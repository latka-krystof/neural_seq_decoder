import torch
from torch import nn

from .augmentations import GaussianSmoothing
from .attention import AttentionBlock


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        post_gru_layers=0,
        post_gru_hidden_dim=None,
        post_gru_dropout=None,
        use_pre_gru_layernorm=False,
        use_day_specific_params=True,
        cnn_layers=0,
        cnn_channels=None,
        cnn_kernel_sizes=None,
        cnn_strides=None,
        cnn_padding=None,
        use_cnn_instead_of_unfold=False,
        use_attention=False,
        attention_heads=8,
        attention_ff_dim=None,
        attention_dropout=None,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        
        # CNN layers for local feature extraction (hybrid CNN+GRU architecture)
        self.cnn_layers = cnn_layers
        self.use_cnn_instead_of_unfold = use_cnn_instead_of_unfold
        cnn_output_dim = neural_dim  # Default: no CNN, use original neural_dim
        
        if cnn_layers > 0:
            if cnn_channels is None:
                raise ValueError("cnn_channels must be specified when cnn_layers > 0")
            if cnn_kernel_sizes is None:
                cnn_kernel_sizes = [3] * cnn_layers
            if cnn_strides is None:
                cnn_strides = [1] * cnn_layers
            if cnn_padding is None:
                cnn_padding = [1] * cnn_layers  # 'same' padding approximation
            
            # Ensure lists are the right length
            if len(cnn_channels) != cnn_layers:
                raise ValueError(f"cnn_channels length ({len(cnn_channels)}) must match cnn_layers ({cnn_layers})")
            if len(cnn_kernel_sizes) != cnn_layers:
                raise ValueError(f"cnn_kernel_sizes length ({len(cnn_kernel_sizes)}) must match cnn_layers ({cnn_layers})")
            if len(cnn_strides) != cnn_layers:
                raise ValueError(f"cnn_strides length ({len(cnn_strides)}) must match cnn_layers ({cnn_layers})")
            if len(cnn_padding) != cnn_layers:
                raise ValueError(f"cnn_padding length ({len(cnn_padding)}) must match cnn_layers ({cnn_layers})")
            
            # Build CNN stack: Conv1d -> BatchNorm1d -> ReLU -> Dropout
            self.cnn_stack = nn.ModuleList()
            in_channels = neural_dim
            for i in range(cnn_layers):
                out_channels = cnn_channels[i]
                kernel_size = cnn_kernel_sizes[i]
                stride = cnn_strides[i]
                padding = cnn_padding[i]
                
                self.cnn_stack.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
                self.cnn_stack.append(nn.BatchNorm1d(out_channels))
                self.cnn_stack.append(nn.ReLU())
                if dropout > 0:
                    self.cnn_stack.append(nn.Dropout(dropout))
                
                in_channels = out_channels
            
            cnn_output_dim = cnn_channels[-1]  # Final CNN output dimension
            # Store for input length calculation
            self.cnn_total_stride = 1
            for s in cnn_strides:
                self.cnn_total_stride *= s
        else:
            self.cnn_stack = None
            self.cnn_total_stride = 1
        
        # Day-specific parameters (can be disabled if using pre-GRU layer norm)
        self.use_day_specific_params = use_day_specific_params
        if use_day_specific_params:
            self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
            self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
            for x in range(nDays):
                self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        else:
            # Register as None to avoid issues, but won't be used
            self.dayWeights = None
            self.dayBias = None

        # GRU layers
        # If CNN is used, GRU input dimension depends on CNN output and whether unfold is used
        if cnn_layers > 0 and use_cnn_instead_of_unfold:
            # CNN replaces unfold: GRU receives CNN output directly (no temporal windows)
            gru_input_dim = cnn_output_dim
        else:
            # Standard case: unfold creates temporal windows
            # If CNN is used, unfold operates on CNN output channels
            gru_input_dim = cnn_output_dim * self.kernelLen
        self.gru_decoder = nn.GRU(
            gru_input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers (only used if day-specific params are enabled)
        if use_day_specific_params:
            for x in range(nDays):
                setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))
            for x in range(nDays):
                thisLayer = getattr(self, "inpLayer" + str(x))
                thisLayer.weight = torch.nn.Parameter(
                    thisLayer.weight + torch.eye(neural_dim)
                )
        
        # Pre-GRU layer normalization (as in [2] with Transformer architecture)
        # Applied before GRU to potentially remove need for day-specific parameters
        self.use_pre_gru_layernorm = use_pre_gru_layernorm
        if use_pre_gru_layernorm:
            # Layer norm is applied to the inputs before GRU
            self.pre_gru_layernorm = nn.LayerNorm(gru_input_dim)

        # Attention mechanism (multi-head self-attention after GRU)
        self.use_attention = use_attention
        gru_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        if use_attention:
            attention_dropout_rate = attention_dropout if attention_dropout is not None else dropout
            self.attention_block = AttentionBlock(
                d_model=gru_output_dim,
                n_heads=attention_heads,
                ff_dim=attention_ff_dim,
                dropout=attention_dropout_rate
            )
        else:
            self.attention_block = None
        
        # Post-GRU stack: linear + layer norm + dropout (as in Linderman Lab approach)
        self.post_gru_layers = post_gru_layers
        if post_gru_layers > 0:
            # Determine input and hidden dimensions
            # If attention is used, post-GRU stack operates on attention output
            post_gru_input_dim = gru_output_dim
            post_gru_hidden = post_gru_hidden_dim if post_gru_hidden_dim is not None else gru_output_dim
            post_gru_dropout_rate = post_gru_dropout if post_gru_dropout is not None else dropout
            
            # Build stack: Linear -> LayerNorm -> Dropout -> ReLU (repeat)
            self.post_gru_stack = nn.ModuleList()
            for i in range(post_gru_layers):
                input_dim = post_gru_input_dim if i == 0 else post_gru_hidden
                self.post_gru_stack.append(nn.Linear(input_dim, post_gru_hidden))
                self.post_gru_stack.append(nn.LayerNorm(post_gru_hidden))
                if post_gru_dropout_rate > 0:
                    self.post_gru_stack.append(nn.Dropout(post_gru_dropout_rate))
                self.post_gru_stack.append(nn.ReLU())
            
            # Final output layer takes post-GRU hidden dimension
            final_input_dim = post_gru_hidden
        else:
            self.post_gru_stack = None
            # Final output layer takes GRU output dimension (or attention output if used)
            final_input_dim = gru_output_dim

        # rnn outputs
        self.fc_decoder_out = nn.Linear(final_input_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer (if enabled)
        if self.use_day_specific_params:
            dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
            transformedNeural = torch.einsum(
                "btd,bdk->btk", neuralInput, dayWeights
            ) + torch.index_select(self.dayBias, 0, dayIdx)
            transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        else:
            # Skip day-specific transformation, just apply nonlinearity
            transformedNeural = self.inputLayerNonlinearity(neuralInput)

        # Apply CNN layers if enabled (for local feature extraction)
        if self.cnn_stack is not None:
            # CNN expects (batch, channels, time) format
            cnn_input = torch.permute(transformedNeural, (0, 2, 1))  # (batch, features, time)
            for layer in self.cnn_stack:
                cnn_input = layer(cnn_input)
            # Convert back to (batch, time, features) for unfold or GRU
            cnn_output = torch.permute(cnn_input, (0, 2, 1))
        else:
            cnn_output = transformedNeural

        # stride/kernel (unfold operation)
        if self.cnn_layers > 0 and self.use_cnn_instead_of_unfold:
            # CNN replaces unfold: use CNN output directly
            gru_input = cnn_output
        else:
            # Standard unfold operation (on CNN output if CNN is enabled, otherwise on transformedNeural)
            gru_input = torch.permute(
                self.unfolder(
                    torch.unsqueeze(torch.permute(cnn_output, (0, 2, 1)), 3)
                ),
                (0, 2, 1),
            )
        
        # Apply pre-GRU layer normalization (as in [2])
        if self.use_pre_gru_layernorm:
            gru_input = self.pre_gru_layernorm(gru_input)

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                gru_input.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                gru_input.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(gru_input, h0.detach())

        # Apply attention mechanism if enabled (multi-head self-attention)
        if self.attention_block is not None:
            hid = self.attention_block(hid)

        # Apply post-GRU stack if enabled (Linderman Lab approach)
        if self.post_gru_stack is not None:
            for layer in self.post_gru_stack:
                hid = layer(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
