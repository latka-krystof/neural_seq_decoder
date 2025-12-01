import torch
from torch import nn

from .augmentations import GaussianSmoothing


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
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
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

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # Post-GRU stack: linear + layer norm + dropout (as in Linderman Lab approach)
        self.post_gru_layers = post_gru_layers
        if post_gru_layers > 0:
            # Determine input and hidden dimensions
            gru_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
            post_gru_hidden = post_gru_hidden_dim if post_gru_hidden_dim is not None else gru_output_dim
            post_gru_dropout_rate = post_gru_dropout if post_gru_dropout is not None else dropout
            
            # Build stack: Linear -> LayerNorm -> Dropout -> ReLU (repeat)
            self.post_gru_stack = nn.ModuleList()
            for i in range(post_gru_layers):
                input_dim = gru_output_dim if i == 0 else post_gru_hidden
                self.post_gru_stack.append(nn.Linear(input_dim, post_gru_hidden))
                self.post_gru_stack.append(nn.LayerNorm(post_gru_hidden))
                if post_gru_dropout_rate > 0:
                    self.post_gru_stack.append(nn.Dropout(post_gru_dropout_rate))
                self.post_gru_stack.append(nn.ReLU())
            
            # Final output layer takes post-GRU hidden dimension
            final_input_dim = post_gru_hidden
        else:
            self.post_gru_stack = None
            # Final output layer takes GRU output dimension
            final_input_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # rnn outputs
        self.fc_decoder_out = nn.Linear(final_input_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # Apply post-GRU stack if enabled (Linderman Lab approach)
        if self.post_gru_stack is not None:
            for layer in self.post_gru_stack:
                hid = layer(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
