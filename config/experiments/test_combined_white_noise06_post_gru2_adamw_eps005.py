"""Combined test: White noise std 0.6 + Post-GRU stack (2 layers) + AdamW eps 0.05.
This configuration combines the best-performing hyperparameters from individual experiments:
- White noise augmentation with std 0.6
- Post-GRU stack with 2 layers (Linderman Lab approach)
- AdamW optimizer with epsilon 0.05"""
modelName = 'test_combined_white_noise06_post_gru2_adamw_eps005'

args = {}
args['outputDir'] = '/kaggle/working/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/kaggle/working/neural_seq_decoder/processedData'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 20000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# White noise augmentation with std 0.6
args['augmentation'] = {
    'white_noise': {
        'enabled': True,
        'std': 0.6  # Lower than baseline (0.8)
    },
    'constant_offset': {
        'enabled': True,
        'std': 0.2  # Keep same as baseline
    }
}

# Old-style args (ignored when using modular augmentation)
args['whiteNoiseSD'] = 0.6
args['constantOffsetSD'] = 0.2

# Post-GRU stack configuration (Linderman Lab approach)
# Stack structure: Linear -> LayerNorm -> Dropout -> ReLU (repeated)
args['post_gru_layers'] = 2  # Number of linear layers in the stack
args['post_gru_hidden_dim'] = None  # None = use same as GRU output (1024)
args['post_gru_dropout'] = None  # None = use same as main dropout (0.4)

# Use AdamW optimizer with epsilon 0.05 (optimal value from experiments)
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.02,
        'betas': (0.9, 0.999),
        'eps': 0.05,  # Optimal epsilon value for AdamW
        'weight_decay': 1e-5,  # Same as baseline's l2_decay
    }
}

