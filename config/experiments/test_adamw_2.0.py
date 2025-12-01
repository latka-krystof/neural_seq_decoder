"""Test AdamW optimizer - Improved version with gradient clipping and better hyperparameters."""
modelName = 'test_adamw_2.0'

args = {}
args['outputDir'] = '/kaggle/working/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/kaggle/working/neural_seq_decoder/processedData'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['nUnits'] = 1024
args['nBatch'] = 5000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8  # Keep same as baseline
args['constantOffsetSD'] = 0.2  # Keep same as baseline
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False

# IMPROVED: Use modular optimizer (AdamW) with lower weight decay
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.01,  # Increased slightly - 0.005 was too conservative
        'weight_decay': 1e-4,  # IMPROVED: Lower than 1e-3, closer to baseline but still using AdamW benefits
        'eps': 1e-8,
    }
}

# IMPROVED: Use constant LR for longer, then decay (helps maintain learning)
# This keeps LR constant for first 60% of training, then decays
args['scheduler'] = {
    'type': 'linear',
    'params': {
        'start_factor': 1.0,
        'end_factor': 0.2,  # IMPROVED: Decay to 20% instead of 10% - keep more learning capacity
    }
}

# IMPROVED: Add gradient clipping for stability
args['grad_clip'] = 1.0  # Clip gradients to max norm of 1.0

# Note: lrStart and lrEnd are ignored when using modular scheduler
args['lrStart'] = 0.01  # For backward compatibility
args['lrEnd'] = 0.002
args['l2_decay'] = 1e-5  # Ignored when using modular optimizer

