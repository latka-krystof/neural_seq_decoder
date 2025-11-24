"""Test AdamW optimizer - Fixed version with better hyperparameters."""
modelName = 'test_adamw_fixed'

args = {}
args['outputDir'] = '/home/latka/github/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/latka/github/neural_seq_decoder/processedData'
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

# FIXED: Use modular optimizer (AdamW) with reasonable weight decay
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.005,  # Reduced from 0.02 - more stable with AdamW
        'weight_decay': 1e-3,  # FIXED: Much lower than 0.05, but still higher than baseline
        'eps': 1e-8,
    }
}

# FIXED: Use linear scheduler (simpler, more stable) or shorter warmup
# Option 1: Linear decay (like baseline, but with AdamW)
args['scheduler'] = {
    'type': 'linear',
    'params': {
        'start_factor': 1.0,
        'end_factor': 0.1,  # Decay to 10% of initial LR
    }
}

# Option 2: Cosine with shorter warmup (commented out - uncomment to try)
# args['scheduler'] = {
#     'type': 'cosine_warmup',
#     'params': {
#         'warmup_steps': 200,  # FIXED: Much shorter warmup
#         'T_max': 4800,  # total_steps - warmup_steps
#         'eta_min': 0.001,  # FIXED: Don't decay to zero, keep some learning
#     }
# }

# Note: lrStart and lrEnd are ignored when using modular scheduler
args['lrStart'] = 0.01  # For backward compatibility
args['lrEnd'] = 0.001
args['l2_decay'] = 1e-5  # Ignored when using modular optimizer

