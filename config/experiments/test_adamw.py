"""Test AdamW optimizer - Phase 1 Component 4."""
modelName = 'test_adamw'

args = {}
args['outputDir'] = '/home/latka/github/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/latka/github/neural_seq_decoder/processedData'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['nUnits'] = 1024
args['nBatch'] = 10000
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

# NEW: Use modular optimizer (AdamW)
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.02,
        'weight_decay': 0.05,  # Higher than baseline (1e-5)
        'eps': 1e-8,  # Standard PyTorch default (smaller than baseline 0.1)
    }
}

# NEW: Use modular scheduler (cosine with warmup)
args['scheduler'] = {
    'type': 'cosine_warmup',
    'params': {
        'warmup_steps': 1000,
        'T_max': 9000,  # total_steps - warmup_steps
        'eta_min': 0.0,
    }
}

# Note: lrStart and lrEnd are ignored when using modular scheduler
args['lrStart'] = 0.02  # For backward compatibility
args['lrEnd'] = 0.02
args['l2_decay'] = 1e-5  # Ignored when using modular optimizer

