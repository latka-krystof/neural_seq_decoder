"""Test AdamW optimizer - Alternative with constant LR phase."""
modelName = 'test_adamw_2.0_alt'

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
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False

# Alternative: Lower LR, even lower weight decay
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.005,  # Conservative LR
        'weight_decay': 5e-5,  # Very close to baseline (1e-5), just slightly higher
        'eps': 1e-8,
    }
}

# Alternative: No scheduler - constant LR (let AdamW's adaptive learning handle it)
args['scheduler'] = {
    'type': 'none',  # Constant learning rate
}

# Gradient clipping for stability
args['grad_clip'] = 1.0

args['lrStart'] = 0.005
args['lrEnd'] = 0.005
args['l2_decay'] = 1e-5

