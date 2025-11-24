"""Test SpecAugment (time + feature masking combined) - Phase 2 Components 1+2."""
modelName = 'test_specaugment'

args = {}
args['outputDir'] = '/Users/krystoflatka/Documents/GitHub/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/Users/krystoflatka/Documents/GitHub/neural_seq_decoder/processedData'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000
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

# NEW: Use SpecAugment (time + feature masking combined)
args['augmentation'] = {
    'white_noise': {
        'enabled': True,
        'std': 0.8  # Keep same as baseline
    },
    'constant_offset': {
        'enabled': True,
        'std': 0.2  # Keep same as baseline
    },
    'time_masking': {
        'enabled': True,  # Phase 2 Component 1
        'num_masks': 2,
        'mask_width': 20,
        'p': 1.0
    },
    'feature_masking': {
        'enabled': True,  # Phase 2 Component 2
        'num_masks': 2,
        'mask_width': 16,
        'p': 1.0
    }
}

# Old-style args (ignored when using modular augmentation)
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2

