"""Test Focal CTC loss - Phase 2 Component 4."""
modelName = 'test_focal_ctc'

args = {}
args['outputDir'] = '/home/latka/github/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/latka/github/neural_seq_decoder/processedData'
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
args['whiteNoiseSD'] = 0.8  # Keep same as baseline
args['constantOffsetSD'] = 0.2  # Keep same as baseline
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# NEW: Use modular loss function (Focal CTC)
args['loss'] = {
    'type': 'focal_ctc',  # NEW - Phase 2
    'params': {
        'blank': 0,
        'reduction': 'mean',
        'gamma': 2.0,  # Focusing parameter (higher = more focus on hard examples)
        'alpha': None,  # Optional class-specific weighting
        'zero_infinity': True
    }
}

