"""Test AdamW optimizer with even larger epsilon (0.2) than baseline (0.1).
Exploring if more conservative epsilon provides better stability."""
modelName = 'test_adamw_eps'

args = {}
args['outputDir'] = '/kaggle/working/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/kaggle/working/neural_seq_decoder/processedData'
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
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# Use AdamW optimizer with even larger epsilon (0.2) than baseline (0.1)
# Testing if more conservative epsilon provides better long-term stability
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.02,
        'betas': (0.9, 0.999),
        'eps': 0.05,  # Even larger than baseline's 0.1 for maximum stability
        'weight_decay': 1e-5,  # Same as baseline's l2_decay
    }
}

