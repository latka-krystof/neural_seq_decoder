"""Test white noise augmentation with std 1.4."""
modelName = 'test_white_noise_std14'

args = {}
args['outputDir'] = '/kaggle/working/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/kaggle/working/neural_seq_decoder/processedData'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 2000
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

args['augmentation'] = {
    'white_noise': {
        'enabled': True,
        'std': 1.4
    },
    'constant_offset': {
        'enabled': True,
        'std': 0.2
    }
}

args['whiteNoiseSD'] = 1.0
args['constantOffsetSD'] = 0.2

