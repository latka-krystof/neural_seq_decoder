"""Test post-GRU stack with 3 layers.
Varying number of post-GRU layers to find optimal architecture depth."""
modelName = 'test_post_gru_stack_layers3'

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
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# Post-GRU stack configuration with 3 layers (deeper)
args['post_gru_layers'] = 3
args['post_gru_hidden_dim'] = None
args['post_gru_dropout'] = None

