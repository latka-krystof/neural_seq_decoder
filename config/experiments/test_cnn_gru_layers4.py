"""Test hybrid CNN+GRU architecture with 4 CNN layers.
Varying number of CNN layers to find optimal feature extraction depth."""
modelName = 'test_cnn_gru_layers4'

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

# CNN configuration with 4 layers (deepest)
args['cnn_layers'] = 4
args['cnn_channels'] = [256, 256, 256, 256]  # Four layers with 256 channels each
args['cnn_kernel_sizes'] = [3, 3, 3, 3]
args['cnn_strides'] = [1, 1, 1, 1]
args['cnn_padding'] = [1, 1, 1, 1]
args['use_cnn_instead_of_unfold'] = False

