"""Test hybrid CNN+GRU architecture for local feature extraction.
CNN layers are added before the GRU to extract local temporal patterns,
which can help the GRU focus on longer-range dependencies.

Architecture: Neural Input -> CNN Stack -> Unfold -> GRU -> Output
"""
modelName = 'test_cnn_gru'

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

# CNN configuration for hybrid CNN+GRU architecture
# CNN layers extract local features before GRU processes longer-range dependencies
args['cnn_layers'] = 2  # Number of CNN layers
args['cnn_channels'] = [256, 256]  # Output channels for each CNN layer
args['cnn_kernel_sizes'] = [3, 3]  # Kernel size for each layer
args['cnn_strides'] = [1, 1]  # Stride for each layer (1 = no downsampling)
args['cnn_padding'] = [1, 1]  # Padding to maintain sequence length
args['use_cnn_instead_of_unfold'] = False  # Keep unfold operation after CNN

