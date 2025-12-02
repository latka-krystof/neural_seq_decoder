"""Test pre-GRU layer normalization to potentially remove need for day-specific parameters.
Based on [2] which used layer normalization before Transformer layers to handle day-to-day variation.

This experiment tests:
1. Adding layer normalization before GRU layers
2. Optionally disabling day-specific parameters to see if layer norm can handle variation
"""
modelName = 'test_pre_gru_layernorm_no_day_specific'

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

# Pre-GRU layer normalization configuration
# Layer norm is applied to strided inputs before GRU (as in [2] with Transformer)
args['use_pre_gru_layernorm'] = True

# Option to disable day-specific parameters
# If True, keeps day-specific weights/bias (baseline behavior)
# If False, removes day-specific params to test if layer norm alone can handle variation
args['use_day_specific_params'] = False  # Set to False to test if layer norm removes need for day params
