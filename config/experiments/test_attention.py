"""Test attention mechanism architecture (multi-head self-attention after GRU).
Attention allows the model to attend to different parts of the sequence,
potentially capturing long-range dependencies more effectively than GRU alone.

Architecture: Neural Input -> GRU -> Attention Block -> Output
"""
modelName = 'test_attention'

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

# Attention mechanism configuration
# Multi-head self-attention is applied after GRU layers
args['use_attention'] = True
args['attention_heads'] = 8  # Number of attention heads
args['attention_ff_dim'] = None  # None = 4 * GRU output dim (4096 for bidirectional, 4096 for unidirectional with 1024 units)
args['attention_dropout'] = None  # None = use same as main dropout (0.4)

