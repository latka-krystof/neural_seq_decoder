modelName = 'test_adamw_minimal'

args = {}
args['outputDir'] = '/kaggle/working/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/kaggle/working/neural_seq_decoder/processedData'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
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
args['l2_decay'] = 1e-5

# Use AdamW optimizer with minimal weight decay (matching baseline's l2_decay)
args['optimizer'] = {
    'type': 'adamw',
    'params': {
        'lr': 0.02,
        'weight_decay': 1e-5,  # Same as baseline's l2_decay
        'eps': 1e-8,
    }
}

# Add learning rate warmup: ramp from 2% to 100% over 150 steps, then stay constant
args['scheduler'] = {
    'type': 'warmup_constant',
    'params': {
        'warmup_steps': 150,  # Warmup over first 150 steps (3% of training)
        'start_factor': 0.02,  # Start at 2% of max LR (0.02 * 0.02 = 0.0004)
    }
}

