from collections import namedtuple

# Define the named tuple with all the parameters


Config = namedtuple('Config', ['epochs',
    'pred_len', 'seq_len', 'epoch', 'model_name', 'dataset',
    'crps', 'metrics', 'optimiser', 'lr', 'dropout',
    'hidden_units1', 'hidden_units2', 'sde_parameters'
])