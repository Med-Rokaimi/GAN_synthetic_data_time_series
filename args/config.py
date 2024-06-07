from collections import namedtuple

# Define the named tuple with all the parameters


Config = namedtuple('Config', ['epochs',
    'pred_len', 'seq_len', 'n_critic','model_name', 'dataset',
    'crps', 'optimiser', 'lr', 'dropout',
    'hidden_units1', 'hidden_units2', 'sde_parameters', 'batch_size',
        'noise_size',
        'noise_type',
        'generator_latent_size',
        'discriminator_latent_size', 'loss', 'sde'
])