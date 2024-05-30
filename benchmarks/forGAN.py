import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_size, x_batch_size, generator_latent_size, feature_no):
        super().__init__()

        self.noise_size = noise_size
        self.x_batch_size = x_batch_size
        self.generator_latent_size = generator_latent_size
        self.cond_to_latent = nn.GRU(input_size=feature_no,
                                     hidden_size=generator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=generator_latent_size + noise_size,
                      out_features=generator_latent_size + noise_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size + noise_size, out_features=1)
        )

    def forward(self, noise, x_batch):   # x_batch [16, 10, 2] 10 = seq_len, 2 = features. noisr = [16, 32]. 16 is the bach size
        x_batch = x_batch.view(-1, self.x_batch_size, 2)
        x_batch = x_batch.transpose(0, 1) # [10, 16, 2]
        x_batch_latent, _ = self.cond_to_latent(x_batch) #[10, 16, 4]
        x_batch_latent = x_batch_latent[-1] #[16, 4]
        g_input = torch.cat((x_batch_latent, noise), dim=1) # conat torch.Size([16, 4]) torch.Size([16, 32])

        output = self.model(g_input) # [16, 1]

        return output








class Discriminator(nn.Module):
    def __init__(self, x_batch_size, discriminator_latent_size):
        super().__init__()
        self.discriminator_latent_size = discriminator_latent_size
        self.x_batch_size = x_batch_size
        self.input_to_latent = nn.GRU(input_size=1,
                                      hidden_size=discriminator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, prediction, x_batch):
        #prediction.shape, x_batch.shape = [batch_size , pred_len] [16 x 1], [batch x seq_len x featurs_no] [16 x 10 x 2]

        # Ignore the extrnal feature SENT
        x_batch= x_batch[:, :, 0] # batch x seq_len [16 x 10]

        d_input = torch.cat((x_batch, prediction.view(-1, 1)), dim=1) # [batch x seq_len + 1] [16 x 11].  add Xt+1 to the end of each sequence
                                                                      #/ concatantae sequnces and predcited value

        d_input = d_input.view(-1, self.x_batch_size + 1, 1) #[16, 11, 1]

        d_input = d_input.transpose(0, 1) # [11, 16, 1]


        d_latent, _ = self.input_to_latent(d_input)  # [11, 16, 64] GRU layer withy 64 hidden dim

        d_latent = d_latent[-1] # [16, 64]

        output = self.model(d_latent)  # pass through linear layer and return [16, 1]

        return output

