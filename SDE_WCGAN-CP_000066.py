'''

The architecture provided in the code is for a Wasserstein GAN with Gradient Penalty (WGAN-GP). Here are the key components and modifications that make it a WGAN-GP rather than a standard GAN:

Wasserstein Loss:

The discriminator (or critic) loss uses the Wasserstein loss, which is simply the difference between the real and fake sample scores: d_loss = d_real_loss + d_fake_loss + lambda_gp * gradient_penalty.
The generator loss is the negative of the discriminator's prediction on the generated data: g_loss = -torch.mean(d_g_decision).
Gradient Penalty:

The code includes a gradient penalty term to enforce the Lipschitz constraint, which is crucial for WGAN-GP. This is done in the compute_gradient_penalty function.
No Sigmoid in Discriminator:

The discriminator does not use a sigmoid activation in the output layer. Instead, it outputs raw scores which are then used to compute the Wasserstein distance.
Use of RMSprop Optimizer:

Although Adam can be used, WGANs often use RMSprop or Adam with specific hyperparameters to ensure stability. The code uses RMSprop with a small learning rate.
Here's the architecture summarized in one complete, cohesive code snippet:
'''


import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, grad

from SDEs.sdes import levy_solver, generate_noise
from utils.evaluation import calc_crps, metric, plot_trues_preds, plot_distibuation, save_results
from data.data import data_prep, create_dataset
from utils.helper import save
from utils.layer import LipSwish


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Generator(nn.Module):
    def __init__(self, hidden_dim, feature_no, seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = 1
        self.input_dim = feature_no
        self.output_dim = 1
        self.dropout = 0.33
        self.mean, self.std = 0, 1
        self.seq_len = seq_len
        self.LipSwish = LipSwish()

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim + noise_size, self.hidden_dim, self.layer_dim, batch_first=True, bidirectional=True,
            dropout=self.dropout
        )
        #nn.init.xavier_normal(self.lstm.weight)

        # Fully connected layer
        self.fc_1 = nn.Linear(self.hidden_dim * 2, 12)
        nn.init.xavier_normal(self.fc_1.weight)
        self.fc_2 = nn.Linear(12, self.output_dim)
        nn.init.xavier_normal(self.fc_2.weight)
        self.relu = nn.ReLU()
        self.r = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.m = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.v = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.lam = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(0.02), requires_grad=True)

    def forward(self, x, noise, batch_size):
        x = (x - self.mean) / self.std

        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        lev = levy_solver(self.r, self.m, self.v, self.lam, self.sigma, self.output_dim, batch_size, 1)

        noise_downsampled = noise.unsqueeze(1).expand(-1, self.seq_len, -1)
        noise_downsampled = noise_downsampled[:, :self.seq_len, :]

        x_combined = torch.cat((x, noise_downsampled), dim=-1)

        out, (hn, cn) = self.lstm(x_combined, (h0, c0))

        out = out[:, -1, :]
        out = self.fc_1(out)
        out = self.LipSwish(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = out * lev
        return out

class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.discriminator_latent_size = hidden_dim
        self.x_batch_size = seq_len
        self.input_to_latent = nn.LSTM(input_size=1, hidden_size=hidden_dim)


        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_dim + seq_len, out_features=64),
            #nn.init.xavier_normal(),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            LipSwish()
        )

    def forward(self, prediction, x_batch):
        if len(prediction.shape) == 1:
            prediction = prediction.unsqueeze(-1)

        x_batch = x_batch[:, :, 0]
        # Ensure prediction has the same batch size as x_batch
        prediction = prediction.view(x_batch.size(0), -1)
        # Concatenate x_batch and prediction along the sequence length dimension
        d_input = torch.cat((x_batch, prediction), dim=1)
        d_input = d_input.unsqueeze(-1)
        d_input = d_input.transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        d_latent = d_latent[-1]
        d_input_flat = torch.cat((d_latent, x_batch.view(x_batch.size(0), -1)), dim=1)
        output = self.model(d_input_flat)
        return output


def load_real_samples(batch_size):
    idx = rs.choice(x_train.shape[0], batch_size)
    x_batch = x_train[idx]
    y_batch = y_train[idx]

    return x_batch, y_batch


def generate_sde_motion(noise_size, x_batch):
    r, m, v, T = torch.tensor(0.02), torch.tensor(0.02), torch.tensor(0.02), 1
    sigma = torch.tensor(0.0891)
    lam = torch.tensor(0.0302)
    steps = x_batch.shape[0]
    Npaths = noise_size

    lev = levy_solver(r, m, v, lam, sigma, T, steps, Npaths)

    return lev


def generate_fake_samples(generator, noise_size, x_batch):
    noise_batch = generate_noise(noise_size, x_batch.size(0), noise_type, rs)
    #_ = generate_sde_motion(noise_size, x_batch)
    y_fake = generator(x_batch, noise_batch, x_batch.size(0)).detach()

    return x_batch, y_fake


def compute_gradient_penalty(discriminator, real_samples, fake_samples, x_batch):
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    alpha = alpha.unsqueeze(-1).unsqueeze(-1)

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates, x_batch)

    fake = Variable(torch.ones(d_interpolates.size(), device=real_samples.device), requires_grad=False)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def train(best_crps):
    print("epochs", epochs)

    lambda_gp = 0.00001


    for step in range(epochs):

        for _ in range(2):
            x_batch, y_batch = load_real_samples(batch_size)

            discriminator.zero_grad()
            d_real_decision = discriminator(y_batch, x_batch)
            d_real_loss = -torch.mean(d_real_decision)

            x_batch, y_fake = generate_fake_samples(generator, noise_size, x_batch)
            d_fake_decision = discriminator(y_fake, x_batch)
            d_fake_loss = torch.mean(d_fake_decision)

            gradient_penalty = compute_gradient_penalty(discriminator, y_batch, y_fake, x_batch)
            d_loss = d_real_loss + d_fake_loss + lambda_gp * gradient_penalty
            d_loss.backward()

        optimizer_d.step()

        generator.zero_grad()
        noise_batch = generate_noise(noise_size, batch_size, noise_type, rs)
        _ = generate_sde_motion(noise_size, x_batch)
        y_fake = generator(x_batch, noise_batch, batch_size)

        d_g_decision = discriminator(y_fake, x_batch)
        g_loss = -torch.mean(d_g_decision)

        g_loss.backward()
        optimizer_g.step()

        g_loss = g_loss.detach().cpu().numpy()

        if step % 100 == 0:
            with torch.no_grad():
                generator.eval()
                predictions = []
                for _ in range(200):
                    noise_batch = generate_noise(noise_size, x_val.size(0), noise_type, rs)

                    predictions.append(generator(x_val, noise_batch, batch_size=1
                                                 ).cpu().detach().numpy())

                predictions = np.stack(predictions)

                generator.train()
            crps = calc_crps(y_val, predictions[:100], predictions[100:])

            if crps <= best_crps:
                best_crps = crps

            print("step : {} , d_loss : {} , g_loss : {}, crps : {}, best crps : {}".format(step, d_loss.item(), g_loss,
                                                                                            crps,
                                                                                            best_crps))
    saved_model_path = save(generator, result_path, str(best_crps), save_model)

    return saved_model_path, generator


if __name__ == '__main__':

    torch.manual_seed(2020)
    rs = np.random.RandomState(4)

    result_path = "./results/SDE_CGAN_v2/"
    saved_model_path = ""
    dataset = 'brent.csv'

    df = pd.read_csv('dataset/' + dataset)
    df = df[6:]
    df = df[['Price', 'SENT']]
    target = 'Price'

    seq_len, pred_len, feature_no = 16, 1, len(df.columns)

    epochs = 10000
    batch_size = 16
    noise_size = 16
    noise_type = 'normal'
    generator_latent_size = 12
    discriminator_latent_size = 64
    save_model = False

    train_size, valid_size, test_size = 2000, 180, 200
    data = create_dataset(df, target, train_size, valid_size, test_size, seq_len, pred_len)

    generator = Generator(
        hidden_dim=generator_latent_size, feature_no=feature_no, seq_len=seq_len).to(device)

    discriminator = Discriminator(seq_len=seq_len,
                                  hidden_dim=discriminator_latent_size).to(device)

    print(generator)
    print(discriminator)

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']

    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=0.003)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=0.003)

    best_crps = np.inf

    is_train = True

    if is_train:
        saved_model_path, trained_model = train(best_crps)
    else:
        print('training mode is off')
        print(saved_model_path, "has been loaded")
        checkpoint = torch.load(saved_model_path)
        generator.load_state_dict(checkpoint['g_state_dict'])

    x_test = torch.tensor(data['X_test'], device=device, dtype=torch.float32)
    predictions = []

    with torch.no_grad():
        generator.eval()
        noise_batch = generate_noise(noise_size, x_test.size(0), noise_type, rs)
        predictions.append(generator(x_test, noise_batch, batch_size=1).detach().cpu().numpy().flatten())

    predictions = np.stack(predictions).flatten()

    y_test = data['y_test'].flatten()

    trues = data['y_test'].flatten()
    preds = predictions.flatten()
    plot_distibuation(trues, preds)
    plot_trues_preds(trues, preds)
    metrics = metric(trues, preds)

    save_results(trues, preds, metrics, saved_model_path)

    merton_ou = levy_solver(r=torch.tensor(0.02), v=torch.tensor(0.02), m=torch.tensor(0.02),
                            lam=torch.tensor(generator.lam), sigma=torch.tensor(generator.sigma), T=1, steps=16,
                            Npaths=10)
    plt.plot(merton_ou)
    plt.xlabel('Days')
    plt.ylabel(' Price')
    plt.title(' Jump Diffusion Process')
    plt.show()
    plt.savefig(saved_model_path + "merton:jump.jpg")