'''
Summary of Additions:
Evaluation Metrics:

Added Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) calculations.
Statistical Test:

Added Kolmogorov-Smirnov (KS) test to compare the distributions of real and generated data.
Visualization:

Added plots for true vs. predicted values and autocorrelation analysis.
Evaluation Function:

Added evaluate_model function to handle the evaluation process.
These additions will help you to robustly evaluate and analyze the performance of your WGAN-GP model.
'''

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, grad
from torch.cuda.amp import GradScaler, autocast
from arch import arch_model
from scipy.stats import ks_2samp

from SDEs.sdes import levy_solver, generate_noise
from utils.evaluation import calc_crps, metric, plot_trues_preds, plot_distibuation, save_results
from data.data import data_prep
from utils.helper import save

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(4)
rs = np.random.RandomState(4)

result_path = "./results/SDE_CGAN_v1/"
saved_model_path = ""


# Better weight initialization
def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.fc(x))


class Generator_LSTM_LEVY(nn.Module):
    def __init__(self, hidden_dim, feature_no, seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = 2
        self.input_dim = feature_no
        self.output_dim = 1
        self.dropout = 0.33
        self.mean, self.std = 0, 1
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            self.input_dim + noise_size, self.hidden_dim, self.layer_dim,
            batch_first=True, bidirectional=True, dropout=self.dropout
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)
        self.fc_1 = nn.Linear(self.hidden_dim * 2, 12)
        self.fc_2 = nn.Linear(12, self.output_dim)
        self.relu = nn.ReLU()
        self.res_block = ResidualBlock(self.hidden_dim * 2)
        self.r = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.m = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.v = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.lam = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(0.02), requires_grad=True)

    def forward(self, x, noise, garch_volatility, batch_size):
        x = (x - self.mean) / self.std

        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        lev = levy_solver(self.r, self.m, self.v, self.lam, self.sigma, self.output_dim, batch_size, 1)

        noise_downsampled = noise.unsqueeze(1).expand(-1, self.seq_len, -1)
        noise_downsampled = noise_downsampled[:, :self.seq_len, :]

        # Integrate GARCH volatility into the input
        garch_volatility = garch_volatility.unsqueeze(-1).expand_as(x)

        x_combined = torch.cat((x, noise_downsampled, garch_volatility), dim=-1)

        out, (hn, cn) = self.lstm(x_combined, (h0, c0))

        out = out[:, -1, :]
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.res_block(out)
        out = self.layer_norm(out)  # Apply Layer Normalization
        out = self.fc_2(out)
        out = out * lev

        return out


class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.discriminator_latent_size = hidden_dim
        self.x_batch_size = seq_len
        self.input_to_latent = nn.GRU(input_size=1, hidden_size=hidden_dim)

        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_dim + seq_len, out_features=128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, prediction, x_batch):
        if len(prediction.shape) == 1:
            prediction = prediction.unsqueeze(-1)

        x_batch = x_batch[:, :, 0]

        prediction = prediction.view(x_batch.size(0), -1)

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


def generate_fake_samples(generator, noise_size, x_batch, garch_volatility):
    noise_batch = generate_noise(noise_size, x_batch.size(0), noise_type, rs)

    _ = generate_sde_motion(noise_size, x_batch)

    y_fake = generator(x_batch, noise_batch, garch_volatility, x_batch.size(0)).detach()

    return x_batch, y_fake

def visualize_samples(generator, x_val, noise_size, noise_type, rs, step):
    with torch.no_grad():
        generator.eval()
        noise_batch = generate_noise(noise_size, x_val.size(0), noise_type, rs)
        generated_samples = generator(x_val, noise_batch, batch_size=1).cpu().detach().numpy()
        generator.train()

    plt.figure(figsize=(10, 5))
    for i in range(min(10, generated_samples.shape[0])):
        plt.subplot(2, 5, i + 1)
        plt.imshow(generated_samples[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Generated Samples at Step {step}')
    plt.show()
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


def evaluate_model(generator, x_test, y_test, garch_volatility):
    # Generate predictions
    predictions =[]
    with torch.no_grad():
        generator.eval()
        noise_batch = generate_noise(noise_size, x_test.size(0), noise_type, rs)
        predictions = generator(x_test, noise_batch, garch_volatility, batch_size=1).detach().cpu().numpy().flatten()

    # Calculate evaluation metrics
    mae = np.mean(np.abs(y_test - predictions))
    mse = np.mean((y_test - predictions) ** 2)
    rmse = np.sqrt(mse)
    ks_stat, ks_pvalue = ks_2samp(y_test.flatten(), predictions)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"KS Statistic: {ks_stat}, p-value: {ks_pvalue}")

    # Plot true vs. predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.flatten(), label='True Values')
    plt.plot(predictions, label='Predicted Values')
    plt.legend()
    plt.title('True vs. Predicted Values')
    plt.show()

    # Autocorrelation analysis
    plt.figure(figsize=(10, 5))
    pd.plotting.autocorrelation_plot(pd.Series(y_test.flatten()), label='True Values')
    pd.plotting.autocorrelation_plot(pd.Series(predictions), label='Predicted Values')
    plt.legend()
    plt.title('Autocorrelation Analysis')
    plt.show()

    return predictions


def train(best_crps):
    print("epochs", epochs)

    lambda_gp = 10
    scaler = GradScaler()

    for step in range(epochs):
        for _ in range(5):  # Update the discriminator more frequently
            x_batch, y_batch = load_real_samples(batch_size)

            # Fit GARCH model to the training data
            garch_model = arch_model(y_batch.squeeze(), vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            garch_volatility = torch.tensor(garch_fit.conditional_volatility, device=device, dtype=torch.float32)

            discriminator.zero_grad()
            with autocast():
                d_real_decision = discriminator(y_batch, x_batch)
                d_real_loss = -torch.mean(d_real_decision)

                x_batch, y_fake = generate_fake_samples(generator, noise_size, x_batch, garch_volatility)
                d_fake_decision = discriminator(y_fake, x_batch)
                d_fake_loss = torch.mean(d_fake_decision)

                gradient_penalty = compute_gradient_penalty(discriminator, y_batch, y_fake, x_batch)
                d_loss = d_real_loss + d_fake_loss + lambda_gp * gradient_penalty

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

        generator.zero_grad()
        with autocast():
            noise_batch = generate_noise(noise_size, batch_size, noise_type, rs)
            _ = generate_sde_motion(noise_size, x_batch)
            y_fake = generator(x_batch, noise_batch, garch_volatility, batch_size)

            d_g_decision = discriminator(y_fake, x_batch)
            g_loss = -torch.mean(d_g_decision)

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_g)
        scaler.update()

        g_loss = g_loss.detach().cpu().numpy()

        # Apply gradient clipping
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

        if step % 100 == 0:
            with torch.no_grad():
                generator.eval()
                predictions = []
                for _ in range(200):
                    noise_batch = generate_noise(noise_size, x_val.size(0), noise_type, rs)

                    predictions.append(
                        generator(x_val, noise_batch, garch_volatility, batch_size=1).cpu().detach().numpy())

                predictions = np.stack(predictions)

                generator.train()
            crps = calc_crps(y_val, predictions[:100], predictions[100:])

            if crps <= best_crps:
                best_crps = crps

            print("step : {} , d_loss : {} , g_loss : {}, crps : {}, best crps : {}".format(step, d_loss.item(), g_loss,
                                                                                            crps, best_crps))
            # Visualize generated samples
            visualize_samples(generator, x_val, noise_size, noise_type, rs, step)

        scheduler_g.step()
        scheduler_d.step()

    saved_model_path = save(generator, result_path, str(best_crps), save_model)

    return saved_model_path, generator


if __name__ == '__main__':

    df = pd.read_csv('dataset/brent.csv')
    df = df[6:]
    df = df[['Price', 'SENT']]

    seq_len, pred_len, feature_no = 10, 1, len(df.columns)

    dim = 128
    epochs = 300
    batch_size = 16
    noise_size = 16
    noise_type = 'normal'
    generator_latent_size = 8
    discriminator_latent_size = 64
    save_model = True
    target = 'Price'

    train_size, valid_size, test_size = 2000, 180, 200

    data = data_prep(df, target, seq_len, pred_len, train_size, valid_size, test_size)
    print(data['X_test'].shape)
    print(data['y_test'].shape)

    generator = Generator_LSTM_LEVY(hidden_dim=generator_latent_size, feature_no=feature_no, seq_len=seq_len).to(device)
    discriminator = Discriminator(seq_len=seq_len, hidden_dim=discriminator_latent_size).to(device)

    print(generator)
    print(discriminator)

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=0.005)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=0.005)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=1000, gamma=0.95)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=1000, gamma=0.95)

    best_crps = np.inf

    is_train = True

    if is_train:
        saved_model_path, trained_model = train(best_crps)
    else:
        print('training mode is off')

    if save_model:
        print(saved_model_path, "has been loaded")
        checkpoint = torch.load(saved_model_path)
        generator.load_state_dict(checkpoint['g_state_dict'])

    x_test = torch.tensor(data['X_test'], device=device, dtype=torch.float32)
    y_test = data['y_test'].flatten()

    # Fit GARCH model to the test data
    garch_model = arch_model(y_test.squeeze(), vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    garch_volatility = torch.tensor(garch_fit.conditional_volatility, device=device, dtype=torch.float32)

    predictions = evaluate_model(generator, x_test, y_test, garch_volatility)

    trues = y_test.flatten()
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
