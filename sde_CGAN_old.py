import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SDEs.sdes import levy_solver, generate_noise
from utils.evaluation import calc_crps, metric, plot_trues_preds, plot_distibuation, save_results
from data.data import data_prep
from utils.helper import save

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(4)
rs = np.random.RandomState(4)

result_path = "./results/SDE_CGAN/"
saved_model_path = ""


class Generator_LSTM_LEVY(nn.Module):
    def __init__(self, hidden_dim, feature_no):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = 1
        self.input_dim = feature_no
        self.output_dim = 1
        self.dropout = 0.3

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, bidirectional=True, dropout=self.dropout
        )

        # Fully connected layer
        self.fc_1 = nn.Linear(self.hidden_dim * 2, 12)  # fully connected
        self.fc_2 = nn.Linear(12, self.output_dim)  # fully connected last layer
        self.relu = nn.ReLU()
        self.r = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.m = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.v = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.lam = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(0.02), requires_grad=True)

    def forward(self, x, batch_size):
        lev = levy_solver(self.r, self.m, self.v, self.lam, self.sigma, self.output_dim, batch_size, 1)

        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        # could be an SDE noise as SDE-GAN does ملاحظة
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # x = torch.cat((x, mm), dim=2)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        # print(lev[0:5])

        out = self.fc_2(out)  # final output
        out = out * lev

        return out


class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.discriminator_latent_size = hidden_dim
        self.x_batch_size = seq_len
        self.input_to_latent = nn.GRU(input_size=1,
                                      hidden_size=hidden_dim)

        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, prediction, x_batch):
        # prediction.shape, x_batch.shape = [batch_size , pred_len] [16 x 1], [batch x seq_len x featurs_no] [16 x 10 x 2]

        # Ignore the extrnal feature SENT
        x_batch = x_batch[:, :, 0]  # batch x seq_len [16 x 10]

        d_input = torch.cat((x_batch, prediction.view(-1, 1)),
                            dim=1)  # [batch x seq_len + 1] [16 x 11].  add Xt+1 to the end of each sequence
        # / concatantae sequnces and predcited value

        d_input = d_input.view(-1, self.x_batch_size + 1, 1)  # [16, 11, 1]

        d_input = d_input.transpose(0, 1)  # [11, 16, 1]

        d_latent, _ = self.input_to_latent(d_input)  # [11, 16, 64] GRU layer withy 64 hidden dim

        d_latent = d_latent[-1]  # [16, 64]

        output = self.model(d_latent)  # pass through linear layer and return [16, 1]

        return output


def load_real_samples(batch_size):
    idx = rs.choice(x_train.shape[0], batch_size)
    x_batch = x_train[idx]
    y_batch = y_train[idx]
    # print("loading real samples")

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
    # generate points in latent space

    # noise_batch = torch.tensor(rs.normal(0, 1, (batch_size, noise_size)),
    # device=device, dtype=torch.float32)
    noise_batch = generate_noise(noise_size, x_batch.size(0), noise_type, rs)

    # print("noise_batch shape", noise_batch.shape)
    sde = generate_sde_motion(noise_size, x_batch)

    y_fake = generator(x_batch, x_batch.size(0)).detach()
    # labels = zeros((x_batch.size(0), 1))  #Label=0 indicating they are fake
    return x_batch, y_fake


def train(best_crps):
    print("epochs", epochs)

    for step in range(epochs):

        d_loss = 0

        # load real samples
        # x_bach = batch x seq_len x feature_no [16, 10, 2]
        # y_batch = batch_size x pred_len  [16, 1]
        x_batch, y_batch = load_real_samples(batch_size)

        # train D on real samples
        discriminator.zero_grad()
        d_real_decision = discriminator(y_batch, x_batch)
        d_real_loss = adversarial_loss(d_real_decision,
                                       torch.full_like(d_real_decision, 1, device=device))
        d_real_loss.backward()
        d_loss += d_real_loss.detach().cpu().numpy()

        # train discriminator on fake data
        x_batch, y_fake = generate_fake_samples(generator, noise_size, x_batch)
        d_fake_decision = discriminator(y_fake, x_batch)
        d_fake_loss = adversarial_loss(d_fake_decision,
                                       torch.full_like(d_fake_decision, 0, device=device))
        d_fake_loss.backward()

        optimizer_d.step()
        d_loss += d_fake_loss.detach().cpu().numpy()

        d_loss = d_loss / 2

        generator.zero_grad()
        # noise_batch = torch.tensor(rs.normal(0, 1, (batch_size, noise_size)), device=device,
        # dtype=torch.float32)
        noise_batch = generate_noise(noise_size, batch_size, noise_type, rs)
        sde = generate_sde_motion(noise_size, x_batch)
        y_fake = generator(x_batch, batch_size)

        # print("y_fake", y_fake.shape)
        d_g_decision = discriminator(y_fake, x_batch)
        g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=device))

        g_loss.backward()
        optimizer_g.step()

        g_loss = g_loss.detach().cpu().numpy()

        # Validation
        if step % 100 == 0:
            with torch.no_grad():
                generator.eval()
                predictions = []
                for _ in range(200):
                    noise_batch = torch.tensor(rs.normal(0, 1, (x_val.size(0), noise_size)),
                                               device=device,
                                               dtype=torch.float32)
                    # predictions.append(generator(noise_batch, x_val, sde = generate_sde_motion(noise_size, x_batch)).cpu().detach().numpy())
                    predictions.append(generator(x_val, batch_size=1
                                                 ).cpu().detach().numpy())

                predictions = np.stack(predictions)

                generator.train()
            # print(y_val.shape)
            crps = calc_crps(y_val, predictions[:100], predictions[100:])

            if crps <= best_crps:
                best_crps = crps
                # save_model(generator, saved_model_path, str(best_crps))

            print("step : {} , d_loss : {} , g_loss : {}, crps : {}, best crps : {}".format(step, d_loss, g_loss, crps,
                                                                                            best_crps))
    saved_model_path = save(generator, result_path, str(best_crps), save_model)

    return saved_model_path, generator


if __name__ == '__main__':

    df = pd.read_csv('dataset/oil.csv')
    df = df[6:]
    df = df[['Price', 'SENT']]

    seq_len, pred_len, feature_no = 10, 1, len(df.columns)

    dim = 128
    epochs = 10000
    batch_size = 16
    noise_size = 32
    noise_type = 'normal'
    generator_latent_size = 4
    discriminator_latent_size = 64
    save_model = True

    train_size, valid_size, test_size = 2000, 180, 200

    data = data_prep(df, seq_len, pred_len, train_size, valid_size, test_size)
    print(data['X_test'].shape)
    print(data['y_test'].shape)

    generator = Generator_LSTM_LEVY(
        hidden_dim=generator_latent_size, feature_no=feature_no).to(device)

    discriminator = Discriminator(seq_len=seq_len,
                                  hidden_dim=discriminator_latent_size).to(device)

    print(generator)
    print(discriminator)

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']

    optimizer_g = torch.optim.RMSprop(generator.parameters())
    optimizer_d = torch.optim.RMSprop(discriminator.parameters())
    adversarial_loss = nn.BCELoss()
    adversarial_loss = adversarial_loss.to(device)

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
    predictions = []

    with torch.no_grad():
        generator.eval()
        # noise_batch = torch.tensor(rs.normal(0, 1, (x_test.size(0), noise_size)), device=device,
        # dtype=torch.float32)
        noise_batch = generate_noise(noise_size, x_test.size(0), noise_type, rs)
        predictions.append(generator(x_test, batch_size=1).detach().cpu().numpy().flatten())

    predictions = np.stack(predictions).flatten()
    # print("preds", predictions.shape)

    y_test = data['y_test'].flatten()
    # print("trues and preds", y_test.shape, predictions.shape)

    trues = data['y_test'].flatten()
    preds = predictions.flatten()
    plot_distibuation(trues, preds)
    plot_trues_preds(trues, preds)
    metrics = metric(trues, preds)
    '''

    for name, param in generator.state_dict().items():
        print(name, param.size(), param.data)
    print(f"sigma: {generator.sigma}, lam {generator.lam}, r {generator.r}")

    save_results(trues, preds, metrics, saved_model_path)

    '''

    merton_ou = levy_solver(r=torch.tensor(0.02), v=torch.tensor(0.02), m=torch.tensor(0.02),
                            lam=torch.tensor(generator.lam), sigma=torch.tensor(generator.sigma), T=1, steps=16,
                            Npaths=10)
    plt.plot(merton_ou)
    plt.xlabel('Days')
    plt.ylabel(' Price')
    plt.title(' Jump Diffusion Process')
    plt.show()
    plt.savefig(saved_model_path + "merton:jump.jpg")


