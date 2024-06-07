'''

Added the GP and loss from
https://github.com/hanyoseob/pytorch-WGAN-GP/blob/master/layer.py#L266

added :LipSwish
'''

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SDEs.sdes import levy_solver, generate_noise
from args.config import Config
from utils.evaluation import calc_crps, metric, plot_trues_preds, plot_distibuation, save_results
from data.data import create_dataset
from utils.helper import save, create_exp, append_to_excel
from utils.layer import LipSwish


class Generator(nn.Module):
    def __init__(self, hidden_dim, feature_no, seq_len, output_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = 1
        self.input_dim = feature_no
        self.output_dim = output_dim
        self.dropout = dropout
        self.mean, self.std = 0, 1
        self.seq_len = seq_len
        self.LipSwish = LipSwish()

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim + noise_size, self.hidden_dim, self.layer_dim, batch_first=True, bidirectional=True, dropout=self.dropout
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

    def forward(self, x, noise, batch_size): # x = [16, 10, 2]

        x = (x-self.mean)/self.std

        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        # could be an SDE noise as SDE-GAN does ملاحظة
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        lev = levy_solver(self.r, self.m, self.v, self.lam, self.sigma, self.output_dim, batch_size, 1)

        # Downsample the noise to match the sequence length of x_batch
        noise_downsampled = noise.unsqueeze(1).expand(-1, self.seq_len, -1)  # Shape: (batch_size, seq_length, noise_dim)
        noise_downsampled = noise_downsampled[:, :self.seq_len, :]  # Downsample to match the sequence length

        # Concatenate the noise with the input features along the feature dimension
        x_combined = torch.cat((x, noise_downsampled),
                               dim=-1)  # Shape: (batch_size, seq_length, features_number + noise_dim)

        out, (hn, cn) = self.lstm(x_combined, (h0, c0))


        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc_1(out)  # first dense
        out = self.LipSwish(out)
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
            nn.ReLU(),
            LipSwish()
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

    noise_batch = generate_noise(noise_size, x_batch.size(0), noise_type, rs)

    # print("noise_batch shape", noise_batch.shape)
    _ = generate_sde_motion(noise_size, x_batch)

    y_fake = generator(x_batch, noise_batch, x_batch.size(0)).detach()
    # labels = zeros((x_batch.size(0), 1))  #Label=0 indicating they are fake
    return x_batch, y_fake


def train(best_crps):
    print("epochs", config.epochs)

    for step in range(config.epochs):
        d_loss = 0
        # load real samples
        # x_bach = batch x seq_len x feature_no [16, 10, 2]
        # y_batch = batch_size x pred_len  [16, 1]
        x_batch, y_batch = load_real_samples(batch_size)

        # train D on real samples
        discriminator.zero_grad()
        d_real_decision = discriminator(y_batch, x_batch)
         # WGAN Loss
        d_real_loss = torch.mean(d_real_decision)

        d_real_loss.backward()
        d_loss += d_real_loss.detach().cpu().numpy()

        # train discriminator on fake data
        x_batch, y_fake = generate_fake_samples(generator, noise_size, x_batch)
        d_fake_decision = discriminator(y_fake, x_batch)
        d_fake_loss = -torch.mean(d_fake_decision)
        d_fake_loss.backward()

        optimizer_d.step()
        d_loss += d_fake_loss.detach().cpu().numpy()

        d_loss = d_loss / 2

        generator.zero_grad()
        # noise_batch = torch.tensor(rs.normal(0, 1, (batch_size, noise_size)), device=device,
        # dtype=torch.float32)
        noise_batch = generate_noise(noise_size, batch_size, noise_type, rs)
        _ = generate_sde_motion(noise_size, x_batch)
        y_fake = generator(x_batch, noise_batch, batch_size)

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
                    noise_batch = generate_noise(noise_size,x_val.size(0),noise_type, rs)

                    predictions.append(generator(x_val, noise_batch, batch_size=1
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
    saved_model_path = save(generator, ex_results_path, str(best_crps), save_model)

    return saved_model_path, generator


if __name__ == '__main__':




    #################################################
    # General settings
    #################################################

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(4)
    rs = np.random.RandomState(4)
    result_path = "./results"
    saved_model_path = ""
    dataset = "brent_wti"    #WTI, BRENT
    dataset_path = 'dataset/' + dataset + '.csv'
    model_decriptipn = 'CGAN + Merton Jump '

    config = Config(
        epochs=10000,
        pred_len=1,
        seq_len=10,
        n_critic= 5,
        model_name="SDE_CGAN_v1",
        dataset=dataset,
        crps=0.5,
        metrics={"mse": None, "rmse": None},
        optimiser=None,
        lr=0.001,
        dropout=0.33,
        hidden_units1=64,
        hidden_units2=32,
        sde_parameters={"param1": 0.1, "param2": 0.2}
    )

    # create a new job
    jobID, ex_results_path = create_exp(result_path , 'exp.xlsx', config.model_name)

    dim = 128

    batch_size = 16
    noise_size = 16
    noise_type = 'normal'  # normal, gbm
    generator_latent_size = 8
    discriminator_latent_size = 64
    save_model = False


    #################################################
    # Dataset
    #################################################

    df = pd.read_csv(dataset_path)
    data, features = {}, []

    if dataset == 'brent':
        features = ['Price', 'SENT']
        df = df[features]
    elif dataset == 'brent_wti':
        features = ['WTI', 'SENT']
        df = df[features]
        df = df.rename(columns={'WTI': 'Price'})
    else:
        print("unknown dataset name")

    train_size, valid_size, test_size = 2000, 180, 200
    data = create_dataset(df, train_size, valid_size, test_size, config.seq_len, config.pred_len)


    print(f"Data : {dataset}, {data['X_train'].shape} , {data['y_train'].shape}")
    print()

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']

    #################################################
    # Build the model
    #################################################

    generator = Generator(hidden_dim=generator_latent_size, feature_no=len(features),
                          seq_len= config.seq_len, output_dim=config.pred_len, dropout=config.dropout).to(device)

    discriminator = Discriminator(seq_len=config.seq_len,
                                  hidden_dim=discriminator_latent_size).to(device)


    optimizer_g = torch.optim.RMSprop(generator.parameters())
    optimizer_d = torch.optim.RMSprop(discriminator.parameters())
    adversarial_loss = nn.BCELoss()
    adversarial_loss = adversarial_loss.to(device)

    #################################################
    # Training the model
    #################################################

    best_crps = np.inf
    is_train = True
    if is_train:
        saved_model_path, trained_model = train(best_crps)
    else:
        print('training mode is off')

    #################################################
    # Testing the model
    #################################################

    if save_model:
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

    #################################################
    # Ploting
    #################################################

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