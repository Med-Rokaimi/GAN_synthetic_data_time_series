import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SDEs.sdes import generate_noise
from SDEs.sdes_new import print_merton_example
from args.config import Config
from utils.evaluation import calc_crps, plot_trues_preds, plot_distibuation, metric, save_results, \
    plot_distibuation_all, plot_err_histogram, scatter_plot, scatter_plot_res, plot_losses, plot_gradiants
from data.data import data_prep, create_dataset
from utils.helper import save_config_to_excel, create_exp, save

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(4)
rs = np.random.RandomState(4)

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

    def forward(self, noise,
                x_batch):  # x_batch [16, 10, 2] 10 = seq_len, 2 = features. noisr = [16, 32]. 16 is the bach size
        x_batch = x_batch.view(-1, self.x_batch_size, 2)
        x_batch = x_batch.transpose(0, 1)  # [10, 16, 2]
        x_batch_latent, _ = self.cond_to_latent(x_batch)  # [10, 16, 4]
        x_batch_latent = x_batch_latent[-1]  # [16, 4]
        g_input = torch.cat((x_batch_latent, noise), dim=1)  # conat torch.Size([16, 4]) torch.Size([16, 32])

        output = self.model(g_input)  # [16, 1]

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
    # print("x_batch shape: ", x_batch.shape)
    # print("y_batch shape", y_batch.shape)
    # lables = ones((batch_size, 1))  #Labels=1 indicating they are real
    return x_batch, y_batch


def generate_fake_samples(generator, noise_size, x_batch):
    # generate points in latent space

    noise_batch = torch.tensor(rs.normal(0, 1, (x_batch.size(0), noise_size)),
                               device=device, dtype=torch.float32)

    # print("noise_batch shape", noise_batch.shape)

    y_fake = generator(noise_batch, x_batch).detach()
    # labels = zeros((x_batch.size(0), 1))  #Label=0 indicating they are fake
    return x_batch, y_fake


def train(best_crps, save_path):
    print("epochs", config.epochs)
    itr = 0
    best_gen = None
    import time
    start_time = time.time()  # Record the start time
    generator_losses, discriminator_losses, d_loss =  [], [], 0
    generator_gradients, discriminator_gradients = [], []

    for step in range(config.epochs):

        d_loss = 0

        # load real samples
        # x_bach = batch x seq_len x feature_no [16, 10, 2]
        # y_batch = batch_size x pred_len  [16, 1]
        x_batch, y_batch = load_real_samples(config.batch_size)

        # train D on real samples
        discriminator.zero_grad()
        d_real_decision = discriminator(y_batch, x_batch)
        d_real_loss = adversarial_loss(d_real_decision,
                                       torch.full_like(d_real_decision, 1, device=device))
        d_real_loss.backward()
        d_loss += d_real_loss.detach().cpu().numpy()

        # train discriminator on fake data
        x_batch, y_fake = generate_fake_samples(generator, config.noise_size, x_batch)
        d_fake_decision = discriminator(y_fake, x_batch)
        d_fake_loss = adversarial_loss(d_fake_decision,
                                       torch.full_like(d_fake_decision, 0, device=device))
        d_fake_loss.backward()

        optimizer_d.step()
        d_loss += d_fake_loss.detach().cpu().numpy()

        d_loss = d_loss / 2

        generator.zero_grad()
        noise_batch = torch.tensor(rs.normal(0, 1, (config.batch_size, config.noise_size)), device=device,
                                   dtype=torch.float32)
        y_fake = generator(noise_batch, x_batch)

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
                    noise_batch = torch.tensor(rs.normal(0, 1, (x_val.size(0), config.noise_size)),
                                               device=device,
                                               dtype=torch.float32)
                    predictions.append(generator(noise_batch, x_val).cpu().detach().numpy())

                predictions = np.stack(predictions)

                generator.train()
            # print(y_val.shape)
            crps = calc_crps(y_val, predictions[:100], predictions[100:])

            if crps < best_crps:
                best_crps = crps
                checkpoint_path = f'{save_path}/chkpt.pt'
                torch.save({
                    'step': step,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'best_crps': best_crps
                }, checkpoint_path)
                itr = step
                _ = save(generator, save_path, 'best', save_model)
                best_gen = generator

            print("step : {} , d_loss : {} , g_loss : {}, crps : {}, best crps : {}".format(step, d_loss, g_loss, crps,
                                                                                            best_crps))

            generator_losses.append(g_loss)
            discriminator_losses.append(d_loss)

        # Adjust learning rates
        # scheduler_d.step()
        # scheduler_g.step()

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime

    plot_losses(generator_losses, discriminator_losses, ex_results_path)
    plot_gradiants(generator_gradients, discriminator_gradients, ex_results_path)

    saved_model_path = save(best_gen, save_path, 'best_' + str(best_crps) + 'epochs_' + str(itr), save_model)
    return saved_model_path, generator, runtime


if __name__ == '__main__':
    #################################################
    # General settings
    #################################################

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch_seed, rs_seed = 43, 1234
    torch.manual_seed(torch_seed)
    rs = np.random.RandomState(rs_seed)
    result_path = "results"
    saved_model_path = ""
    dataset_name = "oil.csv"
    target_column = 'WTI'  # the column name of the target time series (brent or WTI)
    dataset_path = 'dataset/' + dataset_name
    model_decriptipn = 'ForGan orginal model '
    model_name = 'forGAN_v16_june'
    save_model = True
    config = Config(
        epochs=7700,
        pred_len=1,
        seq_len=10,
        n_critic=1,
        model_name=model_name,
        dataset=target_column,
        crps=0.5,
        optimiser=None,
        lr=0.0033,
        dropout=0.33,
        hidden_units1=64,
        hidden_units2=32,
        sde_parameters={"Merton": 0, "CGAN-LSTM": 1},
        batch_size=16,
        noise_size=16,
        noise_type='normal',
        generator_latent_size=4,
        discriminator_latent_size=64,
        loss='BCELoss',
        seeds={'torch_seed': torch_seed, 'rs_seed': rs_seed},
        sde='Merton + LSTM -CGAN'
    )
    #################################################
    # Dataset
    #################################################

    # read the csv file
    df = pd.read_csv(dataset_path)
    df = df[6:]

    df = df[[target_column, 'SENT']]  # Price, WTI, SENT, GRACH

    train_size, valid_size, test_size = 2000, 260, 100
    data = create_dataset(df, target_column, train_size, valid_size, test_size, config.seq_len, config.pred_len)

    print(f"Data : {dataset_name}, {data['X_train'].shape} , {data['y_train'].shape}")
    print()

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']
    x_batch_size = data['X_train'].shape[1]  # 24 sequence length

    #################################################
    # create expermint instance
    #################################################
    jobID, ex_results_path = create_exp(result_path, 'exp.csv', config.model_name)
    #################################################
    # Build the model
    #################################################

    generator = Generator(noise_size=config.noise_size,
                          x_batch_size=x_batch_size,
                          generator_latent_size=config.generator_latent_size, feature_no=len(df.columns)).to(device)

    discriminator = Discriminator(x_batch_size=x_batch_size,
                                  discriminator_latent_size=config.discriminator_latent_size).to(device)

    print(generator)
    print(discriminator)

    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=config.lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=config.lr)
    adversarial_loss = nn.BCELoss()
    adversarial_loss = adversarial_loss.to(device)

    # scheduler_d = StepLR(optimizer_d, step_size=10, gamma=0.5)
    # scheduler_g = StepLR(optimizer_g, step_size= 10, gamma=0.5)

    #################################################
    # Training the model
    #################################################

    best_crps = np.inf
    is_train = True
    if is_train:
        saved_model_path, trained_model, runtime = train(best_crps, ex_results_path)
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
        noise_batch = torch.tensor(rs.normal(0, 1, (x_test.size(0), config.noise_size)), device=device,
                                   dtype=torch.float32)
        predictions.append(generator(noise_batch, x_test).detach().cpu().numpy().flatten())

    predictions = np.stack(predictions).flatten()
    y_test = data['y_test'].flatten()
    trues = data['y_test'].flatten()
    preds = predictions.flatten()

    #################################################
    # Ploting
    #################################################

    plot_distibuation(trues, preds, ex_results_path)
    plot_trues_preds(trues, preds, ex_results_path)
    metrics = metric(trues, preds)
    metrics['crps'] = best_crps
    '''

    for name, param in generator.state_dict().items():
        print(name, param.size(), param.data)
    print(f"sigma: {generator.sigma}, lam {generator.lam}, r {generator.r}")
    '''

    save_results(trues, preds, metrics, ex_results_path)


    plot_distibuation(trues, preds, ex_results_path)
    # save result details with configs to exp.csv file
    save_config_to_excel(jobID, ex_results_path, result_path + '/exp.csv', config, model_decriptipn,
                         generator, metrics, {'train_size': train_size,
                                              'valid_size': valid_size, 'test_size': test_size}, runtime, SDE=False)

    # Plot distribution of actual and predicted values
    plot_distibuation_all(trues, preds, ex_results_path)
    plot_err_histogram(trues, preds, ex_results_path)

    scatter_plot(trues, preds, ex_results_path)
    scatter_plot_res(trues, preds, ex_results_path)

    print(config)
    print(df.columns)
    print("runtime: ", runtime)

