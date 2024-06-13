'''
version info:
SDE_CGAN_v1:
 - integrating a noise vector. the noise vector is conctanted with the condition x. the output is feded to the LSTM
'''

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from SDEs.sdes import levy_solver, generate_noise
from SDEs.sdes_new import print_merton_example
from args.config import Config
from utils.evaluation import calc_crps, metric, plot_trues_preds, plot_distibuation, save_results, \
    plot_distibuation_all, plot_err_histogram, plot_gradiants, plot_losses, scatter_plot, scatter_plot_res, \
    get_gradient_statistics
from data.data import create_dataset, grach_model
from utils.helper import save, create_exp, append_to_excel, save_config_to_excel
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
        self.lipSwish = LipSwish()

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim + config.noise_size, self.hidden_dim, self.layer_dim, batch_first=True, bidirectional=True, dropout=self.dropout
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

        #x = (x-self.mean)/self.std
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
        #out = self.lipSwish(out)
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
            nn.Sigmoid(),
            LipSwish(),

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
    r, m, v, T = torch.tensor(0.02), torch.tensor(0.02), torch.tensor(0.02), 2
    sigma = torch.tensor(0.0891)
    lam = torch.tensor(0.0302)
    steps = x_batch.shape[0]
    Npaths = noise_size

    lev = levy_solver(r, m, v, lam, sigma, T, steps, Npaths)

    return lev


def generate_fake_samples(generator, noise_size, x_batch):
    noise_batch = generate_noise(noise_size, x_batch.size(0), config.noise_type, rs)
    # print("noise_batch shape", noise_batch.shape)
    _ = generate_sde_motion(noise_size, x_batch)
    y_fake = generator(x_batch, noise_batch, x_batch.size(0)).detach()
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
        # Track Discriminator gradients
        discriminator_gradients.append(
            torch.mean(torch.tensor([p.grad.norm() for p in discriminator.parameters() if p.grad is not None])).item()
        )

        generator.zero_grad()
        # noise_batch = torch.tensor(rs.normal(0, 1, (batch_size, noise_size)), device=device,
        # dtype=torch.float32)
        noise_batch = generate_noise(config.noise_size, config.batch_size, config.noise_type, rs)
        _ = generate_sde_motion(config.noise_size, x_batch)
        y_fake = generator(x_batch, noise_batch, config.batch_size)

        # print("y_fake", y_fake.shape)
        d_g_decision = discriminator(y_fake, x_batch)
        g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=device))

        g_loss.backward()
        optimizer_g.step()

        g_loss = g_loss.detach().cpu().numpy()
        generator_gradients.append(
            torch.mean(torch.tensor([p.grad.norm() for p in discriminator.parameters() if p.grad is not None])).item()
        )

        # Validation
        if step % 100 == 0:
            with torch.no_grad():
                generator.eval()
                predictions = []
                for _ in range(200):
                    noise_batch = generate_noise(config.noise_size,x_val.size(0),config.noise_type, rs)

                    predictions.append(generator(x_val, noise_batch, batch_size=1
                                                 ).cpu().detach().numpy())

                predictions = np.stack(predictions)

                generator.train()
            # print(y_val.shape)
            crps = calc_crps(y_val, predictions[:100], predictions[100:])

            if crps <= best_crps:
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
        #scheduler_d.step()
        #scheduler_g.step()

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime

    plot_losses(generator_losses,discriminator_losses, ex_results_path)
    plot_gradiants(generator_gradients, discriminator_gradients, ex_results_path)

    saved_model_path = save(best_gen, save_path, 'best_' + str(best_crps)+ 'epochs_' + str(itr), save_model)
    return saved_model_path, generator, runtime


if __name__ == '__main__':

    #################################################
    # General settings
    #################################################

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch_seed, rs_seed = 42,4
    torch.manual_seed(torch_seed)
    rs = np.random.RandomState(rs_seed)
    result_path = "results"
    saved_model_path = ""
    dataset_name = "oil.csv"
    target_column = 'WTI' # the column name of the target time series (brent or WTI)
    dataset_path = 'dataset/' + dataset_name
    model_decriptipn = 'CGAN + Merton Jump '
    model_name = 'SDE_CGAN_v10_june'
    save_model = True
    grach_feature = None #'estimated_volatility'  # ['estimated_volatility', 'returns', None]
    config = Config(
        epochs=5200,
        pred_len=1,
        seq_len=15,
        n_critic = 1,
        model_name=model_name,
        dataset=target_column,
        crps=0.5,
        optimiser=None,
        lr=0.0001,
        dropout=0.33,
        hidden_units1=64,
        hidden_units2=32,
        sde_parameters={"Merton": 1, "CGAN-LSTM": 1},
        batch_size = 32,
        noise_size = 32,
        noise_type = 'normal',
        generator_latent_size = 8,
        discriminator_latent_size = 64,
        loss = 'BCELoss',
        seeds = {'torch_seed': torch_seed, 'rs_seed': rs_seed},
        sde = 'Merton + LSTM -CGAN'
    )
    #################################################
    # Dataset
    #################################################

    # read the csv file
    df = pd.read_csv(dataset_path)
    df = df[6:]

    # add grach as feature
    grach_dic = grach_model(df, target_column, horizon=config.batch_size)
    #print(len(returns), len(estimated_volatility),len(forecast_volatility))

    df = df[[target_column, 'SENT']]  # Price, WTI, SENT, GRACH

    if grach_feature is not None:
        df['grach'] = grach_dic[grach_feature]
        df['grach'].iloc[0] = df['grach'].iloc[1]

    # Exploratory Data Analysis (EDA) of Volatility Data
    #from utils.helper import eda
    #eda(df, target)

    train_size, valid_size, test_size = 1800, 200, 200
    data = create_dataset(df, target_column, train_size, valid_size, test_size, config.seq_len, config.pred_len)

    print(f"Data : {dataset_name}, {data['X_train'].shape} , {data['y_train'].shape}")
    print()

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']


    #################################################
    # create expermint instance
    #################################################
    jobID, ex_results_path = create_exp(result_path , 'exp.csv', config.model_name)
    #################################################
    # Build the model
    #################################################

    generator = Generator(hidden_dim=config.generator_latent_size, feature_no=len(df.columns),
                          seq_len= config.seq_len, output_dim=config.pred_len, dropout=config.dropout).to(device)
    discriminator = Discriminator(seq_len=config.seq_len,
                                  hidden_dim=config.discriminator_latent_size).to(device)

    optimizer_g = torch.optim.RMSprop(generator.parameters())
    optimizer_d = torch.optim.RMSprop(discriminator.parameters())
    adversarial_loss = nn.BCELoss()
    adversarial_loss = adversarial_loss.to(device)

    #scheduler_d = StepLR(optimizer_d, step_size=10, gamma=0.5)
    #scheduler_g = StepLR(optimizer_g, step_size= 10, gamma=0.5)

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
        noise_batch = generate_noise(config.noise_size, x_test.size(0), config.noise_type, rs)
        predictions.append(generator(x_test, noise_batch, batch_size=1).detach().cpu().numpy().flatten())

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

    print_merton_example(r=0.02, v=0.02, m=0.02, lam=generator.lam, sigma=generator.sigma)

    plot_distibuation(trues, preds, ex_results_path)
    # save result details with configs to exp.csv file
    save_config_to_excel(jobID, ex_results_path, result_path + '/exp.csv', config, model_decriptipn,
                         generator, metrics, {'train_size': train_size,
                                              'valid_size': valid_size, 'test_size': test_size}, runtime)

    # Plot distribution of actual and predicted values
    plot_distibuation_all(trues, preds, ex_results_path)
    plot_err_histogram(trues, preds, ex_results_path)

    scatter_plot(trues, preds, ex_results_path)
    scatter_plot_res(trues, preds, ex_results_path)

    print(config)
    print(df.columns)

