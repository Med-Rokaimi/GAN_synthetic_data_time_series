'''


'''


import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.autograd import Variable, grad

from data.data import grach_model
from SDEs.sdes import levy_solver, generate_noise, print_merton_example
from args.config import Config
from utils.evaluation import calc_crps, metric, plot_trues_preds, plot_distibuation, save_results, \
    get_gradient_statistics, plot_samples, plot_losses, plot_losses_avg, plot_losses_max, plot_gradiants, plot_distibuation_all
from data.data import create_dataset, data_to_tensor
from utils.helper import save, create_exp, save_config_to_excel
from utils.layer import LipSwish


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Generator(nn.Module):
    def __init__(self, hidden_dim, feature_no, seq_len, noise_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = 1
        self.input_dim = feature_no
        self.output_dim = 1
        self.dropout = 0.33
        self.mean, self.std = 0, 1
        self.seq_len = seq_len
        self.noise_size = noise_size
        self.LipSwish = LipSwish()

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim + self.noise_size, self.hidden_dim, self.layer_dim, batch_first=True, bidirectional=True,
            dropout=self.dropout
        )

        # Fully connected layer
        self.fc_1 = nn.Linear(self.hidden_dim * 2, 12)
        nn.init.xavier_normal_(self.fc_1.weight)
        self.fc_2 = nn.Linear(12, self.output_dim)
        nn.init.xavier_normal_(self.fc_2.weight)
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

class Critic(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.critic_latent_size = hidden_dim
        self.x_batch_size = seq_len
        self.input_to_latent = nn.LSTM(input_size=1, hidden_size=hidden_dim)


        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_dim + seq_len, out_features=64),
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

def generate_levy_jump_path(noise_size, x_batch):
    r, m, v, T = torch.tensor(0.02), torch.tensor(0.02), torch.tensor(0.02), 1
    sigma = torch.tensor(0.0891)
    lam = torch.tensor(0.0302)
    steps = x_batch.shape[0]
    Npaths = noise_size
    lev = levy_solver(r, m, v, lam, sigma, T, steps, Npaths)
    return lev

def generate_fake_samples(generator, noise_size, x_batch):
    noise_batch = generate_noise(noise_size, x_batch.size(0), config.noise_type, rs)
    y_fake = generator(x_batch, noise_batch, x_batch.size(0)).detach()
    return x_batch, y_fake


def compute_gradient_penalty(critic, real_samples, fake_samples, x_batch):
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    #alpha = generate_noise(noise_size,real_samples.size(0),noise_type,rs) #torch.rand(real_samples.size(0), 1, device=real_samples.device)
    alpha = alpha.unsqueeze(-1).unsqueeze(-1)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = Variable(interpolates, requires_grad=True)
    d_interpolates = critic(interpolates, x_batch)
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

def train(best_crps, ex_results_path):

    import time
    start_time = time.time()  # Record the start time
    gen_losses , critic_losses = [],[]
    gen_max_losses, critic_max_losses = [],[]
    gen_gradients , critic_gradients = [], []

    for step in range(config.epochs):
        d_loss_avg = 0
        for _ in range(config.n_critic):
            x_batch, y_batch = load_real_samples(config.batch_size)
            critic.zero_grad()
            d_real_decision = critic(y_batch, x_batch)
            d_real_loss = -torch.mean(d_real_decision)

            x_batch, y_fake = generate_fake_samples(generator, config.noise_size, x_batch)
            d_fake_decision = critic(y_fake, x_batch)
            d_fake_loss = torch.mean(d_fake_decision)

            lambda_gp = 0.00001
            gradient_penalty = compute_gradient_penalty(critic, y_batch, y_fake, x_batch)
            d_loss = d_real_loss + d_fake_loss  + lambda_gp * gradient_penalty

            d_loss.backward()
            d_loss_avg = d_loss_avg + d_loss.item()

            # Track critic gradients
            critic_gradients.append(
                torch.mean(torch.stack([p.grad.norm() for p in critic.parameters() if p.grad is not None])).item())

            optimizer_d.step()

        critic_losses.append(d_loss_avg/config.n_critic)

        generator.zero_grad()
        noise_batch = generate_noise(config.noise_size, config.batch_size, config.noise_type, rs)
        _ = generate_levy_jump_path(config.noise_size, x_batch)
        y_fake = generator(x_batch, noise_batch, config.batch_size)
        d_g_decision = critic(y_fake, x_batch)
        g_loss = -torch.mean(d_g_decision)
        g_loss.backward()

        # Track generator gradients
        gen_gradients.append(
            torch.mean(torch.stack([p.grad.norm() for p in generator.parameters() if p.grad is not None])).item())

        optimizer_g.step()
        gen_losses.append(g_loss.item())
        gen_gradients.append(gen_gradients[-1])


        if step % 100 == 0:
            with torch.no_grad():
                generator.eval()
                predictions = []
                for _ in range(400):
                    noise_batch = generate_noise(config.noise_size, x_val.size(0), config.noise_type, rs)
                    predictions.append(generator(x_val, noise_batch, batch_size=1
                                                 ).cpu().detach().numpy())
                predictions = np.stack(predictions)

                generator.train()
            crps = calc_crps(y_val, predictions[:200], predictions[200:])

            if crps <= best_crps:
                best_crps = crps
                #total_crps = total_crps.append(crps.item())

            #trainig_critic_loss, trainig_gen_loss, trainig_crps_ = trainig_critic_loss.append(d_loss), trainig_gen_loss.append(g_loss), trainig_crps_.append(crps)
            print("step : {} , d_loss : {} , g_loss : {}, crps : {}, best crps : {}".format(step, d_loss.item(), g_loss,
                                                                                            crps,
                                                                                            best_crps))
            # monitor generated samples
            plot_samples(y_val, predictions, step)


    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime
    saved_model_path = save(generator, ex_results_path, str(best_crps), save_model)

    plot_losses(gen_losses, critic_losses)
    plot_losses_avg(gen_losses, critic_losses)
    plot_losses_max(gen_max_losses, critic_max_losses)
    plot_gradiants(gen_gradients, critic_gradients)

    return saved_model_path, generator, runtime


if __name__ == '__main__':

    #################################################
    # General settings
    #################################################

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch_seed = 2020
    rs_seed = 4
    torch.manual_seed(torch_seed)
    rs = np.random.RandomState(rs_seed)
    result_path = "./results"
    saved_model_path = ""
    dataset = "oil"  # WTI, BRENT
    dataset_path = 'dataset/' + dataset + '.csv'
    model_decriptipn = 'WCGAN + Merton Jump '
    save_model = True
    model_name = "SDE-WCGAN_v3"
    runtime = 0

    config = Config(
        epochs=100,
        pred_len=1,
        seq_len=10,
        n_critic=1,
        model_name=model_name,
        dataset=dataset,
        crps=0.5,
        optimiser=None,
        lr=0.003,
        dropout=0.33,
        hidden_units1=64,
        hidden_units2=32,
        sde_parameters={"param1": 0.02, "param2": 0.02},
        batch_size=16,
        noise_size=16,
        noise_type='normal',
        generator_latent_size=12,
        discriminator_latent_size=64,
        loss='WGAN-PG',
        sde= {'torch_seed': torch_seed, 'rs_seed':rs_seed}
    )

    # create a new job
    jobID, ex_results_path = create_exp(result_path, 'exp.csv', config.model_name)

    #################################################
    # Dataset
    #################################################

    # read the csv file
    df = pd.read_csv(dataset_path)
    df = df[6:]

    #select target column name
    target = 'brent'  # Price, WTI

    #add grach as feature
    actual_volatility,estimated_volatility, forecast_volatility = grach_model(df[[target]], horizon=config.batch_size)
    print(len(actual_volatility), len(estimated_volatility),len(forecast_volatility))


    df = df[[target, 'SENT']] # Price, WTI, SENT, GRACH


    #Exploratory Data Analysis (EDA) of Volatility Data
    from utils.helper import eda
    eda(df, target)

    train_size, valid_size, test_size = 2000, 180, 200
    data = create_dataset(df, target, train_size, valid_size, test_size, config.seq_len, config.pred_len)

    print(f"Data : {dataset}, {data['X_train'].shape} , {data['y_train'].shape}")
    print()

    x_train = torch.tensor(data['X_train'], device=device, dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], device=device, dtype=torch.float32)
    x_val = torch.tensor(data['X_valid'], device=device, dtype=torch.float32)
    y_val = data['y_valid']


    #################################################
    # Build the model
    #################################################

    generator = Generator(
        hidden_dim=config.generator_latent_size, feature_no=len(df.columns), seq_len=config.seq_len, noise_size=config.noise_size).to(device)
    critic = Critic(seq_len=config.seq_len,
                                  hidden_dim=config.discriminator_latent_size).to(device)
    x_train, y_train, x_val, y_val = data_to_tensor(data, device)

    ########################################
    # initialize weights
    '''def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)'''
    ########################################

    #generator.apply(weight_norm_func)
    #critic.apply(weight_norm_func)

    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=config.lr)
    optimizer_d = torch.optim.RMSprop(critic.parameters(), lr=config.lr)

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

    plot_distibuation(trues, preds, saved_model_path)
    plot_trues_preds(trues, preds, saved_model_path)
    metrics = metric(trues, preds)
    metrics['crps'] = best_crps

    #save trues and preds as npy files
    save_results(trues, preds, metrics, saved_model_path)
    '''
    for name, param in generator.state_dict().items():
        print(name, param.size(), param.data)
    print(f"sigma: {generator.sigma}, lam {generator.lam}, r {generator.r}")
    '''

    print_merton_example(r=0.02,v=0.02,m=0.02, lam =generator.lam, sigma = generator.sigma)
    plot_distibuation(trues,preds, saved_model_path)


    # save result details with configs to exp.csv file
    save_config_to_excel(jobID, saved_model_path, result_path + '/exp.csv' , config, model_decriptipn,
                         generator, metrics, {'train_size':train_size,
                                              'valid_size':valid_size, 'test_size':test_size},runtime)

    # Plot distribution of actual and predicted values
    plot_distibuation_all(trues, preds, saved_model_path)



