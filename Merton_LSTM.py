# Bidirectional=
from SDEs.sdes import levy_solver
import torch.nn as nn
import pandas as pd

import random

import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from data.data import data_prep, pytorch_data_input
from utils.evaluation import plot_trues_preds, metric
from utils.helper import EarlyStopping

device = "cuda" if torch.cuda.is_available() else "cpu"


class Metron_LSTM(nn.Module):

    def __init__(self, hidden_dim, layer_dim, input_dim, pred_len, batch_size, dropou):
        super(Metron_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.output_dim = pred_len
        self.batch_size = batch_size
        self.dropou = dropou

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim , self.hidden_dim, self.layer_dim, batch_first=True, bidirectional=True, dropout=dropou
        )

        # Fully connected layer
        self.fc_1 = nn.Linear(self.hidden_dim * 2, 60)  # fully connected
        self.fc_2 = nn.Linear(60, self.output_dim)  # fully connected last layer
        self.relu = nn.ReLU()
        self.r = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.m = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.v = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.lam = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(0.02), requires_grad=False)


    def forward(self, x, eva=False):


        if eva:
            lev = levy_solver(self.r, self.m, self.v, self.lam, self.sigma, self.output_dim, 1, 1)
        else:
            lev = levy_solver(self.r, self.m, self.v, self.lam, self.sigma, self.output_dim, self.batch_size, 1)

        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        #x = torch.cat((x, mm), dim=2)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        #print(lev[0:5])

        out = self.fc_2(out)  # final output
        out = out * lev

        return out

class TorchTrainer:
    def __init__(self, model, loss_fn, optimizer):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):

        self.model.train()
        # Makes predictions
        yhat = self.model(x)
        # Computes loss
        loss = self.loss_fn(y, yhat)
        # Computes gradients
        #loss.requires_grad = True
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()


    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, patience, result_path, save_results=False):
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()

                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
            early_stopping(validation_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")

                break

        if save_results:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            torch.save(self.model.state_dict(), result_path + 'best-model-parameters.pt')

        for name, param in self.model.state_dict().items():
            print(name, param.size(), param.data)
        print(f"sigma: {self.model.sigma}, lam {self.model.lam}, r {self.model.r}")


    def evaluate(self, test_loader, batch_size=1, n_features=2):

        with torch.no_grad():
            preds = []
            trues = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                #print("y_test shape", y_test.shape)

                self.model.eval()
                yhat = self.model(x_test, True)
                yhat=yhat.cpu().data.numpy()
                #print("Yhat shape", yhat.shape)
                preds.append(yhat)
                #print("preds shape", len(preds[0]))
                y_test=y_test.cpu().data.numpy()
                trues.append(y_test)

        preds = np.array(preds)
        trues = np.array(trues)
        #print("preds shape", preds.shape)

        #print('Optm preds and test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-1] )
        #print("preds shape 2", preds.shape)
        trues = trues.reshape(-1, trues.shape[-1])
        #print('Optm preds and test shape:', preds.shape, trues.shape)
        #self.plot_losses()
        return trues, preds, self.model

    def plot_losses(self):

        plt.figure(figsize=(4,2))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()





if __name__ == '__main__':
    fix_seed = 2012 #2600
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    EXCEL_EXP_PATH = "./results/exp.xlsx"
    results_path = "./results/Metron_LSTM/"

    epochs = 2000
    batch_size = 16
    hidden_dim = 8
    layer_dim = 1
    dropout = 0.3992
    lr = 0.0032
    patience = 7
    opt= 'AdamW'
    save_results = False

    df = pd.read_csv('dataset/oil.csv')
    df = df[20:]
    df = df[['Price', 'SENT']]
    seq_len, pred_len, feature_no = 10, 1, len(df.columns)
    train_size, valid_size, test_size = 2040,259,65


    data = data_prep(df, seq_len, pred_len, train_size, valid_size, test_size)
    print(data['X_test'].shape)
    print(data['y_test'].shape)

    model = Metron_LSTM(hidden_dim, layer_dim, feature_no, pred_len, batch_size, dropout)

    train_loader, val_loader, test_loader , test_loader_one= pytorch_data_input(data, batch_size)
    optimizer = getattr(torch.optim, opt)(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction="mean")

    opt = TorchTrainer(model, loss_fn=loss, optimizer=optimizer)
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=epochs,
              n_features=feature_no, patience=patience, result_path=results_path,
              save_results=save_results)
    trues, preds, model = opt.evaluate(
        test_loader_one,
        batch_size=1,
        n_features=data['X_test'].shape[2])
    print("opt", opt)

    plot_trues_preds(trues, preds)
    metrics = metric(trues, preds)