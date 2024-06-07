from datetime import datetime as dt
import pandas as pd
import os
import numpy as np
import torch
from openpyxl import load_workbook
def create_time_stamp():
    now = dt.now()
    timestr = now.strftime("%Y_%m_%d__%H_%M_%S")
    return timestr


def save(model, path , score, save_model):

    score = str(score)

    if save_model:
        stamped_name__ = score + "_" + create_time_stamp() + ".torch"
        torch_model = os.path.join(path,stamped_name__)
        torch.save({'g_state_dict': model.state_dict()}, torch_model)
    else:
        torch_model = path

    return torch_model




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, result_path, save_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if save_model:
                save(model, result_path , score, save_model)
            self.save_checkpoint(val_loss, model , result_path, save_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save_model:
                save(model, result_path, score, save_model)
            self.save_checkpoint(val_loss, model, result_path, save_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, result_path, save_model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss





def create_exp(result_path, file_name , model_name):

        # Read the Excel file
        new_job_id = 0

        full_path = os.path.join(result_path, file_name)
        print(full_path)
        df = pd.read_csv(full_path)

        # Check if the jobID column exists and has values
        if 'jobID' in df.columns:
            last_job_id = df['jobID'].dropna().iloc[-1] if not df['jobID'].dropna().empty else None

            # Generate new job ID
            if last_job_id is not None:
                new_job_id = int(last_job_id) + 1
        else:
            # If the jobID column does not exist
            print('Eerror: JobID column is not exist')
            exit()



        model_folder = os.path.join(result_path, model_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        timestr= create_time_stamp()
        exp_result_path = os.path.join(model_folder, str(new_job_id) + model_name + timestr)
        if not os.path.exists(exp_result_path):
            os.makedirs(exp_result_path)

        return new_job_id, exp_result_path


import pandas as pd
from openpyxl import load_workbook



def append_to_excel(file_path, new_data):

    print(file_path)
    import csv

    """
    Appends a new results to the exp.csv.

    Parameters:
    file_path (str): The path to the Excel file.
    new_data (dict): A dictionary containing the new data to append. The keys should match the column names in the Excel file.
    """
    # get the excel file column names
    df = pd.read_csv(file_path)
    columns = df.columns

    # Convert the new data to a DataFrame with the correct column order
    new_df = pd.DataFrame([new_data], columns=columns)

    # Open the CSV file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        # Create a DictWriter object
        csvwriter = csv.DictWriter(csvfile, fieldnames=new_df.keys())
        # Check if the file is empty to write the header
        file_is_empty = csvfile.tell() == 0
        if file_is_empty:
            csvwriter.writeheader()

        # Append the new row
        csvwriter.writerow(new_data)
    print(f"New row appended to {file_path} ")


def save_config_to_excel(jobID, exp_path, results_folder, config, model_decriptipn, generator, metrics, dataset, runtime):
    data =  { 'jobID':jobID, 'timestamp': create_time_stamp() , 'path':exp_path,
              'Model': config.model_name,
              'epoch':config.epochs,
               'Dataset': config.dataset,
              'noise':config.noise_type, 'loss':config.loss,'sde':config.sde,
              'crps':metrics['crps'], 'mse':metrics['mse'],
              'hidden_unites1': config.hidden_units1,
              'hidden_unites2': config.hidden_units2,
              'lr':config.lr, 'droupout': config.dropout,
              'pred_len': config.pred_len, 'seq_len':config.seq_len,
              'runtime':runtime, 'config':config,
              'sde_params': [generator.lam, generator.sigma],
              'seeds':'', 'model_decription':model_decriptipn,
              'dataset_shape':dataset}

    append_to_excel(results_folder,data)
    print(results_folder)
    print("append to excel")


def eda(ts, target):
    import  matplotlib.pyplot as plt
    # Calculate the rolling mean and standard deviation
    print(ts.columns)
    rolling_mean = ts[target].rolling(window=30).mean()
    rolling_std = ts[target].rolling(window=30).std()

    # Plot the rolling mean and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts[target], label="Volatility")
    plt.plot(rolling_mean.index, rolling_mean, label="Rolling Mean")
    plt.plot(rolling_std.index, rolling_std, label="Rolling Std")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("Rolling Mean and Standard Deviation of Volatility Data")
    plt.legend()
    plt.grid(True)
    plt.show()