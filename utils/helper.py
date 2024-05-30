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
        torch_model = path + score + "_" + create_time_stamp() + ".torch"
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





def create_exp(result_path, model_name):

        # Read the Excel file
        new_job_id = 0
        df = pd.read_excel(result_path)

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

        timestr= create_time_stamp()
        exp_result_path = result_path + '/' + str(new_job_id) +model_name + timestr

        if not os.path.exists(exp_result_path):
            os.makedirs(exp_result_path)

        return new_job_id, exp_result_path


import pandas as pd
from openpyxl import load_workbook


def append_to_excel(file_path, new_data):
    """
    Appends a new results to the exp.xlsx.

    Parameters:
    file_path (str): The path to the Excel file.
    new_data (dict): A dictionary containing the new data to append. The keys should match the column names in the Excel file.
    """
    # Define the column order in the Excel file
    columns = ['jobID', 'timestamp', 'epoch', 'Model', 'Dataset', 'crps', 'mse', 'hidden_unites1',
               'hidden_unites2', 'lr', 'droupout', 'pred_len', 'seq_len', 'runtime', 'config',
               'sde_params', 'seeds']

    # Load the existing Excel file
    book = load_workbook(file_path)

    # Convert the new data to a DataFrame with the correct column order
    new_df = pd.DataFrame([new_data], columns=columns)

    # Load the sheet you want to append data to
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    writer.book = book

    # Get the last row in the existing Excel file
    sheet_name = 'Sheet1'  # Replace with your sheet name
    startrow = book[sheet_name].max_row

    # Append the new data to the existing data in the sheet
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        new_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow)

    print(f"New row appended to {file_path} in sheet {sheet_name}")
