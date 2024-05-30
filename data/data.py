from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

mm = MinMaxScaler()
ss = StandardScaler()


def create_dataset(df, train_size, valid_size, test_size, seq_len, pred_len):

    data = data_prep(df, seq_len, pred_len, train_size, valid_size, test_size)
    return data
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def split_train_test_pred (X_ss, y_mm , train_test_cutoff, vald_size, predict_size):
    X_train = X_ss[:train_test_cutoff]
    X_valid = X_ss[train_test_cutoff: train_test_cutoff + vald_size]

    y_train = y_mm[:train_test_cutoff]
    y_valid = y_mm[train_test_cutoff: train_test_cutoff + vald_size]

    X_test = X_ss[-predict_size:]
    y_test = y_mm[-predict_size:]

    data = {"X_train": X_train, "y_train": y_train, "X_valid": X_valid, "y_valid": y_valid, "X_test": X_test,
            "y_test": y_test}
    return data

def normalize__my_data_(X, y):
  X_trans = ss.fit_transform(X)
  y_trans = mm.fit_transform(y.reshape(-1, 1))
  return X_trans, y_trans

def denormolize_data(trues, preds):
    return mm.inverse_transform(trues), mm.inverse_transform(preds)


def pytorch_data_input(data, batch_size):
    train_features = torch.Tensor(data['X_train'])
    train_targets = torch.Tensor(data['y_train'])
    val_features = torch.Tensor(data['X_valid'])
    val_targets = torch.Tensor(data['y_valid'])
    test_features = torch.Tensor(data['X_test'])
    test_targets = torch.Tensor(data['y_test'])

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    # print(train.tensors)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader , test_loader_one

def data_prep(df, seq_len, pred_len, train_size, valid_size, test_size):
    X, y = df, df.Price.values
    # 5- Normalize
    X_trans, y_trans = normalize__my_data_(X, y)

    # 7- Build the sequence
    X_ss, y_mm = split_sequences(X_trans, y_trans, seq_len, pred_len)

    train_size = train_size
    train_test_cutoff = train_size
    vald_size = valid_size
    test_size = test_size
    # understand_data_values_for_split(num_of_samples, X, y)
    data = split_train_test_pred(X_ss, y_mm, train_test_cutoff, vald_size, test_size)
    # data = split_train_test_pred(X_trans, y_trans, train_test_cutoff, vald_size, test_size)
    return data




