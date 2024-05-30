
# https://github.com/pere98diaz/Neural-SDEs-for-Conditional-Time-Series-Generation-and-the-Signature-Wasserstein-1-metric/blob/main/Stocks/nsde_sigwgan/NSDE-sigwgan.ipynb
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'
if not is_cuda:
    print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(suppress=True)

from SDEs.Signature import Signature, Basepoint, sig_lreg, Cumsum2, LeadLag
from SDEs.Utilities import plot_nsde as plot, get_n_params
from SDEs.NSDE import SigNSDE
from SDEs.Training_NSDE_sigwgan import train_sigwgan

data = torch.load('../data/data.pt')
print(data['X_train'].shape)

sig_X = Signature(depth=4, augmentations = [Basepoint, Cumsum2],
                  data_size=data['X_train'].shape[2],
                  interval=[0, data['X_train'].shape[1]+1],
                  q=1,
                  t_norm = data['X_train'][:, :, 0].max()).to('cpu')

sig_Y = Signature(depth=4, augmentations = [Cumsum2],
                  data_size=data['Y_train'].shape[2],
                  interval=[0, data['Y_train'].shape[1]+1],
                  q=1,
                  t_norm = data['Y_train'][:, :, 0].max()).to('cpu')

signatures_X, signatures_Y, signatures_Y_pred, sig_Y = sig_lreg(sig_X, sig_Y, data, 528, alpha=0.1, normalize_sig = True)


#if __name__ == '__main__':




