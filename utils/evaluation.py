import numpy as np
import matplotlib.pyplot as plt


#Evluation, Calculate CRPS,
def calc_crps(ground_truth, predictions, predictions2):
    return np.absolute(predictions - ground_truth).mean() - 0.5 * np.absolute(predictions - predictions2).mean()









def model_evaluation_r(trues, preds):
    preds= np.round(preds,2)
    trues = np.round(trues, 2)
    mae,mse,rmse = metric(trues, preds)

    return mae,mse,rmse


def plot_trues_preds(trues, preds):
    print(trues.shape, preds.shape)
    plt.plot(trues)
    plt.plot(preds)
    plt.title('tures vs plots')
    plt.legend(['trues', 'preds'], loc='upper left')
    # plt.savefig(path + 'basic_dec_bigru.png', bbox_inches='tight')
    plt.show()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))
def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(trues, preds):
    preds = np.round(preds, 2)
    trues = np.round(trues, 2)
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rmse = RMSE(preds, trues)
    mspe = MSPE(preds, trues)
    mape = MAPE(preds, trues)
    print(" MAE: {:.6f}  , MSE {:.6f}, RMSE {:.6f}, MSPE {:.6f}, MAPE {:.6f}".format(mae, mse ,rmse ,mspe ,mape))
    return mae,mse,rmse, mspe, mape

def plot_distibuation(trues, preds):
    f, axarr = plt.subplots(1, 1, figsize=(8, 8))
    axarr.hist(trues, bins=100, color='orange', edgecolor='red', density=True, label="target", range=(-10, 10))
    axarr.hist(preds, bins=100, color='cyan', edgecolor='blue', density=True, alpha=0.5, label="forecast",
               range=(-10, 10))
    plt.legend()

    plt.plot()
    plt.show()


def save_results(trues, preds, scores, PATH):

    np.save(PATH + 'preds.npy', preds)
    np.save(PATH + 'trues.npy', trues)
    name = '_scores_' + 'timestep_'
    np.save(PATH + name +'.npy', scores)