import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

#Evluation, Calculate CRPS,
def calc_crps(ground_truth, predictions, predictions2):
    return np.absolute(predictions - ground_truth).mean() - 0.5 * np.absolute(predictions - predictions2).mean()


def plot_trues_preds(trues, preds, path=False):
    print(trues.shape, preds.shape)
    plt.plot(trues)
    plt.plot(preds)
    plt.title('Actual vs generated data')
    plt.legend(['Actual', 'Generated'], loc='upper left')
    if path:
        plt.savefig(path + '/line.png', bbox_inches='tight')
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

def r_2(preds, trues):
    return metrics.r2_score(trues, preds)  # R-Squared
from scipy.special import rel_entr
def kl_divergence(preds, trues):

    # Ensure that P and Q are probability distributions
    preds /= preds.sum()
    trues /= trues.sum()

    # Compute KL Divergence
    kl_div = np.sum(rel_entr(preds, trues))

    return kl_div
def metric(trues, preds):
    preds = np.round(preds, 2)
    trues = np.round(trues, 2)
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rmse = RMSE(preds, trues)
    mspe = MSPE(preds, trues)
    mape = MAPE(preds, trues)
    r2 = r_2(preds, trues)
    kl = kl_divergence(preds, trues)

    print(" MAE: {:.6f}  , MSE {:.6f}, RMSE {:.6f}, MSPE {:.6f}, MAPE {:.6f}, R2 {:.6f}, KL {:.6f}".format(mae, mse ,rmse ,mspe ,mape ,r2,kl))
    return {'mae':mae,'mse':mse,'rmse':rmse, 'mspe':mspe, 'mape':mape, 'r2':r2}

def plot_distibuation(trues, preds, path, save = False):
    f, axarr = plt.subplots(1, 1, figsize=(8, 8))
    axarr.hist(trues, bins=100, color='orange', edgecolor='red', density=True, label="target", range=(-10, 10))
    axarr.hist(preds, bins=100, color='cyan', edgecolor='blue', density=True, alpha=0.5, label="forecast",
               range=(-10, 10))
    plt.legend()

    plt.plot()
    plt.show()
    if save:
        plt.savefig(path + '/distrib.png')


def save_results(trues, preds, scores, PATH, save=True):

    np.save(PATH + '/preds.npy', preds)
    np.save(PATH + '/trues.npy', trues)
    name = str(f"scores_mse_{scores['mse']:.6f}")
    np.save(PATH + '/' + name +'.npy', scores)


def get_gradient_statistics(model):
    grad_means = []
    grad_maxs = []
    for param in model.parameters():
        if param.grad is not None:
            grad_means.append(param.grad.abs().mean().item())
            grad_maxs.append(param.grad.abs().max().item())
    return grad_means, grad_maxs


def plot_samples(y_val, predictions, step ):
    plt.figure(figsize=(10, 5))
    plt.plot(y_val.flatten(), label='Actual Data')
    plt.plot(predictions[:100].mean(axis=0).flatten(), label='Generated Data')
    plt.title(f'Generated vs Actual Data at Step {step}')
    plt.legend()
    plt.show()

def plot_losses(gen_losses, critic_losses, path, save = None):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator')
    plt.plot(critic_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel(' Loss')
    plt.title('Generator and Critic Loss During Training')
    plt.legend()
    if save:
        plt.savefig(path + "/loss.png")
    plt.show()


def plot_losses_avg(gen_losses, critic_losses, path, save= None):
    print( "  m " , len(critic_losses), len(gen_losses))
    mov_avg_gen_losses = [sum(gen_losses[i:i + 10]) / 10 for i in range(0, len(gen_losses), 10)]
    mov_avg_critic_losses = [sum(critic_losses[i:i + 10]) / 10 for i in range(0, len(critic_losses), 10)]
    plt.figure(figsize=(10, 5))
    plt.plot(mov_avg_gen_losses, label='Generator Mean Loss')
    plt.plot(mov_avg_critic_losses, label='Critic Mean Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss per Epoch')
    plt.legend()
    plt.yscale('symlog')  # Use symlog scale to better visualize different magnitudes
    if save:
        plt.savefig(path + '/avg_loss.png')
    plt.show()


def plot_losses_max(gen_max_losses, critic_max_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_max_losses, label='Generator Max Loss')
    plt.plot(critic_max_losses, label='Critic Max Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Max Loss')
    plt.title('Max Loss per Epoch')
    plt.legend()
    plt.show()


def plot_gradiants(gen_gradients, critic_gradients, path, save = False):

    plt.figure(figsize=(10, 5))
    plt.plot(gen_gradients, label='Generator Gradients')
    plt.plot(critic_gradients, label='Discriminator Gradients')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude per Epoch')
    plt.legend()
    if save:
        plt.savefig(path + '/grad.png')
    plt.show()

def plot_distibuation_all(trues, preds, path, save = False):

        plt.figure(figsize=(10, 5))
        actual_values = trues
        predicted_values = preds

        sns.histplot(actual_values, color='blue', label='Actual Values', kde=True)
        sns.histplot(predicted_values, color='orange', label='Predicted Values', kde=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Actual and Predicted Values')
        plt.legend()
        if save:
            plt.savefig(path + '/distrib_all.png')
        plt.show()

def plot_err_histogram(trues, preds, path, save = False):
    # Calculate the forecast errors
    errors = preds - trues

    # Plot the histogram of forecast errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, density=True)
    plt.xlabel("Forecast Error")
    plt.ylabel("Density")
    plt.title("Histogram of Forecast Errors")
    plt.grid(True)
    if save:
        plt.savefig(path + '/hist_err.png')
    plt.show()


def scatter_plot(trues, preds, path, save = False):
    plt.scatter(trues, preds)
    plt.xlabel('Actual Petal Width')
    plt.ylabel('Predicted Petal Width')
    plt.title('Actual vs Predicted Petal Width')
    if save:
        plt.savefig(path + '/scatter.png')
    plt.show()

def scatter_plot_res(trues, preds, path, save = False):
    residuals = trues - preds
    plt.scatter(preds, residuals)
    plt.xlabel('Predicted Petal Width')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    if save:
        plt.savefig(path + '/scatter_res.png')
    plt.show()