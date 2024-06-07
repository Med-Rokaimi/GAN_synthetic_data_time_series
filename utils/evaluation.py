import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Evluation, Calculate CRPS,
def calc_crps(ground_truth, predictions, predictions2):
    return np.absolute(predictions - ground_truth).mean() - 0.5 * np.absolute(predictions - predictions2).mean()









def model_evaluation_r(trues, preds):
    preds= np.round(preds,2)
    trues = np.round(trues, 2)
    mae,mse,rmse = metric(trues, preds)

    return mae,mse,rmse


def plot_trues_preds(trues, preds, path):
    print(trues.shape, preds.shape)
    plt.plot(trues)
    plt.plot(preds)
    plt.title('tures vs plots')
    plt.legend(['trues', 'preds'], loc='upper left')
    plt.savefig(path + 'line.png', bbox_inches='tight')
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
    return {'mae':mae,'mse':mse,'rmse':rmse, 'mspe':mspe, 'mape':mape}

def plot_distibuation(trues, preds, path):
    f, axarr = plt.subplots(1, 1, figsize=(8, 8))
    axarr.hist(trues, bins=100, color='orange', edgecolor='red', density=True, label="target", range=(-10, 10))
    axarr.hist(preds, bins=100, color='cyan', edgecolor='blue', density=True, alpha=0.5, label="forecast",
               range=(-10, 10))
    plt.legend()

    plt.plot()
    plt.show()
    plt.savefig(path + 'distrib.png')




def save_results(trues, preds, scores, PATH):

    np.save(PATH + 'preds.npy', preds)
    np.save(PATH + 'trues.npy', trues)
    name = '_scores_' + 'timestep_'
    np.save(PATH + name +'.npy', scores)


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

def plot_losses(gen_losses, critic_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Mean Loss')
    plt.plot(critic_losses, label='Critic Mean Loss')
    plt.xlabel('Epoch')
    plt.ylabel(' Loss')
    plt.title(' Loss per Epoch')
    plt.legend()
    plt.show()


def plot_losses_avg(gen_losses, critic_losses):
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


def plot_gradiants(gen_gradients, critic_gradients):

    plt.figure(figsize=(10, 5))
    plt.plot(gen_gradients, label='Generator Gradients')
    plt.plot(critic_gradients, label='Critic Gradients')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude per Epoch')
    plt.legend()
    plt.show()

def plot_distibuation_all(trues, preds, path):

        plt.figure(figsize=(10, 5))
        actual_values = trues
        predicted_values = preds

        sns.histplot(actual_values, color='blue', label='Actual Values', kde=True)
        sns.histplot(predicted_values, color='orange', label='Predicted Values', kde=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Actual and Predicted Values')
        plt.legend()
        plt.savefig(path + 'distrib_all.png')
        plt.show()