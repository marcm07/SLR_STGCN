import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import matplotlib.ticker as mtick

# loss curve plot
def plot_loss_curve(avg_train_losses, avg_val_losses, file_path, name):
    fig = plt.figure(figsize=(10, 8))
    # plot train and validation losses
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss', linewidth=3)
    plt.plot(range(1, len(avg_val_losses) + 1), avg_val_losses, label='Validation Loss', linewidth=3)

    # find position of lowest validation loss
    minposs = avg_val_losses.index(min(avg_val_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('Epochs', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.ylim(0, 4.0)  # consistent scale
    plt.xlim(1, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.title('Training and Validation Loss', fontsize=36)
    save_path = os.path.join(file_path, name)
    fig.savefig(save_path, bbox_inches='tight')

# accuracy plot
def plot_accuracy_curve(train_accuracies, val_accuracies, avg_val_losses, file_path, name):
    fig = plt.figure(figsize=(10, 8))
    # plot train and validation accuracy
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', linewidth=3)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', linewidth=3)

    # find position of lowest validation loss
    minposs = avg_val_losses.index(min(avg_val_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('Epochs', fontsize=26)
    plt.ylabel('Accuracy', fontsize=26)
    plt.ylim(0, 1)  # consistent scale
    plt.xlim(1, len(train_accuracies) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.title('Training and Validation Accuracy', fontsize=36)
    # convert y-axis to percentage 0-100%
    ax = plt.subplot()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    save_path = os.path.join(file_path, name)
    fig.savefig(save_path, bbox_inches='tight')


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
    def __call__(self, val_loss, model, path, EPOCH, train_loss, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, EPOCH, train_loss, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
            
            if self.counter <= 5:
                self.save_5_consecutive_checkpoints(val_loss, model, path, EPOCH, train_loss, optimizer) 
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, EPOCH, train_loss, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, EPOCH, train_loss, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({'epoch':EPOCH,
            'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(), 
                    'loss':train_loss
                   }, path)
        self.val_loss_min = val_loss
        
    def save_5_consecutive_checkpoints(self, val_loss, model, path, EPOCH, train_loss, optimizer):
        ''''Saves 5 checkpoints after current best epoch'''
        path_split = path.split('.')
        path = path_split[0] + "_" + str(self.counter) + "." + path_split[1]
        print(path)
        torch.save({'epoch':EPOCH,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':train_loss
                   }, path)
        