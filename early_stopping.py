import torch
import numpy as np
class EarlyStopping:
    """Early stop to prevent overfitting, based on accuracy"""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): When the training epochs of the model exceeds this value and model performance is not improved, the model training will be stopped.
            verbose (bool): Whether to print early stop information.
            delta (float): A value. Only when the change is greater than this value, it is considered that the performance of the model has improved.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Save model checkpoint"""
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_acc_max = val_acc
