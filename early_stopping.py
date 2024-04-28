import torch
import numpy as np
class EarlyStopping:
    """早停来防止过拟合，基于准确率"""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): 指标停止提升后等待多少个epoch后停止训练。
            verbose (bool): 是否打印早停时的信息。
            delta (float): 最小变化量，只有变化大于这个值时才认为是一个提升。
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
        """保存模型"""
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_acc_max = val_acc