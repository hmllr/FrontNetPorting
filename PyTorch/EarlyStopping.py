# from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import numpy as np
from ModelManager import ModelManager
import logging

#Dario's code!!!
#But I edited it and added my own stuff :)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, file_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, file_name)
        elif score < self.best_score:
            self.counter += 1
            logging.info("[EarlyStopping] counter: {} out of {}".format(self.counter, self.patience))
            self.save_checkpoint(val_loss, model, epoch, file_name)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, file_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, file_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info("[EarlyStopping]  Validation loss decreased {} --> {}. Saving model as {}".format(self.val_loss_min, val_loss, file_name))
        ModelManager.Write(model, epoch, file_name)
        self.val_loss_min = val_loss