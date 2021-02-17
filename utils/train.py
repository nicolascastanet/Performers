import os.path

import numpy as np
from tqdm.auto import tqdm
import torch

class InfiniteLoader():
    """
    To iterate indefinitely on a loader
    """
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
        
    def get_batch(self):
        try:
            x = next(self.iterator)
        except:
            self.iterator = iter(self.loader)
            x = next(self.iterator)
        return x


def train(train_loader, test_loader, model, optimizer, criterion, nb_step, nb_step_val, interval_step_val, \
            path=None, path_early_stopping=None, patience=20, verbose=False):
    
    ############################
    # Prepare verbose settings #
    ############################
    
    progress_bar = tqdm if verbose else lambda first_arg, **kwargs: first_arg
    
    ###################
    # Infinite Loader #
    ###################
    train_batchs = InfiniteLoader(train_loader)
    test_batchs = InfiniteLoader(test_loader)
    
    ##################
    # Initialisation #
    ##################
    if(path!=None and os.path.isfile(path)):
        
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint["model_params"])
        
        if("early_stopping" in checkpoint):
            early_stopping = checkpoint["early_stopping"]
            path_early_stopping = checkpoint["path_early_stopping"]
        
        current_step = checkpoint["current_step"]
        
        
        optimizer.load_state_dict(checkpoint["optimizer_params"])
        
        train_loss = checkpoint["train_loss"]
        eval_loss = checkpoint["eval_loss"]
    else:

        current_step = 0

        train_loss = []
        eval_loss = []
        
        if(path_early_stopping is not None):
            early_stopping = EarlyStopping(patience=patience, verbose=verbose, delta=1e-10, path=path_early_stopping)
        
    
    ############
    # Training #
    ############
    tmp_train_loss = []
    model.train()
    for step in progress_bar(range(current_step, nb_step), initial=current_step, total=nb_step):
        
        data, labels = train_batchs.get_batch()

        optimizer.zero_grad()

        outputs = model(data)


        loss = criterion(outputs, labels)
        tmp_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        ##############
        # Evaluation #
        ##############
        if(step%interval_step_val==0 or step==nb_step-1):
            
            with torch.no_grad():
                
                tmp_eval_loss = []
                model.eval()
                for step_val in range(nb_step_val):
                    
                    data, labels = test_batchs.get_batch()

                    outputs = model(data)

                    loss = criterion(outputs, labels)
                    tmp_eval_loss.append(loss.item())

                eval_loss.append(np.mean(tmp_eval_loss)) 
            model.train()
            
            ##################
            # Early stopping #
            ##################
            if(path_early_stopping is not None):
                early_stopping(eval_loss[-1], model)
            
            ##############################
            # Saving Parameters and loss #
            ##############################
            train_loss.append(np.mean(tmp_train_loss))
            
            tmp_train_loss = []
            
            if(path!=None):
                checkpoint = {
                    "model": model.__class__,
                    "model_params": model.state_dict(),

                    "current_step": step+1,

                    "optimizer": optimizer.__class__,
                    "optimizer_params": optimizer.state_dict(),

                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                }
                if(path_early_stopping is not None):
                    checkpoint["early_stopping"] = early_stopping
                    checkpoint["path_early_stopping"] = path_early_stopping
                    
                torch.save(checkpoint, path)
                
            if(path_early_stopping is not None and early_stopping.early_stop):
                print("Early stopping")
                break
    
    if(path_early_stopping is not None):
        model.load_state_dict(torch.load(path_early_stopping))
            
    return train_loss, eval_loss


def get_prediction(model, loader, nb_step=None, verbose=False):
    
    progress_bar = tqdm if verbose else lambda first_arg, **kwargs: first_arg
    
    batchs = InfiniteLoader(loader)
    
    predictions = torch.tensor([])
    targets = torch.tensor([])
    
    with torch.no_grad():
        for step in progress_bar(range(nb_step)):
            
            data, labels = batchs.get_batch()
            outputs, _ = model(data)
            predictions = torch.cat((predictions, outputs.argmax(dim=1)))
            targets = torch.cat((targets, labels))
        
    return predictions, targets

def get_all_prediction(model, loader, verbose=False):
    
    progress_bar = tqdm if verbose else lambda first_arg, **kwargs: first_arg
    
    predictions = torch.tensor([])
    targets = torch.tensor([])
    
    with torch.no_grad():
        for data, labels in progress_bar(loader):
            outputs = model(data)
            predictions = torch.cat((predictions, outputs.argmax(dim=1)))
            targets = torch.cat((targets, labels))
        
    return predictions, targets


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=1e-7, path='checkpoint.pt', trace_func=print):
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
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss