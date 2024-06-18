import numpy as np
import torch

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def adjust_learning_rate(optimizer, epoch, args):
    if args["lradj"] == 'constant':
        lr_adjust = {epoch: args["learning_rate"]}
    elif args["lradj"] == 'type1':
        lr_adjust = {epoch: args["learning_rate"] * (0.5 ** ((epoch - 1) // 1))}
    elif args["lradj"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args["lradj"] == 'type3':
        lr_adjust = {epoch: args["learning_rate"] if epoch < 3 else args["learning_rate"] * (0.9 ** ((epoch - 3) // 1))}
    elif args["lradj"] == 'type4':
        lr_adjust = {epoch: args["learning_rate"] * (args["decay_fac"] ** ((epoch) // 1))}
    elif args["lradj"] == 'type5':
        lr_adjust = {epoch: args["learning_rate"] if epoch < 15 else args["learning_rate"]*0.1}
    elif args["lradj"] == 'type6':
        lr_adjust = {epoch: args["learning_rate"] if epoch < 50 else args["learning_rate"]*0.1}
    elif args["lradj"] == 'type7':
        lr_adjust = {2: args["learning_rate"] * 0.5 ** 1, 4: args["learning_rate"] * 0.5 ** 2,
                     6: args["learning_rate"] * 0.5 ** 3, 8: args["learning_rate"] * 0.5 ** 4,
                     10: args["learning_rate"] * 0.5 ** 5}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def save_results(self, file_path, df):
    with open(file_path, 'a') as file:
        file.write('\n' + self.args['model_name'] + ' ' + self.args['validation'] +'\n')
        df.to_csv(file, header=True, index=False)

