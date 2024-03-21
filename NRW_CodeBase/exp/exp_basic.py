import os
import torch
import numpy as np


class Exp_Basic(object):
    """
    Exp_basic and Exp_Main are python files containing different experiments and configurations for the models. 
    These files contain scripts, configurations, and dataset preprocessing managers for running experiments with the models on 
    various time series forecasting tasks. 
    
    The exp_basic file contains basic or standard configurations and experiments for the models, while the exp_main directory 
    contains more advanced or specific experiments that explore different settings or variations of the models.
     
    These files help organize the codebase and make it easier to run different experiments and compare results.
    """
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            #print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass