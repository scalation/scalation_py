import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.Linear = nn.Linear(self.seq_len, self.pred_len, bias = True)
        #self.activation = nn.GELU()
        #self.layer_norm = nn.LayerNorm(self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, use_tf=True):
        x = x_enc
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        
        return x
