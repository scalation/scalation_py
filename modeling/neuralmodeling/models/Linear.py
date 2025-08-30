import torch.nn as nn
from modeling.neuralmodeling.layers.RevIN import RevIN

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin = configs.revin
        self.subtract_last = configs.subtract_last
        if self.revin: self.revin_layer = RevIN(7, affine=False, subtract_last=False)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.target_idx = configs.target_idx

    def forward(self, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False, count=-1):
        if self.revin: 
            x_enc = self.revin_layer(x_enc, 'norm')
        elif self.subtract_last:
            seq_last = x_enc[:, -1:, :].detach()
            x_enc = x_enc - seq_last
        x = x_enc
        x = x.permute(0,2,1)
        x = self.Linear(x)
        x = x.permute(0,2,1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        elif self.subtract_last:
            x = x + seq_last
        return x