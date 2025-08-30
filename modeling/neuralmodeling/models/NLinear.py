import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.subtract_last = configs['subtract_last']
        self.Linear = nn.Linear(self.seq_len, self.pred_len, bias = True)

    def forward(self, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False, count=-1):
        x = x_enc
        if self.subtract_last:
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.subtract_last:
            x = x + seq_last

        return x