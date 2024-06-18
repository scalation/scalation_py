import torch
from torch import nn
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs['d_model']
        self.layer_dim = configs['rnn_layer_dim']
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.c_out = configs['c_out']
        self.lstm = nn.LSTM(self.enc_in, self.d_model, self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.d_model, self.pred_len*self.c_out)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, use_tf = False):
        out, (hidden, cell) = self.lstm(x_enc)
        output = self.fc(hidden[-1])
        output = torch.reshape(output, (output.shape[0],self.pred_len, self.c_out))
        return output