import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        hidden_dim = configs.d_model
        batch_size = configs.batch_size
        layer_dim = configs.e_layers
        n_past = configs.seq_len
        n_future = configs.pred_len
        features = configs.enc_in
        self.n_future = configs.pred_len
        self.enc_in = configs.enc_in
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.n_future = n_future
        self.lstm = nn.LSTM(features, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_future * self.enc_in)

    def forward(self, src, trg, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=False):
        h0 = torch.zeros(self.layer_dim, src.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, src.size(0), self.hidden_dim).to(device)
        out, (hidden, cell) = self.lstm(src, (h0, c0))
        out = self.fc(out[:, -1, :]).reshape(out.shape[0], self.n_future, self.enc_in)
        return out