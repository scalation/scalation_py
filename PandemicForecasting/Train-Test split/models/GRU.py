import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GRU(nn.Module):
    """
    A Gated Recurrent Unit (GRU) consists of a GRU layer followed by a fully connected (linear) layer.
    """
    def __init__(self, configs):
        super(GRU, self).__init__()
        hidden_dim = configs.d_model
        layer_dim = configs.e_layers
        n_future = configs.pred_len
        horizons = configs.pred_len
        features = configs.enc_in
        hidden_size = hidden_dim
        self.n_future = configs.pred_len
        self.enc_in = configs.enc_in
        self.hidden_dim = hidden_dim
        self.n_future = n_future
        self.hidden_size = hidden_dim
        self.num_layers = layer_dim
        num_layers = layer_dim
        self.gru = nn.GRU(features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizons * self.enc_in)

    def forward(self, src, trg, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=False):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(device)
        # Forward propagate GRU
        out, _ = self.gru(src, h0)
        # Decode the hidden state of the last time step

        out = self.fc(out[:, -1, :]).reshape(out.shape[0], self.n_future, self.enc_in)
        return out
