import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FFNN(nn.Module):
    def __init__(self, configs):
        super(FFNN, self).__init__()
        units_layer1 = configs.units_layer1
        units_layer2 = configs.units_layer2
        lags = configs.seq_len
        horizons = configs.pred_len
        n_features = configs.enc_in
        self.linear1 = nn.Linear(lags * n_features, units_layer1)
        self.linear2 = nn.Linear(units_layer1, units_layer2)
        self.linear3 = nn.Linear(units_layer2, horizons * n_features)
        self.horizons = horizons
        self.n_features = n_features

    def forward(self, src, trg, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=False):
        src = torch.flatten(src, start_dim=1)
        src = torch.relu(self.linear1(src))
        src = torch.relu(self.linear2(src))
        output = self.linear3(src)
        output = torch.reshape(output, (output.shape[0], self.horizons, self.n_features)).to(device)
        return output