import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RandomWalk(nn.Module):
    def __init__(self, configs):
        super(RandomWalk, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.paramlist = nn.ParameterList()
        self.param = torch.nn.Parameter(torch.rand(1, 1, configs.enc_in), requires_grad=True).to(device)
        self.paramlist.append(self.param)

    def forward(self, x, _, ___, ____):
        pred = x[:, :, :]
        for walk in range(self.pred_len):
            pred = torch.cat((pred, pred[:, self.seq_len + walk - 1, :].unsqueeze(1) + self.param * 0), 1)
        return pred[:, -self.pred_len:, :]