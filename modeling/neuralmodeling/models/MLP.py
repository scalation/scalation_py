from torch import nn

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(configs['seq_len'], configs['d_model'])
        self.linear2 = nn.Linear(configs['d_model'], configs['d_model']//2)
        self.linear3 = nn.Linear(configs['d_model']//2, configs['d_model']//4)
        self.linear4 = nn.Linear(configs['d_model']//4, configs['pred_len'])
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
    def forward(self, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False, count=-1):
        x_enc = x_enc.permute(0,2,1)
        output = self.linear1(x_enc)
        output = self.act1(self.linear2(output))
        output = self.act2(self.linear3(output))
        output = self.linear4(output)
        output = output.permute(0,2,1)
        return output