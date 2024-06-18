import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_orthogonal_polynomials(seq_length, num_polynomials, device):
    # Normalize t to the range [-1, 1]
    t = torch.linspace(-1, 1, steps=seq_length, dtype=torch.float32, device=device)
    polynomials = [torch.ones_like(t), t]
    
    # Generate Chebyshev polynomials up to the specified degree
    for n in range(2, num_polynomials + 1):
        p_n = 2 * t * polynomials[-1] - polynomials[-2]
        polynomials.append(p_n)
    
    # Stack polynomials for all harmonics, excluding the first constant polynomial for variety
    polynomial_features = torch.stack(polynomials[1:], dim=0)  # Skip T0
    return polynomial_features.transpose(0, 1)  # Transpose to match input data shape

def preprocess_input_data_with_orthogonal(input_data, num_polynomials):
    seq_length = input_data.shape[1]  # Assuming [Batch, Seq_Length, Features]
    device = input_data.device
    
    polynomial_features = compute_orthogonal_polynomials(seq_length, num_polynomials, device)
    polynomial_features = polynomial_features.unsqueeze(0).expand(input_data.shape[0], -1, -1)
    
    input_data_with_polynomials = torch.cat([input_data, polynomial_features], dim=-1)
    return input_data_with_polynomials


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.Linear = nn.Linear(self.seq_len, self.pred_len, bias = True)
        #self.activation = nn.GELU()
        #self.layer_norm = nn.LayerNorm(self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, use_tf=True):
        """x = x_enc
        # x: [Batch, Input length, Channel]
        #seq_last = x[:, -1:, :].detach()
        #x = x - seq_last
        #num_polynomials = 10
        #x = preprocess_input_data_with_orthogonal(x, num_polynomials)
        #x = x_enc.sum(dim = -1).unsqueeze(-1)
        x = self.activation(self.Linear(x.permute(0, 2, 1)))
        x = self.layer_norm(x).permute(0, 2, 1)
        #x = x + seq_last"""
        
        x = x_enc
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        
        return x
