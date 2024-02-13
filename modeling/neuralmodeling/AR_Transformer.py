import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DSW_embedding, enc_embedding, enc_embedding_enc, enc_embedding_dec
import numpy as np
import random
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from layers.RevIN import RevIN

import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.output_attention = configs['output_attention']
        self.label_len = configs['label_len']
        self.c_out = configs['c_out']
        self.seg_len_enc = configs['patch_len']
        self.stride = configs['stride']
        self.patch_len_dec = configs['patch_len_dec']
        self.stride_dec = configs['patch_len_dec']
        self.d_model = configs['d_model']
        self.segments = int(((configs['seq_len'] - configs['patch_len'])/configs['stride'])+1)
        self.freq = configs['freq']
        self.tf_ratio = configs['tf_ratio']
        self.revin_layer = RevIN(configs["enc_in"], affine=False, subtract_last=0)
        self.revin = configs["revin"]
        self.enc_value_embedding = enc_embedding(self.d_model, self.seq_len, configs['patch_len'], self.stride, configs['dropout'], freq = 'h')

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=configs['output_attention']), configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.dec_value_embedding = enc_embedding(self.d_model, self.pred_len, self.patch_len_dec, self.stride_dec, configs['dropout'], freq = 'h')

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs['factor'], attention_dropout=configs['dropout'], output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    AttentionLayer(
                        FullAttention(False, configs['factor'], attention_dropout=configs['dropout'], output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                )
                for l in range(configs['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs['d_model'], configs['patch_len_dec'])
        )
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, train=False): 
        batch_size = x_enc.shape[0]
        x_enc_temp = x_enc
        #seq_last = x_enc[:, -1:, :].detach()
        #x_enc = x_enc - seq_last
        #x_enc = x_enc[:, 1:, :] - x_enc[:, :-1, :]
        #x_enc = torch.cat([x_enc_temp[:, :1, :], x_enc], dim=1)
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        x_seq = self.enc_value_embedding(x_enc, x_mark_enc)
        x_seq = rearrange(x_seq, 'b ts_d seg_dec_num seg_len -> (b ts_d) seg_dec_num seg_len', ts_d = self.c_out, b = batch_size)
        enc_out, attns = self.encoder(x_seq)
        #decoder_input = torch.zeros(x_enc.shape[0],self.patch_len_dec,x_enc.shape[-1], device = 'cuda:0')
        decoder_input = x_enc[:,-self.patch_len_dec:,:]
        outputs = []
        for t in range(0, int(self.pred_len/self.patch_len_dec)):
            true_input = decoder_input
            dec_in = self.dec_value_embedding(decoder_input, x_mark_dec[:,:self.label_len+t*self.patch_len_dec+self.patch_len_dec,:])
            dec_in = rearrange(dec_in, 'b ts_d seg_dec_num d_model -> (b ts_d) seg_dec_num d_model', b = batch_size)
            dec_out = self.decoder(dec_in, enc_out)
            dec_out = rearrange(dec_out, '(b ts_d) seg_dec_num seg_len -> b ts_d seg_dec_num seg_len', b = batch_size)
            dec_out = rearrange(dec_out, 'b ts_d seg_dec_num seg_len -> b (seg_dec_num seg_len) ts_d', ts_d = self.c_out)
            r = random.random()
            if r < self.tf_ratio and train is True:
                decoder_input = torch.cat((true_input, x_dec[:, self.label_len + (t * self.patch_len_dec):self.label_len + (t * self.patch_len_dec) + self.patch_len_dec, :]), dim=1)
            else:
                decoder_input = torch.cat((true_input, dec_out[:, -self.patch_len_dec:, :]), 1)
            outputs.append(dec_out[:, -self.patch_len_dec:, :])
        outputs = torch.cat(outputs, dim=1)
        #cumulative_sum = torch.cumsum(outputs, dim=1)
        #final_prediction = torch.zeros_like(outputs)
        #final_prediction[:, 0, :] = x_enc_temp[:, -5:, :].mean(dim=1)
        #final_prediction[:, 1:, :] = x_enc_temp[:, -5:, :].mean(dim=1).unsqueeze(1) + cumulative_sum[:, :-1, :]
        if self.revin:
            outputs = self.revin_layer(outputs, 'denorm')
        if self.output_attention:
            return outputs, attns
        else:
            return outputs
