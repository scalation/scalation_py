import torch
import torch.nn as nn
import math

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, subtract_last=False,
                 per_h_enable = True,
                 per_h_cutoff =6,
                 input_blend=True,           # True -> blend last-K vs mean on inputs
                 blend_mode='hard',          # 'hard'
                 blend_tail_steps=6         # K: last K time steps use subtract_last
                 ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.subtract_last = subtract_last

        self.per_h_enable = per_h_enable  
        self.per_h_cutoff = per_h_cutoff    

        # --- NEW: input blending state ---
        self.input_blend = input_blend
        self.blend_mode = blend_mode
        self.blend_tail_steps = blend_tail_steps
      
        print("per_h_enable: "+str(per_h_enable)+" , per_h_cutoff: "+str(per_h_cutoff)+" , input_blend: "
              +str(input_blend)+
              " , blend_mode: "+str(blend_mode)+" , blend_tail_steps: "+str(blend_tail_steps))
        

    # public knobs
    def set_per_h(self, enable: bool, cutoff: int = 6):
        self.per_h_enable = bool(enable)
        self.per_h_cutoff = int(cutoff)

    def forward(self, x, mode: str):
        # x: [B,T,C] when 'norm'; [B,H,C] when 'denorm'
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x


    def _get_statistics(self, x):
     
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()      # [B,1,C]
        self.last = x[:, -1, :].unsqueeze(1).detach()                         # [B,1,C]
        var  = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False).detach()
        self.stdev = torch.sqrt(var + self.eps).detach()                      # [B,1,C]


    def _normalize(self, x):
        
        if not self.input_blend:
            if self.subtract_last:
                z = (x - self.last) / self.stdev
            else:
                z = (x - self.mean) / self.stdev
            x_out = z
        else:
            z_last = (x - self.last) / self.stdev
            z_mean = (x - self.mean) / self.stdev

            B, T, C = x.shape
            if self.blend_mode == 'hard':
                # mask 1 on last K steps, 0 elsewhere
                K = min(max(self.blend_tail_steps, 0), T)
                m = x.new_zeros(B, T, 1)
                if K > 0:
                    m[:, -K:, :] = 1.0
                z = m * z_last + (1.0 - m) * z_mean
            x_out = z
        return x_out

    def _denormalize(self, x):
 
        if self.per_h_enable and x.ndim == 3: #only works when blend_mode = hard

            B, H, C = x.shape
            stdBH  = self.stdev.expand(B, H, C) #broadcasting
            meanBH = self.mean.expand(B, H, C)
            lastBH = self.last.expand(B, H, C)
            idx = torch.arange(1, H+1, device=x.device).view(1, H, 1).expand(B, H, C)
            short = (idx <= self.per_h_cutoff)
            #short = [True, True, True, True, True, True, False, False, False, False, False, False] (cutoff=6, h=12)

            out = torch.empty_like(x)
            out[ short] = x[ short] * stdBH[ short] + lastBH[ short]
            out[~short] = x[~short] * stdBH[~short] + meanBH[~short] #negated
            return out
        
        elif self.per_h_enable == False and x.ndim == 3: 
            
            if self.subtract_last:
                x = x * self.stdev
                x = x + self.last
            else:
                x = x * self.stdev
                x = x + self.mean
            return x
