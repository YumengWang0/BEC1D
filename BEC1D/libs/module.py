# %%
import numpy 

import torch  
import torch.nn as nn    
import torch.nn.functional as F
 

# ============================================================
# Decoder model, decoded from the latent representation 
# ============================================================



class UpsampleBlock(nn.Module):
    """
    Upsample(x2, linear) + Conv1d  — no checkerboard artifact.
    Replaces ConvTranspose1d stride-2.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.act  = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(self.up(x)))


class Decoder(nn.Module):
    """
    (B, latent_dim, n_latent) → (B, N)

    n_latent=64, hidden_conv_dims=[256,128,64,64,32,32], N=4097
    Each UpsampleBlock doubles the spatial dim:
      64 → 128 → 256 → 512 → 1024 → 2048 → 4096
    Final F.interpolate resizes 4096 → 4097 exactly.
    """
    def __init__(self, latent_dim, hidden_conv_dims, output_sol_dim,
                 output_size=4097):
        super().__init__()
        self.output_size = output_size

        dims   = [latent_dim] + hidden_conv_dims
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(UpsampleBlock(dims[i], dims[i + 1]))

        # final 1×1 conv → output channels
        blocks.append(nn.Conv1d(dims[-1], output_sol_dim, kernel_size=1))
        self.model = nn.Sequential(*blocks)

    def enforce_symmetry(self, y):
        """y: (B, C, L)"""
        return 0.5 * (y + torch.flip(y, dims=[-1]))

    def forward(self, x, apply_symmetry=True):
        out = self.model(x)                  # (B, 1, ~4096)

        # resize to exact output_size (handles 4096 → 4097)
        if out.shape[-1] != self.output_size:
            out = F.interpolate(out, size=self.output_size,
                                mode="linear", align_corners=False)

        if apply_symmetry:
            out = self.enforce_symmetry(out)

        return out.permute(0, 2, 1).squeeze(-1)   # (B, N)






# ============================================================
# Latent model: Mapping from the parameters 
# to a latent represnetation  with (n_latent, feature_dimension)
# ============================================================
 
class LatentModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dims,
                 latent_feature_dim , n_latent, 
                 activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.n_latent = n_latent
        self.latent_feature_dim = latent_feature_dim 
        self.output_dim = n_latent  * latent_feature_dim 
        layers = []
        
        dims = [input_dim] + list(hidden_dims) + [self.output_dim]
      
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x is the (B, number_of_parameters )
        output = self.net(x) 
        
        output = output.view(-1, self.latent_feature_dim, self.n_latent)
        return output 
    
