# %%
import numpy 
import torch  
import torch.nn as nn    
import torch.nn.functional as F


#===============================================================================
# 2D module, Add here but not yet test. 
# 
#===============================================================================
class UpsampleBlock2D(nn.Module):
    """
    Upsample(x2, bilinear) + Conv2d  — no checkerboard artifact.
    Replaces ConvTranspose2d stride-2.

    Input : (B, in_channels,  H,   W)
    Output: (B, out_channels, H*2, W*2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.act  = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(self.up(x)))


class Decoder2D(nn.Module):
    """
    (B, latent_dim, h_latent, w_latent) → (B, Nx, Ny)

    Example with h_latent=w_latent=8, hidden_conv_dims=[256,128,64,64,32,32], Nx=Ny=513:
      Each UpsampleBlock2D doubles both spatial dims:
      8 → 16 → 32 → 64 → 128 → 256 → 512  (6 stages)
      Final F.interpolate resizes 512×512 → 513×513 exactly.

    Number of stages required:
      stages = ceil(log2(max(Nx, Ny) / h_latent))

    hidden_conv_dims must have exactly (stages) entries.
    """
    def __init__(self, latent_dim, 
                 hidden_conv_dims, 
                 output_sol_dim,
                 output_size):
        """
        latent_dim       : int          channel depth of latent input
        hidden_conv_dims : list[int]    channel widths per upsample stage
        output_sol_dim   : int          number of output channels (usually 1)
        output_size      : int or (int, int)
                           target spatial size, e.g. 513 or (513, 257)
        """
        super().__init__()

        # normalise output_size to (H, W)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

        dims   = [latent_dim] + hidden_conv_dims
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(UpsampleBlock2D(dims[i], dims[i + 1]))

        # final 1×1 conv → output channels
        blocks.append(nn.Conv2d(dims[-1], output_sol_dim, kernel_size=1))
        self.model = nn.Sequential(*blocks)

    def enforce_symmetry(self, y):
        """
        y: (B, C, H, W)
        Enforce symmetry along both spatial axes (x and y).
        Flip H axis (dim=-2) and W axis (dim=-1).
        """
        y_flipH = torch.flip(y, dims=[-2])          # flip along H
        y_flipW = torch.flip(y, dims=[-1])          # flip along W
        y_flipHW = torch.flip(y, dims=[-2, -1])    # flip both
        return 0.25 * (y + y_flipH + y_flipW + y_flipHW)

    def forward(self, x, apply_symmetry=True):
        """
        x   : (B, latent_dim, h_latent, w_latent)
        out : (B, Nx, Ny)   if output_sol_dim == 1
              (B, C, Nx, Ny) if output_sol_dim > 1  (squeeze removed)
        """
        out = self.model(x)                          # (B, C, H', W')

        # resize to exact output_size
        if out.shape[-2:] != self.output_size:
            out = F.interpolate(out, size=self.output_size,
                                mode="bilinear", align_corners=False)

        if apply_symmetry:
            out = self.enforce_symmetry(out)

        # (B, 1, Nx, Ny) → (B, Nx, Ny)
        if out.shape[1] == 1:
            out = out.squeeze(1)

        return out   # (B, Nx, Ny)  or  (B, C, Nx, Ny)


# ============================================================
# LatentModel 2D
# (B, P) → (B, latent_dim, h_latent, w_latent)
# ============================================================

class LatentModel2D(nn.Module):
    """
    MLP: params (B, input_dim) → latent (B, latent_dim, h_latent, w_latent)

    The MLP output is reshaped to a 2D spatial latent grid,
    matching what Decoder2D expects as input.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim,
                 h_latent, w_latent, activation=nn.ReLU,
                 output_activation=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.h_latent   = h_latent
        self.w_latent   = w_latent
        self.output_dim = latent_dim * h_latent * w_latent

        dims   = [input_dim] + list(hidden_dims) + [self.output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, input_dim)
        out = self.net(x)   # (B, latent_dim * h_latent * w_latent)
        return out.view(-1, self.latent_dim, self.h_latent, self.w_latent)

