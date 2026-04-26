# %%
import numpy 

import torch  
import torch.nn as nn    
 

# ============================================================
# Decoder model, decoded from the latent representation 
# ============================================================

 
class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_conv_dims, output_sol_dim):
        super().__init__()
        self.latent_dimension = latent_dim

        dims = [latent_dim] + hidden_conv_dims  # 需要 8 个间隔 → len=9

        layers = []
        for i in range(len(dims) - 1):
            is_last_upsample = (i == len(dims) - 2)  # 最后一个上采样层

            layers.append(
                nn.ConvTranspose1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=1 if is_last_upsample else 0  # ← 只在最后加
                )
            )
            layers.append(nn.ReLU())

        layers.append(
            nn.Conv1d(dims[-1], output_sol_dim, kernel_size=1)
        )
        self.model = nn.Sequential(*layers)
        print(self.model)


    def enforce_symmetry(self, y):
        """
        y: (B, C, L)
        """
        y_flip = torch.flip(y, dims=[-1])   # reverse spatial dimension
        return 0.5 * (y + y_flip)


    def forward(self, x):
        output = self.model(x)
        output_symmetry = self.enforce_symmetry(output) 
        
         
        return output_symmetry.permute(0, 2, 1).squeeze(-1)
    
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
    