# %%
import numpy 

import torch  
import torch.nn as nn    
 


# %% [markdown]
# ## Requirment 
# 1.  We will be working on a system in free space, so the V(x) = 0.
#  
# 2. The predicted solution must satisfy a symmetry constraint (which may need to be enforced, e.g., in the decoder);
# 
# 3. the solution should satisfy a prescribed (L^2) norm, i.e., |\psi|^2 = N, where N depends on parameters (i.e., it is not an independent parameter). 


# %%
class decoder(nn.Module): 
    
    def __init__(self,  latent_dim, hidden_conv_dims, output_sol_dim):
        super().__init__()
        self.latent_dimension = latent_dim 
        
    
        dims = [latent_dim] + hidden_conv_dims

        layers = []

        for i in range(len(dims) - 1):
            layers.append(
                nn.ConvTranspose1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            layers.append(nn.ReLU())

       
        layers.append(
            nn.Conv1d(dims[-1], output_sol_dim, kernel_size=1)
        )

        self.model = nn.Sequential(*layers)

        print(self.model)
    
    def enforce_even_symmetry(self, y):
        """
        y: (B, C, L)
        """
        y_flip = torch.flip(y, dims=[-1])   # reverse spatial dimension
        return 0.5 * (y + y_flip)
        
    
    def enforce_l2_norm(self, y, N=1.0, eps=1e-12):
        """
        y: (B, C, L)
        N: desired L2 norm squared
        """
        # compute L2 norm squared over spatial dim
        norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)  # (B, C, 1)

        # avoid division by zero
        scale = torch.sqrt(N / (norm_sq + eps))

        return y * scale
        

        
    def forward(self, x):
        output = self.model(x)
        # enforce the symmetry 
        
        output_symmetry = self.enforce_even_symmetry(output) 
        
        output_symmetry_normalized = self.enforce_l2_norm( output_symmetry )
         
        return output_symmetry_normalized.permute(0, 2, 1)
            

# %%
class latent_models(nn.Module):
    
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
