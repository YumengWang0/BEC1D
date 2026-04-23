from lib.module import * 
from lib.train_module import * 
import torch 
import torch.nn as nn 
import numpy as np 

def main(train_loader, val_loader, config, device  ):
    # %%
    
    

    # %%
    model = latent_models(input_dim = config["input_dim"], 
                        hidden_dims= config["hidden_dims"],
                        latent_feature_dim=  config["latent_feature_dim"], 
                        n_latent= config["n_latent"],  
                        activation= config["activation"], 
                        output_activation=config["output_activation"]  )
 
 
    decoder_model = decoder(latent_dim = config["latent_feature_dim"],
            hidden_conv_dims = config["hidden_conv_dims"],
            output_sol_dim = config["output_sol_dim"]).to(device)

    models = nn.ModuleList([
    model,
    decoder_model ]).to(device)

    trainer(models, train_loader, val_loader, config, device)



# %%
if "__name__" == "main": 
 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "num_layers": 4,
        "hidden_dims": [128, 128, 128], 
        "latent_feature_dim":16,
        "n_latent":16,
        
        "hidden_conv_dims": [128, 128, 128, 128, ],

        
        "activation":nn.ReLU,
        "output_activation": None, 
        "learning_rate": 1e-3,

        "batch_size": 32,
        "epochs": 100,
        "weight_decay": 1e-5,
        "dropout": 0.1,


        # data
        "input_dim": 2,
        "output_sol_dim": 1
        
        } 
    
    train_loader= 
    val_loader = 
    main() 