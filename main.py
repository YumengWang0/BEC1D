
import torch 
import torch.nn as nn 
import numpy as np 
from libs.data_gene import * 
from libs.module import *
from libs.train_module import * 
from libs.evaluate_module import * 
import matplotlib.pyplot as plt
from libs.config import *
# %% [markdown]
# \begin{equation}
# \Psi_{D2}(x, \mu) =
# \frac{3\mu}{
# \delta \left(
# 1 + \sqrt{1 + \frac{9\mu}{2\delta^2}}
# \cosh\left(\sqrt{-2\mu}(x - x_0)\right)
# \right)}
# \end{equation}


def main():

    # %%
    # ============================================================
    # Build train / test
    # ============================================================
    
    if  not os.path.exists(config["sol_name"]):
        print(f"[DATA] {config['sol_name']} not found → generating...")

        train_pairs = build_train_pairs(TRAIN_CONFIG)
        test_pairs  = build_test_pairs(TEST_CONFIG)

        print(f"Train pairs : {len(train_pairs)}")
        print(f"Test  pairs : {len(test_pairs)}")

        train_data = build_dataset(train_pairs, PHYSICS_DOMAIN)
        test_data  = build_dataset(test_pairs,  PHYSICS_DOMAIN)

        print(f"\nTrain phi shape : {train_data['phi'].shape}")
        print(f"Test  phi shape : {test_data['phi'].shape}")
        

        save_hdf5( config["sol_name"], train_data, test_data, now)
        print(f"\nSaved → config['sol_name']")

    else: 

        print(f"[DATA] Found {config['sol_name']} → skip generation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # ============================================================
    # Data Load 
    # ============================================================
    x_ref, train_loader, test_loader = get_dataloaders(config["sol_name"], batch_size=config["batch_size"])

    # verify
    for params_batch, phi_batch in train_loader:
        print("train params :", params_batch.shape)   # (B, 2)
        print("train phi    :", phi_batch.shape)       # (B, N)
        break

    for params_batch, phi_batch in test_loader:
        print("test  params :", params_batch.shape)
        print("test  phi    :", phi_batch.shape)
        break 


    # ============================================================
    # Model Load 
    # ============================================================
    
    model = LatentModel(input_dim = config["input_dim"], 
                        hidden_dims= config["hidden_dims"],
                        latent_feature_dim=  config["latent_feature_dim"], 
                        n_latent= config["n_latent"],  
                        activation= config["activation"], 
                        output_activation=config["output_activation"]  )


    decoder_model = Decoder(latent_dim = config["latent_feature_dim"],
            hidden_conv_dims = config["hidden_conv_dims"],
            output_sol_dim = config["output_sol_dim"])


    models = nn.ModuleList([  model  ,  decoder_model])

    # ── train ────────────────────────────────────────────────
    if config["model_train"]: 
        trainer(
            models       = models,
            train_loader = train_loader,
            test_loader  = test_loader,
            config       = config,
            logs         = None,           # replace with wandb.init() if needed
            device       = device,
        )
    else: 
        params_all, phi_pred, phi_true, rel_errors = evaluate_model(models, test_loader, device)

        #params = np.concatenate([test_data['mu'][:, None], test_data['delta'][:, None]], axis = 1) 
      
        plot_results(x_ref, params_all, phi_pred, phi_true,
                 indices=None, save_path=config["save_path"]) 

if __name__ == "__main__":
    main() 


 
