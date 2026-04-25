import datetime
import torch
import torch.nn as nn
import libs
from libs.data_gene import *
from libs.config import * 
from libs.module import LatentModel, Decoder
from libs.train_module import trainer

def main():

    #now = datetime.datetoime.now()
    """
    config = {
        "eq_name": config.eq_name,
        
        # model
        "hidden_dims":       [128, 128, 128],
        "latent_feature_dim": 16,
        "n_latent":           64,
        "hidden_conv_dims":  [256, 128, 64, 64, 32, 32],
        "activation":         nn.ReLU,
        "output_activation":  None,
        "dropout":            0.1,

        # training
        "epochs":      5,
        "epoch_save":  1,
        "batch_size":  32,

        # optimizer
        "learning_rate": 1e-3,
        "weight_decay":  1e-5,
        "step_size":     1,
        "gamma":         0.99,

        # data / naming
        "input_dim":      2,
        "output_sol_dim": 1,
      
    }
    """

    # log path set after eq_name is defined
    #config["log_path"] = f"./logs/train_{config['eq_name']}_{now:%Y-%m-%d_%H%M%S}.log"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #sol_name = f"./data/solution_{config['eq_name']}.h5"

    print(f"Using device : {device}")
    print(f"Loading data : {config["sol_name"]}")

    # ── data ─────────────────────────────────────────────────
    x_ref, train_loader, test_loader = get_dataloaders(
        config["sol_name"], batch_size=config["batch_size"]
    )

    for params_batch, phi_batch in train_loader:
        print("train params :", params_batch.shape)   # (B, 2)
        print("train phi    :", phi_batch.shape)       # (B, N)
        break

    # ── models ───────────────────────────────────────────────
    model1 = LatentModel(
        input_dim          = config["input_dim"],
        hidden_dims        = config["hidden_dims"],
        latent_feature_dim = config["latent_feature_dim"],
        n_latent           = config["n_latent"],
        activation         = config["activation"],
        output_activation  = config["output_activation"],
    ).to(device)

    model2 = Decoder(
        latent_dim       = config["latent_feature_dim"],
        hidden_conv_dims = config["hidden_conv_dims"],
        output_sol_dim   = config["output_sol_dim"],
    ).to(device)

    models = nn.ModuleList([model1, model2])

    # ── train ────────────────────────────────────────────────
    trainer(
        models       = models,
        train_loader = train_loader,
        test_loader  = test_loader,
        config       = config,
        logs         = None,
        device       = device,
    )


if __name__ == "__main__":
    main()
