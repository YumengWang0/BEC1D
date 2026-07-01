"""
configs/Ex2.py  —  
Experiment 2: The output solution is a vector of two elements, 
the real and imaginary parts of the solution. 
"""
import os
import datetime
import torch.nn as nn

now = datetime.datetime.now()

# Equation name  
EQ_NAME     = "Ex2"

# How you select the data 
DATA_SOURCE = "analytical"

# The physic information of the experiment 
PHYSICS_DOMAIN = {
    "x_a": -20,
    "x_b":  20,
    "N":    128,
    "x0":   0,
}

# Parameters name 
PARAM_NAMES  = ["mu", "delta"]


# Training parameters selection for the analytical types of data 
TRAIN_CONFIG = {
    "mu_left":  -0.2222,
    "mu_right": -0.05,
    "N_mu":     20,
    
    "delta_max":  1.1,
    "N_test":     120,
    "seed":    20260425,
}

# For the .mat data, unused for analytical 
MAT_CONFIG = {}   


# Model and training hyperparameters 
config = {
    "eq_name":    EQ_NAME,
    "model_train": False,
    # Loss function: "mse" | "relative_l2"
    "criterion": "relative_l2",


    # Architecture
    "hidden_dims":        [128, 128, 128],
    "latent_feature_dim": 16,
    "n_latent":           64,
    "hidden_conv_dims":   [256, 128, 128, 64, 64, 32, 32],
    "activation":         nn.ReLU,
    "output_activation":  None,
    "learn_boundary":     False,
    "complex_solution":           True,  # The output solution is complex-valued, represented as a vector of two elements (real and imaginary parts)    
 
    # Training
    "epochs":      2000,
    "epoch_save":  200,
    "batch_size":  32,


    # Optimiser
    "learning_rate": 1e-3,
    "step_size":     1,
    "weight_decay":  1e-5,
    "gamma":         0.998,

    # Change unless predict more than one solution
    "input_dim":      len(PARAM_NAMES),
    "output_sol_dim": 2, ### Revise here for image and real 
    "output_size" : PHYSICS_DOMAIN["N"]
}

# Path create for the data, log, checkpoints and result
def build_paths(cfg):
    name = cfg["eq_name"]
    ts   = now.strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(f"./data/{name}",        exist_ok=True)
    os.makedirs(f"./logs/{name}",        exist_ok=True)
    os.makedirs(f"./result/{name}",      exist_ok=True)
    os.makedirs(f"./checkpoints/{name}", exist_ok=True)
    cfg["sol_name"]            = f"./data/{name}/solution.h5"
    cfg["log_path"]            = f"./logs/{name}/train_{ts}.log"
    cfg["save_path"]           = f"./result/{name}/plot.png"
    cfg["save_path_witherror"] = f"./result/{name}/witherror.png"
    cfg["model_save_path"]     = f"./checkpoints/{name}/best_model.pt"
    cfg["error_save_path"]     = f"./result/{name}/RelL2.npz"

build_paths(config)
