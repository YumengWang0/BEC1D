import datetime 
import torch.nn as nn

now  = datetime.datetime.now()

# %%
# ============================================================
# Config for data 
# ============================================================
RANDOM_SEED = 20260425

PHYSICS_DOMAIN = {
    "x_a": -20,
    "x_b":  20,
    "N": 4097,
    "x0":    0,
}

TRAIN_CONFIG = {
    "mu_left":  -0.2222,
    "mu_right": -0.05,
    "N_mu":      20, 
}


TEST_CONFIG = {
    "mu_left":  -0.2222,
    "mu_right": -0.05,
    "delta_max":  1.1,
    "N_test":     120,       # 总随机采样数
    "seed":    RANDOM_SEED,
}

# %%
# ============================================================
# Config for  model and model training 
# ============================================================
config = {
        "model_train":False,
    "eq_name": "Ex1", 
    
    # Model parameters 
    "hidden_dims": [128, 128, 128], 
    "latent_feature_dim":16,
    "n_latent":64,
    "hidden_conv_dims": [256, 128 ,128, 64, 64, 32, 32],  # [64, 64, 64, 128, 128,128,  256],  #[256,128, 128, 64, 64, 32, 32],
    "activation":nn.ReLU,
    "output_activation": None, 
    "dropout": 0.1,
    
    
    # Training parameters 
    "epochs" : 1501, 
    "epoch_save": 500, 
    "batch_size": 32,

    # Optimizer parameters 
    "learning_rate": 1e-3,
    "step_size": 100,
    "weight_decay": 1e-5,
    "gamma":0.75, 


    # data
    "input_dim": 2,
    "output_sol_dim": 1

} 

config["log_path"] = f"./logs/train_{config['eq_name']}_{now:%Y-%m-%d_%H%M%S}.log"
config["sol_name"] = f"./data/solution_{config['eq_name']}.h5"
config["save_path"] = f"./result/{config['eq_name']}_sampled_plot.png"
config["save_path_witherror"] = f"./result/{config['eq_name']}_witherror_sampled_plot.png"

config["model_save_path"] = f"./checkpoints/best_model.pt" 

#config["model_save_path"] = f"./checkpoints/model_epoch_00500.pt" 
config["error_save_path"] = f"./result/{config['eq_name']}_RelL2.npy" 
