    now  = datetime.datetime.now()
  
    # %%
    # ============================================================
    # Config for data 
    # ============================================================
    RANDOM_SEED = 20260425

    PHYSICS_DOMAIN = {
        "x_a": -20,
        "x_b":  20,
        "N":  4096,
        "x0":    0,
    }

    TRAIN_CONFIG = {
        "mu_left":  -0.2222,
        "mu_right": -0.05,
        "N_mu":      15,
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
        "model_train": True,
        "eq_name": "Ex1", 
        
        # Model parameters 
        "hidden_dims": [128, 128, 128], 
        "latent_feature_dim":16,
        "n_latent":64,
        "hidden_conv_dims": [256, 128, 64, 64, 32, 32],
        "activation":nn.ReLU,
        "output_activation": None, 
        "dropout": 0.1,
        
        
        # Training parameters 
        "epochs" : 5, 
        "epoch_save":1, 
        "batch_size": 32,
   
        # Optimizer parameters 
        "learning_rate": 1e-3,
        "step_size": 1,
        "weight_decay": 1e-5,
        "gamma":0.99, 
  
   
        # data
        "input_dim": 2,
        "output_sol_dim": 1, 


    } 
    
    config["log_path"] = f"./logs/train_{config['eq_name']}_{now:%Y-%m-%d_%H%M%S}.log"
    config["sol_name"] = f"../data/solution_{config['eq_name']}.h5"

  