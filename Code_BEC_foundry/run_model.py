"""
run_model.py
============
Train or evaluate a surrogate model. Reads config from --config argument.

Usage:
    python run_model.py --config configs/Ex1.py --train   # train
    python run_model.py --config configs/Ex1.py           # evaluate only
"""
import argparse
import importlib.util

import torch
import torch.nn as nn


def load_config(config_path):
    spec   = importlib.util.spec_from_file_location("cfg", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file, e.g. configs/Ex1.py")
    
    parser.add_argument("--train", action="store_true",
                        help="Train the model. Omit to evaluate only.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config = cfg.config
    config["model_train"] = args.train   # CLI overrides config file value


    from libs.dataloader      import get_dataloaders
   
    from libs.train_module    import trainer, setup_logger
    from libs.evaluate_module import load_model, evaluate_model, plot_results_witherror


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device      : {device}")
    print(f"Experiment  : {config['eq_name']}")
    print(f"Mode        : {'train' if config['model_train'] else 'evaluate'}")

    # ── data ──────────────────────────────────────────────────
    x_ref, train_loader, val_loader, param_names = get_dataloaders(
        config["sol_name"], batch_size=config["batch_size"]
    )

    # quick shape check
    for params_b, phi_b in train_loader:
        print(f"params batch: {params_b.shape}   "   # (B, P)
              f"phi batch: {phi_b.shape}")             # (B, N)
        break

    logger = setup_logger(config["log_path"])
    
    models = load_model(config, device, learn_boundary=config["learn_boundary"])
      
    # ── models ────────────────────────────────────────────────
    if config["model_train"]:
        
        trainer(
            models       = models,
            train_loader = train_loader,
            val_loader  = val_loader,
            config       = config,
            logger       = logger,
            device       = device,
      
        )
        # reload best weights for evaluation
        
    else:
        
    
        # ── evaluate ──────────────────────────────────────────────
        params_all, phi_pred, phi_true, rel_errors = evaluate_model(
            x_ref, models, test_loader, 
            config["model_save_path"], 
            logger,  
            device, 
            learn_boundary=config["learn_boundary"],
            error_save_path=config["error_save_path"],
        )


        plot_results_witherror(
            x_ref, params_all, phi_pred, phi_true,
            param_names=param_names,
            save_path=config["save_path_witherror"],
        )


if __name__ == "__main__":
    main()
