"""
run_data.py
===========
Standalone data generation. Reads config from --config argument.

Usage:
    python run_data.py --config configs/Ex1.py
    python run_data.py --config configs/Ex2.py
"""
import os
import argparse
import datetime
import importlib.util


def load_config(config_path):
    """Dynamically load a config .py file and return its module."""
    spec   = importlib.util.spec_from_file_location("cfg", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file, e.g. configs/Ex1.py")
    

    
    args = parser.parse_args()
    cfg = load_config(args.config)

    # avoid circular import — import data functions here
    from libs.data import (
        build_train_pairs, build_test_pairs,
        build_dataset_analytical,
        build_dataset_mat_folder,
        build_dataset_mat_file,
        save_hdf5,
        droplet_analytical_solution,
    )

    sol_path = cfg.config["sol_name"]

    if os.path.exists(sol_path):
        print(f"[DATA] Found {sol_path} → skip (delete to regenerate)")
        return

    now = datetime.datetime.today()
    print(f"[DATA] Generating data for {cfg.EQ_NAME} ...")
    print(f"       Source: {cfg.DATA_SOURCE}")


    # ── choose data source ────────────────────────────────────
    if cfg.DATA_SOURCE == "analytical":
        train_pairs = build_train_pairs(cfg.TRAIN_CONFIG)
        test_pairs  = build_test_pairs(cfg.TRAIN_CONFIG)
        print(f"  Train pairs: {len(train_pairs)}")
        print(f"  Test  pairs: {len(test_pairs)}")
        
        #build_dataset_analytical_notuse
        train_data = build_dataset_analytical(
            train_pairs, cfg.PHYSICS_DOMAIN,  cfg.PARAM_NAMES
        )
        


        test_data = build_dataset_analytical(
            test_pairs, cfg.PHYSICS_DOMAIN,  cfg.PARAM_NAMES
        )


    elif cfg.DATA_SOURCE == "mat_folder":
        mc = cfg.MAT_CONFIG
        train_data = build_dataset_mat_folder(
            mc["train_path"], mc["x_key"], mc["phi_key"], mc["param_keys"]
        )
        test_data = build_dataset_mat_folder(
            mc["test_path"],  mc["x_key"], mc["phi_key"], mc["param_keys"]
        )

    elif cfg.DATA_SOURCE == "mat_file":
        mc = cfg.MAT_CONFIG
        train_data = build_dataset_mat_file(
            mc["train_path"], mc["x_key"], mc["phi_key"], mc["param_keys"]
        )
        test_data = build_dataset_mat_file(
            mc["test_path"],  mc["x_key"], mc["phi_key"], mc["param_keys"]
        )

    else:
        raise ValueError(f"Unknown DATA_SOURCE: {cfg.DATA_SOURCE!r}. "
                         "Choose 'analytical', 'mat_folder', or 'mat_file'.")

    save_hdf5(sol_path, train_data, test_data, now)


if __name__ == "__main__":
    main()
