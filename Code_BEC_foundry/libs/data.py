"""
libs/data_prep.py
=================
Responsible for producing the standard dataset dict:

    {
        "x":      (N,)    spatial grid
        "phi":    (M, N)  snapshots
        "params": (M, P)  parameter matrix  (P = number of parameters)
        "param_names": [str, ...]  column labels for params
    }

Supports:
  - Analytical droplet solution
  - MATLAB: one .mat file per snapshot (folder)
  - MATLAB: all snapshots in one .mat file
"""

import os
import numpy as np
import h5py
import scipy.io


# ============================================================
# Analytical solution
# ============================================================
def droplet_analytical_solution(mu, delta, domain):
    if mu >= 0:
        raise ValueError("mu must be negative")
    if delta == 0:
        raise ValueError("delta cannot be zero")
    inside = 1 + (9 * mu) / (2 * delta ** 2)
    if inside <= 0:
        return None, None
    x         = np.linspace(domain["x_a"], domain["x_b"], domain["N"])
    k         = np.sqrt(-2 * mu)
    cosh_term = np.cosh(k * (x - domain["x0"]))
    denom     = delta * (1 + np.sqrt(inside) * cosh_term)
    return x, -(3 * mu) / denom


def delta_min(mu):
    return np.sqrt(-4.5 * mu) + 0.00005


def build_train_pairs(cfg):
    mu_list  = np.linspace(cfg["mu_left"], cfg["mu_right"], cfg["N_mu"])
    n_deltas = 2 * np.arange(1, cfg["N_mu"] + 1)
    pairs = []
    for mu, n_d in zip(mu_list, n_deltas):
        for delta in np.linspace(delta_min(mu), cfg["delta_max"], n_d):
            pairs.append([mu, delta])
    return pairs


def build_test_pairs(cfg):
    rng = np.random.default_rng(cfg["seed"])
    pairs = []
    for mu in rng.uniform(cfg["mu_left"], cfg["mu_right"], cfg["N_test"]):
        pairs.append([mu, rng.uniform(delta_min(mu), cfg["delta_max"])])
    return pairs



# ============================================================
# Works for ANY source
# ============================================================
def assemble_dataset(x_ref, phi_list, params_matrix, param_names):
    """
    x_ref        : (N,)
    phi_list     : list of (N,)  or np.array (M, N)
    params_matrix: (M, P)  each row is one sample's parameters
    param_names  : list of P strings, column labels

    Returns standard dataset dict.
    """
    phi = np.array(phi_list) if not isinstance(phi_list, np.ndarray) else phi_list
    return {
        "x":           x_ref,
        "phi":         phi,                    # (M, N)
        "params":      np.array(params_matrix, dtype=np.float64),  # (M, P)
        "param_names": list(param_names),
    }


# ============================================================
# Source 1 — Analytical. 
# This function should change according to your experiment 
# 
# ============================================================
def build_dataset_analytical(pairs, domain, param_names=None):
    """
    pairs       : list of [mu, delta]
    param_names : defaults to ["mu", "delta"]
    """
    if param_names is None:
        param_names = ["mu", "delta"]

    phi_list, params_rows, x_ref = [], [], None
    for mu, delta in pairs:
        x, phi = droplet_analytical_solution(mu, delta, domain)
        if x is None:
            continue
        if x_ref is None:
            x_ref = x
        phi_list.append(phi)
        params_rows.append([mu, delta])

    return assemble_dataset(x_ref, phi_list, params_rows, param_names)


# ============================================================
# Source 2 — MATLAB folder (one .mat file per snapshot)
# ============================================================
def build_dataset_mat_folder(folder_path, x_key, phi_key, param_keys):
    """
    folder_path : path to folder of .mat files
    x_key       : variable name for spatial grid
    phi_key     : variable name for solution
    param_keys  : dict  {param_name: mat_variable_name}
                  e.g. {"mu": "mu", "delta": "delta", "gamma": "gam"}
                  Can have any number of parameters.

    Each .mat file is one snapshot.
    """
    mat_files   = sorted(f for f in os.listdir(folder_path) if f.endswith(".mat"))
    param_names = list(param_keys.keys())

    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder_path}")

    phi_list, params_rows, x_ref = [], [], None

    for fname in mat_files:
        data = scipy.io.loadmat(os.path.join(folder_path, fname))

        if x_ref is None:
            x_ref = data[x_key].flatten()

        phi = data[phi_key].flatten()
        row = [float(data[mat_var].flat[0]) for mat_var in param_keys.values()]

        phi_list.append(phi)
        params_rows.append(row)
        print(f"  {fname}  " + "  ".join(
            f"{n}={v:.4f}" for n, v in zip(param_names, row)
        ))

    return assemble_dataset(x_ref, phi_list, params_rows, param_names)


# ============================================================
# Source 3 — MATLAB single file (all snapshots)
# ============================================================
def build_dataset_mat_file(mat_path, x_key, phi_key, param_keys):
    """
    mat_path    : path to a single .mat file
    param_keys  : dict {param_name: mat_variable_name}
                  The mat variables should each be (M,) or (M,1).
    Expected shapes inside .mat:
        x     : (N,)   or (1, N)
        phi   : (M, N)
        each param variable : (M,) or (M, 1)
    """
    data        = scipy.io.loadmat(mat_path)
    param_names = list(param_keys.keys())

    x_ref = data[x_key].flatten()          # (N,)
    phi   = data[phi_key]                   # (M, N)

    # stack all parameter columns → (M, P)
    param_cols  = [data[mat_var].flatten() for mat_var in param_keys.values()]
    params_matrix = np.column_stack(param_cols)  # (M, P)

    return assemble_dataset(x_ref, phi, params_matrix, param_names)


# ============================================================
# HDF5 save / load  (generic, works for any P)
# ============================================================
def save_hdf5(path, train, test, now):
    """
    Saves the standard dataset dict to HDF5.
    Stores params as (M, P) matrix + param_names as attribute.
    """
    with h5py.File(path, "w") as f:
        f.attrs["description"]  = "Surrogate model snapshots"
        f.attrs["created"]      = str(now)
        f.attrs["param_names"]  = train["param_names"]   # list of strings

        f.create_dataset("x", data=train["x"])

        for split_name, data in [("train", train), ("test", test)]:
            grp = f.create_group(split_name)
            grp.attrs["n_samples"] = data["phi"].shape[0]
            grp.create_dataset("phi",    data=data["phi"])     # (M, N)
            grp.create_dataset("params", data=data["params"])  # (M, P)

    print(f"Saved → {path}")
    print(f"  train: phi={train['phi'].shape}  params={train['params'].shape}")
    print(f"  test:  phi={test['phi'].shape}   params={test['params'].shape}")
    print(f"  param_names: {train['param_names']}")


def load_hdf5(path, split="train"):
    """
    Returns
        x           : (N,)
        params      : (M, P)
        phi         : (M, N)
        param_names : list of P strings
    """
    with h5py.File(path, "r") as f:
        x           = f["x"][:]
        param_names = list(f.attrs["param_names"])
        grp         = f[split]
        phi         = grp["phi"][:]     # (M, N)
        params      = grp["params"][:] # (M, P)
    return x, params, phi, param_names
