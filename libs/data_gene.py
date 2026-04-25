
import numpy as np
import datetime
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


# %%
# ============================================================
# Analytical solution
# ============================================================
def droplet_analytical_solution(mu, delta, domain):
    """
    ψ(x) analytical droplet solution.

    Args:
        mu    : chemical potential  (must be < 0)
        delta : interaction param   (must be ≠ 0)
        domain: dict with x_a, x_b, N, x0

    Returns:
        x   : (N,) grid
        psi : (N,) solution, or (None, None) if params are invalid
    """
    if mu >= 0:
        raise ValueError("mu must be negative")
    if delta == 0:
        raise ValueError("delta cannot be zero")

    inside = 1 + (9 * mu) / (2 * delta**2)
    if inside <= 0:
        print(f"[SKIP] invalid params  mu={mu:.4f}, delta={delta:.4f}")
        return None, None

    x         = np.linspace(domain["x_a"], domain["x_b"], domain["N"])
    sqrt_term = np.sqrt(inside)
    k         = np.sqrt(-2 * mu)
    cosh_term = np.cosh(k * (x - domain["x0"]))
    denom     = delta * (1 + sqrt_term * cosh_term)
    psi       = (3 * mu) / denom

    return x, psi

# %%
# ============================================================
# Parameter pair builders
# ============================================================
def delta_min(mu):
    """Strict lower bound: |δ| > 3√(−μ/2)"""
    return np.sqrt(-4.5 * mu) + 0.05          # small margin above boundary

def build_train_pairs(cfg):
    """
    Increasing number of delta samples per mu:
    mu_i  →  N_delta = 2*(i+1)
    """
    mu_list   = np.linspace(cfg["mu_left"], cfg["mu_right"], cfg["N_mu"])
    n_deltas  = 2 * np.arange(1, cfg["N_mu"] + 1)
    pairs = []
    for mu, n_d in zip(mu_list, n_deltas):
        for delta in np.linspace(delta_min(mu), 1.1, n_d):
            pairs.append([mu, delta])
    return pairs


def build_test_pairs(cfg):
    """
    Random sample in test range.
    mu    ~ Uniform(mu_left,  mu_right)
    delta ~ Uniform(delta_min(mu), delta_max)
    """
    rng   = np.random.default_rng(cfg["seed"])
    pairs = []

    mu_samples = rng.uniform(cfg["mu_left"], cfg["mu_right"], cfg["N_test"])

    for mu in mu_samples:
        d_min = delta_min(mu)
        delta = rng.uniform(d_min, cfg["delta_max"])
        pairs.append([mu, delta])

    return pairs

# %%
# ============================================================
# Dataset builder
# ============================================================
def build_dataset(pairs, domain):
    phi_all   = []
    mu_all    = []
    delta_all = []
    x_ref     = None

    for mu, delta in pairs:
        x, phi = droplet_analytical_solution(mu, delta, domain)
        if x is None:
            continue
        if x_ref is None:
            x_ref = x
        phi_all.append(phi)
        mu_all.append(mu)
        delta_all.append(delta)

    return {
        "x":     x_ref,
        "phi":   np.array(phi_all),    # (M, N)
        "mu":    np.array(mu_all),     # (M,)
        "delta": np.array(delta_all),  # (M,)
    }

# %%
# ============================================================
# Save to HDF5
# ============================================================
def save_hdf5(path, train, test, now):
    with h5py.File(path, "w") as f:

        f.attrs["description"] = "Analytical droplet solutions"
        f.attrs["created"]     = str(now)

        # shared x grid
        f.create_dataset("x", data=train["x"])

        for split_name, data in [("train", train), ("test", test)]:
            grp_split = f.create_group(split_name)
            grp_split.attrs["n_samples"] = data["phi"].shape[0]

            for i, (mu, delta, phi) in enumerate(
                zip(data["mu"], data["delta"], data["phi"])
            ):
                run = grp_split.create_group(f"run_{i:04d}")
                run.attrs["mu"]    = mu
                run.attrs["delta"] = delta
                run.create_dataset("phi", data=phi)


# %%
# ============================================================
# Load train and test data
# ============================================================

def load_split(path, split="train"):
    with h5py.File(path, "r") as f:
        x      = f["x"][:]
        grp    = f[split]
        names  = sorted(grp.keys())
        B, N   = len(names), len(x)

        params  = np.zeros((B, 2))
        phi_all = np.zeros((B, N))

        for i, name in enumerate(names):
            run           = grp[name]
            params[i, 0]  = run.attrs["mu"]
            params[i, 1]  = run.attrs["delta"]
            phi_all[i]    = run["phi"][:]

    return x, params, phi_all  # params[:,0]=mu  params[:,1]=delta 



# ============================================================
# Dataset
# ============================================================
class DropletDataset(Dataset):
    def __init__(self, params, phi):
        """
        params : (B, 2)  [mu, delta]
        phi    : (B, N)  analytical solution
        """
        self.params = torch.tensor(params, dtype=torch.float32)
        self.phi    = torch.tensor(phi,    dtype=torch.float32)

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx], self.phi[idx]   # (2,), (N,)


# ============================================================
# Load from HDF5  →  DataLoader
# ============================================================
def get_dataloaders(path, batch_size=32, num_workers=0):

    x_ref, train_params, train_phi = load_split(path, "train")
    _,     test_params,  test_phi  = load_split(path, "test")

    train_dataset = DropletDataset(train_params, train_phi)
    test_dataset  = DropletDataset(test_params,  test_phi)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
 
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
       
    )

    return x_ref, train_loader, test_loader

