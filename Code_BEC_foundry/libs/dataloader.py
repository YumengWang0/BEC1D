"""
libs/dataloader.py
==================
Responsible only for:
  - DropletDataset  (wraps params + phi as tensors)
  - get_dataloaders (HDF5 → DataLoader)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from libs.data import load_hdf5


class DropletDataset(Dataset):
    """
    params : (M, P)  float32  — P parameters per sample (any number)
    phi    : (M, N)  float32  — spatial solution
    """
    def __init__(self, params, phi):
        self.params = torch.tensor(params, dtype=torch.float32)
        self.phi    = torch.tensor(phi,    dtype=torch.float32)

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx], self.phi[idx]   # (P,), (N,)

    @property
    def n_params(self):
        return self.params.shape[1]

    @property
    def n_spatial(self):
        return self.phi.shape[1]


def get_dataloaders(sol_path, batch_size=32):
    """
    Load HDF5 → train and test DataLoaders.
    
    Returns
        x_ref        : (N,)   spatial grid
        train_loader : DataLoader
        test_loader  : DataLoader
        param_names  : list of str  column labels for params
    """
    x_ref, train_params, train_phi, param_names = load_hdf5(sol_path, "train")
    _,     test_params,  test_phi,  _           = load_hdf5(sol_path, "test")

    print(f"Loaded {sol_path}")
    print(f"  train: params={train_params.shape}  phi={train_phi.shape}")
    print(f"  test:  params={test_params.shape}   phi={test_phi.shape}")
    print(f"  param_names: {param_names}")

    train_loader = DataLoader(
        DropletDataset(train_params, train_phi),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        DropletDataset(test_params, test_phi),
        batch_size=batch_size, shuffle=False,
    )
    return x_ref, train_loader, test_loader, param_names
