"""
libs/criterion.py
=================
Training loss functions.
Select via config["criterion"]: "mse" | "relative_l2"
"""
import torch
import torch.nn as nn


def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)

# Mean relative error  
def relative_l2_loss(pred, target):
    """
    Mean relative L2 loss over the batch.
    loss = mean( ||pred - target||_2 / ||target||_2 )
    pred, target : (B, N)
    """
    num = (pred - target).norm(dim=1)        # (B,)
    den = target.norm(dim=1).clamp(min=1e-8) # (B,)
    return (num / den).mean() 

def get_criterion(config):
    """
    Returns criterion(pred, target) callable from config["criterion"].
    """
    # If not given the criterion, use mse 
    name = config.get("criterion", "mse")
    
    if name == "mse":
        return mse_loss
    elif name == "relative_l2":
        return relative_l2_loss
    else:
        raise ValueError(
            f"Unknown criterion: {name!r}. Choose 'mse' or 'relative_l2'."
        )
