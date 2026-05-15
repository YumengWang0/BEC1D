"""
libs/criterion.py
=================
Training loss functions.
Select via config["criterion"]: "mse" | "relative_l2"
"""
import torch
import torch.nn as nn


 
class RelLpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(RelLpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.sum(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        target = torch.sum(abs(target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        rloss = torch.mean(error / target)
        return rloss


class LpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        loss = torch.mean(error)
        return loss



def get_criterion(config):
    """
    Returns criterion(pred, target) callable from config["criterion"].
    """
    # If not given the criterion, use mse 
    name = config.get("criterion", "relative_l2")


    
    if name == "mse":
        mse_loss = LpLoss(p = 2)
        return mse_loss

    elif name == "relative_l2":
        relative_l2_loss = RelLpLoss(p = 2)
        return relative_l2_loss
        
    else:
        raise ValueError(
            f"Unknown criterion: {name!r}. Choose 'mse' or 'relative_l2'."
        )
