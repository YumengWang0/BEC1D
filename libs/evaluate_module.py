import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from libs.module import LatentModel, Decoder
from libs.data_gene import get_dataloaders


# ============================================================
# Load model
# ============================================================
def load_model(checkpoint_path, config, device):

    model1 = LatentModel(
        input_dim          = config["input_dim"],
        hidden_dims        = config["hidden_dims"],
        latent_feature_dim = config["latent_feature_dim"],
        n_latent           = config["n_latent"],
        activation         = config["activation"],
        output_activation  = config["output_activation"],
    ).to(device)

    model2 = Decoder(
        latent_dim       = config["latent_feature_dim"],
        hidden_conv_dims = config["hidden_conv_dims"],
        output_sol_dim   = config["output_sol_dim"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model1.load_state_dict(checkpoint["model1"])
    model2.load_state_dict(checkpoint["model2"])

    models = nn.ModuleList([model1, model2])
    models.eval()

    return models


# ============================================================
# Evaluate
# ============================================================
def evaluate_model(models, test_loader,  device, model_path, error_save_path = None):
    """
    Returns
        params_all : (M, 2)   [mu, delta]
        phi_pred   : (M, N)   model prediction
        phi_true   : (M, N)   ground truth
        rel_errors : (M,)     relative L2 error per sample
    """

    checkpoint = torch.load(model_path, map_location=device)

    models[0].load_state_dict(checkpoint["model1"])
    models[1].load_state_dict(checkpoint["model2"])
    model1, model2 = models[0], models[1]



    model1.to(device)
    model2.to(device)
    
    model1.eval()
    model2.eval()
    
    criterion = nn.MSELoss()

    params_all, phi_pred_all, phi_true_all = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for params, phi in test_loader:
            params, phi = params.to(device), phi.to(device)

            pred = model2(model1(params)).squeeze(-1)   # (B, N)
            loss = criterion(pred, phi)
            total_loss += loss.item()

            params_all.append(params.cpu().numpy())
            phi_pred_all.append(pred.cpu().numpy())
            phi_true_all.append(phi.cpu().numpy())

    params_all  = np.concatenate(params_all,  axis=0)   # (M, 2)
    phi_pred    = np.concatenate(phi_pred_all, axis=0)   # (M, N)
    phi_true    = np.concatenate(phi_true_all, axis=0)   # (M, N)

    # relative L2 error per sample
    rel_errors = np.linalg.norm(phi_pred - phi_true, axis=1) / \
                 np.linalg.norm(phi_true,             axis=1)

    if error_save_path is not None: 
        # save
        np.save(error_save_path, rel_errors)
        print(f"The relative L2 error for each parameter is stored")
        

    avg_loss = total_loss / len(test_loader)
    print(f"Test MSE loss         : {avg_loss:.6f}")
    print(f"Mean relative L2 error: {rel_errors.mean():.4f} ± {rel_errors.std():.4f}")

    return params_all, phi_pred, phi_true, rel_errors

# ============================================================
# Plot  (paper quality, 4 samples)
# ============================================================
def plot_results(x, params, phi_pred, phi_true,
                 indices=None, save_path=None):
    """
    Plot 4 samples: prediction vs ground truth.

    Args:
        x          : (N,)   spatial grid
        params     : (M, 2) [mu, delta]
        phi_pred   : (M, N)
        phi_true   : (M, N)
  
        indices    : list of 4 indices to plot (default: spread across dataset)
        save_path  : if given, saves figure
    """
    if indices is None:
        M = phi_pred.shape[0]
        indices = [0, M // 3, 2 * M // 3, M - 1]
        print (indices)

    # ── style ────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":         11,
        "axes.labelsize":    12,
        "axes.titlesize":    11,
        "legend.fontsize":    9,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.linewidth":     0.8,
        "lines.linewidth":    1.8,
    })

    colors = ["#2166ac", "#d6604d", "#4dac26", "#7b3294"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()

    
    for ax, idx, color in zip(axes, indices, colors):
        print(  zip(axes, indices, colors), )
        print(  params[idx, 0], idx, " ---- ")
        mu    = params[idx, 0]
        delta = params[idx, 1]
     

        # plot |ψ|²
        ax.plot(x, phi_true[idx]**2, color=color,
                linestyle="-",  label="Analytical",  linewidth=1.8)
        ax.plot(x, phi_pred[idx]**2, color="k",
                linestyle="--", label="Surrogate",   linewidth=1.4, alpha=0.85)

        ax.set_title(
            rf"$\mu={mu:.4f},\ \delta={delta:.3f}$"
            f"\n",
            pad=6
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|\Psi|^2$")

    fig.suptitle(r"Surrogate vs Analytical Droplet Solution $|\phi_{D2}|^2$",
                 fontsize=13, y=1.01)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved → {save_path}")

    plt.show()
