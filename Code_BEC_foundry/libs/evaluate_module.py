import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from libs.module import LatentModel, Decoder


def integrate_phi_squared(phi, x_ref):
    """
    Compute ∫ φ²(x) dx for each sample using the trapezoidal rule.

    Parameters
    ----------
    phi   : (M, N)  snapshots
    x_ref : (N,)    spatial grid (can be non-uniform)

    Returns
    -------
    integral : (M,)  ∫ φ² dx per sample
    """
    phi_sq   = phi ** 2                              # (M, N)
    integral = np.trapz(phi_sq, x=x_ref, axis=1)    # (M,)
    return integral



# ============================================================
# Load model from checkpoint
# If 2D model is needy, can change the load_model only  
# ============================================================
def load_model(config, device):
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
        output_size = config["output_size"]
    ).to(device)

    models = nn.ModuleList([model1, model2])
    return models
 

# ============================================================
# Evaluate
# NOTE: rel_errors is ALWAYS standard relative L2.
#       It is independent of whatever criterion was used during training.
# ============================================================
def evaluate_model(x_ref, models, test_loader, checkpoint_path, logger,  device, error_save_path=None):
    """
    
    Returns
        params_all : (M, P)
        phi_pred   : (M, N)
        phi_true   : (M, N)
        rel_errors : (M,)   standard relative L2 = ||pred-true|| / ||true||
        
    The checkpoint path can choose the best model or the last model. 
    """
    #model1, model2 = load_model(checkpoint_path, config, device)
   
    ckpt = torch.load(checkpoint_path, map_location=device)
    model1.load_state_dict(ckpt["model1"])
    model2.load_state_dict(ckpt["model2"])
    model1, model2 = models[0].to(device), models[1].to(device)
    
    model1.eval()
    model2.eval()

    params_list, pred_list, true_list = [], [], []

    with torch.no_grad():
        for params, phi in test_loader:
            params, phi = params.to(device), phi.to(device)
            pred        = model2(model1(params))
            params_list.append(params.cpu().numpy())
            pred_list.append(pred.cpu().numpy())
            true_list.append(phi.cpu().numpy())

    params_all = np.concatenate(params_list, axis=0)
    phi_pred   = np.concatenate(pred_list,   axis=0)
    phi_true   = np.concatenate(true_list,   axis=0)

    # always standard relative L2 — independent of training criterion
    num        = np.linalg.norm(phi_pred - phi_true, axis=1)
    den        = np.linalg.norm(phi_true,             axis=1).clip(1e-8)
    rel_errors = num / den

    logger.info(f"Mean rel. L2 error : {rel_errors.mean():.4f} ± {rel_errors.std():.4f}")
    logger.info(f"Max  rel. L2 error : {rel_errors.max():.4f}")

    phi_norm_pred = integrate_phi_squared(phi_pred, x_ref)
    phi_norm_true = integrate_phi_squared(phi_true, x_ref)
    phi_error = np.abs( phi_norm_pred -  phi_norm_true )
    
    logger.info(f"Phi predict norm: {phi_norm_pred.mean():.4f}± {phi_norm_pred.std():.4f} ")
    logger.info(f"Phi true norm: {phi_norm_pred.mean():.4f}± {phi_norm_pred.std():.4f} ")
    logger.info(f"Phi error norm: {phi_error.mean():.4f}± {phi_error.std():.4f} ")
    logger.info(f"Phi error norm maximum: {phi_error.max():.4f}")
    
    if error_save_path is not None:
        np.savez(
            error_save_path,
            params     = params_all,
            rel_errors = rel_errors,
            phi_pred   = phi_pred,
            phi_true   = phi_true,
        )
        logger.info(f"Results saved → {error_save_path}")

    return params_all, phi_pred, phi_true, rel_errors


# ============================================================
# Plotting
# ============================================================
def _plot_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12,   "axes.titlesize": 11,
        "legend.fontsize": 9,   "xtick.labelsize": 10,
        "ytick.labelsize": 10,  "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
    })

COLORS = ["#2166ac", "#d6604d", "#4dac26", "#7b3294"]

def _default_indices(M):
    return [0, M // 3, 2 * M // 3, M - 1]

def _param_title(params_row, param_names):
    parts = [rf"${n}={v:.4f}$" for n, v in zip(param_names, params_row)]
    return ",\\ ".join(parts)


def plot_results_witherror(x, params, phi_pred, phi_true,
                           param_names=None, indices=None, save_path=None):
    _plot_style()
    if param_names is None:
        param_names = [f"p{i}" for i in range(params.shape[1])]

    indices    = indices or _default_indices(phi_pred.shape[0])
    global_max = max(np.abs(phi_pred[i]**2 - phi_true[i]**2).max() for i in indices)

    fig = plt.figure(figsize=(11, 11))
    gs  = fig.add_gridspec(5, 2, height_ratios=[3, 1, 0.5, 3, 1],
                           hspace=0.08, wspace=0.35)
    layout     = [(0, 0), (1, 0), (0, 3), (1, 3)]
    ax_err_ref = None

    for (idx, color), (col, main_row) in zip(zip(indices, COLORS), layout):
        ax_main = fig.add_subplot(gs[main_row,     col])
        ax_err  = fig.add_subplot(gs[main_row + 1, col],
                                  sharex=ax_main, sharey=ax_err_ref)
        if ax_err_ref is None:
            ax_err_ref = ax_err

        pred_sq = phi_pred[idx] ** 2
        true_sq = phi_true[idx] ** 2
        abs_err = np.abs(pred_sq - true_sq)

        ax_main.plot(x, true_sq, color=color, ls="-",  label="Reference")
        ax_main.plot(x, pred_sq, color="k",   ls="--", label="Surrogate", alpha=0.85)
        ax_main.set_title(_param_title(params[idx], param_names), pad=5)
        ax_main.set(xlim=(x[0], x[-1]), ylim=(0, None), ylabel=r"$|\Psi|^2$")
        ax_main.legend(frameon=False)
        ax_main.tick_params(labelbottom=False)

        ax_err.fill_between(x, abs_err, alpha=0.25, color=color)
        ax_err.plot(x, abs_err, color=color, lw=1.2)
        ax_err.set(xlim=(x[0], x[-1]), ylim=(0, global_max * 1.1),
                   xlabel=r"$x$", ylabel=r"$|\mathrm{err}|$")
        ax_err.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.1e}")
        )

    fig.suptitle(r"Surrogate vs Reference $|\Psi|^2$", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
