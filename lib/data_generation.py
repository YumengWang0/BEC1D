# %%
import numpy as np 
import datetime 

today = datetime.date.today()
timestamp = datetime.datetime.now().timestamp()

# %% [markdown]
# \begin{equation}
# \Psi_{D2}(x, \mu) =
# \frac{3\mu}{
# \delta \left(
# 1 + \sqrt{1 + \frac{9\mu}{2\delta^2}}
# \cosh\left(\sqrt{-2\mu}(x - x_0)\right)
# \right)}
# \end{equation}

# %%
# -----------------------------
# analytical solution
# -----------------------------
def droplet_analytical_solution(mu, delta, domain):
    """
    Generate analytical droplet solution ψ(x)

    Args:
        mu: chemical potential (must be < 0)
        delta: interaction parameter (≠ 0)
        x0: center of droplet
        N: number of grid points

    Returns:
        x: grid points
        psi: solution ψ(x)
    """
    
    if mu >= 0:
        raise ValueError("mu must be negative")
    if delta == 0:
        raise ValueError("delta cannot be zero")

    inside = 1 + (9 * mu) / (2 * delta**2)
    if inside < 0:
        print(f"invalid for the current parameters mu={mu} and delta ={delta}")
        return None, None  # skip invalid

    x = np.linspace(domain["x_a"], domain["x_b"], domain["N"])

    sqrt_term = np.sqrt(inside)
    k = np.sqrt(-2 * mu)
    cosh_term = np.cosh(k * (x - domain["x0"]))

    denom = delta * (1 + sqrt_term * cosh_term)
    psi = (3 * mu) / denom

    return x, psi



# %%
if "__name__" == "main": 
    
    # -----------------------------
    # physics domain
    # -----------------------------
    physics_domain_params = {
        "x_a": -1,
        "x_b": 1,
        "N": 10,
        "x0": 1
    }

    # -----------------------------
    # parameter grid（注意 mu < 0）
    # -----------------------------
    sets_params = {
        "mu_left": -1.0,
        "mu_right": -0.2,
        "N_mu": 3,

        "delta_left": 0.2,
        "delta_right": 1.0,
        "N_delta": 3
    }

    # -----------------------------
    # build dataset
    # -----------------------------
    mu_list = np.linspace(
        sets_params["mu_left"],
        sets_params["mu_right"],
        sets_params["N_mu"]
    )

    delta_list = np.linspace(
        sets_params["delta_left"],
        sets_params["delta_right"],
        sets_params["N_delta"]
    )

    phi_all = []
    mu_all = []
    delta_all = []

    x_ref = None

    for mu in mu_list:
        print(mu)
        for delta in delta_list:
            
            x, phi = droplet_analytical_solution(
                mu, delta, physics_domain_params
            )
            

            if x is None:
                continue  # skip invalid combo

            if x_ref is None:
                x_ref = x  # store once

            phi_all.append(phi)
            mu_all.append(mu)
            delta_all.append(delta)

            print(len(mu_all),"000")
    # convert to numpy
    phi_all = np.array(phi_all)        # shape: (M, N)
    mu_all = np.array(mu_all)          # shape: (M,)
    delta_all = np.array(delta_all)    # shape: (M,)

    solution = {
        "mu": mu_all,
        "delta": delta_all,
        "x": x_ref,
        "phi": phi_all
    }

    sol_name = f"./data/solution_{today}_{timestamp}.npz"
    np.savez(sol_name, **solution)
    print("This data is generated based on the analytical solution.") 
