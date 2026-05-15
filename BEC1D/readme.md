
# Surrogate Model for the Droplet Solution of the Schrödinger Equation

A data-driven surrogate model that learns the mapping from physical parameters $(\mu, \delta)$ to the analytical or numerical droplet solution $\Psi(x)$ of the nonlinear Schrödinger equation.

---

## An example

**Analystical solutions:**

$$\Psi_{D2}(x, \mu) = - \frac{3\mu}{\delta \left[ 1 + \sqrt{1 + \frac{9\mu}{2\delta^2}} \cosh\left( \sqrt{-2\mu}(x - x_0) \right) \right]}$$


**Parameter constraints:**
- $\mu < 0$  
- $\delta > \sqrt{-9\mu/2}$ 

---

## Repository Structure
**The logic in the original github code keeps**. But reorganize for training different experiments 
```
.
├── libs/                      # Core library modules
│   ├── data.py           # Data generation
    ├── data_gene.py           # Prepare DataLoader 
│   ├── module.py              # Model architectures (Model1, Model2)
│   ├── train_module.py        # Training, evaluation, and trainer logic
│   └── evaluate_module.py     # Testing and result visualization
│
├── configs
    ├── config1.py              # Configuration settings (hyperparameters, paths)
    ├── config2.py              # Configuration settings (hyperparameters, paths)
    |── config3.py              # Configuration settings (hyperparameters, paths)
    ... 

├── run_data.py                # Data generation pipeline
├── run_model.py               # Training + testing pipeline
│
├── Test_separate.sub          # Job script: separate pipeline on Mill
│
├── data/                      # Generated datasets (.h5 files)
    ├── ex1 
    ├── ex2 
    ... 
├── checkpoints/               # Saved model weights
    ├── ex1 
    ├── ex2 
    ... 

├── logs/                      # Training and evaluation logs
    ├── ex1 
    ├── ex2 
    ... 

├── Result/                    # Error file and plots result
    ├── ex1 
    ├── ex2 
    ...  
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data
```bash
python run_data.py --eq_name Ex1
```

### 3. Train the model
```bash
# local
python run_model.py --configs/config1.py --train 
```

### 4. Evaluate (no `--train` flag)
```bash
python run_model.py --configs/config1.py 
```

### 5. Mill 
```bash
sbatch Test_separate.sub
``` 
---


## Changing the Model

### Parameters selection: 

    The training parameters are the most **representative parameters**, which can covers the parameters space. 
    **Uniformly sampling each parameters including the boundary parameters.**

### Data preparation. 

#### Option A — Numerical solution from MATLAB
   The solution $\Psi(x;\mu,\delta)$ is computed directly from the formula like above. 

    No external files needed — just changing the analytical solution function and configures `config.py`.

#### Option B — Numerical solution from MATLAB

Place `.mat` files in `./data/matlab/`. Two layouts are supported:

**One file per snapshot** (folder of `.mat` files):
```
./data/matlab/train/
    run_001.mat   # contains: x, phi, mu, delta
    run_002.mat
    ...
```
**All snapshots in one file:**
```
./data/matlab/train_all.mat   # contains: x (N,), phi (M,N), mu (M,), delta (M,)
```

Variable names inside `.mat` files are configurable:
```python
build_dataset_from_mat_folder(
    "./data/matlab/train/",
    x_key     = "xgrid",      # default: "x"
    phi_key   = "psi",        # default: "phi"
    mu_key    = "chem_pot",   # default: "mu"
    delta_key = "interaction"  # default: "delta"
)
```

## Configuration

All settings live in `libs/config.py`.  
**Recommended: create one `config.py` per experiment.**

### Key constants

| Constant | Description |
|---|---|
| `RANDOM_SEED` | Ensures reproducibility |
| `PHYSICS_DOMAIN` | Spatial grid: `x_a`, `x_b`, `N`, `x0` |
| `TRAIN_CONFIG` | Parameter sampling range for training and testing data|
| `config` | Model parameters and the training parameters | 
**model_train, criterion, eq_name**

If more hyperparameters is needy, add in the config file.
**Recommend to build each experiment one `config.py`**. In the `run_data.py` can write if-else to add the function.  I will add this part in the github    


### Command-line arguments 
| Argument | Type | Description |
|---|---|---|
| `--config` | `.py` | Configuration file|
| `--train` | flag | Include to train; omit to evaluate only|

### Model Architecture parameters
**Neural network parameters** in the `config.py` is important.  
**No need to change the neural network**. 
| Parameter | Meaning | Default |
|---|---|---|
|`config["hidden_conv_dims"]`| Upsampling width and depth | [256, 128, 64, 64, 32, 32]|, 
|`config["n_latent"]` | Number of latent tokens | 64 |


Change `hidden_conv_dims` to match — **one entry per stage**:

```python
# n_latent=64 → 4097 requires 6 stages
-- 128 -- 256 -- 512 -- 1024 -- 2048 -- 4096(4097)  
"hidden_conv_dims": [256, 128, 64, 64, 32, 32],    # 6 entries ✅

# n_latent=128 → 4097 requires 5 stages
# -- 256 -- 512 -- 1024 -- 2048 -- 4096(4097) 
"hidden_conv_dims": [256, 128, 64, 32, 32],        # 5 entries ✅
```

### Verify shapes before training

```python
from libs.module import LatentModel, Decoder
import torch

model1 = LatentModel(input_dim=2, hidden_dims=[128, 128, 128],
                     latent_feature_dim=16, n_latent=64)
model2 = Decoder(latent_dim=16, hidden_conv_dims=[256, 128, 64, 64, 32, 32],
                 output_sol_dim=1, output_size=4097)

params = torch.randn(4, 2)  # 2 is the number of parameters 
z      = model1(params)
out    = model2(z) # out shape should be the same as the data dimension 

print("latent :", z.shape)    # (4, 16, 64)
print("output :", out.shape)  # (4, 4097)  ← must match N
```


