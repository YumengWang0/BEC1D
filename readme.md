# Surrogate Model for the Droplet Solution of the Schrödinger Equation

A data-driven surrogate model that learns the mapping from physical parameters $(\mu, \delta)$ to the analytical droplet solution $\Psi(x)$ of the nonlinear Schrödinger equation.

---

## Analytical solution is 

$$\Psi_{D2}(x, \mu) = \frac{3\mu}{\delta \left(1 + \sqrt{1 + \frac{9\mu}{2\delta^2}} \cosh\left(\sqrt{-2\mu}(x - x_0)\right)\right)}$$

**Parameter constraints:**
- $\mu < 0$  
- $\delta > \sqrt{-9\mu/2}$ 

---

## Repository Structure

```
.
├── libs/                      # Core library modules
│   ├── config.py              # Configuration settings (hyperparameters, paths)
│   ├── data_gene.py           # Data generation and DataLoader utilities
│   ├── module.py              # Model architectures (Model1, Model2)
│   ├── train_module.py        # Training, evaluation, and trainer logic
│   └── evaluate_module.py     # Testing and result visualization
│
├── main.py                    # End-to-end pipeline (data → train → evaluate)
│
├── run_data.py                # Standalone data generation pipeline
├── run_model.py               # Standalone training + testing pipeline
│
├── Test.sub                   # Job script: full pipeline on Mill
├── Test_separate.sub          # Job script: separate pipeline on Mill
│
├── data/                      # Generated datasets (.h5 files)
├── checkpoints/               # Saved model weights
├── logs/                      # Training and evaluation logs
```

---

## Run the experiment 

### Full pipeline (data generation + training)
- Local
```bash
python generate_data.py    # run once, or when parameters change
python train.py
```
- Mill
```bash
sbatch Test.sub
```

### Separate pipeline 
- Local
```bash
python main.py
```

- Mill 
```bash
sbatch Test.sub
```

---

## Changing the Model
The parameters changes accordding to the rquirement in the `config.py` file: 
- `RANDOM_SEED`
- `PHYSICS_DOMAIN` 
- `TRAIN_CONFIG` 
- `TEST_CONFIG` 
- `config` 


### LatentModel  (`libs/module.py`)

Controls the **parameter encoder**. Change these in `config`:

| Parameter | What it does | Default |
|---|---|---|
| `hidden_dims` | MLP width and depth | `[128, 128, 128]` |
| `latent_feature_dim` | Feature size of each latent token | `16` |
| `n_latent` | Number of latent tokens | `64` |
| `activation` | Nonlinearity | `nn.ReLU` |

**The `n_latent`  is important for the design of the decoder. Be careful to test to fit.**


### Decoder  (`libs/module.py`)

Controls the **spatial upsampling** from latent → φ(x). The CNN must upsample from `n_latent` to `N=4096`.

Number of layers required: $\log_2(4096 / n\_latent)$

| `n_latent` | Upsample stages needed |
|---|---|
| 64 | 6 |
| 128 | 5 |
| 256 | 4 |

Change `hidden_conv_dims` to match — **one entry per stage**:

```python
# n_latent=64 → 4096 requires 6 stages
"hidden_conv_dims": [256, 128, 64, 64, 32, 32],   # 6 entries ✅

# n_latent=128 → 4096 requires 5 stages
"hidden_conv_dims": [256, 128, 64, 32, 32],        # 5 entries ✅
```

### Verify output shape before training

```python
model1 = LatentModel(...)
model2 = Decoder(...)

x = torch.randn(4, 2)
z = model1(x)
print("latent :", z.shape)        # (4, n_latent, latent_feature_dim)

out = model2(z)
print("output :", out.shape)      # must be (4, 4096)
```


