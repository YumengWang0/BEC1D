import os
import logging
import torch
import torch.nn as nn
from libs.criterion import get_criterion


# ============================================================
# Logger
# ============================================================
def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.FileHandler(log_path), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def save_checkpoint(path, state):
    tmp = path + ".tmp"        # e.g. "checkpoints/Ex1/best_model.pt.tmp"
    torch.save(state, tmp)     # step 1: save to the temp file first
    os.replace(tmp, path)      # step 2: rename .tmp → .pt  (atomic)


# ============================================================
# Train / Evaluate one epoch
# ============================================================
def train(models, dataloader, optimizer, criterion, device):
    model1, model2 = models[0].to(device), models[1].to(device)
    model1.train()
    model2.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model2(model1(x))
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(models.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(models, dataloader, criterion, device):
    model1, model2 = models[0].to(device), models[1].to(device)
    model1.eval()
    model2.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model2(model1(x))
            total_loss += criterion(pred, y).item()
    return total_loss / len(dataloader)


# ============================================================
# Trainer
# ============================================================
def trainer(models, train_loader, test_loader, config, logger, device, ):
    
    logger.info("=" * 60)
    logger.info(f"Experiment    : {config['eq_name']}")
    logger.info(f"criterion     : {config.get('criterion', 'mse')}")
    logger.info(f"epochs        : {config['epochs']}")
    logger.info(f"learning_rate : {config['learning_rate']}")
    logger.info(f"weight_decay  : {config['weight_decay']}")
    logger.info(f"step_size     : {config['step_size']}")
    logger.info(f"gamma         : {config['gamma']}")
    logger.info(f"epoch_save    : {config['epoch_save']}")
    logger.info("=" * 60)

    criterion = get_criterion(config)

    optimizer = torch.optim.Adam(
        models.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )

    best_val = float("inf")
    ckpt_dir = os.path.dirname(config["model_save_path"])
  

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train(models, train_loader, optimizer, criterion, device)
        scheduler.step()
        logger.info(f"Epoch {epoch:05d} | train: {train_loss:.6f}")


        if epoch % config["epoch_save"] == 0:
            test_loss = evaluate(models, test_loader, criterion, device)
            logger.info(
                f"Epoch {epoch:05d} | test : {test_loss:.6f}  "
                f"[lr={scheduler.get_last_lr()[0]:.2e}]"
            )

            # periodic checkpoint
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pt"),
                {"model1": models[0].state_dict(),
                 "model2": models[1].state_dict()},
            )
            

            # best checkpoint, best model 
            if test_loss < best_val:
                best_val = test_loss
                save_checkpoint(
                    config["model_save_path"],
                    {"model1": models[0].state_dict(),
                     "model2": models[1].state_dict()},
                )
                logger.info(f"Epoch {epoch:05d} | best saved ({best_val:.6f})")

    logger.info("=" * 60)
    logger.info(f"Training finished | best: {best_val:.6f}")
    logger.info("=" * 60)
    return models
