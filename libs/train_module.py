import logging
import os
import numpy  as np 

import torch  
import torch.nn as nn    





# ============================================================
# Logger setup
# ============================================================
def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    
    logger.addHandler(ch)

    return logger


# ============================================================
# Train
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
        print(loss.item(), "loss-----------")
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Evaluate
# ============================================================
def evaluate(models, dataloader, criterion, device):
    model1, model2 = models[0].to(device), models[1].to(device)
    model1.eval()
    model2.eval()

    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model2(model1(x))
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Trainer
# ============================================================
def trainer(models, train_loader, test_loader, config, logs, device):


    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs",        exist_ok=True)

    logger = setup_logger(f"./{config['log_path']}.log")
    logger.info("=" * 60)
    logger.info("Training started")
    logger.info(f"epochs       : {config['epochs']}")
    logger.info(f"learning_rate: {config['learning_rate']}")
    logger.info(f"weight_decay : {config['weight_decay']}")
    logger.info(f"step_size    : {config['step_size']}")
    logger.info(f"gamma        : {config['gamma']}")
    logger.info(f"epoch_save   : {config['epoch_save']}")
    logger.info("=" * 60)

    optimizer = torch.optim.Adam(
        models.parameters(),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = config["step_size"],
        gamma     = config["gamma"],
    )
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(config["epochs"]):

        # ── train ────────────────────────────────────────────
        train_loss = train(models, train_loader, optimizer, criterion, device)
        scheduler.step()

        # ── logging (every epoch) ────────────────────────────
        logger.info(f"Epoch {epoch:05d} | train: {train_loss:.6f}")

        if logs is not None:
            logs.log({"epoch": epoch, "train_loss": train_loss})

        # ── test + save  (every epoch_save epochs) ───────────
        if epoch % config['epoch_save'] == 0:

            test_loss = evaluate(models, test_loader, criterion, device)

            logger.info(
                f"Epoch {epoch:05d} | test : {test_loss:.6f}  "
                f"[lr={scheduler.get_last_lr()[0]:.2e}]"
            )

            if logs is not None:
                logs.log({"epoch": epoch, "test_loss": test_loss})

            # periodic checkpoint
            torch.save(
                {"model1": models[0].state_dict(),
                 "model2": models[1].state_dict()},
                f"./checkpoints/model_epoch_{epoch:05d}.pt"
            )

            # best model
            if test_loss < best_val:
                best_val = test_loss
                torch.save(
                    {"model1": models[0].state_dict(),
                     "model2": models[1].state_dict()},
                    "./checkpoints/best_model.pt"
                )
                logger.info(
                    f"Epoch {epoch:05d} | best model saved  "
                    f"(test_loss={best_val:.6f})"
                )

    logger.info("=" * 60)
    logger.info(f"Training finished | best test loss: {best_val:.6f}")
    logger.info("=" * 60)

    return models

