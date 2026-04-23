import torch
import torch.nn as nn


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train(models, dataloader, optimizer, criterion, device):

    model1 = models[0]
    model2 = models[1]

    model1.train()
    model2.train()

    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        z = model1(x)
        pred = model2(z)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate(models, dataloader, criterion, device):

    model1 = models[0]
    model2 = models[1]

    model1.eval()
    model2.eval()

    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            z = model1(x)
            pred = model2(z)

            loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------------
# TRAINER
# -------------------------
def trainer(models, train_loader, val_loader, config, epoch_save, logs, device):

    optimizer = torch.optim.Adam(
        models.parameters(),   # ✅ ModuleList gives all params automatically
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.95
    )

    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(config["epochs"]):

        # -----------------
        # train
        # -----------------
        train_loss = train(
            models, train_loader, optimizer, criterion, device
        )

        # -----------------
        # validate
        # -----------------
        val_loss = evaluate(
            models, val_loader, criterion, device
        )

        # -----------------
        # scheduler
        # -----------------
        scheduler.step()

        # -----------------
        # logging
        # -----------------
        if logs is not None:
            logs.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

        print(f"Epoch {epoch} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        # -----------------
        # save best model
        # -----------------
        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model1": models[0].state_dict(),
                    "model2": models[1].state_dict(),
                },
                "./checkpoints/best_model.pt"
            )

        # -----------------
        # periodic saving
        # -----------------
        if epoch % epoch_save == 0:
            torch.save(
                {
                    "model1": models[0].state_dict(),
                    "model2": models[1].state_dict(),
                },
                f"./checkpoints/model_epoch_{epoch}.pt"
            )

    return models