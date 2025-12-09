import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from flask import json
from mlflow import pytorch as mlflow_pytorch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from .utils import save_checkpoint, set_seed


def train(
    train_name: str,
    data_root: Path,
    out_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
    pretrained: bool,
):
    """Main training loop.

    Args:
        data_root (Path): Path to the root directory containing 'train/' and 'val/' subdirectories.
        out_dir (Path): Directory where checkpoints will be saved.
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        img_size (int): Size to which images are resized.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        num_workers (int): Number of worker processes for data loading.
        seed (int): Random seed for reproducibility.
        pretrained (bool): If True, use pretrained weights.
    """
    with mlflow.start_run(run_name=train_name):
        set_seed(seed)  # reproducibility

        # Load data loaders
        train_loader, val_loader, class_to_idx = get_dataloaders(
            data_root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Save training parameters to MLflow
        mlflow.log_params(
            {
                "data_root": str(data_root),
                "out_dir": str(out_dir),
                "device": device,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_workers": num_workers,
                "seed": seed,
                "pretrained": pretrained,
            }
        )

        # Save class_to_idx mapping
        with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=4)

        # Log class_to_idx as MLflow artifact
        mlflow.log_artifacts(str(out_dir / "class_to_idx.json"), artifact_path=".")

        num_classes = len(class_to_idx)
        model = build_algorithm(num_classes, pretrained=pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Log training components to MLflow
        mlflow.log_params({"criterion": type(criterion).__name__, "optimizer": type(optimizer).__name__})

        best_val_acc, best_idx, best_model_uri = 0.0, 0, None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            dt = time.time() - t0

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "process_time": dt,
                },
                step=epoch,
            )

            print(
                f"[{epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_idx = epoch
                ckpt = save_checkpoint(model, optimizer, epoch, best_val_acc, class_to_idx, out_dir)

                # Log best model checkpoint to MLflow
                mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
                # Log best model to MLflow
                mlflow_pytorch.log_model(model, artifact_path="model")

                run: mlflow.ActiveRun = mlflow.active_run()  # type: ignore
                best_model_uri = f"runs:/{run.info.run_id}/model"

                print(f" New best model val_acc={best_val_acc:.4f} → saved: {ckpt}")

        print("Training complete.")
        return best_val_acc, best_idx, best_model_uri


def get_dataloaders(data_root: Path, img_size=224, batch_size=32, num_workers=4):
    """Returns train and validation DataLoaders.

    Args:
        data_root (Path): Path to the root directory containing 'train/' and 'val/' subdirectories.
        img_size (int): Size to which images are resized.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for data loading.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.class_to_idx


def build_algorithm(num_classes: int, pretrained=True):
    """Builds a ResNet18 model modified for the given number of classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    # Modify the final layer to match num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += (outputs.argmax(1) == targets).float().sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates the model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in tqdm(loader, desc="Val", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        running_acc += (outputs.argmax(1) == targets).float().sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


if __name__ == "__main__":
    train(
        "example_train",
        data_root=Path("data\data"),
        out_dir=Path("checkpoints/"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=3,
        batch_size=32,
        img_size=224,
        lr=1e-3,
        weight_decay=1e-4,
        num_workers=1,
        seed=42,
        pretrained=True,
    )
