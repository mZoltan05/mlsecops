import json
import random
from pathlib import Path
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def save_checkpoint(model, optimizer, epoch, best_val_acc, class_to_idx, out_dir: Path):
    """Saves the model checkpoint and class mapping."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model_best.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_to_idx": class_to_idx,
        },
        ckpt_path,
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    return ckpt_path

def load_checkpoint(model, ckpt_path: Path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    return ckpt

@torch.no_grad()
def accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()
