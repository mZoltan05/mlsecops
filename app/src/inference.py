import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from mlflow import pytorch as mlflow_pytorch
from PIL import Image
from torchvision import models, transforms


# TODO require Model
def infer(model, idx_to_class, img, img_size=224, topk=1):
    """Perform inference on images using a pre-trained model.

    Loads the model and class mappings from the specified directory, processes the input images,
    and outputs the top-k predictions for each image.
    Args:
        model: The pre-trained PyTorch model for inference.
        idx_to_class (dict): Mapping from class indices to class names.
        img (str or Path): Path to a single image or a directory of images.
        img_size (int): Size to which images are resized for inference.
        topk (int): Number of top predictions to return.
    """

    num_classes = len(idx_to_class)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    # Prepare image transformation
    tfm = get_infer_transform(img_size)

    # Handle single image or directory of images
    img = Path(img)
    return predict_image(model, img, tfm, device, idx_to_class, topk=topk)


def load_latest_staging_model(client: mlflow.MlflowClient, model_name: str):
    """Loads the latest model in 'Staging' from MLflow Model Registry."""
    latest_staging = client.get_latest_versions(name=model_name, stages=["Staging"])
    if not latest_staging:
        raise ValueError("No model found in 'Staging' stage.")

    model_uri = latest_staging[0].source
    model = mlflow_pytorch.load_model(model_uri)
    artifacts = client.download_artifacts(latest_staging[0].source, "classes.json")

    # Load class index mapping
    with open(artifacts, "r") as f:
        idx_to_class = json.load(f)

    return model, idx_to_class


def build_model_for_infer(num_classes: int):
    """Builds a ResNet18 model for inference with the specified number of classes."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.eval()
    return model


def get_infer_transform(img_size=224):
    """Returns the image transformation pipeline for inference."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@torch.no_grad()
def predict_image(model, image_path: Path, transform, device, idx_to_class: dict, topk=1):
    """Predicts the top-k classes for a single image using the provided model and transformation."""
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)

    # Compute probabilities and get top-k predictions
    probs = torch.softmax(logits, dim=1)
    top_probs, top_idxs = probs.topk(topk, dim=1)
    out = []

    # Map indices to class names and probabilities
    for p, i in zip(top_probs[0].tolist(), top_idxs[0].tolist()):
        out.append((idx_to_class[str(i)], float(p)))
    return out
