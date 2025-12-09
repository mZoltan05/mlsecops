from datetime import datetime
from functools import cache
from pathlib import Path

import mlflow
import uvicorn
from fastapi import FastAPI
from mlflow import tracking as mlflow_tracking
from pydantic import BaseModel

from src.inference import infer, load_latest_staging_model
from src.train import train
import os

HOST = os.getenv("HOST","0.0.0.0")
PORT = os.getenv("PORT","8000")

print(f"Starting app on {HOST}:{PORT}")

# Set mlflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:50000"))

ACCURACY_CRITERION = 0.0  # Define a criterion for acceptable accuracy
MODEL_NAME = "AnimalClassifier"

# Set default experiment
experiment_name = "resnet18_animals141"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

client = mlflow.MlflowClient()
app = FastAPI()


class TrainRequest(BaseModel):
    data_root: str
    out_dir: str
    device: str
    epochs: int
    batch_size: int
    img_size: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    pretrained: bool


class InferRequest(BaseModel):
    img_pth: str
    topk: int = 1


@app.post("/train")
def train_endpoint(req: TrainRequest):
    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M")
    train_name = f"train_{timestamp}"

    params = req.model_dump()
    params["data_root"] = Path(req.data_root)
    params["out_dir"] = Path(req.out_dir)
    params["train_name"] = train_name

    best_accuracy, best_idx, best_model_uri = train(**params)
    if best_accuracy > ACCURACY_CRITERION:
        # Register the best model
        registered_model_version = mlflow.register_model(best_model_uri, MODEL_NAME)

        # Set the model stage to "Staging"
        mlflow_tracking.MlflowClient().transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model_version.version,
            stage="Staging",
        )
    else:
        registered_model_version = None

    return {
        "status": "Training complete.",
        "best_accuracy": best_accuracy,
        "best_epoch": best_idx,
        "registered_model_version": registered_model_version,
    }


@app.post("/infer")
def infer_endpoint(req: InferRequest):
    model, idx_to_class = get_model()
    # Use the last staging model for inference
    result = infer(
        model=model,
        idx_to_class=idx_to_class,
        img=req.img_pth,
        topk=req.topk,
    )
    return {"result": result}


@cache
def get_model():
    return load_latest_staging_model(client, MODEL_NAME)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=int(PORT))
