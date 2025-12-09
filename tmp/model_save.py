import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://192.168.8.13:50004"))


RUN_NAME = "torchscript_run2"
RUN_ARTIFACTS = Path("artifacts")  # TODO fix
RUN_ARTIFACTS.mkdir(exist_ok=True)


def log_fn(path: str, msg: str):
    log_file = os.path.join(path, "train.log")
    with open(log_file, "a") as f:
        f.write(msg + "\n")


# -----------------------------------------------------------------------------
# 1. Define a simple model
# -----------------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Dummy data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# -----------------------------------------------------------------------------
# 2. Train the model (very short dummy training)
# -----------------------------------------------------------------------------
def train():
    for epoch in range(5):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return loss


data = {
    "data": {
        "dataset": {
            "augment": True,
            "create_new_ds_split": True,
            "noise_rglob_str": None,
            "max_len": 800,
        },
        "data_loader": {
            "num_workers": 16,
            "prefetch_factor": 12,
            "pin_memory": True,
        },
        "sampler": {},
    }
}
# -----------------------------------------------------------------------------
# 3. Track with MLflow
# -----------------------------------------------------------------------------
mlflow.set_experiment("TorchScript_Example2")

with mlflow.start_run(run_name="train_2025_11_04") as run:
    # Log training params and metrics
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("epochs", 5)
    mlflow.log_param("config", data)
    final_loss = -1
    for epoch in range(5):
        optimizer.zero_grad()
        pred = model(x)
        step_loss = criterion(pred, y)
        step_loss.backward()

        if step_loss.item() < final_loss or final_loss == -1:
            final_loss = step_loss.item()

        optimizer.step()

        Path(f"artifacts/step_{epoch+1}").mkdir(exist_ok=True)

        log_fn(f"artifacts/step_{epoch+1}", f"Epoch {epoch+1}, Loss: {step_loss.item():.4f}")

        mlflow.log_metric("step_loss", step_loss.item(), step=epoch + 1)
        # # Log the eager (standard) model
        # mlflow.pytorch.log_model(model, artifact_path="model_eager", step=epoch + 1)

        # -----------------------------------------------------------------------------
        # 4. Convert and log the TorchScript version
        # -----------------------------------------------------------------------------
        example_input = torch.randn(1, 10)
        scripted_model = torch.jit.trace(model, example_input)

        mlflow.pytorch.log_model(scripted_model, artifact_path="model_torchscript_" + str(epoch + 1), step=epoch + 1)

        # # Optional: also save raw .pt file as an artifact
        ts_path = f"artifacts/step_{epoch+1}/model_scripted.pt"
        scripted_model.save(ts_path)
        # mlflow.log_artifact(ts_path, artifact_path="torchscript_artifacts")

    mlflow.log_metric("final_loss", final_loss)
    mlflow.log_artifacts(str(RUN_ARTIFACTS), artifact_path="run_artifacts")

# -----------------------------------------------------------------------------
# 5. (Optional) Register the TorchScript model in the Model Registry
# -----------------------------------------------------------------------------
model_uri = f"runs:/{run.info.run_id}/model_torchscript_4"
mlflow.register_model(model_uri=model_uri, name="SimpleTorchScriptModel")
print("Model registered in MLflow Model Registry.")
