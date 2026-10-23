
import os
import json
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path

# Project files
model_path = Path("artifacts/training/model_v2.h5")
scores_path = Path("scores.json")

# Verify files exist
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not scores_path.exists():
    raise FileNotFoundError(f"Scores not found: {scores_path}")

# Configure DagsHub MLflow tracking
tracking_uri = (
    "https://dagshub.com/AmruthGowda34/"
    "kidney-Disease-Classifiaction-DLProject.mlflow"
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Kidney-Disease-Classification")

# Read evaluation results
with open(scores_path, "r") as file:
    scores = json.load(file)

print("Evaluation scores:", scores)

# Load the existing trained model; do not retrain
model = tf.keras.models.load_model(
    str(model_path),
    compile=False
)

# Log model and available evaluation metrics
with mlflow.start_run(run_name="kidney-model-oct21"):
    for key, value in scores.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(key, float(value))

    mlflow.log_param("model_architecture", "VGG16")
    mlflow.log_param("num_classes", 4)
    mlflow.log_param("training_epochs", 5)

    mlflow.keras.log_model(
        model,
        artifact_path="model"
    )

    # Preserve the original evaluation report
    mlflow.log_artifact(str(scores_path))

    print("Model and evaluation scores logged successfully!")
