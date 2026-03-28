import os
import subprocess
import sys

def ensure_ultralytics():
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

ensure_ultralytics()

from ultralytics import YOLO
from pathlib import Path

WORKSPACE = Path(__file__).parent
DATASET_YAML = WORKSPACE / "dataset.yaml"
PROJECT_DIR  = WORKSPACE / "runs"
RUN_NAME     = "pothole_yolo26n"

# Model choices (uncomment to switch):
#   yolo26n.pt  — nano,   fastest, least accurate
#   yolo26s.pt  — small
#   yolo26m.pt  — medium
#   yolo26l.pt  — large
#   yolo26x.pt  — extra-large, slowest, most accurate
MODEL_WEIGHTS = "yolo26n.pt"

EPOCHS      = 50
IMG_SIZE    = 416  # smaller size = much faster on CPU (640 → 416 ~2x speedup)
BATCH_SIZE  = 8    # lower batch is more CPU-friendly
WORKERS     = 2    # avoid CPU thread contention (no GPU dataloader needed)
PATIENCE    = 10   # early-stop if no improvement for N epochs


def train():
    print(f"Dataset : {DATASET_YAML}")
    print(f"Model   : {MODEL_WEIGHTS}")
    print(f"Epochs  : {EPOCHS}  |  Img size: {IMG_SIZE}  |  Batch: {BATCH_SIZE}")
    print("-" * 50)

    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        plots=True,
    )

    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved at:\n  {best_weights}")
    return best_weights


def evaluate(weights_path):
    print(f"\nEvaluating on test set with: {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=RUN_NAME + "_test_eval",
        exist_ok=True,
    )
    print(f"\nTest mAP50     : {metrics.box.map50:.4f}")
    print(f"Test mAP50-95  : {metrics.box.map:.4f}")
    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    best = train()
    if best.exists():
        evaluate(best)
