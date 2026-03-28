import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

WORKSPACE    = Path(__file__).parent
BEST_WEIGHTS = WORKSPACE / "runs" / "pothole_yolo26n" / "weights" / "best.pt"


def run(image_path: str, weights: Path, conf: float, show: bool):
    if not weights.exists():
        print(f"Error: weights not found at {weights}")
        print("Make sure training is complete before running inference.")
        sys.exit(1)

    image = Path(image_path)
    if not image.exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    model = YOLO(str(weights))
    results = model.predict(source=str(image), conf=conf, save=True, show=show)

    result = results[0]
    boxes  = result.boxes

    print(f"\nImage : {image.name}")
    print(f"Detections: {len(boxes)}")
    print("-" * 40)

    if len(boxes) == 0:
        print("No potholes detected.")
    else:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence      = box.conf[0].item()
            print(f"  [{i+1}] pothole  conf={confidence:.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    save_dir = Path(result.save_dir)
    print(f"\nAnnotated image saved to: {save_dir / image.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole inference using trained YOLO26n")
    parser.add_argument("image",          type=str,            help="Path to input image")
    parser.add_argument("--weights",      type=str,            default=str(BEST_WEIGHTS), help="Path to model weights")
    parser.add_argument("--conf",         type=float,          default=0.25,  help="Confidence threshold (0.0–1.0)")
    parser.add_argument("--show",         action="store_true", default=False, help="Display image in a window")
    args = parser.parse_args()

    run(args.image, Path(args.weights), args.conf, args.show)
