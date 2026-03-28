import sys
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

WORKSPACE    = Path(__file__).parent
BEST_WEIGHTS = WORKSPACE / "runs" / "pothole_yolo26n" / "weights" / "best.pt"
OUTPUT_DIR   = WORKSPACE / "runs" / "pothole_count"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather_images(source: Path):
    """Return a list of image paths from a single file or a directory."""
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [source]


def draw_numbered(image_path: Path, boxes, out_path: Path):
    """Draw each detection with a sequential number (1, 2, 3 ...) on the image."""
    img = cv2.imread(str(image_path))

    GREEN  = (0, 200, 0)
    WHITE  = (255, 255, 255)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = str(i)

        # bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)

        # filled background tag for the number so it stays readable
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), GREEN, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), font, 0.7, WHITE, 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def count(source_path: str, weights: Path, conf: float):
    if not weights.exists():
        print(f"Error: weights not found at {weights}")
        print("Make sure training is complete before counting.")
        sys.exit(1)

    source = Path(source_path)
    if not source.exists():
        print(f"Error: path not found at {source_path}")
        sys.exit(1)

    images = gather_images(source)
    if not images:
        print(f"No images found in {source_path}")
        sys.exit(1)

    model = YOLO(str(weights))

    print(f"Model : {weights.name}   |   Conf threshold: {conf}")
    print("=" * 50)

    total_potholes = 0
    for image in images:
        results = model.predict(source=str(image), conf=conf, save=False, verbose=False)
        boxes = results[0].boxes
        n = len(boxes)
        total_potholes += n

        out_path = OUTPUT_DIR / image.name
        draw_numbered(image, boxes, out_path)

        print(f"  {image.name:<30} -> {n} pothole(s)")

    print("=" * 50)
    print(f"Images scanned    : {len(images)}")
    print(f"TOTAL potholes    : {total_potholes}")
    if len(images) > 1:
        print(f"Average per image : {total_potholes / len(images):.2f}")
    print(f"\nNumbered images saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count potholes and label each with a number on the image")
    parser.add_argument("source",    type=str,            help="Path to an image OR a folder of images")
    parser.add_argument("--weights", type=str,            default=str(BEST_WEIGHTS), help="Path to model weights")
    parser.add_argument("--conf",    type=float,          default=0.25,  help="Confidence threshold (0.0-1.0)")
    args = parser.parse_args()

    count(args.source, Path(args.weights), args.conf)
