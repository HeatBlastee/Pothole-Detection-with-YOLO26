import os
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# The backend folder lives inside the project root, so go one level up to
# reach the trained weights produced by train.py.
PROJECT_ROOT    = Path(__file__).parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "pothole_yolo26n" / "weights" / "best.pt"

# Allow overriding the weights path via environment variable.
WEIGHTS_PATH = Path(os.getenv("POTHOLE_WEIGHTS", str(DEFAULT_WEIGHTS)))

# Tuned tracker config (keeps the same pothole on one id across frames).
TRACKER_CFG = PROJECT_ROOT / "tracker_pothole.yaml"

_model = None


def get_model():
    """Load the YOLO model once and cache it (lazy singleton)."""
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Trained weights not found at {WEIGHTS_PATH}. "
                "Run train.py first, or set POTHOLE_WEIGHTS to a valid best.pt path."
            )
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def _draw_numbered(img, boxes):
    """Draw each detection box and label it with a sequential number (1, 2, 3 ...)."""
    GREEN = (0, 200, 0)
    WHITE = (255, 255, 255)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = str(i)

        # bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)

        # filled tag with the number so it stays readable on any background
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), GREEN, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), font, 0.7, WHITE, 2, cv2.LINE_AA)

    return img


def detect(image_bytes: bytes, conf: float = 0.25):
    """
    Run pothole detection on raw image bytes.

    Returns:
        count          (int)   number of potholes detected
        annotated_jpg  (bytes) JPEG image with numbered boxes drawn
    """
    model = get_model()

    # decode uploaded bytes into an OpenCV image
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")

    results = model.predict(source=img, conf=conf, save=False, verbose=False)
    boxes   = results[0].boxes
    count   = len(boxes)

    annotated = _draw_numbered(img.copy(), boxes)
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("Failed to encode the annotated image.")

    return count, buf.tobytes()


def _draw_line_count(img, boxes, count, line_y, counted_ids):
    """Draw detections, the counting line, and the running counted-total banner.

    Boxes already counted are drawn GREEN, not-yet-counted ones YELLOW.
    """
    GREEN  = (0, 200, 0)
    YELLOW = (0, 215, 255)
    WHITE  = (255, 255, 255)
    RED    = (0, 0, 255)
    BLACK  = (0, 0, 0)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    h, w = img.shape[:2]

    # the counting line
    cv2.line(img, (0, line_y), (w, line_y), RED, 2)
    cv2.putText(img, "count line", (10, max(20, line_y - 8)), font, 0.6, RED, 2, cv2.LINE_AA)

    ids = boxes.id
    for j in range(len(boxes)):
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[j].tolist())
        tid     = int(ids[j]) if ids is not None else None
        color   = GREEN if tid in counted_ids else YELLOW
        label   = f"ID {tid}" if tid is not None else "pothole"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), font, 0.6, WHITE, 2, cv2.LINE_AA)

    banner = f"Potholes counted: {count}"
    (tw, th), _ = cv2.getTextSize(banner, font, 0.8, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), BLACK, -1)
    cv2.putText(img, banner, (10, th + 10), font, 0.8, YELLOW, 2, cv2.LINE_AA)
    return img


def detect_video(input_path: str, output_path: str, conf: float = 0.25,
                 vid_stride: int = 1, line_pos: float = 0.60, min_hits: int = 2):
    """
    Count UNIQUE potholes in a video using LINE-CROSSING counting.

    A horizontal line is placed at `line_pos` (fraction of the frame height from
    the top). A pothole is counted ONCE, the moment its tracked box center
    crosses that line moving downward — i.e. as the vehicle passes it.

    Why this beats counting track ids: a pothole approaching a forward-facing
    camera grows and slides downward fast, which makes the tracker churn through
    several ids. Counting ids would count it many times. With a line, all those
    far-away ids sit ABOVE the line and are ignored; the pothole is counted a
    single time when it sweeps down across the line.

    `min_hits` requires a track to have been seen at least that many times before
    a crossing counts, which rejects one-off false detections.

    Returns:
        count   (int)  number of potholes that crossed the line
        frames  (int)  number of frames processed
    """
    model = get_model()

    cap     = cv2.VideoCapture(input_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    out_fps = max(1.0, (src_fps or 25.0) / vid_stride)

    tracker_arg = str(TRACKER_CFG) if TRACKER_CFG.exists() else "bytetrack.yaml"

    writer  = None
    frames  = 0
    count   = 0
    id_hits = Counter()   # track id -> times seen
    prev_cy = {}          # track id -> previous normalized center-y
    counted = set()       # track ids already counted at the line

    for result in model.track(
        source=input_path, stream=True, conf=conf,
        vid_stride=vid_stride, tracker=tracker_arg, verbose=False,
    ):
        frame   = result.orig_img.copy()
        h, w    = frame.shape[:2]
        line_y  = int(line_pos * h)
        boxes   = result.boxes

        if boxes.id is not None:
            ids  = boxes.id.tolist()
            xyxy = boxes.xyxy.tolist()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                tid = int(tid)
                id_hits[tid] += 1
                cy   = ((y1 + y2) / 2.0) / h          # normalized center-y
                prev = prev_cy.get(tid)
                crossed_down = prev is not None and prev < line_pos <= cy
                if crossed_down and tid not in counted and id_hits[tid] >= min_hits:
                    counted.add(tid)
                    count += 1
                prev_cy[tid] = cy

        annotated = _draw_line_count(frame, boxes, count, line_y, counted)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

        writer.write(annotated)
        frames += 1

    if writer is not None:
        writer.release()

    return count, frames
