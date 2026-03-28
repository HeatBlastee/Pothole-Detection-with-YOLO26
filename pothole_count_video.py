import sys
import argparse
from pathlib import Path
from collections import Counter
import cv2
from ultralytics import YOLO

WORKSPACE    = Path(__file__).parent
BEST_WEIGHTS = WORKSPACE / "runs" / "pothole_yolo26n" / "weights" / "best.pt"
OUTPUT_DIR   = WORKSPACE / "runs" / "pothole_count_video"
TRACKER_CFG  = WORKSPACE / "tracker_pothole.yaml"


def draw_line_count(img, boxes, count, line_y, counted_ids):
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

    # counting line
    cv2.line(img, (0, line_y), (w, line_y), RED, 2)
    cv2.putText(img, "count line", (10, max(20, line_y - 8)), font, 0.6, RED, 2, cv2.LINE_AA)

    ids = boxes.id
    for j in range(len(boxes)):
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[j].tolist())
        tid   = int(ids[j]) if ids is not None else None
        color = GREEN if tid in counted_ids else YELLOW
        label = f"ID {tid}" if tid is not None else "pothole"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), font, 0.6, WHITE, 2, cv2.LINE_AA)

    # running counted-total banner (top-left)
    banner = f"Potholes counted: {count}"
    (tw, th), _ = cv2.getTextSize(banner, font, 0.8, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), BLACK, -1)
    cv2.putText(img, banner, (10, th + 10), font, 0.8, YELLOW, 2, cv2.LINE_AA)
    return img


def process(source: str, weights: Path, conf: float, vid_stride: int, line_pos: float, min_hits: int):
    if not weights.exists():
        print(f"Error: weights not found at {weights}")
        print("Make sure training is complete before running video detection.")
        sys.exit(1)

    src = Path(source)
    if not src.exists():
        print(f"Error: video not found at {source}")
        sys.exit(1)

    model = YOLO(str(weights))

    # read source fps so the output video keeps a sensible duration
    cap     = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    out_fps = max(1.0, (src_fps or 25.0) / vid_stride)

    tracker_arg = str(TRACKER_CFG) if TRACKER_CFG.exists() else "bytetrack.yaml"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / (src.stem + "_detected.mp4")

    print(f"Model : {weights.name}   |   Conf: {conf}   |   Stride: {vid_stride}   |   Line: {line_pos}   |   Min hits: {min_hits}")
    print("Counting potholes by line-crossing (this can be slow on CPU)...")
    print("=" * 55)

    writer  = None
    frames  = 0
    count   = 0
    id_hits = Counter()   # track id -> times seen
    prev_cy = {}          # track id -> previous normalized center-y
    counted = set()       # track ids already counted at the line

    # A pothole is counted ONCE, when its tracked box center crosses the line
    # (at line_pos, fraction of height) moving downward -> as the vehicle passes
    # it. Far-away ids stay above the line and are never counted.
    for result in model.track(
        source=str(src), stream=True, conf=conf,
        vid_stride=vid_stride, tracker=tracker_arg, verbose=False,
    ):
        frame  = result.orig_img.copy()
        h, w   = frame.shape[:2]
        line_y = int(line_pos * h)
        boxes  = result.boxes

        if boxes.id is not None:
            ids  = boxes.id.tolist()
            xyxy = boxes.xyxy.tolist()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                tid = int(tid)
                id_hits[tid] += 1
                cy   = ((y1 + y2) / 2.0) / h
                prev = prev_cy.get(tid)
                crossed_down = prev is not None and prev < line_pos <= cy
                if crossed_down and tid not in counted and id_hits[tid] >= min_hits:
                    counted.add(tid)
                    count += 1
                prev_cy[tid] = cy

        annotated = draw_line_count(frame, boxes, count, line_y, counted)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))

        writer.write(annotated)
        frames += 1
        if frames % 20 == 0:
            print(f"  processed {frames} frames... potholes counted so far: {count}")

    if writer is not None:
        writer.release()

    print("=" * 55)
    print(f"Frames processed       : {frames}")
    print(f"TOTAL potholes counted : {count}   (line-crossing at y={line_pos:.2f})")
    print(f"\nAnnotated video saved to: {out_path}")
    return count, frames, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and count unique potholes in a video using the trained YOLO26 model")
    parser.add_argument("source",       type=str,   help="Path to the input video")
    parser.add_argument("--weights",    type=str,   default=str(BEST_WEIGHTS), help="Path to model weights")
    parser.add_argument("--conf",       type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--vid-stride", type=int,   default=1, help="Process every Nth frame (higher = faster, less accurate count)")
    parser.add_argument("--line-pos",   type=float, default=0.60, help="Counting line position, fraction of height from top (0.1-0.95)")
    parser.add_argument("--min-hits",   type=int,   default=2, help="Times a track must be seen before a crossing counts (filters noise)")
    args = parser.parse_args()

    process(args.source, Path(args.weights), args.conf, args.vid_stride, args.line_pos, args.min_hits)
