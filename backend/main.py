import os
import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from starlette.background import BackgroundTask

from detector import detect, detect_video, WEIGHTS_PATH

app = FastAPI(title="Pothole Detection API", version="1.1")

# Allow simple browser/API clients to call this service during local demos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Pothole-Count", "X-Frames-Processed"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Quick check that the API is up and the weights are reachable."""
    return {
        "status": "ok",
        "weights": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.exists(),
    }


@app.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Accept an uploaded image, run pothole detection, and return:
      - count   : number of potholes detected
      - message : human-readable count string
      - image   : annotated image (numbered boxes) as a base64 data URI
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        count, annotated_bytes = detect(image_bytes, conf=conf)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    b64     = base64.b64encode(annotated_bytes).decode()
    message = f"{count} pothole{'s' if count != 1 else ''} detected"

    return {
        "count": count,
        "message": message,
        "image": f"data:image/jpeg;base64,{b64}",
    }


@app.post("/detect_video")
async def detect_video_endpoint(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    vid_stride: int = Query(1, ge=1, le=10, description="Process every Nth frame"),
    line_pos: float = Query(0.60, ge=0.1, le=0.95, description="Counting line position (fraction of height from top)"),
    min_hits: int = Query(2, ge=1, le=30, description="Times a track must be seen before a crossing counts"),
):
    """
    Accept an uploaded video, count potholes via LINE-CROSSING, and return the
    annotated video file. The total count is sent in the `X-Pothole-Count`
    response header (and frames processed in `X-Frames-Processed`).

    A pothole is counted once, when its track crosses the line at `line_pos`
    moving downward — robust to one approaching pothole being re-counted as it
    grows/moves toward a forward-facing camera.
    """
    if file.content_type is None or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    in_fd, in_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(in_fd, "wb") as f:
        f.write(await file.read())
    out_path = in_path + "_detected.mp4"

    def cleanup():
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass

    try:
        count, frames = detect_video(in_path, out_path, conf=conf,
                                     vid_stride=vid_stride, line_pos=line_pos, min_hits=min_hits)
    except FileNotFoundError as e:
        cleanup()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        cleanup()
        raise HTTPException(status_code=500, detail=f"Video detection failed: {e}")

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=f"detected_{Path(file.filename or 'video').stem}.mp4",
        headers={
            "X-Pothole-Count": str(count),
            "X-Frames-Processed": str(frames),
        },
        background=BackgroundTask(cleanup),
    )
