# Pothole Detection — Backend (FastAPI)

Serves the trained YOLO26 model as a REST API. Open the app in a browser and use FastAPI's built-in docs UI to upload demo files.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`; it redirects to `http://localhost:8000/docs`.

The API loads `../runs/pothole_yolo26n/weights/best.pt` automatically.
To use different weights:
```bash
POTHOLE_WEIGHTS=/path/to/best.pt uvicorn main:app --port 8000
```

## Endpoints
- `GET  /health`  — status + weights path
- `POST /detect`  — upload an image (`file`), optional `?conf=0.25`
  - returns JSON: `{ "count": int, "message": str, "image": "data:image/jpeg;base64,..." }`
- `POST /detect_video` — upload a video (`file`), optional `?conf=0.25&vid_stride=1`
  - returns the annotated **video file** (`video/mp4`)
  - total UNIQUE potholes in the `X-Pothole-Count` response header
  - frames processed in the `X-Frames-Processed` header
  - counting uses object **tracking** (persistent IDs), so the same pothole
    across many frames is counted once
- Interactive docs/upload UI at `http://localhost:8000/docs`

## Quick test
```bash
# image
curl -X POST "http://localhost:8000/detect?conf=0.45" \
  -F "file=@../test_image.jpg"

# video (saves annotated video, prints the count header)
curl -D - -X POST "http://localhost:8000/detect_video?conf=0.25&vid_stride=3" \
  -F "file=@../test_video.mp4" -o detected.mp4
```
