# Vehicle & Object Detection

Real-time vehicle and object detection using YOLOv8, running straight inside
a Streamlit app — no separate backend or cloud job needed. Point it at a
webcam, an RTSP/IP camera stream, or just upload an image, and it'll draw
boxes and log everything it finds.

I split detections into two buckets (vehicles vs. everything else) since
that's usually what you actually care about when you're doing traffic or
parking-lot type monitoring.

## What it does

- Live detection from a webcam or an RTSP stream, or from a single uploaded
  image
- Draws bounding boxes on the frame using YOLOv8n (the small model — fast
  enough to run without a GPU)
- Separates vehicle detections (car, truck, bus, motorcycle, bicycle, train)
  from other object detections
- Keeps a running log with timestamps and confidence scores
- Exportable as two separate CSVs
- Adjustable confidence threshold so you can tune how strict it is

## Stack

- Streamlit for the UI
- Ultralytics YOLOv8 for detection (runs locally, pretrained COCO weights)
- OpenCV for video capture
- Pandas for the logs/export

## Running it

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py
```

First run will download the YOLOv8n weights automatically (~6MB), no
account or API key needed.

For an RTSP camera, just paste the stream URL in the sidebar — format is
usually `rtsp://username:password@ip:port/stream`, check your camera's
manual for the exact path.

## Deploying

This one's a bit trickier to put on Streamlit Cloud since webcam access
needs to happen on the viewer's own machine, not the server. For a public
demo, the "Upload image" mode works fine on Cloud. For live webcam/RTSP
detection, it's meant to be run locally or on a machine that actually has
the camera attached.

## Notes

- Using the nano YOLOv8 model for speed — swap `MODEL_PATH` in `app.py` for
  `yolov8s.pt` or bigger if you want better accuracy and have the compute
  for it
- Detections are logged every ~2 seconds during live mode rather than every
  frame, otherwise the CSV fills up with near-duplicate rows
- No cloud backend, no external API — this used to call out to a hosted
  endpoint for inference but running the model locally turned out to be
  simpler and doesn't depend on anything else being up

## License

MIT
