Vehicle & Object Detection

Real-time vehicle and object detection with YOLOv8, plus license plate text extraction on any detected vehicle. Works from your browser's camera (phone or laptop) or from an uploaded image.

What it does
Live detection from your browser's camera (desktop or mobile)
Or upload a single image if you'd rather not deal with camera permissions
YOLOv8n for detection, split into vehicles vs. other objects
Reads plate text off detected vehicles using EasyOCR
Detection log with CSV export, separated by category
Stack
Streamlit for the UI
streamlit-webrtc for live browser camera input
Ultralytics YOLOv8 (nano weights) for detection
EasyOCR for plate text extraction
OpenCV / Pillow for image handling
Running it
bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py

First run downloads the YOLO and EasyOCR weights automatically — EasyOCR's first load takes a bit longer (~30s) while it pulls the weights.

Notes
Plate reading works best on clear, front/rear-facing shots — angled or low-res plates are hit or miss
Lower the confidence threshold in the sidebar if it's missing objects, raise it if you're getting false positives
License

MIT
