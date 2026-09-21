# YOLO Bottle Detection, Tracking & Counting

An automated computer vision pipeline using Ultralytics YOLOv8 and OpenCV to detect, track, and dynamically count unique bottles in video streams or live camera feeds.

---

## Key Features

- **Object Detection:** Detects bottles with high precision using the pre-trained `yolov8n.pt` architecture (COCO class ID `39`).
- **Persistent Tracking:** Tracks bottle instances across sequential frames using YOLOv8's built-in tracking module.
- **Unique Counting:** Assigns and tracks persistent IDs to calculate a non-duplicative total count of bottles in real time.
- **Visual Feedback:** Overlays bounding boxes, object IDs, and live counting metrics directly onto the video feed.

---

## Project Structure

```text
Bottle Tracking/
├── 00.py                # Main detection, tracking, and counting script
├── bottle vedio.mp4     # Input test video sample
├── yolov8n.pt           # YOLOv8 nano pre-trained weights
├── .gitignore           # Ignores virtual environments and cache
└── README.md            # Project documentation
