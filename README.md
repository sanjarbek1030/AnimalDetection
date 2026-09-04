# # 🐾 Restricted Zone Animal Detector

A real-time computer vision system that uses **YOLOv8** and **OpenCV** to detect animals in a video and identify when they enter a predefined restricted zone.

The system currently detects:

* 🐦 Birds
* 🐱 Cats
* 🐶 Dogs

Animals outside the restricted zone are displayed with a **green bounding box**. When an animal enters the restricted area, the system displays a **red bounding box** and a warning message.

---

## 🎯 Project Goal

The goal of this project is to demonstrate how object detection can be combined with **geographical/zone-based rules** to create a simple video surveillance system.

Instead of simply detecting an animal, the system asks:

> **"Is this animal inside an area where it should not be?"**

This concept can be adapted for applications such as:

* 🏠 Pet monitoring
* 🌱 Protected gardens
* 🚜 Agricultural areas
* 🏭 Industrial zones
* 🐄 Farm monitoring
* 🏞️ Wildlife protection
* 🚧 Restricted areas

---

## ✨ Features

* YOLOv8 object detection
* Detects birds, cats, and dogs
* User-defined rectangular restricted zone
* Detects whether an animal enters the zone
* Green bounding boxes for normal detections
* Red bounding boxes for restricted-zone intrusions
* Warning message when an intruder is detected
* Confidence percentage displayed for each detection
* Processes videos frame by frame
* Automatically creates an annotated output video
* Beginner-friendly Python implementation

The project uses `ultralytics` for YOLOv8 and `opencv-python` for video processing and visualization.

---

## 🧠 How It Works

The processing pipeline is:

```text
Input Video
     │
     ▼
Read Video Frame
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
Filter Animals
(Bird / Cat / Dog)
     │
     ▼
Calculate Bounding Box Center
     │
     ▼
Is Center Inside Restricted Zone?
     │
   ┌─┴──────────────┐
   │                │
  NO               YES
   │                │
   ▼                ▼
GREEN BOX        RED BOX
                  +
               WARNING
   │                │
   └───────┬────────┘
           ▼
      Output Video
```

---

## 🔍 Detection Logic

The YOLOv8 model is pretrained on the **COCO dataset**.

This project filters the detections to only three classes:

| Class ID | Animal |
| -------: | ------ |
|       14 | Bird   |
|       15 | Cat    |
|       16 | Dog    |

The confidence threshold is set to **0.4**, meaning detections below 40% confidence are ignored.

---

## 🚨 Restricted Zone

The restricted area is represented by a rectangle.

You can customize its position using four coordinates:

```python
ZONE_X_MIN = 640
ZONE_Y_MIN = 200
ZONE_X_MAX = 1100
ZONE_Y_MAX = 600
```

The coordinates represent:

```text
(ZONE_X_MIN, ZONE_Y_MIN)
          ┌──────────────────────┐
          │                      │
          │   RESTRICTED ZONE    │
          │                      │
          └──────────────────────┘
                    (ZONE_X_MAX, ZONE_Y_MAX)
```

The system checks the **center point of each animal's bounding box** to determine whether the animal is inside the restricted zone.

---

## 🟢 Normal Detection

When an animal is outside the restricted area:

```text
┌─────────────────┐
│      🐶         │
│     DOG         │
│                 │
└─────────────────┘
```

The bounding box is **green**.

---

## 🔴 Restricted Zone Detection

When the center of an animal enters the restricted area:

```text
┌─────────────────────────────┐
│   RESTRICTED ZONE           │
│                             │
│        ┌─────────┐          │
│        │   🐶    │          │
│        │   DOG   │          │
│        └─────────┘          │
│                             │
└─────────────────────────────┘

WARNING: ANIMAL IN RESTRICTED AREA
```

The bounding box becomes **red** and a warning banner appears at the top of the video.

---

## 🛠️ Technologies

* **Python**
* **YOLOv8**
* **Ultralytics**
* **OpenCV**
* **Computer Vision**
* **Object Detection**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/restricted-zone-animal-detector.git
cd restricted-zone-animal-detector
```

Install the required libraries:

```bash
pip install ultralytics opencv-python
```

The required dependencies are `ultralytics` and `opencv-python`.

---

## ▶️ How to Run

Place your input video in the project folder and name it:

```text
input_video.mp4
```

Then run:

```bash
python main.py
```

The first time you run the program, YOLOv8 Nano (`yolov8n.pt`) will be downloaded automatically if it is not already available.

After processing, the result will be saved as:

```text
output_video.mp4
```

---

## ⚙️ Configuration

The most important settings are located near the beginning of the Python script.

### Input / Output

```python
INPUT_VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"
```

### YOLO Model

```python
MODEL_NAME = "yolov8n.pt"
```

### Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 0.4
```

### Restricted Zone

```python
ZONE_X_MIN = 640
ZONE_Y_MIN = 200
ZONE_X_MAX = 1100
ZONE_Y_MAX = 600
```

## Change these values to match the area you want to monitor.

## 📁 Project Structure

```text
restricted-zone-animal-detector/
│
├── main.py
├── input_video.mp4
├── output_video.mp4
├── yolov8n.pt
└── README.md
```

---

## 📊 Output

The generated video contains:

* Animal bounding boxes
* Animal names
* Detection confidence
* Restricted-zone overlay
* Red/green detection states
* Intrusion warning

The program preserves the input video's width, height, and FPS when creating the output video.

---

## 🚀 Possible Future Improvements

This project can be expanded significantly.

### 1. Multiple Restricted Zones

Allow users to create multiple protected areas instead of one rectangle.

### 2. Real-Time CCTV

Replace video-file input with:

* Webcam
* CCTV camera
* RTSP stream
* IP camera

### 3. Animal Tracking

Add object tracking so the system can maintain an ID for each animal.

Example:

```text
Dog #1 → Entered Zone
Dog #2 → Outside Zone
Cat #3 → Entered Zone
```

### 4. Notifications

Send an alert when an animal enters the restricted area:

* Email
* Telegram
* Discord
* Mobile notification

### 5. Event Logging

Store intrusion events:

```text
2026-09-04 21:31:02
Dog entered restricted zone
Confidence: 87%
```

### 6. Automatic Zone Selection

Allow users to draw the restricted area directly on the video instead of manually entering coordinates.

---

## 🎓 What I Learned

This project helped me practice several important computer vision concepts:

* Object detection with YOLO
* Working with pretrained models
* Processing videos frame by frame
* Bounding boxes
* Confidence scores
* Coordinate systems
* Region/zone-based detection
* OpenCV visualization
* Video input/output
* Building rule-based logic on top of AI detection

---

## 📌 Disclaimer

This project is intended for educational and demonstration purposes. Detection performance can vary depending on video quality, lighting, camera angle, animal size, and other environmental conditions.

---

## ⭐ If You Like This Project

If you found this project useful, consider giving the repository a ⭐ on GitHub!

More computer vision projects coming soon.
