# Navigate Eye

**AI Vision Assist for Visually Impaired People**

Navigate Eye is a capstone project that uses real-time computer vision and audio guidance to support visually impaired users during indoor and outdoor navigation. The system captures live video through a camera, detects important objects using a trained YOLOv8n model, then converts the detection result into spoken alerts through a connected speaker.

The system is designed to run locally on a Raspberry Pi 4 with limited hardware resources. It does not depend on an internet connection during detection, which makes it more private, portable, and suitable for real-world use.

---

## Project Information

| Item | Details |
|---|---|
| Project Name | Navigate Eye |
| Project Type | Bachelor of Computer Science Capstone Project |
| Main Goal | Assist visually impaired users using AI object detection and audio alerts |
| Team Members | Abdullah Hourani, Nour AbuAlsoud, Omar Almonayer |
| Supervisor | Dr. Malik Louzi |
| Main Model | YOLOv8n |
| Deployment Device | Raspberry Pi 4 |
| Main Language | Python |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Features](#main-features)
- [Detection Classes](#detection-classes)
- [System Workflow](#system-workflow)
- [Hardware Components](#hardware-components)
- [Software and Tools](#software-and-tools)
- [Current Project Structure](#current-project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Run on Raspberry Pi](#run-on-raspberry-pi)
- [Run with a USB Camera](#run-with-a-usb-camera)
- [Run on a Video File](#run-on-a-video-file)
- [Alert Logic](#alert-logic)
- [English and Arabic Audio](#english-and-arabic-audio)
- [Model and Dataset Notes](#model-and-dataset-notes)
- [Performance and Results](#performance-and-results)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

Navigate Eye is a wearable assistive system. A camera captures the surrounding environment and sends frames to the Raspberry Pi. The trained YOLOv8n model detects important navigation objects such as traffic lights, crosswalks, doors, cars, people, stairs, and obstacles.

After detection, the system checks the object priority, distance, position, confidence score, and alert cooldown. Then it speaks the most important alert to the user.

Example alerts:

- `Stop! Red light!`
- `Car ahead. Careful.`
- `Stairs ahead. Watch your step.`
- `Crosswalk ahead.`
- `Door ahead.`

---

## Main Features

- Real-time object detection using YOLOv8n.
- Raspberry Pi 4 support.
- USB camera support.
- Raspberry Pi CSI camera support through `picamera2` when available.
- English and Arabic voice alerts.
- Priority-based alert system.
- Anti-spam cooldown system to avoid repeated audio.
- Temporal validation to reduce false detections.
- Distance estimation using bounding box size.
- Object position guidance: left, center, or right.
- Video-file testing and output video generation.
- Offline text-to-speech using `pyttsx3` and `espeak`.

---

## Detection Classes

The trained model uses 9 classes. The order must not be changed because it matches `data.yaml` and the trained model weights.

| Class ID | Class Name | Purpose | Alert Importance |
|---:|---|---|---|
| 0 | `traffic_light_red` | Warns the user to stop | Critical |
| 1 | `traffic_light_yellow` | Warns the user to be careful | High |
| 2 | `traffic_light_green` | Gives safe-crossing awareness | Information |
| 3 | `crosswalk` | Helps with road crossing navigation | Information |
| 4 | `door` | Supports indoor navigation | Information |
| 5 | `person` | Warns about nearby people | Warning |
| 6 | `car` | Warns about traffic danger | Critical |
| 7 | `stairs` | Helps prevent falls | Critical |
| 8 | `obstacle` | Warns about general hazards | Warning |

---

## System Workflow

The active detection pipeline works as follows:

1. The camera captures a frame.
2. YOLOv8n detects objects in the frame.
3. Low-confidence detections are filtered.
4. Extra class-specific filters are applied.
5. Non-Maximum Suppression removes duplicate boxes.
6. The history buffer checks if the object appears across frames.
7. The system estimates object distance and position.
8. The alert logic chooses the most important warning.
9. The audio engine speaks the alert in English or Arabic.

This workflow helps the system avoid noisy alerts and focus on useful guidance.

---

## Hardware Components

| Component | Purpose |
|---|---|
| Raspberry Pi 4 | Main processing device |
| USB Camera / Pi Camera | Captures real-time video |
| Speaker | Plays audio alerts |
| SD Card | Stores operating system, code, and model weights |
| Power Supply / Power Bank | Powers the Raspberry Pi |

---

## Software and Tools

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Camera access, frame processing, and drawing detections |
| Ultralytics YOLO | Object detection framework |
| YOLOv8n | Lightweight model suitable for embedded devices |
| PyTorch | Model execution backend |
| pyttsx3 | Offline text-to-speech in Python |
| espeak / espeak-ng | Speech synthesis backend on Raspberry Pi |
| ONNX / ONNX Runtime | Optional optimization path for faster inference |
| Roboflow | Dataset labeling and dataset management |

---

## Current Project Structure

This README is written to match the actual files in this repository.

```text
GradProj/
├── README.md
├── data.yaml
├── install_pi.sh
├── requirements-pi.txt
├── run_usb.py
├── run_video.py
├── configs/
│   └── model_config.yaml
├── runs/
│   └── detect/
│       └── smart_glass/
│           ├── args.yaml
│           ├── labels.jpg
│           ├── results.csv
│           └── weights/
│               ├── best.pt
│               └── last.pt
└── src/
    ├── audio_alerts.py
    └── detect.py
```

### Important files

| File | Description |
|---|---|
| `src/detect.py` | Main live detection program |
| `src/audio_alerts.py` | Audio alert engine |
| `run_usb.py` | Helper script for USB camera use |
| `run_video.py` | Runs detection on a video file and saves the output |
| `data.yaml` | Dataset class configuration |
| `configs/model_config.yaml` | Reference configuration for model, detection, priority, audio, and camera settings |
| `runs/detect/smart_glass/weights/best.pt` | Best trained YOLO model |
| `runs/detect/smart_glass/weights/last.pt` | Last training checkpoint |
| `install_pi.sh` | Raspberry Pi setup script |
| `requirements-pi.txt` | Python dependencies for Raspberry Pi |

---

## How to Run the Project

### 1. Go to the project folder

```bash
cd GradProj
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For Raspberry Pi, use:

```bash
pip install -r requirements-pi.txt
```

For desktop testing, these packages are usually enough:

```bash
pip install ultralytics opencv-python pyttsx3 pyyaml numpy torch torchvision
```

### 4. Check that the trained model exists

The default model path is:

```text
runs/detect/smart_glass/weights/best.pt
```

The project will not run if this file is missing.

---

## Run on Raspberry Pi

The project includes a setup script for Raspberry Pi.

```bash
bash install_pi.sh
```

After installation, activate the environment:

```bash
source venv/bin/activate
```

Run with display:

```bash
python3 src/detect.py --device pi --language en
```

Run headless through SSH:

```bash
python3 src/detect.py --device pi --no-display --language en
```

Run with Arabic audio:

```bash
python3 src/detect.py --device pi --no-display --language ar
```

Use a custom model path:

```bash
python3 src/detect.py --device pi --model runs/detect/smart_glass/weights/best.pt --language en
```

---

## Run with a USB Camera

Use the helper script:

```bash
python3 run_usb.py --language en
```

Headless mode:

```bash
python3 run_usb.py --language en --no-display
```

Arabic mode:

```bash
python3 run_usb.py --language ar --no-display
```

You can also run the main script directly:

```bash
python3 src/detect.py --source 0 --device pi --language en
```

Try another camera index if needed:

```bash
python3 src/detect.py --source 1 --device pi --language en
```

---

## Run on a Video File

Use this when testing the model on a recorded video.

```bash
python3 run_video.py --source input.mp4 --output output_detected.mp4 --language en
```

Use a custom model:

```bash
python3 run_video.py --source input.mp4 --output output_detected.mp4 --model runs/detect/smart_glass/weights/best.pt --language en
```

The output video will include bounding boxes and FPS information.

---

## Alert Logic

The system does not speak every detection. It uses filtering and priority logic first.

### Detection filters

- Confidence threshold.
- Class-specific threshold.
- Object size and shape checks.
- Non-Maximum Suppression to remove duplicate boxes.
- Temporal history boost for objects seen across several frames.

### Frame confirmation

| Level | Frames Needed Before Alert |
|---|---:|
| Danger | 1 frame |
| Warning | 2 frames |
| Information | 3 frames |

This means dangerous objects can be spoken faster, while less urgent objects need more confirmation.

### Cooldown system

Each class has a cooldown timer. This prevents the speaker from repeating the same alert again and again.

| Class | Cooldown |
|---|---:|
| `traffic_light_red` | 4 seconds |
| `traffic_light_yellow` | 4 seconds |
| `traffic_light_green` | 8 seconds |
| `crosswalk` | 12 seconds |
| `door` | 10 seconds |
| `person` | 6 seconds |
| `car` | 5 seconds |
| `stairs` | 8 seconds |
| `obstacle` | 6 seconds |

### Distance estimation

The system estimates distance using the object bounding box height compared with the frame height.

| Distance | Meaning |
|---|---|
| Close | Object is large in the frame and may be near the user |
| Medium | Object is ahead but not extremely close |
| Far | Object is too far, so no audio alert is usually spoken |

### Position guidance

For most objects, the alert includes the object position:

- left
- center
- right

Example:

```text
Car ahead. Careful. left
```

---

## English and Arabic Audio

The project supports English and Arabic alert messages.

English example:

```bash
python3 src/detect.py --device pi --language en
```

Arabic example:

```bash
python3 src/detect.py --device pi --language ar
```

The audio system uses offline speech, so internet is not required during real-time detection.

---

## Model and Dataset Notes

The model is trained for the 9 classes listed in `data.yaml`.

The dataset contains indoor and outdoor images, including different object distances, camera angles, and lighting conditions. The final project report mentions around 60,000 images used during development.

The raw dataset is not included in this repository. This repository includes the trained model output and the runtime code needed to test the system.

Training output included in this repository:

```text
runs/detect/smart_glass/
├── args.yaml
├── labels.jpg
├── results.csv
└── weights/
    ├── best.pt
    └── last.pt
```

---

## Performance and Results

The system was tested on Raspberry Pi 4 with limited hardware resources.

Main observed results:

- Around 5 FPS on Raspberry Pi 4.
- Real-time detection is possible with frame skipping and smaller input size.
- Larger objects such as cars and doors were easier to detect.
- Traffic lights were more difficult because they are small, affected by distance, lighting, and color variation.
- The audio cooldown system improved usability by reducing repeated alerts.

Main optimizations used:

- YOLOv8n lightweight model.
- Raspberry Pi mode with smaller prediction image size.
- Frame skipping.
- Camera resolution reduction.
- Temporal filtering.
- Class-based alert cooldowns.

---

## Known Limitations

- Raspberry Pi 4 with 1GB RAM limits speed and model size.
- Poor lighting can reduce detection accuracy.
- Traffic lights can be hard to detect from far distances.
- Crosswalk detection can be affected by faded road lines and camera angle.
- The system estimates distance from bounding box size, not from a real depth sensor.
- Audio quality depends on the speaker and operating system voice support.

---

## Troubleshooting

### Model not found

Make sure this file exists:

```text
runs/detect/smart_glass/weights/best.pt
```

Or pass the model path manually:

```bash
python3 src/detect.py --model runs/detect/smart_glass/weights/best.pt --language en
```

### Camera not opening

Try a different camera source:

```bash
python3 src/detect.py --source 0 --language en
python3 src/detect.py --source 1 --language en
python3 src/detect.py --source 2 --language en
```

On Raspberry Pi, check connected cameras:

```bash
libcamera-hello --list-cameras
v4l2-ctl --list-devices
```

Install Raspberry Pi camera packages:

```bash
sudo apt install python3-libcamera python3-picamera2 v4l-utils
```

### Audio not working

Install speech packages:

```bash
sudo apt install espeak espeak-ng
pip install pyttsx3
```

Test speaker output from the operating system first. If the system has no audio output device, the Python script cannot speak alerts.

### Detection is slow

Use Raspberry Pi mode:

```bash
python3 src/detect.py --device pi --language en
```

Run without display:

```bash
python3 src/detect.py --device pi --no-display --language en
```

You can also reduce the input size in the code or use a lighter export format such as ONNX.

### Too many or too few detections

Change the confidence threshold:

```bash
python3 src/detect.py --conf 0.35 --language en
```

Lower confidence can show more detections but may increase false positives. Higher confidence can reduce false positives but may miss some objects.

---

## Educational Use

This project was developed for academic and research purposes for Al-Hussein Technical University.
