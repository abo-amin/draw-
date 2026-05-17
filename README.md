# Virtual Whiteboard

A real-time virtual whiteboard application that uses hand tracking to draw, erase, and change colors using hand gestures. Built with Python and MediaPipe.

## Features

- **Draw** with your index finger
- **Pause** drawing to reposition your hand (index + middle finger)
- **Erase** the entire canvas by holding a closed fist
- **Change colors** using three fingers
- **Smooth lines** for natural handwriting and drawing
- **Real-time FPS** display

## Gestures

| Gesture | Action |
|---------|--------|
| ☝️ Index finger only | Draw on the canvas |
| ✌️ Index + Middle finger | Pause drawing (move hand freely) |
| 🤟 Three fingers | Change brush color |
| ✊ Closed fist (hold) | Clear the entire canvas |

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abo-amin/draw-.git
cd draw-
```

2. Install dependencies:
```bash
pip install mediapipe opencv-python numpy
```

3. Download the hand landmarker model:
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

## Usage

Run the application:
```bash
python virtual_whiteboard.py
```

Press `ESC` to exit.

## How It Works

1. **Hand Detection**: Uses MediaPipe's HandLandmarker model to detect 21 hand landmarks in real-time
2. **Gesture Recognition**: Analyzes finger positions (tip vs PIP joint) to determine which gesture is being performed
3. **Drawing**: Maps the index finger tip position to canvas coordinates and draws lines/circles
4. **Smoothing**: Applies exponential smoothing to reduce hand tremor and create clean, readable lines
5. **Canvas Overlay**: Combines the camera feed with the drawing canvas using weighted blending

## Project Structure

```
├── virtual_whiteboard.py    # Main application
├── hand_tracking.py         # Basic hand tracking demo
├── test_hand_detection.py   # Hand detection test utility
├── hand_landmarker.task     # MediaPipe hand detection model
└── README.md                # Project documentation
```

## Technologies

- **MediaPipe**: Hand landmark detection and tracking
- **OpenCV**: Image processing and display
- **NumPy**: Canvas and array operations

## License

MIT License
