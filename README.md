# Project:Charm 👁️
### Eye-Tracking Mouse Control via Standard Webcam

**Project:Charm** is a free, open-source, Python-based accessibility tool that transforms any standard webcam into a fully functional eye-tracking mouse controller. No specialized infrared hardware required.

---

## ✨ Features

- **Iris-based gaze tracking** — Uses MediaPipe Face Mesh (478-point model with iris refinement) to detect iris position in real time
- **Full-screen cursor control** — Maps iris position to screen coordinates with calibrated, smoothed mathematical model
- **Eye gesture input** — Left blink → left click, right blink → right click, both blinks → middle click
- **Eye-roll scrolling** — Look up/down to activate scroll mode
- **EMA smoothing** — Exponential Moving Average filter eliminates jitter while maintaining responsiveness
- **5-point calibration** — Interactive routine adapts to your unique eye geometry
- **ESC kill switch** — Instantly pause/resume mouse control without stopping the application
- **Runs 100% locally** — No cloud calls, no data leaves your machine
- **Debug overlay** — Real-time visualization of landmarks, iris tracking, EAR values, and performance metrics

---

## 🖥️ Requirements

- **Python 3.10+**
- A standard webcam (built-in laptop camera works fine)
- Reasonable ambient lighting (avoid extreme backlight)

### Supported Platforms
- ✅ Windows 10/11
- ✅ macOS 12+
- ✅ Linux (Ubuntu 20.04+, see notes below)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/project-charm.git
cd project-charm
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### Linux-specific notes
Some MediaPipe dependencies require system-level packages:
```bash
sudo apt install python3-dev python3-pip
sudo apt install libgl1-mesa-glx  # OpenCV display support
```

### macOS-specific notes
If using an M1/M2 Mac, ensure you're using an ARM-native Python installation for best performance.

---

## 🎮 Usage

### Basic Usage
```bash
python main.py
```

### Command-Line Options
| Flag | Description |
|---|---|
| `--debug` | Enable verbose DEBUG-level logging |
| `--calibrate` | Force recalibration on startup |
| `--no-mouse` | Disable mouse control (overlay-only testing mode) |

### Controls
| Key | Action |
|---|---|
| `ESC` | Pause / Resume mouse control (toggle) |
| `c` | Trigger recalibration |
| `q` | Quit the application |

---

## 🎯 Calibration Guide

On first run, Project:Charm will guide you through a **5-point calibration**:

1. A fullscreen window appears with a green crosshair target
2. **Look directly at the target** (don't move your head — move only your eyes)
3. **Press SPACE** when you're focused on the target
4. The system collects 30 frames of iris data (first 10 discarded for settling)
5. Repeat for all 5 points: top-left, top-right, bottom-right, bottom-left, center

**Tips for best calibration:**
- Keep your head still throughout — only move your eyes
- Maintain consistent distance from the screen (~50–70 cm)
- Ensure even lighting on your face (avoid strong backlighting)
- Calibration is saved to `~/.charm_calibration.json` and reloaded automatically

To recalibrate at any time, press `c` during runtime or use `--calibrate` flag.

---

## 🏗️ Architecture

```
project_charm/
├── main.py                  # Entry point, main loop orchestration
├── config.py                # All constants, thresholds, and tunable params
├── capture.py               # Webcam abstraction layer
├── landmark_processor.py    # MediaPipe integration, landmark extraction
├── gaze_mapper.py           # Iris-to-screen coordinate mapping & EMA
├── gesture_detector.py      # Blink/gesture classification logic
├── mouse_controller.py      # PyAutoGUI mouse action abstraction
├── calibration.py           # Interactive calibration routine
├── overlay.py               # OpenCV debug/status overlay renderer
├── requirements.txt         # All pip dependencies
└── README.md                # This file
```

### Pipeline Flow
```
Webcam Frame → MediaPipe Face Mesh (478 landmarks)
  → Iris Center Extraction (binocular average)
    → EMA Smoothing Filter
      → Calibrated Linear Interpolation → Screen Coordinates
        → PyAutoGUI Mouse Movement

  → EAR Blink Detection (per-eye state machines)
    → Gesture Classification → Mouse Click/Scroll Events
```

---

## ⚙️ Configuration

All tunable parameters are centralized in `config.py`. Key values:

| Parameter | Default | Description |
|---|---|---|
| `EMA_ALPHA` | `0.12` | Smoothing factor (lower = smoother, higher = more responsive) |
| `EAR_BLINK_THRESHOLD` | `0.21` | EAR below this triggers a blink |
| `BLINK_MAX_DURATION_MS` | `300` | Max blink duration before it's classified as a squint |
| `SCROLL_SPEED_PIXELS` | `30` | Scroll amount per frame in scroll mode |
| `TARGET_MONITOR_INDEX` | `0` | Which monitor to map gaze to (0 = primary) |

---

## ⚠️ Safety

- **ESC kill switch** — Press ESC at any time to instantly pause mouse control. Press again to resume. The camera feed continues running so you can see the overlay.
- **Face loss protection** — If your face leaves the frame, the cursor freezes in place (no erratic movement).
- **Squint disambiguation** — Sustained low EAR (tired eyes) won't trigger phantom clicks.

---

## 🔮 Planned Future Features

- **Dwell-click** — Click by holding gaze steady on a target for N milliseconds (no blink required)
- **Left-eye squint = drag lock** — Hold left eye nearly closed to engage click-and-drag
- **Head-pose fallback** — Use nose tip position for coarse cursor movement when iris tracking confidence is low
- **GUI settings panel** — A PyQt6 or Tkinter window to adjust EAR thresholds, EMA alpha, and monitor target at runtime
- **Profiles** — Save/load named calibration profiles for different users or lighting environments

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) — Face Mesh & Iris landmark detection
- [Soukupová & Čech (2016)](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) — Eye Aspect Ratio blink detection
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — Cross-platform mouse automation
