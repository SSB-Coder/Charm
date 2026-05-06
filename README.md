# Project:Charm 👁️
### Eye-Tracking Mouse Control via Standard Webcam

**Project:Charm** is a free, open-source, Python-based accessibility tool that transforms any standard webcam into a fully functional eye-tracking mouse controller. Designed for users with motor disabilities and spectacle wearers. No specialized infrared hardware required.

---

## ✨ Features

- **Iris-based gaze tracking** — MediaPipe Face Mesh (478-point model with iris refinement)
- **One Euro Filter smoothing** — Adaptive filter: smooth when still, responsive when moving
- **Dead zone filtering** — Suppresses micro-jitter below 8px threshold
- **Blink-freeze** — Cursor freezes during blinks to prevent downward jump
- **Eye gesture input** — Left/right/both blinks for clicks, double-blink for double-click
- **Copy shortcut** — Double right-blink sends Ctrl+C
- **Eye-roll scrolling** — Look up/down to activate scroll mode
- **Adaptive EAR baseline** — Auto-calibrates blink threshold for spectacle wearers
- **Cross-eye suppression** — Filters out sympathetic eye dips to prevent false clicks
- **Runtime sensitivity controls** — Adjust smoothing and blink threshold via keyboard
- **5-point calibration** — Interactive routine with persistent save to disk
- **ESC kill switch** — Pause/resume mouse control without stopping the application
- **Runs 100% locally** — No cloud calls, no data leaves your machine

---

## 🖥️ Requirements

- **Python 3.10+**
- A standard webcam (built-in laptop camera works fine)
- Reasonable ambient lighting (avoid extreme backlight)
- **MediaPipe model file**: `face_landmarker.task` in the project root
  - Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

### Supported Platforms
- ✅ Windows 10/11
- ✅ macOS 12+
- ✅ Linux (Ubuntu 20.04+)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/SSB-Coder/Charm.git
cd Charm
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

### 4. Download the MediaPipe model
```bash
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

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

### Keyboard Controls
| Key | Action |
|---|---|
| `ESC` | Pause / Resume mouse control |
| `q` | Quit the application |
| `c` | Trigger recalibration |
| `[` / `]` | Decrease / Increase cursor smoothing |
| `-` / `=` | Decrease / Increase blink sensitivity |

### Gesture Controls
| Gesture | Action |
|---|---|
| Left eye blink | Left click |
| Right eye blink | Right click |
| Both eyes blink | Middle click |
| Double left-blink | Double click |
| Double right-blink | Copy (Ctrl+C) |
| Look up (sustained) | Scroll up |
| Look down (sustained) | Scroll down |

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
Charm/
├── main.py                  # Entry point, pipeline orchestration
├── config.py                # All constants, thresholds, and tunable params
├── capture.py               # Webcam abstraction layer
├── landmark_processor.py    # MediaPipe FaceLandmarker integration
├── gaze_mapper.py           # One Euro Filter, Dead Zone, gaze mapping
├── gesture_detector.py      # Blink detection, gestures, scroll mode
├── mouse_controller.py      # PyAutoGUI mouse/keyboard abstraction
├── calibration.py           # Interactive 5-point calibration routine
├── overlay.py               # Debug/status overlay renderer
├── requirements.txt         # Python dependencies
└── face_landmarker.task     # MediaPipe model (download separately)
```

### Pipeline Flow
```
Webcam Frame → MediaPipe FaceLandmarker (478 landmarks)
  → Iris Center Extraction (binocular average)
    → Blink-Freeze EAR Check (freeze cursor if eyes closing)
      → One Euro Filter (adaptive smoothing)
        → Dead Zone Filter (suppress micro-jitter)
          → Gaze-to-Screen Mapping → Mouse Movement

  → EAR Blink Detection (per-eye state machines with depth tracking)
    → Cross-Eye Sympathetic Suppression
      → Gesture Classification → Mouse Click/Scroll Events
```

---

## ⚙️ Configuration

All tunable parameters are centralized in `config.py`. Key values:

| Parameter | Default | Description |
|---|---|---|
| `ONE_EURO_MIN_CUTOFF` | `0.8` | Smoothing strength (lower = smoother) |
| `ONE_EURO_BETA` | `0.01` | Speed responsiveness (higher = more responsive) |
| `DEAD_ZONE_PIXELS` | `8.0` | Minimum movement to register (pixels) |
| `EAR_BLINK_THRESHOLD` | `0.19` | EAR below this triggers a blink |
| `BLINK_MAX_DURATION_MS` | `400` | Max blink duration before classified as squint |
| `BLINK_BOTH_TOLERANCE_MS` | `40` | Window for dual-blink detection (ms) |
| `CROSS_EYE_SUPPRESSION_MS` | `120` | Sympathetic dip suppression window (ms) |
| `DUAL_BLINK_MIN_DEPTH_RATIO` | `0.65` | Depth required for dual-blink qualification |
| `SCROLL_SPEED_PIXELS` | `30` | Scroll amount per frame |

---

## ♿ Accessibility Features

- **Adaptive EAR Baseline** — Auto-calibrates per-user resting eye state over 60 frames, compensating for spectacle frame occlusion
- **Cross-Eye Sympathetic Suppression** — When one eye blinks, the other may dip involuntarily; shallow dips are filtered to prevent false dual-blinks
- **Blink Depth Tracking** — Measures how deeply each eye closes; only deliberate blinks trigger clicks
- **Blink-Freeze** — Cursor freezes during blinks so clicks fire at the pre-blink position
- **Squint Disambiguation** — Holding eyes closed >400ms is ignored (squinting, not blinking)

---

## ⚠️ Safety

- **ESC kill switch** — Press ESC to instantly pause mouse control. Press again to resume.
- **Face loss protection** — If your face leaves the frame, the cursor freezes and filters reset.
- **Landmark stability check** — Implausible detection artifacts are discarded.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) — Face Mesh & Iris landmark detection
- [Soukupová & Čech (2016)](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) — Eye Aspect Ratio blink detection
- [Casiez, Roussel & Vogel (CHI 2012)](https://cristal.univ-lille.fr/~casiez/1euro/) — One Euro Filter
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — Cross-platform mouse automation
