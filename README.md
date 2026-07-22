# Eye Gaze Estimation — Hands-Free Mouse, Keyboard & Game Control

Control your computer's mouse, type on an on-screen keyboard, play the Chrome dinosaur
game, and view real-time face/hand/pose tracking — all with a regular webcam and no
extra hardware. Built as a Human-Computer Interaction (HCI) exploration of using eye
gaze, blinks, winks, mouth movement, and head pose as input signals.

## What it does

| Script | What it lets you do |
|---|---|
| `main.py` | Tkinter launcher — one window with a button per mode below |
| `Eye_Direction.py` | Live overlay showing detected gaze direction (LEFT / CENTER / RIGHT) and blink state — a diagnostic viewer for the core gaze logic |
| `Virtual_Mouse.py` | Move the real system mouse cursor with head movement, click via winks, scroll via blink-and-hold |
| `Virtual_KeyBoard.py` | Type on an on-screen keyboard using gaze (to pick left/right half) and blinks (to select a letter) |
| `dinoGameWBlinkDetector.py` | Blink to jump — plays Chrome's offline dinosaur game hands-free |
| `Head_pose_and_Full_body.py` | Full-body + face mesh + both hands tracking overlay (MediaPipe Holistic) |

## How it works

1. **Face landmarks**: `dlib`'s 68-point facial landmark predictor locates the eyes,
   mouth, and nose on each webcam frame.
2. **Eye/mouth signals** ([utils.py](code/code/utils.py)):
   - **Eye Aspect Ratio (EAR)** — how open each eye is, used to detect blinks and winks.
   - **Mouth Aspect Ratio (MAR)** — how open the mouth is, used as an on/off toggle.
   - **Gaze ratio** — masks each eye, thresholds it to black/white, and compares the
     amount of white pixels on the left vs. right half to estimate gaze direction.
3. **Action mapping**: thresholds on these ratios drive `pyautogui` mouse/keyboard
   events (move, click, scroll, key press) or draw on-screen feedback via OpenCV.

<p align="center">
  <img src="facial%20landmark.png" alt="68-point facial landmarks" width="320">
  <img src="eye%20points.png" alt="Eye landmark points used for EAR/gaze" width="420">
</p>

## Getting started

### Prerequisites

- Python 3.9+
- A webcam
- Windows/macOS/Linux with a C++ build toolchain available (needed to build `dlib` —
  on Windows, install [CMake](https://cmake.org/download/) and the "Desktop
  development with C++" workload from Visual Studio Build Tools first; on
  macOS/Linux, `cmake` + a compiler via your package manager is usually enough)

### Installation

```bash
git clone https://github.com/bopanna012/Eye_Gaze_Estimation.git
cd Eye_Gaze_Estimation
pip install -r requirements.txt
```

The `dlib` 68-point facial landmark model (`shape_predictor_68_face_landmarks.dat`)
is already included in the repo, so no separate model download is required for the
`dlib`-based scripts. `Head_pose_and_Full_body.py` downloads its own MediaPipe
Holistic model bundle (~13MB) automatically the first time it runs.

### Running it

Launch the GUI, which lets you start any mode from one window:

```bash
python code/code/main.py
```

Or run any mode directly:

```bash
python code/code/Virtual_Mouse.py
```

Press **Esc** in the OpenCV window to quit any mode.

## Controls

**Virtual_Mouse.py**
| Action | Trigger |
|---|---|
| Enter/exit head-tracking mode | Open your mouth for ~5 frames |
| Move cursor | Tilt/move your head (tracked relative to the anchor point set on entry) |
| Left click | Wink left eye |
| Right click | Wink right eye |
| Toggle scroll mode | Close both eyes for ~5 frames |

**Virtual_KeyBoard.py**
| Action | Trigger |
|---|---|
| Select left/right keyboard half | Look left or right and hold |
| Type highlighted letter | Blink and hold while it's highlighted |
| Space | Select the `_` key |

**dinoGameWBlinkDetector.py**
| Action | Trigger |
|---|---|
| Jump | Blink |

## Project structure

```
Eye_Gaze_Estimation/
├── code/code/               # All Python source
│   ├── main.py               # Tkinter GUI launcher
│   ├── utils.py               # Shared EAR/MAR/gaze/blink math
│   ├── Eye_Direction.py
│   ├── Virtual_Mouse.py
│   ├── Virtual_KeyBoard.py
│   ├── dinoGameWBlinkDetector.py
│   └── Head_pose_and_Full_body.py
├── shape_predictor_68_face_landmarks.dat   # dlib 68-point face model
├── left.wav / right.wav / sound.wav        # Virtual keyboard feedback sounds
├── requirements.txt
└── LICENSE
```

## Tech stack

`OpenCV` · `dlib` · `MediaPipe Tasks` · `imutils` · `PyAutoGUI` · `pyglet` · `NumPy` · `Tkinter`

## Known limitations

- Single-user, single-face assumption — behavior with multiple faces in frame is undefined.
- Thresholds (EAR/MAR/gaze cutoffs) are hardcoded constants tuned by eye, not calibrated
  per-user or per-lighting-condition.
- Requires reasonably even, front-facing lighting for reliable landmark detection.

## License

[MIT](LICENSE) — originally authored by Sidharth Rai; this fork includes bug fixes
(MediaPipe API migration, webcam-failure handling, resource-path fixes, and
de-duplicated gaze/blink logic).
