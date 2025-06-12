# Hand-Tracking Robotic-Arm Controller

Real-time 14-gesture control for a robotic arm.  
The pipeline uses **MediaPipe Hands** to extract 21 hand landmarks, then classifies each frame with either a lightweight **Neural Network** (TensorFlow) or a **Random-Forest** (scikit-learn).

---

## What’s inside

| Folder | Purpose |
|--------|---------|
| `src/data_collection/record_gestures.py` | Capture and label webcam images (press **Space** to save). |
| `src/data_collection/make_npz.py` | Convert those images to `*.npz` landmark arrays for ML training. |
| `src/training/` | • `train_nn_14classes.py` – two-layer MLP (Keras/TensorFlow)<br>• `train_rf_14classes.py` – Random-Forest (scikit-learn). |
| `src/model/` | Ready-to-use weights:<br>  `model_nn_14classes.h5`, `model_rf_14classes.pkl`. |
| `src/inference/` | • `print_gesture_nn.py` – live demo (neural-net)<br>• `print_gesture_rf.py` – live demo (random-forest)<br>• `hand_gesture_reader.py` – maps gestures to keyboard / ROS commands. |
| `docs/` | Screenshots, diagrams, attribution notes. |
| `ignore/` | Bulky artefacts kept out of Git: raw `dataset/`, `gesture_npz.zip`, etc. |

---

## Installation

```bash
# clone and enter the repo
git clone https://github.com/<your-user>/HandTracking_RoboticArm.git
cd HandTracking_RoboticArm

# create an isolated environment
python -m venv venv && source venv/bin/activate

# install all dependencies
pip install -r requirements.txt
````

> Dependencies include `mediapipe`, `opencv-python`, `tensorflow`, `scikit-learn`, `joblib`, `numpy`, `matplotlib`.

---

## Quick demo

```bash
# Neural-Net recogniser (prints gesture label)
python src/inference/print_gesture_nn.py
```

or

```bash
# Random-Forest recogniser
python src/inference/print_gesture_rf.py
```

Show a gesture in front of the webcam; the predicted label appears in the terminal (and on the preview window). Press **Q** to quit.

---

## Data collection

1. **Record images**

   ```bash
   python src/data_collection/record_gestures.py
   ```

   *Press* **Space** to save a frame, **q** to switch to the next label.
   Images are stored under `ignore/dataset/<label>/`.

2. **Convert to NumPy arrays**

   ```bash
   python src/data_collection/make_npz.py
   ```

   This script detects the hand with MediaPipe, flattens the 21 landmarks into a 42-float vector, and saves `data_<label>.npz` (arrays `X` and `y`). All `.npz` files can be zipped into `gesture_npz.zip` for Colab training.

---

## Training

```bash
# (A) Two-layer neural net
python src/training/train_nn_14classes.py

# (B) Random-Forest
python src/training/train_rf_14classes.py
```

Both scripts read the `*.npz` files, split 80/20, report accuracy, and save the model into `src/model/`.

---

## Inference / Deployment

* `print_gesture_nn.py` / `print_gesture_rf.py` – stand-alone demos that just print the label.
* `hand_gesture_reader.py` – maps each gesture to a keyboard key (via `pyautogui`) or to a ROS topic (enable the flag in the script).

---

## Credits & Attribution

* **Landmark-MLP concept** and the original four-class demo code were adapted from
  Odilbek Tokhirov – *“How I Built a Hand Gesture Recognition Model in Python — Part 2”* (Medium, 2023).
  See `docs/medium_reference.md` for the excerpt and link.

* MediaPipe Hands – © Google (Apache-2.0).

<sub>Portions of the codebase were drafted with the help of OpenAI ChatGPT and subsequently refined and integrated by the author.</sub>