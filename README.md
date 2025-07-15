# Hand-Tracking Robotic-Arm Controller

*Evaluating multiple gesture-recognition models to find the sweet spot between accuracy and latency.*

> **Project goal** – Compare several architecture families (MediaPipe-landmark + MLP, MobileNet, MobileViT-v2, ConvNeXt-tiny, Random-Forest, etc.) on the same 14-gesture set and decide which best suits a real-time robotic-arm controller. Training ran on a GPU; all demos run **CPU-only** on standard laptops.

---

## Recognition Methods

This repository provides **two primary approaches** for hand gesture recognition:

### 1. MediaPipe-based Methods (Landmark Extraction)

* **a. Complete MediaPipe Pipeline:**
  Use MediaPipe’s built-in gesture recognizer to train a `.task` file for the custom gesture set (see `src/mediapipe/`).
* **b. Machine Learning with Random Forest:**
  Extract 21 hand landmarks using MediaPipe, flatten to a 42-dimension vector, and classify gestures with a Random Forest model.
* **c. Machine Learning with Neural Network:**
  Extract 21 landmarks with MediaPipe and classify gestures with a lightweight Neural Network (Keras/TensorFlow).

### 2. Image Classification Methods

* Use raw webcam images as input and train image classification models.
  Multiple architectures are included (e.g., MobileViT, MobileNet, ConvNeXt-Tiny), classifying gestures directly from images, without landmark extraction.

---

## 📂 Folder structure — top level

```text
.
├── docs/                     ← screenshots, diagrams, references
├── ignore/                   ← large artefacts excluded via .gitignore
├── src/                      ← codebase (see next tree)
├── LICENSE
├── README.md
└── requirements.txt
```

```text
src/
├── benchmark.py              ← script to aggregate metrics into CSV
├── benchmark_results.csv     ← ✨ numbers below
├── data_collection/          ← capture webcam images + convert to .npz
├── inference/                ← live webcam + offline testers
├── mediapipe/                ← full MediaPipe pipeline (.task export)
├── model/                    ← current weights & pickles
├── model_new/ model_old/     ← archived checkpoints (ignored in git)
└── training/                 ← Jupyter notebooks for each model type
```

> **`ignore/`** stores raw images (`dataset/`) and heavyweight checkpoints that don’t belong in the repo.

* `data_collection/` contains scripts to collect and organize data for different model types.
* `mediapipe/` implements the complete MediaPipe pipeline, including landmark extraction and exporting a `.task` file with the current gesture set.
* `model/` contains trained model files for all current approaches (MediaPipe-based and image classification).

---

## 🔧 Installation

```bash
git clone https://github.com/BrisingrArelius/HandTracking_RoboticArm.git
cd HandTracking_RoboticArm
python -m venv venv && source venv/bin/activate   # Python 3.10.2
pip install -r requirements.txt
```

*Dependencies*: `mediapipe-python`, `opencv-python`, `tensorflow`, `torch`, `scikit-learn`, `joblib`, `numpy`, `matplotlib`.

GPU (CUDA 11+) is required **only** to retrain models; CPU inference achieves 30 FPS + on a mid-range laptop.

---

## 🚀 Quick demo

```bash
# Live webcam demo – Neural-Net
python src/inference/print_gesture_nn.py

# Live webcam demo – Random-Forest
python src/inference/print_gesture_rf.py
```

Move your hand in front of the webcam; the predicted label appears in the terminal and overlay window. Press **Q** to quit.

*(Placeholder: add demo GIF at **`docs/media/demo.gif`**)*

---

## 📸 Data collection

| Step                    | Script                                   | Description                                                                                                      |
| ----------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **1. Record images**    | `src/data_collection/record_gestures.py` | Capture labelled frames (press **Space** to save, **Q** to switch label). Saved under `ignore/dataset/<label>/`. |
| **2. Convert to NumPy** | `src/data_collection/make_npz.py`        | Detect 21 landmarks with MediaPipe Hands → flatten to 42-float vector → save `data_<label>.npz` (`X`, `y`).      |

---

## 🏋️ Training (current checkpoints)

```bash
# Two-layer MLP on landmarks
python src/training/train_nn.ipynb       # or open in Jupyter

# Random-Forest baseline
python src/training/train_random_forest.ipynb
```

> **Important** – Models were trained on *different* snapshots of the dataset as it grew. Metrics below are *not* apples-to-apples; unifying the dataset is first on the roadmap.

---

## 📊 Benchmark results

| Model file                              | Accuracy  | F1        | Precision | Recall    | Latency (ms) |
| --------------------------------------- | --------- | --------- | --------- | --------- | ------------ |
| `model_nn.h5`                           | **0.983** | **0.982** | 0.983     | **0.981** | **0.065**    |
| `mobilenet_augmented.pth`               | 0.893     | 0.880     | **0.923** | 0.893     | 44.6         |
| `model_nn_14classes.h5`                 | 0.901     | 0.886     | 0.909     | 0.887     | 0.057        |
| `best_model.pth`                        | 0.729     | 0.695     | 0.778     | 0.729     | 83.0         |
| `model_rf_14classes.pkl`                | 0.706     | 0.689     | 0.744     | 0.697     | **0.017**    |
| `10_epoch_mobilevitv2_050_gestures.pth` | 0.114     | 0.034     | 0.021     | 0.114     | 63.7         |
| `convext_tiny.pt`                       | 0.071     | 0.009     | 0.005     | 0.071     | 144.5        |
| `model_nn_14classes_small.h5`           | 0.467     | 0.433     | 0.579     | 0.456     | **0.056**    |

*(CSV generated by **`src/benchmark.py`**; see **`src/benchmark_results.csv`** for full metrics.)*

---

## 🔎 Inference helpers

| Script                                                   | Purpose                                   |
| -------------------------------------------------------- | ----------------------------------------- |
| `print_gesture_nn.py`                                    | Webcam demo with Neural-Net (.h5)         |
| `print_gesture_rf.py`                                    | Webcam demo with Random-Forest (.pkl)     |
| `cam_test.py`                                            | FPS + latency profiler                    |
| `convext_tiny.py`, `live_webcam_mvit.py`, `mobilenet.py` | Alternative backbones for experimentation |

---

## 🛣️ Future work

* **Unified dataset v2** – retrain all models on the *same* balanced split for fair comparison.
* **Gazebo simulation** – map predictions to a 3-D articulated hand in Gazebo for closed-loop testing before hardware deployment.
* Extend label set (currently 14) to include continuous gestures (e.g. pinch, rotate).
* Package a lightweight `pip install hand-controller` CLI for end-users.

---

## 📝 Credits & Attribution

* MediaPipe Hands – © Google (Apache-2.0).
* Landmark-MLP concept adapted from Odilbek Tokhirov, “How I Built a Hand Gesture Recognition Model in Python — Part 2” (Medium, 2023). See `docs/medium_reference.md`.
* Portions of the codebase were drafted with OpenAI ChatGPT and refined by the author.

MIT License – see `LICENSE` for details.

---

*(Placeholders: **`docs/media/gestures_grid.png`**, **`docs/media/demo.gif`** – drop files and update image links above.)*
