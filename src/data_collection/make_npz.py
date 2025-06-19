"""
Dataset Pre-processing Script for Hand-Gesture Recognition
---------------------------------------------------------

This script scans a folder structure like:

    dataset/
        up_fine/
            img1.jpg
            img2.jpg
            ...
        up_fast/
        ...

For every image it:
1. Detects the first hand using MediaPipe.
2. Flattens the (x, y) coordinates of the 21 landmarks into a 42-element
   feature vector.
3. Stores the feature vectors (`X`) and their integer class labels (`y`)
   into a compressed NumPy archive (`.npz`)—one file per gesture class.

No images are modified on disk; only numeric data are exported to `data/`.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# Ordered list of gesture class names; index acts as the numeric label.
class_names = [
    'up_fine', 'up_fast', 'down_fine', 'down_fast',
    'left_fine', 'left_fast', 'right_fine', 'right_fast',
    'forward_fine', 'forward_fast', 'backward_fine', 'backward_fast',
    'pickup', 'neutral'
]

DATA_DIR = 'dataset'                        # Root directory containing class sub-folders.

# Initialise MediaPipe Hands in single-image mode.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,                 # Each frame is independent; no temporal smoothing.
    min_detection_confidence=0.5            # Trade-off between recall and false positives.
)

# ---------------------------------------------------------------------------
# Main loop: iterate over every gesture class and convert its images to .npz
# ---------------------------------------------------------------------------
for idx, cname in enumerate(class_names):
    X, y = [], []                           # Feature matrix and label vector for this class.

    folder = os.path.join(DATA_DIR, cname)  # Folder containing images of gesture `cname`.

    for fname in os.listdir(folder):
        # Skip non-image files.
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)          # Read image in BGR format.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)    # Run hand landmark detection.

        # If no hand is detected, ignore this image.
        if not results.multi_hand_landmarks:
            continue

        # -------------------------------------------------------------------
        # Convert the first detected hand’s 21 landmarks into a flat vector:
        # [x1, y1, x2, y2, ..., x21, y21]
        # -------------------------------------------------------------------
        lm_vec = []
        for lm in results.multi_hand_landmarks[0].landmark:
            lm_vec.extend([lm.x, lm.y])

        X.append(lm_vec)                    # Store feature vector.
        y.append(idx)                       # Store numeric label for this class.

    # -----------------------------------------------------------------------
    # Persist the collected samples for this class to a compressed .npz file
    # -----------------------------------------------------------------------
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    out_path = os.path.join('data', f'data_{cname}.npz')
    os.makedirs('data', exist_ok=True)      # Create output directory if needed.
    np.savez_compressed(out_path, X=X, y=y)

    print(f"Saved {X.shape[0]} samples to {out_path}")
