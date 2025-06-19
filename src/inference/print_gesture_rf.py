"""
Real-Time Gesture Recognition Demo (Random Forest + MediaPipe)
=============================================================

This standalone script:

1. Loads a **pre-trained Random Forest** classifier stored with `joblib`.
2. Opens the default webcam (index 0), mirrored for a “selfie” view.
3. Detects a *single* hand each frame using **MediaPipe Hands**.
4. Flattens the 21 detected landmarks into a 42-dimensional vector
   `[x1, y1, x2, y2, … x21, y21]`.
5. Classifies the vector and prints the gesture *only* if
   • the prediction changed, **and**
   • at least 0.3 s have elapsed since the last print  
   (basic debouncing to reduce console spam).
6. Overlays the current prediction on the video stream.
7. Press **q** to quit.

Dependencies
------------
* OpenCV (`cv2`)
* NumPy (`numpy`)
* MediaPipe (`mediapipe`)
* scikit-learn (`joblib` for model serialisation)
"""

import cv2, numpy as np, mediapipe as mp, joblib, time

# ───────────────────────── Configuration ─────────────────────────
MODEL_PATH = "model/model_rf_14classes.pkl"    # ← Adjust if your model lives elsewhere.

class_names = [
    'up_fine','up_fast','down_fine','down_fast',
    'left_fine','left_fast','right_fine','right_fast',
    'forward_fine','forward_fast','backward_fine','backward_fast',
    'pickup','neutral'
]

# ───────────────────────── Model & MediaPipe init ────────────────
print("[INFO] loading RF model …")
model = joblib.load(MODEL_PATH)               # Random Forest classifier

mp_hands = mp.solutions.hands                 # Convenience alias
# Hands(static_image_mode, max_num_hands, model_complexity,
#       min_detection_confidence, min_tracking_confidence)
hands = mp_hands.Hands(False, 1, 1, 0.5, 0.5) # Real-time mode, one hand, medium complexity

# ───────────────────────── Video-capture loop ────────────────────
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "❌  No webcam found"

prev, stamp = None, 0                         # (prev prediction, timestamp of last emit)
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)                # Mirror for user-friendly view
    img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(img)

    if res.multi_hand_landmarks:
        # ─── Build feature vector (42 floats) ────────────────────
        vec = []
        for lm in res.multi_hand_landmarks[0].landmark:
            vec.extend([lm.x, lm.y])

        # ─── Predict gesture ─────────────────────────────────────
        pred = class_names[int(model.predict([vec])[0])]

        # ─── Debounce console output ─────────────────────────────
        if pred != prev and time.time() - stamp > 0.3:
            print(pred)
            prev, stamp = pred, time.time()

        # ─── On-screen label ─────────────────────────────────────
        cv2.putText(frame, pred, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RF demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):     # Press 'q' to quit
        break

# ───────────────────────── Clean-up ──────────────────────────────
cap.release()
cv2.destroyAllWindows()
