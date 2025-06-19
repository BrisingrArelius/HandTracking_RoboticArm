"""
Real-Time Hand-Gesture Classification Demo (Neural-Network Version)
===================================================================

*   Loads a **Keras** model trained on 14 gesture classes.
*   Opens the default webcam (index 0) and mirrors the feed for a natural
    “selfie” view.
*   Uses **MediaPipe Hands** to extract 21 hand-landmark coordinates and
    flattens them into a 42-element feature vector `[x0, y0, …, x20, y20]`.
*   Runs the vector through the neural network to obtain class probabilities
    and prints / overlays the highest-probability gesture.
*   Basic debouncing: a new prediction is printed only if it differs from the
    previous one **or** `cooldown` seconds have elapsed.
*   Exit cleanly by pressing **Q** in the video window.

Dependencies
------------
`opencv-python`, `mediapipe`, `tensorflow`, `numpy`
"""

# ───────────────────────── Imports ────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# ───────────────────────── Configuration ──────────────────────
MODEL_PATH = "model/model_nn_14classes.h5"   # ← update if you relocate the model

class_names = [
    'move_up_fine', 'move_up_fast',
    'move_down_fine', 'move_down_fast',
    'move_left_fine', 'move_left_fast',
    'move_right_fine', 'move_right_fast',
    'move_forward_fine', 'move_forward_fast',
    'move_backward_fine', 'move_backward_fast',
    'pickup', 'drop'
]

# ───────────────────────── Model Loading ──────────────────────
print("[INFO] loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
model.make_predict_function()          # Build graph *once* to avoid first-call latency.

# ───────────────────────── MediaPipe Hands ────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,           # Stream/real-time mode
    max_num_hands=1,                   # We care about one hand at a time
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ───────────────────────── Webcam Setup ───────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not available")

# ───────────────────────── Debounce Vars ──────────────────────
last_gesture = None
cooldown     = 0.3                     # Seconds before printing the *same* gesture again
last_time    = 0.0

print("Press  Q  in the video window to quit.\n")

# ───────────────────────── Main Loop ──────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)                           # Mirror for user convenience
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # ─── Gesture inference only if a hand is found ────────────
    if results.multi_hand_landmarks:
        # Build 42-D feature vector.
        lm_vec = []
        for lm in results.multi_hand_landmarks[0].landmark:
            lm_vec.extend([lm.x, lm.y])
        x_arr = np.array(lm_vec, dtype=np.float32)[None, :]   # Shape (1, 42)

        # Predict gesture probabilities; take argmax.
        probs   = model.predict(x_arr, verbose=0)[0]
        gesture = class_names[int(np.argmax(probs))]

        # ─── Debounce console output ──────────────────────────
        now = time.time()
        if gesture != last_gesture or (now - last_time) > cooldown:
            print(gesture)
            last_gesture = gesture
            last_time    = now

        # ─── Optional on-screen overlay ──────────────────────
        cv2.putText(frame, gesture, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ─── Display preview & check for quit key ────────────────
    cv2.imshow("Gesture-only demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ───────────────────────── Clean-up ───────────────────────────
cap.release()
cv2.destroyAllWindows()
