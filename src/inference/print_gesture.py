
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# ---------- CONFIG ----------
MODEL_PATH = "model_nn_14classes.h5" 
class_names = [
    'move_up_fine', 'move_up_fast',
    'move_down_fine', 'move_down_fast',
    'move_left_fine', 'move_left_fast',
    'move_right_fine', 'move_right_fast',
    'move_forward_fine', 'move_forward_fast',
    'move_backward_fine', 'move_backward_fast',
    'pickup', 'drop'
]
# ----------------------------

print("[INFO] loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
model.make_predict_function()  # warm-up

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not available")

last_gesture = None
cooldown = 0.3          # seconds to ignore duplicate prints
last_time  = 0

print("Press  Q  in the video window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # Build a 42-length vector: [x0, y0, x1, y1, …]
        lm_vec = []
        for lm in results.multi_hand_landmarks[0].landmark:
            lm_vec.extend([lm.x, lm.y])
        x_arr = np.array(lm_vec, dtype=np.float32)[None, :]   # shape (1, 42)

        probs   = model.predict(x_arr, verbose=0)[0]
        gesture = class_names[int(np.argmax(probs))]

        now = time.time()
        if gesture != last_gesture or (now - last_time) > cooldown:
            print(gesture)
            last_gesture = gesture
            last_time    = now

        # optional: overlay on video
        cv2.putText(frame, gesture, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture-only demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
