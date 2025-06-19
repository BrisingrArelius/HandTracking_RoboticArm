"""
Real-Time Hand-Gesture Recognition Demo (MediaPipe Tasks API)
=============================================================

How it works
------------
1. Loads a MediaPipe *GestureRecognizer* model from `gesture_recognizer_new.task`.
2. Opens the default webcam (device *0*) and flips each frame horizontally so
   the on-screen motion matches the user’s perspective (a “mirror” view).
3. Converts each BGR frame (OpenCV default) to RGB and wraps it in a
   `mediapipe.tasks.python.vision.Image`.
4. Runs the recognizer in **VIDEO** mode, which expects a monotonically
   increasing millisecond timestamp (`frame_ts`) for temporal consistency.
5. If a gesture is detected, the top category’s name and confidence score are
   drawn on the frame.
6. Press **q** in the display window to quit.

Dependencies
------------
* `opencv-python` (cv2)
* `mediapipe`
* The MediaPipe *Tasks* runtime (`mediapipe.tasks.python`)
"""

import cv2, time, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ───────────────────────── Model & recognizer setup ────────────────────────
model_path = 'gesture_recognizer_new.task'        # Pre-trained .task model

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,        # Use VIDEO mode for time-aware inference
)

recognizer = vision.GestureRecognizer.create_from_options(options)

# ───────────────────────── Webcam initialisation ───────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Gesture recognition started. Press 'q' to quit.")
frame_ts = 0  # Millisecond timestamp fed into recognizer; increments every frame

# ───────────────────────── Main capture / inference loop ───────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)                    # Mirror for user-friendly view
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the image for MediaPipe Tasks (expects RGB numpy array).
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # In VIDEO mode we must pass both the image and its timestamp (ms).
    result = recognizer.recognize_for_video(mp_img, frame_ts)

    # Advance timestamp by ~one frame period (uses FPS if available, else 33 ms).
    frame_ts += int(1000 / cap.get(cv2.CAP_PROP_FPS) or 33)

    # ─── Draw the top gesture (if any) ─────────────────────────────────────
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        score   = result.gestures[0][0].score
        cv2.putText(frame, f"{gesture} ({score:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ─── Display the annotated frame ───────────────────────────────────────
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):         # Quit on 'q'
        break

# ───────────────────────── Clean-up ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
