"""
Simple Real-Time Hand-Landmark Visualizer (MediaPipe)
====================================================

* Opens the default webcam (index 0).
* Detects a single hand per frame using **MediaPipe Hands** and overlays both
  the 21 landmarks and their canonical connections.
* Displays a mirrored view so the on-screen motion matches the user’s
  perspective.
* Press **ESC** to quit.

Requirements
------------
`opencv-python`, `mediapipe`
"""

import cv2
import mediapipe as mp

# ─────────────────────────── MediaPipe setup ──────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands(min_detection_confidence, min_tracking_confidence)
hands = mp_hands.Hands(
    min_detection_confidence=0.7,    # ↑ Higher → fewer false positives
    min_tracking_confidence=0.5      # ↑ Higher → smoother landmark tracking
)

# ─────────────────────────── Webcam initialisation ────────────────────────
cap = cv2.VideoCapture(0)

# ─────────────────────────── Main loop ────────────────────────────────────
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)                       # Mirror for a “selfie” view
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run hand-landmark detection.
    results = hands.process(image_rgb)

    # Draw landmarks and the canonical hand skeleton if a hand is detected.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Show the annotated frame.
    cv2.imshow('MediaPipe Hands', image)

    # Exit when ESC (ASCII 27) is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

# ─────────────────────────── Clean-up ─────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
