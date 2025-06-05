import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the gesture model
model_path = 'gesture_recognizer.task'

# Create base options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)

# Create GestureRecognizer in video mode
recognizer = vision.GestureRecognizer.create_from_options(options)

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Gesture recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run recognition
    result = recognizer.recognize(mp_image)

    # Draw result
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        score = result.gestures[0][0].score
        print(f"Gesture: {gesture}, Score: {score:.2f}")
        cv2.putText(frame, f"{gesture} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
