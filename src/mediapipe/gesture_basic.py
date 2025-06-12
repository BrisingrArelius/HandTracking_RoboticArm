import cv2, time, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'gesture_recognizer_new.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,   # ◀️ NEW
)

recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Gesture recognition started. Press 'q' to quit.")
frame_ts = 0  # milliseconds

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # VIDEO mode call
    result = recognizer.recognize_for_video(mp_img, frame_ts)
    frame_ts += int(1000 / cap.get(cv2.CAP_PROP_FPS) or 33)  # ~33 ms per frame

    if result.gestures:
        gesture = result.gestures[0][0].category_name
        score   = result.gestures[0][0].score
        cv2.putText(frame, f"{gesture} ({score:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
