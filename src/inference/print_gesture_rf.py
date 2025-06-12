import cv2, numpy as np, mediapipe as mp, joblib, time

MODEL_PATH = "model/model_rf_14classes.pkl"  # adjust

class_names = [
    'up_fine','up_fast','down_fine','down_fast',
    'left_fine','left_fast','right_fine','right_fast',
    'forward_fine','forward_fast','backward_fine','backward_fast',
    'pickup','neutral'
]

print("[INFO] loading RF model …")
model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 1, 1, 0.5, 0.5)

cap = cv2.VideoCapture(0)
assert cap.isOpened()

prev, stamp = None, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img)

    if res.multi_hand_landmarks:
        vec = []
        for lm in res.multi_hand_landmarks[0].landmark:
            vec.extend([lm.x, lm.y])
        pred = class_names[int(model.predict([vec])[0])]
        if pred != prev and time.time()-stamp > .3:
            print(pred)
            prev, stamp = pred, time.time()
        cv2.putText(frame, pred, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("RF demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
