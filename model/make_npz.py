import os
import cv2
import numpy as np
import mediapipe as mp

class_names = [
    'up_fine','up_fast','down_fine','down_fast',
    'left_fine','left_fast','right_fine','right_fast',
    'forward_fine','forward_fast','backward_fine','backward_fast',
    'pickup','neutral'
]

DATA_DIR = 'dataset'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.5
)
  
for idx, cname in enumerate(class_names):
    X, y = [], []
    folder = os.path.join(DATA_DIR, cname)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue

        lm_vec = []
        for lm in results.multi_hand_landmarks[0].landmark:
            lm_vec.extend([lm.x, lm.y])
        
        X.append(lm_vec)
        y.append(idx)

    X = np.array(X, dtype=np.float32)  
    y = np.array(y, dtype=np.int32) 

    out_path = os.path.join('data', f'data_{cname}.npz')
    os.makedirs('data', exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved {X.shape[0]} samples to {out_path}")
