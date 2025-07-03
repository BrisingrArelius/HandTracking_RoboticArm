"""
Live Gesture-Recognition Demo Using MobileNet V3
===============================================

Run from project root:

    python -m src.inference.live_webcam_mnet \
           --weights mobilenet_augmented.pth \
           --imgsize 224 --cam 0
"""

import argparse, json, time, cv2, torch
from pathlib import Path
from torchvision import transforms, models
from PIL import Image

# ─────────────────────── CLI ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Real-time MobileNetV3 gesture recogniser from webcam feed")
parser.add_argument("--weights", default="mobilenet_augmented.pth",
                    help="File inside HANDTRACKING_ROBOTICARM/src/model/")
parser.add_argument("--imgsize", type=int, default=224,
                    help="Square side length fed into the network.")
parser.add_argument("--cam", type=int, default=0,
                    help="Index of the webcam to open (0 = default).")
args = parser.parse_args()

# ─────────────────────── Paths ─────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]          # …/HANDTRACKING_ROBOTICARM
MODEL_DIR  = ROOT / "src/model"
WEIGHTS    = MODEL_DIR / args.weights
LABELS     = MODEL_DIR / "class_to_idx.json"
assert WEIGHTS.exists(), f"❌ Weights not found: {WEIGHTS}"
assert LABELS.exists(),  f"❌ class_to_idx.json missing in {MODEL_DIR}"

# ─────────────────────── Model (CHANGED) ───────────────────────────────────
NUM_CLASSES = 14
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v3_large(weights=None, num_classes=NUM_CLASSES)
state = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state if isinstance(state, dict) else state)
model.eval().to(device)
print("✅ MobileNetV3 loaded on", device)

idx2class = {v: k for k, v in json.load(open(LABELS)).items()}

# ─────────────────────── Pre-processing (CHANGED) ─────────────────────────
tf = transforms.Compose([
    transforms.Resize((args.imgsize, args.imgsize)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────── Webcam loop params (unchanged) ───────────────────
fps_target      = 30.0
frame_interval  = 1.0 / fps_target
last_frame_t    = 0.0

cap = cv2.VideoCapture(args.cam)
assert cap.isOpened(), f"Cannot open webcam {args.cam}"
fps_time = time.time()

# ─────────────────────── Main live-inference loop ─────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed. Exiting.")
            break

        now = time.time()
        if now - last_frame_t < frame_interval:
            time.sleep(frame_interval - (now - last_frame_t))
            continue
        last_frame_t = now

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor).argmax(1).item()
        label = idx2class[pred]

        fps    = 1.0 / (time.time() - fps_time)
        fps_time = time.time()

        cv2.putText(frame, f"{label}  {fps:0.1f} FPS",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gesture Demo (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
