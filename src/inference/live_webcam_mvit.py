"""
Live Gesture-Recognition Demo Using MobileViTv2
===============================================

Place **this** file at:

    HANDTRACKING_ROBOTICARM/src/inference/live_webcam_mvit.py

and run from the project root (or with `-m`) like:

    python -m src.inference.live_webcam_mvit \
           --weights best_model.pth          \
           --imgsize 224                    \
           --cam 0

Purpose
-------
* Opens a webcam feed, throttled to an *approximate* frame-rate you set
  (`fps_target` below, default ≈ 5 FPS).
* Every processed frame is:
    1. Resized / gray-scaled / normalised.
    2. Passed through a MobileViTv2 classifier loaded from `--weights`.
    3. Labelled with the predicted gesture name and the realised FPS.
* Exits cleanly when **ESC** is pressed.

Requirements
------------
* OpenCV (`cv2`) for camera I/O and visualisation.
* PyTorch + `timm` for the MobileViTv2 model.
* Pillow for quick RGB→`torch.Tensor` conversion.
* A `class_to_idx.json` saved during training that maps string labels to
  numeric indices.

Notes
-----
* **No model weights are downloaded automatically**; you *must* place the
  `.pth` file in `src/model/`.
* The script leaves the original BGR frame untouched except for a simple
  text overlay, so colour information is preserved if you change the model
  preprocessing later.
"""

import argparse, json, time, cv2, torch, timm
from pathlib import Path
from torchvision import transforms
from PIL import Image

# ───────────────────────── Command-line arguments ──────────────────────────
parser = argparse.ArgumentParser(
    description="Real-time MobileViTv2 gesture recogniser from webcam feed")
parser.add_argument(
    "--weights",
    default="10_epoch_mobilevitv2_050_gestures.pth",
    help="File name inside HANDTRACKING_ROBOTICARM/src/model/ containing the "
         "trained weights."
)
parser.add_argument("--imgsize", type=int, default=224,
                    help="Square side length fed into the network.")
parser.add_argument("--cam", type=int, default=0,
                    help="Index of the webcam to open (0 = default).")
args = parser.parse_args()

# ───────────────────────── Project paths ───────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]         # …/HANDTRACKING_ROBOTICARM
MODEL_DIR  = ROOT / "src/model"
WEIGHTS    = MODEL_DIR / args.weights
LABELS     = MODEL_DIR / "class_to_idx.json"

assert WEIGHTS.exists(), f"❌  Weights not found: {WEIGHTS}"
assert LABELS.exists(),  f"❌  class_to_idx.json is missing in {MODEL_DIR}"

# ───────────────────────── Model initialisation ────────────────────────────
NUM_CLASSES = 14
ARCH        = "mobilevitv2_050.cvnets_in1k"              # Change if you trained a
                                                        # different width/depth.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an *untrained* backbone and then load *your* weights.
model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval().to(device)                                 # Switch to inference mode.

# Reverse mapping: int → string label.
idx2class = {v: k for k, v in json.load(open(LABELS)).items()}

# ───────────────────────── Input preprocessing ─────────────────────────────
# The model was trained on 224 × 224, 3-channel, mean-0.5 / std-0.5 data.
tf = transforms.Compose([
    transforms.Resize((args.imgsize, args.imgsize)),
    transforms.Grayscale(num_output_channels=3),        # Ensure 3 channels.
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# ───────────────────────── Webcam loop parameters ──────────────────────────
fps_target      = 5.0                                   # Desired *processed* FPS.
frame_interval  = 1.0 / fps_target                      # Seconds to wait per frame.
last_frame_t    = 0.0                                   # Time when last frame was processed.

cap = cv2.VideoCapture(args.cam)
assert cap.isOpened(), f"Cannot open webcam {args.cam}"
fps_time = time.time()                                  # Used for instantaneous FPS calc.

# ───────────────────────── Main live-inference loop ───────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame grab failed. Exiting.")
            break

        now = time.time()

        # ─── Throttle to fps_target ────────────────────────────────────────
        if now - last_frame_t < frame_interval:
            time.sleep(frame_interval - (now - last_frame_t))
            continue                                    # Skip processing; grab new frame.
        last_frame_t = now

        # ─── Pre-process the frame exactly as during training ─────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        tensor = tf(pil).unsqueeze(0).to(device)        # Shape (1, 3, H, W).

        # ─── Forward pass ────────────────────────────────────────────────
        with torch.no_grad():
            pred = model(tensor).argmax(1).item()
        label = idx2class[pred]

        # ─── Overlay prediction & FPS on original frame ──────────────────
        now_fps = time.time()
        fps     = 1.0 / (now_fps - fps_time)
        fps_time = now_fps

        cv2.putText(frame, f"{label}  {fps:0.1f} FPS",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gesture Demo (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:                 # ESC key
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
