#!/usr/bin/env python3
# -----------------------------------------------------------
# ConvNeXt-Tiny live-webcam gesture demo
#   • Works with the checkpoint produced by the Colab notebook
#   • PyTorch ≥ 2.0, OpenCV ≥ 4.5 required
# -----------------------------------------------------------

import argparse, time, sys, pathlib
import torch, torchvision.transforms as T
import cv2
from torch import nn

# -------------------------- CLI ------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Real-time gesture classification with ConvNeXt-Tiny")
    p.add_argument("--weights", required=True,
                   help="Path to .pt checkpoint (state_dict or entire model)")
    p.add_argument("--labels",  default=None,
                   help="Optional text file with class names, one per line")
    p.add_argument("--imgsize", type=int, default=224,
                   help="Model input resolution (default 224)")
    p.add_argument("--cam",     type=int, default=0,
                   help="Which camera index OpenCV should open (default 0)")
    p.add_argument("--fps",     type=float, default=0.0,
                   help="Target FPS (0 = no throttle)")
    return p.parse_args()

# ---------------------- model + transforms -------------------------------
@torch.no_grad()
def load_model(ckpt_path: pathlib.Path, num_classes: int):
    """Return a ConvNeXt-Tiny model with weights loaded."""
    try:
        model = torchvision.models.convnext_tiny()          # torchvision >=0.15
    except Exception:
        import timm
        model = timm.create_model("convnext_tiny", pretrained=False)
    # replace classifier head
    if hasattr(model, "classifier"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        model.reset_classifier(num_classes)

    sd = torch.load(ckpt_path, map_location="cpu")
    # accept either full‐save or state_dict
    state_dict = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def build_transforms(img_size):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

# ----------------------------- main --------------------------------------
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load labels
    if args.labels:
        with open(args.labels) as f:
            class_names = [ln.strip() for ln in f if ln.strip()]
    else:
        # fallback to generic names until provided
        class_names = [f"class_{i}" for i in range(14)]

    model = load_model(pathlib.Path(args.weights), num_classes=len(class_names))
    model.to(device)
    tfms  = build_transforms(args.imgsize)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit(f"❌ Could not open camera index {args.cam}")

    print("▶️  Press ESC to quit")
    t_last, fps = time.time(), 0.0
    throttle = 1.0 / args.fps if args.fps > 0 else 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Frame grab failed, exiting."); break

        if throttle:                                    # rough FPS cap
            now = time.time()
            if (now - t_last) < throttle: continue
            t_last = now

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tfms(rgb).unsqueeze(0).to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type=="cuda"):
            logits = model(tensor)
        prob, pred = logits.softmax(1).max(dim=1)
        label      = class_names[pred.item()]
        fps        = 0.9*fps + 0.1*(1.0/(time.time()-t_last+1e-6))

        # overlay
        txt = f"{label}: {prob.item()*100:.1f}%  |  {fps:5.1f} FPS"
        cv2.putText(frame, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("ConvNeXt-Tiny gesture demo (ESC to quit)", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release(); cv2.destroyAllWindows()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
