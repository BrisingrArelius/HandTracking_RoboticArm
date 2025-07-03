#!/usr/bin/env python3
"""
Unified benchmark 2.1 – per-gesture .npz folder + mixed image models
-------------------------------------------------------------------
Usage examples
--------------

# Landmarks only
python src/benchmark.py \
       --models "./model:./model_new" \
       --landmark_folder "src/data_collection/data"

# Landmarks + CNNs (absolute path to avoid cwd issues)
python src/benchmark.py \
       --models "./model:./model_new" \
       --landmark_folder "src/data_collection/data" \
       --images "$PWD/ignore/mini_val" \
       --batch 256
"""
import argparse, pathlib, time, csv, sys, warnings
from collections import namedtuple
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

# ───────────────────── add project dirs to PYTHONPATH ──────────────────────
project_root = pathlib.Path(__file__).resolve().parent
sys.path.append(str(project_root))                     # e.g. src/
sys.path.append(str(project_root / "training"))        # where mobilevit_v2.py lives
# add more as needed: sys.path.append(str(project_root / "some/other/dir"))

# ───────────────────────── CLI ──────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--models', type=str, default='./model',
               help='colon-separated list of directories that hold model files')
p.add_argument('--landmark_folder', type=pathlib.Path, required=True,
               help='folder full of data_<gesture>.npz files')
p.add_argument('--images', type=pathlib.Path, default=None,
               help='root folder with sub-dir per class for image models')
p.add_argument('--batch', type=int, default=256, help='batch size for latency test')
args = p.parse_args()
model_dirs = [pathlib.Path(d) for d in args.models.split(':')]

# ─────────────────────── loaders ────────────────────────
def load_landmark_folder(folder: pathlib.Path):
    files = sorted(folder.glob('data_*.npz'))
    if not files:
        raise FileNotFoundError(f'No data_*.npz in {folder}')
    Xs, ys = [], []
    for idx, f in enumerate(files):
        arr = np.load(f)
        Xs.append(arr["X"].astype("float32"))
        ys.append(arr["y"].astype("int64") if "y" in arr
                  else np.full(len(arr["X"]), idx, dtype="int64"))
    return np.vstack(Xs), np.concatenate(ys)

def load_image_test(img_root):
    import torch
    from torchvision import datasets, transforms
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(img_root, transform=trans)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                     shuffle=False, num_workers=2, pin_memory=True)
    return dl, ds.classes

# ───── simple PyTorch factory – extend as you add backbones ─────
import re, timm

def infer_arch_from_name(fname: str):
    fname = fname.lower()

    # ---- ConvNeXt tiny (unchanged) ----
    if 'convext' in fname or 'convnext' in fname:
        from torchvision.models import convnext_tiny
        return convnext_tiny(num_classes=14)

    # ---- MobileViT v2 via timm ----
    if 'mobilevit' in fname or 'best_model' in fname:
        # pick width from filename, e.g. mobilevitv2_050 → 0.5, _150 → 1.5
        m = re.search(r'v2_(\d+)', fname)
        width = m.group(1) if m else '050'
        model_name = f'mobilevitv2_{width}'
        return timm.create_model(model_name, num_classes=14, pretrained=False)

    # ---- MobileNet V3 small (unchanged) ----
    if 'mobilenet' in fname:
        from torchvision.models import mobilenet_v3_large
        return mobilenet_v3_large(num_classes=14)

    raise ValueError(f'Add factory branch for {fname}')


def load_model(path: pathlib.Path):
    """Return ('kind', model) where kind ∈ {'sk','tf','torch'}."""
    ext = path.suffix.lower()

    # ───────── sklearn pickle ─────────
    if ext == '.pkl':
        return 'sk', joblib.load(path)

    # ───────── Keras / TF ─────────
    if ext == '.h5':
        import tensorflow as tf
        return 'tf', tf.keras.models.load_model(path, compile=False)

    # ───────── PyTorch ─────────
    if ext in {'.pth', '.pt'}:
        import torch, collections

        raw = torch.load(path, map_location='cpu')

        # Case C: file **is** a state-dict
        if isinstance(raw, (collections.OrderedDict, dict)):
            sd = raw
            try:
                net = infer_arch_from_name(path.name)
            except ValueError as e:
                raise RuntimeError(f"state_dict but unknown arch hint → {e}")

            try:                                 # first try strict
                net.load_state_dict(sd, strict=True)
            except RuntimeError:
                net.load_state_dict(sd, strict=False)  # Case B
            net.eval()
            return 'torch', net

        # Otherwise raw is (hopefully) the full model object
        if hasattr(raw, 'eval'):
            raw.eval()
            return 'torch', raw

        raise RuntimeError("Checkpoint is neither model nor state_dict")

    raise ValueError(f'Unknown extension {path}')

# ───────────────────────── main ─────────────────────────
print('📂  Loading landmark test-set …')
land_X, land_y = load_landmark_folder(args.landmark_folder)
print(f'    → {land_X.shape[0]:,} samples  /  {land_X.shape[1]} features')

if args.images:
    print('🖼️   Preparing image test-loader …')
    img_dl, img_classes = load_image_test(args.images)
    print(f'    → {len(img_dl.dataset):,} images  /  {len(img_classes)} classes')
else:
    img_dl = None

Result = namedtuple('Result', 'name acc f1 prec recall latency_ms')
results = []

for model_dir in model_dirs:
    for path in sorted(model_dir.glob('*')):
        try:
            kind, mdl = load_model(path)
        except Exception as e:
            print(f'⚠️  {path.name} skipped → {e}')
            continue

        print(f'\n▶ Evaluating {path.name}  ({kind})')
        if kind in {'sk', 'tf'}:                 # landmark models
            t0 = time.perf_counter()
            y_pred = (mdl.predict(land_X) if kind == 'sk'
                      else mdl.predict(land_X, verbose=0).argmax(axis=1))
            latency = (time.perf_counter() - t0) / len(land_X) * 1000
            y_true = land_y

        elif kind == 'torch' and img_dl:         # image models
            import torch, torch.nn.functional as F
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mdl.to(device)
            preds, gts = [], []
            t0 = time.perf_counter()
            with torch.no_grad():
                for imgs, labels in tqdm(img_dl, leave=False):
                    preds.append(mdl(imgs.to(device)).argmax(1).cpu())
                    gts.append(labels)
            latency = (time.perf_counter() - t0) / len(img_dl.dataset) * 1000
            y_pred = torch.cat(preds).numpy()
            y_true = torch.cat(gts).numpy()

        else:
            print(f'⚠️  {path.name} skipped → no suitable dataset (add --images)')
            continue

        res = Result(
            name       = path.name,
            acc        = accuracy_score (y_true, y_pred),
            f1         = f1_score        (y_true, y_pred, average='macro'),
            prec       = precision_score (y_true, y_pred, average='macro'),
            recall     = recall_score    (y_true, y_pred, average='macro'),
            latency_ms = round(latency, 3)
        )
        results.append(res)
        print(f'   acc {res.acc:.3%}  f1 {res.f1:.3f}  {res.latency_ms:.2f} ms/img')

# ───────────────────────── save CSV ─────────────────────────
csv_path = 'benchmark_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(Result._fields)
    writer.writerows(results)

print(f'\n✅ All done – results saved to {csv_path}')
