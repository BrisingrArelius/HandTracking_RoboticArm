# python - <<'PY'
from torchvision import datasets, transforms
root = 'ignore/mini_val'
try:
    ds = datasets.ImageFolder(root, transform=transforms.ToTensor())
    print(f"Found {len(ds)} images, {len(ds.classes)} classes:", ds.classes)
except Exception as e:
    print("ImageFolder error:", e)
