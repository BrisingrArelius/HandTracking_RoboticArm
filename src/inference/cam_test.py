# cam_test.py  –  run:  python cam_test.py --cam 0
import cv2, argparse, time, os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # stay on XWayland

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
args = ap.parse_args()

cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)   # CAP_V4L2 = Linux backend
print("cap.isOpened() ?", cap.isOpened())

for i in range(100):          # ~3-4 s
    ok, frame = cap.read()
    if ok:
        print("min/max/mean:", frame.min(), frame.max(), frame.mean())
    print(i, ok, frame is None, frame.shape if ok else None)
    if ok:
        cv2.imshow("Raw feed – ESC to quit", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
