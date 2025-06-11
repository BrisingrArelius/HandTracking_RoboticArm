import cv2, pathlib, time

labels = [
    "up_fine","up_fast","down_fine","down_fast",
    "left_fine","left_fast","right_fine","right_fast",
    "forward_fine","forward_fast","backward_fine","backward_fast",
    "pickup","neutral"
]

# make sure the root dir itself exists
root = pathlib.Path(__file__).resolve().parent / "dataset"
root.mkdir(exist_ok=True)

cam = cv2.VideoCapture(0)
assert cam.isOpened(), "No webcam found"

for label in labels:
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)        # create <dataset>/<label>/
    counter = len(list(folder.glob("*.jpg")))

    print(f"\n--- {label}:  SPACE = save   q = next label ---")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        cv2.putText(frame, f"{label}  {counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("rec", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):                    # save current frame
            path = folder / f"{label}_{counter:04d}.jpg"
            success = cv2.imwrite(str(path), frame)
            if success:
                counter += 1
                print("saved ->", path.name)
            else:
                print("⚠️  failed to save", path)
        elif k == ord('q'):                  # move to next label
            break
    time.sleep(0.4)

cam.release()
cv2.destroyAllWindows()
