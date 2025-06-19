"""
Live Image-Capture Utility for Building a Hand-Gesture Dataset
--------------------------------------------------------------

This script lets you step through a predefined list of gesture labels and
capture webcam frames for each label.  For every saved frame it writes a JPEG
named `<label>_<index>.jpg` to a folder structure like:

    <script_dir>/dataset/
        up_fine/
            up_fine_0000.jpg
            up_fine_0001.jpg
            ...
        up_fast/
        ...

Key bindings while the capture window is focused
------------------------------------------------
SPACE : save the current frame to disk
q     : finish the current label and move to the next one
ESC   : (handled implicitly by `cv2.waitKey`) closes the window / aborts
"""

import cv2, pathlib, time   # OpenCV for video I/O, pathlib for paths, time for debounce

# Ordered list of gesture names; the folder name doubles as the class label.
labels = [
    "up_fine", "up_fast", "down_fine", "down_fast",
    "left_fine", "left_fast", "right_fine", "right_fast",
    "forward_fine", "forward_fast", "backward_fine", "backward_fast",
    "pickup", "neutral"
]

# -------------------------------------------------------------------------
# Ensure that the dataset root directory exists next to this script file.
# -------------------------------------------------------------------------
root = pathlib.Path(__file__).resolve().parent / "dataset"
root.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
# Initialise webcam (device 0). Abort early if no camera is found.
# -------------------------------------------------------------------------
cam = cv2.VideoCapture(0)
assert cam.isOpened(), "No webcam found"

# -------------------------------------------------------------------------
# Loop over every label and capture images for it.
# -------------------------------------------------------------------------
for label in labels:
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)            # e.g. dataset/up_fine/
    counter = len(list(folder.glob("*.jpg")))            # Continue numbering if files exist.

    print(f"\n--- {label}:  SPACE = save   q = next label ---")
    while True:
        ok, frame = cam.read()                           # Grab a frame from the webcam.
        if not ok:
            break

        # Overlay the current label and running count in the preview window.
        cv2.putText(frame, f"{label}  {counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("rec", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):                                # SPACE → save frame
            path = folder / f"{label}_{counter:04d}.jpg"
            success = cv2.imwrite(str(path), frame)
            if success:
                counter += 1
                print("saved ->", path.name)
            else:
                print("⚠️  failed to save", path)
        elif k == ord('q'):                              # q → proceed to next label
            break

    time.sleep(0.4)                                      # Small delay to avoid key-bounce

# -------------------------------------------------------------------------
# Clean-up: release camera and close all OpenCV windows.
# -------------------------------------------------------------------------
cam.release()
cv2.destroyAllWindows()
