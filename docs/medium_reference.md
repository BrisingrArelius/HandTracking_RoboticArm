# Medium Reference

## Original Article

**Title:** How I Built a Hand Gesture Recognition Model in Python — Part 2  
**Author:** Odilbek Tokhirov  
**Published:** 3 Oct 2023 on Medium  
**URL:** <https://medium.com/@odil.tokhirov/how-i-built-a-hand-gesture-recognition-model-in-python-part-2-5d8987bb0756>

## What We Re-used

| Section in Article | How it is used here |
|--------------------|---------------------|
| Landmark-based feature idea (21 × 2 = 42 floats) | Adopted unchanged for both NN and RF classifiers. |
| Four-class MLP example code | Expanded to 14 classes; hidden layers widened; soft-max added. |
| Webcam demo structure (`cv2.VideoCapture`, `cv2.imshow`, debounce) | Used as a starting point and heavily modified. |

All other code (data-collection script, RF training, ROS integration, etc.) was written specifically for this repository.

> **Copyright notice:**  
> © Odilbek Tokhirov. The referenced article is cited here for educational and attribution purposes under fair-use guidelines.  
> If you are the author and prefer a different form of attribution, please open an issue.

