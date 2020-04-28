haarcascade_frontalface_alt.xml:
It is the HAAR that detects faces. Keep in the same directory otherwise change path to its location.

model_1.1.hdf5:
It is the CNN-model that predicts the emotion given a face. It receives an array of images of size (70x70x3) and it outputs a single value, if negative the emotion is predicted to be negative and if positive the emotion is predicted to be positive.

In FECNN.py:
Dependencies needed to run script are cv2 (Computer Vision version 2), tensorflow (version 2) and numpy.
Time that takes since reading image from camera until it shows it again with prediction is approx 0.2s..
