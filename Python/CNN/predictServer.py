import tensorflow as tf
import cv2
import numpy as np
from math import floor


predEmo = ["negative", "positive"]
size = 2
IMG_SIZE = 48
NEW_SIZE = (IMG_SIZE, IMG_SIZE)


def predict(model, face):
    im = cv2.resize(face, NEW_SIZE) / 255

    im = (np.expand_dims(im, 2))
    im = (np.expand_dims(im, 0))

    prediction = model.predict(x=im)  # NEGATIVE = -X, POSITIVE = X.

    pred = prediction[0]
    max_val = max(pred)
    max_idx = np.where(pred == max_val)
    i = max_idx[0]

    return predEmo[i[0]]


def isFace(face_cascade, model, pic_path):
    img_name = pic_path
    im = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    if im is not None: #If image was open succesfully
        # Resize the image to speed up detection
        im = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = im[y*4:y*4 + h*4, floor(x*4+(x+4)*0.08):floor(4*(x+w) - 4*(x+w)*0.08)]

            emo = predict(model, face)
            return 1 if emo == predEmo[1] else -1 #Return 1 if positive -1 if negative
        else:
            return 0
    else:
        print("image: ", img_name, " couldnt be read")
        return 0


if __name__ == '__main__':
    MODEL_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\Models\\model_2emotions_30.hdf5"
    model = tf.keras.models.load_model(MODEL_DIR)
    emotion = predict(model)
    print(emotion)

