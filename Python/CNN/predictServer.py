import tensorflow as tf
import cv2
import numpy as np
import time

predEmo = ["anger_disgust_fear_sadness", "happy_surprise"]


def predict(model):
    imgname = "emotion.png"

    im = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    im = tf.cast(im, tf.float32)
    im = (np.expand_dims(im, 2))
    im = (np.expand_dims(im, 0))

    t1 = time.time()
    prediction = model.predict(x=im)  # NEGATIVE = -X, POSITIVE = X.
    t2 = time.time()

    print("Time taken was: ", t2-t1)

    pred = prediction[0]
    max_val = max(pred)
    max_idx = np.where(pred == max_val)
    i = max_idx[0]

    return predEmo[i[0]]



if __name__ == '__main__':
    MODEL_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\Models\\model_2emotions_30.hdf5"
    model = tf.keras.models.load_model(MODEL_DIR)
    emotion = predict(model)
    print(emotion)

