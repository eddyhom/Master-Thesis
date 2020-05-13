from typing import List, Union

import tensorflow as tf
import cv2
import numpy as np
import os
import time

IMG_DIR_POS = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/ValidationGray/Positive/"
IMG_DIR_NEG = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/ValidationGray/Negative/"
IMG_DIR_NEU = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/ValidationGray/Neutral/"
MODEL_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Models/model_2.8.hdf5"
IMG_SIZE = 70
NEW_SIZE = (IMG_SIZE, IMG_SIZE)

model = tf.keras.models.load_model(MODEL_DIR)

Dir = [IMG_DIR_NEG, IMG_DIR_NEU, IMG_DIR_POS]

emotion = ["Negative", "Neutral", "Positive"]

L = []

count = []


for ind, dir_emo in enumerate(Dir):
    counter = 0

    IMG_DIR = dir_emo
    images = os.listdir(dir_emo)
    L.append(len(images))

    for image in images:
        pic_dir = IMG_DIR + image
        im = cv2.imread(pic_dir, 0) / 255
        im = (np.expand_dims(im, 2))
        im = (np.expand_dims(im, 0))

        print(np.shape(im))

        t0 = time.time()
        prediction = model.predict(x=im)  # NEGATIVE = -X, POSITIVE = X.
        t1 = time.time()
        dt = t1 - t0
        print("Time to predict in (s): " + str(dt))
        pred = prediction[0]
        max_val = max(pred)
        max_idx = np.where(pred == max_val)
        i = max_idx[0]
        i = i[0]
        print(pred)
        print(max_val)
        print(i)

        print("---------------Image was: " + emotion[ind] + " prediction was: " + emotion[i] + "---------------")
        if (i == ind):
            counter += 1

    count.append(counter)

accNeg = (count[0] / L[0]) * 100
accNeu = (count[1] / L[1]) * 100
accPos = (count[2] / L[2]) * 100

print("Accuracy for Negative: " + str(accNeg) + "%")
print("Accuracy for Neutral: " + str(accNeu) + "%")
print("Accuracy for Positive: " + str(accPos) + "%")
