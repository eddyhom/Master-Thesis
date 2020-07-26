#import tensorflow as tf
import cv2
import numpy as np
import time
from math import floor
import os
from random import choice


predEmo = ["negative", "positive"] #["fear", "neutral", "happy"]#
size = 2
IMG_SIZE = 48
NEW_SIZE = (IMG_SIZE, IMG_SIZE)
dir_happy = 'D:\\Users\\eddy_\\Downloads\\65125_128470_bundle_archive\\CK+48\\Test\\happy\\'
dir_angry = 'D:\\Users\\eddy_\\Downloads\\65125_128470_bundle_archive\\CK+48\\Test\\fear_sadness\\'
img_happy = os.listdir(dir_happy)
img_angry = os.listdir(dir_angry)


def predict(model, face):
    im = cv2.resize(face, NEW_SIZE) / 255

    im = (np.expand_dims(im, 2))
    im = (np.expand_dims(im, 0))

    t1 = time.time()
    prediction = model.predict(x=im)  # NEGATIVE = -X, POSITIVE = X.
    t2 = time.time()

    print("Time taken was: ", t2-t1)
    #print(prediction)

    pred = prediction[0]
    max_val = max(pred)
    max_idx = np.where(pred == max_val)
    i = max_idx[0]

    return "1" if i == 1 else "2"   #predEmo[i[0]]


def isFace(face_cascade, model, pic_path):
    img_name = pic_path
    im = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    if im is not None:
        # Resize the image to speed up detection
        im = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))
        faces = face_cascade.detectMultiScale(im, 1.05, 3)
        msg = ""
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            fa = cv2.imread(dir_happy+choice(img_happy), cv2.IMREAD_GRAYSCALE)
            #im[y*4:y*4 + h*4, floor(x*4+(x+4)*0.08):floor(4*(x+w) - 4*(x+w)*0.08)]
            if len(faces) > 1:
                (x2, _, _, _) = faces[1]
                fa2 = cv2.imread(dir_angry+choice(img_angry), cv2.IMREAD_GRAYSCALE)
                if x < x2:
                    msg = predict(model, fa)
                    msg = msg + predict(model, fa2)
            else:
                msg = predict(model, cv2.imread(dir_angry+choice(img_angry), cv2.IMREAD_GRAYSCALE))

        if len(msg) == 0:
            return "0"
        else:
            return msg
    else:
        print("image: ", img_name, " couldnt be read")
        return "0" ##Change this to zero


if __name__ == '__main__':
    MODEL_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\Models\\model_2.11.hdf5"
    #MODEL_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\Models\\model_2emotions_30.hdf5"

    directory = 'D:\\Users\\eddy_\\Downloads\\65125_128470_bundle_archive\\CK+48\\Test\\happy\\'
    directory = 'D:\\Users\\eddy_\\Desktop\\Master-Thesis\\Master-Thesis\\DB\\Emotions\\fear\\'
    directory = 'D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\GazeboPics\\FoundFaces\\Angry\\'

    model = tf.keras.models.load_model(MODEL_DIR)

    images = os.listdir(directory)
    dic = {"fear": 0, "neutral": 0, "happy": 0}

    for i, img in enumerate(images):
        file = directory + img
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        emotion = predict(model, img)
        dic[emotion] = dic[emotion] + 1
        print(emotion, i, dic[emotion])

    print(dic)

