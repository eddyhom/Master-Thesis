import cv2
import tensorflow as tf
import numpy as np
import time

#IMPORTANT PATHS CHANGE TO THE ONE IN YOUR COMPUTER
haarPath =  'haarcascade_frontalface_alt.xml' #Change this to your own directory
MODEL_DIR = 'model_1.1.hdf5'

IMG_SIZE = 70
NEW_SIZE = (IMG_SIZE, IMG_SIZE)
IMG_RESIZE = (160, 120)

#LOAD CLASSIFIER AND CNN_MODEL
classifier = cv2.CascadeClassifier(haarPath)
model = tf.keras.models.load_model(MODEL_DIR)

webcam = cv2.VideoCapture(0)

while True:
	t0 = time.time()
	(rval, im) = webcam.read()

	#im = cv2.flip(im,1,0) #Mirror image -- 
	im = cv2.resize(im, IMG_RESIZE)
	#Detect Multiscale / faces
	
	faces = classifier.detectMultiScale(im)

	#Draw rectangles around each face
	for f in faces:
		(x, y, w, h) = [v for v in f] #Scale the shapesize backup

		#PART OF THE PICTURE INCLUDING FACE, RESIZE IT TO FIT CNN's INPUT
		
		sub_face = im[y:y+h, x:x+w]
		subface_resize = cv2.resize(sub_face, NEW_SIZE)/255
		subface_resize = (np.expand_dims(subface_resize, 0))

		emotion = model.predict(x=subface_resize) #POSITIVE VALUE = POSITIVE EMOTION // NEGATIVE VALUE = NEGATIVE EMOTION

		font = cv2.FONT_HERSHEY_SIMPLEX
		if(emotion[0] > 0):
			text = ":D"
		else:
			text = ":("

		cv2.putText(im, text, (x+int(w/2),y), font, 0.4, (0,0,255),2)

	
	#Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)
	t1 = time.time()

	dt = t1-t0
	print("TOTAL TIME in (s): "+str(dt))
	print("-----------------------------------------------------------------------------------")

	
	#If Esc key is press the break
	if key == 27:
		break









