import cv2
import label_image
import tensorflow as tf
import numpy as np

#IMPORTANT PATHS CHANGE TO THE ONE IN YOUR COMPUTER
haarPath =  'haarcascade_frontalface_alt.xml' #Change this to your own directory
FaceFileName = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/TrainedCNN/test.jpg" #Save file to this location
MODEL_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/model_1.1.hdf5"

IMG_SIZE = 70
NEW_SIZE = (IMG_SIZE, IMG_SIZE)

#LOAD CLASSIFIER AND CNN_MODEL
classifier = cv2.CascadeClassifier(haarPath)
model = tf.keras.models.load_model(MODEL_DIR)

webcam = cv2.VideoCapture(0)

while True:
	(rval, im) = webcam.read()
	#im = cv2.flip(im,1,0) #Mirror image -- 

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

		cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)

		font = cv2.FONT_HERSHEY_SIMPLEX
		if(emotion[0] > 0):
			text = "POSITIVE"
		else:
			text = "NEGATIVE"

		cv2.putText(im, text, (x+w,y-50), font, 0.7, (0,0,255),3)

	
	#Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)
	
	#If Esc key is press the break
	if key == 27:
		break









