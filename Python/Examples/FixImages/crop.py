import cv2
import os

directory = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/PositiveCropped/"
f_directory = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/NegativeCropped"


classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

size = 4
images = os.listdir(directory)

    
for img in images:
	file = directory + img
	im = cv2.imread(file)

	#Resize the image to speed up detection
	mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

	#Detect Multiscale / faces
	faces = classifier.detectMultiScale(mini)

	#Draw rectangles around each face
	for f in faces:
		(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
		cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)

		#Save just the rectangle faces in SubRecFaces
		sub_face = im[y:y+h, x:x+w]

		FaceFileName = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/TrainedCNN/test.jpg" #Saving the current image from the webcam for testing
		cv2.imwrite(FaceFileName, sub_face)




	
	#Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)
	
	#If Esc key is press the break
	if key == 27:
		break
