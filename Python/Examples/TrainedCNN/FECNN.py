import cv2
import label_image

#Load XML file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

size = 4

webcam = cv2.VideoCapture(0)

while True:
	(rval, im) = webcam.read()
	im = cv2.flip(im,1,0) #Mirror image
	
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

		results, labels = label_image.main(FaceFileName) #Getting the result from the label_image file.

		text1 = str(format(results[0],'.2f'))+": "+labels[0]
		text2 = str(format(results[1],'.2f'))+": "+labels[1] 

		font = cv2.FONT_HERSHEY_SIMPLEX
		if(results[0] > results[1]):
			text = labels[0]
		else:
			text = labels[1]

		cv2.putText(im, text, (x+w,y-50), font, 0.7, (0,0,255),3)
		#cv2.putText(im, text2, (x+w,y-80), font, 0.7, (0,0,255),3)
		


	
	#Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)
	
	#If Esc key is press the break
	if key == 27:
		break









