## This program first ensures if the face of a person exists in the given image or not then if it exists, it crops
## the image of the face and saves to the given directory.

## Importing Modules
import cv2
import os
import time


#################################################################################

##Make changes to these lines for getting the desired results.

## DIRECTORY of the images
directory = "/media/sf_Ubuntu/Linux/JEFF-database/Positive/"
fin_dir = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/JEFF_Database/Positive/"

## directory where the images to be saved:
'''dir_pos = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Positive/"
dir_neg = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Negative/"
dir_neu = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Neutral/"

file1 = open('/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/list_patition_label.txt')
Lines = file1.readlines()
'''
facedata = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/TrainedCNN/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(facedata)


newsize = (70, 70)
################################################################################
            
def facecrop(image, cascade, dest):
	## Reading the given Image with OpenCV
	img = cv2.imread(image)
	
	#minisize = (img.shape[1], img.shape[0])
	#miniframe = cv2.resize(image, newsize)

	faces = cascade.detectMultiScale(img)
	count = 0

	for f in faces:
		count += 1
		x, y, w, h = [ v for v in f ]
		#cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

		sub_face = img[y:y+h, x:x+w]
		img_resize = cv2.resize(sub_face, newsize)
		#f_name = image.split('/')
		#f_name = f_name[-1]

		#if(count!=1):
			#f_name = f_name.replace(".jpg","_"+count+".jpg")

		cv2.imwrite(dest, img_resize)

	

if __name__ == '__main__':
	images = os.listdir(directory)
	i = 0
	length = len(images)
	t0 = time.time()

	for img in images:
		t1 = time.time()
		file = directory + img
		dest_dir = fin_dir + img
		print("Processing file {} ({}%)".format(i, 100*i//length), end="")
		facecrop(file, cascade, dest_dir)
		i += 1
		t2 = time.time()
		print(" (total: {}s)".format(t2-t0,'.2f'))








