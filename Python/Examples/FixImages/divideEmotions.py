import os
import shutil
import cv2
import time
import face_crop

rootdir = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Positive/"
#dest_dir = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Neutral/"
dir_pos = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Positive/"
dir_neg = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Negative/"
dir_neu = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Neutral/"

#dest_dir_pos = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Positive/"
#dest_dir_neg = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Negative/"

#count = 0

file1 = open('/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/list_patition_label.txt')
Lines = file1.readlines()

facedata = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/TrainedCNN/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(facedata)

countpos = 0
countneg = 0
countneu = 0
length = len(Lines)
t0 = time.time()
i=0
dest = rootdir

for line in Lines:
	words = line.split()
	
	filepath = rootdir + words[0]

	image = cv2.imread(filepath)

	print("Processing file {} ({}%)".format(i, 100*i//length), end="")

	if words[1] in ('2','3','5','6'):
		countneg += 1
		dest = dir_neg + words[0]
	
	elif words[1] in ('4'):	
		countpos += 1
		dest = dir_pos + words[0]

	elif words[1] in ('7'):
		countneu += 1
		dest = dir_neu + words[0]

	face_crop.facecrop(image, cascade, dest)
	i+=1
	t2 = time.time()

	print(" (total: {}s)".format(int(t2-t0), '.2f'))
		
	
print("Positive: {}".format(countpos))
print("Negative: {}".format(countneg))
print("Neutral: {}".format(countneu))
















