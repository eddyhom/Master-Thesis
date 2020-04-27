import os
import shutil
import cv2
import time
import face_crop

rootdir = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Training/Positive/"
#dest_dir = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Training/Neutral/"
dir_pos = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Validation/Positive/"
dir_neg = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Validation/Negative/"
dir_neu = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/Validation/Neutral/"

f_dir = rootdir


images = os.listdir(rootdir)
newsize = (70,70)
i = 0
length = len(images)
t0 = time.time()

for file in images:
	t1 = time.time()
	filepath = rootdir + file
	destpath = f_dir + file
	
	print("Processing file {} ({}%)".format(i, 100*i//length), end="")

	im = cv2.imread(filepath)
	#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	imresize = cv2.resize(im,newsize)	

	cv2.imwrite(destpath,imresize)
	i += 1
	t2 = time.time()
	print(" (total: {}s)".format(int(t2-t0)))











