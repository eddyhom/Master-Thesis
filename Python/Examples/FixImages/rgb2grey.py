import os
import shutil

rootdir = os.getcwd()
dst_dir = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/happy"
file1 = open('SMILE_list.txt')
Lines = file1.readlines()
count = 0

for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		for line in Lines:
			if (file == line.strip()):
				filepath = subdir + os.sep + file
				shutil.move(filepath, dst_dir)
			

