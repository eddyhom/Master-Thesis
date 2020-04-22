import cv2
import label_image
import time

FaceFileName = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/TrainedCNN/test.jpg" 



im = cv2.imread(FaceFileName)

t = time.time()
results, labels = label_image.main(FaceFileName) #Getting the result from the label_image file.
elapsed = time.time() - t

print(results)
print(labels)
print("Elapsed time: "+str(elapsed))

text1 = str(results[0])+": "+labels[0]
text2 = str(results[1])+": "+labels[1] 

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(im, text1, (1,15), font, 0.5, (0,0,255),2)
cv2.putText(im, text2, (1,175), font, 0.5, (0,0,255),2)

cv2.imshow('Capture',   im)
key = cv2.waitKey(0)




