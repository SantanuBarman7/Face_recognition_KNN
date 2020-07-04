# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:11:02 2020

@author: Santanu
"""

import cv2,time
import numpy as np
import os
##KNN
def distance(x1, x2):
	return np.sqrt(((x1-x2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
    
## camera
cam = cv2.VideoCapture(0)

face_in_image = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

d_path = "./face_data/"

skip = 0 
face_data_file =[]
labels = []

class_id = 0 # Labels for the given file
names = {} #Mapping btw id - name

##Data Preparation
for fx in os.listdir(d_path):
	if fx.endswith('.npy'):
		#Create a mapping btw class_id and name
		names[class_id] = fx[:-4]
		print("Loaded "+fx)
		data_item = np.load(d_path+fx)
		face_data_file.append(data_item)

		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)
        
face_dataset = np.concatenate(face_data_file,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

##Testing

while True:
    check, frame = cam.read()
    
     
    if check == False:
        continue
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_in_image.detectMultiScale(gray_image, scaleFactor = 1.05, minNeighbors = 5)
    
    ##if no face in detected then do not store
    if len(face) == 0:
        continue
    
    ##pick the last face from sorted array
    face = sorted(face,key=lambda f:f[2]*f[3])
    
    ##as last face will be the one which is closer
    for last_face in face[-1:]:
        x,y,w,h = last_face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        ##store a more featured image
        crop = 10
        face_region_of_interest = frame[y-crop:y+h+crop, x-crop:x+w+crop]
        face_region_of_interest = cv2.resize(face_region_of_interest, (200,200))
        
        ##Predicted Label (out)
        out = knn(trainset, face_region_of_interest.flatten())
		
        ##Display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


























