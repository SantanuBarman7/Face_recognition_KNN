# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:12:03 2020

@author: Santanu
"""

import cv2, time
import numpy as np

## camera
cam = cv2.VideoCapture(0)

face_in_image = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
## list to store face data
face_data_file = []

d_path = "./face_data/"

file_name = input("Enter the name of the person : ")

skip = 0

while True:
    check, frame = cam.read()
    
     
    if check == False:
        continue
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_in_image.detectMultiScale(gray_image, scaleFactor = 1.05, minNeighbors = 5)
    
    ##if no face in detected then do not store
    if len(face) == 0:
        continue
    
    ##pic the last face from sorted array
    face = sorted(face,key=lambda f:f[2]*f[3])
    
    ##as last face will be the one which is closer
    for last_face in face[-1:]:
        x,y,w,h = last_face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
        ##store a more featured image
        crop = 20
        face_region_of_interest = frame[y-crop:y+h+crop, x-crop:x+w+crop]
        face_region_of_interest = cv2.resize(face_region_of_interest, (200,200))
        
        ##store every 10th frame
        
        skip += 1
        
        if skip % 20 == 0:
            face_data_file.append(face_region_of_interest)
			#print(len(face_data_file))
    cv2.imshow("webcam", frame)
    cv2.imshow("stored face", face_region_of_interest)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


##face list to nupy array
        
face_data_file = np.asarray(face_data_file)
face_data_file = face_data_file.reshape((face_data_file.shape[0],-1))

##save to file

np.save(d_path + file_name + ".npy", face_data_file)
            
print("Successfully saved face data at " + d_path+file_name + '.npy')

cam.release()
cv2.destroyAllWindows()
    
    
