# file này là để chụp hình ảnh lấy làm mẫu
#Run file này trước
import cv2
import os
import numpy as np

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640) 
cam.set(4, 480) 

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


face_id = input('\n enter user id end press <return> ==>  ')



count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count = count + 1

        cv2.imwrite("dataSet/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('Face', img)

    if cv2.waitKey(150) & 0xFF == ord('q'):
        break
    elif count >= 20:
        break



cam.release()
cv2.destroyAllWindows()