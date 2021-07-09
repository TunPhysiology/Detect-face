#file nhận diện 
import os
import cv2
import numpy as np


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
font = cv2.FONT_HERSHEY_SIMPLEX
#a = 0
#while a < 15:
while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)    ##dao chieu camera
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
	if ret == True:
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			id, conf = recognizer.predict(gray[y:y+h,x:x+w])
			if id == 1:
				id = "tuan"
				conf =  "  {0}%".format(round(100 - conf))
			else:
				id = "unknown"
				conf =  "  {0}%".format(round(100 - conf))
			cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
			cv2.putText(frame, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)
		cv2.imshow('face recognition', frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
	else:
		break
	#a = a +1


cap.release()
cv2.destroyAllWindows()

