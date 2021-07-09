#file này nhằm tạo 1 file .yml để lưu id face
#run file này sau khi có ảnh mẫu

import cv2
import numpy as np
import os
from PIL import Image

path = "dataSet"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

def getImageswithID(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
	facesSample = []
	ids = []
	for imagePath in imagePaths:
		faceImg = Image.open(imagePath).convert("L")
		faceNp = np.array(faceImg, "uint8")
		id = int(os.path.split(imagePath)[-1].split(".")[1])
		faces = detector.detectMultiScale(faceNp)
		for (x,y,w,h) in faces:
			facesSample.append(faceNp[y:y+h, x:x+w])
			ids.append(id)
	return facesSample, ids
faces, ids = getImageswithID(path)
recognizer.train(faces, np.array(ids))
recognizer.write("trainer/trainer.yml")
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

