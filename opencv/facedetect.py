import cv2 as cv


img = cv.imread('./Resources/Photos/group 2.jpg')

grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('grey',grey)


haarcascade = cv.CascadeClassifier('facedetect.xml')

face_rect = haarcascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=9)

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=1)

cv.imshow("img",img)

cv.waitKey(0)