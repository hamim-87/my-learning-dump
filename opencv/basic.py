import cv2 as cv
import numpy as np

# ######################### read immage ###########################

# img = cv.imread('../Resources/Photos/cats.jpg')

# cv.imshow("cat",img)
# cv.waitKey(0)


########################## vedio ###########################


# capture = cv.VideoCapture("./Resources/Videos/dog.mp4")

# while True:
#     isTrue, frame = capture.read()

#     if not isTrue:
#         break

#     cv.imshow("video",frame)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# capture.release()


###################### rescale ##################

# def rescaleFrame(frame, scale=0.5):

#     width = int( frame.shape[1] * scale)
#     height = int( frame.shape[0] * scale)

#     dimension = (width,height)

#     return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)


# capture = cv.VideoCapture("./Resources/Videos/dog.mp4")

# while True:
#     isTrue, frame = capture.read()

#     if not isTrue:
#         break

#     cv.imshow('dog',rescaleFrame(frame))

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# capture.release()


########################### draw ##########################

# blank = np.zeros((500,500,3),dtype='uint8')

# blank[:] = 0,255,0

# cv.imshow('blank',blank)
# cv.waitKey(0)


################# function ##############

#color

img = cv.imread('./Resources/Photos/cat.jpg')

# cv.imshow('cat',img)
# grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('grey car',grey)

# cv.waitKey(0)


#blur
blur = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)

# cv.imshow('blur',blur)

# cv.waitKey(0)


#edge detection

canny = cv.Canny(blur,175,225)

cv.imshow('cany',canny)

#dilated 

dilated = cv.dilate(canny, (7,7),iterations=3)

cv.imshow('dilate',dilated)

#ERODE
erode = cv.erode(dilated,(7,7),iterations=3)
cv.imshow('erode',erode)

cv.waitKey(0)