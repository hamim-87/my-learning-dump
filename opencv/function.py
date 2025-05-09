import cv2 as cv
import numpy as np


img = cv.imread('./Resources/Photos/park.jpg')

# ##################### transform ######################

# cv.imshow('park',img)

# def transform(img,x,y):
#     transMat = np.float32([[1,0,x],[0,1,y]])
#     dim = (img.shape[1],img.shape[0])
#     return cv.warpAffine(img,transMat,dim)


# nw = transform(img,100,200)

# cv.imshow('transform',nw)

# ###################### rotation ##########################

# def rotate(img,angle,rotationPoint=None):
#     if rotationPoint is None:
#         rotationPoint = (img.shape[1]//2,img.shape[0]//2)

#     dim = (img.shape[1],img.shape[0])

#     rotMat = cv.getRotationMatrix2D(rotationPoint,angle,1.0)

#     return cv.warpAffine(img,rotMat,dim)

# rot = rotate(img,20)
# cv.imshow('rot',rot)

# ################### flip #############################
# flip = cv.flip(img,1)
# cv.imshow("flip", flip)


##################### contour #########################

# canny = cv.Canny(img,125,175)

# contours, hierarchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)


# grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# ret,thresh = cv.threshold(grey,125,255,cv.THRESH_BINARY)

# cv.imshow('thresh',thresh)
# contours, hierarchies = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

# blank = np.zeros(img.shape,dtype='uint8')

# cv.drawContours(blank,contours,-1,(0,0,255),1)

# cv.imshow('contour',blank)
# cv.waitKey(0)


################## blur #############

# blur = cv.blur(img,(3,3))

# cv.imshow("blur", blur)

# cv.waitKey(0)


############## Edge detection #############

grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('grey',grey)

lap = cv.Laplacian(grey,cv.CV_64F)
lap = np.uint8(np.absolute(lap))

cv.imshow('laplace',lap)


#sobel

sobelx = cv.Sobel(grey,cv.CV_64F,1,0)
sobely = cv.Sobel(grey,cv.CV_64F,0,1)

cv.imshow('x',sobelx)
cv.imshow('y',sobely)


sobel_combine = cv.bitwise_or(sobelx,sobely)
cv.imshow('combine',sobel_combine)

##canny

canny = cv.Canny(grey,150,175)
cv.imshow('canny',canny)

cv.waitKey(0)