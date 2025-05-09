import cv2 as cv


img = cv.imread('./Resources/Photos/cat.jpg')

grey_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


#sift algo
# sift = cv.SIFT.create()

# keypoints, descriptor = sift.detectAndCompute(grey_img,None)

#orb algo
orb = cv.ORB_create()

keypoints, descriptor = orb.detectAndCompute(grey_img,None)

print(len(keypoints))

img_with_kp = cv.drawKeypoints(grey_img,keypoints,None)






cv.imshow("cat",img)
cv.imshow("grey cat",grey_img)
cv.imshow("keypoint img",img_with_kp)
cv.waitKey(0)