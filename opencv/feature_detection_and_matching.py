import cv2 as cv
import numpy as np

#img 1
img = cv.imread('./Resources/Photos/cat.jpg')
grey_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


#img 2
img2 = cv.imread("./Resources/Photos/catSS.png")
img2 = cv.resize(img2,(600,400))
grey_img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)



# #orb algo
# orb = cv.ORB_create()

# keypoints, descriptor = orb.detectAndCompute(grey_img,None)



#sift algo
sift = cv.SIFT.create()


#finding keypints
keypoints1, descriptor1 = sift.detectAndCompute(grey_img,None)
kp2, des2 = sift.detectAndCompute(grey_img2,None)


#use bruteforce to match
bf = cv.BFMatcher(cv.NORM_L2)

# des1 = np.uint8(descriptor1)
# des2 = np.uint8(des2)

matched = bf.knnMatch(descriptor1,des2,k=2)


#lowe's ratio

good_match = []

for m,n in matched:
    if m.distance < 0.75 * n.distance:
        good_match.append(m)

# matches = cv.drawMatches(grey_img,keypoints1,grey_img2,kp2,matched,None)

goodmatches = cv.drawMatches(grey_img,keypoints1,grey_img2,kp2,good_match,None)


# img_with_kp = cv.drawKeypoints(grey_img,keypoints1,None)
# cv.imshow("cat",img)
# cv.imshow("ss cat",img2)
# cv.imshow("grey cat",grey_img)
# cv.imshow("keypoint img",img_with_kp)

# cv.imshow("match", matches)
cv.imshow("good match", goodmatches)
cv.waitKey(0)