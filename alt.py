import cv2
from skimage import data, io, color
import numpy as np
import matplotlib.pyplot as plt

## Read
img = io.imread("img/1 (1).png")

mask = cv2.inRange(img, (0,80,0), (255,255,75))

# save
cv2.imwrite("out/mask.png", mask)


## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

# save
cv2.imwrite("out/green.png", green)

gray = cv2.cvtColor(1-green,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

## save 
cv2.imwrite("out/opening.png", opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

## save 
cv2.imwrite("out/background.png", sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

## save 
cv2.imwrite("out/foreground.png", sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_fg, sure_bg)

## save 
cv2.imwrite("out/unknown.png", unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

## save 
cv2.imwrite("out/final.png", markers)