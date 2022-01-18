import cv2
from skimage import data, io, color
import numpy as np
import matplotlib.pyplot as plt

## Read
img = io.imread("img/1 (1).png")

## convert to hsv
#img_hsv = color.rgb2hsv(img)
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# plt.imshow(img_hsv)
# plt.show()

## mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
#mask = cv2.inRange(img_hsv, (36, 25, 25), (70, 255,255))
mask = cv2.inRange(img, (0, 0, 0), (255, 80, 255))

## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

## save 
cv2.imwrite("out/green.png", green)