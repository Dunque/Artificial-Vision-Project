import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils

import numpy as np

imgPath = "img/"
imgPathOut = "out/"

def main():

    #Loading the input images
    list_files = os.listdir(imgPath)

    image_list = []
    for filename in list_files:
        inputImg = cv2.imread(imgPath+filename, 1)
        #Convert to RGB
        rgbimage = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
        image_list.append(rgbimage)


    #image1 example
    img = image_list[0]
    
    #Green thresholding
    mask = cv2.inRange(img, (0,80,0), (255,255,75))

    """ ## slice the green
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    # save
    cv2.imwrite("out/green.png", green) """

    """ kernel = np.ones((8,8))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) """

    # noise removal
    kernel = np.ones((4,4),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    gaussian = cv2.GaussianBlur(opening, (15, 15), 0)

    ret, bw_img = cv2.threshold(opening,127,255,cv2.THRESH_BINARY)

    trimmedPlants = cv2.bitwise_and(img,img,mask = bw_img)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(trimmedPlants, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))


    colorArray = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255),
                  (0,0,128),(0,128,0),(128,0,0),(128,128,0),(128,0,128),(0,128,128),
                  (128,128,255),(128,255,128),(255,128,128),(255,255,128),(255,128,255),(128,255,255)]
    i = 0;

    # loop over the unique labels returned by the Watershed
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)


        #We draw the leaves
        if (i==len(colorArray)-1):
            i=0
        else:
            i=i+1
        cv2.fillPoly(trimmedPlants, cnts, colorArray[i])
        #cv2.drawContours(img, cnts, -1, colorArray[i], 3)

        """ # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) """
            
    # show the output image
    cv2.imwrite("out/1.png", trimmedPlants)

if __name__ =='__main__':
    main()