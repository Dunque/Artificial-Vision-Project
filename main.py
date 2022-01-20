import os
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils

import numpy as np

imgPath = "input/"
imgPathOut = "output/"


def main():

    #Loading the input images
    list_files = os.listdir(imgPath)

    for filename in list_files:
        inputImg = cv2.imread(imgPath+filename, 1)
        #Convert to RGB
        img = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)

        # -------------------- BACKGROUND REMOVAL --------------------

        #Green thresholding
        mask = cv2.inRange(img, (0,80,0), (255,255,75))

        # noise removal
        kernel = np.ones((4,4),np.uint8)
        opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
        closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE, kernel, iterations = 2)

        bw_img = cv2.threshold(closing,127,255,cv2.THRESH_BINARY)[1]

        trimmedPlants = cv2.bitwise_and(img,img,mask = bw_img)

        # convert to grayscale, then apply Otsu
        gray = cv2.cvtColor(trimmedPlants, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # show the output image
        cv2.imwrite("otsu/"+filename, thresh)

        # -------------------- LEAF SEGMENTATION --------------------

        # Euclidean distance from every binary pixel to the nearest zero pixel, 
        # then find peaks in the distance map, with a minimum distance of eachother
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=17, labels=thresh)

        # show the output image
        cv2.imwrite("euclideanDistance/"+filename, D+128)

        # perform a connected component analysis on the local peaks with 8-connectivity
        # then appy the Watershed algorithm
        # D is negated because watershed interprets it as local minimums
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print("Found " + str((len(np.unique(labels)) - 1)) + " unique segments in Image " + filename)

        # -------------------- RESULTS --------------------

        colorArray = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255),
                    (0,0,128),(0,128,0),(128,0,0),(128,128,0),(128,0,128),(0,128,128),
                    (128,128,255),(128,255,128),(255,128,128),(255,255,128),(255,128,255),(128,255,255)]
        i = 0;

        # loop over the unique labels returned by the Watershed
        for label in np.unique(labels):
            # label 0 is the background, so we ignore it
            if label == 0:
                continue
            # otherwise, we draw the regions in the mask
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #We draw the contours, choosing a color from the list
            if (i==len(colorArray)-1):
                i=0
            else:
                i=i+1
                
            cv2.fillPoly(trimmedPlants, cnts, colorArray[i])
                
        # show the output image
        cv2.imwrite(imgPathOut+filename, trimmedPlants)

if __name__ =='__main__':
    main()