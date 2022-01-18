import os
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hsv
import numpy as np
from skimage.color import rgb2gray
from skimage import data, io, color
from skimage.filters import threshold_otsu

imgPath = "img/"
imgPathOut = "out/"

def main():

    list_files = os.listdir(imgPath)

    image_list = []
    for filename in list_files:
        image_list.append(io.imread(imgPath+filename))

    img = image_list[0]

    img_hsv = color.rgb2hsv(img)

    plt.imshow(img_hsv)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = np.where(np.logical_and(img_hsv>= (36, 25, 25), img_hsv<= (70, 255,255)))

    plt.imshow(mask)
    ## slice the green
    imask = mask > (0,0,0)
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    io.imsave(os.path.join(imgPathOut, 'prueba.png'), green)


if __name__ =='__main__':
    main()