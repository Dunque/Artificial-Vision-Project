import sys
import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt

inputImageStr = './img/1 (1).png'
outputImageStr = './output1.jpeg'
verbose = False


def printResults(input,outImage):
	title = 'Input Image'
	fig1 = plt.figure(num=title)
	plt.imshow(input)
	plt.title(title)
	plt.axis('off')

	title = 'Output Image'
	fig2 = plt.figure(num=title)
	plt.imshow(outImage)
	plt.title(title)
	plt.axis('off')

	plt.show()

def segmentar_plantones(inputImage):
    light_green = (255,255,75)
    dark_green = (0,110,0)

    inputImage= cv2.GaussianBlur(inputImage, (15, 15), 0)
    mask = cv2.inRange(inputImage, dark_green,light_green)
    kernel = np.ones((8,8))
    close = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    open = cv2.morphologyEx(close,cv2.MORPH_OPEN, kernel)

    sure_bg = cv2.dilate(open, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(open, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(inputImage, markers)

    return markers

def main():
    inputImage = cv2.imread(inputImageStr, 1)
    inputImage2 = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    outputImage = segmentar_plantones(inputImage2)

    cv2.imwrite(outputImageStr, outputImage)

    if verbose:
        printResults(inputImage2,outputImage)

def printHelp():
    print(f"\nSegmentación de Plantones\nUso:\n")
    print(f"  -i  --input:\t\tImagen de entrada\n\t\t\tPor defecto: [{inputImageStr}]\n")
    print(f"  -o  --output:\t\tImagen de salida\n\t\t\tPor defecto: [{outputImageStr}]\n")
    print(f"  -h  --help:\t\tAyuda\n")
    print(f"  -v  --verbose:\tMostrar debug\n")
    quit()

def parseArguments(argv):
    global inputImageStr
    global outputImageStr
    global verbose

    arguments=iter(argv)
    for arg in arguments:
        if arg=='-h' or arg =='--help':
            printHelp()
        if arg=='-i' or arg=='--input':
            next=arguments._next_()
            inputImageStr=next if os.path.isfile(next) else inputImageStr
        if arg=='-o' or arg=='--output':
            next=arguments._next_()
            outputImageStr=next
        if arg=='-v' or arg=='--verbose':
            verbose=True

    if verbose:
        print(f"\nUsing:")
        print(f"Input Image: {inputImageStr}")
        print(f"Output Image: {outputImageStr}\n")


if __name__ == "__main__":
    parseArguments(sys.argv[1:])
    main()