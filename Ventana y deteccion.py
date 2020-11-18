## import the necessary packages

import time
import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np

img= cv2.imread("1.jpg") # leer una imagen
image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #conversion a escala de grises

def pyramid(image, scale=1.5, minSize=(30, 30)): #función ventana piramide
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize): #funcion ventana deslizante
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):

        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

(winW, winH) = (100, 100)

# declarate the matrix

A=np.empty((100,winW,winH))
b=1

window=[]
#%%
# loop over the image pyramid
for resized in pyramid(image, scale=2):
    # loop over the sliding window for each layer of the pyramid

    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):

        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        # since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.25)
   #______ ###### the problem #### ______ i want to save the windows obtein in A, but I get the error in the next two lines of code
        #A[b,:,:]=window
        #b=b+1;

cv2.destroyAllWindows()