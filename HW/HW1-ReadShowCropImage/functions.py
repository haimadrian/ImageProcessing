__author__ = "Haim Adrian"

import numpy as np
import cv2
from matplotlib import pyplot as plt


def readImage(imageName):
    if imageName.__class__ != "".__class__:
        return None

    # cv2 loads an image in BGR format
    image = cv2.imread(imageName)
    return image


def validateImage(image):
    if not isinstance(image, np.ndarray):
        return None

    dim = image.shape
    if len(dim) < 2 or len(dim) > 3:
        return None

    if len(dim) == 2:
        image = cv2.merge((image, image, image))

    return image


def showImageUsingMatPlotLib(title, image, isGray=False):
    plt.figure(title)

    if isGray:
        plt.imshow(image, cmap="gray")
    else:
        # Mat Plot Lib uses RGB and not BGR.
        plt.imshow(image[:, :, ::-1])

    plt.axis('off')
    plt.title(title)
    plt.show()


def showImageUsingCV2AndMatPlotLib(image):
    image = validateImage(image)
    cv2.imshow("Image using CV2", image)

    showImageUsingMatPlotLib("Image Using MatPlotLib", image)


def showImageAsBinary(image):
    image = validateImage(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Do this using cv2
    ret, bwImage = cv2.threshold(image.copy(), 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary Image using cv2", bwImage)

    # Do this manually
    bwImage2 = image.copy()
    bwImage2[bwImage2 >= 127] = 255
    bwImage2[bwImage2 < 127] = 0
    showImageUsingMatPlotLib("Binary Image", bwImage2, True)


def showImageCropped(image, newDimension, fromXY):
    image = validateImage(image)
    image = image[fromXY[0]: fromXY[0] + newDimension[0], fromXY[1]: fromXY[1] + newDimension[1], :]

    showImageUsingMatPlotLib("Image Cropped", image)
