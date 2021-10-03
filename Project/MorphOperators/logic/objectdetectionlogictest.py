__author__ = "Haim Adrian"

from logic.functions import *
import cv2
import numpy as np
import imutils

MORPH_ITERATIONS_COUNT = 2
BLUR_KERNEL_SIZE = 13
THRESH_MIN = 92
THRESH_MAX = 255
MIN_CONTOURS_IN_OBJECT_THRESH = 5
IMAGE_SHAPE = (400, 400)
GAUSS_BLUR_MASK_SHAPE = (3, 3)
MORPH_MASK_SHAPE = (3, 3)
COIN_AREA = (200, 450)
BILL_AREA = (5500, 10000)
OBJECT_MARKER_THICKNESS = 2
OBJECT_MARKER_COLOR = 255


def runObjectDetection(image, consoleConsumer=None):
    image = cv2.resize(image, (400, 400))
    # 1. Pre-Processing: Image Contrast Adjustment is done so we can ease edge detection
    #    by gradient, when object edges color is similar to the background color.
    contrastAdjustmentImage = doImageContrastAdjustment(image, 1.3)

    # Blur image so we will reduce amount of sharp lines, to make it easier for us
    # focusing on objects as whole
    imageBlur = cv2.medianBlur(contrastAdjustmentImage, BLUR_KERNEL_SIZE)

    # Now convert images to gray, cause object detection is going to be as binary. (black/white)
    imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)

    # After the gradient, we get image with contours. Background is black and contours in white.
    # Use threshold to remove non-interesting contours, and leave only those we are interested in,
    # those are the objects.
    _, imageBinary = cv2.threshold(imageGray, THRESH_MIN, THRESH_MAX, cv2.THRESH_BINARY)

    kernel = np.ones(MORPH_MASK_SHAPE, np.uint8)

    # Use Closing, so first we will use Dilation, to fill in the shapes, and then Erosion,
    # to reduce the shapes to their original size. This way we try to fill in little holes inside
    # objects.
    # imageDilated = cv2.dilate(imageBinary, kernel, iterations=4)
    # imageEroded = cv2.erode(imageDilated, kernel, iterations=4)
    imageClosing = \
        cv2.morphologyEx(imageBinary.copy(), cv2.MORPH_CLOSE, kernel, iterations=8)

    # Now it is time to replace the value of each pixel with the distance to the nearest background pixel.
    imageDilate = \
        cv2.morphologyEx(imageClosing.copy(), cv2.MORPH_DILATE, kernel, iterations=1)

    # We use findContours to detect objects.
    # Then we make our array regular with the grap_contours method
    # contours, hierarchy = cv2.findContours(theImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(imageDilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    imageMarks = image.copy()
    highlightObjectsInImage(contours, imageMarks)
    showImageUsingMatPlotLib('Test Morphology Object Detection',
                             True,
                             None,
                             (imageBlur, 231, 'Blur', None),
                             (imageGray, 232, 'Gray', 'gray'),
                             (imageBinary, 233, 'Binary', 'gray'),
                             (imageDilate, 234, 'Dilate', 'gray'),
                             (imageClosing, 235, 'Closing', 'gray'),
                             (imageMarks, 236, 'Contours', None))


def highlightObjectsInImage(objectContours, imageToHighlight):
    for (index, contour) in enumerate(objectContours):
        ((x, y), _) = cv2.minEnclosingCircle(contour)
        cv2.drawContours(imageToHighlight, [contour], -1, (0, 255, 0), 1)
        cv2.putText(imageToHighlight,
                    "#{}".format(index + 1),
                    (int(x) - 10, int(y) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)


runObjectDetection(cv2.imread('../images/all.jpg'))
