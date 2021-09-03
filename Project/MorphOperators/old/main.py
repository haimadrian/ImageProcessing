__author__ = "Haim Adrian"


from functions import *
import cv2
import numpy as np


MORPH_ITERATIONS_COUNT = 2
THRESH_MIN = 13
THRESH_MAX = 255
MIN_CONTOURS_IN_OBJECT_THRESH = 5
IMAGE_SHAPE = (400, 400)
GAUSS_BLUR_MASK_SHAPE = (3, 3)
MORPH_MASK_SHAPE = (3, 3)
COIN_AREA = (200, 450)
BILL_AREA = (5500, 10000)
OBJECT_MARKER_THICKNESS = 2
OBJECT_MARKER_COLOR = 255


def test():
    image = cv2.imread('../money_with_other.jpg', cv2.IMREAD_GRAYSCALE)
    image = resizeImage(image, IMAGE_SHAPE)

    # 1. Pre-Processing: Image Contrast Adjustment is done so we can ease edge detection
    #    by gradient, when object edges color is similar to the background color.
    contrastAdjustmentImage = doImageContrastAdjustment(image, gamma=1.3)

    # Use sigmaX = 0, so cv2 will depend on mask size for calculating the standard deviation.
    sigmaX = 0
    gray_blur = cv2.GaussianBlur(contrastAdjustmentImage, GAUSS_BLUR_MASK_SHAPE, sigmaX)

    # Perform gradient on the image, so we will transform the image into image of contours,
    # which makes it easier for us to concentrate on objects in an image.
    gradient = doGradientEdgeDetection(gray_blur)

    # After the gradient, we get image with contours. Background is black and contours in white.
    # Use threshold to remove non-interesting contours, and leave only those we are interested in,
    # those are the objects.
    _, imageBinary = cv2.threshold(gradient, THRESH_MIN, THRESH_MAX, cv2.THRESH_BINARY)

    kernel = np.ones(MORPH_MASK_SHAPE, np.uint8)

    # Use Closing, so first we will use Dilation, to fill in the shapes, and then Erosion,
    # to reduce the shapes to their original size. This way we try to fill in little holes inside
    # objects.
    # imageDilated = cv2.dilate(imageBinary, kernel, iterations=4)
    # imageEroded = cv2.erode(imageDilated, kernel, iterations=4)
    imageClosing = \
        cv2.morphologyEx(imageBinary, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_COUNT)

    cont_img = imageClosing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imageMarks = image.copy()
    highlightObjectsInImage(contours, imageMarks)

    showImageUsingMatPlotLib('My Signature',
                             True,
                             (image, 231, 'Original'),
                             (gray_blur, 232, 'Blur'),
                             (gradient, 233, 'Gradient'),
                             (imageBinary, 234, 'Binary'),
                             (imageClosing, 235, 'Closing'),
                             (imageMarks, 236, 'Contours'))


def highlightObjectsInImage(objectContours, imageToHighlight):
    for contour in objectContours:
        if len(contour) < MIN_CONTOURS_IN_OBJECT_THRESH:
            continue

        area = int(cv2.contourArea(contour))
        if COIN_AREA[0] <= area <= COIN_AREA[1] or BILL_AREA[0] <= area <= BILL_AREA[1]:
            highlightObjectInImage(contour, area, imageToHighlight)


def highlightObjectInImage(contour, area, image):
    if area >= BILL_AREA[0]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, OBJECT_MARKER_COLOR, thickness=OBJECT_MARKER_THICKNESS)
    else:
        # ellipse is a RotatedRect, holds the center, size and angle of a rotated rect, in
        # which there is our ellipse.
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, color=OBJECT_MARKER_COLOR, thickness=OBJECT_MARKER_THICKNESS)


if __name__ == '__main__':
    test()
