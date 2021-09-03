__author__ = "Haim Adrian"

import numpy as np
import cv2
from matplotlib import pyplot as plt


DEFAULT_COLOR_MAP = "gray"


def validateImage(image, isGrayscale=True):
    """
    Make sure a specified object is an ndarray and it has two dimensions only.

    If it has less than 2, or more than 3 dimensions, the result is None. Otherwise, if it has
    3 dimensions, we assume it is a BGR image and convert it to grayscale (2D image)

    :param image: (numpy.ndarray)
        The image to validate
    :param isGrayscale: (bool)
        Whether the image should be in grayscale mode (2D array) or not
    :return: (numpy.ndarray)
        The image after validation. (2D Array or None in case specified argument was not an image)
    """
    if not isinstance(image, np.ndarray):
        print("ERROR - validateImage: Not a tensor. Was: ", image.__class__)
        return None

    if image.ndim < 2 or image.ndim > 3:
        print("ERROR - validateImage: Illegal tensor dimension. "
              "Image can be 2D or 3D array only. Was: ",
              image.ndim)
        return None

    if isGrayscale and image.ndim == 3:
        print("INFO - validateImage: Input image not in grayscale mode. "
              "Converting it to grayscale.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def showImageUsingMatPlotLib(caption, isGray=True, *images):
    """
    Plots an image, titled with custom title, using MatPlotLib.

    if title is not a string, or image is not an image, nothing will happen.

    For example: showImageUsingMatPlotLib('Caption',
                                          True,
                                          (image1, 121, 'First'),
                                          (image2, 122, 'Second'))
    (The number is the subplot, first digit is the amount of rows, so we used one row.
    Second digit is the amount of columns, so we used 2 columns and third digit is the ordinal
    value of the image)

    :param caption: (str)
        Title to use for the dialog
    :param images: (tuple[np.ndarray, int, str])
        The images to display, as tuples where first is the image, and second is its location
    :param isGray: Whether to display the image in grayscale mode, or not (colored)
    :return: None
    """
    if caption.__class__ != "".__class__:
        print("ERROR - showImageUsingMatPlotLib: Caption type must be a string. Was: ",
              caption.__class__)
        return None

    if images is None:
        print("ERROR - showImageUsingMatPlotLib: Missing images.")
        return None

    figure = plt.figure(caption)
    for currImageInfo in images:
        if len(currImageInfo) != 3:
            print("WARN - showImageUsingMatPlotLib: Input image details is not a tuple of 3 "
                  "values. Was: ",
                  len(currImageInfo))
            continue

        image = validateImage(currImageInfo[0], isGray)
        if image is None:
            print("WARN - showImageUsingMatPlotLib: Input image is missing.")
            continue

        plt.subplot(currImageInfo[1])
        plt.axis('off')
        plt.title(currImageInfo[2])

        if isGray:
            plt.imshow(image, cmap=DEFAULT_COLOR_MAP)
        else:
            # Mat Plot Lib uses RGB and not BGR.
            plt.imshow(image[:, :, ::-1])

    figure.tight_layout()
    plt.show()


def resizeImage(image, newDimension):
    """
    Resize an image to a new dimension.

    If the specified image is not an image, result will be None.
    In addition, if the specified newDimension differs from the shape of the specified image, e.g.
    image is 2D array and new dimension is of a 3D array, the result will be None.

    :param image: The image to resize
    :param newDimension: The new dimension to resize the image to
    :return: The resized image, or None in case of any input fault
    """
    image = validateImage(image)
    if image is None:
        print("ERROR - resizeImage: Image is missing.")
        return None

    if not isinstance(newDimension, tuple) or len(newDimension) != image.ndim:
        print("ERROR - resizeImage: Specified dimension is illegal. Dimension=",
              len(newDimension),
              ", ImageDimension=",
              image.ndim)
        return None

    return cv2.resize(image, newDimension)


def doImageContrastAdjustment(image, gamma=1.0):
    """
    Pre-Processing step.
    Receives an image, and and apply contrast adjustment (Gamma Correction or Histogram
    Equalization) to make it easier for us detecting edges later.
    The problem is that objects might have a color similar to the background, which causes
    problems while detecting edges. Hence we adjust image contrast.

    :param image: Image to adjust
    :param gamma: Gamma value. gamma < 1 will shift the image towards the darker end of the
    spectrum while gamma > 1 will make the image appear lighter. gamma = 1 means no affect.
    :return: Adjusted image
    """
    image = validateImage(image)
    if image is None:
        print("ERROR - doImageContrastAdjustment: Image is missing.")
        return None

    # Calculate the LUT once only and keep it as attribute of this function. (To fake a static var)
    if not hasattr(doImageContrastAdjustment, "LUT"):
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        # Note that np.arange is faster than for loop over built-in range.
        invGamma = 1.0 / gamma
        doImageContrastAdjustment.LUT = \
            np.array(((np.arange(256) / 255.0) ** invGamma) * 255.0, dtype=np.uint8)

    # Now apply gamma correction using the lookup table
    return cv2.LUT(image, doImageContrastAdjustment.LUT)


def doGradientEdgeDetection(image):
    """
    Receives an image and return it after executing gradient edge detection.

    We calculate dy and dx (gradients of axis) and then returning a normalized image
    where gray level of each cell is calculated based on sqrt(dy^2 + dx^2) formula.

    :param image: A gray-scale image to apply gradient edge detection on
    :return: The gradient edge detection result. (Image, not filtered by any threshold)
    """
    image = validateImage(image)
    if image is None:
        print("ERROR - doGradientEdgeDetection: Image is missing.")
        return None

    # img is a 2D array, so the result of gradient is two arrays ordered by axis
    dy, dx = np.gradient(np.float64(image))

    # sqrt(dy^2 + dx^2)
    newImage = (dy ** 2 + dx ** 2) ** 0.5

    # Normalize the image to make sure we are within uint8 boundaries
    newImage = newImage - np.min(newImage)
    newImage = np.round(newImage * 255 / np.max(newImage))

    return np.uint8(newImage)
