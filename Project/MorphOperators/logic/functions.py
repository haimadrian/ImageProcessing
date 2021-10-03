__author__ = "Haim Adrian"

import numpy as np
import cv2
from matplotlib import pyplot as plt

DEFAULT_COLOR_MAP = "gray"


def log(consumer, *message):
    text = ' '.join(message)
    if consumer is not None:
        consumer(text)
    else:
        print(text)


def validateImage(image, isGrayscale=True, consoleConsumer=None):
    """
    Make sure a specified object is an ndarray and it has two dimensions only.

    If it has less than 2, or more than 3 dimensions, the result is None. Otherwise, if it has
    3 dimensions, we assume it is a BGR image and convert it to grayscale (2D image)

    :param image: (numpy.ndarray)
        The image to validate
    :param isGrayscale: (bool)
        Whether the image should be in grayscale mode (2D array) or not
    :param consoleConsumer: (function)
        Used to print messages at the UI layer
    :return: (numpy.ndarray)
        The image after validation. (2D Array or None in case specified argument was not an image)
    """
    if not isinstance(image, np.ndarray):
        log(consoleConsumer, 'ERROR - validateImage: Not a tensor. Was:', str(image.__class__))
        return None

    if image.ndim < 2 or image.ndim > 3:
        log(consoleConsumer,
            'ERROR - validateImage: Illegal tensor dimension. Image can be 2D or 3D array only. Was:',
            str(image.ndim))
        return None

    if isGrayscale and image.ndim == 3:
        log(consoleConsumer,
            'INFO - validateImage: Input image not in grayscale mode. Converting it to grayscale.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def showImageUsingMatPlotLib(caption, isGray=True, consoleConsumer=None, *images):
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
    :param isGray: Whether to display the image in grayscale mode, or not (colored)
    :param consoleConsumer: (function)
        Used to print messages at the UI layer
    :param images: (tuple[np.ndarray, int, str])
        The images to display, as tuples where first is the image, and second is its location
    :return: None
    """
    if caption.__class__ != "".__class__:
        log(consoleConsumer,
            'ERROR - showImageUsingMatPlotLib: Caption type must be a string. Was:',
            str(caption.__class__))
        return None

    if images is None:
        log(consoleConsumer, 'ERROR - showImageUsingMatPlotLib: Missing images.')
        return None

    figure = plt.figure(caption)
    for currImageInfo in images:
        if len(currImageInfo) != 4:
            log(consoleConsumer,
                'WARN - showImageUsingMatPlotLib: Input image details is not a tuple of 3 values. Was: ',
                str(len(currImageInfo)))
            continue

        image = validateImage(currImageInfo[0], False)
        if image is None:
            log(consoleConsumer, 'WARN - showImageUsingMatPlotLib: Input image is missing.')
            continue

        plt.subplot(currImageInfo[1])
        plt.axis('off')
        plt.title(currImageInfo[2])

        if currImageInfo[3] is None:
            # Mat Plot Lib uses RGB and not BGR.
            plt.imshow(image[:, :, ::-1])
        else:
            plt.imshow(image, cmap=currImageInfo[3])

    figure.tight_layout()
    plt.show()


def resizeImage(image, newDimension, consoleConsumer=None):
    """
    Resize an image to a new dimension.

    If the specified image is not an image, result will be None.
    In addition, if the specified newDimension differs from the shape of the specified image, e.g.
    image is 2D array and new dimension is of a 3D array, the result will be None.

    :param image: The image to resize
    :param newDimension: The new dimension to resize the image to
    :param consoleConsumer: Used to print messages at the UI layer
    :return: The resized image, or None in case of any input fault
    """
    image = validateImage(image)
    if image is None:
        log(consoleConsumer, 'ERROR - resizeImage: Image is missing.')
        return None

    if not isinstance(newDimension, tuple) or len(newDimension) != image.ndim:
        log(consoleConsumer,
            'ERROR - resizeImage: Specified dimension is illegal. Dimension=',
            str(len(newDimension)),
            ', ImageDimension=',
            str(image.ndim))
        return None

    return cv2.resize(image, newDimension)


def doImageContrastAdjustment(image, gamma=1.0):
    """
    Pre-Processing step.
    Receives an image and apply contrast adjustment (Gamma Correction or Histogram
    Equalization) to make it easier for us detecting edges later.
    The problem is that objects might have a color similar to the background, which causes
    problems while detecting edges. Hence we adjust image contrast.

    :param image: Image to adjust
    :param gamma: Gamma value. gamma &lt; 1 will shift the image towards the darker end of the
    spectrum while gamma &gt; 1 will make the image appear lighter. gamma = 1 means no affect.
    :return: Adjusted image
    """
    # Calculate the LUT once only and keep it as attribute of this function. (To fake a static var)
    if not hasattr(doImageContrastAdjustment, 'LUT') or doImageContrastAdjustment.GAMMA != gamma:
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        # Note that np.arange is faster than for loop over built-in range.
        doImageContrastAdjustment.GAMMA = gamma
        invGamma = 1.0 / gamma
        doImageContrastAdjustment.LUT = \
            np.array(((np.arange(256) / 255.0) ** invGamma) * 255.0, dtype=np.uint8)

    # Now apply gamma correction using the lookup table
    return cv2.LUT(image, doImageContrastAdjustment.LUT)


def doGradientEdgeDetection(image, consoleConsumer=None):
    """
    Receives an image and return it after executing gradient edge detection.

    We calculate dy and dx (gradients of axis) and then returning a normalized image
    where gray level of each cell is calculated based on sqrt(dy^2 + dx^2) formula.

    :param image: A gray-scale image to apply gradient edge detection on
    :param consoleConsumer: Used to print messages at the UI layer
    :return: The gradient edge detection result. (Image, not filtered by any threshold)
    """
    image = validateImage(image)
    if image is None:
        log(consoleConsumer, 'ERROR - doGradientEdgeDetection: Image is missing.')
        return None

    # img is a 2D array, so the result of gradient is two arrays ordered by axis
    dy, dx = np.gradient(np.float64(image))

    # sqrt(dy^2 + dx^2)
    newImage = (dy ** 2 + dx ** 2) ** 0.5

    # Normalize the image to make sure we are within uint8 boundaries
    newImage = newImage - np.min(newImage)
    newImage = np.round(newImage * 255 / np.max(newImage))

    return np.uint8(newImage)


def fullyPrintArray(array, consoleConsumer=None):
    if not isinstance(array, np.ndarray):
        log(consoleConsumer, 'ERROR - fullyPrintArray: Not a tensor. Was:', str(array.__class__))
        return None

    # Configure numpy such that it will not truncate array, and use a line width which is wide enough
    # for big arrays, without word-wrapping it.
    lastThreshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=np.inf, linewidth=800)
    print(array)
    np.set_printoptions(threshold=lastThreshold)

