__author__ = "Haim Adrian"

import numpy as np
import cv2
from matplotlib import pyplot as plt


ZERO_PADDING = 0
EXTENDED_PADDING = 1


def my_imfilter(image, twoDFilter, paddingType=ZERO_PADDING, dtype=np.uint8):
    """
    Convolve a filter to some image

    If the specified argument is not an image, the result will be None. In addition,
    if the dimension of the images differ, the result will be None. Note that the shape of the
    filter must be odd. e.g. 3x3, 5x5, 7x7, etc. Otherwise the result will be None.

    :param image: (numpy.ndarray)
        The image to apply a filter on
    :param twoDFilter: (numpy.ndarray)
        A filter to convolve to the specified image
    :param paddingType: (int)
        0 for Zero Padding, 1 for Extended Padding.
    :param dtype: Type ot output
    :return:
        The signed image
    """
    image = validateImage(image)
    twoDFilter = validateFilter(twoDFilter)
    if image is None or twoDFilter is None:
        print("ERROR - my_imfilter: Missing image or filter.")
        return None

    if paddingType not in [0, 1]:
        print("WARN - my_imfilter: Unsupported padding type. only 0 or 1 are supported. Using 0.")
        paddingType = 0

    # Find the pad size. For example, we will get 1 in case shape is 3, or 2 in case shape is 5, and so on
    # Bonus 1: Filter can be rectangle, and not necessarily square
    # Bonus 2: Calculate the padding size based on the specified filter size
    padSizeVertical = (twoDFilter.shape[0] - 1) // 2
    padSizeHorizontal = (twoDFilter.shape[1] - 1) // 2
    paddedImage = myPadding(image, padSizeVertical, padSizeHorizontal, paddingType).astype(np.float64)

    twoDFilter = twoDFilter.astype(np.float64)
    result = np.zeros(image.shape).astype(np.float64)

    # Now apply the filter - Bonus 1: Matrix can be a rectangle and not necessarily square
    for i in range(padSizeVertical, paddedImage.shape[0] - padSizeVertical):
        for j in range(padSizeHorizontal, paddedImage.shape[1] - padSizeHorizontal):
            windowSum = np.sum(twoDFilter * paddedImage[i - padSizeVertical: i + padSizeVertical + 1,
                                                        j - padSizeHorizontal: j + padSizeHorizontal + 1])

            result[i - padSizeVertical, j - padSizeHorizontal] = windowSum

    # Make sure we do not exceed bounds of the specified data type (i=int, u=unsigned)
    if np.issubdtype(dtype, np.integer):
        maxValue = np.iinfo(dtype).max
        minValue = np.iinfo(dtype).min
    else:
        maxValue = np.finfo(dtype).max
        minValue = np.finfo(dtype).min

    # Make sure we do not exceed the bounds of requested data type, by clipping the matrix.
    # Clip means that any value below minimum will be modified to the minimum, and the same with maximum.
    result = np.clip(result, minValue, maxValue)

    return result.astype(dtype=dtype)


def myPadding(matrix, padSizeVertical=1, padSizeHorizontal=1, paddingType=0):
    """
    Do a padding with custom size and custom padding type (Zero or Extended) to a matrix.

    If matrix dimension is not a 2D array, result will be None. If padSize is negative, we'll do nothing.

    :param matrix: (numpy.ndarray)
        The matrix to return padded
    :param padSizeVertical: (int)
        Size of the padding. (Top, Bottom)
    :param padSizeHorizontal: (int)
        Size of the padding. (Left, Right)
    :param paddingType: (int)
        0 for Zero Padding, 1 for Extended Padding.
    :return:
        The matrix with padding
    """
    if paddingType not in [0, 1]:
        print("WARN - myPadding: Unsupported padding type. only 0 or 1 are supported. Using 0.")
        paddingType = 0

    if paddingType == 0:
        return myZeroPadding2D(matrix, padSizeVertical, padSizeHorizontal)

    return myExtendedPadding2D(matrix, padSizeVertical, padSizeHorizontal)


def myZeroPadding2D(matrix, padSizeVertical=1, padSizeHorizontal=1):
    """
    Do zero padding to a matrix.

    If matrix dimension is not a 2D array, result will be None. If padSize is negative, we'll do nothing.

    :param matrix: (numpy.ndarray)
        The matrix to return padded with zeroes
    :param padSizeVertical: (int)
        Size of the padding. (Top, Bottom)
    :param padSizeHorizontal: (int)
        Size of the padding. (Left, Right)
    :return:
        The matrix with padding
    """
    if matrix is None:
        print("ERROR - myZeroPadding2D: Missing matrix.")
        return None

    if not isinstance(matrix, np.ndarray):
        print("ERROR - myZeroPadding2D: Not a tensor. Was: ", matrix.__class__)
        return None

    if matrix.ndim != 2:
        print("ERROR - myZeroPadding2D: Not a 2D array. Was: ", matrix.ndim)
        return None

    if padSizeVertical < 0 or padSizeHorizontal < 0:
        print("ERROR - myZeroPadding2D: Padding size was negative.")
        return None

    # Create a zero matrix, with enough space to contain the specified matrix and padding
    dim = matrix.shape
    paddedMatrix = np.zeros((dim[0] + (padSizeVertical * 2), dim[1] + (padSizeHorizontal * 2)), dtype=matrix.dtype)

    # Copy the specified matrix and return the result
    paddedMatrix[padSizeVertical: padSizeVertical + dim[0], padSizeHorizontal: padSizeHorizontal + dim[1]] = matrix
    return paddedMatrix


def myExtendedPadding2D(matrix, padSizeVertical=1, padSizeHorizontal=1):
    """
    Do extended padding to a matrix.

    If matrix dimension is not a 2D array, result will be None. If padSize is negative, we'll do nothing.

    :param matrix: (numpy.ndarray)
        The matrix to return padded with extended. (Replicating the border of the specified matrix)
    :param padSizeVertical: (int)
        Size of the padding. (Top, Bottom)
    :param padSizeHorizontal: (int)
        Size of the padding. (Left, Right)
    :return:
        The matrix with padding
    """
    paddedMatrix = myZeroPadding2D(matrix, padSizeVertical, padSizeHorizontal)

    # It means something in the input was wrong
    if paddedMatrix is None:
        return None

    matrixShape = matrix.shape

    # First we fill the top and bottom parts, leaving the left and right top corners black (zero)
    # Second, we depend on the size and top and bottom padding in order to fill the whole left and right
    # parts, including the corners.

    # Duplicate top row of matrix, padSize times. Result of tile should have padSize rows, and each row
    # is the same origin row, without repeating it.
    paddedMatrix[:padSizeVertical, padSizeHorizontal: padSizeHorizontal + matrixShape[1]] = \
        np.tile(matrix[0, :], (padSizeVertical, 1))

    # Duplicate bottom row of matrix
    paddedMatrix[padSizeVertical + matrixShape[0]:, padSizeHorizontal: padSizeHorizontal + matrixShape[1]] = \
        np.tile(matrix[matrixShape[0] - 1, :], (padSizeVertical, 1))

    # Use tile to duplicate the column at padSize, padSize times, and then transpose it because the column
    # was duplicated into several rows, but we want to use it for filling in the columns.
    # Left
    paddedMatrix[:, :padSizeHorizontal] = \
        np.tile(paddedMatrix[:, padSizeHorizontal], (padSizeHorizontal, 1)).transpose()

    # Right
    paddedMatrix[:, padSizeHorizontal + matrixShape[1]:] = \
        np.tile(paddedMatrix[:, paddedMatrix.shape[1] - padSizeHorizontal - 1], (padSizeHorizontal, 1)).transpose()

    return paddedMatrix


def normalizeImage(image):
    """
    Normalizes an image to uint8 data type, such that we make sure the values in the image are between 0 to 255.
    :param image: (numpy.ndarray)
        Image to normalize
    :return:
        The image, normalized. (uint8 where all values are between 0 to 255)
    """
    if image is None:
        print("ERROR - normalizeImage: Missing image.")
        return None

    image = image - np.min(image)
    image = np.round(image * 255 / np.max(image))
    return np.uint8(image)


def readImage(imageName, flags=cv2.IMREAD_GRAYSCALE):
    """
    Reads an image from disk into memory and return it as numpy.ndarray.

    If the specified reference is not a string, result will be None.

    :param imageName: (str)
        Name (or full path) of the image to read
    :param flags: (int)
        Flags to use for reading an image. Default value is 0, to read image in grayscale mode.
        For colored image, use cv2.IMREAD_COLOR
    :return: (numpy.ndarray)
        The image loaded as ndarray with dimensions as specified by the flags parameter
    """
    if imageName.__class__ != "".__class__:
        print("ERROR - readImage: Input type must be a string (image name). Was: ", imageName.__class__)
        return None

    # cv2 loads an image in BGR format. Here we have a default value used to load it in grayscale mode
    image = cv2.imread(imageName, flags)
    return image


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
        print("ERROR - validateImage: Illegal tensor dimension. Image can be 2D or 3D array only. Was: ", image.ndim)
        return None

    if isGrayscale and image.ndim == 3:
        print("INFO - validateImage: Input image not in grayscale mode. Converting it to grayscale.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def validateFilter(twoDFilter):
    """
    Make sure a specified object is an ndarray and it has two dimensions only.

    If dimension differs from 2, the result is None. In addition, if filter is not
    a square, or its shape is not odd, the result will be None.

    :param twoDFilter: (numpy.ndarray)
        The filter to validate
    :return: (numpy.ndarray)
        The filter after validation. (2D Array or None in case specified argument was not a filter)
    """
    if not isinstance(twoDFilter, np.ndarray):
        print("ERROR - validateFilter: Not a tensor. Was: ", twoDFilter.__class__)
        return None

    if twoDFilter.ndim != 2:
        print("ERROR - validateFilter: Illegal tensor dimension. Filter can be 2D array only. Was: ", twoDFilter.ndim)
        return None

    twoDFilterShape = twoDFilter.shape
    if (twoDFilterShape[0] != twoDFilterShape[1]) or (twoDFilterShape[0] % 2 == 0):
        print("ERROR - validateFilter: Filter is not supported. Only two square odd dimensional filters. "
              "e.g. 3x3 or 5x5, etc.")
        return None

    return twoDFilter


def showImageUsingMatPlotLib(caption, isGray=True, *images):
    """
    Plots an image, titled with custom title, using MatPlotLib.

    if title is not a string, or image is not an image, nothing will happen.

    For example: showImageUsingMatPlotLib('Caption', True, (image1, 121, 'First'), (image2, 122, 'Second')) (The number
    is the subplot, first digit is the amount of rows, so we used one row. Second digit is the amount of columns, so we
    used 2 columns and third digit is the ordinal value of the image)

    :param caption: (str)
        Title to use for the dialog
    :param images: (tuple[np.ndarray, int, str])
        The images to display, as tuples where first is the image, and second is its location (subplot)
    :param isGray: Whether to display the image in grayscale mode, or not (colored)
    :return: None
    """
    if caption.__class__ != "".__class__:
        print("ERROR - showImageUsingMatPlotLib: Caption type must be a string. Was: ", caption.__class__)
        return None

    if images is None:
        print("ERROR - showImageUsingMatPlotLib: Missing images.")
        return None

    figure = plt.figure(caption)
    for currImageInfo in images:
        if len(currImageInfo) != 3:
            print("ERROR - showImageUsingMatPlotLib: Input image details is not a tuple of 3 values. Was: ",
                  len(currImageInfo))
            continue

        image = validateImage(currImageInfo[0], isGray)
        if image is None:
            print("ERROR - showImageUsingMatPlotLib: Input image is missing.")
            continue

        plt.subplot(currImageInfo[1])
        plt.axis('off')
        plt.title(currImageInfo[2])

        if isGray:
            plt.imshow(image, cmap="gray")
        else:
            # Mat Plot Lib uses RGB and not BGR.
            plt.imshow(image[:, :, ::-1])

    figure.tight_layout()
    plt.show()


def resizeImage(image, newDimension):
    """
    Resize an image to a new dimension.

    If the specified image is not an image, result will be None.
    In addition, if the specified newDimension differs from the shape of the specified image, e.g. image is 2D array
    and new dimension is of a 3D array, the result will be None.
    :param image: The image to resize
    :param newDimension: The new dimension to resize the image to
    :return: The resized image, or None in case of any input fault
    """
    image = validateImage(image)
    if image is None:
        print("ERROR - resizeImage: Image is missing.")
        return None

    if not isinstance(newDimension, tuple) or len(newDimension) != image.ndim:
        print("ERROR - resizeImage: Specified dimension is illegal. Dimension=", len(newDimension), ", ImageDimension=",
              image.ndim)
        return None

    return cv2.resize(image, newDimension)
