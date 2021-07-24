__author__ = "Haim Adrian"

import numpy as np
import cv2
from matplotlib import pyplot as plt


def my_sign(imageToSign, signatureImage):
    """
    Signs an image using the specified signature.

    If the specified arguments are not images, the result will be None. In addition,
    if the dimension of the images differ, the result will be None. It is your responsibility
    to ensure both images have the same dimension and same shape.
    Note that values in the ndarray's must be of type uint8! Otherwise result will be None.

    :param imageToSign: (numpy.ndarray)
        The image to sign with the specified signature
    :param signatureImage: (numpy.ndarray)
        The signature to sign the specified image with
    :return: The signed image
    """
    imageToSign = validateImage(imageToSign)
    signatureImage = validateImage(signatureImage)

    if imageToSign is None or signatureImage is None:
        print("ERROR - my_sign: Image to sign or signature are missing.")
        return None

    if imageToSign.shape != signatureImage.shape:
        print("ERROR - my_sign: Image to sign has a different shape than signature's shape. Image=",
              imageToSign.shape, ", Signature=", signatureImage.shape)
        return None

    if imageToSign[0, 0].__class__ != np.uint8:
        print("WARN - my_sign: Image's tensor data type must be uint8. Converting it. Was: ",
              imageToSign[0, 0].__class__)
        imageToSign = np.uint8(imageToSign)

    imageToSign = imageToSign.copy()

    # Clear the 3 LSB of the image to sign. (7 = 00000111b, ~7 = 11111000b)
    imageToSign = imageToSign & ~7

    # Set the 3 MSB from signature to the 3 LSB of image to sign. (224 = 11100000b)
    imageToSign = imageToSign | ((signatureImage & 224) >> 5)

    return np.uint8(imageToSign)


def verifySignedImage(imageToVerify, signatureImage):
    """
    Tests if a given image is valid (signed by the specified signature) or not.

    If the specified arguments are not images, the result will be None. In addition,
    if the dimension images differ, the result will be None. It is your responsibility
    to ensure both images have the same dimension and same shape.
    Note that values in the ndarray's must be of type uint8! Otherwise result will be None.

    :param imageToVerify: (numpy.ndarray)
        The image to verify
    :param signatureImage: (numpy.ndarray)
        The signature we expect to find in the specified image
    :return: Whether the image is signed using the specified signature or not.
    """
    imageToVerify = validateImage(imageToVerify)
    signatureImage = validateImage(signatureImage)

    if imageToVerify is None or signatureImage is None:
        print("ERROR - verifySignedImage: Image to verify or signature are missing.")
        return None

    if imageToVerify.shape != signatureImage.shape:
        print("INFO - verifySignedImage: Image to verify has a different shape than signature's shape. Image=",
              imageToVerify.shape, ", Signature=", signatureImage.shape)
        return False

    # Prepare 3 LSB from 3 MSB of signature. (224 = 11100000b)
    expectation = (signatureImage & 224) >> 5

    # Now make sure the 3 LSB of the image equal to the 3 LSB we have prepared
    return ((imageToVerify & expectation) == expectation).all()


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
