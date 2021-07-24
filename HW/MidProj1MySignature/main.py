__author__ = "Haim Adrian"

from functions import *


# This method reads our images from disk, and resize them to 400x400.
def readAndResizeImages(*imageNames):
    imagesSize = (400, 400)
    result = []

    # Read and resize all images
    for currImageName in imageNames:
        image = readImage(currImageName)
        image = resizeImage(image, imagesSize)
        result.append(image)

    return result


if __name__ == '__main__':
    # Read and resize images
    signature, gotenksImage = readAndResizeImages('guitar.jpg', 'gotenks.jpg')

    # Sign image
    gotenksImageSigned = my_sign(gotenksImage, signature)

    # Get a copy of the image and change its signature
    # Here we clear the third bit from right
    gotenksFake = gotenksImageSigned.copy()
    gotenksFake = np.uint8(gotenksFake & ~4)
    isImageSignatureValid = verifySignedImage(gotenksFake, signature)

    showImageUsingMatPlotLib('My Signature', True, (gotenksImage, 221, 'Original'), (signature, 222, 'Signature'),
                             (gotenksImageSigned, 223, 'Signed Image'), (gotenksFake, 224, 'Is valid signature? (' +
                                                                         str(isImageSignatureValid) + ')'))

