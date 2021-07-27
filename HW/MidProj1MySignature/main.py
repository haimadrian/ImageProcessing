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


def testOnSmallTensor():
    # 00000001, 00000010, 00000011
    # 00000100, 00010111, 00000110
    # 00000111, 00001111, 11111111
    imageTest = np.array([[1, 2,  3],
                          [4, 23, 6],
                          [7, 15, 255]], dtype=np.uint8)

    # 11100000, 10100000, 00000011
    # 11100000, 10100000, 00000011
    # 11100000, 10100000, 00000011
    signatureTest = np.array([[224, 160, 3],
                              [224, 160, 3],
                              [224, 160, 3]], dtype=np.uint8)

    signedImage = my_sign(imageTest, signatureTest)
    print('Image:')
    print(np.vectorize(np.binary_repr)(imageTest, width=8))
    print('Signature:')
    print(np.vectorize(np.binary_repr)(signatureTest, width=8))
    print('Signed Image:')
    print(np.vectorize(np.binary_repr)(signedImage, width=8))


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
else:
    testOnSmallTensor()

