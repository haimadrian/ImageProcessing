__author__ = "Haim Adrian"

from functions import *


ZERO_PADDING = 0
EXTENDED_PADDING = 1


# This method reads our images from disk, and resize them to 400x400.
def readAndResizeImages(*imageNames):
    imagesSize = (400, 400)
    result = []

    # Read and resize all images
    for currImageName in imageNames:
        img = readImage(currImageName)
        img = resizeImage(img, imagesSize)
        result.append(img)

    return result[0] if len(result) == 1 else result


def doFilterAndShow(img, imageFilter):
    # Do my filter
    imgFiltered = my_imfilter(img, imageFilter, ZERO_PADDING, np.uint8)
    imgFilteredFloat64 = my_imfilter(img, imageFilter, ZERO_PADDING, np.float64)

    # Do cv2 filter
    imgFilteredUsingCv2 = cv2.filter2D(img, cv2.CV_8U, imageFilter)
    imgFilteredUsingCv2Float64 = cv2.filter2D(img, cv2.CV_64F, imageFilter)

    return img, imgFiltered, imgFilteredFloat64, imgFilteredUsingCv2, imgFilteredUsingCv2Float64


if __name__ == '__main__':
    # Read and resize image
    trunksImage = readAndResizeImages('trunks.jpg')

    # Create the laplacian filter
    laplacianFilter = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    laplacianFilter5 = np.array([[0,   0, -1,  0,  0],
                                 [0,  -1, -2, -1,  0],
                                 [-1, -2, 17, -2, -1],
                                 [0,  -1, -2, -1,  0],
                                 [0,   0, -1,  0,  0]])

    image, imageFiltered, imageFilteredFloat64, imageFilteredUsingCv2, imageFilteredUsingCv2Float64 = \
        doFilterAndShow(trunksImage.copy(), laplacianFilter)
    showImageUsingMatPlotLib('Laplacian 3x3', True, (image, 231, 'Original'),
                             (np.uint8(imageFiltered), 232, 'My Filter (uint8)'),
                             (np.uint8(imageFilteredUsingCv2), 233, 'CV2 Filter (uint8)'),
                             (normalizeImage(imageFilteredFloat64), 235, 'My Filter (float64)'),
                             (normalizeImage(imageFilteredUsingCv2Float64), 236, 'CV2 Filter (float64)'))

    image, imageFiltered, imageFilteredFloat64, imageFilteredUsingCv2, imageFilteredUsingCv2Float64 = \
        doFilterAndShow(trunksImage.copy(), laplacianFilter5)
    showImageUsingMatPlotLib('Laplacian 5x5', True, (image, 131, 'Original'),
                             (normalizeImage(imageFilteredFloat64), 132, 'My Filter (float64, Laplacian 5x5)'),
                             (normalizeImage(imageFilteredUsingCv2Float64), 133, 'CV2 Filter (float64, Laplacian 5x5)'))
