__author__ = "Haim Adrian"

from functions import *


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


def doFilter(img, imageFilter):
    img = img.copy()

    # Do my filter
    imgFiltered = my_imfilter(img, imageFilter, ZERO_PADDING, np.uint8)
    imgFilteredFloat64 = my_imfilter(img, imageFilter, ZERO_PADDING, np.float64)

    # Do cv2 filter
    imgFilteredUsingCv2 = cv2.filter2D(img, cv2.CV_8U, imageFilter, borderType=cv2.BORDER_CONSTANT)
    imgFilteredUsingCv2Float64 = cv2.filter2D(img, cv2.CV_64F, imageFilter, borderType=cv2.BORDER_CONSTANT)

    return img, imgFiltered, imgFilteredFloat64, imgFilteredUsingCv2, imgFilteredUsingCv2Float64


def testOnSmallTensor():
    imageTest = np.array([i * 2 for i in range(1, 26)], dtype=np.uint8).reshape((5, 5))

    laplacianFilterTest = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])

    imgFilteredZeroPadding = my_imfilter(imageTest, laplacianFilterTest, ZERO_PADDING, np.uint8)
    imgFilteredExtendedPadding = my_imfilter(imageTest, laplacianFilterTest, EXTENDED_PADDING, np.uint8)
    imgFilteredUsingCv2ZeroPadding = cv2.filter2D(imageTest, cv2.CV_8U, laplacianFilterTest,
                                                  borderType=cv2.BORDER_CONSTANT)
    imgFilteredUsingCv2ExtendedPadding = cv2.filter2D(imageTest, cv2.CV_8U, laplacianFilterTest,
                                                      borderType=cv2.BORDER_REPLICATE)

    print('Image:')
    print(imageTest)
    print('MyFilter, Zero Padding:')
    print(imgFilteredZeroPadding)
    print('CV2Filter, Zero Padding:')
    print(imgFilteredUsingCv2ZeroPadding)
    print('MyFilter, Extended Padding:')
    print(imgFilteredExtendedPadding)
    print('CV2Filter, Extended Padding:')
    print(imgFilteredUsingCv2ExtendedPadding)
    print()
    print('Max difference (Zero): ', np.max(np.abs(np.int32(imgFilteredZeroPadding) -
                                                   np.int32(imgFilteredUsingCv2ZeroPadding))))
    print('Max difference (Extended): ', np.max(np.abs(np.int32(imgFilteredExtendedPadding) -
                                                       np.int32(imgFilteredUsingCv2ExtendedPadding))))


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
        doFilter(trunksImage, laplacianFilter)
    showImageUsingMatPlotLib('Laplacian 3x3', True, (image, 231, 'Original'),
                             (imageFiltered, 232, 'My Filter (uint8)'),
                             (imageFilteredUsingCv2, 233, 'CV2 Filter (uint8)'),
                             (normalizeImage(imageFilteredFloat64), 235, 'My Filter (float64)'),
                             (normalizeImage(imageFilteredUsingCv2Float64), 236, 'CV2 Filter (float64)'))

    maxDifference = np.max(np.abs(np.int32(imageFiltered) - np.int32(imageFilteredUsingCv2)))
    print("Laplacian 3x3")
    print('MyFiltered shape: ', imageFiltered.shape, 'CV2Filtered shape: ', imageFilteredUsingCv2.shape)
    print('Max difference is: ', maxDifference)

    image, imageFiltered, imageFilteredFloat64, imageFilteredUsingCv2, imageFilteredUsingCv2Float64 = \
        doFilter(trunksImage, laplacianFilter5)
    showImageUsingMatPlotLib('Laplacian 5x5', True, (image, 231, 'Original'),
                             (imageFiltered, 232, 'My Filter (uint8, Laplacian 5x5)'),
                             (imageFilteredUsingCv2, 233, 'CV2 Filter (uint8, Laplacian 5x5)'),
                             (normalizeImage(imageFilteredFloat64), 235, 'My Filter (float64, Laplacian 5x5)'),
                             (normalizeImage(imageFilteredUsingCv2Float64), 236, 'CV2 Filter (float64, Laplacian 5x5)'))

    maxDifference = np.max(np.abs(np.int32(imageFiltered) - np.int32(imageFilteredUsingCv2)))
    print()
    print("Laplacian 5x5")
    print('MyFiltered shape: ', imageFiltered.shape, 'CV2Filtered shape: ', imageFilteredUsingCv2.shape)
    print('Max difference is: ', maxDifference)
else:
    testOnSmallTensor()
