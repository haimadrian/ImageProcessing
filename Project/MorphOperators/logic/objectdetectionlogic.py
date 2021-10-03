__author__ = "Haim Adrian"

from logic.functions import *
import cv2
import numpy as np
import imutils


def runObjectDetection(obj1Image, obj2Image, image, settings, consoleConsumer, progressConsumer):
    consoleConsumer('Running Object Detection using Morphological Operators...')

    # Make sure objects do not exceed image size
    obj1Image, obj2Image, image = validateImagesSize(obj1Image, obj2Image, image, settings)

    # Pre-Processing: Image Contrast Adjustment is done so we can ease edge detection
    # by gradient, when object edges color is similar to the background color.
    contrastAdjustmentObj1, contrastAdjustmentObj2, contrastAdjustmentImage = \
        doImagesContrastAdjustment(obj1Image, obj2Image, image, 1.3)

    # Blur image so we will reduce amount of sharp lines, to make it easier for us
    # focusing on objects as whole
    obj1Blur, obj2Blur, imageBlur = blurImages(contrastAdjustmentObj1,
                                               contrastAdjustmentObj2,
                                               contrastAdjustmentImage,
                                               settings)

    # Now convert images to gray, cause object detection is going to be as binary. (black/white)
    obj1Gray, obj2Gray, imageGray = convertImagesToGray(obj1Blur, obj2Blur, imageBlur)

    # Optional:
    # Perform gradient on the image, so we will transform the image into image of contours,
    # which makes it easier for us to concentrate on objects in an image.
    if settings.isUsingGradientEdgeDetector:
        obj1Gray, obj2Gray, imageGray = \
            doImagesGradientEdgeDetection(obj1Gray, obj2Gray, imageGray, consoleConsumer)

    # After the gradient, we get image with contours. Background is black and contours in white.
    # Use threshold to remove non-interesting contours, and leave only those we are interested in,
    # those are the objects.
    obj1Binary, obj2Binary, imgBinary = doImagesThresholding(obj1Gray, obj2Gray, imageGray, settings)

    # Use Closing, so first we will use Dilation, to fill in the shapes, and then Erosion, to reduce
    # the shapes to their original size. This way we try to fill in little holes inside objects.
    obj1Closing, obj2Closing, imgClosing = doImagesClosing(obj1Binary, obj2Binary, imgBinary, settings)

    # This method will iteratively try looking up for the objects in the given image, using
    # multiple sizes of the objects, depend on settings
    # Once we gather objects using findNonZero, we can filter them based on hit & miss results
    hitMissObj1, hitMissObj2, progress = \
        doHitMiss(imgClosing, obj1Closing, obj2Closing, settings, consoleConsumer, progressConsumer)

    # We use findContours to detect objects.
    # Then we make our array regular with the grab_contours method
    contours = cv2.findContours(imgClosing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # And now, the finale, highlight findings in the source image
    imgMarks = image.copy()
    highlightObjectsInImage(contours, hitMissObj1, hitMissObj2, imgMarks, settings, consoleConsumer,
                            progressConsumer, progress)

    objsImg = concatenateImages3D(obj1Image, obj2Image)
    objsBinaryImg = concatenateImages2D(obj1Binary, obj2Binary)
    objsClosingImg = concatenateImages2D(obj1Closing, obj2Closing)

    return objsImg, objsBinaryImg, objsClosingImg, imgBinary, imgClosing, hitMissObj1, hitMissObj2, imgMarks


def validateImageSize(image, settings):
    shape = image.shape
    if shape[0] > settings.imageShape[1] - 2:
        shape = (shape[0] - 2, shape[1], shape[2]) if len(shape) == 3 else (shape[0] - 2, shape[1])

    if shape[1] > settings.imageShape[0] - 2:
        shape = (shape[0], shape[1] - 2, shape[2]) if len(shape) == 3 else (shape[0], shape[1] - 2)

    if shape != image.shape:
        if len(shape) == 3:
            b, g, r = cv2.split(image)
            shape2D = (shape[0], shape[1])
            image = cv2.merge((cv2.resize(b, shape2D), cv2.resize(g, shape2D), cv2.resize(r, shape2D)))
        else:
            image = cv2.resize(image, shape)

    return image


def validateImagesSize(obj1Image, obj2Image, image, settings):
    obj1Image = validateImageSize(obj1Image, settings)
    obj2Image = validateImageSize(obj2Image, settings)
    image = validateImageSize(image, settings)
    return obj1Image, obj2Image, image


def doImagesContrastAdjustment(obj1Image, obj2Image, image, gamma):
    contrastAdjustmentObj1 = doImageContrastAdjustment(obj1Image, gamma)
    contrastAdjustmentObj2 = doImageContrastAdjustment(obj2Image, gamma)
    contrastAdjustmentImage = doImageContrastAdjustment(image, gamma)
    return contrastAdjustmentObj1, contrastAdjustmentObj2, contrastAdjustmentImage


def blurImages(obj1Image, obj2Image, image, settings):
    obj1Blur = cv2.medianBlur(obj1Image, settings.blurKernelSize)
    obj2Blur = cv2.medianBlur(obj2Image, settings.blurKernelSize)
    imageBlur = cv2.medianBlur(image, settings.blurKernelSize)
    return obj1Blur, obj2Blur, imageBlur


def convertImagesToGray(obj1Image, obj2Image, image):
    obj1Gray = cv2.cvtColor(obj1Image, cv2.COLOR_BGR2GRAY)
    obj2Gray = cv2.cvtColor(obj2Image, cv2.COLOR_BGR2GRAY)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return obj1Gray, obj2Gray, imageGray


def doImagesGradientEdgeDetection(obj1Image, obj2Image, image, consoleConsumer):
    obj1Gradient = doGradientEdgeDetection(obj1Image, consoleConsumer)
    obj2Gradient = doGradientEdgeDetection(obj2Image, consoleConsumer)
    imageGradient = doGradientEdgeDetection(image, consoleConsumer)
    return obj1Gradient, obj2Gradient, imageGradient


def doImagesThresholding(obj1Image, obj2Image, image, settings):
    thresholdingType = cv2.THRESH_BINARY

    if settings.isBrightBackground and not settings.isUsingGradientEdgeDetector:
        thresholdingType = cv2.THRESH_BINARY_INV

    _, obj1Binary = cv2.threshold(obj1Image, settings.threshold1, settings.threshold2, thresholdingType)
    _, obj2Binary = cv2.threshold(obj2Image, settings.threshold1, settings.threshold2, thresholdingType)
    _, imageBinary = cv2.threshold(image, settings.threshold1, settings.threshold2, thresholdingType)

    return obj1Binary, obj2Binary, imageBinary


def doImagesClosing(obj1Image, obj2Image, image, settings):
    kernel = np.ones(settings.morphologicalMaskShape, np.uint8)
    obj1Closing = doObjsClosing(obj1Image, settings)
    obj2Closing = doObjsClosing(obj2Image, settings)
    imageClosing = \
        cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=settings.morphCloseIterationsCount)

    # Now use OPEN to discard little noise (1-3 pixels wide elements here and there)
    imageClosing = \
        cv2.morphologyEx(imageClosing, cv2.MORPH_OPEN, kernel, iterations=settings.morphOpenIterationsCount)

    return obj1Closing, obj2Closing, imageClosing


def doObjsClosing(objImage, settings):
    kernel = np.ones(settings.morphologicalMaskShape, np.uint8)

    if not settings.isUsingGradientEdgeDetector and not settings.isBrightBackground:
        # Use OPEN to discard little noise (1-3 pixels wide elements here and there)
        objImage = \
            cv2.morphologyEx(objImage, cv2.MORPH_OPEN, kernel, iterations=settings.morphOpenIterationsCount)

    # Before we can dilate the object, we must copy it to a container which has a bigger background area
    # so we will avoid of having the object filling up all of its array bounds.
    objShape = objImage.shape
    padWidth = settings.morphCloseIterationsCount + 2
    objImagePadded = np.pad(objImage, padWidth)

    # Now it is safe to do Closing (Dilate & Erode)
    objClosing = cv2.morphologyEx(objImagePadded,
                                  cv2.MORPH_CLOSE,
                                  kernel,
                                  iterations=settings.morphCloseIterationsCount)

    # Now revert back to normal shape
    result = objClosing[padWidth: padWidth + objShape[0], padWidth: padWidth + objShape[1]]

    return result


def objectsToHitMissStructuringElement(obj1Closing, obj2Closing, settings, dilateOrErodeWidth):
    structuringElement1 = objectToHitMissStructuringElement(obj1Closing, settings, dilateOrErodeWidth)
    structuringElement2 = objectToHitMissStructuringElement(obj2Closing, settings, dilateOrErodeWidth)

    print('\n############### Structuring Element 1 ###############')
    fullyPrintArray(structuringElement1)
    print('\n############### Structuring Element 2 ###############')
    fullyPrintArray(structuringElement2)

    return structuringElement1, structuringElement2


def objectToHitMissStructuringElement(obj, settings, dilateOrErodeWidth):
    kernel = np.ones(settings.morphologicalMaskShape, np.uint8)
    pad = abs(dilateOrErodeWidth)

    # Resizing the object - Decrease
    if dilateOrErodeWidth < 0:
        obj = cv2.erode(obj, kernel, iterations=pad)
        obj = obj[pad: obj.shape[0] - pad, pad: obj.shape[1] - pad]
    # Resizing the object - Increase
    elif dilateOrErodeWidth > 0:
        obj = np.pad(obj, pad)
        obj = cv2.dilate(obj, kernel, iterations=pad)

    structuringElementDontCareWidth = settings.structuringElementDontCareWidth
    structuringElement = np.zeros((1, 2))
    while np.count_nonzero(structuringElement >= 127) < 5:
        structuringElement = cv2.erode(obj, kernel, iterations=structuringElementDontCareWidth)
        structuringElement = np.array(structuringElement, dtype=np.int16)

        # In order to avoid of infinite loop, use this check
        if structuringElementDontCareWidth == 1 and np.count_nonzero(structuringElement >= 127) < 5:
            structuringElement = obj
            break

        structuringElementDontCareWidth = int(structuringElementDontCareWidth / 2)

    # White area is mandatory in the structuring element. Such cells will take value 1
    # Black area must not present. Such cells will take value -1 (exclude)
    # And the difference between the object and the eroded one - Don't care. Value = 0
    differenceBetweenObjAndEroded = np.array(obj - structuringElement, dtype=np.int16)
    differenceBetweenObjAndEroded[differenceBetweenObjAndEroded <= 127] = 0
    differenceBetweenObjAndEroded[differenceBetweenObjAndEroded > 127] = 1
    structuringElement[structuringElement <= 127] = -1
    structuringElement[structuringElement > 127] = 1
    structuringElement = structuringElement + differenceBetweenObjAndEroded

    return structuringElement


def doHitMiss(imgClosing, obj1Closing, obj2Closing, settings, consoleConsumer, progressConsumer):
    # Arrays that will sum up all findings. (Might exceed 255 in case we find an object several times)
    hitMissObj1 = np.zeros(imgClosing.shape, dtype=np.int64)
    hitMissObj2 = np.zeros(imgClosing.shape, dtype=np.int64)

    consoleConsumer('Running Hit & Miss to detect objects in image...')

    # Prepare progress calculation
    progress = 0
    progressConsumer(progress)
    totalMorphIterations = float(settings.morphErodeIterationsCount + settings.morphDilateIterationsCount)
    totalSteps = totalMorphIterations * (360.0 / float(settings.objectRotationDegreeInc))
    progressStep = 92 / totalSteps

    for i in range(settings.morphErodeIterationsCount):
        # Prepare structuring elements out of the objects
        structuringElement1, structuringElement2 = \
            objectsToHitMissStructuringElement(obj1Closing, obj2Closing, settings, -i)
        hitMissObj1Inner, hitMissObj2Inner, progress = doHitMissWithRotation(hitMissObj1,
                                                                             hitMissObj2,
                                                                             imgClosing,
                                                                             settings,
                                                                             structuringElement1,
                                                                             structuringElement2,
                                                                             progressConsumer,
                                                                             progress,
                                                                             progressStep)
        hitMissObj1 += hitMissObj1Inner
        hitMissObj2 += hitMissObj2Inner

    for i in range(settings.morphDilateIterationsCount):
        # Prepare structuring elements out of the objects
        structuringElement1, structuringElement2 = \
            objectsToHitMissStructuringElement(obj1Closing, obj2Closing, settings, i)
        hitMissObj1Inner, hitMissObj2Inner, progress = doHitMissWithRotation(hitMissObj1,
                                                                             hitMissObj2,
                                                                             imgClosing,
                                                                             settings,
                                                                             structuringElement1,
                                                                             structuringElement2,
                                                                             progressConsumer,
                                                                             progress,
                                                                             progressStep)
        hitMissObj1 += hitMissObj1Inner
        hitMissObj2 += hitMissObj2Inner

    hitMissObj1[hitMissObj1 > 255] = 255
    hitMissObj2[hitMissObj2 > 255] = 255
    return np.uint8(hitMissObj1), np.uint8(hitMissObj2), progress


def doHitMissWithRotation(hitMissObj1,
                          hitMissObj2,
                          imgClosing,
                          settings,
                          structuringElement1,
                          structuringElement2,
                          progressConsumer,
                          startingProgress,
                          progressStep):
    progress = startingProgress

    # We might get an empty, or very little structure element when user plays with the erode, using
    # a big erosion
    checkStructure1 = np.count_nonzero(structuringElement1 == 1) > 4
    checkStructure2 = np.count_nonzero(structuringElement2 == 1) > 4

    # Loop over the rotation angles, ensuring no part of the image is cut off
    for angle in np.arange(0, 360, settings.objectRotationDegreeInc):
        if checkStructure1:
            structuringElement1Rotated = imutils.rotate_bound(structuringElement1, angle)
            hitMissObj1 = \
                hitMissObj1 + cv2.morphologyEx(imgClosing, cv2.MORPH_HITMISS, structuringElement1Rotated)

        if checkStructure2:
            structuringElement2Rotated = imutils.rotate_bound(structuringElement2, angle)
            hitMissObj2 = \
                hitMissObj2 + cv2.morphologyEx(imgClosing, cv2.MORPH_HITMISS, structuringElement2Rotated)

        progress += progressStep
        progressConsumer(progress)

    return hitMissObj1, hitMissObj2, progress


def highlightObjectsInImage(objectContours,
                            hitMissObj1,
                            hitMissObj2,
                            imageToHighlight,
                            settings,
                            consoleConsumer,
                            progressConsumer,
                            startingProgress):
    consoleConsumer('Highlighting objects in image...')

    hitMissObj1Locations = extractLocations(hitMissObj1, settings)
    hitMissObj2Locations = extractLocations(hitMissObj2, settings)
    obj1Count = 0
    obj2Count = 0

    # Prepare progress calculation
    progress = startingProgress
    progressConsumer(progress)
    maxLocations = max(len(hitMissObj1Locations) if hitMissObj1Locations is not None else 1,
                       len(hitMissObj2Locations) if hitMissObj2Locations is not None else 1)
    totalSteps = len(objectContours) * maxLocations
    progressStep = (100 - startingProgress) / totalSteps

    for contour in objectContours:
        if len(contour) < 5:
            continue

        isObj2 = False

        isObj1, obj1Count, progress = checkIfContourIsAnObject(contour, hitMissObj1Locations,
                                                               obj1Count, progressConsumer, progress,
                                                               progressStep)

        if not isObj1:
            isObj2, obj2Count, progress = checkIfContourIsAnObject(contour, hitMissObj2Locations,
                                                                   obj2Count, progressConsumer, progress,
                                                                   progressStep)

        if isObj1 or isObj2:
            # Get coordinates of minimal enclosing circle so we can get the center point
            ((x, y), _) = cv2.minEnclosingCircle(contour)
            objNumStr = "{}".format(obj1Count if isObj1 else obj2Count)

            cv2.drawContours(imageToHighlight, [contour], -1, settings.markColor, settings.markThickness)
            cv2.putText(imageToHighlight,
                        objNumStr,
                        (int(x) - 7, int(y) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (196, 146, 53) if isObj1 else (50, 167, 240),
                        3)

    consoleConsumer('First Object Count: ' + str(obj1Count) + ',  Second Object Count: ' + str(obj2Count))


def extractLocations(hitMissObjResult, settings):
    locations = []

    if hitMissObjResult is not None:
        kernel = np.ones(settings.morphologicalMaskShape, np.uint8)
        dilated = cv2.dilate(hitMissObjResult, kernel, iterations=2)
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            # Compute the center of the contour, using moments
            moments = cv2.moments(contour)
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            locations.append((x, y))

    return locations


def checkIfContourIsAnObject(contour, locations, objCount, progressConsumer, progress, progressStep):
    isObjectFound = False

    if locations is not None:
        for objPoint in locations:
            if cv2.pointPolygonTest(contour, objPoint, False) >= 0:
                isObjectFound = True
                objCount += 1
                break
            progress += progressStep
            progressConsumer(progress)

    return isObjectFound, objCount, progress


def concatenateImages2D(image1, image2):
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape

    # Create empty matrix
    result = np.zeros((rows1 + rows2 + 20, max(cols1, cols2)), np.uint8)

    # Combine 2 images, vertically, with padding
    result[: rows1, : cols1] = image1
    result[rows1 + 20: rows1 + 20 + rows2, : cols2] = image2

    return result


def concatenateImages3D(image1, image2):
    rows1, cols1 = image1.shape[:2]
    rows2, cols2 = image2.shape[:2]

    # Create empty matrix
    result = np.zeros((rows1 + rows2 + 20, max(cols1, cols2), 3), np.uint8)

    # Combine 2 images, vertically, with padding
    result[: rows1, : cols1, :] = image1
    result[rows1 + 20: rows1 + 20 + rows2, : cols2, :] = image2

    return result
