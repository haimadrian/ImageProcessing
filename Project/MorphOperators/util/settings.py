__author__ = "Haim Adrian"

import os
from ast import literal_eval

SETTINGS_FILE_NAME = 'settings.txt'

# Defaults
DEFAULT_GAMMA_CORRECTION = 1.3
DEFAULT_BLUR_KERNEL_SIZE = 13
DEFAULT_IS_USING_GRADIENT = False
DEFAULT_THRESH_MIN = 92
DEFAULT_THRESH_MAX = 255
DEFAULT_IS_BRIGHT_BACKGROUND = False
DEFAULT_IMAGE_SHAPE = (400, 400)
DEFAULT_MORPH_CLOSE_MASK_SHAPE = (3, 3)
DEFAULT_MORPH_CLOSE_ITERATIONS_COUNT = 8
DEFAULT_MORPH_OPEN_ITERATIONS_COUNT = 2
DEFAULT_MORPH_DILATE_ITERATIONS_COUNT = 6
DEFAULT_MORPH_ERODE_ITERATIONS_COUNT = 4
DEFAULT_STRUCTURE_ELEMENT_DONTCARE_WIDTH = 5
DEFAULT_OBJECT_MARKER_THICKNESS = 2
DEFAULT_OBJECT_MARKER_COLOR = (0, 255, 0)
DEFAULT_OBJECT_ROTATE_DEGREE_INC = 3


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


class Settings(Singleton):
    """
    Settings class is used to let user to configure some advanced properties, for the application
    and for the algorithm itself.
    This class is written to file so we can save settings between launches of the application.
    """

    def __init__(self,
                 gammaCorrectionValue=DEFAULT_GAMMA_CORRECTION,
                 blurKernelSize=DEFAULT_BLUR_KERNEL_SIZE,
                 isUsingGradientEdgeDetector=DEFAULT_IS_USING_GRADIENT,
                 threshold1=DEFAULT_THRESH_MIN,
                 threshold2=DEFAULT_THRESH_MAX,
                 isBrightBackground=DEFAULT_IS_BRIGHT_BACKGROUND,
                 morphCloseIterationsCount=DEFAULT_MORPH_CLOSE_ITERATIONS_COUNT,
                 morphOpenIterationsCount=DEFAULT_MORPH_OPEN_ITERATIONS_COUNT,
                 morphDilateIterationsCount=DEFAULT_MORPH_DILATE_ITERATIONS_COUNT,
                 morphErodeIterationsCount=DEFAULT_MORPH_ERODE_ITERATIONS_COUNT,
                 structuringElementDontCareWidth=DEFAULT_STRUCTURE_ELEMENT_DONTCARE_WIDTH,
                 markColor=DEFAULT_OBJECT_MARKER_COLOR,
                 markThickness=DEFAULT_OBJECT_MARKER_THICKNESS,
                 imageShape=DEFAULT_IMAGE_SHAPE,
                 morphologicalMaskShape=DEFAULT_MORPH_CLOSE_MASK_SHAPE,
                 objectRotationDegreeInc=DEFAULT_OBJECT_ROTATE_DEGREE_INC):
        """
        Constructs a new Settings instance.

        :param gammaCorrectionValue: See gammaCorrectionValue
        :param blurKernelSize: See blurKernelSize
        :param isUsingGradientEdgeDetector: See isUsingGradientEdgeDetector
        :param threshold1: See threshold1
        :param threshold2: See threshold2
        :param isBrightBackground: See isBrightBackground
        :param morphCloseIterationsCount: See morphologicalIterationsCount
        :param morphOpenIterationsCount: See morphOpenIterationsCount
        :param morphDilateIterationsCount: See morphDilateIterationsCount
        :param morphErodeIterationsCount: See morphErodeIterationsCount
        :param structuringElementDontCareWidth: See structuringElementDontCareWidth
        :param markColor: See markColor
        :param markThickness: See markThickness
        :param imageShape: See imageShape
        :param morphologicalMaskShape: See morphologicalMaskShape
        :param objectRotationDegreeInc: See objectRotationDegreeInc
        """
        self.gammaCorrectionValue = gammaCorrectionValue
        self.blurKernelSize = blurKernelSize
        self.isUsingGradientEdgeDetector = isUsingGradientEdgeDetector
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.isBrightBackground = isBrightBackground
        self.morphCloseIterationsCount = morphCloseIterationsCount
        self.morphOpenIterationsCount = morphOpenIterationsCount
        self.morphDilateIterationsCount = morphDilateIterationsCount
        self.morphErodeIterationsCount = morphErodeIterationsCount
        self.structuringElementDontCareWidth = structuringElementDontCareWidth
        self.markColor = markColor
        self.markThickness = markThickness
        self.imageShape = imageShape
        self.morphologicalMaskShape = morphologicalMaskShape
        self.objectRotationDegreeInc = objectRotationDegreeInc

    @property
    def gammaCorrectionValue(self):
        """
        The Gamma value to use when adjusting image contrast.
        We adjust image contrast as enhancement, to make it easier to find edges later even
        when object color is similar to the background color.
        gamma values < 1 will shift the image towards the darker end of the spectrum
        while gamma values > 1 will make the image appear lighter.
        gamma = 1 means no affect.
        Default value is 1.3

        :return: Gamma correction value (Remember to invert)
        """
        return self.__gammaCorrectionValue

    @gammaCorrectionValue.setter
    def gammaCorrectionValue(self, value):
        self.__gammaCorrectionValue = value

    @property
    def blurKernelSize(self):
        """
        A blurring rate to be used when applying Median Blur.
        We blur the images, to reduce noise and make it easier to focus on objects
        and avoid of having sharp lines that will reduce efficiency in finding objects.
        Default value is 20

        :return: Blurring rate for Median Blur
        """
        return self.__blurKernelSize

    @blurKernelSize.setter
    def blurKernelSize(self, value):
        self.__blurKernelSize = value

    @property
    def isUsingGradientEdgeDetector(self):
        """
        A flag indicating whether we are using gradient edge detector during the algorithm or not.
        By default the value is set to true, but it can be modified.
        Default value is True

        :return: Whether we should use gradient edge detector during image processing or not
        """
        return self.__isUsingGradientEdgeDetector

    @isUsingGradientEdgeDetector.setter
    def isUsingGradientEdgeDetector(self, value):
        self.__isUsingGradientEdgeDetector = value

    @property
    def threshold1(self):
        """
        Minimum threshold when performing cv2.threshold, to pickup all gray-scale values
        bigger than that threshold.
        Default value is 13

        :return: Threshold1 for cv2.threshold
        """
        return self.__threshold1

    @threshold1.setter
    def threshold1(self, value):
        self.__threshold1 = value

    @property
    def threshold2(self):
        """
        Maximum threshold when performing cv2.threshold, to pickup all gray-scale values
        below that threshold.
        Default value is 255

        :return: Threshold2 for cv2.threshold. (MaxVal)
        """
        return self.__threshold2

    @threshold2.setter
    def threshold2(self, value):
        self.__threshold2 = value

    @property
    def isBrightBackground(self):
        """
        Whether input image got a bright background. When background is bright, and we would like to
        do thresholding such that we keep bright colors only, then we use threshold binary invert,
        to make sure the background becomes black and the objects are white.

        :return: If the image got a bright background
        """
        return self.__isBrightBackground

    @isBrightBackground.setter
    def isBrightBackground(self, value):
        self.__isBrightBackground = value

    @property
    def morphCloseIterationsCount(self):
        """
        How many iterations to perform during "Closing" morphological operator.
        Reminder: Closing means first dilate and then erode.
        Default value is 6

        :return: Amount of iterations for cv2.morphologyEx
        """
        return self.__morphCloseIterationsCount

    @morphCloseIterationsCount.setter
    def morphCloseIterationsCount(self, value):
        self.__morphCloseIterationsCount = value

    @property
    def morphOpenIterationsCount(self):
        """
        How many iterations to perform during "Opening" morphological operator.
        Reminder: Opening means first erode and then dilate.
        Default value is 2

        :return: Amount of iterations for cv2.morphologyEx
        """
        return self.__morphOpenIterationsCount

    @morphOpenIterationsCount.setter
    def morphOpenIterationsCount(self, value):
        self.__morphOpenIterationsCount = value

    @property
    def morphDilateIterationsCount(self):
        """
        How many iterations to perform during "Dilate" morphological operator.
        We take an input image to look for, then dilate it over and over, trying to
        find that object in another image.
        This way we can try to overcome DISTANCE of the objects from camera.
        Default value is 10

        :return: Amount of iterations for cv2.morphologyEx
        """
        return self.__morphDilateIterationsCount

    @morphDilateIterationsCount.setter
    def morphDilateIterationsCount(self, value):
        self.__morphDilateIterationsCount = value

    @property
    def morphErodeIterationsCount(self):
        """
        How many iterations to perform during "Erode" morphological operator.
        We take an input image to look for, then erode it over and over, trying to
        find that object in another image.
        This way we can try to overcome DISTANCE of the objects from camera.
        Default value is 10

        :return: Amount of iterations for cv2.morphologyEx
        """
        return self.__morphErodeIterationsCount

    @morphErodeIterationsCount.setter
    def morphErodeIterationsCount(self, value):
        self.__morphErodeIterationsCount = value

    @property
    def structuringElementDontCareWidth(self):
        """
        The border width to define as Don't Care when preparing a structuring element for hit&miss
        Default value is 5

        :return: Border width of don't care for structuring element
        """
        return self.__structuringElementDontCareWidth

    @structuringElementDontCareWidth.setter
    def structuringElementDontCareWidth(self, value):
        self.__structuringElementDontCareWidth = value

    @property
    def markColor(self):
        """
        The color (RGB Tuple) we use for drawing a mark sign around objects.
        Default value is (0, 255, 0) - Green

        :return: Marks color
        """
        return self.__markColor

    @markColor.setter
    def markColor(self, value):
        self.__markColor = value

    @property
    def markThickness(self):
        """
        How thick a mark around object should be.
        Default value is 2

        :return: Marks thickness
        """
        return self.__markThickness

    @markThickness.setter
    def markThickness(self, value):
        self.__markThickness = value

    @property
    def imageShape(self):
        """
        Shape of the image - we resize the input images to this shape.
        Default value is (400, 400)

        :return: Shape of working image
        """
        return self.__imageShape

    @imageShape.setter
    def imageShape(self, value):
        self.__imageShape = value

    @property
    def morphologicalMaskShape(self):
        """
        Shape of Morphological Operator mask. We use cv2's morphologyEx method in order to
        execute a "Closing" (Dilate -> Erode) on gradient image, trying to fill in holes
        inside the objects.
        Default value is (3, 3)

        :return: Shape of Morphological Operator mask
        """
        return self.__morphologicalMaskShape

    @morphologicalMaskShape.setter
    def morphologicalMaskShape(self, value):
        self.__morphologicalMaskShape = value

    @property
    def objectRotationDegreeInc(self):
        """
        How much to increase degree rotation while looking up for objects with different angle rotation
        such that we can detect objects that are rotated in an image

        :return: Amount of degree incrementing
        """
        return self.__objectRotationDegreeInc

    @objectRotationDegreeInc.setter
    def objectRotationDegreeInc(self, value):
        self.__objectRotationDegreeInc = value

    def save(self):
        """
        Store settings to file

        :return: self
        """
        return self.saveAs(SETTINGS_FILE_NAME)

    def saveAs(self, outFilePath):
        """
        Store settings to file

        :param outFilePath: Path of the file to store settings to
        :return: self
        """
        print('INFO - Storing settings to file:', outFilePath)
        with open(outFilePath, 'w') as outFile:
            outFile.writelines([str(self.gammaCorrectionValue) + '\n',
                                str(self.blurKernelSize) + '\n',
                                str(self.isUsingGradientEdgeDetector) + '\n',
                                str(self.threshold1) + '\n',
                                str(self.threshold2) + '\n',
                                str(self.isBrightBackground) + '\n',
                                str(self.morphCloseIterationsCount) + '\n',
                                str(self.morphOpenIterationsCount) + '\n',
                                str(self.morphDilateIterationsCount) + '\n',
                                str(self.morphErodeIterationsCount) + '\n',
                                str(self.structuringElementDontCareWidth) + '\n',
                                str(self.markColor) + '\n',
                                str(self.markThickness) + '\n',
                                str(self.imageShape) + '\n',
                                str(self.morphologicalMaskShape) + '\n',
                                str(self.objectRotationDegreeInc)])
        return self

    def load(self):
        return self.loadFrom(SETTINGS_FILE_NAME)

    def loadFrom(self, filePath):
        """
        Load settings from a previously saved file.
        Does nothing if the file does not exist.
        :param filePath: Path of the file to load from
        :return: self
        """
        if os.path.exists(filePath) and os.path.isfile(filePath):
            try:
                print('INFO - Loading settings from file:', filePath)
                with open(filePath, 'r') as inFile:
                    self.gammaCorrectionValue = float(inFile.readline().strip())
                    self.blurKernelSize = int(inFile.readline().strip())
                    self.isUsingGradientEdgeDetector = (inFile.readline().strip() == 'True')
                    self.threshold1 = int(inFile.readline().strip())
                    self.threshold2 = int(inFile.readline().strip())
                    self.isBrightBackground = (inFile.readline().strip() == 'True')
                    self.morphCloseIterationsCount = int(inFile.readline().strip())
                    self.morphOpenIterationsCount = int(inFile.readline().strip())
                    self.morphDilateIterationsCount = int(inFile.readline().strip())
                    self.morphErodeIterationsCount = int(inFile.readline().strip())
                    self.structuringElementDontCareWidth = int(inFile.readline().strip())
                    self.markColor = literal_eval(inFile.readline().strip())
                    self.markThickness = int(inFile.readline().strip())
                    self.imageShape = literal_eval(inFile.readline().strip())
                    self.morphologicalMaskShape = literal_eval(inFile.readline().strip())
                    self.objectRotationDegreeInc = int(inFile.readline().strip())
            except Exception as e:
                print('ERROR - Error has occurred while reading settings file. File has to be ' +
                      'overridden. Error:', str(e))
        return self

    def reset(self):
        self.gammaCorrectionValue = DEFAULT_GAMMA_CORRECTION
        self.blurKernelSize = DEFAULT_BLUR_KERNEL_SIZE
        self.isUsingGradientEdgeDetector = DEFAULT_IS_USING_GRADIENT
        self.threshold1 = DEFAULT_THRESH_MIN
        self.threshold2 = DEFAULT_THRESH_MAX
        self.isBrightBackground = DEFAULT_IS_BRIGHT_BACKGROUND
        self.morphCloseIterationsCount = DEFAULT_MORPH_CLOSE_ITERATIONS_COUNT
        self.morphOpenIterationsCount = DEFAULT_MORPH_OPEN_ITERATIONS_COUNT
        self.morphDilateIterationsCount = DEFAULT_MORPH_DILATE_ITERATIONS_COUNT
        self.morphErodeIterationsCount = DEFAULT_MORPH_ERODE_ITERATIONS_COUNT
        self.structuringElementDontCareWidth = DEFAULT_STRUCTURE_ELEMENT_DONTCARE_WIDTH
        self.markColor = DEFAULT_OBJECT_MARKER_COLOR
        self.markThickness = DEFAULT_OBJECT_MARKER_THICKNESS
        self.imageShape = DEFAULT_IMAGE_SHAPE
        self.morphologicalMaskShape = DEFAULT_MORPH_CLOSE_MASK_SHAPE
        self.objectRotationDegreeInc = DEFAULT_OBJECT_ROTATE_DEGREE_INC


# Modules are imported only once, so this variable will be a singleton of Settings.
settingsInstance = Settings()
