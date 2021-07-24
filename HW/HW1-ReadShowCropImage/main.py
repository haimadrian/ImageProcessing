__author__ = "Haim Adrian"

from functions import *


image = readImage("vegeta.jpg")
showImageUsingCV2AndMatPlotLib(image)
showImageAsBinary(image)
showImageCropped(image, (image.shape[0] - 100, image.shape[1] - 200), (50, 100))

# Wait so the application will not be terminated automatically
cv2.waitKey()
