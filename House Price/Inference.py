import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('CNN_HousePrices.h5')
Images = []
outputImage = np.zeros((64, 64, 3), dtype="uint8")

for image_name in os.listdir('TestData'):
    image = cv2.imread('TestData/' + image_name)
    image = cv2.resize(image, (32, 32))
    Images.append(image)

outputImage[0:32, 0:32] = Images[0]
outputImage[0:32, 32:64] = Images[1]
outputImage[32:64, 32:64] = Images[2]
outputImage[32:64, 0:32] = Images[3]

outputImage = outputImage/255
outputImage = outputImage.reshape(1, 64, 64, 3)
prediction = model.predict([outputImage])
print('The House Price is: ', prediction)
