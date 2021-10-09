import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pathlib
import time
import datetime
import cv2
import os
import numpy as np
import argparse

from matplotlib import pyplot as plt
from IPython import display
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--Input", default='Input', type=str)
args = parser.parse_args()

model = load_model('facades.h5')

input_image = tf.data.Dataset.list_files(str('Input/*.jpg'))

IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = tf.io.read_file(f'Input/{filename}')
        img = tf.image.decode_jpeg(img)
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if img is not None:
            images.append(img)
    return images


input = load_images_from_folder(args.Input)

predict = []
for image in input:
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False, name=None)
    image = np.expand_dims(image, axis=0)
    prediction = model(image, training=True)
    prediction = prediction * 0.5 + 0.5
    prediction = tf.image.convert_image_dtype(prediction, dtype=tf.uint8, saturate=False, name=None)
    prediction = np.squeeze(prediction, axis=0)
    predict.append(prediction)

for i, pred in enumerate(predict):
    plt.imshow(predict[i])
    plt.show()
