from imp import load_module
import os
from pyexpat import model
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt


def CheckFileExists():
    if os.path.isfile('models/ParkingSpotStatusModel.h5') is False:
        model.save(os.path.join('models', 'ParkingSpotStatusModel.h5'))

def SpotStatus(dataDir, folder, images, model):
    availableSpots = len(images)
    for spot in images:
        #read image
        img = cv2.imread(os.path.join(dataDir, folder, spot))
        resizedImg = tf.image.resize(img, (256, 256))
        result = model.predict(np.expand_dims(resizedImg/256,0))
        if result > 0.5:
            print(spot)
            availableSpots = availableSpots - 1
    return availableSpots


if __name__ == "__main__":
    data_dir = 'data'
    folder = 'available'

    CheckFileExists()
    new_model = load_model(os.path.join('models', 'ParkingSpotStatusModel.h5'))

    images = os.listdir(os.path.join('data', 'available'))
    print(len(images))
    print(SpotStatus(data_dir, folder, images, new_model))
  



