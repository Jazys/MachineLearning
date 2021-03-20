# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:09:08 2020

@author: CupakabraNo1
"""

import dataset_loader as dl
from model import model
import constants as const
import keras
import tensorflowjs as tfjs
from accuracy import AccuracyHistory
import matplotlib.pyplot as plt

def visualize_data(images, categories):
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    for i in range(3 * 2):
        plt.subplot(3, 7, i+1)
        plt.xticks([])
        plt.yticks([])
        img=None
        print (images[i].shape)
        if images[i].shape[2] == 1:
          img=images[i][:, :,0]
        plt.imshow(img)

    plt.show()

history = AccuracyHistory()

#On ajoute de la distorsion à nos images
datagen = keras.preprocessing.image.ImageDataGenerator(
  rotation_range=30,
  width_shift_range=0.25,
  height_shift_range=0.25,
  shear_range=0.25,
  zoom_range=0.2
)

train_generator = datagen.flow(dl.train_image_data, dl.train_label_data)

#Pour visualiser les images
#visualize_data(dl.train_image_data, dl.train_label_data)
#exit(0)
model.fit(train_generator, batch_size = const.BATCH_SIZE, epochs = const.EPOCHS, verbose = 1, validation_data = (dl.test_image_data, dl.test_label_data), callbacks = [history])
test_loss,test_acc = model.evaluate(dl.test_image_data, dl.test_label_data)

#conversion h5
model.save('mnist_shift.h5')

#conversion pour le format web
tfjs.converters.save_keras_model(model, 'models3')

#conversion pour Android
#utiliser le script convert-model-anroid.py

