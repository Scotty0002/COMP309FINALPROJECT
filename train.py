#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from time import time

import numpy as np
import tensorflow as tf
import random
import os


# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)
TRAINDATADIR = "data/Train_data"
TESTDATADIR = "data/Test_data"
VALIDDATADIR = "data/Valid_data"
def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = (200,200,1)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3,activation = tf.nn.softmax))
    model.compile(loss='categorical_crossentropy', optimizer='AdaDelta', metrics=['accuracy'])
    return model


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    tensorboard = TensorBoard(log_dir = 'logs\{}'.format(time()))
    datagen = ImageDataGenerator()
    train_batches = datagen.flow_from_directory(TRAINDATADIR, target_size = (200,200), color_mode = 'grayscale', classes = ['cherry','strawberry','tomato'], batch_size = 80)
    valid_batches = datagen.flow_from_directory(VALIDDATADIR, target_size = (200,200), color_mode = 'grayscale', classes = ['cherry','strawberry','tomato'], batch_size = 80)
    test_batches = datagen.flow_from_directory(TESTDATADIR, target_size = (200,200), color_mode = 'grayscale', classes = ['cherry','strawberry','tomato'], batch_size = 10)
    model.fit_generator(train_batches, epochs = 16, callbacks = [tensorboard], validation_data = valid_batches, verbose = 1)
    return model

def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':
    model = construct_model()
    model = train_model(model)
    save_model(model)
