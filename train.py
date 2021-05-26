from data_loader import Dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Dropout, BatchNormalization,Concatenate,ReLU,LeakyReLU,Activation,Input,MaxPool2D
from keras.layers import Add
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import cv2
from matplotlib import pyplot as plt
from main import UNET
from data_loader import Dataset
from utils import *


class FIT(keras.Model):
    def __init__(self,model):
        super(FIT,self).__init__()
        self.model = model

    def train_step(self,data):
        x,y = data

        with tf.GradientTape() as tape:
            y_pred = model.forward(x)
            loss = self.compiled_loss(y,y_pred)

        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients,training_vars))
        self.compiled_metrics.update_state(y,y_pred)

        return {m.name : m.result() for m in self.metrics}



model = UNET()
training = FIT(model)
training.compile(
    optimizer=Adam(0.0001,0.05),
    loss = keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
    )

ig = 'vedant/Semantic segmentation dataset/Tile 1/images'
mk = 'vedant/Semantic segmentation dataset/Tile 1/masks'

img,mask = get_loaders(ig, mk)
mask.shape


training.fit(img,mask,epochs=2)

