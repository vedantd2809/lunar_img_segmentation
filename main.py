import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Dropout, BatchNormalization,Concatenate,ReLU,LeakyReLU,Activation,Input,MaxPool2D
from keras.layers import Add
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import cv2
from matplotlib import pyplot as plt


def Down_Conv(filters, size):
    model = Sequential()

    model.add(Conv2D(filters, size, strides = 1, padding = 'same' , use_bias=False))

    model.add(ReLU())

    model.add(Conv2D(filters,size , strides = 1, padding='same', use_bias=True))

    model.add(ReLU())


    return model


def Up_Conv(filters, size, batchnorm=True):

    model = Sequential()
    model.add(Conv2DTranspose(filters*2, size, strides= 2, padding='same', use_bias=False))

    if batchnorm:
        model.add(BatchNormalization())

    model.add(Conv2D(filters, size, strides = 1, padding = 'same' , use_bias=False))

    model.add(LeakyReLU())

    model.add(Conv2D(filters, size, strides = 1, padding = 'same' , use_bias=False))

    model.add(LeakyReLU())

    return model





class UNET(tf.Module):


    def __init__(self, out_channel=3, filters=[64,128,256,512]
    ):
        super(UNET,self).__init__()

        self.ups = []
        self.downs = []
        self.pool = MaxPool2D((2,2))

        for feature in filters:
            self.downs.append(Down_Conv(feature, 3))


        for feature in reversed(filters):

            self.ups.append(Up_Conv(feature, 3))


        self.bottleneck = Down_Conv(filters[-1]*2, 3)

        self.final = Conv2D(out_channel, 1, 1, padding='same')


    def forward(self, x):

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

            x = self.pool(x)



        x = self.bottleneck(x)


        skip_connections = skip_connections[::-1]

        count = 0
        for up in self.ups:
            x = up(x)
            x = Concatenate()([skip_connections[count],x])
            count+=1

        x = self.final(x)

        return x

