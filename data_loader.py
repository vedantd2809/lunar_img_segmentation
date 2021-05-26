import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import glob




class Dataset(tf.Module):
    def __init__(self,img_dir,mask_dir,transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)


    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):

        img_pth = os.path.join(self.img_dir,self.images[index])
        mask_pth = os.path.join(self.mask_dir,self.masks[index])

        image = cv2.imread(img_pth)
        image = image/255

        mask = cv2.imread(mask_pth)
        mask = mask/255


        image = np.array(image)
        mask = np.array(mask,dtype=np.float32)

        image = cv2.resize(image,(128,128))
        mask = cv2.resize(mask,(128,128))

        if self.transform is not None:
            augment = self.transform(image = image, mask=mask)
            image = augment['image']
            mask = augment['mask']

        return image, mask


