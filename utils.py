import numpy as np
import cv2
from data_loader import Dataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    model.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



def get_loaders(train_dir, train_maskdir,
    train_transform=None,num_workers=4,pin_memory=True):

    train_ds = Dataset(
        train_dir,
        train_maskdir,
    )

    train_img = []
    train_mask = []

    for im,ma in train_ds:
        train_img.append(im)
        train_mask.append(ma)

    train_img=np.array(train_img)
    train_mask = np.array(train_mask)

    return train_img, train_mask
