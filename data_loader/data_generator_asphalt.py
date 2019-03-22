import numpy as np
import random
import cv2
import pandas as pd
import os
import imgaug as ia
import imgaug
from imgaug import augmenters as iaa
from sklearn.cross_validation import train_test_split

from shutil import copyfile

HDF_ANNOTATIONS = 'annotations'
HDF_IMAGES = 'images'

DF_IMAGE_ID = 'image_id'
DF_IMG_PATH = 'src_image_path'
DF_MASK_PATH = 'mask_path'

def random_crop(img, mask, crop_shape):
    shp = img.shape
    start = [random.randint(0, shp[0] - crop_shape[0]),
             random.randint(0, shp[1] - crop_shape[1])]
    end = [start[0] + crop_shape[0],
           start[1] + crop_shape[1]]
             
    img_cropped = img[start[0]:end[0], start[1]:end[1],:]
    mask_cropped = mask[start[0]:end[0], start[1]:end[1],:]

    return img_cropped, mask_cropped


class DataGeneratorAsphalt:
    def __init__(self, config):
        self.config = config
        
        ann_h5 = self.config.ann_h5
        img_h5 = self.config.img_h5

        with pd.HDFStore(ann_h5, mode="r") as hdf:
            self.ann_df = hdf[HDF_ANNOTATIONS]
        with pd.HDFStore(img_h5, mode="r") as hdf:
            self.img_df = hdf[HDF_IMAGES]
        
        self.img_dirname = os.path.dirname(img_h5)
        self.mask_dirname = os.path.dirname(ann_h5)

        ann_imgids = self.ann_df[DF_IMAGE_ID].unique()
        self.ann_imgids_train, self.ann_imgids_test = train_test_split(
            ann_imgids, test_size=config.test_size,
            random_state=config.seed)

        self.augs = self.init_augs()

    def init_augs(self):
        augs = [
            iaa.Crop(percent=((0, 0.4), (0, 0.4), (0, 0.4), (0, 0.4))),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.9, 1.1)),
            iaa.Affine(
                rotate=(-30, 30),
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            )
        ]
        seq = iaa.SomeOf(4, augs, random_order=True)
        return seq

    def augment_image(self, image, mask):
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        det = self.augs.to_deterministic()
        image = det.augment_image(image)
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        
        return image, mask.astype(np.float32)

    def get_img(self, mode):
        img_size = self.config.img_size

        # get random img id
        if mode == 'train':
            img_id = random.choice(self.ann_imgids_train)
        elif mode == 'test':
            img_id = random.choice(self.ann_imgids_test)
        else:
            raise KeyError

        img_path_rel = getattr(self.img_df[
            self.img_df[DF_IMAGE_ID] == img_id], DF_IMG_PATH).values[0]
        

        img_path_abs = os.path.join(self.img_dirname, img_path_rel)
        
        ann_per_img = self.ann_df[self.ann_df[DF_IMAGE_ID] == img_id]
        
        masks = []
        for mask_row in ann_per_img.itertuples(): 
            mask_path_rel = getattr(mask_row, DF_MASK_PATH)
            mask_path_abs = os.path.join(self.mask_dirname, mask_path_rel)
            mask_path_abs = mask_path_abs.replace('\\', '/')
            mask = cv2.imread(mask_path_abs, cv2.IMREAD_UNCHANGED)/255.0
            masks.append(mask)
        mask = np.array(masks).max(axis=0)
        mask = mask[:, :, np.newaxis]
          
        # load imgs
        img = cv2.imread(img_path_abs, cv2.IMREAD_UNCHANGED)/255.0
        img = img[:, :, np.newaxis]
        
        crop_shape = self.config.img_size
        img, mask = random_crop(img, mask, crop_shape)

        # print('before aug, img.min(): ', img.min())
        # print('before aug, img.max(): ', img.max())
        # print('before aug, img.dtype: ', img.dtype)
        # print('before aug, img.shape: ', img.shape)        

        # print('before aug, mask.min(): ', mask.min())
        # print('before aug, mask.max(): ', mask.max())
        # print('before aug, mask.dtype: ', mask.dtype)
        # print('before aug, mask.shape: ', mask.shape)
        
        img, mask = self.augment_image(
            img.astype(np.float32), mask.astype(np.float32))
        
        mask_cls2 = np.ones_like(mask) - mask
        mask = np.concatenate((mask, mask_cls2), axis=2)

        # print(' ')
        # print(' ')
        # print(' ')

        # print('after aug, img.min(): ', img.min())
        # print('after aug, img.max(): ', img.max())
        # print('after aug, img.dtype: ', img.dtype)
        # print('after aug, img.shape: ', img.shape)        

        # print('after aug, mask.min(): ', mask.min())
        # print('after aug, mask.max(): ', mask.max())
        # print('after aug, mask.dtype: ', mask.dtype)
        # print('after aug, mask.shape: ', mask.shape)
        # #exit(0)

        return img.astype(np.float32), mask.astype(np.float32)

    def next_batch(self, batch_size, mode='train'):
        y = []
        x = []
        for _ in range(batch_size):
            img, mask = self.get_img(mode=mode)

            y.append(mask)
            x.append(img)

        yield np.array(x), np.array(y)
