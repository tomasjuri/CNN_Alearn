import tensorflow as tf

from models.fcn_alearn_model import FCNAlearn
from utils.config import process_config
import logging
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)

HDF_IMAGES = 'images'

DF_IMAGE_ID = 'image_id'
DF_IMG_PATH = 'src_image_path'

HDF_INFERENCE = 'inference'

def crop_center(img, size):
    cropy, cropx = size[:2]
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty + cropy,
               startx:startx + cropx]

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        help='The Configuration file',
        required=True)
    argparser.add_argument(
        '-i', '--in_h5',
        help='Input h5 file with images',
        required=True)
    argparser.add_argument(
        '-o', '--out_h5',
        help='Output h5 file with inference results',
        required=True)
    args = argparser.parse_args()
    return args

def main():
    args = parse_args()
    config = process_config(args.config)

    tf.set_random_seed(config.seed)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)

    # create an instance of the model
    model = FCNAlearn(config)
    model.load(sess)
    logging.info('Model created')  

    logging.info('Opening input image h5: %s', args.in_h5)    
    with pd.HDFStore(args.in_h5, mode="r") as hdf:
        img_df = hdf[HDF_IMAGES]

    basepath_in = os.path.dirname(args.in_h5)
    basepath_out = os.path.dirname(args.out_h5)
    
    data = {DF_IMAGE_ID: [], DF_IMG_PATH: []}

    for img_row in tqdm(img_df.itertuples(), total=len(img_df)):
        img_id = getattr(img_row, DF_IMAGE_ID)
        rel_path = getattr(img_row, DF_IMG_PATH)
        abs_path = os.path.join(basepath_in, rel_path)
        img = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)/255.0
        img = crop_center(img, config.img_size)
        img = img[:,:,np.newaxis]

        feed_dict = {model.x: [img], model.is_training: False}    
        run_list = [model.pred_masks]
        
        logging.debug('Running inference of image id: %d', img_id)
        (pred_masks, ) = sess.run(run_list, feed_dict=feed_dict)

        pred_masks = pred_masks*255.0
        pred_mask = pred_masks[0, ...]
        pred_mask_0, pred_mask_1 = np.split(pred_mask, 2, axis=2)

        rel_mask_path = rel_path + '_infer_model_' + config.model_name + '.png'
        abs_mask_path = os.path.join(basepath_out, rel_mask_path)

        if not os.path.exists(os.path.dirname(abs_mask_path)):
                os.makedirs(os.path.dirname(abs_mask_path))
        
        cv2.imwrite(abs_mask_path, pred_mask_0.astype(np.uint8))
        
        # TMP image saving
        img = img * 255.0
        cv2.imwrite(abs_mask_path + '_img.png', img.astype(np.uint8))

        logging.debug('Saving inference: %s', rel_mask_path)

        data[DF_IMAGE_ID].append(img_id)
        data[DF_IMG_PATH].append(rel_mask_path)

    inf_df = pd.DataFrame(data)
    
    logging.info('Saving inference h5: %s', args.out_h5)
    with pd.HDFStore(args.out_h5, mode="w") as hdf:
        hdf[HDF_INFERENCE] = inf_df




if __name__ == '__main__':
    main()
