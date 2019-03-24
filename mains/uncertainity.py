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

logging.basicConfig(level=logging.DEBUG)

HDF_INFERENCE = 'inference'

DF_IMAGE_ID = 'image_id'
DF_IMG_PATH = 'src_image_path'

HDF_UNCERTAINITY = 'uncertainity'

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
        '-i', '--in_h5',
        help='List of uncertainity h5s files',
        nargs="*",
        type=str,
        required=True)
    argparser.add_argument(
        '-o', '--out_h5',
        help='Output h5 file with uncertainity results',
        required=True)
    args = argparser.parse_args()
    return args

def main():
    args = parse_args()
    
    inf_dfs = []
    for in_h5 in args.in_h5:
        logging.info('Opening dataframes: %s', in_h5)    
        with pd.HDFStore(in_h5, mode="r") as hdf:
            inf_dfs.append(hdf[HDF_INFERENCE])

    basepath_in = [os.path.dirname(in_h5) for in_h5 in args.in_h5]
    basepath_out = os.path.dirname(args.out_h5)
    
    data = {DF_IMAGE_ID: [], DF_IMG_PATH: []}

    for img_id in tqdm(inf_dfs[0][DF_IMAGE_ID].unique()):
        imgid_dfs = [df[df[DF_IMAGE_ID] == img_id] \
            for df in inf_dfs]
        rel_paths = [getattr(df, DF_IMG_PATH).values[0] \
            for df in imgid_dfs]
        abs_paths = [os.path.join(base, rel) \
            for base, rel in zip(basepath_in, rel_paths)]
        imgs = [cv2.imread(pth, cv2.IMREAD_UNCHANGED) \
            for pth in abs_paths]
        imgs = [img[np.newaxis,:,:] for img in imgs]
        imgs = np.concatenate(imgs, axis=0)

        diff = imgs.max(axis=0) - imgs.min(axis=0)
        diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        rel_mask_path = rel_paths[0] + '_uncertainity.png'
        abs_mask_path = os.path.join(basepath_out, rel_mask_path)

        if not os.path.exists(os.path.dirname(abs_mask_path)):
                os.makedirs(os.path.dirname(abs_mask_path))
        
        cv2.imwrite(abs_mask_path, diff.astype(np.uint8))
        
        logging.debug('Saving uncertainity: %s', abs_mask_path)

        data[DF_IMAGE_ID].append(img_id)
        data[DF_IMG_PATH].append(rel_mask_path)

        continue

    
    unc_df = pd.DataFrame(data)
    
    logging.info('Saving uncertainity h5: %s', args.out_h5)
    with pd.HDFStore(args.out_h5, mode="w") as hdf:
        hdf[HDF_UNCERTAINITY] = unc_df


if __name__ == '__main__':
    main()
