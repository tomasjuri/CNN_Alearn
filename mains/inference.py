import tensorflow as tf

from data_loader.data_generator_asphalt import DataGeneratorAsphalt
from models.fcn_alearn_model import FCNAlearn
from trainers.asphalt_trainer import AsphaltTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
import logging
import argparse
import numpy as np
import cv2

logging.basicConfig(level=logging.DEBUG)

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        help='The Configuration file',
        required=True)
#    argparser.add_argument(
#        '-i', '--image',
#        help='The Configuration file',
#        required=True)
    args = argparser.parse_args()
    return args

def main():
    args = parse_args()
    config = process_config(args.config)

    tf.set_random_seed(config.seed)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)

    
    # create your data generator
    data = DataGeneratorAsphalt(config, apply_augs=False)

    # create an instance of the model
    model = FCNAlearn(config)
    model.load(sess)
    logging.debug('Model created')  
    
    batch_x, batch_y = next(data.next_batch(
        config.batch_size, mode='train'))
    feed_dict = {model.x: batch_x, model.y: batch_y,
                 model.is_training: False}    
    run_list = [model.accuracy, model.pred_masks]
        
    accuracy, pred_masks = sess.run(run_list, feed_dict=feed_dict)

    print('accuracy: ', accuracy)
    print('pred_masks.shape: ', pred_masks.shape)

    print('batch_y.shape: ', batch_y.shape)
    for i in range(pred_masks.shape[0]):
        x = batch_x[i, ...]*255.0

        y = batch_y[i, ...]*255.0
        y_0, y_1 = np.split(y, 2, axis=2)

        pred_mask = pred_masks[i, ...]*255.0
        pred_mask_0, pred_mask_1 = np.split(pred_mask, 2, axis=2)

        filename_mask = 'outputs/mask_{}.png'.format(str(i))
        filename_x = 'outputs/x_{}.png'.format(str(i))
        filename_y = 'outputs/y_{}.png'.format(str(i))
        
        print('saving: ', filename_mask)
        cv2.imwrite(filename_x, x.astype(np.uint8))
        cv2.imwrite(filename_y, y_0.astype(np.uint8))
        cv2.imwrite(filename_mask, pred_mask_0.astype(np.uint8))


if __name__ == '__main__':
    main()
