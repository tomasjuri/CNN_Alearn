import tensorflow as tf

from data_loader.data_generator_asphalt import DataGeneratorAsphalt
from models.fcn_alearn_model import FCNAlearn
from trainers.asphalt_trainer import AsphaltTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import logging

logging.basicConfig(level=logging.DEBUG)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    
    # create tensorflow session
    tf.set_random_seed(config.seed)
    tf_config = tf.ConfigProto() #device_count = {'GPU': 2})
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)
    
    # create your data generator
    data = DataGeneratorAsphalt(config)

    # create an instance of the model you want
    model = FCNAlearn(config)
    logging.debug('Model created')  
    
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = AsphaltTrainer(sess, model, data, config, logger)
    #load model if exists
    sess.run(tf.global_variables_initializer())
    #model.load(sess)

    trainer.train()


if __name__ == '__main__':
    main()
