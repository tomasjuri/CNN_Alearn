from data_loader.data_generator_asphalt import DataGeneratorAsphalt
import numpy as np
from utils.config import process_config
import cv2


# Sample script to test augmentation correctness

config = process_config('../configs/config.json')
data = DataGeneratorAsphalt(config)

for i in range(0,20):
    img, mask = data.get_img(mode='test')

    cv2.imwrite('imgs_test/{}_mask.png'.format(str(i)), mask[:,:,0]*255.0)
    cv2.imwrite('imgs_test/{}_img.png'.format(str(i)), img*255.0)
    
    print(i)
    print(img.shape)
    print(mask.shape)