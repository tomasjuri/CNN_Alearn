import numpy as np
import random
import cv2

class DataGenerator:
    def __init__(self, config):
        
        self.config = config
        # load data here
        
        #self.input = np.ones((500, 784))
        #self.y = np.ones((500, 10))

    def gen_img(self):
        img_size = self.config.img_size
        img = np.ones(img_size, dtype=np.float32)
        mask = np.zeros(img_size, dtype=np.int32)
        
        # generate circle in the image
        y_center = random.randint(0, img.shape[0])
        x_center = random.randint(0, img.shape[1])
        diameter = random.randint(10, 150)

        img = cv2.circle(
            img, (x_center, y_center), diameter, (0.0), -1)
        # add noise
        row, col, c = img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, c))
        gauss = gauss.reshape(row, col, c)
        img = img + gauss

        # generate mask
        mask = cv2.circle(
            mask, (x_center, y_center), diameter, (1), -1)

        return img, mask

    def next_batch(self, batch_size):
        y = []
        x = []
        for _ in range(batch_size):
            img, mask = self.gen_img()
            y.append(mask)
            x.append(img)

        yield np.array(x), np.array(y)
