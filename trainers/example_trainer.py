from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc, x, y, pred_mask, argmax_mask = self.train_step()
            
            losses.append(loss)
            accs.append(acc)
            
        loss = np.mean(losses)
        acc = np.mean(accs)

        pred_mask = pred_mask*255.0
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask_0, pred_mask_1 = np.split(pred_mask, 2, axis=3)

        print('pred_mask_0.mean(): ', pred_mask_0.mean())
        print('pred_mask_0.max(): ', pred_mask_0.max())
        print('pred_mask_0.min): ', pred_mask_0.min())

        print('pred_mask_1.mean(): ', pred_mask_1.mean())
        print('pred_mask_1.max(): ', pred_mask_1.max())
        print('pred_mask_1.min): ', pred_mask_1.min())

        y_0, y_1 = np.split(y, 2, axis=3)
        argmax_mask = argmax_mask[:,:,:,np.newaxis] * 255.0

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'x': x,
            'y': y_0,
            'pred_mask_0': pred_mask_0,
            'pred_mask_1': pred_mask_1,
            'argmax_mask': argmax_mask,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc, pred_mask, argmax_mask = self.sess.run(
            [self.model.train_step, self.model.cross_entropy,
             self.model.accuracy, self.model.pred_masks, self.model.argmax_mask],
            feed_dict=feed_dict)
        return loss, acc, batch_x, batch_y, pred_mask, argmax_mask
