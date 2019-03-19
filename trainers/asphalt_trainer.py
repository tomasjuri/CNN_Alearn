from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class AsphaltTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(AsphaltTrainer, self).__init__(sess, model, data, config,logger)

    def postproc_masks(self, pred_mask, y, argmax_mask):
        pred_mask = pred_mask*255.0
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask_0, pred_mask_1 = np.split(pred_mask, 2, axis=3)
        y_0, y_1 = np.split(y, 2, axis=3)
        argmax_mask = argmax_mask[:,:,:,np.newaxis] * 255.0

        return {'argmax_mask': argmax_mask, 'pred_mask_0':pred_mask_0,
                'pred_mask_1':pred_mask_1, 'y_0':y_0, 'y_1': y_1}

    def train_epoch(self):
        losses = {'train':[], 'test':[]}
        accs = {'train':[], 'test':[]}

        train_loop = tqdm(range(self.config.num_iter_per_epoch))
        for _ in train_loop:
            loss, acc, x, y, pred_mask, argmax_mask = self.train_step() 
            losses['train'].append(loss)
            accs['train'].append(acc)
        imgs = self.postproc_masks(pred_mask, y, argmax_mask)
        
        summaries_dict_train = {
            'loss': np.array(losses['train']).mean(),
            'acc': np.array(accs['train']).mean(),
            'x': x,
            'y': imgs['y_0'],
            'pred_mask_0': imgs['pred_mask_0'],
            'argmax_mask': imgs['argmax_mask'],
        }

        cur_it = self.model.global_step_tensor.eval(self.sess)

        eval_loop = tqdm(range(self.config.num_evals_per_epoch))
        for _ in eval_loop:
            loss, acc, x, y, pred_mask, argmax_mask = self.eval_step() 
            losses['test'].append(loss)
            accs['test'].append(acc)
        imgs = self.postproc_masks(pred_mask, y, argmax_mask)
        
        summaries_dict_eval = {
            'loss': np.array(losses['test']).mean(),
            'acc': np.array(accs['test']).mean(),
            'x': x,
            'y': imgs['y_0'],
            'pred_mask_0': imgs['pred_mask_0'],
            'argmax_mask': imgs['argmax_mask'],
        }
        
        self.logger.summarize(
            cur_it, summarizer="train", summaries_dict=summaries_dict_train)
        self.logger.summarize(
            cur_it, summarizer="test", summaries_dict=summaries_dict_eval)
        self.model.save(self.sess)


    def train_step(self):
        batch_x, batch_y = next(
            self.data.next_batch(self.config.batch_size, mode='train'))
        feed_dict = {
            self.model.x: batch_x,
            self.model.y: batch_y,
            self.model.is_training: True
            }
        
        run_list = [self.model.train_step, self.model.cross_entropy,
                    self.model.accuracy, self.model.pred_masks,
                    self.model.argmax_mask]
        
        _, loss, acc, pred_mask, argmax_mask = self.sess.run(run_list,
            feed_dict=feed_dict)
        return loss, acc, batch_x, batch_y, pred_mask, argmax_mask

    def eval_step(self):
        batch_x, batch_y = next(
            self.data.next_batch(self.config.batch_size, mode='test'))
        
        feed_dict = {
            self.model.x: batch_x,
            self.model.y: batch_y,
            self.model.is_training: False
            }
        
        loss, acc, pred_mask, argmax_mask = self.sess.run(
            [self.model.cross_entropy, self.model.accuracy,
             self.model.pred_masks, self.model.argmax_mask],
            feed_dict=feed_dict)
        return loss, acc, batch_x, batch_y, pred_mask, argmax_mask
