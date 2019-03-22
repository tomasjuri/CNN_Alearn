from base.base_model import BaseModel
import tensorflow as tf
from itertools import repeat

class FCNAlearn(BaseModel):
    def __init__(self, config):
        super(FCNAlearn, self).__init__(config)
        self.n_class = self.config.n_classes
        self.build_model()
        self.init_saver()

    def bottleneck_layer(self, x, n_chan, is_training, use_projection=True):
        shortcut = x

        if use_projection:
            # Projection shortcut in first layer to match filters and strides
            shortcut = tf.layers.conv2d(
                inputs=shortcut, filters=4*n_chan, kernel_size=[1, 1],
                padding="same", activation=None)
            
            shortcut = tf.nn.relu(shortcut)
            shortcut = tf.layers.batch_normalization(shortcut, training=is_training)

        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        
        x = tf.layers.conv2d(
            inputs=x, filters=n_chan, kernel_size=[1, 1], padding="same",
            activation=None)

        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        
        x = tf.layers.conv2d(
            inputs=x, filters=n_chan, kernel_size=[3, 3], padding="same",
            activation=None)

        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        
        x = tf.layers.conv2d(
            inputs=x, filters=4*n_chan, kernel_size=[1, 1],
            padding="same", activation=None)
        
        x += shortcut
        return x


    def features_to_mask(self, immediate_features, n_classes=2):      
        with tf.name_scope("features_1"):
            f1 = immediate_features['x1']
            f1 = tf.layers.conv2d_transpose(f1, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f1 = tf.nn.relu(f1)
            f1 = tf.layers.batch_normalization(f1, training=self.is_training)            

        with tf.name_scope("features_2"):
            f2 = immediate_features['x2']
            f2 = tf.layers.conv2d_transpose(f2, filters=64,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f2 = tf.nn.relu(f2)
            f2 = tf.layers.batch_normalization(f2, training=self.is_training)
            f2 = tf.layers.conv2d_transpose(f2, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f2 = tf.nn.relu(f2)
            f2 = tf.layers.batch_normalization(f2, training=self.is_training)

        with tf.name_scope("features_3"):
            f3 = immediate_features['x3']
            f3 = tf.layers.conv2d_transpose(f3, filters=128,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f3 = tf.nn.relu(f3)
            f3 = tf.layers.batch_normalization(f3, training=self.is_training)

            f3 = tf.layers.conv2d_transpose(f3, filters=64,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f3 = tf.nn.relu(f3)
            f3 = tf.layers.batch_normalization(f3, training=self.is_training)

            f3 = tf.layers.conv2d_transpose(f3, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f3 = tf.nn.relu(f3)
            f3 = tf.layers.batch_normalization(f3, training=self.is_training)

        with tf.name_scope("features_4"):
            f4 = immediate_features['x4']
            f4 = tf.layers.conv2d_transpose(f4, filters=256,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f4 = tf.nn.relu(f4)
            f4 = tf.layers.batch_normalization(f4, training=self.is_training)

            f4 = tf.layers.conv2d_transpose(f4, filters=128,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f4 = tf.nn.relu(f4)
            f4 = tf.layers.batch_normalization(f4, training=self.is_training)

            f4 = tf.layers.conv2d_transpose(f4, filters=64,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f4 = tf.nn.relu(f4)
            f4 = tf.layers.batch_normalization(f4, training=self.is_training)

            f4 = tf.layers.conv2d_transpose(f4, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f4 = tf.nn.relu(f4)
            f4 = tf.layers.batch_normalization(f4, training=self.is_training)

        with tf.name_scope("features_5"):
            f5 = immediate_features['x5']
            f5 = tf.layers.conv2d_transpose(f5, filters=256,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f5 = tf.nn.relu(f5)
            f5 = tf.layers.batch_normalization(f5, training=self.is_training)

            f5 = tf.layers.conv2d_transpose(f5, filters=128,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f5 = tf.nn.relu(f5)
            f5 = tf.layers.batch_normalization(f5, training=self.is_training)

            f5 = tf.layers.conv2d_transpose(f5, filters=64,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f5 = tf.nn.relu(f5)
            f5 = tf.layers.batch_normalization(f5, training=self.is_training)

            f5 = tf.layers.conv2d_transpose(f5, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f5 = tf.nn.relu(f5)
            f5 = tf.layers.batch_normalization(f5, training=self.is_training)

            f5 = tf.layers.conv2d_transpose(f5, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f5 = tf.nn.relu(f5)
            f5 = tf.layers.batch_normalization(f5, training=self.is_training)

        with tf.name_scope("features_6"):
            f6 = immediate_features['x6']
            f6 = tf.layers.conv2d_transpose(f6, filters=256,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f6 = tf.nn.relu(f6)
            f6 = tf.layers.batch_normalization(f6, training=self.is_training)

            f6 = tf.layers.conv2d_transpose(f6, filters=128,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f6 = tf.nn.relu(f6)
            f6 = tf.layers.batch_normalization(f6, training=self.is_training)

            f6 = tf.layers.conv2d_transpose(f6, filters=64,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f6 = tf.nn.relu(f6)
            f6 = tf.layers.batch_normalization(f6, training=self.is_training)

            f6 = tf.layers.conv2d_transpose(f6, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f6 = tf.nn.relu(f6)
            f6 = tf.layers.batch_normalization(f6, training=self.is_training)

            f6 = tf.layers.conv2d_transpose(f6, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f6 = tf.nn.relu(f6)
            f6 = tf.layers.batch_normalization(f6, training=self.is_training)

            f6 = tf.layers.conv2d_transpose(f6, filters=32,
                        strides=[2,2], kernel_size=[3,3], padding='same')
            f6 = tf.nn.relu(f6)
            f6 = tf.layers.batch_normalization(f6, training=self.is_training)

        with tf.name_scope("features_combined"):
            mask = tf.zeros_like(f1)
            fs = [f1, f2, f3, f4, f5, f6]
            for f in fs:
                mask += f

        with tf.name_scope("mask_conv_3x3"):
            mask = tf.layers.conv2d(
                inputs=mask, filters=n_classes, kernel_size=[3, 3],
                strides=[1, 1], padding="same", activation=None)
            mask = tf.nn.relu(mask)
            mask = tf.layers.batch_normalization(mask, training=self.is_training)

        with tf.name_scope("mask_conv_1x1"):
            mask = tf.layers.conv2d(
                inputs=mask, filters=n_classes, kernel_size=[1, 1],
                strides=[1, 1], padding="same", activation=None)
            mask = tf.nn.relu(mask)
            mask = tf.layers.batch_normalization(mask, training=self.is_training)

        return mask

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(
            tf.float32, shape=[None] + self.config.img_size)
        self.y = tf.placeholder(
            tf.int32, shape=[None] + self.config.mask_size)
        
        immediate_features = {}
        with tf.name_scope("1_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=self.x, pool_size=[2, 2], strides=2)
        
        with tf.name_scope("2_conv"):
            x = tf.layers.conv2d(
                inputs=x, filters=32, kernel_size=[3, 3],
                strides=[1, 1], padding="same", activation=None)
            x = tf.nn.relu(x)
            x = tf.layers.batch_normalization(x, training=self.is_training)
        
        with tf.name_scope("3_conv"):
            x = tf.layers.conv2d(
                inputs=x, filters=32, kernel_size=[3, 3],
                strides=[1, 1], padding="same", activation=None)
            x = tf.nn.relu(x)
            x = tf.layers.batch_normalization(x, training=self.is_training)
            immediate_features['x1'] = x

        with tf.name_scope("4_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)
    
        with tf.name_scope("5_resnet"):
            x = self.bottleneck_layer(x, 64, self.is_training)

        with tf.name_scope("6_resnet"):
            x = self.bottleneck_layer(x, 64, self.is_training)
            immediate_features['x2'] = x

        with tf.name_scope("7_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)

        with tf.name_scope("8_resnet"):
            x = self.bottleneck_layer(x, 128, self.is_training)

        with tf.name_scope("9_resnet"):
            x = self.bottleneck_layer(x, 128, self.is_training)
            immediate_features['x3'] = x

        with tf.name_scope("10_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)

        with tf.name_scope("11_resnet"):
            x = self.bottleneck_layer(x, 256, self.is_training)

        with tf.name_scope("12_resnet"):
            x = self.bottleneck_layer(x, 256, self.is_training)
            immediate_features['x4'] = x

        with tf.name_scope("13_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)

        with tf.name_scope("14_resnet"):
            x = self.bottleneck_layer(x, 256, self.is_training)

        with tf.name_scope("15_resnet"):
            x = self.bottleneck_layer(x, 256, self.is_training)
            immediate_features['x5'] = x

        with tf.name_scope("16_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)

        with tf.name_scope("17_resnet"):
            x = self.bottleneck_layer(x, 256, self.is_training)

        with tf.name_scope("18_resnet"):
            x = self.bottleneck_layer(x, 256, self.is_training)
            immediate_features['x6'] = x

        self.image_descriptor = tf.reduce_mean(x, axis=[1, 2])
        pred_masks = self.features_to_mask(immediate_features)
        
        with tf.name_scope("loss"):           
            flat_logits = tf.reshape(pred_masks, [-1, self.n_class])
            flat_labels = tf.reshape(self.y, [-1, self.n_class])

            # softmax_cross_entropy_with_logits_v2
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=flat_logits, labels=flat_labels)
            self.cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):    
                self.train_step = tf.train.AdamOptimizer(
                    self.config.learning_rate).minimize(self.cross_entropy,
                        global_step=self.global_step_tensor)
            
            correct_prediction = tf.equal(
                tf.argmax(pred_masks, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

            self.pred_masks = tf.nn.softmax(pred_masks, axis=3)    
            self.argmax_mask = tf.argmax(pred_masks, axis=3)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

