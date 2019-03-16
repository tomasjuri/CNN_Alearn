from base.base_model import BaseModel
import tensorflow as tf
from itertools import repeat

def bottleneck_layer(x, n_chan, use_projection=True):
    shortcut = x

    if use_projection:
    # Projection shortcut in first layer to match filters and strides
        shortcut = tf.layers.conv2d(
            inputs=shortcut, filters=4*n_chan, kernel_size=[1, 1],
            padding="same", activation=None)
        #shortcut = tf.layers.batch_normalization(shortcut)

    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        inputs=x, filters=n_chan, kernel_size=[1, 1], padding="same",
        activation=None)

    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        inputs=x, filters=n_chan, kernel_size=[3, 3], padding="same",
        activation=None)

    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        inputs=x, filters=4*n_chan, kernel_size=[1, 1],
        padding="same", activation=None)
    
    x += shortcut
    x = tf.layers.batch_normalization(x)
    return x

class FCNAlearn(BaseModel):
    def __init__(self, config):
        super(FCNAlearn, self).__init__(config)
        self.build_model()
        self.init_saver()
        self.init_variables()

    def init_variables(self):
        pass

    def features_to_mask(self, immediate_features, n_classes=2):
        f_list = []
        for i in range(1,7):
            f_key = 'x' + str(i)
            f = immediate_features[f_key]
            for j in range(i):
                print(j)
                print('f before resize: ', f.shape)
                f_shp = f.get_shape().as_list()
                n_filters = 32 if f_shp[3] == 32 else f_shp[3]//2
                n_filters = 32 if j+1 == i else n_filters
                print('f before convolution: ', f.shape)
                f = tf.layers.conv2d_transpose(f, filters=n_filters,
                    strides=[2,2], kernel_size=[3,3], padding='same')

                print('f after convolution: ', f.shape)
                print('-----')
            f_list.append(f)
        
        mask = tf.zeros_like(f_list[0])
        print('mask.shape before addition: ', mask.shape)
        for f in f_list:
            mask += f
            print('adding f to mask: ', f.shape)
        print('mask.shape after addition: ', mask.shape)

        with tf.name_scope("mask_conv_3x3"):
            mask = tf.layers.conv2d(
                inputs=mask, filters=n_classes, kernel_size=[3, 3],
                strides=[1, 1], padding="same", activation=None)
            mask = tf.layers.batch_normalization(mask)
            mask = tf.nn.relu(mask)

        with tf.name_scope("mask_conv_1x1"):
            mask = tf.layers.conv2d(
                inputs=mask, filters=n_classes, kernel_size=[1, 1],
                strides=[1, 1], padding="same", activation=None)
            mask = tf.layers.batch_normalization(mask)
            mask = tf.nn.relu(mask)
            
        return mask

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(
            tf.float32, shape=[None] + self.config.img_size)
        self.y = tf.placeholder(
            tf.int32, shape=[None] + self.config.img_size)
        
        immediate_features = {}
        with tf.name_scope("1_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=self.x, pool_size=[2, 2], strides=2)
        
        with tf.name_scope("2_conv"):
            x = tf.layers.conv2d(
                inputs=x, filters=32, kernel_size=[3, 3],
                strides=[1, 1], padding="same", activation=None)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            
        with tf.name_scope("3_conv"):
            x = tf.layers.conv2d(
                inputs=x, filters=32, kernel_size=[3, 3],
                padding="same", strides=[1, 1], activation=None)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            immediate_features['x1'] = x

        with tf.name_scope("4_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)
        
        with tf.name_scope("5_bottlenec"):
            x = bottleneck_layer(x, n_chan=64)
        
        with tf.name_scope("6_bottleneck"):
            x = bottleneck_layer(x, n_chan=64)
            immediate_features['x2'] = x
        
        with tf.name_scope("7_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)
        
        with tf.name_scope("8_bottleneck"):
            x = bottleneck_layer(x, n_chan=128)
        
        with tf.name_scope("9_bottleneck"):
            x = bottleneck_layer(x, n_chan=128)
            immediate_features['x3'] = x
        
        with tf.name_scope("10_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2) 
        
        with tf.name_scope("11_bottleneck"):
            x = bottleneck_layer(x, n_chan=256)
        
        with tf.name_scope("12_bottleneck"):
            x = bottleneck_layer(x, n_chan=256)
            immediate_features['x4'] = x
        
        with tf.name_scope("13_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)
        
        with tf.name_scope("14_bottleneck"):
            x = bottleneck_layer(x, n_chan=256)
        
        with tf.name_scope("15_bottleneck"):
            x = bottleneck_layer(x, n_chan=256)
            immediate_features['x5'] = x
        
        with tf.name_scope("16_pooling"):
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=[2, 2], strides=2)
        
        with tf.name_scope("17_bottleneck"):
            x = bottleneck_layer(x, n_chan=256)
        
        with tf.name_scope("18_bottleneck"):
            x = bottleneck_layer(x, n_chan=256)
            immediate_features['x6'] = x

        image_descriptor = tf.reduce_mean(x, axis=[1, 2])
        pred_masks = self.features_to_mask(immediate_features)
        
        with tf.name_scope("loss"):
            #self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=pred_masks))
            #self.cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            #    multi_class_labels=self.y, logits=pred_masks))
            
            logits = tf.reshape(pred_masks, [-1, 2])
            labels = tf.reshape(self.y, [-1])

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            self.cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):    
                self.train_step = tf.train.AdamOptimizer(
                    self.config.learning_rate).minimize(self.cross_entropy,
                        global_step=self.global_step_tensor)
            
            correct_prediction = tf.equal(tf.argmax(pred_masks, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.pred_mask = pred_masks


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

