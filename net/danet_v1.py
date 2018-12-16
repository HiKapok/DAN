# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True

# vgg_16/conv2/conv2_1/biases
# vgg_16/conv4/conv4_3/biases
# vgg_16/conv1/conv1_1/biases
# vgg_16/fc6/weights
# vgg_16/conv3/conv3_2/biases
# vgg_16/conv5/conv5_3/biases
# vgg_16/conv3/conv3_1/weights
# vgg_16/conv4/conv4_2/weights
# vgg_16/conv1/conv1_1/weights
# vgg_16/conv5/conv5_3/weights
# vgg_16/conv4/conv4_1/weights
# vgg_16/conv3/conv3_3/weights
# vgg_16/conv5/conv5_2/biases
# vgg_16/conv3/conv3_2/weights
# vgg_16/conv4/conv4_2/biases
# vgg_16/conv5/conv5_2/weights
# vgg_16/conv3/conv3_1/biases
# vgg_16/conv2/conv2_2/weights
# vgg_16/fc7/weights
# vgg_16/conv5/conv5_1/biases
# vgg_16/conv1/conv1_2/biases
# vgg_16/conv2/conv2_2/biases
# vgg_16/conv4/conv4_1/biases
# vgg_16/fc7/biases
# vgg_16/fc6/biases
# vgg_16/conv4/conv4_3/weights
# vgg_16/conv2/conv2_1/weights
# vgg_16/conv5/conv5_1/weights
# vgg_16/conv3/conv3_3/biases
# vgg_16/conv1/conv1_2/weights

class VGG16Backbone(object):
    def __init__(self, data_format='channels_first', bn_epsilon=1e-5, bn_momentum=0.997, use_fused_bn=True):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        self._bn_epsilon = bn_epsilon
        self._bn_momentum = bn_momentum
        self._use_fused_bn = use_fused_bn
        #initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)

    def l2_normalize(self, inputs, init_value, training, name=None):
        with tf.variable_scope(name, "l2_normalize", [inputs]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            num_channels = inputs.get_shape().as_list()[axis]
            weight_scale = tf.get_variable(name='weight', shape=(num_channels,), trainable=training, initializer=tf.constant_initializer([init_value] * num_channels))
            if self._data_format == 'channels_last':
                weight_scale = tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            else:
                weight_scale = tf.reshape(weight_scale, [1, -1, 1, 1], name='reshape')
            square_sum = tf.reduce_sum(tf.square(inputs), axis, keepdims=True)
            inputs_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(tf.multiply(inputs, inputs_inv_norm, 'normalize'), weight_scale, 'rescale')

    def conv_relu(self, inputs, filters, kernel_size, strides, scope, padding='same', dilate_rate=1, reuse=None):
        with tf.variable_scope(scope, 'conv_relu', [inputs]):
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=True, padding=padding,
                                dilation_rate=(dilate_rate, dilate_rate),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            return inputs

    def conv_bn_relu(self, inputs, filters, kernel_size, strides, scope, training, padding='same', dilate_rate=1, reuse=None):
        with tf.variable_scope(scope, 'conv_bn_relu', [inputs]) as sc:
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=False, padding=padding,
                                dilation_rate=(dilate_rate, dilate_rate),
                                data_format=self._data_format, activation=None,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=None, reuse=reuse)
            inputs = tf.layers.batch_normalization(inputs, axis=self._bn_axis, training=training, name='bn',
                                reuse=reuse, momentum=self._bn_momentum, epsilon=self._bn_epsilon, fused=self._use_fused_bn)
            return tf.nn.relu(inputs)

    def bn_relu(self, inputs, scope, training, reuse=None):
        with tf.variable_scope(scope, 'bn_relu', [inputs]) as sc:
            inputs = tf.layers.batch_normalization(inputs, axis=self._bn_axis, training=training, name='bn',
                                reuse=reuse, momentum=self._bn_momentum, epsilon=self._bn_epsilon, fused=self._use_fused_bn)
            return tf.nn.relu(inputs)

    def conv_bn(self, inputs, filters, kernel_size, strides, scope, training, padding='same', dilate_rate=1, reuse=None):
        with tf.variable_scope(scope, 'conv_bn', [inputs]) as sc:
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=False, padding=padding,
                                dilation_rate=(dilate_rate, dilate_rate),
                                data_format=self._data_format, activation=None,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=None, reuse=reuse)
            inputs = tf.layers.batch_normalization(inputs, axis=self._bn_axis, training=training, name='bn',
                                reuse=reuse, momentum=self._bn_momentum, epsilon=self._bn_epsilon, fused=self._use_fused_bn)
            return inputs

    def conv_block(self, inputs, num_blocks, filters, kernel_size, strides, name, reuse=None):
        with tf.variable_scope(name):
            for ind in range(1, num_blocks + 1):
                inputs = self.conv_relu(inputs, filters, kernel_size, strides, '{}_{}'.format(name, ind), reuse=reuse)
            return inputs

    def get_featmaps(self, inputs, training=False):
        # inputs should in BGR
        feature_layers = []
        # forward vgg layers
        inputs = self.conv_block(inputs, 2, 64, (3, 3), (1, 1), 'conv1')
        inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], padding='same', data_format=self._data_format, name='pool_1')
        inputs = self.conv_block(inputs, 2, 128, (3, 3), (1, 1), 'conv2')
        inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], padding='same', data_format=self._data_format, name='pool_2')
        inputs = self.conv_block(inputs, 3, 256, (3, 3), (1, 1), 'conv3')
        feature_layers.append(self.l2_normalize(inputs, 10, training, 'l2_norm_layer_3'))
        inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], padding='same', data_format=self._data_format, name='pool_3')
        inputs = self.conv_block(inputs, 3, 512, (3, 3), (1, 1), 'conv4')
        feature_layers.append(self.l2_normalize(inputs, 8, training, 'l2_norm_layer_4'))
        inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], padding='same', data_format=self._data_format, name='pool_4')
        inputs = self.conv_block(inputs, 3, 512, (3, 3), (1, 1), 'conv5')
        feature_layers.append(self.l2_normalize(inputs, 5, training, 'l2_norm_layer_5'))
        inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], padding='same', data_format=self._data_format, name='pool_5')

        inputs = self.conv_relu(inputs, 1024, (3, 3), (1, 1), 'fc6', padding='same', dilate_rate=1, reuse=None)
        inputs = self.conv_relu(inputs, 1024, (1, 1), (1, 1), 'fc7', padding='same', reuse=None)
        feature_layers.append(inputs)
        # SFD layers
        with tf.variable_scope('additional_layers') as scope:
            inputs = self.conv_relu(inputs, 256, (1, 1), (1, 1), 'conv6_1', padding='same', reuse=None)
            inputs = self.conv_relu(inputs, 512, (3, 3), (2, 2), 'conv6_2', padding='same', reuse=None)
            feature_layers.append(inputs)
            inputs = self.conv_relu(inputs, 128, (1, 1), (1, 1), 'conv7_1', padding='same', reuse=None)
            inputs = self.conv_relu(inputs, 256, (3, 3), (2, 2), 'conv7_2', padding='same', reuse=None)
            feature_layers.append(inputs)
        return feature_layers

    # def build_lfpn(self, feature_layers, skip_last=3):
    #     output_layers = []
    #     with tf.variable_scope('lfpn'):
    #         axis = -1 if self._data_format == 'channels_last' else 1
    #         for ind, featmap in enumerate(feature_layers[:skip_last]):
    #             with tf.variable_scope('fpn_{}'.format(ind)):
    #                 top_channels = feature_layers[ind+1].get_shape().as_list()[axis]
    #                 down_channels = featmap.get_shape().as_list()[axis]
    #                 up_sampling = tf.layers.conv2d(feature_layers[ind+1], down_channels, (1, 1), strides=1,
    #                                     name='lateral', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)
    #                 if self._data_format == 'channels_first':
    #                     up_sampling_shape = tf.shape(featmap)[-2:]
    #                     up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans')
    #                 else:
    #                     up_sampling_shape = tf.shape(featmap)[1:-1]

    #                 up_sampling = tf.image.resize_bilinear(up_sampling, up_sampling_shape, name='upsample')
    #                 if self._data_format == 'channels_first':
    #                     up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv')
    #                 featmap = featmap * up_sampling
    #                 featmap = tf.layers.conv2d(featmap, down_channels, (3, 3), strides=1,
    #                                     name='fused_conv', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)
    #                 output_layers.append(featmap)
    #         output_layers = output_layers + feature_layers[skip_last:]
    #         return output_layers


    def build_bi_lfpn(self, feature_layers, skip_last=3, name=None):
        output_layers = []
        post_output_layers = []
        with tf.variable_scope(name, 'lfpn'):
            axis = -1 if self._data_format == 'channels_last' else 1
            for ind, featmap in enumerate(feature_layers[:skip_last]):
                with tf.variable_scope('fpn_{}'.format(ind)):
                    top_channels = feature_layers[ind+1].get_shape().as_list()[axis]
                    down_channels = featmap.get_shape().as_list()[axis]
                    up_sampling = tf.layers.conv2d(feature_layers[ind+1], down_channels, (1, 1), strides=1,
                                        name='lateral', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    if self._data_format == 'channels_first':
                        up_sampling_shape = tf.shape(featmap)[-2:]
                        up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans')
                    else:
                        up_sampling_shape = tf.shape(featmap)[1:-1]

                    up_sampling = tf.image.resize_bilinear(up_sampling, up_sampling_shape, name='upsample')
                    if self._data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv')
                    featmap = featmap + up_sampling
                    featmap = tf.layers.conv2d(featmap, 256, (3, 3), strides=1,
                                        name='fused_conv', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    output_layers.append(featmap)
            for ind, featmap in enumerate(feature_layers[skip_last:-1]):
                ind = ind + skip_last
                with tf.variable_scope('fpn_{}'.format(ind)):
                    top_channels = feature_layers[ind+1].get_shape().as_list()[axis]
                    down_channels = featmap.get_shape().as_list()[axis]
                    up_sampling = tf.layers.conv2d(feature_layers[ind+1], down_channels, (1, 1), strides=1,
                                        name='lateral', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    if self._data_format == 'channels_first':
                        up_sampling_shape = tf.shape(featmap)[-2:]
                        up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans')
                    else:
                        up_sampling_shape = tf.shape(featmap)[1:-1]

                    up_sampling = tf.image.resize_bilinear(up_sampling, up_sampling_shape, name='upsample')
                    if self._data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv')
                    featmap = featmap + up_sampling
                    featmap = tf.layers.conv2d(featmap, 256, (3, 3), strides=1,
                                        name='fused_conv', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    post_output_layers.append(featmap)
            output_layers = output_layers + post_output_layers + [feature_layers[-1]]
            return output_layers

    # def build_reverse_lfpn(self, feature_layers, skip_last=3, name=None):
    #     output_layers = []
    #     post_output_layers = []
    #     with tf.variable_scope(name, 'reverse_lfpn'):
    #         axis = -1 if self._data_format == 'channels_last' else 1
    #         num_layers = len(feature_layers)
    #         skip_last = num_layers - 1 - skip_last
    #         feature_layers = list(reversed(feature_layers))
    #         for ind, featmap in enumerate(feature_layers[:skip_last]):
    #             with tf.variable_scope('fpn_{}'.format(num_layers - 2 - ind)):
    #                 top_channels = feature_layers[ind+1].get_shape().as_list()[axis]
    #                 down_channels = featmap.get_shape().as_list()[axis]
    #                 up_sampling = tf.layers.conv2d(feature_layers[ind+1], down_channels, (3, 3), strides=2,
    #                                     name='lateral', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)

    #                 featmap = featmap + up_sampling
    #                 featmap = tf.layers.conv2d(featmap, 256, (3, 3), strides=1,
    #                                     name='fused_conv', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)
    #                 output_layers.append(featmap)
    #         for ind, featmap in enumerate(feature_layers[skip_last:-1]):
    #             ind = ind + skip_last
    #             with tf.variable_scope('fpn_{}'.format(num_layers - 2 - ind)):
    #                 top_channels = feature_layers[ind+1].get_shape().as_list()[axis]
    #                 down_channels = featmap.get_shape().as_list()[axis]
    #                 up_sampling = tf.layers.conv2d(feature_layers[ind+1], down_channels, (3, 3), strides=2,
    #                                     name='lateral', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)

    #                 featmap = featmap + up_sampling
    #                 featmap = tf.layers.conv2d(featmap, 256, (3, 3), strides=1,
    #                                     name='fused_conv', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)
    #                 post_output_layers.append(featmap)
    #         output_layers = output_layers + post_output_layers + [feature_layers[-1]]
    #         return list(reversed(output_layers))



    def build_lfpn(self, feature_layers, skip_last=3, name=None):
        output_layers = []
        with tf.variable_scope(name, 'lfpn'):
            axis = -1 if self._data_format == 'channels_last' else 1
            up_sampling = None

            for ind in range(skip_last, 0, -1):
            #for ind, featmap in enumerate(reversed(feature_layers[:(skip_last + 1)])):
                with tf.variable_scope('fpn_{}'.format(ind - 1)):
                    top_channels = feature_layers[ind].get_shape().as_list()[axis]
                    down_channels = feature_layers[ind-1].get_shape().as_list()[axis]

                    if up_sampling is None:
                        up_sampling = feature_layers[ind]
                    up_sampling = tf.layers.conv2d(up_sampling, down_channels, (1, 1), strides=1,
                                            name='upsample_conv', use_bias=True, padding='same',
                                            data_format=self._data_format, activation=None,
                                            kernel_initializer=self._conv_initializer(),
                                            bias_initializer=tf.zeros_initializer(), reuse=None)
                    lateral = tf.layers.conv2d(feature_layers[ind-1], down_channels, (1, 1), strides=1,
                                        name='lateral', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    if self._data_format == 'channels_first':
                        up_sampling_shape = tf.shape(lateral)[-2:]
                        up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans')
                    else:
                        up_sampling_shape = tf.shape(lateral)[1:-1]

                    up_sampling = tf.image.resize_bilinear(up_sampling, up_sampling_shape, name='upsample')
                    if self._data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv')
                    up_sampling = lateral + up_sampling
                    featmap = tf.layers.conv2d(up_sampling, 256, (3, 3), strides=1,
                                        name='fused_conv', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    output_layers.append(featmap)
            output_layers = list(reversed(output_layers)) + feature_layers[skip_last:]
            return output_layers

    def build_reverse_lfpn(self, feature_layers, skip_last=3, name=None):
        output_layers = []
        with tf.variable_scope(name, 'reverse_lfpn'):
            axis = -1 if self._data_format == 'channels_last' else 1
            down_sampling = None

            for ind in range(0, skip_last, 1):
                with tf.variable_scope('reverse_fpn_{}'.format(ind)):
                    top_channels = feature_layers[ind].get_shape().as_list()[axis]
                    down_channels = feature_layers[ind+1].get_shape().as_list()[axis]

                    if down_sampling is None:
                        down_sampling = feature_layers[ind]
                    down_sampling = tf.layers.conv2d(down_sampling, down_channels, (3, 3), strides=2,
                                        name='downsample_conv', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    lateral = tf.layers.conv2d(feature_layers[ind+1], down_channels, (1, 1), strides=1,
                                        name='lateral', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    down_sampling = lateral + down_sampling
                    featmap = tf.layers.conv2d(down_sampling, 256, (3, 3), strides=1,
                                        name='fused_conv', use_bias=True, padding='same',
                                        data_format=self._data_format, activation=None,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(), reuse=None)
                    output_layers.append(featmap)
            output_layers = [feature_layers[0]] + output_layers + feature_layers[(skip_last + 1):]
            return output_layers


    # def build_bifpn(self, feature_layers, skip_last=3):
    #     output_layers = []
    #     with tf.variable_scope('bifpn'):
    #         axis = -1 if self._data_format == 'channels_last' else 1
    #         for ind, featmap in enumerate(feature_layers[:skip_last]):
    #             with tf.variable_scope('fpn_{}'.format(ind)):
    #                 top_channels = feature_layers[ind+1].get_shape().as_list()[axis]
    #                 down_channels = featmap.get_shape().as_list()[axis]
    #                 up_sampling = tf.layers.conv2d(feature_layers[ind+1], down_channels, (1, 1), strides=1,
    #                                     name='lateral', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)
    #                 if self._data_format == 'channels_first':
    #                     up_sampling_shape = tf.shape(featmap)[-2:]
    #                     up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans')
    #                 else:
    #                     up_sampling_shape = tf.shape(featmap)[1:-1]

    #                 up_sampling = tf.image.resize_bilinear(up_sampling, up_sampling_shape, name='upsample')
    #                 if self._data_format == 'channels_first':
    #                     up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv')
    #                 featmap = featmap * up_sampling
    #                 featmap = tf.layers.conv2d(featmap, down_channels, (3, 3), strides=1,
    #                                     name='fused_conv', use_bias=True, padding='same',
    #                                     data_format=self._data_format, activation=None,
    #                                     kernel_initializer=self._conv_initializer(),
    #                                     bias_initializer=tf.zeros_initializer(), reuse=None)
    #                 output_layers.append(featmap)
    #         output_layers = output_layers + feature_layers[skip_last:]
    #         return output_layers


    # def get_shift_upsampled_feature(self, feature_layers, shift=1, name=None):
    #     with tf.variable_scope(name, 'shift_upsampled'):
    #         feature_outputs = list()
    #         axis = -1 if self._data_format == 'channels_last' else 1
    #         for ind, feat in enumerate(feature_layers[shift:]):
    #             top_channels = feat.get_shape().as_list()[axis]
    #             feature_outputs.append(tf.layers.conv2d_transpose(feat, top_channels, (3, 3), strides=(2, 2),
    #                         padding='same', data_format=self._data_format, activation=None, use_bias=True,
    #                         kernel_initializer=self._conv_initializer(),
    #                         bias_initializer=tf.zeros_initializer(), name='upsample_conv_{}'.format(ind)))
    #         for ind, feat in enumerate(feature_layers[-shift:]):
    #             top_channels = feat.get_shape().as_list()[axis]
    #             feature_outputs.append(tf.layers.conv2d(feat, top_channels, (3, 3), use_bias=True,
    #                         name='upsample_conv_{}'.format(ind + len(feature_layers) - shift), strides=(1, 1),
    #                         padding='same', data_format=self._data_format, activation=None,
    #                         kernel_initializer=self._conv_initializer(),
    #                         bias_initializer=tf.zeros_initializer()))

    #         return feature_outputs

    def get_predict_module(self, feature_layers, pos_maxout, neg_maxout, num_anchors_depth_per_layer, name=None):
        with tf.variable_scope(name, 'predict_face'):
            cls_preds = []
            loc_preds = []
            axis = -1 if self._data_format == 'channels_last' else 1
            for ind, feat in enumerate(feature_layers):
                top_channels = feat.get_shape().as_list()[axis]
                feat = self.conv_relu(feat, top_channels, (3, 3), (1, 1), 'shared_conv_{}'.format(ind), padding='same', reuse=None)
                loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
                            name='loc_{}'.format(ind), strides=(1, 1),
                            padding='same', data_format=self._data_format, activation=None,
                            kernel_initializer=self._conv_initializer(),
                            bias_initializer=tf.zeros_initializer()))
                cls_pred = tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * (pos_maxout[ind] + neg_maxout[ind]), (3, 3), use_bias=True,
                            name='cls_{}'.format(ind), strides=(1, 1),
                            padding='same', data_format=self._data_format, activation=None,
                            kernel_initializer=self._conv_initializer(),
                            bias_initializer=tf.zeros_initializer())

                if pos_maxout[ind] + neg_maxout[ind] > 2:
                    if self._data_format == 'channels_first':
                        num_batch, num_channels, feat_height, feat_width = tf.unstack(tf.shape(cls_pred))
                        target_shape = tf.stack([num_batch, num_anchors_depth_per_layer[ind], -1, feat_height, feat_width])
                        final_shape = tf.stack([num_batch, num_anchors_depth_per_layer[ind]*2, feat_height, feat_width])
                        cls_pred = tf.reshape(cls_pred, target_shape)
                        if pos_maxout[ind] > 1:
                            pos_cls_score = tf.reduce_max(cls_pred[:, :, neg_maxout[ind]:, :, :], axis=2)
                        else:
                            pos_cls_score = cls_pred[:, :, -1, :, :]
                        if neg_maxout[ind] > 1:
                            neg_cls_score = tf.reduce_max(cls_pred[:, :, :neg_maxout[ind], :, :], axis=2)
                        else:
                            neg_cls_score = cls_pred[:, :, 0, :, :]
                        neg_cls_score = tf.reshape(neg_cls_score, target_shape)
                        pos_cls_score = tf.reshape(pos_cls_score, target_shape)
                        #print(neg_cls_score, pos_cls_score)
                        cls_pred = tf.concat([neg_cls_score, pos_cls_score], axis=2)
                        #print(cls_pred)
                        #cls_pred = tf.Print(cls_pred, [tf.shape(cls_pred), tf.shape(neg_cls_score), tf.shape(pos_cls_score)], summarize=10)
                        cls_pred = tf.reshape(cls_pred, final_shape)
                    else:
                        num_batch, feat_height, feat_width, num_channels = tf.unstack(tf.shape(cls_pred))
                        target_shape = tf.stack([num_batch, feat_height, feat_width, num_anchors_depth_per_layer[ind], -1])
                        final_shape = tf.stack([num_batch, feat_height, feat_width, num_anchors_depth_per_layer[ind]*2])
                        cls_pred = tf.reshape(cls_pred, target_shape)
                        if pos_maxout[ind] > 1:
                            pos_cls_score = tf.reduce_max(cls_pred[:, :, :, :, neg_maxout[ind]:], axis=-1)
                        else:
                            pos_cls_score = cls_pred[:, :, :, :, -1]
                        if neg_maxout[ind] > 1:
                            neg_cls_score = tf.reduce_max(cls_pred[:, :, :, :, :neg_maxout[ind]], axis=-1)
                        else:
                            neg_cls_score = cls_pred[:, :, :, :, 0]
                        neg_cls_score = tf.reshape(neg_cls_score, target_shape)
                        pos_cls_score = tf.reshape(pos_cls_score, target_shape)
                        #print(neg_cls_score, pos_cls_score)
                        cls_pred = tf.concat([neg_cls_score, pos_cls_score], axis=-1)
                        # print(cls_pred)
                        # cls_pred = tf.Print(cls_pred, [tf.shape(cls_pred), tf.shape(neg_cls_score), tf.shape(pos_cls_score)], summarize=10)

                        cls_pred = tf.reshape(cls_pred, final_shape)
                cls_preds.append(cls_pred)

            return loc_preds, cls_preds

    # def large_separable_kernel(self, features, kernel_size, num_mid, num_out, name=None, reuse=None):
    #     with tf.variable_scope(name, 'large_separable_kernel', reuse=reuse):
    #         left_conv1 = tf.layers.conv2d(features, num_mid, (kernel_size, 1), strides=1,
    #                             name='left_conv1', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=None,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         left_conv2 = tf.layers.conv2d(left_conv1, num_out, (1, kernel_size), strides=1,
    #                             name='left_conv2', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=None,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         right_conv1 = tf.layers.conv2d(features, num_mid, (1, kernel_size), strides=1,
    #                             name='right_conv1', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=None,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         right_conv2 = tf.layers.conv2d(right_conv1, num_out, (kernel_size, 1), strides=1,
    #                             name='right_conv2', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=None,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)

    #         return tf.nn.relu(left_conv2 + right_conv2)
    # def adaptive_feature_aggregation(self, features, num_channels, name=None, reuse=None):
    #     with tf.variable_scope(name, 'adaptive_feature_aggregation', reuse=reuse):
    #         channels_per_feature = num_channels // 4
    #         ch_axis = -1 if self._data_format == 'channels_last' else 1

    #         feature_1x1 = tf.layers.conv2d(features, channels_per_feature, (1, 1), strides=1,
    #                             name='conv_1x1', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=tf.nn.relu,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         feature_3x3 = tf.layers.conv2d(features, channels_per_feature, (3, 3), strides=1,
    #                             name='conv_3x3', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=tf.nn.relu,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         feature_5x5 = self.large_separable_kernel(features, 5, channels_per_feature//4, channels_per_feature, name='conv_5x5', reuse=reuse)
    #         feature_7x7 = self.large_separable_kernel(features, 7, channels_per_feature//4, channels_per_feature, name='conv_7x7', reuse=reuse)
    #         feature_hyper_column = tf.concat([feature_1x1, feature_3x3, feature_5x5, feature_7x7], axis=ch_axis)

    #         scale_features = tf.layers.conv2d(features, num_channels, (1, 1), strides=1,
    #                             name='conv_scale', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=tf.nn.relu,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         conv_down = tf.layers.conv2d(scale_features, num_channels//8, (3, 3), strides=1,
    #                             name='conv_down_1x1', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=tf.nn.relu,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)
    #         conv_up = tf.layers.conv2d(conv_down, num_channels, (1, 1), strides=1,
    #                             name='conv_up_1x1', use_bias=True, padding='same',
    #                             dilation_rate=(1, 1),
    #                             data_format=self._data_format, activation=tf.nn.sigmoid,
    #                             kernel_initializer=self._conv_initializer(),
    #                             bias_initializer=tf.zeros_initializer(), reuse=reuse)

    #         return feature_hyper_column * conv_up + features

    # def get_se_predict_module(self, feature_layers, pos_maxout, neg_maxout, num_anchors_depth_per_layer, name=None, reuse=None):
    #     with tf.variable_scope(name, 'se_predict_face', reuse=reuse):
    #         cls_preds = []
    #         loc_preds = []
    #         axis = -1 if self._data_format == 'channels_last' else 1
    #         for ind, feat in enumerate(feature_layers):
    #             top_channels = feat.get_shape().as_list()[axis]

    #             #feat = self.conv_relu(feat, top_channels, (3, 3), (1, 1), 'shared_conv_{}'.format(ind), padding='same', reuse=None)

    #             feat = self.adaptive_feature_aggregation(feat, top_channels, 'shared_conv_{}'.format(ind))
    #             loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
    #                         name='loc_{}'.format(ind), strides=(1, 1),
    #                         padding='same', data_format=self._data_format, activation=None,
    #                         kernel_initializer=self._conv_initializer(),
    #                         bias_initializer=tf.zeros_initializer()))
    #             cls_pred = tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * (pos_maxout[ind] + neg_maxout[ind]), (3, 3), use_bias=True,
    #                         name='cls_{}'.format(ind), strides=(1, 1),
    #                         padding='same', data_format=self._data_format, activation=None,
    #                         kernel_initializer=self._conv_initializer(),
    #                         bias_initializer=tf.zeros_initializer())

    #             if pos_maxout[ind] + neg_maxout[ind] > 2:
    #                 if self._data_format == 'channels_first':
    #                     num_batch, num_channels, feat_height, feat_width = tf.unstack(tf.shape(cls_pred))
    #                     target_shape = tf.stack([num_batch, num_anchors_depth_per_layer[ind], -1, feat_height, feat_width])
    #                     final_shape = tf.stack([num_batch, num_anchors_depth_per_layer[ind]*2, feat_height, feat_width])
    #                     cls_pred = tf.reshape(cls_pred, target_shape)
    #                     if pos_maxout[ind] > 1:
    #                         pos_cls_score = tf.reduce_max(cls_pred[:, :, neg_maxout[ind]:, :, :], axis=2)
    #                     else:
    #                         pos_cls_score = cls_pred[:, :, -1, :, :]
    #                     if neg_maxout[ind] > 1:
    #                         neg_cls_score = tf.reduce_max(cls_pred[:, :, :neg_maxout[ind], :, :], axis=2)
    #                     else:
    #                         neg_cls_score = cls_pred[:, :, 0, :, :]
    #                     neg_cls_score = tf.reshape(neg_cls_score, target_shape)
    #                     pos_cls_score = tf.reshape(pos_cls_score, target_shape)
    #                     #print(neg_cls_score, pos_cls_score)
    #                     cls_pred = tf.concat([neg_cls_score, pos_cls_score], axis=2)
    #                     #print(cls_pred)
    #                     #cls_pred = tf.Print(cls_pred, [tf.shape(cls_pred), tf.shape(neg_cls_score), tf.shape(pos_cls_score)], summarize=10)
    #                     cls_pred = tf.reshape(cls_pred, final_shape)
    #                 else:
    #                     num_batch, feat_height, feat_width, num_channels = tf.unstack(tf.shape(cls_pred))
    #                     target_shape = tf.stack([num_batch, feat_height, feat_width, num_anchors_depth_per_layer[ind], -1])
    #                     final_shape = tf.stack([num_batch, feat_height, feat_width, num_anchors_depth_per_layer[ind]*2])
    #                     cls_pred = tf.reshape(cls_pred, target_shape)
    #                     if pos_maxout[ind] > 1:
    #                         pos_cls_score = tf.reduce_max(cls_pred[:, :, :, :, neg_maxout[ind]:], axis=-1)
    #                     else:
    #                         pos_cls_score = cls_pred[:, :, :, :, -1]
    #                     if neg_maxout[ind] > 1:
    #                         neg_cls_score = tf.reduce_max(cls_pred[:, :, :, :, :neg_maxout[ind]], axis=-1)
    #                     else:
    #                         neg_cls_score = cls_pred[:, :, :, :, 0]
    #                     neg_cls_score = tf.reshape(neg_cls_score, target_shape)
    #                     pos_cls_score = tf.reshape(pos_cls_score, target_shape)
    #                     #print(neg_cls_score, pos_cls_score)
    #                     cls_pred = tf.concat([neg_cls_score, pos_cls_score], axis=-1)
    #                     # print(cls_pred)
    #                     # cls_pred = tf.Print(cls_pred, [tf.shape(cls_pred), tf.shape(neg_cls_score), tf.shape(pos_cls_score)], summarize=10)

    #                     cls_pred = tf.reshape(cls_pred, final_shape)
    #             cls_preds.append(cls_pred)

    #         return loc_preds, cls_preds

    # def get_density_se_predict_module(self, feature_layers, pos_maxout, neg_maxout, num_anchors_depth_per_layer, offsets=None, var_name_ind=None, name=None, reuse=None):
    #     with tf.variable_scope(name, 'se_predict_face', reuse=tf.AUTO_REUSE):
    #         if offsets is None: offsets = [None] * len(feature_layers)
    #         cls_preds = []
    #         loc_preds = []
    #         axis = -1 if self._data_format == 'channels_last' else 1
    #         for ind, feat in enumerate(feature_layers):
    #             top_channels = feat.get_shape().as_list()[axis]
    #             #feat = self.conv_relu(feat, top_channels, (3, 3), (1, 1), 'shared_conv_{}'.format(var_name_ind[ind]), padding='same', reuse=tf.AUTO_REUSE)
    #             feat = self.adaptive_feature_aggregation(feat, top_channels, 'shared_conv_{}'.format(var_name_ind[ind]), reuse=tf.AUTO_REUSE)
    #             if offsets[ind] is not None:
    #                 if self._data_format == 'channels_first':
    #                     feats = tf.transpose(feat, [0, 2, 3, 1])
    #                 else:
    #                     feats = feat
    #                 feat_size = tf.shape(feats)[1:-1]
    #                 offset_h, offset_w = offsets[ind]
    #                 feat = tf.image.crop_and_resize(feats, tf.expand_dims(tf.stack([offset_h/tf.to_float(feat_size[0]), offset_w/tf.to_float(feat_size[1]), 1. + offset_h/tf.to_float(feat_size[0]), 1. + offset_w/tf.to_float(feat_size[1])], axis=-1), axis=0), tf.constant([0]), feat_size, method='bilinear', extrapolation_value=0)
    #                 if self._data_format == 'channels_first':
    #                     feat = tf.transpose(feat, [0, 3, 1, 2])


    #             loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
    #                         name='loc_{}'.format(var_name_ind[ind]), strides=(1, 1),
    #                         padding='same', data_format=self._data_format, activation=None,
    #                         kernel_initializer=self._conv_initializer(),
    #                         bias_initializer=tf.zeros_initializer(), reuse=tf.AUTO_REUSE))
    #             cls_pred = tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * (pos_maxout[ind] + neg_maxout[ind]), (3, 3), use_bias=True,
    #                         name='cls_{}'.format(var_name_ind[ind]), strides=(1, 1),
    #                         padding='same', data_format=self._data_format, activation=None,
    #                         kernel_initializer=self._conv_initializer(),
    #                         bias_initializer=tf.zeros_initializer(), reuse=tf.AUTO_REUSE)

    #             if pos_maxout[ind] + neg_maxout[ind] > 2:
    #                 if self._data_format == 'channels_first':
    #                     num_batch, num_channels, feat_height, feat_width = tf.unstack(tf.shape(cls_pred))
    #                     target_shape = tf.stack([num_batch, num_anchors_depth_per_layer[ind], -1, feat_height, feat_width])
    #                     final_shape = tf.stack([num_batch, num_anchors_depth_per_layer[ind]*2, feat_height, feat_width])
    #                     cls_pred = tf.reshape(cls_pred, target_shape)
    #                     if pos_maxout[ind] > 1:
    #                         pos_cls_score = tf.reduce_max(cls_pred[:, :, neg_maxout[ind]:, :, :], axis=2)
    #                     else:
    #                         pos_cls_score = cls_pred[:, :, -1, :, :]
    #                     if neg_maxout[ind] > 1:
    #                         neg_cls_score = tf.reduce_max(cls_pred[:, :, :neg_maxout[ind], :, :], axis=2)
    #                     else:
    #                         neg_cls_score = cls_pred[:, :, 0, :, :]
    #                     neg_cls_score = tf.reshape(neg_cls_score, target_shape)
    #                     pos_cls_score = tf.reshape(pos_cls_score, target_shape)
    #                     #print(neg_cls_score, pos_cls_score)
    #                     cls_pred = tf.concat([neg_cls_score, pos_cls_score], axis=2)
    #                     #print(cls_pred)
    #                     #cls_pred = tf.Print(cls_pred, [tf.shape(cls_pred), tf.shape(neg_cls_score), tf.shape(pos_cls_score)], summarize=10)
    #                     cls_pred = tf.reshape(cls_pred, final_shape)
    #                 else:
    #                     num_batch, feat_height, feat_width, num_channels = tf.unstack(tf.shape(cls_pred))
    #                     target_shape = tf.stack([num_batch, feat_height, feat_width, num_anchors_depth_per_layer[ind], -1])
    #                     final_shape = tf.stack([num_batch, feat_height, feat_width, num_anchors_depth_per_layer[ind]*2])
    #                     cls_pred = tf.reshape(cls_pred, target_shape)
    #                     if pos_maxout[ind] > 1:
    #                         pos_cls_score = tf.reduce_max(cls_pred[:, :, :, :, neg_maxout[ind]:], axis=-1)
    #                     else:
    #                         pos_cls_score = cls_pred[:, :, :, :, -1]
    #                     if neg_maxout[ind] > 1:
    #                         neg_cls_score = tf.reduce_max(cls_pred[:, :, :, :, :neg_maxout[ind]], axis=-1)
    #                     else:
    #                         neg_cls_score = cls_pred[:, :, :, :, 0]
    #                     neg_cls_score = tf.reshape(neg_cls_score, target_shape)
    #                     pos_cls_score = tf.reshape(pos_cls_score, target_shape)
    #                     #print(neg_cls_score, pos_cls_score)
    #                     cls_pred = tf.concat([neg_cls_score, pos_cls_score], axis=-1)
    #                     # print(cls_pred)
    #                     # cls_pred = tf.Print(cls_pred, [tf.shape(cls_pred), tf.shape(neg_cls_score), tf.shape(pos_cls_score)], summarize=10)

    #                     cls_pred = tf.reshape(cls_pred, final_shape)
    #             cls_preds.append(cls_pred)


    #         return loc_preds, cls_preds

    def se_inception_block(self, features, name=None, reuse=None):
        with tf.variable_scope(name, 'se_inception_block', reuse=reuse):
            ch_axis = -1 if self._data_format == 'channels_last' else 1
            ###### branch1
            branch1_conv_1x1 = tf.layers.conv2d(features, 64, (1, 1), strides=1,
                                name='branch1_conv_1x1', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            ###### branch2
            branch2_avg_pool = tf.layers.average_pooling2d(features, (2, 2), 1, padding='same',
                            data_format=self._data_format, name='branch2_avg_pool')
            branch2_conv_1x1 = tf.layers.conv2d(branch2_avg_pool, 64, (1, 1), strides=1,
                                name='branch2_conv_1x1', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            ###### branch3
            branch3_conv_1x1 = tf.layers.conv2d(features, 64, (1, 1), strides=1,
                                name='branch3_conv_1x1', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            branch3_conv_3x1 = tf.layers.conv2d(branch3_conv_1x1, 32, (3, 1), strides=1,
                                name='branch3_conv_3x1', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            branch3_conv_1x3 = tf.layers.conv2d(branch3_conv_1x1, 32, (1, 3), strides=1,
                                name='branch3_conv_1x3', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            ###### branch4
            branch4_conv_1x1 = tf.layers.conv2d(features, 64, (1, 1), strides=1,
                                name='branch4_conv_1x1', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            branch4_conv_3x3 = tf.layers.conv2d(branch4_conv_1x1, 64, (3, 3), strides=1,
                                name='branch4_conv_3x3', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            branch4_conv_1x3 = tf.layers.conv2d(branch4_conv_3x3, 32, (3, 1), strides=1,
                                name='branch4_conv_1x3', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            branch4_conv_3x1 = tf.layers.conv2d(branch4_conv_3x3, 32, (1, 3), strides=1,
                                name='branch4_conv_3x1', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            ###### concat
            feature_hyper_column = tf.concat([branch1_conv_1x1, branch2_conv_1x1, branch3_conv_3x1, branch3_conv_1x3, branch4_conv_1x3, branch4_conv_3x1], axis=ch_axis)
            ###### rescale
            if self._data_format == 'channels_first':
                pooled_inputs = tf.reduce_mean(feature_hyper_column, [2, 3], name='global_pool', keep_dims=True)
            else:
                pooled_inputs = tf.reduce_mean(feature_hyper_column, [1, 2], name='global_pool', keep_dims=True)

            down_inputs = tf.layers.conv2d(pooled_inputs, 32, (1, 1), use_bias=True,
                                name='proj_down', strides=(1, 1),
                                padding='same', data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer())

            up_inputs = tf.layers.conv2d(down_inputs, 256, (1, 1), use_bias=True,
                                        name='proj_up', strides=(1, 1),
                                        padding='same', data_format=self._data_format, activation=tf.nn.sigmoid,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer())
            return feature_hyper_column * up_inputs

    def get_features_stage1(self, feature_layers, name=None, reuse=None):
        with tf.variable_scope(name, 'prediction_modules_stage1', reuse=tf.AUTO_REUSE):
            feature_maps = []
            axis = -1 if self._data_format == 'channels_last' else 1
            for ind, feat in enumerate(feature_layers):
                top_channels = feat.get_shape().as_list()[axis]
                feat = self.se_inception_block(feat, name='predict_stage1_{}'.format(ind), reuse=tf.AUTO_REUSE)
                feature_maps.append(feat)

            return feature_maps

    def get_features_stage2(self, feature_stage1, feature_layers, name=None, reuse=None):
        with tf.variable_scope(name, 'prediction_modules_stage2', reuse=tf.AUTO_REUSE):
            feature_maps = []
            axis = -1 if self._data_format == 'channels_last' else 1
            for ind, feat in enumerate(feature_layers):
                top_channels = feat.get_shape().as_list()[axis]
                feat_stage1 = feature_stage1[ind]
                conv_stage1 = tf.layers.conv2d(feat_stage1, top_channels//2, (1, 1), strides=1,
                                name='satge1_conv_1x1_{}'.format(ind), use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
                conv_residual= tf.layers.conv2d(feat, top_channels - top_channels//2, (1, 1), strides=1,
                                name='residual_conv_1x1_{}'.format(ind), use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
                feat = tf.concat([conv_stage1, conv_residual], axis=axis)
                feat = self.se_inception_block(feat, name='predict_stage2_{}'.format(ind), reuse=tf.AUTO_REUSE)
                feature_maps.append(feat)

            return feature_maps
