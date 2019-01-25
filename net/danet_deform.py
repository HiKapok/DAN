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
from utility import custom_op

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

    def se_inception_block(self, features, name=None, reuse=None):
        with tf.variable_scope(name, 'se_inception_block', reuse=reuse):
            ch_axis = -1 if self._data_format == 'channels_last' else 1
            top_channels = features.get_shape().as_list()[ch_axis]

            conv_1x1_down = tf.layers.conv2d(features, 256, (1, 1), strides=1,
                                name='conv_1x1_down', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            deform_conv = custom_op.deform_conv_2d(conv_1x1_down, 256, kernel_size_h=3, kernel_size_w=3, stride=1, dilate_rate=1, deformable_group=4, data_format=self._data_format, no_bias=False, kernel_initializer=self._conv_initializer, name='deform_conv')
            # deform_conv2 = custom_op.deform_conv_2d(tf.nn.relu(deform_conv1), 256, kernel_size_h=3, kernel_size_w=3, stride=1, dilate_rate=1, deformable_group=4, data_format=self._data_format, no_bias=False, kernel_initializer=self._conv_initializer, name='deform_conv2')
            # deform_conv3 = custom_op.deform_conv_2d(tf.nn.relu(deform_conv2), 256, kernel_size_h=3, kernel_size_w=3, stride=1, dilate_rate=1, deformable_group=4, data_format=self._data_format, no_bias=False, kernel_initializer=self._conv_initializer, name='deform_conv3')

            conv_1x1_up = tf.layers.conv2d(tf.nn.relu(deform_conv), top_channels, (1, 1), strides=1,
                                name='conv_1x1_up', use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)

            return conv_1x1_up + features

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
                feat_stage1 = tf.stop_gradient(feature_stage1[ind])
                conv_stage1 = tf.layers.conv2d(feat_stage1, top_channels//3, (1, 1), strides=1,
                                name='satge1_conv_1x1_{}'.format(ind), use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
                conv_residual= tf.layers.conv2d(feat, top_channels - top_channels//3, (1, 1), strides=1,
                                name='residual_conv_1x1_{}'.format(ind), use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
                feat = tf.concat([conv_stage1, conv_residual], axis=axis)
                feat = self.se_inception_block(feat, name='predict_stage2_{}'.format(ind), reuse=tf.AUTO_REUSE)
                feature_maps.append(feat)

            return feature_maps

    def get_features_stage1_conv_only(self, feature_layers, name=None, reuse=None):
        with tf.variable_scope(name, 'prediction_modules_stage1', reuse=tf.AUTO_REUSE):
            feature_maps = []
            axis = -1 if self._data_format == 'channels_last' else 1
            for ind, feat in enumerate(feature_layers):
                top_channels = feat.get_shape().as_list()[axis]

                feat = tf.nn.relu(custom_op.deform_conv_2d(feat, top_channels, kernel_size_h=3, kernel_size_w=3, stride=1, dilate_rate=1, deformable_group=4, data_format=self._data_format, no_bias=False, kernel_initializer=self._conv_initializer, name='predict_stage1_conv{}'.format(ind)))

                feature_maps.append(feat)

            return feature_maps

    def get_features_stage2_conv_only(self, feature_stage1, feature_layers, name=None, reuse=None):
        with tf.variable_scope(name, 'prediction_modules_stage2', reuse=tf.AUTO_REUSE):
            feature_maps = []
            axis = -1 if self._data_format == 'channels_last' else 1
            for ind, feat in enumerate(feature_layers):
                top_channels = feat.get_shape().as_list()[axis]
                feat_stage1 = tf.stop_gradient(feature_stage1[ind])
                conv_stage1 = tf.layers.conv2d(feat_stage1, top_channels//3, (1, 1), strides=1,
                                name='satge1_conv_1x1_{}'.format(ind), use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
                conv_residual= tf.layers.conv2d(feat, top_channels - top_channels//3, (1, 1), strides=1,
                                name='residual_conv_1x1_{}'.format(ind), use_bias=True, padding='same',
                                dilation_rate=(1, 1),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
                feat = tf.concat([conv_stage1, conv_residual], axis=axis)

                feat = tf.nn.relu(custom_op.deform_conv_2d(feat, top_channels, kernel_size_h=3, kernel_size_w=3, stride=1, dilate_rate=1, deformable_group=4, data_format=self._data_format, no_bias=False, kernel_initializer=self._conv_initializer, name='predict_stage2_conv{}'.format(ind)))

                feature_maps.append(feat)

            return feature_maps
