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

import numpy as np
import tensorflow as tf

from utility import bbox_util
from utility import custom_op


'''
v0: the structure in the original paper
v1: the modified version implemented in fair.torch(https://github.com/facebook/fb.resnet.torch)
v2: pre-activation version

v1: This implementation differs from the ResNet paper in a few ways:
    Scale augmentation: We use the scale and aspect ratio augmentation from Going Deeper with Convolutions, instead of scale augmentation used in the ResNet paper. We find this gives a better validation error.
    Color augmentation: We use the photometric distortions from Andrew Howard in addition to the AlexNet-style(https://arxiv.org/abs/1312.5402) color augmentation used in the ResNet paper.
    Weight decay: We apply weight decay to all weights and biases instead of just the weights of the convolution layers.
    Strided convolution: When using the bottleneck architecture, we use stride 2 in the 3x3 convolution, instead of the first 1x1 convolution.
'''

'''
-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style) For V0
function M.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic')
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style) For V1
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end
'''
class ResNetBackbone(object):
    def __init__(self, depth=50, data_format='channels_last', freeze_bn=False, bn_epsilon=1e-05, bn_momentum=0.997, use_fused_bn=True):
        super(ResNetBackbone, self).__init__()
        self._depth = depth
        self._data_format = data_format
        self._bn_axis = -1 if self._data_format == 'channels_last' else 1
        self._bn_trainable = (not freeze_bn)
        self._bn_epsilon = bn_epsilon
        self._bn_momentum = bn_momentum
        self._use_fused_bn = use_fused_bn
        #initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)
        self._block_settings = {
            50:  (3, 4, 6,  3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
        }
    # input image order: BGR, range [-128-128]
    # mean_value: 104, 117, 123
    # only subtract mean is used
    # for root block, use dummy input_filters, e.g. 128 rather than 64 for the first block
    def get_featmaps(self, inputs, training=False, freeze=False):
        input_depth = [128, 256, 512, 1024] # the input depth of the the first block is dummy input

        num_units = self._block_settings[self._depth]
        training_sts = training
        if freeze:
            training = False
        with tf.variable_scope('block_0', [inputs]):
            if self._data_format == 'channels_first':
                inputs = tf.pad(inputs, paddings = [[0, 0], [0, 0], [3, 3], [3, 3]])
            else:
                inputs = tf.pad(inputs, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])

            inputs = self.conv_bn_relu(inputs, input_depth[0] // 2, (7, 7), (2, 2), 'conv_1', training, padding='valid', reuse=None)

            inputs = tf.layers.max_pooling2d(inputs, [3, 3], [2, 2], padding='same', data_format=self._data_format, name='pool_1')

        collected_featmaps = []
        is_root = True
        for ind, num_unit in enumerate(num_units):
            with tf.variable_scope('block_{}'.format(ind+1), [inputs]):
                need_reduce = True
                for unit_index in range(1, num_unit+1):
                    inputs = self.bottleneck_block(inputs, input_depth[ind], 'conv_{}'.format(unit_index), training, need_reduce=need_reduce, is_root=is_root)
                    need_reduce = False
                    is_root = False
                if freeze and (ind==0):
                    # from here, we un-freeze all layers
                    inputs = tf.stop_gradient(inputs)
                    training = training_sts
                collected_featmaps.append(inputs)

        with tf.variable_scope('additional_layers') as scope:
            inputs = self.conv_bn_relu(inputs, 512, (1, 1), (1, 1), 'conv6_1', training, padding='same', reuse=None)
            inputs = self.conv_bn_relu(inputs, 512, (3, 3), (2, 2), 'conv6_2', training, padding='same', reuse=None)
            collected_featmaps.append(inputs)

            inputs = self.conv_bn_relu(inputs, 128, (1, 1), (1, 1), 'conv7_1', training, padding='same', reuse=None)
            inputs = self.conv_bn_relu(inputs, 256, (3, 3), (2, 2), 'conv7_2', training, padding='same', reuse=None)
            collected_featmaps.append(inputs)

        return collected_featmaps

    def bottleneck_block(self, inputs, filters, scope, training, need_reduce=True, is_root=False, reuse=None):
        with tf.variable_scope(scope, 'bottleneck_block', [inputs]):
            strides = 1 if (not need_reduce) or is_root else 2

            shortcut = self.conv_bn(inputs, filters * 2, (1, 1), (strides, strides), 'shortcut', training, padding='valid', reuse=reuse) if need_reduce else inputs

            inputs = self.conv_bn_relu(inputs, filters // 2, (1, 1), (strides, strides), 'reduce', training, padding='valid', reuse=reuse)

            if self._data_format == 'channels_first':
                inputs = tf.pad(inputs, paddings = [[0, 0], [0, 0], [1, 1], [1, 1]])
            else:
                inputs = tf.pad(inputs, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])

            inputs = self.conv_bn_relu(inputs, filters // 2, (3, 3), (1, 1), 'block_3x3', training, padding='valid', reuse=reuse)
            inputs = self.conv_bn(inputs, filters * 2, (1, 1), (1, 1), 'increase', training, reuse=reuse)

            return tf.nn.relu(inputs + shortcut)


    def conv_bn_relu(self, inputs, filters, kernel_size, strides, scope, training, padding='same', dilate_rate=1, reuse=None):
        with tf.variable_scope(scope, 'conv_bn_relu', [inputs]) as sc:
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=False, padding=padding, dilation_rate=(dilate_rate, dilate_rate),
                                data_format=self._data_format, activation=None,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=None, reuse=reuse)
            inputs = tf.layers.batch_normalization(inputs, axis=self._bn_axis, training=(training and self._bn_trainable), trainable=self._bn_trainable, name='bn',
                                reuse=reuse, momentum=self._bn_momentum, epsilon=self._bn_epsilon, fused=self._use_fused_bn)
            with tf.variable_scope("bn", reuse=True):
                tf.add_to_collection('need_restored', tf.get_variable("beta"))
                tf.add_to_collection('need_restored', tf.get_variable("gamma"))
                tf.add_to_collection('need_restored', tf.get_variable("moving_mean"))
                tf.add_to_collection('need_restored', tf.get_variable("moving_variance"))

            return tf.nn.relu(inputs)
    def bn_relu(self, inputs, scope, training, reuse=None):
        with tf.variable_scope(scope, 'bn_relu', [inputs]) as sc:
            inputs = tf.layers.batch_normalization(inputs, axis=self._bn_axis, training=(training and self._bn_trainable), trainable=self._bn_trainable, name='bn',
                                reuse=reuse, momentum=self._bn_momentum, epsilon=self._bn_epsilon, fused=self._use_fused_bn)
            with tf.variable_scope("bn", reuse=True):
                tf.add_to_collection('need_restored', tf.get_variable("beta"))
                tf.add_to_collection('need_restored', tf.get_variable("gamma"))
                tf.add_to_collection('need_restored', tf.get_variable("moving_mean"))
                tf.add_to_collection('need_restored', tf.get_variable("moving_variance"))

            return tf.nn.relu(inputs)
    def conv_bn(self, inputs, filters, kernel_size, strides, scope, training, padding='same', reuse=None):
        with tf.variable_scope(scope, 'conv_bn', [inputs]) as sc:
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=False, padding=padding,
                                data_format=self._data_format, activation=None,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=None, reuse=reuse)
            inputs = tf.layers.batch_normalization(inputs, axis=self._bn_axis, training=(training and self._bn_trainable), trainable=self._bn_trainable, name='bn',
                                reuse=reuse, momentum=self._bn_momentum, epsilon=self._bn_epsilon, fused=self._use_fused_bn)
            with tf.variable_scope("bn", reuse=True):
                tf.add_to_collection('need_restored', tf.get_variable("beta"))
                tf.add_to_collection('need_restored', tf.get_variable("gamma"))
                tf.add_to_collection('need_restored', tf.get_variable("moving_mean"))
                tf.add_to_collection('need_restored', tf.get_variable("moving_variance"))

            return inputs

    def conv_relu(self, inputs, filters, kernel_size, strides, scope, padding='same', dilate_rate=1, reuse=None):
        with tf.variable_scope(scope, 'conv_relu', [inputs]):
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=True, padding=padding,
                                dilation_rate=(dilate_rate, dilate_rate),
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            return inputs

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

            return tf.layers.conv2d(feature_hyper_column, top_channels, (1, 1), use_bias=True,
                                name='residual_conv', strides=(1, 1),
                                padding='same', data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer()) + features

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
                feat = tf.layers.conv2d(feat, top_channels, (3, 3), use_bias=True,
                                name='predict_stage1_conv{}'.format(ind), strides=(1, 1),
                                padding='same', data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=tf.AUTO_REUSE)
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
                feat = tf.layers.conv2d(feat, top_channels, (3, 3), use_bias=True,
                                name='predict_stage2_conv{}'.format(ind), strides=(1, 1),
                                padding='same', data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=tf.AUTO_REUSE)
                feature_maps.append(feat)

            return feature_maps
