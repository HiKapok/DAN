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

import os
import sys
import time
from datetime import datetime

import tensorflow as tf

from net import pb_net

from dataset import dataset_common
from preprocessing import pb_preprocessing
from utility import anchor_manipulator
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 12,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 48,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', './dataset/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './pyramid_box_logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 7200, # not used
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_steps', 10000,
    'The frequency with which the model is saved, in steps.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 640,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', 120000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 14,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.4, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.4, 'Matching threshold for the negtive examples in the loss function.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180817, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '1000, 80000, 100000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '0.1, 1, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'pyramid_box',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'pyramid_box/additional_layers, pyramid_box/l2_norm_layer_3, pyramid_box/l2_norm_layer_4, pyramid_box/l2_norm_layer_5, pyramid_box/lfpn, pyramid_box/cpm, pyramid_box/predict_face, pyramid_box/predict_head, pyramid_box/predict_body',
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu', True,
    'Whether there is GPU to use for training.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES
def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    if FLAGS.multi_gpu:
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs '
                            'were found. To use CPU, run --multi_gpu=False.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
                    'must be a multiple of the number of available GPUs. '
                    'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                    ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)
        return num_gpus
    return 0

def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
                                            FLAGS.model_scope, FLAGS.checkpoint_model_scope,
                                            FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars,
                                            name_remap={'/conv2d/kernel': '/weights', '/conv2d/bias': '/biases'})

# couldn't find better way to pass params from input_fn to model_fn
# some tensors used by model_fn must be created in input_fn to ensure they are in the same graph
# but when we put these tensors to labels's dict, the replicate_model_fn will split them into each GPU
# the problem is that they shouldn't be splited
global_anchor_info = dict()

def input_pipeline(dataset_pattern='train-*', is_training=True, batch_size=FLAGS.batch_size):
    def input_fn():
        target_shape = [FLAGS.train_image_size] * 2

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold = FLAGS.match_threshold,
                                                        ignore_threshold = FLAGS.neg_threshold,
                                                        prior_scaling=[0.1, 0.1, 0.2, 0.2])

        all_anchor_scales = [(16.,), (32.,), (64.,), (128.,), (256.,), (512.,)]
        all_extra_scales = [(), (), (), (), (), ()]
        all_anchor_ratios = [(1.,), (1.,), (1.,), (1.,), (1.,), (1.,)]
        all_layer_shapes = [(160, 160), (80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]
        all_layer_strides = [4, 8, 16, 32, 64, 128]
        offset_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        total_layers = len(all_layer_shapes)
        anchors_height = list()
        anchors_width = list()
        anchors_depth = list()
        for ind in range(total_layers):
            _anchors_height, _anchors_width, _anchor_depth = anchor_encoder_decoder.get_anchors_width_height(all_anchor_scales[ind], all_extra_scales[ind], all_anchor_ratios[ind], name='get_anchors_width_height{}'.format(ind))
            anchors_height.append(_anchors_height)
            anchors_width.append(_anchors_width)
            anchors_depth.append(_anchor_depth)
        anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask = anchor_encoder_decoder.get_all_anchors(target_shape, anchors_height, anchors_width, anchors_depth,
                                                                        offset_list, all_layer_shapes, all_layer_strides,
                                                                        [FLAGS.train_image_size * 1.] * total_layers, [False] * total_layers)

        num_anchors_per_layer = list()
        for ind, layer_shape in enumerate(all_layer_shapes):
            _, _num_anchors_per_layer = anchor_encoder_decoder.get_anchors_count(anchors_depth[ind], layer_shape, name='get_anchor_count{}'.format(ind))
            num_anchors_per_layer.append(_num_anchors_per_layer)

        image_preprocessing_fn = lambda image_, bboxes_ : pb_preprocessing.preprocess_image(image_, bboxes_, target_shape, [16, 32, 64, 128, 256, 512], is_training=is_training, data_format=FLAGS.data_format, output_rgb=False)

        def anchor_encoder_fn(gbboxes_):
            head_start_indices = num_anchors_per_layer[0]
            body_start_indices = num_anchors_per_layer[0] + num_anchors_per_layer[1]

            _face_targets, _face_labels, _face_scores, _ = anchor_encoder_decoder.encode_pa_anchors(gbboxes_, anchors_ymin[0:],
                                                                                                    anchors_xmin[0:],
                                                                                                    anchors_ymax[0:],
                                                                                                    anchors_xmax[0:],
                                                                                                    inside_mask[0:], FLAGS.neg_threshold, FLAGS.match_threshold, match_mining=True, scale=1.)
            _head_targets, _head_labels, _head_scores, _ = anchor_encoder_decoder.encode_pa_anchors(gbboxes_, anchors_ymin[head_start_indices:],
                                                                                                    anchors_xmin[head_start_indices:],
                                                                                                    anchors_ymax[head_start_indices:],
                                                                                                    anchors_xmax[head_start_indices:],
                                                                                                    inside_mask[head_start_indices:], 0.35, 0.35, match_mining=False, scale=2.)
            _body_targets, _body_labels, _body_scores, _ = anchor_encoder_decoder.encode_pa_anchors(gbboxes_, anchors_ymin[body_start_indices:],
                                                                                                    anchors_xmin[body_start_indices:],
                                                                                                    anchors_ymax[body_start_indices:],
                                                                                                    anchors_xmax[body_start_indices:],
                                                                                                    inside_mask[body_start_indices:], 0.35, 0.35, match_mining=False, scale=4.)

            return [_face_targets, _head_targets, _body_targets], [_face_labels, _head_labels, _body_labels], [_face_scores, _head_scores, _body_scores], [_, _, _]
        #encode_pa_anchors(self, bboxes, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask, match_mining=True, scale=1., debug=False):
        image, filename, shape, face_loc_targets, head_loc_targets, body_loc_targets, \
        face_cls_targets, head_cls_targets, body_cls_targets, \
        face_match_scores, head_match_scores, body_match_scores, _, _, _ = dataset_common.slim_get_batch(FLAGS.num_classes,
                                                                                batch_size,
                                                                                ('train' if is_training else 'valid'),
                                                                                os.path.join(FLAGS.data_dir, dataset_pattern),
                                                                                FLAGS.num_readers,
                                                                                FLAGS.num_preprocessing_threads,
                                                                                image_preprocessing_fn,
                                                                                anchor_encoder_fn,
                                                                                num_epochs=FLAGS.train_epochs,
                                                                                is_training=is_training)
        global global_anchor_info
        global_anchor_info = {'decode_fn': lambda pred : anchor_encoder_decoder.batch_decode_anchors(pred, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax),
                            'num_anchors_per_layer': num_anchors_per_layer,
                            'all_num_anchors_depth': anchors_depth }

        return image, {'filename': filename, 'shape': shape, \
                        'face_loc_targets': face_loc_targets, 'face_cls_targets': face_cls_targets, 'face_match_scores': face_match_scores, \
                        'head_loc_targets': head_loc_targets, 'head_cls_targets': head_cls_targets, 'head_match_scores': head_match_scores, \
                        'body_loc_targets': body_loc_targets, 'body_cls_targets': body_cls_targets, 'body_match_scores': body_match_scores}
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

def mining_hard_neg_across_batch(batch_size, cls_pred, location_pred, cls_targets, match_scores, loc_targets, name=None):
    with tf.variable_scope(name, 'hard_neg_mining'):
        ############## hard negtive mining across batch
        cls_targets = tf.reshape(cls_targets, [-1])
        match_scores = tf.reshape(match_scores, [-1])
        loc_targets = tf.reshape(loc_targets, [-1, 4])

        # each positive examples has one label
        positive_mask = cls_targets > 0
        n_positives = tf.count_nonzero(positive_mask)

        negtive_mask = tf.equal(cls_targets, 0)
        #negtive_mask = tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
        n_negtives = tf.count_nonzero(negtive_mask)

        n_neg_to_select = tf.to_int32(FLAGS.negative_ratio * tf.to_float(n_positives))
        n_neg_to_select = tf.minimum(n_neg_to_select, tf.to_int32(n_negtives))

        # hard negative mining for classification
        predictions_for_bg = tf.nn.softmax(cls_pred)[:, 0]

        prob_for_negtives = tf.where(negtive_mask,
                               0. - predictions_for_bg,
                               # ignore all the positives
                               0. - tf.ones_like(predictions_for_bg))
        topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=n_neg_to_select)
        selected_neg_mask = prob_for_negtives > topk_prob_for_bg[-1]

        # include both selected negtive and all positive examples
        final_mask = tf.stop_gradient(tf.logical_or(tf.logical_and(negtive_mask, selected_neg_mask), positive_mask))
        total_examples = tf.count_nonzero(final_mask)

        flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(cls_targets, 0, FLAGS.num_classes), final_mask)
        cls_pred = tf.boolean_mask(cls_pred, final_mask)
        location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
        flaten_loc_targets = tf.boolean_mask(loc_targets, tf.stop_gradient(positive_mask))

        return cls_pred, location_pred, flaten_cls_targets, flaten_loc_targets

def mining_hard_neg(batch_size, cls_pred, location_pred, cls_targets, match_scores, loc_targets, name=None):
    with tf.variable_scope(name, 'hard_neg_mining'):
        flaten_cls_targets = tf.reshape(cls_targets, [-1])
        flaten_match_scores = tf.reshape(match_scores, [-1])
        flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])

        # each positive examples has one label
        positive_mask = flaten_cls_targets > 0
        n_positives = tf.count_nonzero(positive_mask)

        batch_n_positives = tf.count_nonzero(cls_targets > 0, -1)

        batch_negtive_mask = tf.equal(cls_targets, 0)#tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
        batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

        batch_n_neg_select = tf.to_int32(FLAGS.negative_ratio * tf.to_float(batch_n_positives))
        batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.to_int32(batch_n_negtives))

        # hard negative mining for classification
        predictions_for_bg = tf.nn.softmax(tf.reshape(cls_pred, [batch_size, -1, FLAGS.num_classes]))[:, :, 0]
        prob_for_negtives = tf.where(batch_negtive_mask,
                               0. - predictions_for_bg,
                               # ignore all the positives
                               0. - tf.ones_like(predictions_for_bg))
        topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
        score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(batch_size), batch_n_neg_select - 1], axis=-1))

        selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

        # include both selected negtive and all positive examples
        final_mask = tf.stop_gradient(tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]), positive_mask))
        total_examples = tf.count_nonzero(final_mask)

        cls_pred = tf.boolean_mask(cls_pred, final_mask)
        location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
        flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(flaten_cls_targets, 0, FLAGS.num_classes), final_mask)
        flaten_loc_targets = tf.stop_gradient(tf.boolean_mask(flaten_loc_targets, positive_mask))

        return cls_pred, location_pred, flaten_cls_targets, flaten_loc_targets
# from scipy.misc import imread, imsave, imshow, imresize
# import numpy as np
# from utility import draw_toolbox

# def save_image_with_bbox(image, labels_, scores_, bboxes_):
#     if not hasattr(save_image_with_bbox, "counter"):
#         save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
#     save_image_with_bbox.counter += 1

#     img_to_draw = np.copy(image)

#     img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
#     imsave(os.path.join('./pb_debug/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
#     return save_image_with_bbox.counter

def sfd_model_fn(features, labels, mode, params):
    """model_fn for sfd to be used with our Estimator."""
    shape = labels['shape']

    loc_targets = labels['face_loc_targets']
    cls_targets = labels['face_cls_targets']
    match_scores = labels['face_match_scores']

    head_loc_targets = labels['head_loc_targets']
    head_cls_targets = labels['head_cls_targets']
    head_match_scores = labels['head_match_scores']

    body_loc_targets = labels['body_loc_targets']
    body_cls_targets = labels['body_cls_targets']
    body_match_scores = labels['body_match_scores']

    global global_anchor_info
    decode_fn = global_anchor_info['decode_fn']
    num_anchors_per_layer = global_anchor_info['num_anchors_per_layer']
    all_num_anchors_depth = global_anchor_info['all_num_anchors_depth']

    # bboxes_pred = decode_fn(loc_targets[0])
    # bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
    # bboxes_pred = tf.concat(bboxes_pred, axis=0)
    # save_image_op = tf.py_func(save_image_with_bbox,
    #                         [pb_preprocessing.unwhiten_image(features[0]),
    #                         tf.clip_by_value(cls_targets[0], 0, tf.int64.max),
    #                         match_scores[0],
    #                         bboxes_pred],
    #                         tf.int64, stateful=True)
    # with tf.control_dependencies([save_image_op]):

    #print(all_num_anchors_depth)
    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = pb_net.VGG16Backbone(params['data_format'])
        feature_layers = backbone.get_featmaps(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        feature_layers = backbone.build_lfpn(feature_layers, skip_last=3)
        feature_layers = backbone.context_pred_module(feature_layers)
        #print(feature_layers)
        #print(feature_layers)
        location_pred, cls_pred = backbone.get_predict_module(feature_layers, [1] + [3] * (len(feature_layers) - 1),
                                        [3] + [1] * (len(feature_layers) - 1), all_num_anchors_depth, name='predict_face')
        head_location_pred, head_cls_pred = backbone.get_predict_module(feature_layers[1:], [1] * (len(feature_layers) - 1), [1] * (len(feature_layers) - 1), all_num_anchors_depth[1:], name='predict_head')
        body_location_pred, body_cls_pred = backbone.get_predict_module(feature_layers[2:], [1] * (len(feature_layers) - 2), [1] * (len(feature_layers) - 2), all_num_anchors_depth[2:], name='predict_body')

        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

            head_cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in head_cls_pred]
            head_location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in head_location_pred]

            body_cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in body_cls_pred]
            body_location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in body_location_pred]

        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]

        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)

        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])

        # for head
        head_cls_pred =  tf.concat([tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in head_cls_pred], axis=1)
        head_location_pred = tf.concat([tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in head_location_pred], axis=1)
        head_cls_pred = tf.reshape(head_cls_pred, [-1, params['num_classes']])
        head_location_pred = tf.reshape(head_location_pred, [-1, 4])
        # for body
        body_cls_pred =  tf.concat([tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in body_cls_pred], axis=1)
        body_location_pred = tf.concat([tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in body_location_pred], axis=1)
        body_cls_pred = tf.reshape(body_cls_pred, [-1, params['num_classes']])
        body_location_pred = tf.reshape(body_location_pred, [-1, 4])

    extra_losses = []
    with tf.device('/cpu:0'):
        with tf.control_dependencies([cls_pred, location_pred]):
            with tf.name_scope('post_forward'):
                #bboxes_pred = decode_fn(location_pred)
                # bboxes_pred = decode_fn(tf.reshape(location_pred, [tf.shape(features)[0], -1, 4]))
                # #cls_targets = tf.Print(cls_targets, [tf.shape(bboxes_pred[0]),tf.shape(bboxes_pred[1]),tf.shape(bboxes_pred[2]),tf.shape(bboxes_pred[3])])
                # bboxes_pred = tf.reshape(bboxes_pred, [-1, 4])
                cls_pred, location_pred, flaten_cls_targets, flaten_loc_targets = mining_hard_neg(tf.shape(features)[0], cls_pred, location_pred, cls_targets, match_scores, loc_targets)
                with tf.name_scope('head_loss'):
                    head_cls_pred, head_location_pred, flaten_head_cls_targets, flaten_head_loc_targets = mining_hard_neg(tf.shape(features)[0], head_cls_pred, head_location_pred, head_cls_targets, head_match_scores, head_loc_targets)
                    head_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_head_cls_targets, logits=head_cls_pred) * (params['negative_ratio'] + 1.)
                    head_cross_entropy = tf.identity(head_cross_entropy, name='head_cross_entropy_loss')
                    head_loc_loss = modified_smooth_l1(head_location_pred, flaten_head_loc_targets, sigma=1.)
                    head_loc_loss = tf.reduce_mean(tf.reduce_sum(head_loc_loss, axis=-1), name='head_location_loss')
                    extra_losses.append(head_cross_entropy * 0.66)
                    extra_losses.append(head_loc_loss * 0.66)
                with tf.name_scope('body_loss'):
                    body_cls_pred, body_location_pred, flaten_body_cls_targets, flaten_body_loc_targets = mining_hard_neg(tf.shape(features)[0], body_cls_pred, body_location_pred, body_cls_targets, body_match_scores, body_loc_targets)
                    body_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_body_cls_targets, logits=body_cls_pred) * (params['negative_ratio'] + 1.)
                    body_cross_entropy = tf.identity(body_cross_entropy, name='body_cross_entropy_loss')
                    body_loc_loss = modified_smooth_l1(body_location_pred, flaten_body_loc_targets, sigma=1.)
                    body_loc_loss = tf.reduce_mean(tf.reduce_sum(body_loc_loss, axis=-1), name='body_location_loss')
                    extra_losses.append(body_cross_entropy * 0.33)
                    extra_losses.append(body_loc_loss * 0.33)

                predictions = {
                            'classes': tf.argmax(cls_pred, axis=-1),
                            'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1)}
                            #'loc_predict': bboxes_pred }

                cls_accuracy = tf.metrics.accuracy(flaten_cls_targets, predictions['classes'])
                metrics = {'cls_accuracy': cls_accuracy}

                # Create a tensor named train_accuracy for logging purposes.
                tf.identity(cls_accuracy[1], name='cls_accuracy')
                tf.summary.scalar('cls_accuracy', cls_accuracy[1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    #cross_entropy = tf.cond(n_positives > 0, lambda: tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred), lambda: 0.)# * (params['negative_ratio'] + 1.)
    #flaten_cls_targets=tf.Print(flaten_cls_targets, [flaten_loc_targets],summarize=50000)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred) * (params['negative_ratio'] + 1.)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    #loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flaten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    loc_loss = modified_smooth_l1(location_pred, flaten_loc_targets, sigma=1.)
    #loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if 'bn' not in trainable_var.name:
            if 'l2_norm_layer' not in trainable_var.name:
                if '/bias' not in trainable_var.name: l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.2)
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    total_loss = tf.add(cross_entropy + loc_loss + tf.add_n(extra_losses), tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss'), name='total_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        truncated_learning_rate = tf.maximum(learning_rate, tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype), name='learning_rate')
        # Create a tensor named learning_rate for logging purposes.
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        gradient_multipliers = {}
        for var in tf.trainable_variables():
            if '/bias' in var.name: gradient_multipliers[var] = 2.
            else: gradient_multipliers[var] = 1.

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.contrib.layers.optimize_loss(total_loss, global_step, truncated_learning_rate,
                                    optimizer=lambda lr: tf.contrib.estimator.TowerOptimizer(tf.train.MomentumOptimizer(learning_rate=lr, momentum=params['momentum'])),
                                    gradient_multipliers=gradient_multipliers,
                                    update_ops=update_ops,
                                    summaries=['loss'])
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
                              mode=mode,
                              predictions=predictions,
                              loss=total_loss,
                              train_op=train_op,
                              eval_metric_ops=metrics,
                              scaffold=tf.train.Scaffold(init_fn=get_init_fn()))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)

    num_gpus = validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        tf_random_seed=FLAGS.tf_random_seed).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    replicate_sfd_model_fn = tf.contrib.estimator.replicate_model_fn(sfd_model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    sfd_detector = tf.estimator.Estimator(
        model_fn=replicate_sfd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'num_gpus': num_gpus,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })
    tensors_to_log = {
        'lr': 'learning_rate',
        'ce': 'cross_entropy_loss',
        'lc': 'location_loss',
        'hce': 'post_forward/head_loss/head_cross_entropy_loss',
        'hlc': 'post_forward/head_loss/head_location_loss',
        'bce': 'post_forward/body_loss/body_cross_entropy_loss',
        'blc': 'post_forward/body_loss/body_location_loss',
        'ls': 'total_loss',
        'wd': 'l2_loss',
        'acc': 'post_forward/cls_accuracy',
    }
    def logging_formatter(dicts):
        output_order = ['ls', 'acc', 'ce', 'lc', 'hce', 'hlc', 'bce', 'blc', 'wd']
        return 'lr=%.5f,' % (dicts['lr'], ) + ','.join(['%s=%.2f' % (k, dicts[k]) for k in output_order])

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                            formatter=logging_formatter)

    #hook = tf.train.ProfilerHook(save_steps=50, output_dir='.', show_memory=True)
    tf.logging.info('Starting a training cycle.')
    sfd_detector.train(input_fn=input_pipeline(dataset_pattern='{}-*', is_training=True, batch_size=FLAGS.batch_size),
                    hooks=[logging_hook], max_steps=FLAGS.max_number_of_steps)
    tf.logging.info('Finished runing at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

if __name__ == '__main__':
    import logging
    tf.gfile.MakeDirs(FLAGS.model_dir)
    tf.gfile.Copy(os.path.realpath(__file__), os.path.join(FLAGS.model_dir, 'train.py'), overwrite=True)
    log = logging.getLogger('tensorflow')
    fh = logging.FileHandler(FLAGS.model_dir + 'tensorflow.log')
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting runing at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    tf.app.run()
