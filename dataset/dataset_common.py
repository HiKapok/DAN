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

#import sys; sys.path.insert(0, ".")
from preprocessing import dan_preprocessing

import tensorflow as tf

slim = tf.contrib.slim

# use dataset_inspect.py to get these summary
data_splits_num = {
    'train': 12880,
    'valid': 3226,
    '?????': 12880 + 3226,
}

def slim_get_batch(num_classes, batch_size, split_name, file_pattern, num_readers, num_preprocessing_threads, image_preprocessing_fn, anchor_encoder, num_epochs=None, is_training=True):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      num_classes: total class numbers in dataset.
      batch_size: the size of each batch.
      split_name: 'train', 'valid' or '?????'.
      file_pattern: The file pattern to use when matching the dataset sources (full path).
      num_readers: the max number of reader used for reading tfrecords.
      num_preprocessing_threads: the max number of threads used to run preprocessing function.
      image_preprocessing_fn: the function used to dataset augumentation.
      anchor_encoder: anchor encoder function.
      num_epochs: total epoches for iterate this dataset.
      is_training: whether we are in traing phase.

    Returns:
      A batch of [image, shape, loc_targets, cls_targets, match_scores].
    """
    if split_name not in data_splits_num:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern = file_pattern.format(split_name)
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/blur': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/expression': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/illumination': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/invalid': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/occlusion': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/pose': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/blur': slim.tfexample_decoder.Tensor('image/object/bbox/blur'),
        'object/expression': slim.tfexample_decoder.Tensor('image/object/bbox/expression'),
        'object/illumination': slim.tfexample_decoder.Tensor('image/object/bbox/illumination'),
        'object/invalid': slim.tfexample_decoder.Tensor('image/object/bbox/invalid'),
        'object/occlusion': slim.tfexample_decoder.Tensor('image/object/bbox/occlusion'),
        'object/pose': slim.tfexample_decoder.Tensor('image/object/bbox/pose'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=data_splits_num[split_name],
                items_to_descriptions=None,
                num_classes=num_classes,
                labels_to_names={0: 'bg', 1: 'face'})

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=32 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=is_training,
            num_epochs=num_epochs)

    [org_image, filename, shape, \
    g_bboxes, g_blur, g_expression, \
    g_illumination, g_invalid, g_occlusion, g_pose] = provider.get(['image', 'filename', 'shape',
                                                                     'object/bbox', 'object/blur',
                                                                     'object/expression', 'object/illumination',
                                                                     'object/invalid', 'object/occlusion', 'object/pose'])

    if is_training:
        # if all is invalid, then keep the first one
        # isinvalid_mask =tf.cond(tf.count_nonzero(g_invalid, dtype=tf.int32) < tf.shape(g_invalid)[0],
        #                         lambda : g_invalid < tf.ones_like(g_invalid),
        #                         lambda : tf.one_hot(0, tf.shape(g_invalid)[0], on_value=True, off_value=False, dtype=tf.bool))

        #isinvalid_mask = g_invalid < 1
        isinvalid_mask = tf.ones_like(g_invalid < 1)
        g_bboxes = tf.boolean_mask(g_bboxes, isinvalid_mask)
        g_blur = tf.boolean_mask(g_blur, isinvalid_mask)
        g_expression = tf.boolean_mask(g_expression, isinvalid_mask)
        g_illumination = tf.boolean_mask(g_illumination, isinvalid_mask)
        g_invalid = tf.boolean_mask(g_invalid, isinvalid_mask)
        g_occlusion = tf.boolean_mask(g_occlusion, isinvalid_mask)
        g_pose = tf.boolean_mask(g_pose, isinvalid_mask)

    # Pre-processing image and bboxes.
    if is_training:
        image, gbboxes = image_preprocessing_fn(org_image, g_bboxes)
        # zero_mask = tf.Print(zero_mask, [tf.squeeze(tf.where(zero_mask), axis=-1)])
        # small_mask =tf.cond(tf.count_nonzero(small_mask, dtype=tf.int32) > 0,
        #                         lambda : small_mask,
        #                         lambda : tf.one_hot(tf.squeeze(tf.where(zero_mask), axis=-1)[0],
        #                                             tf.shape(zero_mask)[0], on_value=True,
        #                                             off_value=False, dtype=tf.bool))

        #gbboxes = tf.boolean_mask(gbboxes, small_mask)
        # g_blur = tf.boolean_mask(g_blur, small_mask)
        # g_expression = tf.boolean_mask(g_expression, small_mask)
        # g_illumination = tf.boolean_mask(g_illumination, small_mask)
        # g_invalid = tf.boolean_mask(g_invalid, small_mask)
        # g_occlusion = tf.boolean_mask(g_occlusion, small_mask)
        # g_pose = tf.boolean_mask(g_pose, small_mask)
        #is_padding_bbox = tf.ones_like(tf.reduce_mean(gbboxes), dtype=tf.int64)
        gt_targets, gt_labels, gt_scores, gt_bboxes = anchor_encoder(gbboxes)
        # ver2
        batch_list = [image, filename, shape]
        if isinstance(gt_targets, list) or isinstance(gt_targets, tuple):
            batch_list = batch_list + list(gt_targets)
        else:
            batch_list = batch_list + [gt_targets]
        if isinstance(gt_labels, list) or isinstance(gt_labels, tuple):
            batch_list = batch_list + list(gt_labels)
        else:
            batch_list = batch_list + [gt_labels]
        if isinstance(gt_scores, list) or isinstance(gt_scores, tuple):
            batch_list = batch_list + list(gt_scores)
        else:
            batch_list = batch_list + [gt_scores]
        if isinstance(gt_bboxes, list) or isinstance(gt_bboxes, tuple):
            batch_list = batch_list + list(gt_bboxes)
        else:
            batch_list = batch_list + [gt_bboxes]
        # ver1
        #batch_list = [image, filename, shape, gt_targets, gt_labels, gt_scores]
    else:
        image, output_shape = image_preprocessing_fn(org_image, g_bboxes)
        gbboxes = g_bboxes
        #is_padding_bbox = tf.ones_like(tf.reduce_mean(gbboxes), dtype=tf.int64)
        batch_list = [image, filename, shape, output_shape, gbboxes]

    if is_training:
        return tf.train.maybe_shuffle_batch(batch_list,
                        batch_size = batch_size,
                        capacity = 64 * batch_size,
                        min_after_dequeue = 8 * batch_size,
                        keep_input = (tf.shape(gbboxes)[0] > 0),
                        num_threads = num_preprocessing_threads,
                        enqueue_many = False,
                        shapes = None,
                        allow_smaller_final_batch = (not is_training))
    else:
        return tf.train.maybe_batch(batch_list,
                        (tf.shape(gbboxes)[0] > 0), batch_size,
                        num_threads = num_preprocessing_threads,
                        capacity = 64 * batch_size, enqueue_many = False,
                        shapes = None, dynamic_pad = (not is_training),
                        allow_smaller_final_batch = (not is_training))

