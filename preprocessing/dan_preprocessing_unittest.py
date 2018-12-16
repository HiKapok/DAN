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

import tensorflow as tf
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np
import sys; sys.path.insert(0, ".")
from utility import draw_toolbox
import dan_preprocessing

slim = tf.contrib.slim

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image).astype(np.uint8)

    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2, y_first=True)
    imsave(os.path.join('./debug/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter

def slim_get_split(file_pattern='{}_????'):
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
                            num_samples=0,
                            items_to_descriptions=None,
                            num_classes=2,
                            labels_to_names={0: 'background', 1: 'face'})

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                                                            dataset,
                                                            num_readers=1,
                                                            common_queue_capacity=32,
                                                            common_queue_min=8,
                                                            shuffle=False,
                                                            num_epochs=None)

    [org_image, filename, shape, \
    g_bboxes, g_blur, g_expression, \
    g_illumination, g_invalid, g_occlusion, g_pose] = provider.get(['image', 'filename', 'shape',
                                                                     'object/bbox', 'object/blur',
                                                                     'object/expression', 'object/illumination',
                                                                     'object/invalid', 'object/occlusion', 'object/pose'])

    #return tf.reduce_sum(tf.to_float(tf.equal(g_invalid, tf.zeros_like(g_invalid))))#filename, shape, g_bboxes, g_blur, g_expression, g_illumination, g_invalid, g_occlusion, g_pose
    with tf.device('/cpu:0'):
        image, gbboxes = dan_preprocessing.preprocess_image(org_image, g_bboxes, [640, 640], [16, 32, 64, 128, 256, 512], is_training=True, data_format='channels_first', output_rgb=True)

    image = tf.transpose(image, perm=(1, 2, 0))
    save_image_op = tf.py_func(save_image_with_bbox,
                            [dan_preprocessing.unwhiten_image(image),
                            tf.ones_like(tf.reduce_sum(gbboxes, axis=-1), dtype=tf.int32),
                            tf.ones_like(tf.reduce_sum(gbboxes, axis=-1), dtype=tf.int32),
                            gbboxes],
                            tf.int64, stateful=True)
    return save_image_op

if __name__ == '__main__':
    save_image_op = slim_get_split('/data1/home/changanwang/widerface/tfrecords/*')
    # Create the graph, etc.
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            # if sess.run(save_image_op) < 1:
            #     print('fff')
            #     break
            print(sess.run(save_image_op))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
