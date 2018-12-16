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

from datetime import datetime
import os
import random
import sys
import threading
import xml.etree.ElementTree as xml_tree

import numpy as np
import six
import tensorflow as tf

import dataset_common

'''How to organize your dataset folder:
  WIDERFACE Dataset/
       |->WIDER_train/
       |    |->images/
       |    |->...
       |->WIDER_val/
       |    |->images/
       |    |->...
       |->WIDER_test/
       |    |->images/
       |    |->...
       |->wider_face_split/
       |    |->wider_face_train.mat
       |    |->...
'''
tf.app.flags.DEFINE_string('dataset_directory', '/data1/home/changanwang/widerface',
                           'All datas directory')
tf.app.flags.DEFINE_string('output_directory', '/data1/home/changanwang/widerface/tfrecords',
                           'Output data directory')
tf.app.flags.DEFINE_string('train_split', 'WIDER_train',
                           'Name of the training data sub-directory')
tf.app.flags.DEFINE_string('validation_split', 'WIDER_val',
                           'Name of the validation data sub-directory')
tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
RANDOM_SEED = 180530

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if isinstance(value, six.string_types):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_name, image_buffer, bboxes, blur, expression, illumination, invalid, occlusion, pose, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    bboxes: List of bounding boxes for each image.
    blur: List, clear->0, normal blur->1, heavy blur->2.
    expression: List, typical expression->0, exaggerate expression->1.
    illumination: List, normal illumination->0, extreme illumination->1.
    invalid: List, false->0(valid image), true->1(invalid image).
    occlusion: List, no occlusion->0, partial occlusion->1, heavy occlusion->2.
    pose: List, typical pose->0, atypical pose->1.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  ymin = []
  xmin = []
  ymax = []
  xmax = []
  for b in bboxes:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
    # pylint: enable=expression-not-assigned
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/channels': _int64_feature(channels),
            'image/shape': _int64_feature([height, width, channels]),
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/ymax': _float_feature(ymax),
            'image/object/bbox/blur': _int64_feature(blur),
            'image/object/bbox/expression': _int64_feature(expression),
            'image/object/bbox/illumination': _int64_feature(illumination),
            'image/object/bbox/invalid': _int64_feature(invalid),
            'image/object/bbox/occlusion': _int64_feature(occlusion),
            'image/object/bbox/pose': _int64_feature(pose),
            'image/format': _bytes_feature(image_format),
            'image/filename': _bytes_feature(image_name.encode('utf8')),
            'image/encoded': _bytes_feature(image_buffer)}))
  return example

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

# final_scaless = 0.
# count = 0.
def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  # shorter_side = min(height, width)
  # longer_side = max(height, width)

  # target_shorter_side = np.random.choice([1024., 1200.], 2, p=[0.5, 0.5])[0]
  # target_longer = target_shorter_side * longer_side / shorter_side
  # if target_longer > 1600:
  #     final_scale = 1600./ longer_side
  # else:
  #     final_scale = target_shorter_side / shorter_side

  # global final_scaless
  # global count
  # final_scaless += final_scale
  # #print(final_scale)
  # count+=1.

  return image_data, height, width

def _find_image_bounding_boxes(cur_record, all_ground_truth, height, width):
  """Find the bounding boxes for a given image file.

  Args:
    cur_record: list of strings; the first of which is the sub-directory of cur_record, the second is the image filename.
    all_ground_truth: all the annotations of the faces in this data set.
    height: the height of the current image.
    width: the width of the current image.
  Returns:
    bboxes: List of bounding boxes for each image.
    blur: List, clear->0, normal blur->1, heavy blur->2.
    expression: List, typical expression->0, exaggerate expression->1.
    illumination: List, normal illumination->0, extreme illumination->1.
    invalid: List, false->0(valid image), true->1(invalid image).
    occlusion: List, no occlusion->0, partial occlusion->1, heavy occlusion->2.
    pose: List, typical pose->0, atypical pose->1.
  """
  all_bboxes = all_ground_truth[cur_record]

  bboxes = []
  blur = []
  expression = []
  illumination = []
  invalid = []
  occlusion = []
  pose = []

  for bbox in all_bboxes:
    bbox = bbox.split()
    _x1, _y1, _w, _h, _blur, _expression, _illumination, _invalid, _occlusion, _pose = [int(_.strip()) for _ in bbox]
    # _w = max(_w, 1)
    # _h = max(_h, 1)

    # ymin = _y1 * 1.
    # xmin = _x1 * 1.
    # ymax = (_y1 + _h - 1) * 1.
    # xmax = (_x1 + _w - 1) * 1.
    _w = max(_w, 0)
    _h = max(_h, 0)

    ymin = _y1 * 1.
    xmin = _x1 * 1.
    ymax = (_y1 + _h) * 1.
    xmax = (_x1 + _w) * 1.
    bboxes.append((ymin, xmin, ymax, xmax))
    blur.append(_blur)
    expression.append(_expression)
    illumination.append(_illumination)
    invalid.append(_invalid)
    occlusion.append(_occlusion)
    pose.append(_pose)

  return bboxes, blur, expression, illumination, invalid, occlusion, pose

def _process_image_files_batch(coder, thread_index, ranges, name, directory, all_records, num_shards, all_ground_truth):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    directory: string; the path of all datas
    all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
    num_shards: integer number of shards for this data set.
    all_ground_truth: all the annotations of the faces in this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      cur_record = all_records[i]
      filename = os.path.join(directory, cur_record)

      image_buffer, height, width = _process_image(filename, coder)
      bboxes, blur, expression, illumination, invalid, occlusion, pose = _find_image_bounding_boxes(cur_record, all_ground_truth, height, width)

      example = _convert_to_example(filename, cur_record, image_buffer, bboxes, blur, expression, illumination, invalid, occlusion, pose, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

def _process_image_files(name, directory, all_records, num_shards, all_ground_truth):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    directory: string; the path of all datas
    all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
    num_shards: integer number of shards for this data set.
    all_ground_truth: all the annotations of the faces in this data set.
  """
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(all_records), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, directory, all_records, num_shards, all_ground_truth)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(all_records)))
  sys.stdout.flush()

def _process_dataset(name, directory, split_name, num_shards, all_ground_truth):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    split_name: sub-path to the data set.
    num_shards: integer number of shards for this data set.
    all_ground_truth: all the annotations of the faces in this data set.
  """
  #all_records = []
  jpeg_file_path = os.path.join(directory, split_name, 'images')
  all_records = list(all_ground_truth.keys())
  #all_records = list(zip([jpeg_file_path] * len(jpegs), jpegs))

  shuffled_index = list(range(len(all_records)))
  random.seed(RANDOM_SEED)
  random.shuffle(shuffled_index)
  all_records = [all_records[i] for i in shuffled_index]
  _process_image_files(name, jpeg_file_path, all_records, num_shards, all_ground_truth)

def get_train_or_val_gt(anna_file):
  # assume there is not empty objects in all images
  all_images = {}
  # take the first non-empty record as imagename
  sts_stamp = -1
  cur_image = None
  with open(anna_file) as file:
    for line in file:
      line = line.strip()
      if line == '':
        continue
      elif sts_stamp < 0:
        # encounter a new image
        assert (('jpg' in line) or ('--' in line)), 'mismatch records in {}'.format(anna_file)
        all_images[line] = []
        cur_image = line
        sts_stamp = 0 # set stamp to read total objects at next line
      elif sts_stamp > 0:
        all_images[cur_image].append(line)
        sts_stamp = sts_stamp - 1
        if sts_stamp == 0:
          sts_stamp = -1 # wait for next image
      else:
        sts_stamp = int(line)
  return all_images

all_images_list = {
  'train': os.path.join(FLAGS.dataset_directory, 'wider_face_split', 'wider_face_train_bbx_gt.txt'),
  'valid': os.path.join(FLAGS.dataset_directory, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
}

def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  os.makedirs(FLAGS.output_directory, exist_ok=True)
  _process_dataset('valid', FLAGS.dataset_directory, FLAGS.validation_split, FLAGS.validation_shards, get_train_or_val_gt(all_images_list['valid']))
  _process_dataset('train', FLAGS.dataset_directory, FLAGS.train_split, FLAGS.train_shards, get_train_or_val_gt(all_images_list['train']))
  #global final_scaless
  #global count
  #print(final_scaless/count)
if __name__ == '__main__':
  tf.app.run()

# python dataset/convert_tfrecords.py --dataset_directory=/data1/home/changanwang/widerface/ --output_directory=/data1/home/changanwang/widerface/tfrecords

