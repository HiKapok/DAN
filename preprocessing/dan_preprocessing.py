# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
# ==============================================================================
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
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _ImageDimensions(image, rank = 3):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def pyramid_box_random_sample_patch(image, labels, bboxes, ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.], name=None):
  '''pyramid_box_random_sample_patch.
  select one min_iou
  sample _width and _height from [0-width] and [0-height]
  check if the aspect ratio between 0.5-2.
  select left_top point from (width - _width, height - _height)
  check if this bbox has a min_iou with all ground_truth bboxes
  keep ground_truth those center is in this sampled patch, if none then try again
  '''
  def sample_width_height(width, height):
    with tf.name_scope('sample_width_height'):
      index = 0
      max_attempt = 10
      sampled_width, sampled_height = width, height

      def condition(index, sampled_width, sampled_height, width, height):
        return tf.logical_or(tf.logical_and(tf.logical_or(tf.greater(sampled_width, sampled_height * 2),
                                                        tf.greater(sampled_height, sampled_width * 2)),
                                            tf.less(index, max_attempt)),
                            tf.less(index, 1))

      def body(index, sampled_width, sampled_height, width, height):
        sampled_width = tf.random_uniform([1], minval=0.3, maxval=0.999, dtype=tf.float32)[0] * width
        sampled_height = tf.random_uniform([1], minval=0.3, maxval=0.999, dtype=tf.float32)[0] * height

        return index+1, sampled_width, sampled_height, width, height

      [index, sampled_width, sampled_height, _, _] = tf.while_loop(condition, body,
                                         [index, sampled_width, sampled_height, width, height], parallel_iterations=4, back_prop=False, swap_memory=True)

      return tf.cast(sampled_width, tf.int32), tf.cast(sampled_height, tf.int32)

  def jaccard_with_anchors(roi, bboxes):
    with tf.name_scope('jaccard_with_anchors'):
      int_ymin = tf.maximum(roi[0], bboxes[:, 0])
      int_xmin = tf.maximum(roi[1], bboxes[:, 1])
      int_ymax = tf.minimum(roi[2], bboxes[:, 2])
      int_xmax = tf.minimum(roi[3], bboxes[:, 3])
      h = tf.maximum(int_ymax - int_ymin + 1., 0.)
      w = tf.maximum(int_xmax - int_xmin + 1., 0.)
      inter_vol = h * w
      union_vol = (roi[3] - roi[1] + 1.) * (roi[2] - roi[0] + 1.) + ((bboxes[:, 2] - bboxes[:, 0] + 1.) * (bboxes[:, 3] - bboxes[:, 1] + 1.) - inter_vol)
      jaccard = tf.div(inter_vol, union_vol)
      return jaccard

  def areas(bboxes):
    with tf.name_scope('bboxes_areas'):
      vol = (bboxes[:, 3] - bboxes[:, 1] + 1.) * (bboxes[:, 2] - bboxes[:, 0] + 1.)
      return vol

  def check_roi_center(width, height, labels, bboxes):
    with tf.name_scope('check_roi_center'):
      index = 0
      max_attempt = 20
      float_width = tf.to_float(width)
      float_height = tf.to_float(height)
      roi = [0., 0., float_height - 1., float_width - 1.]

      mask = tf.cast(tf.zeros_like(labels, dtype=tf.uint8), tf.bool)
      center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2

      def condition(index, roi, mask):
        return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.to_int32(mask)) < 1,
                                          tf.less(index, max_attempt)),
                            tf.less(index, 1))

      def body(index, roi, mask):
        sampled_width, sampled_height = sample_width_height(float_width, float_height)

        x = tf.random_uniform([], minval=0, maxval=width - sampled_width, dtype=tf.int32)
        y = tf.random_uniform([], minval=0, maxval=height - sampled_height, dtype=tf.int32)

        roi = [tf.to_float(y), tf.to_float(x), tf.to_float(y + sampled_height), tf.to_float(x + sampled_width)]

        mask_min = tf.logical_and(tf.greater(center_y, roi[0]), tf.greater(center_x, roi[1]))
        mask_max = tf.logical_and(tf.less(center_y, roi[2]), tf.less(center_x, roi[3]))
        mask = tf.logical_and(mask_min, mask_max)

        return index + 1, roi, mask

      [index, roi, mask] = tf.while_loop(condition, body, [index, roi, mask], parallel_iterations=10, back_prop=False, swap_memory=True)

      mask_labels = tf.boolean_mask(labels, mask)
      mask_bboxes = tf.boolean_mask(bboxes, mask)

      return roi, mask_labels, mask_bboxes
  def check_roi_overlap(width, height, labels, bboxes, min_iou):
    with tf.name_scope('check_roi_overlap'):
      index = 0
      max_attempt = 50
      float_width = tf.to_float(width)
      float_height = tf.to_float(height)
      roi = [0., 0., float_height - 1., float_width - 1.]

      mask_labels = labels
      mask_bboxes = bboxes

      def condition(index, roi, mask_labels, mask_bboxes):
        return tf.logical_or(tf.logical_or(tf.logical_and(tf.reduce_sum(tf.to_int32(jaccard_with_anchors(roi, mask_bboxes) < min_iou)) > 0,
                                                        tf.less(index, max_attempt)),
                                          tf.less(index, 1)),
                            tf.less(tf.shape(mask_labels)[0], 1))

      def body(index, roi, mask_labels, mask_bboxes):
        roi, mask_labels, mask_bboxes = check_roi_center(width, height, labels, bboxes)
        return index+1, roi, mask_labels, mask_bboxes

      [index, roi, mask_labels, mask_bboxes] = tf.while_loop(condition, body, [index, roi, mask_labels, mask_bboxes], parallel_iterations=16, back_prop=False, swap_memory=True)

      return tf.cond(tf.greater(tf.shape(mask_labels)[0], 0),
                  lambda : (tf.to_int32([roi[0], roi[1], roi[2] - roi[0] + 1., roi[3] - roi[1] + 1.]), mask_labels, mask_bboxes),
                  lambda : (tf.to_int32([0., 0., float_height, float_width]), labels, bboxes))


  def sample_patch(image, labels, bboxes, min_iou):
    with tf.name_scope('sample_patch'):
      height, width, depth = _ImageDimensions(image, rank=3)

      roi_slice_range, mask_labels, mask_bboxes = check_roi_overlap(width, height, labels, bboxes, min_iou)

      # Add offset.
      offset = tf.cast(tf.stack([roi_slice_range[0], roi_slice_range[1], roi_slice_range[0], roi_slice_range[1]]), mask_bboxes.dtype)
      mask_bboxes = mask_bboxes - offset

      cliped_ymin = tf.maximum(0., mask_bboxes[:, 0])
      cliped_xmin = tf.maximum(0., mask_bboxes[:, 1])
      cliped_ymax = tf.minimum(tf.to_float(roi_slice_range[2]) - 1., mask_bboxes[:, 2])
      cliped_xmax = tf.minimum(tf.to_float(roi_slice_range[3]) - 1., mask_bboxes[:, 3])

      mask_bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)

      return tf.cond(tf.logical_or(tf.less(roi_slice_range[2], 1), tf.less(roi_slice_range[3], 1)),
                  lambda: (image, labels, bboxes),
                  lambda: (tf.slice(image, [roi_slice_range[0], roi_slice_range[1], 0], [roi_slice_range[2], roi_slice_range[3], -1]),
                                  mask_labels, mask_bboxes))

  with tf.name_scope('pyramid_box_random_sample_patch'):
    image = tf.convert_to_tensor(image, name='image')

    min_iou_list = tf.convert_to_tensor(ratio_list)
    samples_min_iou = tf.multinomial(tf.log([[1. / len(ratio_list)] * len(ratio_list)]), 1)

    sampled_min_iou = min_iou_list[tf.cast(samples_min_iou[0][0], tf.int32)]

    return tf.cond(tf.less(sampled_min_iou, 1.), lambda: sample_patch(image, labels, bboxes, sampled_min_iou), lambda: (image, labels, bboxes))

def pyramid_box_random_expand(image, bboxes, ratio=2., name=None):
  with tf.name_scope('pyramid_box_random_expand'):
    image = tf.convert_to_tensor(image, name='image')
    if image.get_shape().ndims != 3:
      raise ValueError('\'image\' must have 3 dimensions.')

    height, width, depth = _ImageDimensions(image, rank=3)

    float_height, float_width = tf.to_float(height), tf.to_float(width)

    canvas_width, canvas_height = tf.to_int32(float_width * ratio), tf.to_int32(float_height * ratio)

    mean_color_of_image = [_R_MEAN/255., _G_MEAN/255., _B_MEAN/255.]#tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

    x = tf.random_uniform([], minval=0, maxval=canvas_width - width, dtype=tf.int32)
    y = tf.random_uniform([], minval=0, maxval=canvas_height - height, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[y, canvas_height - height - y], [x, canvas_width - width - x]])

    big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values = mean_color_of_image[0]),
                          tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values = mean_color_of_image[1]),
                          tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values = mean_color_of_image[2])], axis=-1)

    return big_canvas, bboxes + tf.cast(tf.stack([y, x, y, x]), bboxes.dtype)
# # ssd augumentation
# def pyramid_box_random_sample_patch_wrapper(image, labels, bboxes):
#   with tf.name_scope('pyramid_box_random_sample_patch_wrapper'):
#     orgi_image, orgi_labels, orgi_bboxes = image, labels, bboxes
#     def check_bboxes(image, bboxes):
#       image_shape = tf.shape(image)
#       areas = (bboxes[:, 3] - bboxes[:, 1] + 1.) * (bboxes[:, 2] - bboxes[:, 0] + 1.)
#       return tf.logical_and(tf.logical_and(areas > 64., areas < tf.to_float(image_shape[0] * image_shape[1]) * 0.96),
#                         tf.logical_and((bboxes[:, 3] - bboxes[:, 1] + 1.) > 5., (bboxes[:, 2] - bboxes[:, 0] + 1.) > 5.))

#     index = 0
#     max_attempt = 3
#     def condition(index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes):
#       return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(check_bboxes(image, bboxes), tf.int64)) < 1, tf.less(index, max_attempt)), tf.less(index, 1))

#     def body(index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes):
#       image, bboxes = tf.cond(tf.random_uniform([], minval=0., maxval=1., dtype=tf.float32) < 0.5,
#                       lambda: (orgi_image, orgi_bboxes),
#                       lambda: pyramid_box_random_expand(orgi_image, orgi_bboxes, tf.random_uniform([1], minval=1.1, maxval=4., dtype=tf.float32)[0]))
#       # Distort image and bounding boxes.
#       random_sample_image, labels, bboxes = pyramid_box_random_sample_patch(image, orgi_labels, bboxes, ratio_list=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.])
#       random_sample_image.set_shape([None, None, 3])
#       return index+1, random_sample_image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes

#     [index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes] = tf.while_loop(condition, body, [index,  image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes], parallel_iterations=4, back_prop=False, swap_memory=True)

#     valid_mask = check_bboxes(image, bboxes)
#     labels, bboxes = tf.boolean_mask(labels, valid_mask), tf.boolean_mask(bboxes, valid_mask)
#     return tf.cond(tf.less(index, max_attempt),
#                 lambda : (image, labels, bboxes),
#                 lambda : (orgi_image, orgi_labels, orgi_bboxes))

# def pyramid_box_random_sample_patch_wrapper(image, bboxes, select_face_ind, patch_size):
#   with tf.name_scope('pyramid_box_random_sample_patch_wrapper'):
#     height, width, _ = _ImageDimensions(image, rank=3)
#     float_height, float_width = tf.to_float(height), tf.to_float(width)

#     face_ymin = tf.maximum(tf.to_int32(tf.floor(bboxes[select_face_ind, 0])), 0)
#     face_xmin = tf.maximum(tf.to_int32(tf.floor(bboxes[select_face_ind, 1])), 0)
#     face_ymax = tf.minimum(tf.to_int32(tf.ceil(bboxes[select_face_ind, 2])), height - 1)
#     face_xmax = tf.minimum(tf.to_int32(tf.ceil(bboxes[select_face_ind, 3])), width - 1)
#     face_center_x = tf.to_int32(tf.floor(tf.to_float(face_xmin + face_xmax) / 2.))
#     face_center_y = tf.to_int32(tf.floor(tf.to_float(face_ymin + face_ymax) / 2.))
#     patch_size = tf.to_int32(patch_size)

#     mean_color_of_image = [_R_MEAN / 255., _G_MEAN / 255., _B_MEAN / 255.]#tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

#     #xmin = tf.random_uniform([], minval=tf.minimum(face_xmax - patch_size + 1, face_center_x), maxval=face_xmin + 1, dtype=tf.int32)
#     #ymin = tf.random_uniform([], minval=face_ymax - patch_size + 1, maxval=face_ymin + 1, dtype=tf.int32)
#     xmin = tf.random_uniform([], minval=face_center_x - patch_size + 2, maxval=face_center_x, dtype=tf.int32)
#     ymin = tf.random_uniform([], minval=face_center_y - patch_size + 2, maxval=face_center_y, dtype=tf.int32)
#     xmax = xmin + patch_size - 1
#     ymax = ymin + patch_size - 1

#     x_pad_left = tf.cond(xmin < 0, lambda : -xmin, lambda: 0)
#     x_pad_right = tf.cond(xmax > width - 1, lambda : xmax - width + 1, lambda: 0)
#     y_pad_top = tf.cond(ymin < 0, lambda : -ymin, lambda: 0)
#     y_pad_bottom = tf.cond(ymax > height - 1, lambda : ymax - height + 1, lambda: 0)

#     paddings = tf.convert_to_tensor([[y_pad_top, y_pad_bottom], [x_pad_left, x_pad_right]])

#     bboxes = bboxes + tf.cast(tf.stack([y_pad_top, x_pad_left, y_pad_top, x_pad_left]), bboxes.dtype)
#     image = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values=mean_color_of_image[0]),
#                       tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values=mean_color_of_image[1]),
#                       tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values=mean_color_of_image[2])], axis=-1)

#     xmin = xmin + x_pad_left
#     ymin = ymin + y_pad_top
#     xmax = xmax + x_pad_left
#     ymax = ymax + y_pad_top

#     center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2
#     mask_min = tf.logical_and(tf.greater(center_y, tf.to_float(ymin)), tf.greater(center_x, tf.to_float(xmin)))
#     mask_max = tf.logical_and(tf.less(center_y, tf.to_float(ymax)), tf.less(center_x, tf.to_float(xmax)))
#     mask = tf.logical_and(mask_min, mask_max)
#     bboxes = tf.boolean_mask(bboxes, mask)
#     # start crop
#     cliped_ymin = tf.maximum(0., bboxes[:, 0] - tf.to_float(ymin))
#     cliped_xmin = tf.maximum(0., bboxes[:, 1] - tf.to_float(xmin))
#     cliped_ymax = tf.minimum(tf.to_float(ymax), bboxes[:, 2]) - tf.to_float(ymin)
#     cliped_xmax = tf.minimum(tf.to_float(xmax), bboxes[:, 3]) - tf.to_float(xmin)

#     cliped_ymin = tf.minimum(cliped_ymin, cliped_ymax)
#     cliped_xmin = tf.minimum(cliped_xmin, cliped_xmax)

#     bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)

#     return tf.slice(image, [ymin, xmin, 0], [ymax - ymin + 1, xmax - xmin + 1, -1]), bboxes

def dan_random_sample_patch_wrapper(image, bboxes):
  with tf.name_scope('dan_random_sample_patch_wrapper'):
    height, width, _ = _ImageDimensions(image, rank=3)
    float_height, float_width = tf.to_float(height), tf.to_float(width)

    def sample_around_bbox(bboxes, sampled_shorter_side):
      '''sample around the first bbox
      '''
      with tf.name_scope('sample_around_bbox'):
        sampled_shorter_side = tf.to_float(sampled_shorter_side)
        center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2
        target_ind = tf.random_uniform([], minval=0, maxval=tf.shape(bboxes)[0], dtype=tf.int32)
        roi_center_x, roi_center_y = (bboxes[target_ind, 1] + bboxes[target_ind, 3]) / 2, (bboxes[target_ind, 0] + bboxes[target_ind, 2]) / 2
        roi = [tf.maximum(roi_center_y - sampled_shorter_side/2., 0.), tf.maximum(roi_center_x - sampled_shorter_side/2., 0.), \
                tf.minimum(roi_center_y + sampled_shorter_side/2., float_height - 1.), \
                tf.minimum(roi_center_x + sampled_shorter_side/2., float_width - 1.)]
        mask_min = tf.logical_and(tf.greater_equal(center_y, roi[0]), tf.greater_equal(center_x, roi[1]))
        mask_max = tf.logical_and(tf.less_equal(center_y, roi[2]), tf.less_equal(center_x, roi[3]))
        mask = tf.logical_and(mask_min, mask_max)

        return tf.to_int32([roi[0], roi[1], roi[2] - roi[0] + 1., roi[3] - roi[1] + 1.]), tf.boolean_mask(bboxes, mask)
    def check_roi_center(bboxes, sampled_shorter_side):
      with tf.name_scope('check_roi_center'):
        index = 0
        max_attempt = 25
        roi = [0., 0., float_height - 1., float_width - 1.]

        mask = tf.cast(tf.zeros_like(bboxes[:, 0], dtype=tf.uint8), tf.bool)
        center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2

        def condition(index, roi, mask):
          return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.to_int32(mask)) < 1,
                                            tf.less(index, max_attempt)),
                              tf.less(index, 1))

        def body(index, roi, mask):
          x = tf.random_uniform([], minval=0, maxval=width - sampled_shorter_side + 1, dtype=tf.int32)
          y = tf.random_uniform([], minval=0, maxval=height - sampled_shorter_side + 1, dtype=tf.int32)

          roi = [tf.to_float(y), tf.to_float(x), tf.to_float(y + sampled_shorter_side) - 1., tf.to_float(x + sampled_shorter_side) - 1.]

          mask_min = tf.logical_and(tf.greater(center_y, roi[0]), tf.greater(center_x, roi[1]))
          mask_max = tf.logical_and(tf.less(center_y, roi[2]), tf.less(center_x, roi[3]))
          mask = tf.logical_and(mask_min, mask_max)

          return index + 1, roi, mask

        [index, roi, mask] = tf.while_loop(condition, body, [index, roi, mask], parallel_iterations=10, back_prop=False, swap_memory=True)

        mask_bboxes = tf.boolean_mask(bboxes, mask)

        return tf.cond(tf.greater(tf.shape(mask_bboxes)[0], 0),
                    lambda : (tf.to_int32([roi[0], roi[1], roi[2] - roi[0] + 1., roi[3] - roi[1] + 1.]), mask_bboxes),
                    lambda : sample_around_bbox(bboxes, sampled_shorter_side))

    shorter_side = tf.minimum(float_height, float_width)
    longer_side = tf.maximum(float_height, float_width)

    patch_a = tf.random_uniform([], minval=0.3, maxval=1., dtype=tf.float32, name='patch_a_sample')
    patch_b = tf.random_uniform([], minval=0.3, maxval=1., dtype=tf.float32, name='patch_b_sample')
    patch_c = tf.random_uniform([], minval=0.3, maxval=1., dtype=tf.float32, name='patch_c_sample')
    patch_d = tf.random_uniform([], minval=0.3, maxval=1., dtype=tf.float32, name='patch_d_sample')

    patch_list = tf.stack([patch_a, patch_b, patch_c, patch_d, 1.])
    samples_patch_ind = tf.multinomial(tf.log([[1./5., 1./5., 1./5., 1./5., 1./5.]]), 1)
    sampled_shorter_side = tf.to_int32(patch_list[tf.to_int32(samples_patch_ind[0][0])] * shorter_side)

    roi_slice_range, bboxes = check_roi_center(bboxes, sampled_shorter_side)

    # Add offset.
    offset = tf.cast(tf.stack([roi_slice_range[0], roi_slice_range[1], roi_slice_range[0], roi_slice_range[1]]), bboxes.dtype)
    bboxes = bboxes - offset

    cliped_ymin = tf.maximum(0., bboxes[:, 0])
    cliped_xmin = tf.maximum(0., bboxes[:, 1])
    cliped_ymax = tf.minimum(tf.to_float(roi_slice_range[2]) - 1., bboxes[:, 2])
    cliped_xmax = tf.minimum(tf.to_float(roi_slice_range[3]) - 1., bboxes[:, 3])

    cliped_ymin = tf.minimum(cliped_ymin, cliped_ymax)
    cliped_xmin = tf.minimum(cliped_xmin, cliped_xmax)

    bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)

    return tf.slice(image, [roi_slice_range[0], roi_slice_range[1], 0], [roi_slice_range[2], roi_slice_range[3], -1]), bboxes

def pyramid_box_random_sample_patch_wrapper(image, bboxes, select_face_ind, patch_size):
  with tf.name_scope('pyramid_box_random_sample_patch_wrapper'):
    height, width, _ = _ImageDimensions(image, rank=3)
    float_height, float_width = tf.to_float(height), tf.to_float(width)

    face_ymin = tf.maximum(tf.to_int32(tf.floor(bboxes[select_face_ind, 0])), 0)
    face_xmin = tf.maximum(tf.to_int32(tf.floor(bboxes[select_face_ind, 1])), 0)
    face_ymax = tf.minimum(tf.to_int32(tf.ceil(bboxes[select_face_ind, 2])), height - 1)
    face_xmax = tf.minimum(tf.to_int32(tf.ceil(bboxes[select_face_ind, 3])), width - 1)
    face_center_x = tf.to_int32(tf.floor(tf.to_float(face_xmin + face_xmax) / 2.))
    face_center_y = tf.to_int32(tf.floor(tf.to_float(face_ymin + face_ymax) / 2.))
    face_height = tf.to_int32(face_ymax - face_ymin)
    face_width = tf.to_int32(face_xmax - face_xmin)
    patch_size = tf.to_int32(patch_size)

    mean_color_of_image = [_R_MEAN / 255., _G_MEAN / 255., _B_MEAN / 255.]#tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

    x_lower_sample = tf.minimum(tf.minimum(face_xmax - patch_size + 1, face_center_x), face_xmin)
    x_higher_sample = tf.maximum(tf.minimum(face_xmax - patch_size + 1, face_center_x), face_xmin)
    xmin = tf.random_uniform([], minval=x_lower_sample, maxval=x_higher_sample + 1, dtype=tf.int32)
    y_lower_sample = tf.minimum(tf.minimum(face_ymax - patch_size + 1, face_center_y), face_ymin)
    y_higher_sample = tf.maximum(tf.minimum(face_ymax - patch_size + 1, face_center_y), face_ymin)
    ymin = tf.random_uniform([], minval=y_lower_sample, maxval=y_higher_sample + 1, dtype=tf.int32)
    # xmin = tf.random_uniform([], minval=face_center_x - patch_size + 2, maxval=face_center_x, dtype=tf.int32)
    # ymin = tf.random_uniform([], minval=face_center_y - patch_size + 2, maxval=face_center_y, dtype=tf.int32)
    xmax = xmin + patch_size - 1
    ymax = ymin + patch_size - 1

    x_pad_left = tf.cond(xmin < 0, lambda : -xmin, lambda: 0)
    x_pad_right = tf.cond(xmax > width - 1, lambda : xmax - width + 1, lambda: 0)
    y_pad_top = tf.cond(ymin < 0, lambda : -ymin, lambda: 0)
    y_pad_bottom = tf.cond(ymax > height - 1, lambda : ymax - height + 1, lambda: 0)

    paddings = tf.convert_to_tensor([[y_pad_top, y_pad_bottom], [x_pad_left, x_pad_right]])

    bboxes = bboxes + tf.cast(tf.stack([y_pad_top, x_pad_left, y_pad_top, x_pad_left]), bboxes.dtype)
    # it's same to do either slice or padding first
    slice_xmin = tf.maximum(xmin, 0)
    slice_ymin = tf.maximum(ymin, 0)
    slice_xmax = tf.minimum(xmax, width - 1)
    slice_ymax = tf.minimum(ymax, height - 1)
    image = tf.slice(image, [slice_ymin, slice_xmin, 0], [slice_ymax - slice_ymin + 1, slice_xmax - slice_xmin + 1, -1])

    image = tf.cond(tf.count_nonzero(paddings, dtype=tf.int32) < 1, lambda: image,
                  lambda: tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values=mean_color_of_image[0]),
                                    tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values=mean_color_of_image[1]),
                                    tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values=mean_color_of_image[2])], axis=-1))
    # transfrom bbox and clip to border
    xmin = xmin + x_pad_left
    ymin = ymin + y_pad_top
    xmax = xmax + x_pad_left
    ymax = ymax + y_pad_top

    center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2
    mask_min = tf.logical_and(tf.greater(center_y, tf.to_float(ymin)), tf.greater(center_x, tf.to_float(xmin)))
    mask_max = tf.logical_and(tf.less(center_y, tf.to_float(ymax)), tf.less(center_x, tf.to_float(xmax)))
    mask = tf.logical_and(mask_min, mask_max)
    bboxes = tf.boolean_mask(bboxes, mask)
    # start crop
    cliped_ymin = tf.maximum(0., bboxes[:, 0] - tf.to_float(ymin))
    cliped_xmin = tf.maximum(0., bboxes[:, 1] - tf.to_float(xmin))
    cliped_ymax = tf.minimum(tf.to_float(ymax), bboxes[:, 2]) - tf.to_float(ymin)
    cliped_xmax = tf.minimum(tf.to_float(xmax), bboxes[:, 3]) - tf.to_float(xmin)

    cliped_ymin = tf.minimum(cliped_ymin, cliped_ymax)
    cliped_xmin = tf.minimum(cliped_xmin, cliped_xmax)

    bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)

    return image, bboxes

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def unwhiten_image(image, output_rgb=True):
  means=[_R_MEAN, _G_MEAN, _B_MEAN]
  if not output_rgb:
    image_channels = tf.unstack(image, axis=-1, name='split_bgr')
    image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_rgb')
  num_channels = image.get_shape().as_list()[-1]
  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] += means[i]
  return tf.concat(axis=2, values=channels)

def random_flip_left_right(image, bboxes):
  with tf.name_scope('random_flip_left_right'):
    float_width = tf.to_float(_ImageDimensions(image, rank=3)[1])

    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    # Flip image.
    result = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(image), lambda: image)
    # Flip bboxes.
    mirror_bboxes = tf.stack([bboxes[:, 0], float_width - 1. - bboxes[:, 3],
                              bboxes[:, 2], float_width - 1. - bboxes[:, 1]], axis=-1)
    bboxes = tf.cond(mirror_cond, lambda: mirror_bboxes, lambda: bboxes)
    return result, bboxes

def dan_random_sample(image, bboxes, out_shape):
  with tf.name_scope('dan_random_sample', [image, bboxes]):
    random_sample_image, bboxes = dan_random_sample_patch_wrapper(image, bboxes)
    # Rescale to VGG input scale.
    height, width, _ = _ImageDimensions(random_sample_image, rank=3)
    float_height, float_width = tf.to_float(height), tf.to_float(width)
    ymin, xmin, ymax, xmax = tf.unstack(bboxes, 4, axis=-1)
    target_height, target_width = tf.to_float(out_shape[0]), tf.to_float(out_shape[1])
    ymin, ymax = ymin * target_height / float_height, ymax * target_height / float_height
    xmin, xmax = xmin * target_width / float_width, xmax * target_width / float_width
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    return random_sample_image, bboxes

def data_anchor_sampling(image, bboxes, anchor_scales, out_shape):
  with tf.name_scope('data_anchor_sampling', [image, bboxes]):
    # select one target scale
    face_heights = bboxes[:, 2] - bboxes[:, 0]
    face_widths = bboxes[:, 3] - bboxes[:, 1]
    face_scale = tf.maximum(face_widths, face_heights)
    # select one face randomly
    select_ind = tf.random_uniform([], minval=0, maxval=tf.shape(face_scale)[0], dtype=tf.int32)
    select_face_scale = tf.maximum(face_scale[select_ind], 16.)
    anchor_scales_tensor = tf.constant(anchor_scales, dtype=tf.float32)
    diff_anchor_scales = -tf.abs(anchor_scales_tensor - select_face_scale)
    # get the max anchor ind, some different with the paper
    anchor_ind = tf.minimum(tf.argmax(diff_anchor_scales, output_type=tf.int32) + 1, len(anchor_scales) - 1) + 1
    #anchor_ind = tf.Print(anchor_ind, [anchor_ind])
    # select target ind
    target_scale = anchor_scales_tensor[tf.random_uniform([], minval=0, maxval=anchor_ind, dtype=tf.int32)]
    # get final scale ratio
    final_scale = tf.random_uniform([], minval=target_scale / 2., \
                                  maxval=tf.minimum(2. * tf.sqrt(face_heights[select_ind] * face_widths[select_ind]), target_scale * 2.), \
                                  dtype=tf.float32) / select_face_scale

    height, width, _ = _ImageDimensions(image, rank=3)
    float_height, float_width = tf.to_float(height), tf.to_float(width)

    # patch_size = tf.Print(patch_size, [patch_size/tf.minimum(float_height, float_width)])
    assert out_shape[0] == out_shape[1], 'output patch should be square!'
    patch_size = tf.minimum(tf.maximum(out_shape[0]/final_scale, 64.), tf.minimum(float_height, float_width) * 8.)

    random_sample_image, bboxes = pyramid_box_random_sample_patch_wrapper(image, bboxes, select_ind, patch_size)

    # Rescale to VGG input scale.
    height, width, _ = _ImageDimensions(random_sample_image, rank=3)
    float_height, float_width = tf.to_float(height), tf.to_float(width)

    # target_shape = tf.to_int32(tf.to_float(tf.stack([height, width])) * final_scale)
    final_scale = tf.to_float(out_shape[0]) / float_height
    bboxes = bboxes * final_scale

    return random_sample_image, bboxes

def preprocess_for_train(image, bboxes, out_shape, anchor_scales, data_format='channels_first', scope='pyramid_box_preprocessing_train', output_rgb=True):
  """Preprocesses the given image for training.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after preprocessing.
    anchor_scales: List of all the anchor scales used for data-anchor sampling
    data_format: The data_format of the desired output image.
  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'dan_preprocessing_train', [image, bboxes]):
    if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
    # Convert to float scaled [0, 1].
    orig_dtype = image.dtype
    if orig_dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Randomly distort the colors. There are 4 ways to do it.
    distort_image = apply_with_random_selector(image,
                                          lambda x, ordering: distort_color(x, ordering, False),
                                          num_cases=4)

    random_sample_image, bboxes = tf.cond(tf.random_uniform([], 0, 1.0) < 0.5, lambda: dan_random_sample(distort_image, bboxes, out_shape), \
                                                                              lambda: data_anchor_sampling(distort_image, bboxes, anchor_scales, out_shape))

    #random_sample_image, bboxes = dan_random_sample(distort_image, bboxes, out_shape)

    random_sample_resized_image = tf.image.resize_images(random_sample_image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    random_sample_resized_image.set_shape([None, None, 3])

    # assert out_shape[0] == out_shape[1], 'output patch should be square!'
    # patch_size = out_shape[0]
    # random_sample_resized_image, bboxes = pyramid_box_random_sample_patch_wrapper(resized_image, bboxes, select_ind, patch_size)
    # Randomly flip the image horizontally.
    random_sample_flip_resized_image, bboxes = random_flip_left_right(random_sample_resized_image, bboxes)

    final_image = tf.to_float(tf.image.convert_image_dtype(random_sample_flip_resized_image, orig_dtype, saturate=True))
    final_image.set_shape(out_shape + [3])
    final_image = _mean_image_subtraction(final_image, [_R_MEAN, _G_MEAN, _B_MEAN])

    if not output_rgb:
      image_channels = tf.unstack(final_image, axis=-1, name='split_rgb')
      final_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
    if data_format == 'channels_first':
      final_image = tf.transpose(final_image, perm=(2, 0, 1))

    height_filter = (bboxes[:, 2] - bboxes[:, 0]) > 6.
    width_filter = (bboxes[:, 3] - bboxes[:, 1]) > 3.
    small_mask = tf.logical_and(height_filter, width_filter)

    height_filter = (bboxes[:, 2] - bboxes[:, 0]) > 0.
    width_filter = (bboxes[:, 3] - bboxes[:, 1]) > 0.
    zero_mask = tf.logical_and(height_filter, width_filter)
    bboxes = tf.boolean_mask(bboxes, small_mask)
    return final_image, bboxes

def preprocess_for_eval(image, out_shape, data_format='channels_first', scope='pyramid_box_preprocessing_eval', output_rgb=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    out_shape: The height and width of the image after preprocessing.
    data_format: The data_format of the desired output image.
  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'dan_preprocessing_eval', [image]):
    image = tf.to_float(image)
    if out_shape is not None:
      image = tf.image.resize_images(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      image.set_shape(out_shape + [3])

    height, width, _ = _ImageDimensions(image, rank=3)
    output_shape = tf.stack([height, width])

    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if not output_rgb:
      image_channels = tf.unstack(image, axis=-1, name='split_rgb')
      image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
    # Image data format.
    if data_format == 'channels_first':
      image = tf.transpose(image, perm=(2, 0, 1))
    return image, output_shape

def preprocess_image(image, bboxes, out_shape, anchor_scales, is_training=False, data_format='channels_first', output_rgb=True):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after preprocessing.
    anchor_scales: List of all the anchor scales used for data-anchor sampling
    is_training: Wether we are in training phase.
    data_format: The data_format of the desired output image.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, bboxes, out_shape, anchor_scales, data_format=data_format, output_rgb=output_rgb)
  else:
    return preprocess_for_eval(image, out_shape, data_format=data_format, output_rgb=output_rgb)
