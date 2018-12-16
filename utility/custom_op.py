# MIT License

# Copyright (c) 2018 Changan Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import math


LIB_NAME = 'extra_lib'

def load_op_module(path_root, lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../cpp/{}/build/lib{}.so'.format(path_root, lib_name))
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the original library is updated
  lib_copy_path = './cpp/lib{0}_{1}.so'.format(str(uuid.uuid4())[:8], LIB_NAME)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib

op_module_extra = load_op_module('ExtraLib', LIB_NAME)

small_mining_match = op_module_extra.small_mining_match

ops.NotDifferentiable("SmallMiningMatch")

dynamic_anchor_routing = op_module_extra.dynamic_anchor_routing

ops.NotDifferentiable("DynamicAnchorRouting")




LIB_NAME = 'deform'
op_module = load_op_module('Deform', LIB_NAME)

deform_conv_op = op_module.deform_conv_op
deform_conv_grad_op = op_module.deform_conv_backprop_op


@ops.RegisterGradient("DeformConvOp")
def _deform_conv_grad(op, grad):
  """The gradients for `deform_conv`.
  Args:
    op: The `deform_conv` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  filters = op.inputs[1]
  offset = op.inputs[2]

  strides = op.get_attr('strides')
  rates = op.get_attr('rates')
  num_groups = op.get_attr('num_groups')
  padding = op.get_attr('padding')
  data_format = op.get_attr('data_format')
  deformable_group = op.get_attr('deformable_group')

  # compute gradient
  data_grad = deform_conv_grad_op(data, filters, offset, grad, strides, rates, num_groups, deformable_group, padding, data_format)

  return data_grad  # List of one Tensor, since we have one input

# NCHW only
deform_psroi_pool = op_module.deform_psroi_pool
deform_psroi_pool_grad = op_module.deform_psroi_pool_grad


@ops.RegisterGradient("DeformPSROIPool")
def _deform_psroi_pool_grad(op, grad, _):
  """The gradients for `Deform_PSROI_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  rois = op.inputs[1]
  trans = op.inputs[2]
  mapping_channel = op.outputs[1]
  spatial_scale = op.get_attr('spatial_scale')
  output_dim = op.get_attr('output_dim')
  group_size = op.get_attr('group_size')
  pooled_size = op.get_attr('pooled_size')
  part_size = op.get_attr('part_size')
  sample_per_part = op.get_attr('sample_per_part')
  trans_std = op.get_attr('trans_std')
  no_trans = op.get_attr('no_trans')

  # compute gradient
  #data_grad = psroi_pooling_op.psroi_pool_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)
  data_grad, trans_grad = deform_psroi_pool_grad(data, rois, trans, mapping_channel, grad, spatial_scale,
                                                            output_dim, group_size, pooled_size, part_size, sample_per_part,
                                                            trans_std, no_trans)
  # rois_grad = tf.zeros(rois.shape)
  return [data_grad, None, trans_grad]  # List of one Tensor, since we have one input

def deform_conv_2d(inputs, num_outputs, kernel_size_h=3, kernel_size_w=3, stride=1, dilate_rate=1, deformable_group=1, data_format='channels_first', no_bias=True, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    with tf.variable_scope(name, 'deform_conv'):
        if 'channels_last' == data_format:
            inputs = tf.transpose(inputs, [0, 3, 1, 2], name='trans')
        offset = tf.layers.conv2d(inputs, 2 * deformable_group * kernel_size_h * kernel_size_w, (kernel_size_h, kernel_size_w), padding='SAME', dilation_rate=(dilate_rate, dilate_rate), strides=(stride, stride), data_format='channels_first', kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())

        kernel = tf.get_variable(name='kernel', shape=(num_outputs, inputs.get_shape().as_list()[1], kernel_size_h, kernel_size_w), trainable=True, initializer=kernel_initializer())
        if not no_bias:
            bias_var = tf.get_variable(name='bias', shape=(num_outputs,), trainable=True, initializer=tf.zeros_initializer())
            if 'channels_last' == data_format:
                bias_var = tf.reshape(bias_var, [1, 1, 1, num_outputs])
            else:
                bias_var = tf.reshape(bias_var, [1, num_outputs, 1, 1])
        res = deform_conv_op(inputs, filter=kernel, offset=offset, rates=[1, 1, dilate_rate, dilate_rate], padding='SAME', strides=[1, 1, stride, stride], num_groups=1, deformable_group=deformable_group)
        if 'channels_last' == data_format:
            res = tf.transpose(res, [0, 2, 3, 1], name='trans_inv')
        if not no_bias:
          res = res + bias_var
    return res
