from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops


filename = osp.join(osp.dirname(__file__), 'build/libdeform.so')
_deform_conv_module = tf.load_op_library(filename)
deform_conv_op = _deform_conv_module.deform_conv_op
deform_conv_grad_op = _deform_conv_module.deform_conv_backprop_op


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

deform_psroi_pool = _deform_conv_module.deform_psroi_pool
deform_psroi_pool_grad = _deform_conv_module.deform_psroi_pool_grad


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


def deform_conv_2d(inputs, num_outputs, kernel_size=3, stride=1, dilate_rate=1, deformable_group=1, data_format='channels_first', no_bias=True, name=None):
    with tf.variable_scope(name, 'deform_conv'):
        if 'channels_last' == data_format:
            inputs = tf.transpose(inputs, [0, 3, 1, 2], name='trans')
        offset = tf.layers.conv2d(inputs, 2 * deformable_group * kernel_size**2, kernel_size, padding='SAME', dilation_rate=(dilate_rate, dilate_rate), strides=(stride, stride), data_format='channels_first')

        kernel = tf.get_variable(name='kernel', shape=(num_outputs, inputs.get_shape().as_list()[1], kernel_size, kernel_size), initializer=tf.glorot_uniform_initializer())
        if not no_bias:
            bias_var = tf.get_variable(name='bias', shape=(1, num_outputs, 1, 1), initializer=tf.zeros_initializer())
        res = deform_conv_op(inputs, filter=kernel, offset=offset, rates=[1, 1, dilate_rate, dilate_rate], padding='SAME', strides=[1, 1, stride, stride], num_groups=1, deformable_group=deformable_group)
        if 'channels_last' == data_format:
            res = tf.transpose(res, [0, 2, 3, 1], name='trans_inv')
        if not no_bias:
          res = res + bias_var
    return res


import os
import numpy as np
import tensorflow as tf

arr = np.ones((8, 6, 4, 5))
with tf.Session() as sess:
  with tf.device('/gpu:0'):
    a = tf.constant(arr, dtype=tf.float32)
    result = deform_conv_2d(a, 64, no_bias=False)
    init_op = tf.group([tf.local_variables_initializer(), tf.global_variables_initializer(), tf.tables_initializer()])

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    sm = sess.run(result)
    print(sm)


# import os
# import numpy as np
# import tensorflow as tf

# arr = np.zeros((8, 6, 4, 5))
# with tf.Session() as sess:
#   with tf.device('/gpu:0'):
#     a = tf.constant(arr, dtype=tf.float32)
#     b = tf.constant(np.ones((21,2,2,2), dtype = np.float32))
#     c = tf.constant(np.ones((8,8,2,2), dtype = np.float32))
#     result = deform_conv_op(a, b, c, strides=[1, 1, 2, 2], rates=[1,1,1,1], padding="VALID", num_groups=3, deformable_group=1)
#     sm = sess.run(result)
#     d = tf.constant(np.ones((8,21,2,2), dtype = np.float32))
#     grad = tf.gradients(result, [a, b, c])
#     res = [sess.run(g) for g in grad]

# print(res[0])
# # print(sm)

# data_arr = np.random.rand(1,25,5,5)
# # roi = np.array([[0, 0, 0, 4, 4]],dtype=np.float32)
# trans_arr = np.random.rand(1,2,2,2)


# rois = tf.convert_to_tensor([ [0, 0, 0, 4, 4]], dtype=tf.float32)
# trans = tf.convert_to_tensor(trans_arr, dtype=tf.float32)
# hh=tf.convert_to_tensor(data_arr,dtype=tf.float32)
# [y2, channels] = deform_psroi_pool(hh, rois, trans=trans, pooled_size=2, output_dim=1, group_size=1, spatial_scale=1.0, trans_std=1e-1, sample_per_part=1, part_size=2, no_trans=False)
# s = tf.gradients(y2, [hh, trans])
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # sess.run(s[0])
# # print( sess.run(trans))
# # print( sess.run(y2))
# print( sess.run(s[1]))
# # print( sess.run(s[1]))
# pdb.set_trace()


# res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu,
#                                                       num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
# res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
#                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
#                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)
# # rfcn_cls/rfcn_bbox
# rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
# rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
# # trans_cls / trans_cls
# rfcn_cls_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
# rfcn_bbox_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")

# rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
#                                                       sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
# rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
#                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)

# psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
#                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
#                                                            output_dim=num_classes, spatial_scale=0.0625, part_size=7)
# psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
#                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
#                                                            output_dim=8, spatial_scale=0.0625, part_size=7)
