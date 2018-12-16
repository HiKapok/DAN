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

import tensorflow as tf
import numpy as np
import cv2
import scipy.io as sio
from scipy.misc import imread, imsave, imshow, imresize

from net import danet

from dataset import dataset_common
from preprocessing import dan_preprocessing
from utility import anchor_manipulator
from utility import custom_op
from utility import bbox_util
from utility import draw_toolbox

# scaffold related configuration
tf.app.flags.DEFINE_string(
    'det_dir', './dan_det/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'data_dir', '/data1/home/changanwang/widerface',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'subset', 'val', #val or test
    'The subset of the dataset to predict.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 640,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_last', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
# r-fcn subnet configuration
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.3, 'nms threshold.')
tf.app.flags.DEFINE_float(
    'memory_limit', 577.0, 'the scale ratio to control the max memory.')
tf.app.flags.DEFINE_integer(
    'max_per_image', 750, 'max objects in one image.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './dan_logs/',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'dan',
    'Model scope name used to replace the name_scope in checkpoint.')
#CUDA_VISIBLE_DEVICES
FLAGS = tf.app.flags.FLAGS

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def detect_face(net, image, shrink):
    if shrink != 1:
        image = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    bboxes, scores = net[0].run(net[2:], feed_dict = {net[1] : image})

    det_xmin = bboxes[:, 1] / shrink#image.shape[1] *
    det_ymin = bboxes[:, 0] / shrink#image.shape[0] *
    det_xmax = bboxes[:, 3] / shrink#image.shape[1] *
    det_ymax = bboxes[:, 2] / shrink#image.shape[0] *

    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, scores))

    # keep_index = np.where(det[:, 4] >= FLAGS.select_threshold)[0]
    # det = det[keep_index, :]

    # keep_index = det[:, 4].ravel().argsort()[::-1].astype(np.int64)[:FLAGS.max_per_image]
    # det = det[keep_index, :]

    top_bbox_num = min(det.shape[0] - 1, int(FLAGS.max_per_image * 1.5))
    keep_index = det[:, 4].ravel().argsort()[::-1].astype(np.int64)[:top_bbox_num]
    det = det[keep_index, :]

    return det


def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(net, image, max_shrink):
    # Use image pyramids to detect faces
    det_b = detect_face(net, image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.75, 1.25, 1.5, 1.75]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(net, image, st[i])
            # Enlarged images are only used to detect small faces.
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            # Shrinked images are only used to detect big faces.
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b

# def flip_test(net, image, shrink):
#     image_f = cv2.flip(image, 1)
#     det_f = detect_face(net, image_f, shrink)

#     det_t = np.zeros(det_f.shape)
#     det_t[:, 0] = image.shape[1] - det_f[:, 2]
#     det_t[:, 1] = det_f[:, 1]
#     det_t[:, 2] = image.shape[1] - det_f[:, 0]
#     det_t[:, 3] = det_f[:, 3]
#     det_t[:, 4] = det_f[:, 4]
#     return det_t
def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2] - 1
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0] - 1
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1].astype(np.int64)
    det = det[order, :]
    dets = np.zeros((0, 5), dtype=np.float32)
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= FLAGS.nms_threshold)[0].astype(np.int64)

        det_accu = det[merge_index, :]

        det = np.delete(det, merge_index, 0)
        #print(det.shape[0])
        if merge_index.shape[0] == 0:
            det = np.delete(det, [0], 0)
        if merge_index.shape[0] <= 1:
            # it's totally useless to keep those single detection results
            # if merge_index == 1:
            #     dets = np.row_stack((dets, det_accu))
            # if merge_index == 1:
            #     print('single')
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5), dtype=np.float32)
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        dets = np.row_stack((dets, det_accu_sum))

    dets = dets[list(range(min(FLAGS.max_per_image, dets.shape[0]))), :]
    return dets.astype(np.float32)

def write_to_txt(f, det, event, im_name):
    bbox_xmin = det[:, 0]
    bbox_ymin = det[:, 1]
    bbox_xmax = det[:, 2]
    bbox_ymax = det[:, 3]
    scores = det[:, 4]
    bbox_height = bbox_ymax - bbox_ymin + 1
    bbox_width = bbox_xmax - bbox_xmin + 1

    valid_mask = np.logical_and(np.logical_and((np.ceil(bbox_height) >= 9), (bbox_width > 1)), scores > FLAGS.select_threshold)

    f.write('{:s}\n'.format(event[0][0] + '/' + im_name + '.jpg'))
    f.write('{}\n'.format(np.count_nonzero(valid_mask)))

    for det_ind in range(valid_mask.shape[0]):
        if not valid_mask[det_ind]:
            continue
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(np.floor(bbox_xmin[det_ind]), np.floor(bbox_ymin[det_ind]), np.ceil(bbox_width[det_ind]), np.ceil(bbox_height[det_ind]), scores[det_ind]))
        #f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(bbox_xmin[det_ind], bbox_ymin[det_ind], bbox_width[det_ind], bbox_height[det_ind], scores[det_ind]))

def get_shrink(height, width):
    """
    Args:
        height (int): image height.
        width (int): image width.
    """
    # avoid out of memory
    max_shrink_v1 = (0x7fffffff / FLAGS.memory_limit / (height * width))**0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width))**0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink

def main(_):
    with tf.Graph().as_default():
        target_shape = None

        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        features, output_shape = dan_preprocessing.preprocess_for_eval(image_input, target_shape, data_format=FLAGS.data_format, output_rgb=False)
        features = tf.expand_dims(features, axis=0)
        output_shape = tf.expand_dims(output_shape, axis=0)

        all_anchor_scales = [(16.,), (32.,), (64.,), (128.,), (256.,), (512.,)]
        all_extra_scales = [(), (), (), (), (), ()]
        all_anchor_ratios = [(0.8,), (0.8,), (0.8,), (0.8,), (0.8,), (0.8,)]
        all_layer_strides = [4, 8, 16, 32, 64, 128]
        offset_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            backbone = danet.VGG16Backbone(FLAGS.data_format)
            feature_layers = backbone.get_featmaps(features, training=False)
            feature_layers = backbone.build_bi_lfpn(feature_layers, skip_last=3)

            with tf.device('/cpu:0'):
                anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold=None, ignore_threshold=None, prior_scaling=[0.1, 0.1, 0.2, 0.2])

                if FLAGS.data_format == 'channels_first':
                    all_layer_shapes = [tf.shape(feat)[2:] for feat in feature_layers]
                else:
                    all_layer_shapes = [tf.shape(feat)[1:3] for feat in feature_layers]
                total_layers = len(all_layer_shapes)
                anchors_height = list()
                anchors_width = list()
                anchors_depth = list()
                for ind in range(total_layers):
                    _anchors_height, _anchors_width, _anchor_depth = anchor_encoder_decoder.get_anchors_width_height(all_anchor_scales[ind], all_extra_scales[ind], all_anchor_ratios[ind], name='get_anchors_width_height{}'.format(ind))
                    anchors_height.append(_anchors_height)
                    anchors_width.append(_anchors_width)
                    anchors_depth.append(_anchor_depth)
                anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, _ = anchor_encoder_decoder.get_all_anchors(tf.squeeze(output_shape, axis=0),
                                                                                anchors_height, anchors_width, anchors_depth,
                                                                                offset_list, all_layer_shapes, all_layer_strides,
                                                                                [0.] * total_layers, [False] * total_layers)
                num_anchors_per_layer = list()
                for ind, layer_shape in enumerate(all_layer_shapes):
                    _, _num_anchors_per_layer = anchor_encoder_decoder.get_anchors_count(anchors_depth[ind], layer_shape, name='get_anchor_count{}'.format(ind))
                    num_anchors_per_layer.append(_num_anchors_per_layer)

            feature_layers_stage1 = backbone.get_features_stage1(feature_layers, name='prediction_modules_stage1')
            location_pred, cls_pred = backbone.get_predict_module(feature_layers_stage1, [1] * len(feature_layers_stage1),
                                        [1] + [1] * (len(feature_layers_stage1) - 1), anchors_depth, name='predict_face')
            if FLAGS.data_format == 'channels_first':
                cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
                location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

            cls_pred = [tf.reshape(pred, [-1, FLAGS.num_classes]) for pred in cls_pred]
            location_pred = [tf.reshape(pred, [-1, 4]) for pred in location_pred]

            cls_pred = tf.nn.softmax(tf.concat(cls_pred, axis=0))[:, -1]
            location_pred = tf.concat(location_pred, axis=0)


            feature_layers_stage2 = backbone.get_features_stage2(feature_layers_stage1, feature_layers, name='prediction_modules_stage2')
            final_location_pred, final_cls_pred = backbone.get_predict_module(feature_layers_stage2, [1] * len(feature_layers_stage2),
                                        [3] + [1] * (len(feature_layers_stage2) - 1), anchors_depth, name='predict_cascade')
            if FLAGS.data_format == 'channels_first':
                final_cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in final_cls_pred]
                final_location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in final_location_pred]

            final_cls_pred = [tf.reshape(pred, [-1, FLAGS.num_classes]) for pred in final_cls_pred]
            final_location_pred = [tf.reshape(pred, [-1, 4]) for pred in final_location_pred]

            final_cls_pred = tf.nn.softmax(tf.concat(final_cls_pred, axis=0))[:, -1]
            final_location_pred = tf.concat(final_location_pred, axis=0)

        bboxes_pred = anchor_encoder_decoder.decode_anchors(location_pred, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

        num_anchors_per_layer = tf.stack(num_anchors_per_layer)
        if FLAGS.data_format == 'channels_first':
            feat_height_list = [tf.shape(feat)[2] for feat in feature_layers]
            feat_width_list = [tf.shape(feat)[3] for feat in feature_layers]
        else:
            feat_height_list = [tf.shape(feat)[1] for feat in feature_layers]
            feat_width_list = [tf.shape(feat)[2] for feat in feature_layers]

        decoded_bbox_list = tf.split(bboxes_pred, num_anchors_per_layer, axis=0, name='split_decoded_bbox')
        gt_bboxes_list = tf.split(final_location_pred / tf.expand_dims(tf.constant([10., 10., 5., 5.], dtype=tf.float32) * 2., axis=0), num_anchors_per_layer, axis=0, name='split_gt_bboxes')
        gt_lables_list = tf.split(final_cls_pred, num_anchors_per_layer, axis=0, name='split_gt_lables')
        easy_mask_list = tf.split(tf.to_int32(cls_pred > 0.03), num_anchors_per_layer, axis=0, name='split_easy_mask')
        mask_out_list = []
        decode_out_list = []
        feat_strides = [4, 8, 16, 32, 64, 128]
        for ind in range(len(anchors_depth)):
            with tf.name_scope('routing_{}'.format(ind)):
                with tf.device('/cpu:0'):
                    mask_out, decode_out = custom_op.dynamic_anchor_routing(decoded_bbox_list[ind], gt_bboxes_list[ind], gt_lables_list[ind], easy_mask_list[ind], feat_height_list[ind], feat_width_list[ind], anchors_depth[ind], feat_strides[ind], output_shape[0][0], output_shape[0][1], False, 0.03, 0.0)
                mask_out_list.append(mask_out)
                decode_out_list.append(decode_out)

        mask_out = tf.stop_gradient(tf.concat(mask_out_list, axis=0))
        bboxes_pred = tf.stop_gradient(tf.concat(decode_out_list, axis=0))

        cls_pred = final_cls_pred * tf.to_float(mask_out)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, get_checkpoint())

            os.makedirs(FLAGS.det_dir, exist_ok=True)

            if FLAGS.subset is 'val':
                wider_face = sio.loadmat(os.path.join(FLAGS.data_dir, 'wider_face_split', 'wider_face_val.mat'))    # Val set
            else:
                wider_face = sio.loadmat(os.path.join(FLAGS.data_dir, 'wider_face_split', 'wider_face_test.mat'))     # Test set
            event_list = wider_face['event_list']
            file_list = wider_face['file_list']
            del wider_face

            Path = os.path.join(FLAGS.data_dir, ('WIDER_val' if FLAGS.subset is 'val' else 'WIDER_test'), 'images')
            save_path = os.path.join(FLAGS.det_dir, FLAGS.subset)
            len_event = len(event_list)
            for index, event in enumerate(event_list):
                filelist = file_list[index][0]
                len_files = len(filelist)
                if not os.path.exists(os.path.join(save_path, event[0][0])):
                    os.makedirs(os.path.join(save_path, event[0][0]))

                for num, file in enumerate(filelist):
                    im_name = file[0][0]
                    Image_Path = os.path.join(Path, event[0][0], im_name[:]+'.jpg')

                    image = imread(Image_Path)
                    #image = imread('manymany.jpg')

                    shrink, max_shrink = get_shrink(image.shape[0], image.shape[1])
                    # max_im_shrink = (0x7fffffff / FLAGS.memory_limit / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
                    # #max_im_shrink = (0x7fffffff / 80.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
                    # shrink = max_im_shrink if max_im_shrink < 1 else 1

                    det0 = detect_face([sess, image_input, bboxes_pred, cls_pred], image, shrink)  # origin test
                    det1 = flip_test([sess, image_input, bboxes_pred, cls_pred], image, shrink)    # flip test
                    [det2, det3] = multi_scale_test([sess, image_input, bboxes_pred, cls_pred], image, max_shrink)  #multi-scale test
                    # merge all test results via bounding box voting
                    det4 = multi_scale_test_pyramid([sess, image_input, bboxes_pred, cls_pred], image, max_shrink)
                    det = np.row_stack((det0, det1, det2, det3, det4))
                    dets = bbox_vote(det)

                    f = open(os.path.join(save_path, event[0][0], im_name+'.txt'), 'w')
                    write_to_txt(f, dets, event, im_name)
                    f.close()
                    if num % FLAGS.log_every_n_steps == 0:
                        img_to_draw = draw_toolbox.bboxes_draw_on_img(image, (dets[:, 4] > 0.2).astype(np.int32), dets[:, 4], dets[:, :4], thickness=2)
                        imsave(os.path.join('./dan_debug/{}.jpg').format(im_name), img_to_draw)

                    #imsave(os.path.join('./debug/{}_{}.jpg').format(index, num), draw_toolbox.absolute_bboxes_draw_on_img(image, (dets[:, 4]>0.1).astype(np.int32), dets[:, 4], dets[:, :4], thickness=2))
                    #break
                    sys.stdout.write('\r>> Predicting event:%d/%d num:%d/%d' % (index + 1, len_event, num + 1, len_files))
                    sys.stdout.flush()
                sys.stdout.write('\n')
                sys.stdout.flush()
                #break

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

