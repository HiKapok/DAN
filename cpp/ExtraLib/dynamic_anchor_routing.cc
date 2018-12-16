// Copyright (c) 2018 Changan Wang

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "dynamic_anchor_routing.h"
#include "work_sharder.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <random>
#include <cmath>
#include <queue>

using namespace tensorflow;

REGISTER_OP("DynamicAnchorRouting")
    .Attr("T: {float}")
    .Attr("trainging: bool")
    .Attr("thres: float")
    .Attr("ignore_thres: float")
    .Input("anchors: T")
    .Input("gt_targets: T")
    .Input("labels: T")
    .Input("mask_in: int32")
    .Input("feat_height: int32")
    .Input("feat_width: int32")
    .Input("anchor_depth: int32")
    .Input("feat_strides: int32")
    .Input("img_height: int32")
    .Input("img_width: int32")
    .Output("mask_out: int32")
    .Output("decode_out: T")
    .Doc(R"doc(
        DynamicAnchorRouting is use to get prior box by dynamic anchors which are generate by the last stgae of the detector.
        anchors: num_bbox x 4, is the decoded anchors in the first stage. (ymin, xmin, ymax, xmax)
        gt_targets: num_bbox x 4, is the gt bbox of each anchor before they are decoded. (ymin, xmin, ymax, xmax) (or in test, the pred_targets offsets of the next stage)
        labels: num_bbox, is the gt labels of each anchor before they are decoded (or in test, the pred labels of the next stage).
        mask_in: num_bbox, is the mask after filter some easy-background anchor.
        feat_height, feat_width: 1, height and width of the anchor layer.
        trainging: 1, whether we are in training stage.
        mask_out: num_bbox, the output mask of the next stage anchor.
        decode_out: num_bbox x 4, the prior box of the next stage. (ymin, xmin, ymax, xmax)
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(0));

      return Status::OK();
    });

// CPU specialization of actual computation.
//template <typename T>
// template <typename T>
// struct DynamicAnchorRoutingFunctor<CPUDevice, T> {
//   void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat anchors_in, typename TTypes<T>::ConstFlat gt_targets, typename TTypes<T>::ConstFlat labels_in, typename TTypes<int32_t>::ConstFlat mask_in, const int32_t feat_height, const int32_t feat_width, const int32_t anchor_depth, const int32_t feat_strides, typename TTypes<int32_t>::Flat matched_num, typename TTypes<float>::Flat prior_prob, typename TTypes<int32_t>::Flat mask_out, typename TTypes<T>::Flat decode_out, const bool is_trainging, const int64_t num_anchors, const float thres) {
//     matched_num = matched_num.setZero();
//     mask_out = mask_out.setZero();
//     prior_prob = prior_prob.setZero();
//     decode_out = decode_out.setZero();

//     std::random_device rd;  //Will be used to obtain a seed for the random number engine
//     std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//     std::uniform_real_distribution<> dis(0., 1.0);
//     //
//     //const int64_t num_anchors = anchors_in.dimension(0);
//     // we only implement single thread here, because map_fn may help us to run parralle among batches
//     //std::cout << feat_height << " " << feat_width << " " << num_anchors << " ";
//     if(is_trainging){
//       for(int64_t index = 0;index < num_anchors;++index){
//         if(mask_in.data()[index] < 1 && labels_in.data()[index] < 1.){
//           // easy background
//           if(mask_out.data()[index] < 1) mask_out.data()[index] = -1;
//           continue;
//         }
//         float ymin = static_cast<float>(anchors_in.data()[index * 4]);
//         float xmin = static_cast<float>(anchors_in.data()[index * 4 + 1]);
//         float ymax = static_cast<float>(anchors_in.data()[index * 4 + 2]);
//         float xmax = static_cast<float>(anchors_in.data()[index * 4 + 3]);

//         ymin = std::max(ymin, 0.f);
//         xmin = std::max(xmin, 0.f);
//         ymax = std::max(ymax, feat_height - 1.f);
//         xmax = std::max(xmax, feat_width - 1.f);

//         if(xmax - xmin < 1 || ymax - ymin < 1){
//           // invalid bbox
//           if(mask_out.data()[index] < 1) mask_out.data()[index] = -1;
//           continue;
//         }
//         int64_t int_center_x = static_cast<int64_t>(std::round((xmin + xmax) / (2. * feat_strides)));
//         int_center_x = std::min(int_center_x, static_cast<int64_t>(feat_width - 1));
//         int_center_x = std::max(int_center_x, static_cast<int64_t>(0));
//         int64_t int_center_y = static_cast<int64_t>(std::round((ymin + ymax) / (2. * feat_strides)));
//         int_center_y = std::min(int_center_y, static_cast<int64_t>(feat_height - 1));
//         int_center_y = std::max(int_center_y, static_cast<int64_t>(0));
//         int64_t cur_depth = index % anchor_depth;

//         int64_t next_stage_ind = (int_center_y * feat_width + int_center_x) * anchor_depth + cur_depth;
//         //std::cout << num_anchors << " " << next_stage_ind << " ";
//         //if(next_stage_ind > num_anchors) std::cout  << " false ";
//         if(dis(gen) <= 1. / (matched_num.data()[next_stage_ind] + 1)){
//           if(labels_in.data()[index] > 0.){// && labels_in.data()[next_stage_ind] > 0.
//             matched_num.data()[next_stage_ind] += 1;
//             mask_out.data()[next_stage_ind] = 1;

//             float prior_cy = (ymin + ymax) / 2.;
//             float prior_cx = (xmin + xmax) / 2.;
//             float prior_h = (ymax - ymin + 1.);
//             float prior_w = (xmax - xmin + 1.);

//             float gt_ymin = static_cast<float>(gt_targets.data()[index * 4]);
//             float gt_xmin = static_cast<float>(gt_targets.data()[index * 4 + 1]);
//             float gt_ymax = static_cast<float>(gt_targets.data()[index * 4 + 2]);
//             float gt_xmax = static_cast<float>(gt_targets.data()[index * 4 + 3]);

//             float gt_cy = (gt_ymin + gt_ymax) / 2.;
//             float gt_cx = (gt_xmin + gt_xmax) / 2.;
//             float gt_h = (gt_ymax - gt_ymin + 1.);
//             float gt_w = (gt_xmax - gt_xmin + 1.);

//             decode_out.data()[next_stage_ind * 4] = (gt_cy - prior_cy) / prior_h;
//             decode_out.data()[next_stage_ind * 4 + 1] = (gt_cx - prior_cx) / prior_w;
//             decode_out.data()[next_stage_ind * 4 + 2] = std::log(std::max(gt_h / prior_h, std::numeric_limits<float>::epsilon()));
//             decode_out.data()[next_stage_ind * 4 + 3] = std::log(std::max(gt_w / prior_w, std::numeric_limits<float>::epsilon()));
//           }
//         }
//       }
//     }else{
//       for(int64_t index = 0;index < num_anchors;++index){
//         if(mask_in.data()[index] < 1){
//           // easy background
//           continue;
//         }
//         float ymin = static_cast<float>(anchors_in.data()[index * 4]);
//         float xmin = static_cast<float>(anchors_in.data()[index * 4 + 1]);
//         float ymax = static_cast<float>(anchors_in.data()[index * 4 + 2]);
//         float xmax = static_cast<float>(anchors_in.data()[index * 4 + 3]);

//         ymin = std::max(ymin, 0.f);
//         xmin = std::max(xmin, 0.f);
//         ymax = std::max(ymax, feat_height - 1.f);
//         xmax = std::max(xmax, feat_width - 1.f);

//         if(xmax - xmin < 1 || ymax - ymin < 1){
//           // invalid bbox
//           continue;
//         }
//         int64_t int_center_x = static_cast<int64_t>(std::round((xmin + xmax) / (2. * feat_strides)));
//         int_center_x = std::min(int_center_x, static_cast<int64_t>(feat_width - 1));
//         int_center_x = std::max(int_center_x, static_cast<int64_t>(0));
//         int64_t int_center_y = static_cast<int64_t>(std::round((ymin + ymax) / (2. * feat_strides)));
//         int_center_y = std::min(int_center_y, static_cast<int64_t>(feat_height - 1));
//         int_center_y = std::max(int_center_y, static_cast<int64_t>(0));
//         int64_t cur_depth = index % anchor_depth;

//         int64_t next_stage_ind = (int_center_y * feat_width + int_center_x) * anchor_depth + cur_depth;
//         if(labels_in.data()[index] > prior_prob.data()[next_stage_ind]){
//           prior_prob.data()[next_stage_ind] = labels_in.data()[index];
//           mask_out.data()[next_stage_ind] = 1;

//           decode_out.data()[next_stage_ind * 4] = ymin;
//           decode_out.data()[next_stage_ind * 4 + 1] = xmin;
//           decode_out.data()[next_stage_ind * 4 + 2] = ymax;
//           decode_out.data()[next_stage_ind * 4 + 3] = xmax;

//         }
//       }
//     }
//   }
// };

template <typename T>
struct DynamicAnchorRoutingFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat anchors_in, typename TTypes<T>::ConstFlat gt_targets, typename TTypes<T>::ConstFlat labels_in, typename TTypes<int32_t>::ConstFlat mask_in, const int32_t feat_height, const int32_t feat_width, const int32_t anchor_depth, const int32_t feat_strides, const int32_t img_height, const int32_t img_width, typename TTypes<int32_t>::Flat matched_num, typename TTypes<float>::Flat prior_prob, typename TTypes<int32_t>::Flat mask_out, typename TTypes<T>::Flat decode_out, const bool is_trainging, const int64_t num_anchors, const float thres, const float ignore_thres) {
    matched_num = matched_num.setZero();
    mask_out = mask_out.setZero();
    prior_prob = prior_prob.setZero();
    decode_out = decode_out.setZero();

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0., 1.0);
    //
    //const int64_t num_anchors = anchors_in.dimension(0);
    // we only implement single thread here, because map_fn may help us to run parralle among batches
    //std::cout << feat_height << " " << feat_width << " " << num_anchors << " ";
    if(is_trainging){
      // assign gt bboxes
      for(int64_t index = 0;index < num_anchors;++index){
        if(labels_in.data()[index] > 0.){
            float gt_ymin = static_cast<float>(gt_targets.data()[index * 4]);
            float gt_xmin = static_cast<float>(gt_targets.data()[index * 4 + 1]);
            float gt_ymax = static_cast<float>(gt_targets.data()[index * 4 + 2]);
            float gt_xmax = static_cast<float>(gt_targets.data()[index * 4 + 3]);

            // float _gt_ymin = std::max(gt_ymin, 0.f);
            // float _gt_xmin = std::max(gt_xmin, 0.f);
            // float _gt_ymax = std::max(gt_ymax, img_height - 1.f);
            // float _gt_xmax = std::max(gt_xmax, img_width - 1.f);

            float _gt_ymin = gt_ymin;
            float _gt_xmin = gt_xmin;
            float _gt_ymax = gt_ymax;
            float _gt_xmax = gt_xmax;

            if(_gt_xmax - _gt_xmin < 1 || _gt_ymax - _gt_ymin < 1){
              continue;
            }
            int64_t int_center_x = static_cast<int64_t>(std::round((_gt_xmin + _gt_xmax) / (2. * feat_strides)));
            if(int_center_x < -feat_strides || int_center_x > feat_width + feat_strides - 1.) continue;
            int_center_x = std::min(int_center_x, static_cast<int64_t>(feat_width - 1));
            int_center_x = std::max(int_center_x, static_cast<int64_t>(0));
            int64_t int_center_y = static_cast<int64_t>(std::round((_gt_ymin + _gt_ymax) / (2. * feat_strides)));
            if(int_center_y < -feat_strides || int_center_y > feat_height + feat_strides - 1.) continue;
            int_center_y = std::min(int_center_y, static_cast<int64_t>(feat_height - 1));
            int_center_y = std::max(int_center_y, static_cast<int64_t>(0));
            int64_t cur_depth = index % anchor_depth;

            int64_t next_stage_ind = (int_center_y * feat_width + int_center_x) * anchor_depth + cur_depth;

            matched_num.data()[next_stage_ind] = 1;
            mask_out.data()[next_stage_ind] = 1;
        }
      }
      // assign bbox from last stage
      for(int64_t index = 0;index < num_anchors;++index){
        if(mask_in.data()[index] < 1){
          // easy background
          if(mask_out.data()[index] < 1) mask_out.data()[index] = -1;
          continue;
        }
        float ymin = static_cast<float>(anchors_in.data()[index * 4]);
        float xmin = static_cast<float>(anchors_in.data()[index * 4 + 1]);
        float ymax = static_cast<float>(anchors_in.data()[index * 4 + 2]);
        float xmax = static_cast<float>(anchors_in.data()[index * 4 + 3]);

        // ymin = std::max(ymin, 0.f);
        // xmin = std::max(xmin, 0.f);
        // ymax = std::max(ymax, img_height - 1.f);
        // xmax = std::max(xmax, img_width - 1.f);

        if(xmax - xmin < 1 || ymax - ymin < 1){
          // invalid bbox
          if(mask_out.data()[index] < 1) mask_out.data()[index] = -1;
          continue;
        }
        int64_t int_center_x = static_cast<int64_t>(std::round((xmin + xmax) / (2. * feat_strides)));
        if(xmin / feat_strides < -1 || xmax / feat_strides > feat_width + 1 - 1.) continue;

        //if(int_center_x < -feat_strides || int_center_x > feat_width + feat_strides - 1.) continue;
        int_center_x = std::min(int_center_x, static_cast<int64_t>(feat_width - 1));
        int_center_x = std::max(int_center_x, static_cast<int64_t>(0));
        int64_t int_center_y = static_cast<int64_t>(std::round((ymin + ymax) / (2. * feat_strides)));
        if(ymin / feat_strides < -1 || ymax / feat_strides > feat_height + 1 - 1.) continue;

        //if(int_center_y < -feat_strides || int_center_y > feat_height + feat_strides - 1.) continue;
        int_center_y = std::min(int_center_y, static_cast<int64_t>(feat_height - 1));
        int_center_y = std::max(int_center_y, static_cast<int64_t>(0));
        int64_t cur_depth = index % anchor_depth;

        int64_t next_stage_ind = (int_center_y * feat_width + int_center_x) * anchor_depth + cur_depth;
        //std::cout << num_anchors << " " << next_stage_ind << " ";
        //if(next_stage_ind > num_anchors) std::cout  << " false ";

        float gt_ymin = static_cast<float>(gt_targets.data()[index * 4]);
        float gt_xmin = static_cast<float>(gt_targets.data()[index * 4 + 1]);
        float gt_ymax = static_cast<float>(gt_targets.data()[index * 4 + 2]);
        float gt_xmax = static_cast<float>(gt_targets.data()[index * 4 + 3]);

        float int_ymin = std::max(ymin, gt_ymin);
        float int_xmin = std::max(xmin, gt_xmin);
        float int_ymax = std::min(ymax, gt_ymax);
        float int_xmax = std::min(xmax, gt_xmax);
        float h = std::max(int_ymax - int_ymin + 1., 0.);
        float w = std::max(int_xmax - int_xmin + 1., 0.);
        float int_area = h * w;
        float area_a = (gt_ymax - gt_ymin + 1.) * (gt_xmax - gt_xmin + 1.);
        float area_b = (ymax - ymin + 1.) * (xmax - xmin + 1.);

        float union_vol = area_a + area_b - int_area;

        if(labels_in.data()[index] > 0.){
          if(std::abs(union_vol) <= 1.) continue;
          //if(std::abs(union_vol) <= std::numeric_limits<float>::epsilon()) return 0.;
          else if(int_area/union_vol <= ignore_thres) continue;
          //thres
          if(int_area/union_vol < thres){
            if(mask_out.data()[index] < 1) mask_out.data()[index] = -1;
          }
          if(dis(gen) <= 1. / (matched_num.data()[next_stage_ind] + 1)){
          // && labels_in.data()[next_stage_ind] > 0.
            matched_num.data()[next_stage_ind] += 1;
            mask_out.data()[next_stage_ind] = 1;

            float prior_cy = (ymin + ymax) / 2.;
            float prior_cx = (xmin + xmax) / 2.;
            float prior_h = (ymax - ymin + 1.);
            float prior_w = (xmax - xmin + 1.);

            float gt_cy = (gt_ymin + gt_ymax) / 2.;
            float gt_cx = (gt_xmin + gt_xmax) / 2.;
            float gt_h = (gt_ymax - gt_ymin + 1.);
            float gt_w = (gt_xmax - gt_xmin + 1.);

            decode_out.data()[next_stage_ind * 4] = (gt_cy - prior_cy) / prior_h;
            decode_out.data()[next_stage_ind * 4 + 1] = (gt_cx - prior_cx) / prior_w;
            decode_out.data()[next_stage_ind * 4 + 2] = std::log(std::max(gt_h / prior_h, std::numeric_limits<float>::epsilon()));
            decode_out.data()[next_stage_ind * 4 + 3] = std::log(std::max(gt_w / prior_w, std::numeric_limits<float>::epsilon()));
          }
        }
      }
    }else{
      for(int64_t index = 0;index < num_anchors;++index){
        if(mask_in.data()[index] < 1){
          // easy background
          mask_out.data()[index] = -1;
          continue;
        }
        float ymin = static_cast<float>(anchors_in.data()[index * 4]);
        float xmin = static_cast<float>(anchors_in.data()[index * 4 + 1]);
        float ymax = static_cast<float>(anchors_in.data()[index * 4 + 2]);
        float xmax = static_cast<float>(anchors_in.data()[index * 4 + 3]);

        // ymin = std::max(ymin, 0.f);
        // xmin = std::max(xmin, 0.f);
        // ymax = std::max(ymax, img_height - 1.f);
        // xmax = std::max(xmax, img_width - 1.f);

        if(xmax - xmin < 1 || ymax - ymin < 1){
          // invalid bbox
          continue;
        }
        int64_t int_center_x = static_cast<int64_t>(std::round((xmin + xmax) / (2. * feat_strides)));
        if(xmin / feat_strides < -1 || xmax / feat_strides > feat_width + 1 - 1.) continue;
        //if(int_center_x < 0 || int_center_x > feat_width - 1.) continue;


        //if(int_center_x < -feat_strides || int_center_x > feat_width + feat_strides - 1.) continue;
        int_center_x = std::min(int_center_x, static_cast<int64_t>(feat_width - 1));
        int_center_x = std::max(int_center_x, static_cast<int64_t>(0));
        int64_t int_center_y = static_cast<int64_t>(std::round((ymin + ymax) / (2. * feat_strides)));
        if(ymin / feat_strides < -1 || ymax / feat_strides > feat_height + 1 - 1.) continue;
        //if(int_center_y < 0 || int_center_y > feat_height - 1.) continue;


        //if(int_center_y < -feat_strides || int_center_y > feat_height + feat_strides - 1.) continue;
        int_center_y = std::min(int_center_y, static_cast<int64_t>(feat_height - 1));
        int_center_y = std::max(int_center_y, static_cast<int64_t>(0));
        int64_t cur_depth = index % anchor_depth;

        int64_t next_stage_ind = (int_center_y * feat_width + int_center_x) * anchor_depth + cur_depth;
        if(labels_in.data()[index] > prior_prob.data()[next_stage_ind]){
          if(mask_out.data()[next_stage_ind] < 0) continue;
          prior_prob.data()[next_stage_ind] = labels_in.data()[index];
          mask_out.data()[next_stage_ind] = 1;
          //matched_num.data()[next_stage_ind] += 1;

          decode_out.data()[next_stage_ind * 4] = ymin;
          decode_out.data()[next_stage_ind * 4 + 1] = xmin;
          decode_out.data()[next_stage_ind * 4 + 2] = ymax;
          decode_out.data()[next_stage_ind * 4 + 3] = xmax;

        }
      }

      for(int64_t index = 0;index < num_anchors;++index){
        mask_out.data()[index] = std::max(0, mask_out.data()[index]);
        float ymin = static_cast<float>(decode_out.data()[index * 4]);
        float xmin = static_cast<float>(decode_out.data()[index * 4 + 1]);
        float ymax = static_cast<float>(decode_out.data()[index * 4 + 2]);
        float xmax = static_cast<float>(decode_out.data()[index * 4 + 3]);

        float prior_cy = (ymin + ymax) / 2.;
        float prior_cx = (xmin + xmax) / 2.;
        float prior_h = (ymax - ymin + 1.);
        float prior_w = (xmax - xmin + 1.);

        float pred_cy = static_cast<float>(gt_targets.data()[index * 4]);
        float pred_cx = static_cast<float>(gt_targets.data()[index * 4 + 1]);
        float pred_h = static_cast<float>(gt_targets.data()[index * 4 + 2]);
        float pred_w = static_cast<float>(gt_targets.data()[index * 4 + 3]);

        pred_h = std::exp(pred_h) * prior_h;
        pred_w = std::exp(pred_w) * prior_w;
        pred_cy = pred_cy * prior_h + prior_cy;
        pred_cx = pred_cx * prior_w + prior_cx;

        decode_out.data()[index * 4] = pred_cy - (pred_h - 1.) / 2.;
        decode_out.data()[index * 4 + 1] = pred_cx - (pred_w - 1.) / 2.;
        decode_out.data()[index * 4 + 2] = pred_cy + (pred_h - 1.) / 2.;
        decode_out.data()[index * 4 + 3] = pred_cx + (pred_w - 1.) / 2.;
      }


      // for(int64_t index = 0;index < num_anchors;++index){
      //   float ymin = static_cast<float>(decode_out.data()[index * 4]);
      //   float xmin = static_cast<float>(decode_out.data()[index * 4 + 1]);
      //   float ymax = static_cast<float>(decode_out.data()[index * 4 + 2]);
      //   float xmax = static_cast<float>(decode_out.data()[index * 4 + 3]);

      //   float prior_cy = (ymin + ymax) / 2.;
      //   float prior_cx = (xmin + xmax) / 2.;
      //   float prior_h = (ymax - ymin + 1.);
      //   float prior_w = (xmax - xmin + 1.);

      //   float pred_cy = static_cast<float>(gt_targets.data()[index * 4]);
      //   float pred_cx = static_cast<float>(gt_targets.data()[index * 4 + 1]);
      //   float pred_h = static_cast<float>(gt_targets.data()[index * 4 + 2]);
      //   float pred_w = static_cast<float>(gt_targets.data()[index * 4 + 3]);

      //   pred_h = std::exp(pred_h) * prior_h;
      //   pred_w = std::exp(pred_w) * prior_w;
      //   pred_cy = pred_cy * prior_h + prior_cy;
      //   pred_cx = pred_cx * prior_w + prior_cx;

      //   decode_out.data()[index * 4] = pred_cy - (pred_h - 1.) / 2.;
      //   decode_out.data()[index * 4 + 1] = pred_cx - (pred_w - 1.) / 2.;
      //   decode_out.data()[index * 4 + 2] = pred_cy + (pred_h - 1.) / 2.;
      //   decode_out.data()[index * 4 + 3] = pred_cx + (pred_w - 1.) / 2.;
      // }


      // for(int64_t index = 0;index < num_anchors;++index){
      //   if(mask_in.data()[index] < 1){
      //     // easy background
      //     continue;
      //   }
      //   float ymin = static_cast<float>(anchors_in.data()[index * 4]);
      //   float xmin = static_cast<float>(anchors_in.data()[index * 4 + 1]);
      //   float ymax = static_cast<float>(anchors_in.data()[index * 4 + 2]);
      //   float xmax = static_cast<float>(anchors_in.data()[index * 4 + 3]);

      // ymin = std::max(ymin, 0.f);
      // xmin = std::max(xmin, 0.f);
      // ymax = std::max(ymax, img_height - 1.f);
      // xmax = std::max(xmax, img_width - 1.f);

      //   if(xmax - xmin < 1 || ymax - ymin < 1){
      //     // invalid bbox
      //     continue;
      //   }
      //   int64_t int_center_x = static_cast<int64_t>(std::round((xmin + xmax) / (2. * feat_strides)));
      //   if(int_center_x < -feat_strides || int_center_x > feat_width + feat_strides - 1.) continue;
      //   int_center_x = std::min(int_center_x, static_cast<int64_t>(feat_width - 1));
      //   int_center_x = std::max(int_center_x, static_cast<int64_t>(0));
      //   int64_t int_center_y = static_cast<int64_t>(std::round((ymin + ymax) / (2. * feat_strides)));
      //   if(int_center_y < -feat_strides || int_center_y > feat_height + feat_strides - 1.) continue;
      //   int_center_y = std::min(int_center_y, static_cast<int64_t>(feat_height - 1));
      //   int_center_y = std::max(int_center_y, static_cast<int64_t>(0));
      //   int64_t cur_depth = index % anchor_depth;

      //   int64_t next_stage_ind = (int_center_y * feat_width + int_center_x) * anchor_depth + cur_depth;
      //   if(labels_in.data()[index] >= thres){
      //     //prior_prob.data()[next_stage_ind] = labels_in.data()[index];
      //     mask_out.data()[next_stage_ind] = 1;
      //     matched_num.data()[next_stage_ind] += 1;

      //     decode_out.data()[next_stage_ind * 4] += ymin;
      //     decode_out.data()[next_stage_ind * 4 + 1] += xmin;
      //     decode_out.data()[next_stage_ind * 4 + 2] += ymax;
      //     decode_out.data()[next_stage_ind * 4 + 3] += xmax;

      //   }
      // }
      // for(int64_t index = 0;index < num_anchors;++index){
      //   if(matched_num.data()[index] > 0){
      //     decode_out.data()[index * 4] = decode_out.data()[index * 4] / matched_num.data()[index];
      //     decode_out.data()[index * 4 + 1] = decode_out.data()[index * 4 + 1] / matched_num.data()[index];
      //     decode_out.data()[index * 4 + 2] = decode_out.data()[index * 4 + 2] / matched_num.data()[index];
      //     decode_out.data()[index * 4 + 3] = decode_out.data()[index * 4 + 3] / matched_num.data()[index];
      //   }

      //   float ymin = static_cast<float>(decode_out.data()[index * 4]);
      //   float xmin = static_cast<float>(decode_out.data()[index * 4 + 1]);
      //   float ymax = static_cast<float>(decode_out.data()[index * 4 + 2]);
      //   float xmax = static_cast<float>(decode_out.data()[index * 4 + 3]);

      //   float prior_cy = (ymin + ymax) / 2.;
      //   float prior_cx = (xmin + xmax) / 2.;
      //   float prior_h = (ymax - ymin + 1.);
      //   float prior_w = (xmax - xmin + 1.);

      //   float pred_cy = static_cast<float>(gt_targets.data()[index * 4]);
      //   float pred_cx = static_cast<float>(gt_targets.data()[index * 4 + 1]);
      //   float pred_h = static_cast<float>(gt_targets.data()[index * 4 + 2]);
      //   float pred_w = static_cast<float>(gt_targets.data()[index * 4 + 3]);

      //   pred_h = std::exp(pred_h) * prior_h;
      //   pred_w = std::exp(pred_w) * prior_w;
      //   pred_cy = pred_cy * prior_h + prior_cy;
      //   pred_cx = pred_cx * prior_w + prior_cx;

      //   decode_out.data()[index * 4] = pred_cy - (pred_h - 1.) / 2.;
      //   decode_out.data()[index * 4 + 1] = pred_cx - (pred_w - 1.) / 2.;
      //   decode_out.data()[index * 4 + 2] = pred_cy + (pred_h - 1.) / 2.;
      //   decode_out.data()[index * 4 + 3] = pred_cx + (pred_w - 1.) / 2.;
      // }


    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class DynamicAnchorRoutingOp : public OpKernel {
 public:
  explicit DynamicAnchorRoutingOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("trainging", &is_trainging));
    OP_REQUIRES_OK(context, context->GetAttr("thres", &thres));
    OP_REQUIRES(context, (thres >= 0.) && (thres < 1.), errors::InvalidArgument("Need Attr 1 > thres >= 0., got ", thres));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_thres", &ignore_thres));
    OP_REQUIRES(context, (ignore_thres >= 0.) && (ignore_thres < 1.), errors::InvalidArgument("Need Attr 1 > ignore_thres >= 0., got ", ignore_thres));

  }

  void Compute(OpKernelContext* context) override {
    const Tensor& anchors_in = context->input(0);
    const Tensor& gt_targets = context->input(1);
    const Tensor& labels_in = context->input(2);
    const Tensor& mask_in = context->input(3);
    const Tensor& feat_height = context->input(4);
    const Tensor& feat_width = context->input(5);
    const Tensor& anchor_depth = context->input(6);
    const Tensor& feat_strides = context->input(7);
    const Tensor& img_height = context->input(8);
    const Tensor& img_width = context->input(9);

    OP_REQUIRES(context, anchors_in.shape().dims() == 2, errors::InvalidArgument("anchors must be in 'num_anchors x 4' format."));
    OP_REQUIRES(context, gt_targets.shape().dims() == 2, errors::InvalidArgument("gt_targets must be in 'num_anchors x 4' format."));
    OP_REQUIRES(context, labels_in.shape().dims() == 1, errors::InvalidArgument("labels must be in 'num_anchors' format."));
    OP_REQUIRES(context, mask_in.shape().dims() == 1, errors::InvalidArgument("mask must be in 'num_anchors' format."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(feat_height.shape()), errors::InvalidArgument("the input feat_height should be one scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(feat_width.shape()), errors::InvalidArgument("the input feat_width should be one scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(anchor_depth.shape()), errors::InvalidArgument("the input anchor_depth should be one scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(feat_strides.shape()), errors::InvalidArgument("the input feat_strides should be one scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(img_height.shape()), errors::InvalidArgument("the input img_height should be one scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(img_width.shape()), errors::InvalidArgument("the input img_width should be one scalar."));

    const int64_t num_anchors = anchors_in.dim_size(0);

    Tensor* mask_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {num_anchors}, &mask_out));
    Tensor* decode_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {num_anchors, 4}, &decode_out));
    Tensor matched_num;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {num_anchors}, &matched_num));
    Tensor prior_prob;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {num_anchors}, &prior_prob));

    DynamicAnchorRoutingFunctor<Device, T>()(context, context->eigen_device<Device>(), anchors_in.template flat<T>(),
              gt_targets.template flat<T>(), labels_in.template flat<T>(), mask_in.template flat<int32_t>(),
              *(feat_height.template flat<int32_t>().data()), *(feat_width.template flat<int32_t>().data()), *(anchor_depth.template flat<int32_t>().data()), *(feat_strides.template flat<int32_t>().data()), *(img_height.template flat<int32_t>().data()), *(img_width.template flat<int32_t>().data()),
              matched_num.template flat<int32_t>(), prior_prob.template flat<float>(), mask_out->template flat<int32_t>(), decode_out->template flat<T>(), is_trainging, num_anchors, thres, ignore_thres);
  }

private:
  bool is_trainging{false};
  float thres{0.5};
  float ignore_thres{1.};
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DynamicAnchorRouting").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DynamicAnchorRoutingOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
// #if GOOGLE_CUDA == 1
// #define REGISTER_GPU(T)                                          \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("DynamicAnchorRouting").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//       SmallMiningMatchOp<GPUDevice, T>);
// REGISTER_GPU(float);
// #endif  // GOOGLE_CUDA
