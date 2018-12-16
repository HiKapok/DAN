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
#include "small_mining_match.h"
#include "work_sharder.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <queue>

using namespace tensorflow;

REGISTER_OP("SmallMiningMatch")
    .Attr("T: {float}")
    .Attr("negative_low_thres: float")
    .Attr("negative_high_thres: float")
    .Attr("positive_thres: float")
    .Attr("min_match: int")
    .Attr("stop_positive_thres: float")
    .Input("overlaps: T")
    .Output("match_indices: int32")
    .Output("match_scores: T")
    .Doc(R"doc(
        SmallMiningMatch is use to match ground truth to anchors with compensation wo small scale anchors.
        The input overlaps matrix in shape [num_anchors, num_ground_truth] and each element must be in range [0, 1.].
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle inputs_shape = c->input(0);
      shape_inference::DimensionHandle num_anchors = c->Dim(inputs_shape, 0);
      shape_inference::DimensionHandle num_ground_truth = c->Dim(inputs_shape, 1);

      c->set_output(0, c->MakeShape({num_anchors}));
      c->set_output(1, c->MakeShape({num_anchors}));

      return Status::OK();
    });

struct DistancePair {
  DistancePair(int64_t i1, int64_t i2, float d) : index1(i1), index2(i2), dist(d) {}

  bool operator<(const DistancePair& b1) const { return b1.dist > dist; }

  int64_t index1, index2;
  float dist;
};

// CPU specialization of actual computation.
//template <typename T>
template <typename T>
struct SmallMiningMatchFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat overlaps, typename TTypes<int32_t>::Flat match_indices, typename TTypes<T>::Flat match_scores, typename TTypes<int32_t>::Flat gt_match_num, typename TTypes<int32_t>::Flat gt_small_topk, const int32_t num_anchors, const int32_t num_ground_truth, const float negative_low_thres, const float negative_high_thres, const float positive_thres, const float stop_positive_thres, const int32_t min_match) {
    gt_match_num = gt_match_num.setZero();

    auto anchor_match_routine = [&overlaps, &match_indices, &match_scores, &gt_match_num, num_anchors, num_ground_truth, negative_low_thres, negative_high_thres, positive_thres, min_match](int64_t start, int64_t limit){
      for (int64_t worker_index = start; worker_index < limit; ++worker_index){
        const T * anchor_overlaps = overlaps.data() + worker_index * num_ground_truth;
        int32_t max_gt = 0;
        float max_score = std::numeric_limits<float>::lowest();

        for(int32_t index = 0;index < num_ground_truth;++index){
          if(anchor_overlaps[index] > max_score){
            max_gt = index;
            max_score = anchor_overlaps[index];
          }
        }
        match_scores.data()[worker_index] = max_score;
        //std::cout <<"no" << std::endl;
        if((max_score >= negative_low_thres) && (max_score < negative_high_thres)){
          match_indices.data()[worker_index] = -1;
        }else if(max_score >= positive_thres){
          match_indices.data()[worker_index] = max_gt;
          __atomic_fetch_add(gt_match_num.data() + max_gt, 1, __ATOMIC_SEQ_CST);
          //gt_match_num.data()[max_gt] += 1;
        }else match_indices.data()[worker_index] = -2;
      }
    };

    // auto gt_match_routine = [&overlaps, &match_indices, &match_scores, &gt_match_num, num_anchors, num_ground_truth, negative_low_thres, negative_high_thres, positive_thres, min_match](int64_t start, int64_t limit){
    //   for (int64_t worker_index = start; worker_index < limit; ++worker_index){
    //     int32_t max_anchor = 0;
    //     float max_score = std::numeric_limits<float>::lowest();

    //     for(int32_t index = 0;index < num_anchors;++index){
    //       T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
    //       if(cur_score > max_score){
    //         max_anchor = index;
    //         max_score = cur_score;
    //       }
    //     }

    //     atomic_store_float(match_scores.data() + max_anchor, max_score);
    //     //match_scores.data()[max_anchor] = max_score;
    //     if(match_indices.data()[max_anchor] > -1){
    //        __atomic_fetch_add(gt_match_num.data() + match_indices.data()[max_anchor], 1, __ATOMIC_SEQ_CST);
    //       //gt_match_num.data()[match_indices.data()[max_anchor]] -= 1;
    //     }
    //     match_indices.data()[max_anchor] = worker_index;
    //   }
    // };

    // auto small_match_routine = [&overlaps, &match_indices, &match_scores, &gt_match_num, &gt_small_topk, num_anchors, num_ground_truth, negative_low_thres, negative_high_thres, positive_thres, stop_positive_thres, min_match](int64_t start, int64_t limit){
    //   for (int64_t worker_index = start; worker_index < limit; ++worker_index){
    //     if(gt_match_num.data()[worker_index] >= min_match) continue;
    //     int32_t *small_topk = gt_small_topk.data() + worker_index * min_match;
    //     // we set each of those ground truth's topk small anchor indices to -1, indicates that we have not found them
    //     for(int32_t index = 0;index < min_match;++index){
    //       small_topk[index] = -1;
    //     }
    //     // still have 'min_match' to match
    //     int32_t left_min_match = min_match - gt_match_num.data()[worker_index];
    //     // iterate all anchors to find topk small anchors
    //     for(int32_t index = 0;index < num_anchors;++index){
    //       T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
    //       // these topk anchors should not have been matched or in others topk lists, and the same time, they should have overlap scores greater than 'stop_positive_thres'
    //       if((match_indices.data()[index] < 0) && (match_indices.data()[index] > -3) && (cur_score > stop_positive_thres)){
    //         // update topk lists
    //         int32_t small_index = 0;
    //         for(;small_index < left_min_match;++small_index){
    //           if((small_topk[small_index]<0) || ((overlaps.data()[small_topk[small_index] * num_ground_truth + worker_index]) < cur_score)) break;
    //         }
    //         if(small_index < left_min_match){
    //           small_topk[small_index] = index;
    //           match_indices.data()[index] = -3;
    //         }
    //       }

    //       for(int32_t small_index = 0;small_index < left_min_match;++small_index){
    //         if(small_topk[small_index] < 0) continue;
    //         match_indices.data()[small_topk[small_index]] = worker_index;
    //         match_scores.data()[small_topk[small_index]] = overlaps.data()[small_topk[small_index] * num_ground_truth + worker_index];
    //       }

    //     }
    //   }
    // };

    const DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_anchors, num_ground_truth * 2, anchor_match_routine);
    // Shard(worker_threads.num_threads, worker_threads.workers, num_ground_truth, num_anchors * 2, gt_match_routine);
    // Shard(worker_threads.num_threads, worker_threads.workers, num_ground_truth, num_anchors * 4, small_match_routine);
    //std::cout <<"fini" << std::endl;
    for (int64_t worker_index = 0; worker_index < num_ground_truth; ++worker_index){
      int32_t max_anchor = 0;
      float max_score = std::numeric_limits<float>::lowest();

      std::vector<int32_t> vec_may_max;
      for(int32_t index = 0;index < num_anchors;++index){
        T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
        if(cur_score > max_score){
          max_anchor = index;
          max_score = cur_score;
        }
        if(std::abs(cur_score - max_score) < std::numeric_limits<float>::epsilon()){
          vec_may_max.push_back(index);
        }
      }
      // get all equal max anchors
      //std::vector<int32_t> all_equal_index(1, max_anchor);
      for(int32_t index : vec_may_max){
        T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
        if(std::abs(cur_score - max_score) < std::numeric_limits<float>::epsilon()){
          match_scores.data()[index] = cur_score;
          if(match_indices.data()[index] > -1){
            gt_match_num.data()[match_indices.data()[index]] -= 1;
          }
          match_indices.data()[index] = worker_index;
          gt_match_num.data()[worker_index] += 1;
        }
      }

      // for(int32_t index : all_equal_index){
      //   match_scores.data()[index] = max_score;
      //   if(match_indices.data()[index] > -1){
      //     gt_match_num.data()[match_indices.data()[index]] -= 1;
      //   }
      //   match_indices.data()[index] = worker_index;
      //   gt_match_num.data()[worker_index] += 1;
      // }
    }
    // hard face compensation
    for (int64_t worker_index = 0; worker_index < num_ground_truth; ++worker_index){
      if(gt_match_num.data()[worker_index] >= min_match) continue;

      std::priority_queue<DistancePair> match_queue;

      for(int32_t index = 0;index < num_anchors;++index){
        T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
        if((match_indices.data()[index] < 0) && (cur_score > stop_positive_thres)){
          match_queue.push(DistancePair(index, worker_index, cur_score));
        }
      }

      while(!match_queue.empty()) {
        DistancePair p = match_queue.top();
        //std::cout << p.dist << std::endl;
        if((gt_match_num.data()[worker_index] >= min_match)){
          break;
        }
        gt_match_num.data()[worker_index] += 1;
        match_scores.data()[p.index1] = p.dist;
        match_indices.data()[p.index1] = worker_index;
        match_queue.pop();
      }
    }

    // // Greedy bi-partite matching.
    // std::priority_queue<DistancePair> match_queue;

    // for (int64_t worker_index = 0; worker_index < num_ground_truth; ++worker_index){
    //   if(gt_match_num.data()[worker_index] >= min_match) continue;
    //   // iterate all anchors to find topk small anchors
    //   for(int64_t index = 0;index < num_anchors;++index){
    //     T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
    //     // these topk anchors should not have been matched or in others topk lists, and the same time, they should have overlap scores greater than 'stop_positive_thres'
    //     if((match_indices.data()[index] < 0) && (cur_score > stop_positive_thres)){
    //       match_queue.push(DistancePair(index, worker_index, cur_score));
    //     }
    //   }
    // }

    // while(!match_queue.empty()) {
    //   DistancePair p = match_queue.top();
    //   if((gt_match_num.data()[p.index2] >= min_match) || (match_indices.data()[p.index1] > -1)){
    //     match_queue.pop();
    //     continue;
    //   }
    //   gt_match_num.data()[p.index2] += 1;
    //   match_scores.data()[p.index1] = p.dist;
    //   match_indices.data()[p.index1] = p.index2;
    // }

    // for (int64_t worker_index = 0; worker_index < num_ground_truth; ++worker_index){
    //   if(gt_match_num.data()[worker_index] >= min_match) continue;
    //   int32_t *small_topk = gt_small_topk.data() + worker_index * min_match;
    //   // we set each of those ground truth's topk small anchor indices to -1, indicates that we have not found them
    //   for(int32_t index = 0;index < min_match;++index){
    //     small_topk[index] = -1;
    //   }
    //   // still have 'min_match' to match
    //   int32_t left_min_match = min_match - gt_match_num.data()[worker_index];
    //   // iterate all anchors to find topk small anchors
    //   for(int32_t index = 0;index < num_anchors;++index){
    //     T cur_score = overlaps.data()[index * num_ground_truth + worker_index];
    //     // these topk anchors should not have been matched or in others topk lists, and the same time, they should have overlap scores greater than 'stop_positive_thres'
    //     if((match_indices.data()[index] < 0) && (match_indices.data()[index] > -3) && (cur_score > stop_positive_thres)){
    //       // update topk lists
    //       int32_t small_index = 0;
    //       for(;small_index < left_min_match;++small_index){
    //         if((small_topk[small_index]<0) || ((overlaps.data()[small_topk[small_index] * num_ground_truth + worker_index]) < cur_score)) break;
    //       }
    //       if(small_index < left_min_match){
    //         small_topk[small_index] = index;
    //         match_indices.data()[index] = -3;
    //       }
    //     }

    //     for(int32_t small_index = 0;small_index < left_min_match;++small_index){
    //       if(small_topk[small_index] < 0) continue;
    //       match_indices.data()[small_topk[small_index]] = worker_index;
    //       match_scores.data()[small_topk[small_index]] = overlaps.data()[small_topk[small_index] * num_ground_truth + worker_index];
    //     }

    //   }
    // }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class SmallMiningMatchOp : public OpKernel {
 public:
  explicit SmallMiningMatchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("negative_low_thres", &negative_low_thres));
    OP_REQUIRES(context, (negative_low_thres >= 0) && (negative_low_thres < 1), errors::InvalidArgument("Need Attr 1 > grid_dim_width >= 0, got ", negative_low_thres));

    OP_REQUIRES_OK(context, context->GetAttr("negative_high_thres", &negative_high_thres));
    OP_REQUIRES(context, (negative_high_thres > negative_low_thres) && (negative_high_thres < 1.), errors::InvalidArgument("Need Attr 1 > negative_high_thres > negative_low_thres, got ", negative_high_thres));

    OP_REQUIRES_OK(context, context->GetAttr("positive_thres", &positive_thres));
    OP_REQUIRES(context, (positive_thres >= negative_high_thres) && (positive_thres < 1.), errors::InvalidArgument("Need Attr 1 > positive_thres >= negative_high_thres, got ", positive_thres));

    OP_REQUIRES_OK(context, context->GetAttr("stop_positive_thres", &stop_positive_thres));
    OP_REQUIRES(context, (stop_positive_thres >= 0.) && (stop_positive_thres < 1.), errors::InvalidArgument("Need Attr 1 > stop_positive_thres >= 0., got ", stop_positive_thres));

    OP_REQUIRES_OK(context, context->GetAttr("min_match", &min_match));
    OP_REQUIRES(context, min_match >= 1, errors::InvalidArgument("Need Attr grid_dim_height >= 1, got ", min_match));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& overlaps_in = context->input(0);

    OP_REQUIRES(context, overlaps_in.shape().dims() == 2, errors::InvalidArgument("inputs must be in 'num_anchors x num_ground_truth' format."));

    const int32_t num_anchors = overlaps_in.dim_size(0);
    const int32_t num_ground_truth = overlaps_in.dim_size(1);

    Tensor* match_indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {num_anchors}, &match_indices));
    Tensor* match_scores = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {num_anchors}, &match_scores));
    Tensor gt_match_num;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {num_ground_truth}, &gt_match_num));
    Tensor gt_small_topk;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {num_ground_truth, min_match}, &gt_small_topk));

    SmallMiningMatchFunctor<Device, T>()(context, context->eigen_device<Device>(), overlaps_in.template flat<T>(), match_indices->template flat<int32_t>(), match_scores->template flat<T>(), gt_match_num.template flat<int32_t>(), gt_small_topk.template flat<int32_t>(), num_anchors, num_ground_truth, negative_low_thres, negative_high_thres, positive_thres, stop_positive_thres, min_match);
  }

private:
  float negative_low_thres{0.};
  float negative_high_thres{0.5};
  float positive_thres{0.5};
  float stop_positive_thres{0.05};
  int32_t min_match{1};
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SmallMiningMatch").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SmallMiningMatchOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
// #if GOOGLE_CUDA == 1
// #define REGISTER_GPU(T)                                          \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("SmallMiningMatch").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//       SmallMiningMatchOp<GPUDevice, T>);
// REGISTER_GPU(float);
// #endif  // GOOGLE_CUDA
