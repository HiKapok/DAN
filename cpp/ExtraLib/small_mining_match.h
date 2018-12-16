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
#ifndef SMALL_MINING_MATCH_H_
#define SMALL_MINING_MATCH_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <cstdint>
#include <tuple>
#include <limits>
#include <iostream>

using tensorflow::TTypes;
using tensorflow::OpKernelContext;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template <typename Device, typename T>
struct SmallMiningMatchFunctor {
  void operator()(OpKernelContext* context, const Device& d, typename TTypes<T>::ConstFlat overlaps, typename TTypes<int32_t>::Flat match_indices, typename TTypes<T>::Flat match_scores, typename TTypes<int32_t>::Flat gt_match_num, typename TTypes<int32_t>::Flat gt_small_topk, const int32_t num_anchors, const int32_t num_ground_truth, const float negative_low_thres, const float negative_high_thres, const float positive_thres, const float stop_positive_thres, const int32_t min_match);
};

// #if GOOGLE_CUDA == 1
// template <typename T>
// struct SmallMiningMatchFunctor<GPUDevice, T> {
//   void operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat overlaps, typename TTypes<int32_t>::Flat match_indices, typename TTypes<T>::Flat match_scores, typename TTypes<int32_t>::Flat gt_match_num, typename TTypes<int32_t>::Flat gt_small_topk, const int32_t num_anchors, const int32_t num_ground_truth, const float negative_low_thres, const float negative_high_thres, const float positive_thres, const float stop_positive_thres, const int32_t min_match);
// };
// #endif

#endif // SMALL_MINING_MATCH_H_

