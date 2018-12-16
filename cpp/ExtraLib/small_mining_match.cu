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
#if GOOGLE_CUDA == 1
#define EIGEN_USE_GPU
#include "small_mining_match.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

#include <cstdint>
#include <cmath>
#include <cfloat>

// Define the CUDA kernel.
template <typename T>
__global__ void AnchorMatchCudaKernel(CudaLaunchConfig config, const T * overlaps, int32_t * match_indices, T * match_scores, int32_t * gt_match_num, const int32_t num_anchors, const int32_t num_ground_truth, const float negative_low_thres, const float negative_high_thres, const float positive_thres) {

  CUDA_1D_KERNEL_LOOP(worker_index, config.virtual_thread_count) {
    const T * anchor_overlaps = overlaps + worker_index * num_ground_truth;
    int32_t max_gt = 0;
    float max_score = std::numeric_limits<float>::lowest();

    for(int32_t index = 0;index < num_ground_truth;++index){
      if(ldg(anchor_overlaps+index) > max_score){
        max_gt = index;
        max_score = ldg(anchor_overlaps+index);
      }
    }
    match_scores[worker_index] = max_score;

    if((max_score >= negative_low_thres) && (max_score < negative_high_thres)){
      match_indices[worker_index] = -1;
    }else if(max_score > positive_thres){
      match_indices[worker_index] = max_gt;
      gt_match_num[max_gt] += 1;
    }else match_indices[worker_index] = -2;
  }
}

template <typename T>
__global__ void GtMatchCudaKernel(CudaLaunchConfig config, const T * overlaps, int32_t * match_indices, T * match_scores, int32_t * gt_match_num, const int32_t num_anchors, const int32_t num_ground_truth, const float negative_low_thres, const float negative_high_thres, const float positive_thres) {
  CUDA_1D_KERNEL_LOOP(worker_index, config.virtual_thread_count) {
    int32_t max_anchor = 0;
    float max_score = std::numeric_limits<float>::lowest();

    for(int32_t index = 0;index < num_anchors;++index){
      T cur_score = ldg(overlaps + index * num_ground_truth + worker_index);
      if(cur_score > max_score){
        max_anchor = index;
        max_score = cur_score;
      }
    }

    match_scores[max_anchor] = max_score;
    if(match_indices[max_anchor] > -1){
      gt_match_num[match_indices[max_anchor]] -= 1;
    }
    match_indices[max_anchor] = worker_index;
  }
}

template <typename T>
__global__ void SmallMatchCudaKernel(CudaLaunchConfig config, const T * overlaps, int32_t * match_indices, T * match_scores, int32_t * gt_match_num, int32_t * gt_small_topk, const int32_t num_anchors, const int32_t num_ground_truth, const float stop_positive_thres, const int32_t min_match) {

  CUDA_1D_KERNEL_LOOP(worker_index, config.virtual_thread_count) {
    if(gt_match_num[worker_index] >= min_match) continue;
    int32_t *small_topk = gt_small_topk + worker_index * min_match;
    // we set each of those ground truth's topk small anchor indices to -1, indicates that we have not found them
    for(int32_t index = 0;index < min_match;++index){
      small_topk[index] = -1;
    }
    // still have 'min_match' to match
    int32_t left_min_match = min_match - gt_match_num[worker_index];
    // iterate all anchors to find topk small anchors
    for(int32_t index = 0;index < num_anchors;++index){
      T cur_score = ldg(overlaps+index * num_ground_truth + worker_index);
      // these topk anchors should not have been matched or in others topk lists, and the same time, they should have overlap scores greater than 'stop_positive_thres'
      if((match_indices[index] < 0) && (match_indices[index] > -3) && (cur_score > stop_positive_thres)){
        // update topk lists
        int32_t small_index = 0;
        for(;small_index < left_min_match;++small_index){
          if((small_topk[small_index]<0) || (ldg(overlaps+small_topk[small_index] * num_ground_truth + worker_index) < cur_score)) break;
        }
        if(small_index < left_min_match){
          small_topk[small_index] = index;
          match_indices[index] = -3;
        }
      }

      for(int32_t small_index = 0;small_index < left_min_match;++small_index){
        if(small_topk[small_index] < 0) continue;
        match_indices[small_topk[small_index]] = worker_index;
        match_scores[small_topk[small_index]] = overlaps[small_topk[small_index] * num_ground_truth + worker_index];
      }
    }

  }
}

template <typename T>
void SmallMiningMatchFunctor<GPUDevice, T>::operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat overlaps, typename TTypes<int32_t>::Flat match_indices, typename TTypes<T>::Flat match_scores, typename TTypes<int32_t>::Flat gt_match_num, typename TTypes<int32_t>::Flat gt_small_topk, const int32_t num_anchors, const int32_t num_ground_truth, const float negative_low_thres, const float negative_high_thres, const float positive_thres, const float stop_positive_thres, const int32_t min_match) {

    CudaLaunchConfig gt_config = GetCudaLaunchConfig(num_ground_truth, d);
    CudaLaunchConfig anchor_config = GetCudaLaunchConfig(num_anchors, d);

    SetZero <<<gt_config.block_count, gt_config.thread_per_block, 0, d.stream()>>> (num_ground_truth, gt_match_num.data());

    AnchorMatchCudaKernel <<<anchor_config.block_count, anchor_config.thread_per_block, 0, d.stream()>>> (anchor_config, overlaps.data(), match_indices.data(), match_scores.data(), gt_match_num.data(), num_anchors, num_ground_truth, negative_low_thres, negative_high_thres, positive_thres);

    GtMatchCudaKernel <<<gt_config.block_count, gt_config.thread_per_block, 0, d.stream()>>> (gt_config, overlaps.data(), match_indices.data(), match_scores.data(), gt_match_num.data(), num_anchors, num_ground_truth, negative_low_thres, negative_high_thres, positive_thres);

    SmallMatchCudaKernel <<<gt_config.block_count, gt_config.thread_per_block, 0, d.stream()>>> (gt_config, overlaps.data(), match_indices.data(), match_scores.data(), gt_match_num.data(), gt_small_topk.data(), num_anchors, num_ground_truth, stop_positive_thres, min_match);

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }
}

template struct SmallMiningMatchFunctor<GPUDevice, float>;
// #define DEFINE_GPU_SPECS(T)   \
//   template struct SmallMiningMatchFunctor<T>;

// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
