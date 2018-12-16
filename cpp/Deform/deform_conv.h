/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2017 by Contributors
 * \file deformable_im2col.h
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, and dilation.
 * These functions are mainly used in convolution operators.
 * The implementation of the im2col and col2im algorithms
 * are copied from Caffe with minor interface modifications
 * adapting to MXNet data structures.
 */

#ifndef TENSORFLOW_KERNELS_CONV_OPS_im2col_H_
#define TENSORFLOW_KERNELS_CONV_OPS_im2col_H_

// #define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>

namespace tensorflow {
// typedef Eigen::ThreadPoolDevice CPUDevice;
typedef std::vector<int> TShape;
// typedef Eigen::GpuDevice GPUDevice;

namespace functor {

/*!
 * \brief cpu function of im2col algorithm
 * \param data_im pointer of a image (C, H, W,...) in the image batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_col start pointer of the column buffer to be filled
 */
template <typename Device, typename DType>
struct deformable_im2col {
    void operator()(const Device& d,
                    const DType* data_im, const DType* data_offset,
                    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
                    const TShape& pad, const TShape& stride, const TShape& dilation,
                    const int deformable_group, DType* data_col);
};

/*!\brief
 * cpu function of col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_im start pointer of the image data
 * \param data_offset start pointer of the offset data
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param grad_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename Device, typename DType>
struct deformable_col2im {
    void operator()(const Device& d,
                    const DType* data_col, const DType* data_offset,
                    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
                    const TShape& pad, const TShape& stride,
                    const TShape& dilation, const int deformable_group,
                    DType* grad_im);
};



template <typename Device, typename DType>
struct deformable_col2im_coord {
    void operator()(const Device& d,
                    const DType* data_col, const DType* data_im, const DType* data_offset, const TShape& im_shape,
                    const TShape& col_shape, const TShape& kernel_shape,
                    const TShape& pad, const TShape& stride,
                    const TShape& dilation, const int deformable_group, DType* grad_offset);
};

template <typename Device, typename DType>
struct im2col {
    void operator() (const Device& d,
                     const DType* data_im, const TShape& im_shape,
                     const TShape& col_shape, const TShape& kernel_shape,
                     const TShape& pad, const TShape& stride,
                     const TShape& dilation, DType* data_col);
};

template <typename Device, typename DType>
struct pureAddTo {
    void operator() (const Device& d, const int n, DType* result_data, const DType* right_data);
};

template <typename Device, typename DType>
struct pureSubTo {
    void operator() (const Device& d, const int n, DType* result_data, const DType* right_data);
};

template <typename Device, typename DType>
struct setZero {
    void operator() (const Device& d, const int n, DType* result_data);
};


}  // namespace functor
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_im2col_H_
