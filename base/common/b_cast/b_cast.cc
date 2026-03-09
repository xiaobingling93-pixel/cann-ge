/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/b_cast/b_cast.h"

#include <vector>

#include "graph/utils/math_util.h"
#include "common/util.h"
#include "base/err_msg.h"

namespace ge {
Status BCast::GenerateBcastInfo(const kVecInt &sx, const kVecInt &sy) {
  if ((sx.size() == 0U) && (sy.size() == 0U)) {
    result_.push_back(1);
    x_reshape_.push_back(1);
    x_bcast_.push_back(1);
    y_reshape_.push_back(1);
    y_bcast_.push_back(1);
  } else {
    kVecInt x = sx;
    kVecInt y = sy;
    Reverse(x);
    Reverse(y);
    ExtendTensorDim(x, y);
    GE_RETURN_WITH_LOG_IF_ERROR(SetShapeDifferentInfo(x, y), "[Set][ShapeDifferentInfo] GenerateBcastInfo failed.");
  }
  ReverseAllIntermediateShapes();
  return domi::SUCCESS;
}

Status BCast::SetShapeDifferentInfo(const kVecInt &x, const kVecInt &y) {
  const int64_t n = static_cast<int64_t>(x.size());
  for (int64_t i = 0; i < n; ++i) {
    const int64_t x_i = x[static_cast<size_t>(i)];
    GE_CHECK_GE(x_i, 0);
    const int64_t y_i = y[static_cast<size_t>(i)];
    GE_CHECK_GE(y_i, 0);
    int64_t output_i = 0;
    int64_t x_bcast_i = 0;
    int64_t y_bcast_i = 0;

    if (x_i == y_i) {
      output_i = x_i;
      x_bcast_i = 1;
      y_bcast_i = 1;
      if (x_i == 1) {
        grad_x_reduce_idx_.push_back(n - 1 - i);
        grad_y_reduce_idx_.push_back(n - 1 - i);
      }
    } else if (x_i == 1) {
      output_i = y_i;
      x_bcast_i = y_i;
      y_bcast_i = 1;
      grad_x_reduce_idx_.push_back(n - 1 - i);
    } else if (y_i == 1) {
      output_i = x_i;
      x_bcast_i = 1;
      y_bcast_i = x_i;
      grad_y_reduce_idx_.push_back(n - 1 - i);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "SetShapeDifferentInfo failed. Two tensor shapes are not compatible "
                         "according to the broadcasting rule.");
      GELOGE(domi::PARAM_INVALID,
             "[Check][Param] SetShapeDifferentInfo failed. Two tensor shapes are not compatible "
             "according to the broadcasting rule.");
      return domi::PARAM_INVALID;
    }
    output_.push_back(output_i);
    result_.push_back(output_i);
    x_reshape_.push_back(x_i);
    x_bcast_.push_back(x_bcast_i);
    y_reshape_.push_back(y_i);
    y_bcast_.push_back(y_bcast_i);
  }
  return domi::SUCCESS;
}

void BCast::ExtendTensorDim(kVecInt &v_x, kVecInt &v_y) {
  if (v_x.size() > v_y.size()) {
    v_y.resize(v_x.size(), 1);
  } else {
    v_x.resize(v_y.size(), 1);
  }
}

BCast::kVecInt BCast::TransShapeToDimVec(const GeTensorDesc &shape) {
  const size_t dim_num = shape.GetShape().GetDimNum();
  BCast::kVecInt ret(dim_num);
  for (size_t i = 0U; i < dim_num; ++i) {
    ret[i] = shape.GetShape().GetDim(i);
  }
  return ret;
}

void BCast::Reverse(kVecInt &shape) { std::reverse(shape.begin(), shape.end()); }

void BCast::ReverseAllIntermediateShapes() {
  // Reverse all intermediate shape params
  Reverse(x_reshape_);
  Reverse(x_bcast_);
  Reverse(y_reshape_);
  Reverse(y_bcast_);
  Reverse(result_);
  Reverse(output_);
  Reverse(grad_x_reduce_idx_);
  Reverse(grad_y_reduce_idx_);
}

void BCast::BCastIndexes(kVecInt &x_indexes, kVecInt &y_indexes) {
  Reverse(x_reshape_);
  Reverse(y_reshape_);
  Reverse(output_);

  // Process 0-th dimension
  int64_t x_dim = 1;
  int64_t y_dim = 1;
  int64_t out_dim = 1;

  // If x and y are both scalar, then output_ is empty
  if (!output_.empty()) {
    x_dim = x_reshape_.at(0U);
    y_dim = y_reshape_.at(0U);
    out_dim = output_.at(0U);
  }

  int64_t x_bias = x_dim;
  int64_t y_bias = y_dim;

  for (int64_t i = 0; i < out_dim; i++) {
    const int64_t x_index = (x_dim == 1) ? 0 : i;
    const int64_t y_index = (y_dim == 1) ? 0 : i;
    x_indexes.push_back(x_index);
    y_indexes.push_back(y_index);
  }

  // Process the remaining dimensions
  for (size_t i = 1U; i < output_.size(); i++) {
    x_dim = x_reshape_.at(i);  // i-th dimension of x.
    y_dim = y_reshape_.at(i);  // i-th dimension of y.
    out_dim = output_.at(i);   // i-th dimension of output_.

    const int64_t stride = static_cast<int64_t>(x_indexes.size());
    for (int64_t j = 1; j < out_dim; j++) {
      for (int64_t k = 0; k < stride; k++) {
        const int64_t x_bias_tmp = (x_dim == 1) ? 0 : (j * x_bias);
        const int64_t y_bias_tmp = (y_dim == 1) ? 0 : (j * y_bias);
        const int64_t x_index = x_indexes.at(static_cast<size_t>(k)) + x_bias_tmp;
        const int64_t y_index = y_indexes.at(static_cast<size_t>(k)) + y_bias_tmp;
        x_indexes.push_back(x_index);
        y_indexes.push_back(y_index);
      }
    }
    x_bias *= x_dim;
    y_bias *= y_dim;
  }

  Reverse(x_reshape_);
  Reverse(y_reshape_);
  Reverse(output_);
}
}  // namespace ge
