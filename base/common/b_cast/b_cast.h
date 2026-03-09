/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_B_CAST_H_
#define GE_GRAPH_COMMON_B_CAST_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "common/debug/log.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_adapter.h"
#include "base/err_msg.h"

namespace ge {
class BCast {
 public:
  /// @ingroup domi_calibration
  /// @brief define kVecInt
  using kVecInt = std::vector<int64_t>;

  /// @ingroup domi_calibration
  /// @brief constructor
  BCast() {}
  /// @ingroup domi_calibration
  /// @brief destructor
  ~BCast() = default;

  /// @ingroup domi_calibration
  /// @brief Not optimize intermediate shapes
  /// @decrease dims, more efficient, set by user
  /// @param [in] x   first Tensor dim
  /// @param [in] y   second Tensor dim
  /// @return     SUCCESS broadcast message successfully generated
  /// @return     other   broadcast message failed to generate
  ge::Status GenerateBcastInfo(const kVecInt &sx, const kVecInt &sy);

  /// @ingroup domi_calibration
  /// @brief get result_shape
  const kVecInt &GetOutputShape() const { return output_; }
  const kVecInt &GetGradXReduceIdx() const { return grad_x_reduce_idx_; }
  const kVecInt &GetGradYReduceIdx() const { return grad_y_reduce_idx_; }

  /// @ingroup domi_calibration
  /// @brief convert TensorDescriptor to kVecInt
  /// @param [in] shape   Tensor descriptor
  /// @return     kVecInt     dim info
  static kVecInt TransShapeToDimVec(const GeTensorDesc &shape);

  void BCastIndexes(kVecInt &x_indexes, kVecInt &y_indexes);
  template <typename InT, typename OutT>
  Status BCastCompute(const std::vector<ConstGeTensorPtr> &input, std::vector<OutT> &v_output,
                      const std::function<OutT(InT const &, InT const &)> &func) {
    Status ret;
    if (func == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Check param func nullptr");
      GELOGE(domi::PARAM_INVALID, "Param func is null");
      return domi::PARAM_INVALID;
    }
    // Min input num is 2
    constexpr size_t kMinDimNum = 2U;
    if (input.size() < kMinDimNum) {
      REPORT_INNER_ERR_MSG("E19999", "Param input.size():%zu < %zu, check invalid",
                         input.size(), kMinDimNum);
      GELOGE(domi::PARAM_INVALID, "Input size is smaller than two.");
      return domi::PARAM_INVALID;
    }
    // Only broadcast shape
    ret =
      GenerateBcastInfo(TransShapeToDimVec(input[0U]->GetTensorDesc()),
                        TransShapeToDimVec(input[1U]->GetTensorDesc()));
    if (ret != domi::SUCCESS) {
      GELOGE(ret, "Greater broadcasting failed.");
      return ret;
    }

    kVecInt x_indexes;
    kVecInt y_indexes;
    BCastIndexes(x_indexes, y_indexes);

    const void *const x1_data = input[0U]->GetData().data();
    const void *const x2_data = input[1U]->GetData().data();
    const size_t x1_size = input[0U]->GetData().size();
    const size_t x2_size = input[1U]->GetData().size();

    for (size_t i = 0U; i < x_indexes.size(); i++) {
      const int64_t x_index = x_indexes[i];
      const int64_t y_index = y_indexes[i];
      const InT *const ptr_x1 = reinterpret_cast<const InT *>(x1_data);
      const InT *const ptr_x2 = reinterpret_cast<const InT *>(x2_data);
      const auto value = func(*(PtrAdd<const InT>(ptr_x1, x1_size, x_index)),
                              *(PtrAdd<const InT>(ptr_x2, x2_size, y_index)));
      v_output.push_back(value);
    }

    return domi::SUCCESS;
  }

  template <typename InT, typename OutT>
  Status BCastComputeCheck(const std::vector<ConstGeTensorPtr> &input, std::vector<OutT> &v_output,
                           const std::function<OutT(InT const &, InT const &, DataType &type, Status &)> &func) {
    if (func == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Check param func nullptr");
      GELOGE(PARAM_INVALID, "Param func is null");
      return PARAM_INVALID;
    }
    // Min input num is 2
    constexpr size_t kMinDimNum = 2U;
    if (input.size() < kMinDimNum) {
      REPORT_INNER_ERR_MSG("E19999", "Param input.size():%zu < %zu, check invalid",
                         input.size(), kMinDimNum);
      GELOGE(PARAM_INVALID, "Input size is smaller than two.");
      return PARAM_INVALID;
    }
    // Only broadcast shape
    Status ret =
      GenerateBcastInfo(TransShapeToDimVec(input[0U]->GetTensorDesc()),
                        TransShapeToDimVec(input[1U]->GetTensorDesc()));
    if (ret != SUCCESS) {
      GELOGE(ret, "Greater broadcasting failed.");
      return ret;
    }

    DataType data_type = input[0U]->GetTensorDesc().GetDataType();
    kVecInt x_indexes;
    kVecInt y_indexes;
    BCastIndexes(x_indexes, y_indexes);

    const void *const x1_data = input[0U]->GetData().data();
    const void *const x2_data = input[1U]->GetData().data();
    const size_t x1_size = input[0U]->GetData().size();
    const size_t x2_size = input[1U]->GetData().size();

    for (size_t i = 0U; i < x_indexes.size(); i++) {
      const int64_t x_index = x_indexes[i];
      const int64_t y_index = y_indexes[i];
      const InT *const ptr_x1 = reinterpret_cast<const InT *>(x1_data);
      const InT *const ptr_x2 = reinterpret_cast<const InT *>(x2_data);
      const auto value = func(*(PtrAdd<const InT>(ptr_x1, x1_size, x_index)),
                              *(PtrAdd<const InT>(ptr_x2, x2_size, y_index)), data_type, ret);
      
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "BCastComputeCheck func execute failed, datatype is %d.", data_type);
        GELOGE(ret, "BCastComputeCheck func execute failed, datatype is %d.", data_type);
        return ret;
      }
      v_output.push_back(value);
    }

    return SUCCESS;
  }

 private:
  /// @ingroup domi_calibration
  /// @brief reverse elements in kVecInt
  /// @param [in] shape   dim info
  /// @return null
  static void Reverse(kVecInt &shape);

  /// @ingroup domi_calibration
  /// @brief two Tensor with different shape, set broadcast info
  /// @param [in] x   first input Tensor dim info
  /// @param [in] y   second input Tensor dim info
  /// @return null
  ge::Status SetShapeDifferentInfo(const kVecInt &x, const kVecInt &y);
  /// @ingroup domi_calibration
  /// @brief extend Tensor dim
  /// @param [in] x   first input Tensor dim info
  /// @param [in] y   second input Tensor dim info
  /// @return null
  static void ExtendTensorDim(kVecInt &v_x, kVecInt &v_y);
  /// @ingroup domi_calibration
  /// @brief reverse all intermediate shape params
  /// @param [in] void
  /// @return null
  void ReverseAllIntermediateShapes();

  kVecInt x_reshape_;
  kVecInt x_bcast_;
  kVecInt y_reshape_;
  kVecInt y_bcast_;
  kVecInt result_;
  kVecInt output_;
  kVecInt grad_x_reduce_idx_;
  kVecInt grad_y_reduce_idx_;
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_BCAST_H_
