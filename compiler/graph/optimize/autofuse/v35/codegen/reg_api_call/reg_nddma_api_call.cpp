/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "reg_api_call_utils.h"
#include "reg_nddma_api_call.h"

using namespace ge::ops;
using namespace ge::ascir_op;

namespace {
constexpr uint64_t kNddmaMaxLen = 5UL;
}

namespace codegen {
Status NddmaApiCall::ParseAttr(const ascir::NodeView &node) {
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return ge::SUCCESS;
}

Status NddmaApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                              std::string &result) const {
  std::stringstream ss;
  const auto &gm = inputs[0].get();
  const auto &ub = outputs[0].get();
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    GE_ASSERT_TRUE(gm.axis_size.size() >= 2U, "Nddma src axis-size less than 2 is invalid in CV-Fusion case");
    auto last_index = gm.axis_size.size() - 1U;
    auto second_to_last_index = gm.axis_size.size() - 2U;
    ss << "const int64_t output_dims_" << ub.id << "[2] = {curAivM, curAlignN};" << std::endl;
    std::string gm_offset;
    if ((gm.axis_size[second_to_last_index] == One) && (gm.axis_size[last_index] == One)) {
      ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 0};" << std::endl;
      ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
      gm_offset = "batch_num";
    } else if (gm.axis_size[second_to_last_index] == One) {
      ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 1};" << std::endl;
      ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
      gm_offset = "offset % shapeN + batch_num * shapeN";
    } else if (gm.axis_size[last_index] == One) {
      ss << "const int64_t input_stride_" << ub.id << "[2] = {1, 0};" << std::endl;
      ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
      gm_offset = "offset / shapeN";
    }
    ss << api_name_ << "(" << ub << ", " << gm << "[" << gm_offset << "], " << "output_dims_" << ub.id << ", "
       << "output_stride_" << ub.id << ", " << "input_stride_" << ub.id << ");" << std::endl;
  } else {
    DataCopyParams param;
    GE_ASSERT_TRUE(CalculateDmaParams(tpipe, ub, ub, param), "CalculateDmaParams failed");
    const std::string gm_offset = ub.is_ub_scalar ? "0" : tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides); // 每次从gm搬到ub的偏移量
    if (param.repeats.size() <= kNddmaMaxLen) {
      NddmaParams nddma_param;
      SetNddmaParams(tpipe, param, nddma_param, ub.id, ss);
      ss << api_name_ << "(" << ub << ", " << gm << "[" << gm_offset << " + " << tpipe.tiler.Size(offset_) << "], "
         << "output_dims_" << ub.id << ", " << "output_stride_" << ub.id << ", " << "input_stride_" << ub.id << ");"
         << std::endl;
    } else {
      CreateNddmaCall(tpipe, gm, ub, gm_offset, param, offset_, ss);
    }
  }
  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<NddmaApiCall> register_nddma_api_call("NddmaApiCall");
}  // namespace codegen