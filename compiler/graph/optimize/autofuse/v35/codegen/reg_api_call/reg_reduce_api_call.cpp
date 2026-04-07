/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_reduce_api_call.h"

#include "codegen/api_call/reduce/reduce_api_call_base.h"
#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "common/checker.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "reg_api_call_utils.h"

namespace codegen {
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;
using namespace reduce_base;

Status RegReduceApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHECK_NOTNULL(node);
  auto node_in_anchor = node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(node_in_anchor);
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  const auto &in_node = std::dynamic_pointer_cast<ge::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(in_node);
  if (in_node->GetOutAllNodes().size() == 1UL) {
    is_reuse_source_ = "true";
  }
  return ge::SUCCESS;
}

Status RegReduceApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
   // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end() || is_reuse_source_ == "true", "RegReduceApiCall(id) cannot find tmp buffer id to use.");
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  auto iter = reduce_type_map.find(this->api_name_);
  GE_CHK_BOOL_RET_STATUS(iter != reduce_type_map.end(), ge::FAILED, "Codegen unsupported reg reduce api::%s", this->api_name_.c_str());
  auto &[type_value, instr_type] = iter->second;

  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string reduce_pattern;
  GetIsArAndPattern(y, x.isAr, reduce_pattern);

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen(reg reduce) get data type:%d failed", static_cast<int32_t>(y.dtype));
  GELOGI("Tensor::DtypeName(y.dtype) == %s", dtype_name.c_str());

  std::stringstream ss;

  ReduceMergedSizeCodeGen(tpipe, ss, x, y);

  ReduceDimACodeGen(x, this->api_name_, ss);

  ReduceInitCodeGen(x, y, type_value, ss, tpipe, dtype_name);

  ss << "uint32_t tmp_reduce_shape[] = {first_actual, last};" << std::endl;

  std::string new_api_name = this->api_name_ == "Mean" ? "Sum" : this->api_name_;
  if (!IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back())) {
    ss << "Reduce" << new_api_name << "<" << dtype_name << ", " << reduce_pattern << ", " << is_reuse_source_ << ">("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], ";
    is_reuse_source_ == "true" ? ss << "" : ss << tpipe.tmp_buf << "_" << std::to_string(id) << ", ";
    ss << "tmp_reduce_shape, true);" << std::endl;

    if (this->api_name_== "Mean") {
      ReduceMeanCodeGen(dtype_name, tpipe, y, ss);
    }
  } else {
    life_time_axis_id = 0L;
    int64_t tmp_reduce_id = -1L;
    auto it = this->tmp_buf_id.find(life_time_axis_id);
    GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "RegReduceApiCall(tmp_reduce_id) cannot find tmp buffer id to use.");
    tmp_reduce_id = it->second;
    ss << "LocalTensor<" << dtype_name << "> tmp_reduce;" << std::endl;
    ss << "tmp_reduce = " << tpipe.tmp_buf << "_" << std::to_string(tmp_reduce_id) << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;
    ss << "Reduce" << new_api_name << "<" << dtype_name << "," << reduce_pattern << ", " << is_reuse_source_ << ">"
       << "(tmp_reduce[0], " << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], ";
    is_reuse_source_ == "true" ? ss << "" : ss << tpipe.tmp_buf << "_" << std::to_string(id) << ", ";
    ss << "tmp_reduce_shape, true);" << std::endl;
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
    ss << "uint32_t temp_size = " << KernelUtils::SizeAlign() << "(" << y.actual_size << ", 32/sizeof(" << dtype_name << "));" << std::endl;
    ss << "if (" << tpipe.tiler.GetAxis(current_axis.back()) << " == 0) {" << std::endl;
    ss << "DataCopyExtend(" << y << "[0], " << "tmp_reduce[0], " << "temp_size);" << std::endl;
    ss << "} else {" << std::endl;
    ss << "AscendC::" << instr_type << "(" << y << "[0], " << "tmp_reduce[0], " << y << "[0], temp_size);\n" << "}" << std::endl;
  }

  ss << "}" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<RegReduceApiCall> register_reduce_api_call("RegReduceApiCall");

}  // namespace codegen