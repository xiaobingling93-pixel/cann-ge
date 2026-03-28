/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "truediv_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils//asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

Status TrueDivApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  size_t x1_idx = 0;
  size_t x2_idx = 1;

  auto x1 = inputs[x1_idx].get();
  auto x2 = inputs[x2_idx].get();

  auto y = outputs[0].get();
  stringstream ss;
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "TrueDivApiCall cannot find tmp buffer id to use.");
  id = it->second;

  std::string x1_dtype_name;
  std::string x2_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x1.dtype, x1_dtype_name), "Codegen get x1 data type:%d failed",
                    static_cast<int32_t>(x1.dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, x2_dtype_name), "Codegen get x2 data type:%d failed",
                    static_cast<int32_t>(x2.dtype));
  GE_ASSERT_TRUE(x1_dtype_name == x2_dtype_name, "x1_dtype_name:%s, x2_dtype_name:%s", x1_dtype_name.c_str(),
                 x2_dtype_name.c_str());
  if (x1_dtype_name == "int32_t") {
    ss << this->api_name_ << "<int32_t, float>("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
       << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
       << x1.actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id) << ");" << std::endl;
  } else {
    ss << this->api_name_ << "<" << x1_dtype_name << ">("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
       << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
       << x1.actual_size << ");" << std::endl;
  }

  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<TrueDivApiCall> register_truediv_api_call("TrueDivApiCall");

}  // namespace codegen