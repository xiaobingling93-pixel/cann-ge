/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "square_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "common/checker.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

Status SquareApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                              std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  stringstream ss;
  string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(x.dtype, blk_align), "Codegen blk align failed");

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x.dtype));

  // 使用Mul算子实现Square：Mul(y, x, x, size)
  ss << "Mul(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
     << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
     << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
     << x.actual_size << ");" << std::endl;

  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<SquareApiCall> register_square_api_call("SquareApiCall");
}  // namespace codegen
