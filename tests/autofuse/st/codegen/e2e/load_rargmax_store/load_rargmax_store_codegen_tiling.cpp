/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"

#include <sstream>
#include "e2e_common.h"

extern "C" bool CodegenTiling(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                              std::map<std::string, std::string> &options,
                              std::map<std::string, std::string> &tiling_func) {
  std::stringstream ss;

  ss << OptilingStub(fused_schedule_result) << std::endl;
  ss << "extern \"C\" void GetTiling(AutofuseTilingData& tiling_data) {" << std::endl;
  ss << "  tiling_data.set_z0Tb_size(4);" << std::endl;
  ss << "  tiling_data.set_z0t_size(tiling_data.get_s0() / 48 / 4);" << std::endl;
  ss << "  tiling_data.set_z1t_size(tiling_data.get_s1());" << std::endl;
  ss << "}" << std::endl;

  tiling_func["TilingHead"] += ss.str();
  return true;
}
