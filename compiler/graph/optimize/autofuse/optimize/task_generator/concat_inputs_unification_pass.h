/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_TASK_GENERATOR_CONCAT_INPUTS_UNIFICATION_PASS_H
#define OPTIMIZE_TASK_GENERATOR_CONCAT_INPUTS_UNIFICATION_PASS_H

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"

namespace optimize {

class ConcatInputUnificationPass {
 public:
  static Status Run(std::vector<ascir::ImplGraph> &graphs);

 private:
  static Status RunOneGraph(ascir::ImplGraph &graph);
  static bool NeedOptimize(const ge::AscNodePtr &concat_node);
  static Status DoOptimize(ascir::ImplGraph &graph, const ge::AscNodePtr &concat_node);
  static ge::Expression GetColSize(const ge::AscTensor &tensor, size_t concat_dim);
  static ge::Status GetLoadNum(const ge::AscNodePtr &concat_node, uint32_t &load_num);
  static bool IsSrcColSizeAlignedToB4(const ge::AscNodePtr &concat_node, size_t concat_dim, int32_t dtype_size);
  static bool IsDstColSizeOverLimit(const ge::AscNodePtr &concat_node, size_t concat_dim, int32_t dtype_size);
};

}  // optimize

#endif  // OPTIMIZE_TASK_GENERATOR_CONCAT_INPUTS_UNIFICATION_PASS_H
