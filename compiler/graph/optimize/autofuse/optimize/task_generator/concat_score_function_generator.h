/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCORE_FUNCTION_GENERATOR_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCORE_FUNCTION_GENERATOR_H_

#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"

namespace optimize {
class ConcatScoreFunctionGenerator {
 public:
  ConcatScoreFunctionGenerator(const ascir::HintGraph &graph, ge::AscNodePtr concat_node, uint32_t concat_dim);
  Status Generate(std::string &score_func);
  Status GenerateForCheckSmallTail(std::string &score_func);
  static void GenerateScoreOne(std::string &score_func);

 private:
  Status ParseStride();
  Status GenerateForUnaligned();
  Status TryGetScoreByConstExpr(int32_t &score);
  void GenerateReturnValue(int32_t score);

  const double kMaxUnalignedRate = 0.1;  // TTODO
  const uint32_t kAlignment_ = 32U;

  const ascir::HintGraph *graph_;
  ge::AscNodePtr concat_node_;
  uint32_t concat_dim_;
  ge::Expression stride_;
  ge::Expression elt_num_after_concat_dim_;
  size_t const_part_stride_ = 0U;
  uint32_t num_inputs_ = 0U;
  std::stringstream ss_;
};
}  // optimize
#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCORE_FUNCTION_GENERATOR_H_
