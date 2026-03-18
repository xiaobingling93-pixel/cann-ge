/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_TEMPLATE_NDDMA_TEMPLATE_H
#define OPTIMIZE_PLATFORM_V2_TEMPLATE_NDDMA_TEMPLATE_H

#include "platform/common/base_template.h"

namespace optimize {

class NddmaTemplate : public BaseTemplate {
 public:
  ~NddmaTemplate() override = default;
  explicit NddmaTemplate() = default;

  std::string GenName(const std::string &general_case_name) override;
  static Status GenNddmaNode(const ge::AscNodePtr &node_pre, const ge::AscNodePtr &node_brc, ge::AscGraph &new_case,
                             const bool is_need_realignment = false);
  static Status AddTransposeNodeAfter(ge::AscGraph &graph, const ge::AscNodePtr &node,
                                      ge::AscNodePtr &new_transpose_node, const ge::AscNodePtr &old_transpose_node);
  static Status MergeLoadAndTranspose(const ge::AscNodePtr &load_node, ge::AscGraph& new_case);
  static Status TransposeToNddmaNode(const ge::AscNodePtr &transpose_node, ge::AscGraph& new_case);
  ge::Status Generate(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                      ge::AscGraph &new_case) override;
  static Status SwapCastBrcAndGenNddma(const ge::AscNodePtr &node_cast, const ge::AscNodePtr &node_load,
      ge::AscGraph &new_case);
  bool NeedDropBasedCase(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                         const ge::AscGraph &new_case) override;
  static Status ReAlignVectorizedStrides(const ge::AscNodePtr &node);
  static bool IsSecondaryTailAxisAligned(const ge::AscNodePtr &node);
  std::string GetScoreFunc(const ge::AscGraph &origin_graph, const ge::AscGraph &nddma_graph) override;
  static ge::Status ReorderRepeats(const ge::AscNodePtr &node_src, const ge::AscNodePtr &node_dst);
  NddmaTemplate(const NddmaTemplate &) = delete;
  NddmaTemplate &operator=(const NddmaTemplate &) = delete;
  NddmaTemplate(NddmaTemplate &&) = delete;
  NddmaTemplate &operator=(NddmaTemplate &&) = delete;
};
}  // namespace optimize

#endif //OPTIMIZE_PLATFORM_V2_TEMPLATE_NDDMA_TEMPLATE_H