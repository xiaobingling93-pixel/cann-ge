/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_LOWERING_LOWERING_UTILS_H_
#define AUTOFUSE_LOWERING_LOWERING_UTILS_H_
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include "graph/node.h"
#include "graph/utils/op_desc_utils.h"
#include "asc_lowerer/asc_overrides.h"
#include "asc_lowerer/kernel_box.h"
#include "utils/auto_fuse_config.h"


namespace ge {
struct LoweringConfig {
  size_t max_loop_ops = 64U;
  size_t max_loop_loads = 4U;
  size_t max_buffer_readers = 4U;
};

enum class PrefixType {
  FusedNumber,   // autofuse_fused_N
  PureNumber,    // autofuse_N
  PureString     // autofuse_字符
};

class FuseNodeNameFormatter {
  public:
    // main process func
    static std::string SimplifyProcess(const std::string& node_name);

  private:
    static std::pair<PrefixType, size_t> ParsePrefix(const std::vector<std::string>& parts);

    static std::string FormatContinuous(const std::vector<std::string>& op_parts);

    static std::string FormatGlobal(const std::vector<std::string>& op_parts);

    static std::string FormatWithConcat(const std::vector<std::string>& op_parts);

    static bool IsConcatOperator(const std::string& op) {
      return (op.length() >= 6) && (op.compare(0, 6, "Concat") == 0);
    }

    static bool IsStrInteger(const std::string& str) {
      if (str.empty()) {
        return false;
      }

      size_t start = 0;
      if ((str[0] == '+') || (str[0] == '-')) {
        if (str.length() == 1) {
          return false;
        }
        start = 1;
      }

      for (size_t i = start; i < str.length(); ++i) {
        if (!std::isdigit(str[i])) {
          return false;
        }
      }
      return true;
    }
};

class LoweringUtils {
 public:
  static void PrintReadableAscGraph(const AscGraph &asc_graph);

  // user core num utils
  static bool IsAnyKernelBoxOversize(std::vector<loop::KernelBox> &kernel_boxes, const LoweringConfig &config);
  
  static bool IsNodeCoreNumDif(const NodePtr &node);
  
  static graphStatus SetAttrCoreNum(OpDescPtr &asc_desc, loop::KernelBox &kernel_box);

  // dfx utils, for reconstruct the original graph for each ascbackend
  static std::string GetConstructDumpGraphName(const NodePtr &node);

  static graphStatus GetOriginToReplaced(const Node *const &node, const ComputeGraphPtr &graph,
                                         std::map<const OutDataAnchor *, OutDataAnchorPtr> &origin_to_replaced);

  // graph utils, for replace ascbackend node
  static graphStatus AddDataEdgesForAscNode(const NodePtr &asc_node, const std::vector<const OutDataAnchor *> &inputs,
                                            ge::OutDataAnchor *origin_output, std::set<const ge::Node *> &used_in_nodes);
  
  static graphStatus GetUnusedInNodes(loop::KernelBox &kernel_box, const std::set<const ge::Node *> &used_in_nodes,
                                      std::set<NodePtr> &unused_in_nodes);
  
  static graphStatus MoveControlEdges(const NodePtr &src, const NodePtr &dst);

  static graphStatus AssembleConcreteEdges(loop::KernelBox &kernel_box, AutoFuseAttrs &fuse_attrs,
                                           const std::vector<const ge::OutDataAnchor *> &origin_inputs);
  
  static graphStatus CheckSpecialFuseType(loop::KernelBox &kernel_box, std::shared_ptr<ge::loop::AscOverrides> &asc_graph);

  static graphStatus SetStreamLabelForOpDesc(loop::KernelBox &kernel_box, OpDescPtr &asc_desc);
};
} // namespace ge

#endif  // AUTOFUSE_LOWERING_LOWERING_UTILS_H_
