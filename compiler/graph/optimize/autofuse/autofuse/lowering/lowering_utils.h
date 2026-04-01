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
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include "graph/node.h"
#include "graph/utils/op_desc_utils.h"
#include "asc_lowerer/asc_overrides.h"
#include "asc_lowerer/kernel_box.h"
#include "utils/auto_fuse_config.h"


namespace ge {
struct LoweringConfig {
  uint64_t max_loop_ops = 64U;
  size_t max_loop_loads = 4U;
  size_t max_buffer_readers = 4U;
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

class GraphFusionReasonStore {
  public:
    enum class FailReasonCategory {
      NODE_INFO_ERROR = 0,
      BACKEND_NOT_SUPPORTED = 1,
      TEMPORARILY_NOT_SUPPORTED = 2
    };

    static void StartProcessGraph(const std::string& graph_name);

    static void AddCurrentGraphNode(const std::string& node_name, const std::string& node_type);

    static void CountNodeFuseFailReason(const std::string& node_name, const std::string& reason, FailReasonCategory category);

    static void ShowGraphFusionFailReasons(const std::string& graph_name);

    static void ClearGraphData(const std::string& graph_name);

  private:
    struct NodeInfo {
      std::string node_type;
      int32_t insert_order;
    };

    struct FailReasonInfo {
      std::string reason;
      FailReasonCategory category;
    };

    struct Storage {
      std::string current_graph_;
      std::vector<std::string> graph_process_order_;
      std::unordered_map<std::string, std::unordered_map<std::string, NodeInfo>> graph_node_info_;
      std::unordered_map<std::string, FailReasonInfo> node_fusion_reason_;
      std::atomic<int32_t> global_node_order_{0};
      std::mutex mutex_;
    };
    static Storage& GetGlobalStorage();

    static const char* GetCategoryName(FailReasonCategory category) {
      switch (category) {
        case FailReasonCategory::NODE_INFO_ERROR: return "Node Info Incorrect";
        case FailReasonCategory::BACKEND_NOT_SUPPORTED: return "Backend Not Support";
        case FailReasonCategory::TEMPORARILY_NOT_SUPPORTED: return "Temporary Skip Lowering";
        default: return "Unknown Reason";
      }
    }
};
} // namespace ge

#endif  // AUTOFUSE_LOWERING_LOWERING_UTILS_H_
