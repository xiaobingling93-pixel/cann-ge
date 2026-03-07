/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/preprocess/multi_batch_copy_graph.h"

#include <queue>
#include <set>
#include <string>

#include "formats/utils/formats_trans_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/passes/multi_batch/multi_batch_clone_pass.h"
#include "graph/passes/multi_batch/subgraph_multi_dims_clone_pass.h"
#include "graph/passes/multi_batch/create_subgraph_with_scope_pass.h"
#include "graph/passes/standard_optimize/prune_pass.h"
#include "graph/preprocess/multi_batch_options.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/passes/pass_manager.h"
#include "common/context/local_context.h"
#include "common/omg_util/omg_util.h"


namespace ge {
namespace multibatch {
namespace {
const char *const kGetNextName = "IteratorV2";
const int32_t kStaticOutput = -1;

inline bool IsGetNextType(const NodePtr &node) {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
                  GELOGW("Get original type failed"); return false);
  return (original_type == kGetNextName);
}

//              +-----------+
//              |   Data    |                      +-----------+       +-----------+       +-----------+
//              +-----------+                      |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
//                       \                      /. +-----------+       +-----------+       +-----------+
//                        \                    /.
// +-----------+       +-----------+          /.   +-----------+       +-----------+       +-----------+
// |   Data    | ----> |    Case   |         S---  |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
// +-----------+       +-----------+          \.   +-----------+       +-----------+       +-----------+
//                               \             \.
//                                \             \. +-----------+       +-----------+       +-----------+
//                           +-----------+         |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
//                           | NetOutput |         +-----------+       +-----------+       +-----------+
//                           +-----------+
// +-----------+                  /
// |   Data    | --------------->/
// +-----------+
void GetDynamicShapeByGraph(const ComputeGraphPtr &graph, const NodePtr &node,
                            std::set<size_t> &dynamic_output_index, std::vector<std::string> &dynamic_output_dims) {
  GELOGD("Try get dynamic shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &func_desc = node->GetOpDesc();
  if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
    GELOGD("Graph: %s Not multi-batch, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
    return;
  }

  const auto &dynamic_branch_names = func_desc->GetSubgraphInstanceNames();
  for (size_t i = 0; i < func_desc->GetOutputsSize(); ++i) {
    for (size_t j = 0; j < dynamic_branch_names.size(); ++j) {
      const auto &subgraph = graph->GetSubgraph(dynamic_branch_names[j]);
      if (subgraph == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Get subgraph:%s from graph:%s failed",
                           dynamic_branch_names[j].c_str(), graph->GetName().c_str());
        GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][SubGraph] %s from graph:%s failed",
               dynamic_branch_names[j].c_str(), graph->GetName().c_str());
        dynamic_output_dims.clear();
        return;
      }

      const auto &out_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
      if (out_node == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "No netoutput node exist in subgraph:%s, check invalid",
                           subgraph->GetName().c_str());
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] No netoutput node exist in subgraph:%s",
               subgraph->GetName().c_str());
        dynamic_output_dims.clear();
        return;
      }

      GELOGI("Find the subgraph Output node %s and the index is %zu", out_node->GetName().c_str(), i);
      const auto &out_desc = out_node->GetOpDesc();
      if (out_desc == nullptr) {
        return;
      }
      if (out_desc->GetInputsSize() <= i) {
        REPORT_INNER_ERR_MSG("E19999",
                           "op_desc of node in subgraph:%s is nullptr or input desc size:%zu <= %zu, check invalid",
                           subgraph->GetName().c_str(), out_desc->GetInputsSize(), i);
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL,
               "[Check][Param] op_desc of node in subgraph:%s is nullptr or input desc size:%zu <= %zu",
               subgraph->GetName().c_str(), out_desc->GetInputsSize(), i);
        dynamic_output_dims.clear();
        return;
      }

      const auto &input_tensor = out_desc->GetInputDesc(i);
      const auto &shape_msg = input_tensor.GetShape().ToString();
      std::string output_shape = std::to_string(j) + "," + std::to_string(i) + "," + shape_msg;
      GELOGI("The shape msg in dynamic batch is %s", output_shape.c_str());
      dynamic_output_dims.emplace_back(output_shape);

      uint32_t parent_index = 0;
      (void)AttrUtils::GetInt(input_tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
      dynamic_output_index.insert(parent_index);
    }
  }
}

// Connect NetOutput directly
void GetDirectOutputShape(const ComputeGraphPtr &graph, const NodePtr &node,
                          const std::set<size_t> &dynamic_output_index, std::vector<std::string> &dynamic_output_dims) {
  if (!GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGD("No need to get directly shape info of %s when train.", node->GetName().c_str());
    return;
  }
  GELOGD("Try get directly shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &netoutput_desc = node->GetOpDesc();
  const auto &inputnode_to_netoutput = node->GetInDataNodes();
  for (size_t i = 0; i < inputnode_to_netoutput.size(); ++i) {
    if (dynamic_output_index.count(i) > 0) {
      continue;
    }

    auto tensor_desc = netoutput_desc->GetInputDesc(i);
    auto shape = tensor_desc.GetShape().ToString();
    std::string static_output_shape = std::to_string(kStaticOutput) + "," + std::to_string(i) + "," + shape;
    GELOGI("The static output shape msg is %s", static_output_shape.c_str());
    dynamic_output_dims.emplace_back(static_output_shape);
  }
}
}  // namespace

Status ProcessMultiBatch(const ComputeGraphPtr &graph, const uint64_t session_id) {
  PassManager pass_manager;
  GE_CHK_STATUS_RET(pass_manager.AddPass("CreateSubGraphWithScopePass",
                                         new (std::nothrow) CreateSubGraphWithScopePass));
  GE_CHK_STATUS_RET(pass_manager.AddPass("SubgraphMultiDimsClonePass",
                                         new (std::nothrow) SubgraphMultiDimsClonePass));
  GE_CHK_STATUS_RET(pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(session_id)));
  const auto ret = pass_manager.Run(graph);
  return ret;
}

Status GetDynamicOutputShape(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGI("Start to get output dynamic batch shape message");

  NodePtr net_output;
  std::set<size_t> dynamic_output_index;
  std::vector<std::string> dynamic_output_dims;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      net_output = node;
    } else if (node->GetType() == CASE) {
      GetDynamicShapeByGraph(graph, node, dynamic_output_index, dynamic_output_dims);
    }
  }

  if ((net_output != nullptr) && !dynamic_output_dims.empty()) {
    GetDirectOutputShape(graph, net_output, dynamic_output_index, dynamic_output_dims);
    if (!AttrUtils::SetListStr(net_output->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_dims)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to node:%s(%s) failed",
                        ATTR_NAME_DYNAMIC_OUTPUT_DIMS.c_str(),
                        net_output->GetName().c_str(), net_output->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to node:%s(%s) failed", ATTR_NAME_DYNAMIC_OUTPUT_DIMS.c_str(),
             net_output->GetName().c_str(), net_output->GetType().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}
}  // namespace multibatch
}  // namespace ge
