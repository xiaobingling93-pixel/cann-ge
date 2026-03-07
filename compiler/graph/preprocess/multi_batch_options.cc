/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/preprocess/multi_batch_options.h"

#include "base/err_msg.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/ge_context.h"
#include "common/context/local_context.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"

namespace ge {
namespace multibatch {
namespace {
constexpr int32_t kDecimal = 10;
constexpr uint8_t kMinShapesCount = 2;
const int32_t kDynmaicDims = -1;
const int32_t kDynamicImgSizeDynamciDimsNum = 2;
const size_t kNumOfGetnextNode = 1;
const int32_t kDivisionConst = 2;
const char *const kSubstrOfGetNextNosinkName = "IteratorGetNext";
const char *const kShapeDataName = "ascend_mbatch_shape_data";
const char *const kGetNextName = "IteratorV2";

inline bool IsGetNextType(const NodePtr &node) {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
                  GELOGW("Get original type failed."); return false);
  return (original_type == kGetNextName);
}

bool ParseDynamicSize(std::string dynamic_size, std::vector<std::vector<int64_t>> &shapes) {
  std::vector<std::string> shape_strs = ge::StringUtils::Split(dynamic_size, ';');
  for (const auto &shape_str : shape_strs) {
    if (shape_str.empty()) {
      continue;
    }
    std::vector<int64_t> shape;
    std::vector<std::string> dims = ge::StringUtils::Split(shape_str, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      int64_t dynamic_dim_shape;
      GE_ASSERT_SUCCESS(ChangeStrToNum(dim, dynamic_dim_shape));
      shape.emplace_back(dynamic_dim_shape);
    }
    if (!shape.empty()) {
      shapes.emplace_back(shape);
    }
  }
  return true;
}

Status DistinguishGetNextAndData(const ComputeGraphPtr &graph, std::vector<NodePtr> &data_nodes,
                                 std::vector<NodePtr> &getnext_nosink_nodes,
                                 std::vector<NodePtr> &getnext_sink_nodes) {
  GELOGD("Start distinguish getnext and data node.");
  for (NodePtr &input_node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op_desc = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType()) && (op_desc->GetName() != kShapeDataName)) {
      if (op_desc->GetName().find(kSubstrOfGetNextNosinkName) == std::string::npos) {
        data_nodes.emplace_back(input_node);
        GELOGD("Name of data node is %s.", op_desc->GetName().c_str());
      } else {
        getnext_nosink_nodes.emplace_back(input_node);
        GELOGD("Name of getnext nosink is %s.", op_desc->GetName().c_str());
      }
    }
    if (IsGetNextType(input_node)) {
      GELOGD("Name of getnext sink is %s.", op_desc->GetName().c_str());
      getnext_sink_nodes.emplace_back(input_node);
    }
  }
  GELOGI("Data count is %zu, getnext nosink count is %zu, getnext sink count is %zu.", data_nodes.size(),
         getnext_nosink_nodes.size(), getnext_sink_nodes.size());
  SortDataNodesByName(data_nodes);
  SortDataNodesByName(getnext_nosink_nodes);
  GetLocalOmgContext().data_nodes = data_nodes;
  GetLocalOmgContext().getnext_nosink_nodes = getnext_nosink_nodes;
  return SUCCESS;
}

Status CheckNodeSize(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes) {
  if (data_nodes.size() != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Count:%zu of data_nodes in graph:%s should be equal to "
                       "input_shape count:%zu from option, check invalid",
                       data_nodes.size(), graph->GetName().c_str(), GetLocalOmgContext().user_input_dims.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Count:%zu of data_nodes in graph:%s should be equal to "
           "input_shape count:%zu from option",
           data_nodes.size(), graph->GetName().c_str(), GetLocalOmgContext().user_input_dims.size());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status CheckSequenceOfData(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes) {
  GELOGD("Start check input sequence from data nodes and input shape.");
  GE_ASSERT_SUCCESS(CheckNodeSize(graph, data_nodes));
  for (size_t i = 0; i < data_nodes.size(); ++i) {
    auto data_node = data_nodes.at(i);
    GE_CHECK_NOTNULL(data_node);
    GE_CHECK_NOTNULL(data_node->GetOpDesc());
    auto output_shape = data_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
    auto dynamic_dims = GetLocalOmgContext().user_input_dims.at(i).second;
    GELOGD("The %zu data node is %s, node shape is %s, dynamic dim is %s.", i, data_node->GetName().c_str(),
           ToString(output_shape).c_str(), ToString(dynamic_dims).c_str());
    if (output_shape.empty() && dynamic_dims.size() == 1 && dynamic_dims.at(0) == 0) {
      GELOGI("No need to check sequence for constant.");
      continue;
    }
    if (dynamic_dims.size() != output_shape.size()) {
      REPORT_INNER_ERR_MSG("E19999", "The output shape of %s is %s, the input shape from options of %s is %s, graph:%s,"
                         "check invalid", data_node->GetName().c_str(),
                         ToString(output_shape).c_str(),
                         GetLocalOmgContext().user_input_dims.at(i).first.c_str(),
                         ToString(dynamic_dims).c_str(), graph->GetName().c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] The output shape of %s is %s, "
             "the input shape from options of %s is %s, graph:%s",
             data_node->GetName().c_str(), ToString(output_shape).c_str(),
             GetLocalOmgContext().user_input_dims.at(i).first.c_str(),
             ToString(dynamic_dims).c_str(), graph->GetName().c_str());
      return PARAM_INVALID;
    }
    for (size_t j = 0; j < dynamic_dims.size(); ++j) {
      if (dynamic_dims.at(j) != kDynmaicDims && dynamic_dims.at(j) != output_shape.at(j)) {
        REPORT_INNER_ERR_MSG("E19999", "Value of input shape %s from option and output shape %s of data op:%s "
                           "should be equal to %d, index:%zu, graph:%s, check invalid",
                           ToString(dynamic_dims).c_str(),
                           ToString(output_shape).c_str(), data_node->GetName().c_str(), kDynmaicDims,
                           j, graph->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] Value of input shape %s from option and output shape %s of data op:%s "
               "should be equal to %d, index:%zu, graph:%s",
               ToString(dynamic_dims).c_str(), ToString(output_shape).c_str(),
               data_node->GetName().c_str(), kDynmaicDims, j, graph->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}

Status CheckSequenceOfGetnext(const ComputeGraphPtr &graph, const std::vector<NodePtr> &getnext_sink_node) {
  GELOGD("Start check input sequence from getnext sink nodes and input shape.");
  if (getnext_sink_node.size() != kNumOfGetnextNode) {
    REPORT_INNER_ERR_MSG("E19999", "Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
                       "num of getnext node:%zu, check invalid",
                       graph->GetName().c_str(), getnext_sink_node.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
           "num of getnext node:%zu", graph->GetName().c_str(), getnext_sink_node.size());
    return PARAM_INVALID;
  }
  auto data_node = getnext_sink_node.at(0);
  GE_CHECK_NOTNULL(data_node);
  auto op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  size_t data_count = data_node->GetAllOutDataAnchors().size() / kDivisionConst;
  if (data_count != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Output desc count of %s is %zu, should be equal to count of input shape:%zu, "
                       "graph:%s, check invalid", op_desc->GetName().c_str(), data_count,
                       GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Output desc count of %s is %zu, "
           "should be equal to count of input shape:%zu, graph:%s", op_desc->GetName().c_str(),
           data_count, GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    return PARAM_INVALID;
  }
  for (size_t i = 0; i < data_count; ++i) {
    auto output_shape = data_node->GetOpDesc()->GetOutputDesc(i).GetShape().GetDims();
    auto dynamic_dims = GetLocalOmgContext().user_input_dims.at(i).second;
    GELOGD("The %zu getnext node is %s, node shape is %s, dynamic dim is %s.", i, data_node->GetName().c_str(),
           ToString(output_shape).c_str(), ToString(dynamic_dims).c_str());
    if (output_shape.empty() && dynamic_dims.size() == 1 && dynamic_dims.at(0) == 0) {
      GELOGI("No need to check sequence for constant.");
      continue;
    }
    if (dynamic_dims.size() != output_shape.size()) {
      REPORT_INNER_ERR_MSG("E19999", "The %zu output_shape of %s is %s not equal to the input_shape:%s "
                         "from options of %s, graph:%s, check invalid", i,
                         data_node->GetName().c_str(), ToString(output_shape).c_str(),
                         ToString(dynamic_dims).c_str(),
                         GetLocalOmgContext().user_input_dims.at(i).first.c_str(),
                         graph->GetName().c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] The %zu output_shape of %s is %s not equal to the input_shape:%s "
             "from options of %s, graph:%s", i, data_node->GetName().c_str(),
             ToString(output_shape).c_str(), ToString(dynamic_dims).c_str(),
             GetLocalOmgContext().user_input_dims.at(i).first.c_str(), graph->GetName().c_str());
      return PARAM_INVALID;
    }
    for (size_t j = 0; j < dynamic_dims.size(); ++j) {
      if (dynamic_dims.at(j) != kDynmaicDims && dynamic_dims.at(j) != output_shape.at(j)) {
        REPORT_INNER_ERR_MSG("E19999", "Value of input shape %s from option and output shape %s of data op:%s "
                           "should be equal to %d, index:%zu, graph:%s, check invalid",
                           ToString(dynamic_dims).c_str(),
                           ToString(output_shape).c_str(), data_node->GetName().c_str(), kDynmaicDims,
                           j, graph->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] Value of input shape %s from option and output shape %s of data op:%s "
               "should be equal to %d, index:%zu, graph:%s", ToString(dynamic_dims).c_str(),
               ToString(output_shape).c_str(), data_node->GetName().c_str(), kDynmaicDims,
               j, graph->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}

Status UpdateNameOfGetnext(const ComputeGraphPtr &graph, const std::vector<NodePtr> &getnext_sink_nodes) {
  GELOGD("Update first value of input shape by getnext sink nodes.");
  if (getnext_sink_nodes.size() != kNumOfGetnextNode) {
    REPORT_INNER_ERR_MSG("E19999", "Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
                                 "num of getnext node:%zu, check invalid",
                       graph->GetName().c_str(), getnext_sink_nodes.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
                          "num of getnext node:%zu", graph->GetName().c_str(), getnext_sink_nodes.size());
    return PARAM_INVALID;
  }
  auto input_node = getnext_sink_nodes.at(0);
  GE_CHECK_NOTNULL(input_node);
  auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // user want getnext dynamic, just getnext or data+getnext_sink
  size_t data_count = input_node->GetAllOutDataAnchors().size() / kDivisionConst;
  if (data_count != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Output desc count of %s is %zu, should be equal to count of input shape:%zu, "
                                 "graph:%s, check invalid", op_desc->GetName().c_str(), data_count,
                       GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Output desc count of %s is %zu, "
                          "should be equal to count of input shape:%zu, graph:%s", op_desc->GetName().c_str(), data_count,
           GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < data_count; ++i) {
    std::string data_name = op_desc->GetName() + "_" + std::to_string(i);
    GELOGD("Data just from getnext sink is %s.", data_name.c_str());
    GetLocalOmgContext().user_input_dims.at(i).first = data_name;
  }
  return SUCCESS;
}

std::vector<std::string> SplitInputShape(const std::string &input_shape) {
  std::vector<std::string> shape_pair_vec;
  size_t pos = input_shape.rfind(":");
  if (pos != std::string::npos) {
    shape_pair_vec.emplace_back(input_shape.substr(0, pos));
    shape_pair_vec.emplace_back(input_shape.substr(pos + 1, input_shape.size() - pos));
  }
  return shape_pair_vec;
}
} // namespace

Status CheckSequenceOfOptions(const ComputeGraphPtr &graph, std::vector<NodePtr> &data_nodes,
                              std::vector<NodePtr> &getnext_nosink_nodes,
                              std::vector<NodePtr> &getnext_sink_nodes,
                              bool &need_multi_batch) {
  if (GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGI("No need to CheckSequenceOfOptions.");
    return SUCCESS;
  }

  if (DistinguishGetNextAndData(graph, data_nodes, getnext_nosink_nodes,
                                getnext_sink_nodes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Call][DistinguishGetNextAndData] failed.");
    return PARAM_INVALID;
  }

  if (GetLocalOmgContext().dynamic_node_type == DATA) {
    GELOGD("Users want data nodes to be dynamic.");
    if (CheckSequenceOfData(graph, data_nodes) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Check][Sequence] Of Data nodes failed.");
      return PARAM_INVALID;
    }
  } else {
    GELOGD("Users want getnext nodes to be dynamic.");
    if ((getnext_nosink_nodes.empty()) && (getnext_sink_nodes.empty())) {
      need_multi_batch = false;
      GELOGI("No need multi batch when graph has no GetNext, graph name:%s.", graph->GetName().c_str());
      return SUCCESS;
    }
    if (!getnext_nosink_nodes.empty()) {
      if (CheckSequenceOfData(graph, getnext_nosink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Check][Sequence] of getnext nosink nodes failed.");
        return PARAM_INVALID;
      }
    } else {
      if (CheckSequenceOfGetnext(graph, getnext_sink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Check][Sequence] of getnext sink nodes failed.");
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status UpdateDataShapeByUserInput() {
  if (GetLocalOmgContext().dynamic_node_type == DATA) {
    return UpdateDataShape(GetLocalOmgContext().data_nodes);
  }
  if ((GetLocalOmgContext().dynamic_node_type == GETNEXT) &&
      (!GetLocalOmgContext().getnext_nosink_nodes.empty())) {
    return UpdateDataShape(GetLocalOmgContext().getnext_nosink_nodes);
  }
  return SUCCESS;
}

Status UpdateDataShape(const std::vector<NodePtr> &data_nodes) {
  auto data_name_and_shape = GetLocalOmgContext().user_input_dims;
  GE_ASSERT_TRUE(data_nodes.size() == data_name_and_shape.size());
  for (size_t i = 0UL; i < data_nodes.size(); i++) {
    auto op_desc = data_nodes[i]->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const auto data_shape = data_name_and_shape[i].second;
    auto input_tensor = op_desc->MutableInputDesc(0U);
    GE_ASSERT_NOTNULL(input_tensor);
    input_tensor->SetShape(GeShape(data_shape));
    auto output_tensor = op_desc->MutableOutputDesc(0U);
    GE_ASSERT_NOTNULL(output_tensor);
    output_tensor->SetShape(GeShape(data_shape));
  }
  return SUCCESS;
}

std::vector<int32_t> GetDataNodesUnknownDimIndex(const ge::NodePtr &data_nodes) {
  const auto op_desc = data_nodes->GetOpDesc();
  if (op_desc == nullptr) {
    return {};
  }
  const auto shape = op_desc->GetOutputDesc(0UL).GetShape().GetDims();
  std::vector<int32_t> unknown_shape_index;
  for (size_t i = 0UL; i < shape.size(); i++) {
    if (shape[i] == UNKNOWN_DIM) {
      unknown_shape_index.emplace_back(static_cast<int32_t>(i));
    }
  }
  return unknown_shape_index;
}

void SortDataNodesByIndex(std::vector<NodePtr> &data_nodes) {
  auto cmp_func = [](const NodePtr &node1, const NodePtr &node2) -> bool {
                    int64_t data_index_node1 = 0LL;
                    GE_ASSERT_TRUE(AttrUtils::GetInt(node1->GetOpDesc(), ATTR_NAME_INDEX, data_index_node1));
                    int64_t data_index_node2 = 0LL;
                    GE_ASSERT_TRUE(AttrUtils::GetInt(node2->GetOpDesc(), ATTR_NAME_INDEX, data_index_node2));
                    return (data_index_node1 < data_index_node2);
                  };
  (void)std::sort(data_nodes.begin(), data_nodes.end(), cmp_func);
}

void SortDataNodesByName(std::vector<NodePtr> &data_nodes) {
  auto cmp_func = [](const NodePtr &node1, const NodePtr &node2) -> bool {
                  return (node1->GetName() < node2->GetName());
                  };
  (void)std::sort(data_nodes.begin(), data_nodes.end(), cmp_func);
}

Status UpdateNameOfData(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes) {
  GELOGD("Update first value of input shape by data nodes.");
  GE_ASSERT_SUCCESS(CheckNodeSize(graph, data_nodes));
  for (size_t i = 0; i < data_nodes.size(); ++i) {
    GELOGD("The %zu data name is %s.", i, data_nodes.at(i)->GetOpDesc()->GetName().c_str());
    GetLocalOmgContext().user_input_dims.at(i).first = data_nodes.at(i)->GetOpDesc()->GetName();
  }
  return SUCCESS;
}

// need to distinguish online and offline, offline no need to update the name of input_shape
Status UpdateNameOfInputShape(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes,
                              const std::vector<NodePtr> &getnext_nosink_nodes,
                              const std::vector<NodePtr> &getnext_sink_nodes) {
  if (GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGI("No need to update first value of input shape when offline infer.");
    return SUCCESS;
  }

  if (GetLocalOmgContext().dynamic_node_type == DATA) {
    GELOGD("Users want data nodes to be dynamic.");
    if (UpdateNameOfData(graph, data_nodes) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Call][UpdateNameOfData] update first value of input shape of data nodes failed.");
      return PARAM_INVALID;
    }
  } else {
    GELOGD("Users want getnext nodes to be dynamic.");
    if (!getnext_nosink_nodes.empty()) {
      if (UpdateNameOfData(graph, getnext_nosink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID,
               "[Call][UpdateNameOfData] update first value of input shape of getnext nosink nodes failed.");
        return PARAM_INVALID;
      }
    } else {
      if (UpdateNameOfGetnext(graph, getnext_sink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID,
               "[Call][UpdateNameOfGetnext] update first value of input shape of getnext sink nodes failed.");
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status DeleteIdentityInsertByAdapter(const ComputeGraphPtr &graph) {
  GELOGD("Start delete identity node inserted by adapter.");
  for (NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (IsGetNextType(node)) {
      for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
        GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
        for (auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
          GE_IF_BOOL_EXEC(peer_in_anchor == nullptr, continue);
          auto dst_node = peer_in_anchor->GetOwnerNode();
          GE_IF_BOOL_EXEC(dst_node == nullptr, continue);
          if (dst_node->GetType() == IDENTITY && dst_node->GetOutDataNodes().empty()) {
            GELOGI("Need to remove %s.", dst_node->GetName().c_str());
            if (GraphUtils::RemoveNodeWithoutRelink(graph, dst_node) != GRAPH_SUCCESS) {
              REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) from graph:%s failed",
                                dst_node->GetName().c_str(), dst_node->GetType().c_str(), graph->GetName().c_str());
              GELOGE(FAILED, "[Remove][Node] %s(%s) from graph:%s failed",
                     dst_node->GetName().c_str(), dst_node->GetType().c_str(), graph->GetName().c_str());
              return FAILED;
            }
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status CheckNegativeCountOfOptions(const std::vector<std::vector<int64_t>> &shapes) {
  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    size_t negative_count = 0;
    for (size_t i = 0; i < GetLocalOmgContext().user_input_dims.size(); ++i) {
      for (size_t j = 0; j < GetLocalOmgContext().user_input_dims.at(i).second.size(); ++j) {
        if (GetLocalOmgContext().user_input_dims.at(i).second.at(j) == kDynmaicDims) {
          negative_count++;
        }
      }
    }
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (shapes.at(i).size() != negative_count) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
            std::vector<const char *>({"dynamic_dims's per dims count", std::to_string(shapes.at(i).size()).c_str(),
                                       "dynamic_dims's per dims count should be equal to input_shape's dim size."}));
        GELOGE(PARAM_INVALID, "[Check][Param] gear num of dynamic_dims is %zu should be equal to num:%zu from option",
               shapes.at(i).size(), negative_count);
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}
Status ChangeStrToNum(const std::string &str, int64_t &num) {
  for (const auto &a : str) {
    GE_ASSERT_TRUE(isdigit(a));
  }
  num = std::strtol(str.c_str(), nullptr, kDecimal);
  return SUCCESS;
}
///
/// @ingroup ge
/// @brief Init Dynamic Param from Options.
/// @param [out] std::vector<std::vector<int64_t>> &shapes: Result for Params.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
Status InitDynamicParams(std::vector<std::vector<int64_t>> &shapes) {
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    GELOGD("Found dynamic batch option, value %s", GetLocalOmgContext().dynamic_batch_size.c_str());
    std::vector<std::string> dims = ge::StringUtils::Split(GetLocalOmgContext().dynamic_batch_size, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      int64_t dynamic_batch_shape;
      GE_ASSERT_SUCCESS(ChangeStrToNum(dim, dynamic_batch_shape),
          "Option dynamic_batch_size[%s] should not have non-digital character",
          GetLocalOmgContext().dynamic_batch_size.c_str());
      shapes.emplace_back(std::vector<int64_t>({dynamic_batch_shape}));
      GELOGI("Found dynamic batch, shape %s", ToString(*shapes.rbegin()).c_str());
    }
  }

  if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    GELOGD("Found dynamic image size option, value %s", GetLocalOmgContext().dynamic_image_size.c_str());
    GE_ASSERT_TRUE(ParseDynamicSize(GetLocalOmgContext().dynamic_image_size, shapes),
        "Option dynamic_batch_size[%s] should not have non-digital character",
        GetLocalOmgContext().dynamic_image_size.c_str());
    for (const auto &shape : shapes) {
      GELOGI("Found dynamic image size, shape %s", ToString(shape).c_str());
    }
  }

  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    GELOGD("Found dynamic dims option, value %s", GetLocalOmgContext().dynamic_dims.c_str());
    GE_ASSERT_TRUE(ParseDynamicSize(GetLocalOmgContext().dynamic_dims, shapes),
        "Option dynamic_dims[%s] should not have non-digital character",
        GetLocalOmgContext().dynamic_dims.c_str());
    for (const auto &shape : shapes) {
      GELOGI("Found dynamic dims, shape %s", ToString(shape).c_str());
    }
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief parse each data's own dynamic dims.
/// @param [out] std::map<std::string, std::vector<std::vector<int64_t>>> &data_to_dynamic_info: key:data_name.
///              value:dynamic dims.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
Status ParserDataToDynamicInfo(const std::vector<std::vector<int64_t>> &shapes,
                               std::vector<std::pair<std::string, std::vector<int64_t>>> &data_name_and_shape,
                               std::map<std::string, std::vector<std::vector<int64_t>> > &data_to_dynamic_info) {
  size_t cur_data_index = 0;
  for (size_t index = 0; index < data_name_and_shape.size(); ++index) {
    auto &cur_item = data_name_and_shape[index];
    auto &data_name = cur_item.first;
    auto &data_shape = cur_item.second;
    auto dynamic_dims_num = std::count_if(data_shape.begin(), data_shape.end(),
                                          [&data_shape](int64_t dim){ return dim < 0; });
    GELOGI("Train_Dynamic dynamic_dims_num of %s is %zu", data_name.c_str(), dynamic_dims_num);
    std::vector<std::vector<int64_t> > dynamic_info;
    for (auto &dynamic_gear_info : shapes) {
      GELOGI("Train_Dynamic dynamic_gear_info is %s", ToString(dynamic_gear_info).c_str());
      std::vector<int64_t> one_gear;
      if (dynamic_gear_info.size() == static_cast<size_t>(dynamic_dims_num)) {
        one_gear = dynamic_gear_info;
      } else if (dynamic_gear_info.size() > static_cast<size_t>(dynamic_dims_num)) {
        auto tmp_index = cur_data_index;
        for (size_t i = 0; i < static_cast<size_t>(dynamic_dims_num); ++i) {
          if (tmp_index >= dynamic_gear_info.size()) {
            REPORT_PREDEFINED_ERR_MSG("E10045", std::vector<const char *>({"name", "shape"}),
                                      std::vector<const char *>({data_name.c_str(), ToString(data_shape).c_str()}));
            GELOGE(PARAM_INVALID, "[Check][Param] Data:%s shape:%s make dynamic dims overflow", data_name.c_str(),
                   ToString(data_shape).c_str());
            return FAILED;
          }
          one_gear.push_back(dynamic_gear_info[tmp_index++]);
        }
      } else {
        REPORT_PREDEFINED_ERR_MSG("E10046", std::vector<const char *>({"name", "shape"}),
                                  std::vector<const char *>({data_name.c_str(), ToString(data_shape).c_str()}));
        GELOGE(PARAM_INVALID, "[Check][Param] Dynamic dims num of data: %s shape: %s "
               "can not be more than one gear dynamic info size",
               data_name.c_str(), ToString(data_shape).c_str());
        return FAILED;
      }
      GELOGI("Train_Dynamic one_gear is %s.", ToString(one_gear).c_str());
      dynamic_info.push_back(one_gear);
    }
    cur_data_index += dynamic_dims_num;
    data_to_dynamic_info[data_name] = dynamic_info;
  }
  return SUCCESS;
}


///
/// @ingroup ge
/// @brief Check Dynamic Param is invalid.
/// @param [in] const std::vector<std::vector<int64_t>> &shapes: Params for check.
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
///
Status CheckDynamicParams(const std::vector<std::vector<int64_t>> &shapes) {
  if (shapes.size() < kMinShapesCount) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10035", std::vector<const char *>({"shapesize", "minshapesize"}),
        std::vector<const char *>({std::to_string(shapes.size()).c_str(), std::to_string(kMinShapesCount).c_str()}));
    GELOGE(PARAM_INVALID,
           "[Check][Param] Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
           "value size [%zu] should not less than [%d].",
           shapes.size(), kMinShapesCount);
    return PARAM_INVALID;
  }
  std::set<std::vector<int64_t>> shapes_set;
  size_t shape_size = shapes.at(0).size();
  for (auto &shape : shapes) {
    if (shape_size != shape.size()) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10037", std::vector<const char *>({"shapesize1", "shapesize2"}),
          std::vector<const char *>({std::to_string(shape_size).c_str(), std::to_string(shape.size()).c_str()}));
      GELOGE(PARAM_INVALID,
             "[Check][Param] Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
             "value size must be same, first group's size is %zu and another's is %zu.",
             shape_size, shape.size());
      return PARAM_INVALID;
    }
    for (auto dim : shape) {
      if (dim < 0) {
        REPORT_PREDEFINED_ERR_MSG("E10038", std::vector<const char *>({"dim"}), std::vector<const char *>({std::to_string(dim).c_str()}));
        GELOGE(PARAM_INVALID, "[Check][Param] Invalid dim %ld, all dims must not be less than 0", dim);
        return PARAM_INVALID;
      }
    }
    shapes_set.insert(shape);
  }
  if (shapes_set.size() != shapes.size()) {
    REPORT_PREDEFINED_ERR_MSG("E10039", std::vector<const char *>({}), std::vector<const char *>({}));
    GELOGE(PARAM_INVALID, "[Check][Param] Input parameter[--dynamic_batch_size, "
           "--dynamic_image_size or --dynamic_dims] exist duplicate shapes.");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get GeShape from configed shape.
/// @param [in] const std::vector<int64_t> &batch_shape: Configed shape.
/// @param [out] GeShape &data_shape: GeShape for configed shape.
/// @return SUCCESS / PARAM_INVALID
///
Status CalcShape(const std::vector<int64_t> &batch_shape, GeShape &data_shape) {
  size_t batch_shape_index = 0;
  for (size_t i = 0; i < data_shape.GetDimNum(); ++i) {
    if (data_shape.GetDim(i) < 0) {
      if (batch_shape_index >= batch_shape.size()) {
        REPORT_INNER_ERR_MSG("E19999", "the batch shape count %zu, does not match the data shape %s",
                           batch_shape.size(), data_shape.ToString().c_str());
        GELOGE(PARAM_INVALID, "[Check][Param] Failed to calc tensor shape, the batch shape count %zu, "
               "does not match the data shape %s", batch_shape.size(), data_shape.ToString().c_str());
        return PARAM_INVALID;
      }
      data_shape.SetDim(i, batch_shape[batch_shape_index++]);
    }
  }
  GELOGI("CalcShape size of batch_shape is %zu, batch_shape_index is %zu.", batch_shape.size(), batch_shape_index);
  if (batch_shape_index != batch_shape.size()) {
    REPORT_INNER_ERR_MSG("E19999", "the batch shape count %zu, does not match the data shape %s",
                       batch_shape.size(), data_shape.ToString().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Failed to calc tensor shape, the batch shape count %zu, "
           "does not match the data shape %s", batch_shape.size(), data_shape.ToString().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set mbatch_dynamic_type on node.
/// @param [in] const OpDescPtr &op_desc: Node for set attribute.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
///
Status StampDynamicType(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  int32_t dynamic_type = static_cast<int32_t>(FIXED);
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_BATCH);
  }
  if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_IMAGE);
  }
  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_DIMS);
  }
  if (!AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_DYNAMIC_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
           ATTR_DYNAMIC_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Check dynamic batch Shape.
/// @param [in] const std::vector<int64_t> &shape: data_shape to be checked.
/// @param [in] const std::string &data_name: cur data name.
/// @return 0: true/false
///
bool CheckDynamicBatchShape(const std::vector<int64_t> &shape, const std::string &data_name) {
  if (shape[0] == kDynmaicDims) {
    for (size_t i = 1; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10018", std::vector<const char *>({"index", "shape"}),
            std::vector<const char *>({std::to_string(i).c_str(), std::to_string(shape[i]).c_str()}));
        GELOGE(ge::PARAM_INVALID, "[Check][Param] Only batch N can be -1 when set --dynamic_batch_size, "
               "current data: %s shape[%zu] is %ld", data_name.c_str(), i, shape[i]);
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

///
/// @ingroup ge
/// @brief Check Dynamic image size shape.
/// @param [in] unordered_map<std::string, std::vector<int64_t>> &shape_map: map of data_name and data_shape.
/// @param [in]  const std::string &input_format: format of input.
/// @return 0: true/false
///
bool CheckDynamicImageSizeShape(const std::vector<int64_t> &shape, const std::string &input_format) {
  int64_t height = 0;
  int64_t width = 0;
  if (input_format == "NCHW") {
    height = shape[NCHW_DIM_H];
    width = shape[NCHW_DIM_W];
  } else if (input_format == "NHWC") {
    height = shape[NHWC_DIM_H];
    width = shape[NHWC_DIM_W];
  } else {
    std::string reason = "When --dynamic_image_size is set, the input format only supports NCHW or NHWC.";
    REPORT_PREDEFINED_ERR_MSG("E10003", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"input_format", input_format.c_str(), reason.c_str()}));
    GELOGE(ge::PARAM_INVALID, "when set --dynamic_image_size, input format only support NCHW and NHWC.");
  }

  if (height == kDynmaicDims && width == kDynmaicDims &&
      std::count(shape.begin(), shape.end(), kDynmaicDims) == kDynamicImgSizeDynamciDimsNum) {
    return true;
  } else {
    REPORT_PREDEFINED_ERR_MSG("E10019", std::vector<const char *>({}), std::vector<const char *>({}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param] --input_shape's shape is invalid, only height and width can be -1 "
           "when set --dynamic_image_size.");
    return false;
  }
}

Status ParseInputShapes(const std::string &input_shapes,
                        std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map) {
  std::vector<std::string> shape_vec = ge::StringUtils::Split(input_shapes, ';');
  const int32_t kDefaultShapePairSize = 2;
  for (const auto &shape : shape_vec) {
    std::vector<std::string> shape_pair_vec = SplitInputShape(shape);
    if (shape_pair_vec.size() != kDefaultShapePairSize) {
      GELOGE(INTERNAL_ERROR, "shape[%s] after split by \":\" must contains two parts: name and value", shape.c_str());
      return INTERNAL_ERROR;
    }

    if (shape_pair_vec[1].empty()) {
      GELOGE(INTERNAL_ERROR, "The shape [%s] has a name, it's value can not be empty", shape.c_str());
      return INTERNAL_ERROR;
    }

    std::vector<std::string> shape_value_strs = ge::StringUtils::Split(shape_pair_vec[1], ',');
    std::vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      if (shape_value_str.find('.') != std::string::npos) {
        GELOGE(INTERNAL_ERROR, "unsupport float config value.");
        return INTERNAL_ERROR;
      }
      int64_t result = 0;
      try {
        result = std::stol(StringUtils::Trim(shape_value_str));
      } catch (const std::exception &e) {
        GELOGE(PARAM_INVALID, "Convert %s to int64_t value failed:%s", shape_value_str.c_str(), e.what());
        return INTERNAL_ERROR;
      }
      shape_values.push_back(static_cast<int64_t>(result));
    }
    user_shape_map.push_back(std::make_pair(ge::StringUtils::Trim(shape_pair_vec[0]), shape_values));
  }
  return SUCCESS;
}

Status BuildSubgraphMuliDimsInput(const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                                  const DimsVector &dynamic_dims_vec,
                                  std::vector<std::string> &subgraph_multi_dims_input_shape,
                                  std::vector<std::string> &subgraph_multi_dims_input_dims) {
  const size_t nodes_num = user_shape_map.size();  // e.g. inputshape:  [{"data0", [-1,3]}, {"data1", [-1,4]}]
  size_t count = 0U;
  const size_t dynamic_count = dynamic_dims_vec.size();  // e.g. dims： [["3","3"],["4","4"]]
  for (size_t i = 0U; i < nodes_num; ++i) {
    std::vector<std::string> tmp(dynamic_count);
    auto &nodes_shape = user_shape_map[i].second;
    for (auto &dim : nodes_shape) {
      if (dim != -1) {
        continue;
      }
      for (size_t j = 0U; j < dynamic_count; ++j) {
        (void)tmp[j].append(dynamic_dims_vec[j][count]).append(",");
      }
      ++count;
    }
    std::string tmp_dims;
    for (size_t j = 0U; j < dynamic_count; ++j) {
      if (tmp[j].empty()) {
        GELOGI("input_shapes: %zu matched dims is empty", i);
        tmp_dims.clear();
        break;
      }
      (void)tmp_dims.append(tmp[j].substr(0, tmp[j].size() - 1)).append(";");
    }
    std::string tmp_shape;
    for (size_t j = 0U; (j < nodes_shape.size()) && (!tmp_dims.empty()); ++j) {
      (void)tmp_shape.append(std::to_string(nodes_shape[j])).append(",");
    }
    subgraph_multi_dims_input_dims.push_back(tmp_dims.substr(0, tmp_dims.size() - 1));
    subgraph_multi_dims_input_shape.push_back(tmp_shape.substr(0, tmp_shape.size() - 1));
    GELOGI("index: %zu subgraph_multi_dims_input_dims is: %s.", i, subgraph_multi_dims_input_dims[i].c_str());
    GELOGI("index: %zu subgraph_multi_dims_input_shape is: %s.", i, subgraph_multi_dims_input_shape[i].c_str());
  }
  return SUCCESS;
}

Status ParseDynamicShapesAndDims(const std::string &input_shapes, const std::string &dynamic_dims,
                                 std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                                 DimsVector &dynamic_dims_vec,
                                 std::vector<std::pair<std::string, std::vector<int64_t>>> &max_shape_range_map) {
  GELOGI("input_shapes: %s, dynamic_dims: %s", input_shapes.c_str(), dynamic_dims.c_str());
  GE_RETURN_IF_ERROR(ParseDynamicShapes(input_shapes, user_shape_map));
  std::vector<std::vector<int64_t>> dynamic_dims_digit_vec;
  GE_RETURN_IF_ERROR(ParseDynamicDims(dynamic_dims, dynamic_dims_vec, dynamic_dims_digit_vec, user_shape_map));
  GE_RETURN_IF_ERROR(ParseMaxShapeRange(user_shape_map, dynamic_dims_digit_vec, max_shape_range_map));
  return SUCCESS;
}

Status ParseMaxShapeRange(const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                          const std::vector<std::vector<int64_t>> &dynamic_dims_digit_vec,
                          std::vector<std::pair<std::string, std::vector<int64_t>>> &max_shape_range_map) {
  size_t num = dynamic_dims_digit_vec[0].size();
  std::vector<int64_t> tmp(num, 0);
  for (auto &digit_vec : dynamic_dims_digit_vec) {
    for (size_t i = 0U; i < num; ++i) {
      tmp[i] = std::max(tmp[i], digit_vec[i]);
    }
  }

  size_t count = 0U;
  max_shape_range_map = user_shape_map;
  for (auto &shape_range : max_shape_range_map) {
    std::vector<int64_t> &shapes = shape_range.second;
    for (size_t i = 0U; i < shapes.size(); ++i) {
      if (shapes[i] == -1) {
        shapes[i] = tmp[count++];
      }
    }
  }
  return SUCCESS;
}
Status ParseDynamicDims(const std::string &dynamic_dims, DimsVector &dynamic_dims_vec,
                        std::vector<std::vector<int64_t>> &dynamic_dims_digit_vec,
                        const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map) {
  int64_t dynamic_dim_num = 0;
  for (auto &info_shapes : user_shape_map) {
    auto &shapes = info_shapes.second;
    dynamic_dim_num += std::count(shapes.begin(), shapes.end(), -1);
  }
  GELOGI("dynamic dim num: %ld.", dynamic_dim_num);
  if (dynamic_dims.empty()) {
    GELOGE(INTERNAL_ERROR, "dynamic_dims can not be empty.");
    return INTERNAL_ERROR;
  }
  // Different parameter sets are split by ';'
  std::vector<std::string> split_set = ge::StringUtils::Split(dynamic_dims, ';');
  for (auto split_dim : split_set) {
    std::vector<std::string> one_dim_set = ge::StringUtils::Split(split_dim, ',');
    GE_CHK_BOOL_RET_STATUS(
        one_dim_set.size() == static_cast<size_t>(dynamic_dim_num), FAILED,
        "dynamic_dims: %s invalid. reason: Each gear setting needs to be consistent with the number of -1 in the "
        "inputshape.",
        dynamic_dims.c_str());

    std::vector<int64_t> digit_vec;
    for (auto dim : one_dim_set) {
      for (auto c : dim) {
        GE_CHK_BOOL_RET_STATUS(isdigit(c), FAILED, "dynamic_dims: %s parameter must be positive integer.",
                               dynamic_dims.c_str());
        constexpr int32_t decimal = 10;
        digit_vec.push_back(std::strtol(dim.c_str(), nullptr, decimal));
      }
    }
    dynamic_dims_vec.push_back(one_dim_set);
    dynamic_dims_digit_vec.push_back(digit_vec);
  }
  return SUCCESS;
}

Status ParseDynamicShapes(const std::string &input_shapes,
                          std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map) {
  std::vector<std::string> shape_vec = ge::StringUtils::Split(input_shapes, ';');
  const int32_t kDefaultShapePairSize = 2;
  std::vector<std::string> input_shape_names;
  for (const auto &shape : shape_vec) {
    std::vector<std::string> shape_pair_vec = SplitInputShape(shape);
    GE_CHK_BOOL_RET_STATUS(shape_pair_vec.size() == kDefaultShapePairSize, FAILED, "parse input_shape failed.");
    GE_CHK_BOOL_RET_STATUS(!shape_pair_vec[1].empty(), FAILED, "parse input_shape failed.");
    input_shape_names.emplace_back(shape_pair_vec[0]);

    std::vector<std::string> shape_value_strs = ge::StringUtils::Split(shape_pair_vec[1], ',');
    std::vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      // stoul: The method may throw an exception: invalid_argument/out_of_range
      GE_CHK_BOOL_RET_STATUS(shape_value_str.find('.') == std::string::npos, FAILED, "unsupport float config value.");
      int64_t result = std::strtol(shape_value_str.c_str(), nullptr, kDecimal);
      GE_CHK_BOOL_RET_STATUS(shape_value_str == std::to_string(result), FAILED,
                             "value %s is invalid, should be int, such as 0, 1, 2.", shape_value_str.c_str());
      shape_values.push_back(result);
    }
    user_shape_map.push_back(make_pair(ge::StringUtils::Trim(shape_pair_vec[0]), shape_values));
  }
  std::vector<std::string> sorted_input_shape_names = input_shape_names;
  std::sort(sorted_input_shape_names.begin(), sorted_input_shape_names.end());
  GE_CHK_BOOL_RET_STATUS(sorted_input_shape_names == input_shape_names, FAILED,
                         "input_shape input should be in alphabetical order.");
  return SUCCESS;
}
}  // namespace multibatch
}  // namespace ge
