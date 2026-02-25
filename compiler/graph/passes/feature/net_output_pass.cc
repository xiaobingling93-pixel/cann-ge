/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/net_output_pass.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "common/context/local_context.h"
#include "common/checker.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
static std::map<std::string, ge::DataType> output_type_str_to_datatype = {
    {"FP32", ge::DT_FLOAT},    {"FP16", ge::DT_FLOAT16},  {"INT8", ge::DT_INT8},    {"INT16", ge::DT_INT16},
    {"UINT16", ge::DT_UINT16}, {"UINT8", ge::DT_UINT8},   {"INT32", ge::DT_INT32},  {"INT64", ge::DT_INT64},
    {"UINT32", ge::DT_UINT32}, {"UINT64", ge::DT_UINT64}, {"DOUBLE", ge::DT_DOUBLE}, {"BF16", ge::DT_BF16},
    {"HIF8", ge::DT_HIFLOAT8}, {"FP8E5M2", ge::DT_FLOAT8_E5M2}, {"FP8E4M3FN", ge::DT_FLOAT8_E4M3FN},
};

// the size of user defined output datatype or format std::string after split by ":".
const size_t kUserDefinedElementCount = 2;
const size_t kNodesCount = 2;

///
/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
///
Status NetOutputPass::ClearStatus() {
  return SUCCESS;
}
Status NetOutputPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] Compute graph is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  GELOGI("[NETOUTPUT PASS] Run.graph is [%s]", graph->GetName().c_str());
  GE_ASSERT_SUCCESS(graph->CreateOrUpdateNetoutput());
  NodePtr output_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  GE_ASSERT_NOTNULL(output_node);
  GE_ASSERT_SUCCESS(SetNetOutputFormat(output_node));
  GE_ASSERT_SUCCESS(AddCtrlEdgesBetweenLeafAndNetOutput(graph, output_node));
  (void)TryToSetOutputNodeName(output_node);
  GE_CHK_STATUS_RET(TryToSetOutputMaxSize(output_node), "Failed to set output max size");
  // Add userdef attrs to netoutput node
  return SetUserDefDTypeAndFormatFromAtcParams(output_node);
}

bool NeedUpdateOutputByOutputTypeParm(std::string &output_type, const OpDescPtr &op_desc, const uint32_t &src_index,
                                      ge::DataType &dt) {
  if (output_type_str_to_datatype.find(output_type) != output_type_str_to_datatype.end()) {
    dt = output_type_str_to_datatype[output_type];
    return true;
  }

  std::vector<std::string> output_dt_str;
  if (ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_data_type", output_dt_str)) {
    for (const auto &dt_str : output_dt_str) {
      std::vector<std::string> dt_str_split = StringUtils::Split(dt_str, ':');
      if (dt_str_split.size() == kUserDefinedElementCount) {
        if (dt_str_split[0] == to_string(src_index)) {
          dt = TypeUtils::SerialStringToDataType(dt_str_split[1]);
          return true;
        }
      } else {
        GELOGW("The size of [%s] is not 2 after split.", dt_str.c_str());
        continue;
      }
    }
  }
  return false;
}

bool NeedUpdateOutputFp16Nc1hwc0(const OpDescPtr &op_desc, const uint32_t &src_index) {
  std::vector<std::string> output_dt_str;
  if (ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_fp16_5hd", output_dt_str)) {
    for (const auto &dt_str : output_dt_str) {
      std::vector<std::string> dt_str_split = StringUtils::Split(dt_str, ':');
      if (dt_str_split.size() == kUserDefinedElementCount) {
        if (dt_str_split[0] == to_string(src_index)) {
          return true;
        }
      } else {
        GELOGW("The size of [%s] is not 2 after split.", dt_str.c_str());
        continue;
      }
    }
  }
  return false;
}

Status NetOutputPass::SetUserDefDTypeAndFormatFromAtcParams(const NodePtr &output_node) const {
  if (output_node == nullptr) {
    GELOGI("[NETOUTPUT PASS] The graph no need netoutput node!");
    return SUCCESS;
  }
  auto output_type = GetLocalOmgContext().output_type;
  auto op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<std::string> userdef_dtypes;
  std::vector<std::string> userdef_formats;

  ge::DataType output_data_type = ge::DT_FLOAT;
  for (const auto &in_anchor : output_node->GetAllInDataAnchors()) {
    auto index = static_cast<uint32_t>(in_anchor->GetIdx());
    auto peer_out = in_anchor->GetPeerOutAnchor();
    if (peer_out == nullptr) {
      // If user set target, peer_out anchor will be unlinked.
      continue;
    }
    auto src_index = static_cast<uint32_t>(peer_out->GetIdx());
    auto src_node = peer_out->GetOwnerNode();
    OpDescPtr src_op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);

    // Update datatype
    if (NeedUpdateOutputByOutputTypeParm(output_type, src_op_desc, src_index, output_data_type)) {
      GELOGD("Add user-define datatype:%s to netoutput node.",
             TypeUtils::DataTypeToSerialString(output_data_type).c_str());
      userdef_dtypes.push_back(
          std::to_string(index).append(":").append(TypeUtils::DataTypeToSerialString(output_data_type)));
      continue;
    }
    // Output_node is not set,check if is_output_adjust_hw_layout is set
    bool set_fp16_nc1hwc0 = NeedUpdateOutputFp16Nc1hwc0(src_op_desc, src_index);
    if (set_fp16_nc1hwc0) {
      // Set DT_FLOAT16 & FORMAT_NC1HWC0
      userdef_dtypes.push_back(std::to_string(index).append(":").append(TypeUtils::DataTypeToSerialString(DT_FLOAT16)));
      userdef_formats.push_back(
          std::to_string(index).append(":").append(TypeUtils::FormatToSerialString(FORMAT_NC1HWC0)));
    }
  }
  if (!userdef_dtypes.empty() && !ge::AttrUtils::SetListStr(op_desc, ATTR_ATC_USER_DEFINE_DATATYPE, userdef_dtypes)) {
    REPORT_INNER_ERR_MSG("E19999", "User define datatype is empty or Set Attr:%s to op:%s(%s) failed",
                       ATTR_ATC_USER_DEFINE_DATATYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] User define datatype is empty or Set Attr:%s to op:%s(%s) failed",
           ATTR_ATC_USER_DEFINE_DATATYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (!userdef_formats.empty() && !ge::AttrUtils::SetListStr(op_desc, ATTR_ATC_USER_DEFINE_FORMAT, userdef_formats)) {
    REPORT_INNER_ERR_MSG("E19999", "User define format is empty or Set Attr:%s to op:%s(%s) failed",
                       ATTR_ATC_USER_DEFINE_FORMAT.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] User define format is empty or Set Attr:%s to op:%s(%s) failed",
           ATTR_ATC_USER_DEFINE_FORMAT.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status NetOutputPass::TryToSetOutputNodeName(const NodePtr &output_node) const {
  if (output_node == nullptr) {
    GELOGI("The graph has no net output node.");
    return SUCCESS;
  }
  auto &net_out_nodes = GetLocalOmgContext().net_out_nodes;
  if (!net_out_nodes.empty()) {
    GELOGD("Output node names have been set before, size:%zu.", net_out_nodes.size());
    return SUCCESS;
  }
  OpDescPtr op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<std::string> output_names;
  for (const InDataAnchorPtr &in_data_anchor : output_node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    auto input_desc = op_desc->GetInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    std::string tensor_name;
    bool has_tensor_name = ge::AttrUtils::GetStr(input_desc, ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME, tensor_name);
    if (has_tensor_name && !tensor_name.empty()) {
      std::string output_name = in_node->GetName() + ":" +
                                std::to_string(peer_out_anchor->GetIdx()) + ":" + tensor_name;
      output_names.push_back(output_name);
      GELOGD("Output[%d], name[%s]", in_data_anchor->GetIdx(), output_name.c_str());
    }
  }
  if (output_names.size() == output_node->GetInDataNodes().size()) {
    net_out_nodes.swap(output_names);
    GELOGI("Set output node name in net output pass.");
  } else {
    GELOGD("Not all input have origin output tensor name attr. All input data node size:%zu, tensor attr size:%zu.",
           output_node->GetInDataNodes().size(), output_names.size());
  }
  return SUCCESS;
}

Status NetOutputPass::TryToSetOutputMaxSize(const NodePtr &output_node) const {
  std::string output_max_size_list_str;
  (void)GetContext().GetOption(OUTPUT_MAX_SIZE, output_max_size_list_str);
  if (output_max_size_list_str.empty()) {
    GELOGI("Output max size option[%s] not set.", OUTPUT_MAX_SIZE.c_str());
    return SUCCESS;
  }
  GELOGI("Get output max size option[%s] success, value = %s.",
         OUTPUT_MAX_SIZE.c_str(), output_max_size_list_str.c_str());

  std::vector<int64_t> output_max_size_list;
  auto normalized = StringUtils::ReplaceAll(output_max_size_list_str, " ", "");
  GE_CHK_BOOL_RET_STATUS(!normalized.empty(), PARAM_INVALID, "Option value of after normalized is empty.");
  std::vector<std::string> output_max_size_str_list;
  output_max_size_str_list = StringUtils::Split(normalized, ';');
  for (const auto &output_max_size_str : output_max_size_str_list) {
    GELOGI("Transfer output max size str[%s] to int64.", output_max_size_str.c_str());
    int64_t max_size = -1;  // -1 means placeholder
    if (!output_max_size_str.empty()) {
      try {
        max_size = std::stol(output_max_size_str);
      } catch (std::out_of_range &) {
        GELOGE(PARAM_INVALID, "Value[%s] is out of range.", output_max_size_str.c_str());
        return PARAM_INVALID;
      } catch (std::invalid_argument &) {
        GELOGE(PARAM_INVALID, "Value[%s] is invalid.", output_max_size_str.c_str());
        return PARAM_INVALID;
      }
    }
    output_max_size_list.emplace_back(max_size);
  }
  GELOGI("Get output max size list success, value = %s.", ToString(output_max_size_list).c_str());

  OpDescPtr op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const InDataAnchorPtr &in_data_anchor : output_node->GetAllInDataAnchors()) {
    auto idx = in_data_anchor->GetIdx();
    GE_CHECK_GE(idx, 0);
    GE_CHECK_LE(static_cast<size_t>(idx + 1), output_max_size_list.size());
    auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(idx));
    int64_t max_size = 0UL;
    bool has_max_size = ge::AttrUtils::GetInt(input_desc, ATTR_NAME_GRAPH_OUTPUT_MAX_SIZE, max_size);
    GELOGI("Has output max size:%d.", static_cast<int32_t>(has_max_size));
    if (!has_max_size) {
      AttrUtils::SetInt(input_desc, ATTR_NAME_GRAPH_OUTPUT_MAX_SIZE, output_max_size_list[idx]);
      GELOGI("Node[%s] set attr max size[%ld] success, index = %d",
             output_node->GetName().c_str(), output_max_size_list[idx], idx);
    }
  }
  return SUCCESS;
}

Status NetOutputPass::AddCtrlEdgesBetweenLeafAndNetOutput(const ComputeGraphPtr &compute_graph, const ge::NodePtr &net_out_node) const {
  GE_CHECK_NOTNULL(net_out_node);
  bool is_user_define_output_nodes = false;
  for (const auto &item : compute_graph->GetGraphOutNodesInfo()) {
    auto op_desc = item.first->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->HasAttr(ATTR_ATC_USER_DEFINE_OUTPUT_NODES)) {
      is_user_define_output_nodes = true;
      break;
    }
  }
  if (!GetLocalOmgContext().user_out_nodes.empty() || is_user_define_output_nodes) {
    GELOGI("No need to add ctrl edge to netoutput because user out nodes have been set.");
    return SUCCESS;
  }
  bool graph_has_only_one_node_except_netoutput = (compute_graph->GetDirectNodesSize() == kNodesCount);
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetOpDesc() == nullptr || node->GetOpDesc()->GetType() == NETOUTPUT) {
      continue;
    }
    if ((node->GetInControlNodesSize() != 0 || node->GetInDataNodesSize() != 0 ||
         graph_has_only_one_node_except_netoutput) &&
        node->GetOutDataNodesSize() == 0 && node->GetOutControlNodesSize() == 0) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(node->GetOutControlAnchor(), net_out_node->GetInControlAnchor()),
                              "[Add][ControlEdge] between %s and %s failed", node->GetName().c_str(),
                              net_out_node->GetName().c_str());
      GELOGD("Add ctrl edge success. src name: %s, dst name: %s", node->GetName().c_str(),
             net_out_node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status NetOutputPass::SetNetOutputFormat(const ge::NodePtr &net_output) const {
  auto op_desc = net_output->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  bool is_inner_net_output = false;
  (void) AttrUtils::GetBool(op_desc, "_inner_net_output", is_inner_net_output);
  if (!is_inner_net_output) {
    return SUCCESS;
  }
  for (size_t i = 0U; i < op_desc->GetAllInputsDesc().size(); i++) {
    auto input_desc = op_desc->MutableInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    input_desc->SetFormat(FORMAT_ND);
    input_desc->SetOriginFormat(FORMAT_ND);
  }
  return SUCCESS;
}

REG_PASS_OPTION("NetOutputPass").LEVELS(OoLevel::kO0);
}  // namespace ge
