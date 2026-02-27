/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/omg/omg.h"

#include "base/err_msg.h"

#include <iostream>
#include <memory>

#include "framework/common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/helper/model_helper.h"
#include "common/helper/model_parser_base.h"
#include "common/helper/model_saver.h"
#include "common/context/properties_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/params.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "api/aclgrph/option_utils.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/weights_parser.h"
#include "parser/common/pre_checker.h"
#include "parser/common/convert/pb2json.h"
#include "common/proto_util.h"
#include "graph/utils/op_type_utils.h"
#include "graph_metadef/common/ge_common/util.h"

using std::ostringstream;

namespace ge {
namespace {
const std::string kGraphDefaultName = "domi_default";
const std::string kScopeIdAttr = "fusion_scope";
const char *const kOutputTypeSample = "The parameter is invalid. Valid format \"opname:index:dtype\".";
const char *const kOutputTypeSupport = "The value must be FP32, FP16, UINT8, INT8. A node can only have one type. "
                                       "The correct example is: --output_type=FP32.";
const char *const kOutputTypeError = "In the mode of specified node, the correct example is: node1:0:FP16;node2:0:FP32."
                                     "The nodes set in --output_type must be found in --out_nodes.";
const size_t kNodeNameIndex = 0;
const size_t kIndexStrIndex = 1;
const size_t kDTValueIndex = 2;
const size_t kOmInfoSize = 5;
const uint32_t kSetOutputWithNodeAndIndex = 0x1;
const uint32_t kSetOutputWithTensorName = 0x2;
const uint32_t kSetOutputModeMixed = 0x3;
const size_t kSoStoreIndex = 4;
const size_t kTaskInfoIndex = 3;
const std::set<domi::FrameworkType> kSupportTensorAsOutput = {domi::CAFFE, domi::ONNX};

void UpdateOutputTypeNameAndIndex(std::string &node_name, std::string &index_str) {
  const auto &final_out_nodes_map = domi::GetContext().final_out_nodes_map;
  const auto new_name_it = final_out_nodes_map.find(node_name + ":" + index_str);
  if (new_name_it != final_out_nodes_map.end()) {
    GELOGI("Update output_type node from [%s:%s] to [%s:%d]", node_name.c_str(), index_str.c_str(),
           new_name_it->second.first.c_str(), new_name_it->second.second);
    node_name = new_name_it->second.first;
    index_str = std::to_string(new_name_it->second.second);
  }
}
}  // namespace

// When the model is converted to a JSON file, the following operator attributes in the blacklist will be ignored
const std::set<std::string> kOmBlackFields = {"output",      "data_offset", "data", "workspace", "workspace_bytes",
                                              "memory_size", "weight_size", "size", "bt",        "quantize_factor"};

static std::map<std::string, ge::DataType> output_type_str_to_datatype = {
    {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}, {"INT8", ge::DT_INT8},
    {"HIF8", ge::DT_HIFLOAT8}, {"FP8E5M2", ge::DT_FLOAT8_E5M2}, {"FP8E4M3FN", ge::DT_FLOAT8_E4M3FN},
};

static bool CheckInputTrueOrFalse(const std::string &s, const std::string &atc_param) {
  if ((s == "true") || (s == "false")) {
    return true;
  } else {
    REPORT_PREDEFINED_ERR_MSG("E10005", std::vector<const char *>({"parameter", "value"}),
                              std::vector<const char *>({atc_param.c_str(), s.c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--%s]'s value[%s] must be true or false.",
           atc_param.c_str(), s.c_str());
    return false;
  }
}

static void ParseAtcParms(const std::map<std::string, std::string> &atc_params, const std::string &key,
                          std::string &param) {
  auto iter = atc_params.find(key);
  if (iter != atc_params.end()) {
    param = iter->second;
  }
}

static domi::Status CheckUserInputShape(const ComputeGraphPtr &graph) {
  for (auto it : domi::GetContext().user_input_dims) {
    std::string node_name = it.first;
    ge::NodePtr node = graph->FindNode(node_name);
    if (node == nullptr) {
      REPORT_PREDEFINED_ERR_MSG("E10016", std::vector<const char *>({"parameter", "opname"}),
                                std::vector<const char *>({"input_shape", node_name.c_str()}));
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_shape]'s opname[%s] does not exist in model",
             node_name.c_str());
      return PARAM_INVALID;
    }
    if (!OpTypeUtils::IsDataNode(node->GetType())) {
      REPORT_PREDEFINED_ERR_MSG("E10017", std::vector<const char *>({"parameter", "opname"}),
                                std::vector<const char *>({"input_shape", node_name.c_str()}));
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_shape]'s opname[%s] is not a input opname",
             node_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

static domi::Status CheckInputShapeNode(const ComputeGraphPtr &graph, bool is_dynamic_input,
                                        const std::string &input_shape_range, RunMode run_mode) {
  if ((!is_dynamic_input) && (run_mode != RunMode::MODEL_TO_JSON) && input_shape_range.empty()) {
    for (auto node : graph->GetDirectNode()) {
      if (OpTypeUtils::IsDataNode(node->GetType())) {
        auto data_op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(data_op_desc);
        auto tensor_desc = data_op_desc->MutableInputDesc(0);
        GE_CHECK_NOTNULL(tensor_desc);
        for (auto dim : tensor_desc->GetShape().GetDims()) {
          if (dim < 0) {
            GELOGE(PARAM_INVALID, "[Check][Param]Input op [%s] shape %ld is negative, "
                   "maybe you should set input_shape to specify its shape", node->GetName().c_str(), dim);
            const std::string reason =
                "The shapes of inputs contain -1 in the model. You may need to set input shape to specify its shape.";
            REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                      std::vector<const char *>({"--input_shape", "NULL", reason.c_str()}));
            return PARAM_INVALID;
          }
        }
      }
    }
  }

  return CheckUserInputShape(graph);
}

void AddAttrsForInputNodes(const std::vector<std::string> &adjust_fp16_format_vec,
                           const std::string &fp16_nodes_name, uint32_t index,
                           const OpDescPtr &op_desc) {
  if (AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_DATATYPE, TypeUtils::DataTypeToSerialString(DT_FLOAT16))) {
    if ((index < adjust_fp16_format_vec.size()) && (adjust_fp16_format_vec[index] == "true")) {
      GELOGI("This node [%s] should be set NC1HWC0", fp16_nodes_name.c_str());
      if (!AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_FORMAT, TypeUtils::FormatToSerialString(FORMAT_NC1HWC0))) {
        GELOGW("This node [%s] set NC1HWC0 failed", fp16_nodes_name.c_str());
      }
    }
  }
}

static domi::Status CheckInputFp16Nodes(const ComputeGraphPtr &graph, const std::string &input_fp16_nodes,
                                        const std::string &is_input_adjust_hw_layout) {
  GE_CHECK_NOTNULL(graph);
  std::vector<std::string> adjust_fp16_format_vec;
  if (!is_input_adjust_hw_layout.empty()) {
    adjust_fp16_format_vec = StringUtils::Split(is_input_adjust_hw_layout, ',');
    for (auto &s : adjust_fp16_format_vec) {
      StringUtils::Trim(s);
      if (!CheckInputTrueOrFalse(s, "is_input_adjust_hw_layout")) {
        GELOGE(PARAM_INVALID, "[Check][Param]Invalid Param, is_input_adjust_hw_layout only support true/false:"
               "but is [%s]", is_input_adjust_hw_layout.c_str());
        return PARAM_INVALID;
      }
    }
  }
  if (input_fp16_nodes.empty()) {
    return SUCCESS;
  }
  GELOGI("The input_fp16_nodes is set %s", input_fp16_nodes.c_str());
  std::vector<std::string> input_fp16_nodes_vec = StringUtils::Split(input_fp16_nodes, ';');
  for (uint32_t i = 0; i < input_fp16_nodes_vec.size(); ++i) {
    ge::NodePtr node = graph->FindNode(input_fp16_nodes_vec[i]);
    if (node == nullptr) {
      REPORT_PREDEFINED_ERR_MSG("E10016", std::vector<const char *>({"parameter", "opname"}),
                                std::vector<const char *>({"input_fp16_nodes", input_fp16_nodes_vec[i].c_str()}));
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_fp16_nodes]'s opname[%s] does not exist in model",
             input_fp16_nodes_vec[i].c_str());
      return PARAM_INVALID;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!OpTypeUtils::IsDataNode(op_desc->GetType())) {
      REPORT_PREDEFINED_ERR_MSG("E10017", std::vector<const char *>({"parameter", "opname"}),
                                std::vector<const char *>({"input_fp16_nodes", input_fp16_nodes_vec[i].c_str()}));
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_fp16_nodes]'s opname[%s] is not a input opname",
             input_fp16_nodes_vec[i].c_str());
      return PARAM_INVALID;
    }
    AddAttrsForInputNodes(adjust_fp16_format_vec, input_fp16_nodes_vec[i], i, op_desc);
  }
  return SUCCESS;
}

static domi::Status ParseOutputFp16NodesFormat(const std::string &is_output_fp16) {
  if (is_output_fp16.empty()) {
    return SUCCESS;
  }

  std::vector<domi::domiTensorFormat_t> &output_formats = domi::GetContext().output_formats;
  output_formats.clear();
  std::vector<std::string> node_format_vec = StringUtils::Split(is_output_fp16, ',');
  for (auto &is_fp16 : node_format_vec) {
    StringUtils::Trim(is_fp16);
    if (!CheckInputTrueOrFalse(is_fp16, "is_output_adjust_hw_layout")) {
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid Param, is_output_adjust_hw_layout "
             "only support true/false: but is [%s]", is_output_fp16.c_str());
      return PARAM_INVALID;
    }
    if (is_fp16 == "false") {
      output_formats.push_back(domi::DOMI_TENSOR_ND);
    } else if (is_fp16 == "true") {
      output_formats.push_back(domi::DOMI_TENSOR_NC1HWC0);
    }
  }
  return SUCCESS;
}

void FindParserSo(const std::string &path, std::vector<std::string> &file_list, std::string &caffe_parser_path) {
  // path, Change to absolute path
  std::string real_path = RealPath(path.c_str());
  if (real_path.empty()) {  // plugin path does not exist
    return;
  }
  struct stat stat_buf;
  if ((stat(real_path.c_str(), &stat_buf) != 0) || (!S_ISDIR(stat_buf.st_mode))) {
    GELOGI("The path %s is not a directory.", real_path.c_str());
    return;
  }

  struct dirent *dent(nullptr);
  DIR *dir = opendir(real_path.c_str());

  if (dir == nullptr) {  //  plugin path does not exist
    GELOGW("Open directory %s failed.", path.c_str());
    return;
  }

  while ((dent = readdir(dir)) != nullptr) {
    if ((strcmp(dent->d_name, ".") == 0) || (strcmp(dent->d_name, "..") == 0)) {
      continue;
    }
    std::string name = dent->d_name;
    std::string full_name = real_path + "/" + name;
    const std::string so_suff = ".so";
    const std::string caffe_parser_so_suff = "lib_caffe_parser.so";
    if (name.size() >= so_suff.size() && name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0) {
      if (full_name.size() >= caffe_parser_so_suff.size() &&
          full_name.compare(full_name.size() - caffe_parser_so_suff.size(), caffe_parser_so_suff.size(),
                            caffe_parser_so_suff) == 0) {
        caffe_parser_path = full_name;
      } else {  // save parser so path into file_list vector
        file_list.push_back(full_name);
      }
      continue;
    }

    FindParserSo(full_name, file_list, caffe_parser_path);
  }
  closedir(dir);
  return;
}

bool CheckDigitStr(std::string &str) {
  for (char c : str) {
    if (!isdigit(c)) {
      GELOGE(FAILED, "[Check][Param]value[%s] is not positive integer", str.c_str());
      return false;
    }
  }
  return true;
}

domi::Status StringToInt(std::string &str, int32_t &value) {
  try {
    if (!CheckDigitStr(str)) {
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid of digit std::string: %s ", str.c_str());
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({"--output_type", str.c_str(), "The value is not a positive integer."}));
      return PARAM_INVALID;
    }
    value = stoi(str);
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid of digit std::string: %s, catch invalid_argument.", str.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10014", std::vector<const char *>({"parameter", "value"}),
                              std::vector<const char *>({"--output_type", str.c_str()}));
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid of digit std::string: %s, catch out_of_range.", str.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10013", std::vector<const char *>({"parameter", "value"}),
                              std::vector<const char *>({"--output_type", str.c_str()}));
    return PARAM_INVALID;
  }
  return SUCCESS;
}

domi::Status VerifyOutputTypeAndOutNodes(std::vector<std::string> &out_type_vec) {
  std::vector<std::pair<std::string, int32_t>> user_out_nodes = domi::GetContext().user_out_nodes;
  std::set<std::string> out_nodes_info;
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    // out_nodes set should include output_type and output_format
    std::string tmp = user_out_nodes[i].first + ":" + to_string(user_out_nodes[i].second);
    out_nodes_info.emplace(tmp);
  }
  for (uint32_t i = 0; i < out_type_vec.size(); ++i) {
    if (out_nodes_info.find(out_type_vec[i]) == out_nodes_info.end()) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                std::vector<const char *>({"--output_type", out_type_vec[i].c_str(), kOutputTypeError}));
      GELOGE(FAILED, "[Check][Param]Invalid value for --output_type[%s], %s.",
             out_type_vec[i].c_str(), kOutputTypeError);
      return FAILED;
    }
  }
  return SUCCESS;
}

domi::Status CheckOutPutDataTypeSupport(const std::string &output_type) {
  std::map<std::string, ge::DataType>::const_iterator it = output_type_str_to_datatype.find(output_type);
  if (it == output_type_str_to_datatype.cend()) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"--output_type", output_type.c_str(), kOutputTypeSupport}));
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid value for --output_type[%s], %s.",
           output_type.c_str(), kOutputTypeSupport);
    return FAILED;
  }
  return SUCCESS;
}

domi::Status ParseOutputType(const std::string &output_type, std::map<std::string,
                       std::vector<std::string>> &output_node_dt_map) {
  if (output_type.find(':') == std::string::npos) {
    GELOGI("output_type is not multiple nodes, means all out nodes");
    return CheckOutPutDataTypeSupport(output_type);
  }
  std::vector<std::string> out_type_vec;
  std::vector<std::string> nodes_v = StringUtils::Split(output_type, ';');
  for (const std::string &node : nodes_v) {
    std::vector<std::string> node_index_type_v = StringUtils::Split(node, ':');
    if (node_index_type_v.size() != 3) {  // The size must be 3.
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                std::vector<const char *>({"--output_type", node.c_str(), kOutputTypeSample}));
      GELOGE(PARAM_INVALID, "[Parse][Param]Invalid value for --output_type[%s], %s.", node.c_str(), kOutputTypeSample);
      return FAILED;
    }
    std::string node_name = StringUtils::Trim(node_index_type_v[kNodeNameIndex]);
    std::string index_str = StringUtils::Trim(node_index_type_v[kIndexStrIndex]);
    UpdateOutputTypeNameAndIndex(node_name, index_str);
    int32_t index;
    if (StringToInt(index_str, index) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Convert][Type]This str must be digit string, while the actual input is %s.",
             index_str.c_str());
      return FAILED;
    }
    std::string dt_value = StringUtils::Trim(node_index_type_v[kDTValueIndex]);
    std::map<std::string, ge::DataType>::const_iterator it = output_type_str_to_datatype.find(dt_value);
    if (it == output_type_str_to_datatype.cend()) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                std::vector<const char *>({"--output_type", dt_value.c_str(), kOutputTypeSupport}));
      GELOGE(ge::PARAM_INVALID, "[Parse][Param]Invalid value for --output_type[%s], %s.",
             dt_value.c_str(), kOutputTypeSupport);
      return FAILED;
    }
    const ge::DataType tmp_dt = it->second;

    out_type_vec.push_back(node_name + ":" + index_str);
    std::string index_dt_str = index_str + ":" + TypeUtils::DataTypeToSerialString(tmp_dt);
    auto it1 = output_node_dt_map.find(node_name);
    if (it1 == output_node_dt_map.end()) {
      std::vector<std::string> tmp_vec;
      tmp_vec.push_back(index_dt_str);
      output_node_dt_map.emplace(node_name, tmp_vec);
    } else {
      it1->second.push_back(index_dt_str);
    }
  }
  return VerifyOutputTypeAndOutNodes(out_type_vec);
}

domi::Status CheckOutNode(ge::OpDescPtr op_desc, int32_t index) {
  int32_t out_size = op_desc->GetOutputsSize();
  if ((index < 0) || (index >= out_size)) {
    GELOGE(FAILED,
           "[Check][Param]out_node [%s] output index:%d must be smaller "
           "than node output size:%d and can not be negative",
           op_desc->GetName().c_str(), index, out_size);
    std::string fail_reason = "Output index:\"" + to_string(index) + "\" must be smaller than output size:" +
                              to_string(out_size) + " and cannot be negative.";
    REPORT_PREDEFINED_ERR_MSG("E10003", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"out_nodes", op_desc->GetName().c_str(), fail_reason.c_str()}));
    return FAILED;
  }
  return SUCCESS;
}
domi::Status GetDefaultOutInfo(ge::ComputeGraphPtr &compute_graph,
                               std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  std::vector<std::pair<std::string, int32_t>> default_out_nodes = domi::GetContext().default_out_nodes;
  if (!default_out_nodes.empty()) {
    for (uint32_t i = 0U; i < default_out_nodes.size(); ++i) {
      NodePtr out_node = compute_graph->FindNode(default_out_nodes[i].first);
      // onnx may be set isolated output tensor
      if (out_node == nullptr && domi::GetContext().type != domi::ONNX) {
        REPORT_PREDEFINED_ERR_MSG("E10016", std::vector<const char *>({"parameter", "opname"}),
                                  std::vector<const char *>({"out_nodes", default_out_nodes[i].first.c_str()}));
        GELOGE(FAILED, "[Check][Param]Can not find src node (%s) in graph.", default_out_nodes[i].first.c_str());
        return FAILED;
      }
      if (out_node == nullptr) {
        continue;
      }
      output_nodes_info.push_back(std::make_pair(out_node, default_out_nodes[i].second));
      GELOGD("Get default output node:%s.", out_node->GetName().c_str());
    }
    return SUCCESS;
  }

  for (ge::NodePtr node : compute_graph->GetDirectNode()) {
    if (!node->GetInAllNodes().empty() && node->GetOutAllNodes().empty()) {
      domi::Status ret = GetOutputLeaf(node, output_nodes_info);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "find leaf fail.");
    }
  }
  return SUCCESS;
}

domi::Status SetOutputNodeInfo(ge::Graph &graph, const std::string &output_type) {
  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  std::vector<std::pair<std::string, int32_t>> user_out_nodes = domi::GetContext().user_out_nodes;
  std::vector<domi::domiTensorFormat_t> output_formats = domi::GetContext().output_formats;
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  std::vector<std::string> output_nodes_name;
  std::map<std::string, std::vector<std::string>> output_node_dt_map;
  if (!output_type.empty()) {
    if (ParseOutputType(output_type, output_node_dt_map) != SUCCESS) {
      GELOGE(FAILED, "[Parse][output_type] failed.");
      return FAILED;
    }
  }

  // User declared outputs
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    ge::NodePtr out_node = compute_graph->FindNode(user_out_nodes[i].first);
    if (out_node == nullptr) {
      REPORT_PREDEFINED_ERR_MSG("E10016", std::vector<const char *>({"parameter", "opname"}),
                                std::vector<const char *>({"out_nodes", user_out_nodes[i].first.c_str()}));
      GELOGE(FAILED, "[Check][Param]Can not find src node (%s) in graph.", user_out_nodes[i].first.c_str());
      return FAILED;
    }
    auto op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (CheckOutNode(op_desc, user_out_nodes[i].second) != SUCCESS) {
      GELOGE(FAILED, "[Check][OutNode] (%s) fail.", user_out_nodes[i].first.c_str());
      return FAILED;
    }

    // add user_define_output_nodes attr.
    (void)ge::AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_OUTPUT_NODES, "true");

    if (i < output_formats.size()) {
      if (output_formats[i] == domi::DOMI_TENSOR_NC1HWC0) {
        GELOGI("The output node [%s] should be set NC1HWC0", user_out_nodes[i].first.c_str());
        std::vector<std::string> output_fp16_5hd_vec;
        (void)ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_fp16_5hd", output_fp16_5hd_vec);
        output_fp16_5hd_vec.push_back(std::to_string(user_out_nodes[i].second) + ":" + "NC1HWC0");
        (void)ge::AttrUtils::SetListStr(op_desc, "_user_defined_output_fp16_5hd", output_fp16_5hd_vec);
      }
    }
    std::map<std::string, std::vector<std::string>>::const_iterator it =
        output_node_dt_map.find(user_out_nodes[i].first);
    if (it != output_node_dt_map.cend()) {
      GELOGI("The output node [%s] need to be set output_type", user_out_nodes[i].first.c_str());
      (void)ge::AttrUtils::SetListStr(op_desc, "_user_defined_output_data_type", it->second);
    }
    output_nodes_info.push_back(std::make_pair(out_node, user_out_nodes[i].second));
  }
  // default output node (leaf)
  if (user_out_nodes.empty()) {
    if (GetDefaultOutInfo(compute_graph, output_nodes_info) != SUCCESS) {
      GELOGE(FAILED, "[Get][DefaultOutInfo] failed.");
      return FAILED;
    }
  }
  CreateOutputNodesInfo(output_nodes_info, output_nodes_name);
  GE_ASSERT_SUCCESS(compute_graph->SetGraphOutNodesInfo(output_nodes_info));
  domi::GetContext().net_out_nodes = output_nodes_name;
  return SUCCESS;
}

void CreateOutputNodesInfo(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                           std::vector<std::string> &output_nodes_name) {
  output_nodes_name.clear();
  auto &out_tensor_names = domi::GetContext().out_tensor_names;
  if (domi::GetContext().out_tensor_names.empty()) {
    // tf process, no top name.
    for (const auto &output_node_info : output_nodes_info) {
      std::string node_name = output_node_info.first->GetName();
      int32_t index = output_node_info.second;
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
    return;
  }

  // Need add top name after node_name:index
  for (size_t i = 0; i < output_nodes_info.size(); ++i) {
    auto node = output_nodes_info[i].first;
    int32_t index = output_nodes_info[i].second;
    std::string node_name = node->GetName();
    if (i < out_tensor_names.size()) {
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(index));
      (void)AttrUtils::SetStr(output_desc, ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME, out_tensor_names[i]);
      std::string output_name = node_name + ":" + std::to_string(index) + ":" + out_tensor_names[i];
      output_nodes_name.push_back(output_name);
      GELOGD("Output[%zu] name[%s]", i, output_name.c_str());
    } else {
      GELOGW("Get top name of node [%s] fail.", node_name.c_str());
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
  }
}

domi::Status GetOutputLeaf(NodePtr node, std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  ge::OpDescPtr tmpDescPtr = node->GetOpDesc();
  if (tmpDescPtr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param node has no opdesc.");
    GELOGE(FAILED, "[Check][Param]Get outnode op desc fail.");
    return FAILED;
  }
  size_t size = tmpDescPtr->GetOutputsSize();
  if (node->GetType() != NETOUTPUT) {
    for (size_t index = 0; index < size; ++index) {
      output_nodes_info.push_back(std::make_pair(node, index));
      GELOGD("Get output leaf node:%s.", node->GetName().c_str());
    }
  } else {
    const auto in_anchors = node->GetAllInDataAnchors();
    for (auto in_anchor : in_anchors) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "GetPeerOutAnchor return nullptr, node:%s.", node->GetName().c_str());
        GELOGE(FAILED, "[Invoke][GetPeerOutAnchor]Get leaf node op desc fail.");
        return FAILED;
      }
      auto out_node = out_anchor->GetOwnerNode();
      output_nodes_info.push_back(std::make_pair(out_node, out_anchor->GetIdx()));
    }
  }
  return SUCCESS;
}

///
/// @ingroup domi_common
/// @brief Initialize omgcontext based on command line input
/// @param [in] input_shape Input shape std::string to be parsed
/// @return SUCCESS: parse successfully; PARAM_INVALID：parse failed
///
domi::Status InitDomiOmgContext(const std::string &input_shape, const std::string &input_format,
                                const std::string &net_format,
                                bool is_dynamic_input) {
  (void)net_format;
  // Clear omgcontext data first
  domi::GetContext().input_dims.clear();
  domi::GetContext().user_input_dims.clear();
  domi::GetContext().is_dynamic_input = is_dynamic_input;

  // the default value is ND
  domi::GetContext().format = domi::DOMI_TENSOR_ND;
  if (!input_format.empty()) {
    std::map<std::string, domi::domiTensorFormat_t>::const_iterator iter =
        ge::input_format_str_to_geformat.find(input_format);
    if (iter != ge::input_format_str_to_geformat.cend()) {
      domi::GetContext().format = iter->second;
    } else {
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10061", std::vector<const char *>({"value", "parameter", "expected_value"}),
          std::vector<const char *>({input_format.c_str(), "input_format", "ND, NCHW, NHWC, CHWN, NC1HWC0 or NHWC1C0"}));
      GELOGE(PARAM_INVALID, "[Check][Param]Input format %s not support, "
             "expect ND/NCHW/NHWC/CHWN/NC1HWC0/NHWC1C0.", input_format.c_str());
      return PARAM_INVALID;
    }
  }

  // Input is empty, do not process
  if (input_shape.empty()) {
    return SUCCESS;
  }

  // Analyze the input shape paramete
  std::map<std::string, std::vector<int64_t>> &shape_map = domi::GetContext().input_dims;

  if (!ge::ParseInputShape(input_shape, domi::GetContext().input_dims, domi::GetContext().user_input_dims,
                           is_dynamic_input) || shape_map.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "ParseInputShape failed for %s", input_shape.c_str());
    GELOGE(PARAM_INVALID, "[Parse][InputShape] %s failed.", input_shape.c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

domi::Status ParseOutNodes(const std::string &out_nodes) {
  try {
    // parse output node
    if (!out_nodes.empty()) {
      domi::GetContext().out_nodes_map.clear();
      domi::GetContext().user_out_nodes.clear();
      domi::GetContext().user_out_tensors.clear();
      uint32_t set_output_mode = 0;

      std::vector<std::string> nodes_v = StringUtils::Split(out_nodes, ';');
      for (const std::string &node : nodes_v) {
        std::vector<std::string> key_value_v = StringUtils::Split(node, ':');
        if (key_value_v.size() != 2) {  // The size must be 2.
          if (key_value_v.size() == 1 && kSupportTensorAsOutput.count(domi::GetContext().type) > 0) {
            set_output_mode |= kSetOutputWithTensorName;
            if (set_output_mode == kSetOutputModeMixed) {
              break;
            }
            domi::GetContext().user_out_tensors.push_back(node);
            continue;
          }
          REPORT_PREDEFINED_ERR_MSG(
              "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
              std::vector<const char *>(
                  {"--out_nodes", node.c_str(),
                   "The parameter format is invalid. Valid format: \"node_name1:0;node_name1:1;node_name2:0\"."}));
          GELOGE(PARAM_INVALID,
                 "[Parse][Param]The input format of --out_nodes is invalid, the correct format is "
                 "\"node_name1:0;node_name1:1;node_name2:0\", while the actual input is %s.",
                 node.c_str());
          return PARAM_INVALID;
        }
        set_output_mode |= kSetOutputWithNodeAndIndex;
        if (set_output_mode == kSetOutputModeMixed) {
          break;
        }
        // stoi: The method may throw an exception: invalid_argument/out_of_range
        if (!CheckDigitStr(key_value_v[1])) {
          REPORT_PREDEFINED_ERR_MSG(
              "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
              std::vector<const char *>({"--out_nodes", out_nodes.c_str(), "The index is not a positive integer."}));
          GELOGE(PARAM_INVALID, "[Parse][Param]This str must be digit string, while the actual input is %s",
                 out_nodes.c_str());
          return PARAM_INVALID;
        }

        auto iter = domi::GetContext().out_nodes_map.find(key_value_v[0]);
        int32_t index = stoi(StringUtils::Trim(key_value_v[1]));
        GELOGD("Get output info: node[%s] and index[%d]", key_value_v[0].c_str(), index);
        if (iter != domi::GetContext().out_nodes_map.end()) {
          iter->second.emplace_back(index);
        } else {
          std::vector<int32_t> index_v;
          index_v.emplace_back(index);
          domi::GetContext().out_nodes_map.emplace(key_value_v[0], index_v);
        }
        domi::GetContext().user_out_nodes.push_back(std::make_pair(key_value_v[0], index));
      }
      if (set_output_mode == kSetOutputModeMixed) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
            std::vector<const char *>(
                {"--out_nodes", out_nodes.c_str(), "Only one of index, top_name and output_name can be used."}));
        GELOGE(PARAM_INVALID, "[Parse][Param]This out_nodes str must be all index or tensor_name, "
                              "while the actual input is %s", out_nodes.c_str());
        return PARAM_INVALID;
      }
    }
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "[Parse][Param]Invalid of out_nodes: %s ", out_nodes.c_str());
    REPORT_PREDEFINED_ERR_MSG(
            "E10014", std::vector<const char *>({"parameter", "value"}),
            std::vector<const char *>({"--out_nodes", out_nodes.c_str()}));
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "[Parse][Param]Invalid of out_nodes: %s ", out_nodes.c_str());
    REPORT_PREDEFINED_ERR_MSG(
            "E10013", std::vector<const char *>({"parameter", "value"}),
            std::vector<const char *>({"--out_nodes", out_nodes.c_str()}));
    return PARAM_INVALID;
  }
  return SUCCESS;
}

/// @ingroup domi_common
///  @brief Judge whether the op_Name_Map parameter matches the network
///  @param [in] graph Input network graph
///  @return SUCCESS: Input parameters are correct; PARAM_INVALID: Input parameters are incorrect
///
static domi::Status CheckOpNameMap(const ComputeGraphPtr &graph, const std::string &op_conf) {
  GE_CHECK_NOTNULL(graph);
  std::map<std::string, std::string> graphNodeTypes;
  for (const NodePtr &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "param graph's node has no opdesc.");
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid parameter for opDesc.");
      return PARAM_INVALID;
    }
    graphNodeTypes[op_desc->GetType()] = "";
  }
  std::map<std::string, std::string> &propertiesMap = domi::GetContext().op_conf_map;
  if (propertiesMap.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"op_name_map", op_conf.c_str(), "The file content is empty."}));
    GELOGE(PARAM_INVALID, "[Check][Param]op_name_map file content is empty, please check file!");
    return PARAM_INVALID;
  }
  for (auto iter = propertiesMap.cbegin(); iter != propertiesMap.cend(); iter++) {
    GE_IF_BOOL_EXEC(graphNodeTypes.find(iter->second) == graphNodeTypes.end(),
                    REPORT_PREDEFINED_ERR_MSG(
                        "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
                        std::vector<const char *>({"op_name_map", op_conf.c_str(),
                                                   ("Type[" + iter->second + "] is not found in the model.").c_str()}));
                    GELOGE(PARAM_INVALID, "[Find][NodeType]Invalid parameter for op_name_map."); return PARAM_INVALID;);
  }
  return SUCCESS;
}

domi::Status CheckParamForAirInput(ge::Graph &graph) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_RETURN_IF_ERROR(CheckUserInputShape(compute_graph));
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY domi::Status ParseGraph(ge::Graph &graph, const std::map<std::string, std::string> &atc_params,
                                                 const char *model_file, const char *weights_file,
                                                 domi::FrameworkType type, const char *op_conf, const char *target,
                                                 RunMode run_mode, bool is_dynamic_input) {
  GE_CHECK_NOTNULL(model_file);
  GE_CHECK_NOTNULL(weights_file);
  domi::GetContext().type = type;
  domi::GetContext().run_mode = run_mode;
  // Prevent data residue in multiple calls
  PreChecker::Instance().Clear();

  Params::Instance()->SetTarget(target);

  // Create an empty computegraph
  std::string om_name;
  ParseAtcParms(atc_params, "output", om_name);
  ModelHelper model_helper;
  std::string graph_name = "";
  domi::Status name_ret = model_helper.GetBaseNameFromFileName(om_name, graph_name);
  if (name_ret != SUCCESS) {
    graph_name = kGraphDefaultName + "_" + CurrentTimeInStr();
  }
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(compute_graph);
  graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // initialize omgContext
  std::string input_shape;
  ParseAtcParms(atc_params, "input_shape", input_shape);
  std::string input_format;
  ParseAtcParms(atc_params, "input_format", input_format);
  GE_RETURN_WITH_LOG_IF_ERROR(InitDomiOmgContext(input_shape, input_format, "", is_dynamic_input),
                              "[Call][InitDomiOmgContext] ret fail");

  std::string is_output_adjust_hw_layout;
  ParseAtcParms(atc_params, "is_output_adjust_hw_layout", is_output_adjust_hw_layout);
  GE_RETURN_WITH_LOG_IF_ERROR(ParseOutputFp16NodesFormat(is_output_adjust_hw_layout),
                              "[Call][ParseOutputFp16NodesFormat]Parse is_output_fp16 failed");

  std::string out_nodes;
  ParseAtcParms(atc_params, "out_nodes", out_nodes);
  GE_RETURN_WITH_LOG_IF_ERROR(ParseOutNodes(out_nodes), "[Parse][OutNodes] fail");

  std::string output_type;
  ParseAtcParms(atc_params, "output_type", output_type);

  // parse configuration item
  if ((op_conf != nullptr) && (*op_conf != '\0')) {
    // divided by ":"
    PropertiesManager::Instance().SetPropertyDelimiter(OP_CONF_DELIMITER);
    // Parsing the op_conf configuration item file
    GE_IF_BOOL_EXEC(!PropertiesManager::Instance().Init(op_conf),
                    REPORT_PREDEFINED_ERR_MSG(
                            "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
                            std::vector<const char *>({"op_name_map", op_conf, "File content error."}));
                    GELOGE(FAILED, "[Invoke][Init]op_name_map init failed!");
                    return FAILED);
    // Return map and put it into ATC global variable
    domi::GetContext().op_conf_map = PropertiesManager::Instance().GetPropertyMap();
  }

  // parse network model
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(type);
  if (model_parser == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "CreateModelParser failed, type:%d", type);
    GELOGE(FAILED, "[Create][ModelParser] ret fail, type:%d.", type);
    return FAILED;
  }
  UpdateParserCtxWithOmgCtx();
  domi::Status ret = model_parser->Parse(model_file, graph);
  UpdateOmgCtxWithParserCtx();

  // Generate the report in case of pre inspection failure or only pre inspection mode
  if ((PreChecker::Instance().HasError()) || (run_mode == ge::RunMode::ONLY_PRE_CHECK)) {
    std::string check_report;
    ParseAtcParms(atc_params, "check_report", check_report);
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().Save(check_report),
                                "[Invoke][Save]Generate pre-checking report failed.");
    GEEVENT("The pre-checking report has been saved to %s.", check_report.c_str());
  }

  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "ATC model parse ret fail.");

  std::string input_fp16_nodes;
  ParseAtcParms(atc_params, "input_fp16_nodes", input_fp16_nodes);
  std::string is_input_adjust_hw_layout;
  ParseAtcParms(atc_params, "is_input_adjust_hw_layout", is_input_adjust_hw_layout);
  compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  if ((run_mode == ge::RunMode::ONLY_PRE_CHECK) && (compute_graph == nullptr)) {
    PreChecker::Instance().Clear();
    return SUCCESS;
  }
  GE_RETURN_IF_ERROR(CheckInputFp16Nodes(compute_graph, input_fp16_nodes, is_input_adjust_hw_layout));
  std::string input_shape_range;
  ParseAtcParms(atc_params, INPUT_SHAPE_RANGE, input_shape_range);
  GE_RETURN_IF_ERROR(CheckInputShapeNode(compute_graph, is_dynamic_input, input_shape_range, run_mode));

  // Verify the contents of the op_name_map
  if ((op_conf != nullptr) && (*op_conf != '\0')) {
    GE_RETURN_WITH_LOG_IF_ERROR(CheckOpNameMap(compute_graph, op_conf),
                                "[Invoke][CheckOpNameMap]op_name_map parameter is not fit with input net!");
  }

  // Print parse network structure
  compute_graph->Dump();

  // parse weight
  graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto weights_parser = domi::WeightsParserFactory::Instance()->CreateWeightsParser(type);
  GE_ASSERT_NOTNULL(weights_parser);
  ret = weights_parser->Parse(weights_file, graph);

  // IN ONLY_PRE_CHECK mode, generate pre inspection report only.
  if ((PreChecker::Instance().HasError()) || (run_mode == ge::RunMode::ONLY_PRE_CHECK)) {
    std::string check_report;
    ParseAtcParms(atc_params, "check_report", check_report);
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().Save(check_report),
                                "[Invoke][Save]Generate pre-checking report failed.");
    GEEVENT("The pre-checking report has been saved to %s.", check_report.c_str());
  }
  // Prevent data residue in multiple calls
  PreChecker::Instance().Clear();

  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "[Check][State]ATC weights parse ret fail.");

  // parser input shape range and update op shape range
  GE_RETURN_WITH_LOG_IF_ERROR(UpdateDynamicInputShapeRange(compute_graph, input_shape_range),
                              "[Update][DynamicInputShapeRange] failed");

  return SUCCESS;
}

void GetGroupName(ge::proto::ModelDef &model_def) {
  auto model_attr_map = model_def.mutable_attr();
  auto fusion_model_op_list_iter = model_attr_map->find(MODEL_ATTR_FUSION_MODEL_DEF);
  if (fusion_model_op_list_iter == model_attr_map->end()) {
    return;
  }
  int32_t fusion_op_index = 0;
  const proto::AttrDef &fm_attr_def = fusion_model_op_list_iter->second;
  for (int32_t i = 0; i < model_def.graph_size(); ++i) {
    auto graph = model_def.mutable_graph(i);
    for (int32_t j = 0; j < graph->op_size(); ++j) {
      const auto bt = (fm_attr_def.list().bt_size() <= fusion_op_index) ? "" : fm_attr_def.list().bt(fusion_op_index++);
      if (bt.empty()) {
        GELOGW("Fusion op list bt is empty");
        return;
      }

      proto::OpDef fusion_op_def;
      (void)(fusion_op_def.ParseFromArray(bt.data(), bt.size()));
      auto fusion_attr_map = fusion_op_def.mutable_attr();
      auto fusion_iter = fusion_attr_map->find(kScopeIdAttr);
      if (fusion_iter == fusion_attr_map->end()) {
        continue;
      }

      uint64_t scope_id = static_cast<uint64_t>(fusion_iter->second.i());
      proto::OpDef *op_def = graph->mutable_op(j);
      auto &attr_map = *op_def->mutable_attr();

      int64_t stream_id = op_def->stream_id();
      uint16_t l1_id = ((static_cast<uint64_t>(scope_id) & 0xFFFF0000U)) >> 16U;
      if (l1_id != 0U) {
        std::ostringstream group_name;
        group_name << "group_op_l1_" << l1_id << "_" << stream_id;
        attr_map["group_op_name"].set_s(group_name.str());
        continue;
      }

      uint16_t ub_id = (static_cast<uint64_t>(scope_id & 0xFFFFU));
      if (ub_id != 0U) {
        std::ostringstream group_name;
        group_name << "group_op_ub_" << ub_id << "_" << stream_id;
        attr_map["group_op_name"].set_s(group_name.str());
      }
    }
  }
}

FMK_FUNC_HOST_VISIBILITY void PrintModelInfo(ge::proto::ModelDef *model_def, uint32_t modeldef_size) {
  std::cout << "============ Display Model Info start ============" << std::endl;

  auto model_attr_map = model_def->mutable_attr();
  // system info
  auto iter = model_attr_map->find(ATTR_MODEL_ATC_VERSION);
  auto atc_version = (iter != model_attr_map->end()) ? iter->second.s() : "";
  iter = model_attr_map->find("soc_version");
  auto soc_version = (iter != model_attr_map->end()) ? iter->second.s() : "";
  iter = model_attr_map->find("framework_type");
  auto framework_type = (iter != model_attr_map->end()) ? iter->second.s() : "";
  // original atc cmdline
  iter = model_attr_map->find(ATTR_MODEL_ATC_CMDLINE);
  auto cmdline = (iter != model_attr_map->end()) ? iter->second.s() : "";
  std::cout << "Original Atc command line: "
            << cmdline << std::endl
            << "system   info: "
            <<  ATTR_MODEL_ATC_VERSION
            << "[" << atc_version << "], "
            << "soc_version"
            << "[" << soc_version << "], "
            << "framework_type"
            << "[" << framework_type << "]." << std::endl;

  // resource info
  iter = model_attr_map->find(ATTR_MODEL_MEMORY_SIZE);
  auto memory_size = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  iter = model_attr_map->find(ATTR_MODEL_WEIGHT_SIZE);
  auto weight_size = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  iter = model_attr_map->find(ATTR_MODEL_STREAM_NUM);
  auto stream_num = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  iter = model_attr_map->find(ATTR_MODEL_EVENT_NUM);
  auto event_num = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  std::cout << "resource info: "
            << ATTR_MODEL_MEMORY_SIZE
            << "[" << memory_size << " B], "
            << ATTR_MODEL_WEIGHT_SIZE
            << "[" << weight_size << " B], "
            << ATTR_MODEL_STREAM_NUM
            << "[" << stream_num << "], "
            << ATTR_MODEL_EVENT_NUM
            << "[" << event_num << "]."
            << std::endl;

  // om info
  iter = model_attr_map->find("om_info_list");
  if (iter == model_attr_map->end()) {
    std::cout << "Display Model Info failed, attr \"om_info_list\" is not found in om, check the version is matched."
              << std::endl;
    std::cout << "============ Display Model Info end   ============"  << std::endl;
    return;
  }
  auto list_size = iter->second.list().i_size();
  if (list_size == kOmInfoSize) {
    std::cout << "om       info: "
              << "modeldef_size"
              << "[" << modeldef_size << " B], "
              << "weight_data_size"
              << "[" << iter->second.list().i(0) << " B], "
              << "tbe_kernels_size"
              << "[" << iter->second.list().i(1) << " B], "
              << "cust_aicpu_kernel_store_size"
              << "[" << iter->second.list().i(2) << " B], "
              << "task_info_size"
              << "[" << iter->second.list().i(kTaskInfoIndex) << " B], "
              << "so_store_size"
              << "[" << iter->second.list().i(kSoStoreIndex) << " B]." << std::endl;
  } else {
    std::cout << "Display Model Info error, please check!"  << std::endl;
  };

  std::cout << "============ Display Model Info end   ============"  << std::endl;
}

FMK_FUNC_HOST_VISIBILITY domi::Status ConvertOm(const char *model_file, const char *json_file, bool is_covert_to_json) {
  GE_CHECK_NOTNULL(model_file);
  // Mode 2 does not need to verify the priority, and a default value of 0 is passed
  // Load model from file
  ModelData model;
  GE_CHK_STATUS_RET_NOLOG(ModelParserBase::LoadFromFile(model_file, 0, model));

  GE_MAKE_GUARD(model_guard, [&model]() {
    if (model.model_data != nullptr) {
      delete[] static_cast<char *>(model.model_data);
      model.model_data = nullptr;
    }
  });

  try {
    // Parse the contents of the file to get the modeldef object
    Status ret;
    do {
      OmFileLoadHelper om_load_helper;
      ret = om_load_helper.Init(model);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Om file:%s init failed", model_file);
        GELOGE(ge::FAILED, "[Invoke][Init]Om file init failed.");
        break;
      }

      ModelPartition ir_part;
      ret = om_load_helper.GetModelPartition(MODEL_DEF, ir_part, 0U);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Get model part of om file:%s failed", model_file);
        GELOGE(ge::FAILED, "[Get][ModelPartition] failed.");
        break;
      }

      // De serialization
      ge::proto::ModelDef model_def;
      if (ReadProtoFromArray(ir_part.data, ir_part.size, &model_def)) {
        if (is_covert_to_json) {
          GE_CHECK_NOTNULL(json_file);
          GetGroupName(model_def);

          nlohmann::json j;
          Pb2Json::Message2Json(model_def, kOmBlackFields, j, true);
          ret = ModelSaver::SaveJsonToFile(json_file, j);
        } else {
          PrintModelInfo(&model_def, ir_part.size);
        }
      } else {
        ret = INTERNAL_ERROR;
        REPORT_INNER_ERR_MSG("E19999", "ReadProtoFromArray failed for om file:%s", model_file);
        GELOGE(ret, "[Read][Proto]From Array failed.");
      }
    } while (false);
    return ret;
  } catch (const std::exception &e) {
    REPORT_INNER_ERR_MSG("E19999", "Convert om model to json failed, exception message:%s, model_file:%s",
                       std::string(e.what()).c_str(), model_file);
    GELOGE(FAILED, "[Save][Model]Convert om model to json failed, exception message : %s.", e.what());
    return FAILED;
  }
}

FMK_FUNC_HOST_VISIBILITY domi::Status ConvertPbtxtToJson(const char *model_file, const char *json_file) {
  try {
    ge::proto::ModelDef model_def;
    const bool flag = GraphUtils::ReadProtoFromTextFile(model_file, &model_def);
    if (!flag) {
      REPORT_INNER_ERR_MSG("E19999", "ReadProtoFromTextFile failed for model_file:%s", model_file);
      GELOGE(FAILED, "[Invoke][ReadProtoFromTextFile] failed.");
      return FAILED;
    }
    GetGroupName(model_def);
    nlohmann::json j;
    Pb2Json::Message2Json(model_def, kOmBlackFields, j, true);
    auto ret = ModelSaver::SaveJsonToFile(json_file, j);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "SaveJsonToFile failed.");
      GELOGE(ret, "[Save][Json] to file fail.");
      return ret;
    }
    return SUCCESS;
  } catch (const std::exception &e) {
    REPORT_INNER_ERR_MSG("E19999", "ConvertPbtxtToJson failed, exception message:%s, model_file:%s",
                       std::string(e.what()).c_str(), model_file);
    GELOGE(FAILED, "[Save][pbtxt]Convert pbtxt to json failed, exception message : %s.", e.what());
    return FAILED;
  }
}

FMK_FUNC_HOST_VISIBILITY domi::Status ConvertFwkModelToJson(const domi::FrameworkType framework, const char *model_file,
                                                            const char *json_file) {
  if ((framework == domi::CAFFE) || (framework == domi::TENSORFLOW) || (framework == domi::ONNX)) {
    auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(framework);
    if (model_parser == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "CreateModelParser failed, framework:%d.", framework);
      GELOGE(FAILED, "[Create][ModelParser] ret fail, framework:%d.", framework);
      return FAILED;
    }
    return model_parser->ToJson(model_file, json_file);
  }

  REPORT_PREDEFINED_ERR_MSG(
      "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
      std::vector<const char *>(
          {"--framework", std::to_string(framework).c_str(),
           "The ramework must be selected from {0(Caffe), 3(TensorFlow), 5(Onnx)} when model is set to 1(JSON)."}));
  GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--framework] is mandatory "
         "and it's value must be: 0(Caffe) 3(TensorFlow) or 5(Onnx).");
  return PARAM_INVALID;
}

FMK_FUNC_HOST_VISIBILITY domi::Status DumpInfershapeJson(const ge::Graph &graph, const char *json_file) {
  // Create buffer
  GELOGI("Enter to dump infershape json schedule.");
  ge::Model model("", "");
  model.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  model.Save(buffer, true);

  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    std::string str(PtrToPtr<void, char>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      REPORT_INNER_ERR_MSG("E19999", "ParseFromString failed.");
      GELOGE(GRAPH_FAILED, "[Invoke][ParseFromString] failed.");
      return FAILED;
    }

    nlohmann::json j;
    Pb2Json::Message2Json(ge_proto, std::set<std::string>(), j);

    ModelSaver::SaveJsonToFile(json_file, j);
  }
  return SUCCESS;
}

void UpdateOmgCtxWithParserCtx() {
  domi::GetContext().format = GetParserContext().format;
  domi::GetContext().input_dims = GetParserContext().input_dims;
  domi::GetContext().user_input_dims = GetParserContext().user_input_dims;
  domi::GetContext().is_dynamic_input = GetParserContext().is_dynamic_input;
  domi::GetContext().type = GetParserContext().type;
  domi::GetContext().user_out_nodes = GetParserContext().user_out_nodes;
  domi::GetContext().train_flag = GetParserContext().train_flag;
  domi::GetContext().run_mode = GetParserContext().run_mode;
  domi::GetContext().op_conf_map = GetParserContext().op_conf_map;
  domi::GetContext().out_nodes_map = GetParserContext().out_nodes_map;
  domi::GetContext().final_out_nodes_map = GetParserContext().final_out_nodes_map;
  domi::GetContext().input_nodes_format_map = GetParserContext().input_nodes_format_map;
  domi::GetContext().out_tensor_names = GetParserContext().out_tensor_names;
  domi::GetContext().user_out_tensors = GetParserContext().user_out_tensors;
  domi::GetContext().default_out_nodes = GetParserContext().default_out_nodes;
  domi::GetContext().data_tensor_names = GetParserContext().data_tensor_names;
}

void UpdateParserCtxWithOmgCtx() {
  GetParserContext().format = domi::GetContext().format;
  GetParserContext().input_dims = domi::GetContext().input_dims;
  GetParserContext().user_input_dims = domi::GetContext().user_input_dims;
  GetParserContext().is_dynamic_input = domi::GetContext().is_dynamic_input;
  GetParserContext().type = domi::GetContext().type;
  GetParserContext().user_out_nodes = domi::GetContext().user_out_nodes;
  GetParserContext().train_flag = domi::GetContext().train_flag;
  GetParserContext().run_mode = domi::GetContext().run_mode;
  GetParserContext().op_conf_map = domi::GetContext().op_conf_map;
  GetParserContext().out_nodes_map = domi::GetContext().out_nodes_map;
  GetParserContext().final_out_nodes_map = domi::GetContext().final_out_nodes_map;
  GetParserContext().input_nodes_format_map = domi::GetContext().input_nodes_format_map;
  GetParserContext().out_tensor_names = domi::GetContext().out_tensor_names;
  GetParserContext().user_out_tensors = domi::GetContext().user_out_tensors;
  GetParserContext().data_tensor_names = domi::GetContext().data_tensor_names;
}
}  // namespace ge
