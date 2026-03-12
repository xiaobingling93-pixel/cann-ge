/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/hybrid_model.h"
#include <vector>
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
namespace hybrid {
namespace {
const int64_t kTensorSizeUnknownShape = -1; // Unknown shape mem size
bool IsRtV2Supported(const GeRootModelPtr &model) {
  static_cast<void>(model);
  return true;
}
bool IsRtV2Enabled() {
  // 离线推理从ACL控制入口，当前ACL不设置RT2环境变量默认走RT2的执行器，不会进入本文件RT1的执行流程
  // 为了兼容TF训练场景,这里如果不设置RT2环境变量，还是走RT1流程; 如果设置只有值为0才继续走RT1的流程，否则走RT2的执行流程
  const char_t *enable_rtv2 = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ENABLE_RUNTIME_V2, enable_rtv2);
  if ((enable_rtv2 != nullptr) && (enable_rtv2[0] == '0')) {
    return false;
  }
  return true;
}
}

HybridModel::HybridModel(GeRootModelPtr ge_model) : ge_root_model_(std::move(ge_model)) {
}

HybridModel::~HybridModel() {
  for (const auto &it : node_items_) {
    if (it.second != nullptr) {
      it.second->kernel_task.reset();
    }
  }
  GELOGD("[%s] HybridModel destroyed.", model_name_.c_str());
}

Status HybridModel::Init(const bool is_single_op) {
  if (!is_single_op) {
    PrintDynamicType();
  }
  const bool gert_enabled = IsRtV2Enabled();
  GELOGI("Start to init hybrid model %u, gert enabled: %d, single op: %d", model_id_, gert_enabled, is_single_op);
  execute_by_rt_v2_ = (gert_enabled && IsRtV2Supported(ge_root_model_) && (!is_single_op));
  if (!GetContext().GetHostExecFlag()) {
    GE_ASSERT_SUCCESS(ModelHelper::InitRuntimePlatform());
  }
  if (execute_by_rt_v2_) {
    // 这里赋值是为了HybridModelAsyncExecutor中CheckBlockingOp
    root_graph_ = ge_root_model_->GetRootGraph();
    GELOGI("Succeed init hybrid model %u that will be executed by runtime v2", model_id_);
    return SUCCESS;
  }
  if (!GetContext().GetHostExecFlag()) {
    ModelHelper model_helper;
    GE_CHK_STATUS_RET(model_helper.HandleDeviceInfo(platform_infos_), "Fail to handle device info");
  }
  is_single_op_ = is_single_op;
  if (is_single_op) {
    GE_CHK_STATUS_RET(HybridModelBuilder(*this).BuildForSingleOp(), "[Build][HybridModel] for SingleOp failed.");
  } else {
    GE_CHK_STATUS_RET(HybridModelBuilder(*this).Build(), "[Build][HybridModel] failed.");
  }
  SaveSpecifyAttrValues();
  GELOGD("HybridModel initialized successfully.");
  return SUCCESS;
}

TensorValue *HybridModel::GetVariable(const std::string &name) const {
  const auto it = variable_tensors_.find(name);
  if (it == variable_tensors_.end()) {
    GELOGD("Failed to get variable tensor. var name = [%s]", name.c_str());
    return nullptr;
  }

  GELOGD("Got variable tensor. var name = [%s], tensor = %s", name.c_str(), it->second->DebugString().c_str());
  return it->second.get();
}

NodePtr HybridModel::GetVariableNode(const std::string &name) const {
  const auto it = device_variable_nodes_.find(name);
  if (it != device_variable_nodes_.end()) {
    return it->second;
  }
  const auto host_find = host_variable_nodes_.find(name);
  if (host_find != host_variable_nodes_.end()) {
    return host_find->second;
  }
  GELOGD("Failed to get variable node by name = [%s]", name.c_str());
  return nullptr;
}

const std::vector<domi::TaskDef> *HybridModel::GetTaskDefs(const NodePtr &node) const {
  const auto it = task_defs_.find(node);
  if (it == task_defs_.end()) {
    return nullptr;
  }

  return &it->second;
}

NodeItem *HybridModel::MutableNodeItem(const NodePtr &node) const {
  const auto it = node_items_.find(node);
  if (it == node_items_.end()) {
    return nullptr;
  }

  return it->second.get();
}

const NodeItem *HybridModel::GetNodeItem(const NodePtr &node) const {
  const auto it = node_items_.find(node);
  if (it == node_items_.end()) {
    return nullptr;
  }

  return it->second.get();
}

GeModelPtr HybridModel::GetGeModel(const NodePtr &node) const {
  const auto it = known_shape_sub_models_.find(node);
  if (it == known_shape_sub_models_.end()) {
    GELOGE(INTERNAL_ERROR, "[Check][Param:node][%s(%s)] Failed to get GeModel for subgraph node,"
           "because node not in known_shape_sub_models_.", node->GetName().c_str(), node->GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) Failed to get GeModel for subgraph node,"
                       "because node not in known_shape_sub_models_.",
                       node->GetName().c_str(), node->GetType().c_str());
    return nullptr;
  }

  return it->second;
}

const GraphItem *HybridModel::GetRootGraphItem() const {
  return root_graph_item_.get();
}

const ComputeGraphPtr &HybridModel::GetRootGraph() const {
  return root_graph_;
}

const GraphItem *HybridModel::GetSubgraphItem(const std::string &graph_name) const {
  GELOGD("To find subgraph item by name = %s", graph_name.c_str());
  const auto it = subgraph_items_.find(graph_name);
  if (it == subgraph_items_.end()) {
    GELOGD("Subgraph item not found by node = %s", graph_name.c_str());
    return nullptr;
  }

  return it->second.get();
}

const GraphItem *HybridModel::GetSubgraphItem(const ComputeGraphPtr &subgraph) const {
  if (subgraph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Input param subgraph is nullptr, Graph:%s",
                       root_graph_item_->GetName().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]subgraph is nullptr. graph:%s",
           root_graph_item_->GetName().c_str());
    return nullptr;
  }

  const auto subgraph_name = subgraph->GetName();
  return GetSubgraphItem(subgraph_name);
}

const std::string &HybridModel::GetModelName() const {
  return model_name_;
}

Status HybridModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const {
  // dynamic shape do not need dynamic batch
  batch_info = {};
  dynamic_type = -1;
  return SUCCESS;
}

void HybridModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const {
  // dynamic shape do not need dynamic batch
  user_input_shape_order = {};
}

void HybridModel::GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) const {
  dynamic_output_shape_info = {};
}

Status HybridModel::GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                           std::vector<InputOutputDescInfo> &output_desc,
                                           std::vector<uint32_t> &input_formats,
                                           std::vector<uint32_t> &output_formats) {
  const auto node_item_list = root_graph_item_->GetInputNodes();
  if (!node_item_list.empty()) {
    GE_CHECK_NOTNULL(node_item_list[0U]->node);
    GE_CHECK_NOTNULL(node_item_list[0U]->node->GetOpDesc());
    if (node_item_list[0U]->node->GetOpDesc()->GetInputsSize() != 1U) {
      REPORT_INNER_ERR_MSG("E19999", "Input size of op is not 1, op:%s, type:%s",
                         node_item_list[0U]->node->GetName().c_str(),
                         node_item_list[0U]->node->GetType().c_str());
      GELOGE(FAILED, "[Check][Size]input size of op is not 1! op:%s, type:%s",
             node_item_list[0U]->node->GetName().c_str(),
             node_item_list[0U]->node->GetType().c_str());
      return FAILED;
    }
    GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "[Get][InputDescInfo] failed.");
  }

  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats), "[Get][OutputDescInfo] failed.");

  return SUCCESS;
}

void HybridModel::SetInputDimsAndShapeRangesInfo(const std::vector<int64_t> &model_input_dims,
                                                 const std::vector<std::pair<int64_t, int64_t>> &shape_ranges,
                                                 InputOutputDescInfo &input) const {
  for (const auto model_input_dim : model_input_dims) {
    input.shape_info.dims.push_back(model_input_dim);
  }
  input.shape_info.shape_ranges = shape_ranges;
  return;
}

void HybridModel::CreateInputDimsInfo(const OpDescPtr &op_desc, InputOutputDescInfo &input) const {
  std::vector<std::pair<int64_t, int64_t>> shape_ranges;
  if (is_new_model_desc_ && op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    // When static aipp is set, need to get the model input dims which processed by aipp
    std::vector<int64_t> model_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    SetInputDimsAndShapeRangesInfo(model_input_dims, shape_ranges, input);
    return;
  }
  // judge if this data is linked dynamic aipp first, multiply batch has been considered
  if (op_desc->HasAttr("_dynamic_aipp_input_dims")) {
    std::vector<int64_t> dynamic_aipp_input_dims;
    (void)AttrUtils::GetListInt(op_desc, "_dynamic_aipp_input_dims", dynamic_aipp_input_dims);
    (void)op_desc->GetInputDescPtr(0U)->GetShapeRange(shape_ranges);
    SetInputDimsAndShapeRangesInfo(dynamic_aipp_input_dims, shape_ranges, input);
    return;
  } else {
    const std::vector<int64_t> input_dims = op_desc->GetInputDescPtr(0U)->GetShape().GetDims();
    (void)op_desc->GetInputDescPtr(0U)->GetShapeRange(shape_ranges);
    SetInputDimsAndShapeRangesInfo(input_dims, shape_ranges, input);
    return;
  }
}

Status HybridModel::GetInputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                     std::vector<uint32_t> &formats_result) {
  const auto node_item_list = root_graph_item_->GetInputNodes();
  for (auto &node_item : node_item_list) {
    InputOutputDescInfo input;

    GE_CHECK_NOTNULL(node_item->node);
    const auto op_desc = node_item->node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(0U));

    const Format format = op_desc->GetInputDescPtr(0U)->GetFormat();
    const DataType data_type = op_desc->GetInputDescPtr(0U)->GetDataType();
    input.data_type = static_cast<uint32_t>(data_type);
    input.name = op_desc->GetName();
    const GeShape shape = op_desc->GetInputDescPtr(0U)->GetShape();
    int64_t tensor_size = 0;
    if (TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[Calculate][TensorMemSize] failed input0 desc in node:%s(%s)."
             "shape:%s, format:%s, datatype:%s.", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             shape.ToString().c_str(), TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      REPORT_INNER_ERR_MSG("E19999", "CalcTensorMemSize failed for input0 desc in node:%s(%s),"
                        "shape:%s, format:%s, datatype:%s", op_desc->GetName().c_str(), op_desc->GetName().c_str(),
                        shape.ToString().c_str(), TypeUtils::FormatToSerialString(format).c_str(),
                        TypeUtils::DataTypeToSerialString(data_type).c_str());
      return FAILED;
    }
    if (tensor_size == kTensorSizeUnknownShape) {
      tensor_size = 0;
    }
    input.size = static_cast<uint64_t>(tensor_size);
    CreateInputDimsInfo(op_desc, input);

    formats_result.push_back(format);
    input_desc.push_back(input);
  }
  is_new_model_desc_ = false;
  return SUCCESS;
}

void HybridModel::CreateOutput(const ConstGeTensorDescPtr &output_desc,
                               InputOutputDescInfo &output_desc_info, uint32_t &format_result) const {
  GE_IF_BOOL_EXEC(output_desc == nullptr,
      REPORT_INNER_ERR_MSG("E19999", "param output_desc is nullptr, check invalid.");
      GELOGE(FAILED, "[Check][Param:output_desc]output desc ptr is nullptr");
      return);
  const Format format = output_desc->GetFormat();
  const GeShape shape = output_desc->GetShape();
  std::vector<std::pair<int64_t, int64_t>> shape_ranges;
  (void)output_desc->GetShapeRange(shape_ranges);
  const DataType data_type = output_desc->GetDataType();
  format_result = format;
  if (format == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    const int64_t k = shape.GetDim(0U);                                           // 0: first dim
    const int64_t c = shape.GetDim(1U);                                           // 1: second dim
    const int64_t h = shape.GetDim(2U);                                           // 2: third dim
    const int64_t w = shape.GetDim(3U);                                           // 3: forth dim
    output_desc_info.shape_info.dims.push_back(h);
    output_desc_info.shape_info.dims.push_back(w);
    output_desc_info.shape_info.dims.push_back(c);
    output_desc_info.shape_info.dims.push_back(k);
    if (shape_ranges.size() == 4U) {                   // 4 dims
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[2U]);  // h:2
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[3U]);  // w:3
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[1U]);  // c:1
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[0U]);  // k:0
    }
    format_result = FORMAT_HWCN;
  } else {
    for (size_t j = 0U; j < shape.GetDimNum(); j++) {
      output_desc_info.shape_info.dims.push_back(shape.GetDim(j));
    }
    output_desc_info.shape_info.shape_ranges = shape_ranges;
  }
  int64_t tensor_size = 0;
  (void)TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  if (tensor_size == kTensorSizeUnknownShape) {
    tensor_size = 0;
  }
  output_desc_info.size = static_cast<uint64_t>(tensor_size);
  output_desc_info.data_type = output_desc->GetDataType();
}

Status HybridModel::GetOutputDescInfo(std::vector<InputOutputDescInfo> &output_desc,
                                      std::vector<uint32_t> &formats_result) const {
  std::vector<ConstGeTensorDescPtr> output_desc_list;
  // output_desc_list contains vaild input desc
  GE_CHK_STATUS_RET(root_graph_item_->GetOutputDescList(output_desc_list),
                    "[Invoke][GetOutputDescList]get output desc info failed, Graph:%s",
                    root_graph_item_->GetName().c_str());

  std::vector<std::string> out_node_names;
  (void)ge::AttrUtils::GetListStr(ge_root_model_->GetRootGraph(), ATTR_MODEL_OUT_NODES_NAME, out_node_names);

  GE_CHECK_NOTNULL(root_graph_item_->GetOutputNode());
  const auto op_desc = root_graph_item_->GetOutputNode()->op_desc;
  GE_CHECK_NOTNULL(op_desc);

  const auto out_size = static_cast<uint32_t>(op_desc->GetInputsSize());
  GE_IF_BOOL_EXEC(out_size != output_desc_list.size(),
                  REPORT_INNER_ERR_MSG("E19999", "output size[%u] not match output_desc_list size[%zu]",
                                     out_size, output_desc_list.size());
                  GELOGE(FAILED, "[Check][Size]output size[%u] not match output_desc_list size[%zu]",
                         out_size, output_desc_list.size());
                  return FAILED;);

  for (uint32_t index = 0U; index < out_size; ++index) {
    std::string output_name;
    const std::vector<std::string> src_name = op_desc->GetSrcName();
    const std::vector<int64_t> src_index = op_desc->GetSrcIndex();
    if (out_size == out_node_names.size()) {
      const bool contains_colon = out_node_names[static_cast<size_t>(index)].find(":") != std::string::npos;
      output_name = contains_colon ? out_node_names[static_cast<size_t>(index)] :
                                     out_node_names[static_cast<size_t>(index)] +
                                     ":" + std::to_string(src_index[static_cast<size_t>(index)]);
    } else {
      output_name = std::string("output_") + std::to_string(index) + "_" + src_name[static_cast<size_t>(index)] +
          "_" + std::to_string(src_index[static_cast<size_t>(index)]);
    }

    InputOutputDescInfo output_desc_info;
    output_desc_info.name = output_name;

    uint32_t format_result;
    CreateOutput(output_desc_list[static_cast<size_t>(index)], output_desc_info, format_result);
    output_desc.push_back(output_desc_info);
    formats_result.push_back(format_result);
  }
  return SUCCESS;
}

TensorValue *HybridModel::GetConstant(const NodePtr &node) const {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param:node]node is null.");
    REPORT_INNER_ERR_MSG("E19999", "param node is null, check invalid.");
    return nullptr;
  }

  const auto it = constant_tensors_.find(node);
  if (it == constant_tensors_.end()) {
    GELOGD("constant not found, node name = [%s]", node->GetName().c_str());
    return nullptr;
  }

  GELOGD("Got constant tensor, node name = [%s], tensor = %s",
         node->GetName().c_str(),
         it->second->DebugString().c_str());
  return it->second.get();
}

TensorValue *HybridModel::GetTensor(const NodePtr &node) const {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param:node]node is null.");
    REPORT_INNER_ERR_MSG("E19999", "param node is null, check invalid.");
    return nullptr;
  }

  if (node->GetType() == CONSTANT) {
    return GetConstant(node);
  }

  return GetVariable(node->GetName());
}

const std::map<int64_t, std::vector<std::pair<int32_t, GeTensorPtr>>> &HybridModel::GetHostTensors() const {
  return host_tensors_;
}

void *HybridModel::GetOverflowAddr() const {
  if (globalworkspace_overflow_addr_ == nullptr) {
    return nullptr;
  }
  return globalworkspace_overflow_addr_->GetData();
}

Status HybridModel::SetOverflowAddr(void *const buffer, const size_t size) {
  globalworkspace_overflow_addr_ = TensorBuffer::Create(buffer, size);
  GE_CHECK_NOTNULL(globalworkspace_overflow_addr_);
  return SUCCESS;
}

void *HybridModel::GetGlobalStep() const {
  if (global_step_ == nullptr) {
    return nullptr;
  }
  return global_step_->GetData();
}

TensorBuffer *HybridModel::GetModelWeight(const std::string &subgraph_name) const {
  const auto it = weight_buffer_map_.find(subgraph_name);
  if (it == weight_buffer_map_.end()) {
    GELOGD("Model weight not found, subgraph name = %s", subgraph_name.c_str());
    return nullptr;
  }
  return it->second.get();
}

// save specify attr values of op, such as ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
// it will save more attr values in the future
void HybridModel::SaveSpecifyAttrValues() {
  for (const auto &node : root_graph_->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    const auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    std::vector<std::string> value;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value)) {
      std::map<std::string, std::vector<std::string>> attr_name_to_value;
      attr_name_to_value[ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES] = value;
      op_name_to_attrs_[op_desc->GetName()] = attr_name_to_value;
      GELOGD("Get op:%s attr:%s success.", op_desc->GetName().c_str(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str());
    }
  }
  return;
}
Status HybridModel::GetOpAttr(const std::string &op_name, const std::string &attr_name,
                              std::string &attr_value) const {
  const auto itr = op_name_to_attrs_.find(op_name);
  if (itr == op_name_to_attrs_.end()) {
    GELOGW("Did not save op:%s attr", op_name.c_str());
    return SUCCESS;
  }
  const auto attr_itr = itr->second.find(attr_name);
  if (attr_itr == itr->second.end()) {
    GELOGW("Did not save attr:%s of op:%s", attr_name.c_str(), op_name.c_str());
    return SUCCESS;
  }
  for (const auto &name : attr_itr->second) {
    attr_value += "[" + std::to_string(name.size()) + "]" + name;
  }
  GELOGD("Get attr:%s of op:%s success, attr value:%s", attr_name.c_str(), op_name.c_str(), attr_value.c_str());
  return SUCCESS;
}

Status HybridModel::InitAippInfoAndType() {
  uint32_t data_index = 0U;
  std::map<std::string, uint32_t> data_index_map;
  for (const auto &data_node_item : GetRootGraphItem()->GetInputNodes()) {
    data_index_map[data_node_item->node_name] = data_index;
    data_index++;
  }
  data_index = 0U;
  for (const auto &data_node_item : GetRootGraphItem()->GetInputNodes()) {
    const auto ret = AippUtils::SetAippInfoAndTypeFromOpDesc(data_index_map,
                                                             data_node_item->node->GetOpDesc(), data_index,
                                                             aipp_infos_, aipp_types_);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Set][AippInfoAndType]index:%u, node:%s.", data_index, data_node_item->node_name.c_str());
      return INTERNAL_ERROR;
    }
    ++data_index;
  }
  return SUCCESS;
}

void HybridModel::PrintDynamicType() const {
  const auto &root_graph = ge_root_model_->GetRootGraph();
  bool is_dynamic_input = false;
  bool is_dynamic_in_process = false;
  for (const auto &node : root_graph->GetDirectNode()) {
    bool is_unknown_node = false;
    (void)NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_node);
    if (is_unknown_node) {
      if (OpTypeUtils::IsDataNode(node->GetType())) {
        is_dynamic_input = true;
        break;
      } else {
        is_dynamic_in_process = true;
      }
    }
  }
  if (is_dynamic_input) {
    GEEVENT("current execute mode is dynamic input, model id = %u, model_name = \"%s\".", model_id_,
            model_name_.c_str());
  } else {
    if (is_dynamic_in_process) {
      GEEVENT("current execute mode is dynamic in process, model id = %u, model_name = \"%s\".", model_id_,
              model_name_.c_str());
    }
  }
}
Status HybridModel::GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const {
  return AippUtils::GetAippInfo(aipp_infos_, index, aipp_info);
}

Status HybridModel::GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_data_index) const {
  return AippUtils::GetAippType(aipp_types_, index, aipp_type, aipp_data_index);
}

bool HybridModel::CheckHostMemInputOptimization(const std::vector<NodePtr> &node_with_hostmem) const {
  if (node_with_hostmem.empty()) {
    GELOGD("NOT support host memory input optimization because there is no host memory input node.");
    return false;
  }

  for (auto &node_ptr : node_with_hostmem) {
    const ge::hybrid::NodeItem *const node_item_ptr = GetNodeItem(node_ptr);
    if ((node_item_ptr == nullptr) || (node_item_ptr->kernel_task == nullptr)) {
      return false;
    }
    if ((!node_item_ptr->kernel_task->IsArgsExtendedForHostMemInput())) {
      GELOGD("kernel_task[%s] args are not extended for host memory input optimization",
             node_ptr->GetName().c_str());
      return false;
    }
    if ((!node_item_ptr->kernel_task->IsSupportHostMemInputOpt())) {
      GELOGD("kernel_task[%s] not support host memory input optimization",
             node_ptr->GetName().c_str());
      return false;
    }
  }

  GELOGD("CheckHostMemInputOptimization return true.");
  return true;
}

void HybridModel::SetNeedHostMemOpt(const std::vector<NodePtr> &node_with_hostmem, const bool need_host_mem_opt) const {
  for (auto &node_ptr : node_with_hostmem) {
    const ge::hybrid::NodeItem *const node_item_ptr = MutableNodeItem(node_ptr);
    if ((node_item_ptr != nullptr) && (node_item_ptr->kernel_task != nullptr)) {
      node_item_ptr->kernel_task->SetNeedHostMemOpt(need_host_mem_opt);
    }
  }
}

Status HybridModel::ReportProfilingData() const {
  for (const auto &it : node_items_) {
    auto &node_item = it.second;
    GE_CHECK_NOTNULL(node_item->node_executor);
    (void)node_item->node_executor->ReportProfilingData(*node_item);
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
