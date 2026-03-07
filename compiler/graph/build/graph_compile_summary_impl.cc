/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_compile_summary_impl.h"

#include <memory>
#include <set>
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/memory/external_weight_desc_impl.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/operator_reg.h"
#include "framework/common/types.h"
#include "graph/utils/type_utils.h"
#include "stream/stream_info.h"
#include "graph/utils/node_adapter.h"
#include "graph/build/stream/stream_utils.h"

namespace ge {
namespace {
constexpr uint32_t kDataIndex = 0U;
constexpr uint32_t kGetDynamicDimsCount = 1U;
}  // namespace

CompiledGraphSummary::~CompiledGraphSummary() = default;
ExternalWeightDesc::~ExternalWeightDesc() = default;

bool CompiledGraphSummary::IsStatic() const {
  return data_->IsStatic();
}

Status CompiledGraphSummary::GetConstMemorySize(size_t &size) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get const memory size!");
    size = 0ULL;
    return FAILED;
  }
  size = data_->GetConstMemorySize();
  return SUCCESS;
}

Status CompiledGraphSummary::GetFeatureMemorySize(size_t &size) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get feature memory size!");
    size = 0ULL;
    return FAILED;
  }
  size = data_->GetFeatureMemorySize();
  return SUCCESS;
}

Status CompiledGraphSummary::GetFixedFeatureMemorySize(size_t &size) const {
  // 这个接口支持纯静态图以及静态子图场景
  size = data_->GetFixedFeatureMemorySize();
  return SUCCESS;
}

std::vector<FeatureMemoryPtr> CompiledGraphSummary::GetAllFeatureMemoryTypeSize() const {
  // 这个接口支持纯静态图以及静态子图场景
  return data_->GetAllFeatureMemoryTypeSize();
}

Status CompiledGraphSummary::GetRefreshableFeatureMemorySize(size_t &size) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get feature memory size!");
    size = 0ULL;
    return FAILED;
  }
  size = data_->GetRefreshableFeatureMemorySize();
  return SUCCESS;
}

Status CompiledGraphSummary::GetFeatureMemoryBaseRefreshable (bool &v) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get feature memory refreshable!");
    v = false;
    return FAILED;
  }
  v = data_->IsFeatureMemoryBaseRefreshable();
  return SUCCESS;
}

Status CompiledGraphSummary::GetStreamNum(size_t &num) const {
  num = data_->GetStreamNum();
  return SUCCESS;
}

Status CompiledGraphSummary::GetEventNum(size_t &num) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get event number!");
    num = 0ULL;
    return FAILED;
  }
  num = data_->GetEventNum();
  return SUCCESS;
}

Status CompiledGraphSummary::GetOutputShapes(std::vector<ge::Shape> &shapes) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get output shapes!");
    return FAILED;
  }
  shapes = data_->GetOutputShapes();
  return SUCCESS;
}

Status CompiledGraphSummary::GetOutputDtypes(std::vector<ge::DataType> &dtypes) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get output dtypes!");
    return FAILED;
  }
  dtypes = data_->GetOutputDtypes();
  return SUCCESS;
}

Status CompiledGraphSummary::GetIOIndexesWithSameAddr(std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) const {
  if (!data_->IsStatic()) {
    GELOGW("Compiled graph is not static, can not get IO indexes with same address!");
    return FAILED;
  }
  io_indexes = data_->GetIOIndexesWithSameAddr();
  return SUCCESS;
}

Status CompiledGraphSummary::GetExternalWeightPaths(std::vector<ExternalWeightDescPtr> &paths) const {
  paths = data_->GetExternalWeightPaths();
  return SUCCESS;
}

CompiledGraphSummaryPtr CompiledGraphSummary::Builder::Build(const GeModelPtr &ge_model,
                                                             const GeRootModelPtr &ge_root_model) {
  CompiledGraphSummaryPtr summary(new (std::nothrow) CompiledGraphSummary());
  GE_ASSERT_NOTNULL(summary);

  summary->data_ = MakeShared<CompiledGraphSummary::SummaryData>();
  summary->data_->stream_allocation_summary_ = MakeShared<StreamAllocationSummary>();
  GE_ASSERT_NOTNULL(summary->data_);
  GE_ASSERT_NOTNULL(summary->data_->stream_allocation_summary_);
  GE_ASSERT_NOTNULL(summary->data_->stream_allocation_summary_->impl_);
  GE_ASSERT_SUCCESS(summary->data_->stream_allocation_summary_->impl_->CollectStreamInfos(ge_root_model));
  const auto &root_graph = ge_root_model->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  bool is_dsp_partitioned_graph = false;
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dsp_partitioned_graph);
  summary->data_->is_static_ = !(root_graph->GetGraphUnknownFlag() || is_dsp_partitioned_graph);
  // 支持动态图中静态子图获取对应的fix地址大小数据
  GE_ASSERT_SUCCESS(summary->data_->SetFixedFeatureMemorySize(ge_model, ge_root_model));
  GE_ASSERT_SUCCESS(summary->data_->SetStreamNum(ge_model));
  if (!summary->data_->is_static_) {
    GELOGI("Dynamic graph summary build success, graph_name:%s, model_name:%s",
           root_graph->GetName().c_str(), ge_model->GetName().c_str());
    return summary;
  }
  GE_ASSERT_SUCCESS(summary->data_->SetConstMemorySize(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetFeatureMemorySize(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetRefreshablFeatureMemorySize(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetFeatureMemoryBaseRefreshable(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetEventNum(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetOutputTensorInfo(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetIOIndexesWithSameAddr(ge_model));
  GE_ASSERT_SUCCESS(summary->data_->SetExternalWeightPaths(ge_model));
  GELOGI("Static graph summary build success, graph_name:%s, model_name:%s",
         root_graph->GetName().c_str(), ge_model->GetName().c_str());
  return summary;
}

std::vector<FeatureMemoryPtr> CompiledGraphSummary::SummaryData::GetAllFeatureMemoryTypeSize() const {
  return feature_memory_;
}

Status CompiledGraphSummary::SummaryData::SetConstMemorySize(const GeModelPtr &ge_model) {
  size_t ori_const_mem_size = 0UL;
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, ori_const_mem_size));
  const_mem_size_ = (ori_const_mem_size == 0UL) ? 512UL : ori_const_mem_size;
  GELOGI("model_name:%s const_mem_size:%zu, ori_const_mem_size:%zu.", ge_model->GetName().c_str(), const_mem_size_,
         ori_const_mem_size);
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetFeatureMemorySize(const GeModelPtr &ge_model) {
  size_t feature_size_with_input_output = 0UL;
  size_t input_output_size = 0UL;
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, feature_size_with_input_output));
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, input_output_size));
  GE_ASSERT(feature_size_with_input_output >= input_output_size,
            "feature_size_with_input_output:%zu, input_output_size:%zu",
            feature_size_with_input_output, input_output_size);
  // 临时方案：feature_mem_size为0的情况下，app不会设置fm地址，davinci_model会申请“包含zero_copy_mem_size的fm内存”；对于执行态给
  // device内存的场景，GE不需要申请“包含zero_copy_mem_size的fm内存”，考虑修改davinci_model影响性大，采用以下临时方案：
  // const mem size 或 fm mem size为0时，给外部返回512
  const size_t ori_feature_mem_size = feature_size_with_input_output - input_output_size;
  feature_mem_size_ = (ori_feature_mem_size == 0UL) ? 512UL : ori_feature_mem_size;
  GELOGI("model_name:%s ori_feature_size:%zu, input_output_size:%zu, total_size:%zu, feature_size:%zu.",
         ge_model->GetName().c_str(), ori_feature_mem_size, input_output_size, feature_size_with_input_output,
         feature_mem_size_);
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetFixedFeatureMemorySize(const GeModelPtr &ge_model,
                                                                    const GeRootModelPtr &ge_root_model) {
  (void)ge_model;
  GE_ASSERT_SUCCESS(ge_root_model->GetSummaryFeatureMemory(feature_memory_, fixed_feature_mem_size_));
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetRefreshablFeatureMemorySize(const GeModelPtr &ge_model) {
  size_t feature_size_with_input_output = 0UL;
  size_t input_output_size = 0UL;
  size_t fixed_mem_size = 0UL;
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, feature_size_with_input_output));
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, input_output_size));
  std::vector<std::vector<int64_t>> sub_mem_infos;
  (void)AttrUtils::GetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);
  for (size_t index = 0UL; index < sub_mem_infos.size(); ++index) {
    const auto &sub_memory_info = sub_mem_infos[index];
    // 0U: memory_type, 1U:logic_memory_base, 2U:memory_size, 3U:is_fixed_addr_prior
    bool is_fixed_addr_prior = (sub_memory_info.size() > 3U) ? (sub_memory_info[3U] != 0UL) : false;
    // 2U:memory_size, is_fixed_addr_prior is true set memory size to fixed size
    fixed_mem_size += (is_fixed_addr_prior ? sub_memory_info[2U] : 0UL);
  }
  GE_ASSERT(feature_size_with_input_output >= (input_output_size + fixed_mem_size),
            "feature_size_with_input_output:%zu, input_output_size:%zu, fixed_mem_size:%zu",
            feature_size_with_input_output, input_output_size, fixed_mem_size);
  // 临时方案：feature_mem_size为0的情况下，app不会设置fm地址，davinci_model会申请“包含zero_copy_mem_size的fm内存”；对于执行态给
  // device内存的场景，GE不需要申请“包含zero_copy_mem_size的fm内存”，考虑修改davinci_model影响性大，采用以下临时方案：
  // const mem size 或 fm mem size为0时，给外部返回512
  const size_t ori_feature_mem_size = feature_size_with_input_output - input_output_size - fixed_mem_size;
  refreshable_feature_mem_size_ = (ori_feature_mem_size == 0UL) ? 512UL : ori_feature_mem_size;
  GELOGI("model_name:%s ori_feature_size:%zu, input_output_size:%zu, fixed_mem_size:%zu,"
         " total_size:%zu, feature_size:%zu.",
         ge_model->GetName().c_str(), ori_feature_mem_size, input_output_size, fixed_mem_size,
         feature_size_with_input_output, refreshable_feature_mem_size_);
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetFeatureMemoryBaseRefreshable(const GeModelPtr &ge_model) {
  (void)ge_model;
  std::string is_refreshable;
  (void)GetContext().GetOption(OPTION_FEATURE_BASE_REFRESHABLE, is_refreshable);
  static const std::string kEnabled = "1";
  is_feature_mem_refreshable_ = (is_refreshable == kEnabled);
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetStreamNum(const GeModelPtr &ge_model) {
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_STREAM_NUM, stream_num_));
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetEventNum(const GeModelPtr &ge_model) {
  GE_ASSERT(AttrUtils::GetInt(ge_model, ATTR_MODEL_EVENT_NUM, event_num_));
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetOutputTensorInfo(const GeModelPtr &ge_model) {
  const auto compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == NETOUTPUT) {
      const auto &tensor_descs = node->GetOpDesc()->GetAllInputsDescPtr();
      uint32_t id = 0U;
      bool is_no_tiling = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_OP_NO_TILING, is_no_tiling);
      for (const auto &tensor_desc : tensor_descs) {
        GE_ASSERT_NOTNULL(tensor_desc);
        std::vector<int64_t> output_dims;
        if ((is_no_tiling) && (tensor_desc->GetShape().IsUnknownShape())) {
          // no tiling场景，此场景输出shape不受输入shape影响，虽然输入shape存在-1，但是依然可以下沉
          // 当前子图分档场景下，netoutput是notiling的，他的输入shape存在-1，需要适配成获取shaperange中的
          // max_shape给用户
          std::vector<std::pair<int64_t, int64_t>> shape_ranges;
          GE_ASSERT_SUCCESS(tensor_desc->GetShapeRange(shape_ranges));
          for (const auto &range : shape_ranges) {
            output_dims.emplace_back(range.second);
          }
          GE_ASSERT_TRUE(!output_dims.empty(),
              "No tiling node: %s shape is unknown, but has no shape range.", node->GetNamePtr());
        } else {
          output_dims = tensor_desc->GetShape().GetDims();
        }
        GELOGD("Node:%s input[%u] shape:%s datatype:%s", node->GetName().c_str(), id++,
                GeShape(output_dims).ToString().c_str(),
                TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str());
        netoutput_shapes_.emplace_back(Shape(output_dims));
        netoutput_dtypes_.emplace_back(tensor_desc->GetDataType());
      }
    }
  }
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::ConstructIoOffsetToRoleToIndex(const ComputeGraphPtr &compute_graph,
  std::map<int64_t, map<int32_t, std::vector<uint32_t>>> &io_offset_to_role_to_index) const {
  uint32_t graph_id = compute_graph->GetGraphID();
  uint32_t data_op_index = 0U;
  uint32_t input_index = 0U;
  uint32_t output_index = 0U;
  for (const auto &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      input_index = data_op_index;
      if (AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, input_index)) {
        GELOGD("graph_id:%u, new input index:%u, old input index:%u", graph_id, input_index, data_op_index);
      }

      const auto &input_offset = op_desc->GetOutputOffset();
      if (input_offset.empty()) {
        GELOGW("graph_id:%u op index:%u input offset is empty", graph_id, input_index);
        data_op_index++;
        continue;
      }

      io_offset_to_role_to_index[input_offset[kDataIndex]][static_cast<int32_t>(ModelIORole::KInput)]
        .push_back(input_index);
      GELOGD("graph_id:%u node:%s input index:%d offset:%ld",
        graph_id, node->GetName().c_str(), input_index, input_offset[kDataIndex]);
      data_op_index++;
    } else if (node->GetType() == NETOUTPUT) {
      const auto &output_offsets = op_desc->GetInputOffset();
      if (output_offsets.empty()) {
        GELOGW("graph_id:%u netoutput[%s] input offset is empty", graph_id, node->GetName().c_str());
        continue;
      }

      size_t actual_output_size = output_offsets.size();
      bool getnext_sink_dynamic = false;
      if (AttrUtils::GetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, getnext_sink_dynamic) && getnext_sink_dynamic) {
        actual_output_size -= kGetDynamicDimsCount;
        GELOGI("graph_id:%u ATTR_GETNEXT_SINK_DYNMAIC has been set and is true, node: %s, output size:%zu",
          graph_id, op_desc->GetName().c_str(), actual_output_size);
      }

      for (size_t i = 0U; i < actual_output_size; i++) {
        io_offset_to_role_to_index[output_offsets[i]][static_cast<int32_t>(ModelIORole::KOutput)]
          .push_back(output_index);
        GELOGD("graph_id:%u node:%s output index:%d offset:%lld",
          graph_id, node->GetName().c_str(), output_index, output_offsets[i]);
        output_index++;
      }
    }
  }
  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetIOIndexesWithSameAddr(const GeModelPtr &ge_model) {
  const auto compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);

  std::map<int64_t, map<int32_t, std::vector<uint32_t>>> io_offset_to_role_to_index;
  GE_ASSERT_SUCCESS(ConstructIoOffsetToRoleToIndex(compute_graph, io_offset_to_role_to_index));

  uint32_t graph_id = compute_graph->GetGraphID();
  for (auto &it : io_offset_to_role_to_index) {
    const auto input_to_indexes = it.second.find(static_cast<int32_t>(ModelIORole::KInput));
    const auto output_to_indexes = it.second.find(static_cast<int32_t>(ModelIORole::KOutput));
    if ((input_to_indexes == it.second.end()) || (output_to_indexes == it.second.end())) {
        GELOGD("graph_id:%u offset %lld has no matching inputs and outputs with same addr", graph_id, it.first);
      continue;
    }

    if (input_to_indexes->second.size() != 1U) {
      GELOGW("graph_id:%u offset %lld input index list size %zu, more than one",
        graph_id, it.first, input_to_indexes->second.size());
      continue;
    }

    for (auto output_index : output_to_indexes->second) {
      io_indexes_with_same_addr_.emplace_back(std::make_pair(input_to_indexes->second[kDataIndex], output_index));
      GELOGI("graph_id:%u input index:%u output index:%u has the same offset:%lld",
        graph_id, input_to_indexes->second[kDataIndex], output_index, it.first);
    }
  }

  return SUCCESS;
}

Status CompiledGraphSummary::SummaryData::SetExternalWeightPaths(const GeModelPtr &ge_model) {
  const auto compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  std::map<std::string, std::string> file_id_to_path_map;
  GE_ASSERT_SUCCESS(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_to_path_map));
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == FILECONSTANT) {
      auto op_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      std::string file_path;
      size_t offset = 0U;
      size_t length = 0U;
      GE_ASSERT_SUCCESS(FileConstantUtils::GetFilePath(op_desc, file_id_to_path_map, file_path, offset, length));

      std::string file_id_temp;
      (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, file_id_temp);
      const auto file_id = ConvertToAscendString(file_id_temp);
      const auto file_path_ascend = ConvertToAscendString(file_path);
      ExternalWeightDescPtr external_weight_desc =
        ExternalWeightDesc::Builder::Build(file_path_ascend, length, offset, file_id);
      GE_ASSERT_NOTNULL(external_weight_desc);
      external_weight_paths_.emplace_back(std::move(external_weight_desc));
      GELOGI("file constant node:%s, path:%s, offset:%zu, length:%zu, file_id:%s", node->GetNamePtr(),
        file_path.c_str(), offset, length, file_id_temp.c_str());
    }
  } 
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetStringInfos(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<AscendString>> &graph_to_string_infos) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<AscendString> string_infos;
    for (const auto &stream_info : iter.second) {
      string_infos.emplace_back(stream_info.ToStringInfo());
    }
    graph_to_string_infos[iter.first] = string_infos;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetLogicalStreamIds(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<int64_t>> &graph_to_logical_stream_ids) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<int64_t> logical_stream_ids;
    for (const auto &stream_info : iter.second) {
      logical_stream_ids.emplace_back(stream_info.GetLogicalStreamId());
    }
    graph_to_logical_stream_ids[iter.first] = logical_stream_ids;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetUsrStreamLabels(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<AscendString>> &graph_to_user_stream_labels) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<AscendString> user_stream_labels;
    for (const auto &stream_info : iter.second) {
      user_stream_labels.emplace_back(stream_info.GetUsrStreamLabel());
    }
    graph_to_user_stream_labels[iter.first] = user_stream_labels;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryIsAssignedByStreamPass(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<bool>> &graph_to_is_assigned_by_stream_pass) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<bool> assigned_by_stream_passes;
    for (const auto &stream_info : iter.second) {
      assigned_by_stream_passes.emplace_back(stream_info.IsAssignedByStreamPass());
    }
    graph_to_is_assigned_by_stream_pass[iter.first] = assigned_by_stream_passes;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetAttachedStreamIds(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<std::vector<int64_t>>> &graph_to_attached_stream_ids) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<std::vector<int64_t>> attached_stream_ids;
    for (const auto &stream_info : iter.second) {
      attached_stream_ids.emplace_back(stream_info.GetAttachedStreamIds());
    }
    graph_to_attached_stream_ids[iter.first] = attached_stream_ids;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetPhysicalStreamNums(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<int64_t>> &graph_to_physical_stream_nums) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<int64_t> physical_stream_nums;
    for (const auto &stream_info : iter.second) {
      physical_stream_nums.emplace_back(stream_info.GetPhysicalStreamNum());
    }
    graph_to_physical_stream_nums[iter.first] = physical_stream_nums;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetHcclFollowedStreamNums(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<int64_t>> &graph_to_hccl_followed_stream_nums) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<int64_t> hccl_followed_stream_nums;
    for (const auto &stream_info : iter.second) {
      hccl_followed_stream_nums.emplace_back(stream_info.GetHcclFollowedStreamNum());
    }
    graph_to_hccl_followed_stream_nums[iter.first] = hccl_followed_stream_nums;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetAllNodes(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<std::vector<GNode>>> &graph_to_all_nodes) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  for (const auto &iter : stream_allocation->GetAllLogicalStreamInfos()) {
    std::vector<std::vector<GNode>> all_nodes;
    for (const auto &stream_info : iter.second) {
      all_nodes.emplace_back(stream_info.GetAllNodes());
    }
    graph_to_all_nodes[iter.first] = all_nodes;
  }
  return SUCCESS;
}

extern "C" ge::Status GEStreamAllocationSummaryGetStreamGraphs(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, AscendString> &graph_to_stream_graphs) {
  std::shared_ptr<StreamAllocationSummary> stream_allocation;
  GE_ASSERT_SUCCESS(compiled_graph_summary.GetStreamAllocationSummary(stream_allocation));
  GE_ASSERT_NOTNULL(stream_allocation);
  graph_to_stream_graphs = stream_allocation->ToStreamGraph();
  return SUCCESS;
}

const std::map<AscendString, AscendString> &StreamAllocationSummary::StreamAllocationSummaryImpl::ToStreamGraph()
    const {
  return graph_to_stream_graph_;
}

const std::map<AscendString, std::vector<LogicalStreamAllocationInfo>>
    &StreamAllocationSummary::StreamAllocationSummaryImpl::GetAllLogicalStreamInfos() const {
  return graph_to_logical_stream_infos_;
}

Status StreamAllocationSummary::StreamAllocationSummaryImpl::CollectCustomStreamInfo(const ge::ComputeGraphPtr graph,
  std::map<int64_t, ge::LogicalStreamAllocationInfo> &logical_stream_id_to_stream_info) const {
  std::string vec_str;
  std::vector<int64_t> custom_logical_stream_ids;
  if (AttrUtils::GetStr(graph, "_custom_logical_stream_ids", vec_str)) {
    GE_ASSERT_SUCCESS(StreamUtils::TransStrToVec(vec_str, custom_logical_stream_ids));
  }
  for (const auto custom_logical_stream_id : custom_logical_stream_ids) {
    auto stream_info_iter = logical_stream_id_to_stream_info.find(custom_logical_stream_id);
    if (stream_info_iter != logical_stream_id_to_stream_info.end()) {
      stream_info_iter->second.impl_->SetIsAssignedByUsrStreamPass(true);
    }
  }
  return SUCCESS;
}

Status StreamAllocationSummary::StreamAllocationSummaryImpl::CollectStreamInfosFromKnownGraph(
    const ComputeGraphPtr graph, std::vector<LogicalStreamAllocationInfo> &logical_stream_infos) {
  std::map<int64_t, LogicalStreamAllocationInfo> logical_stream_id_to_stream_info;
  std::map<int64_t, std::set<int64_t>> logical_stream_id_to_real_stream_ids;
  std::map<int64_t, std::set<int64_t>> logical_main_stream_id_to_attached_logical_stream_ids;
  std::map<int64_t, int64_t> hccl_real_stream_id_to_hccl_followed_stream_num;
  std::string split_stream_2_logical_stream_str;
  if (!AttrUtils::GetStr(graph, "_split_logic_stream_2_origin_logic_stream", split_stream_2_logical_stream_str)) {
    return SUCCESS;
  }
  std::map<int64_t, int64_t> real_stream_id_to_logical_stream_id;
  GE_ASSERT_SUCCESS(StreamUtils::TransStrToMap(split_stream_2_logical_stream_str,
                                                               real_stream_id_to_logical_stream_id));
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    int64_t logical_main_stream_id = -1;
    if (!AttrUtils::GetInt(op_desc, "_logic_stream_id", logical_main_stream_id) || (logical_main_stream_id < 0)) {
      continue;
    }
    int64_t real_main_stream_id = op_desc->GetStreamId();
    auto logical_stream_iter = logical_stream_id_to_stream_info.find(logical_main_stream_id);
    if (logical_stream_iter == logical_stream_id_to_stream_info.end()) {
      // 创建逻辑主流的stream info
      LogicalStreamAllocationInfo stream_info;
      GE_ASSERT_NOTNULL(stream_info.impl_);
      stream_info.impl_->SetLogicalStreamId(logical_main_stream_id);
      logical_stream_id_to_stream_info[logical_main_stream_id] = stream_info;
    }

    // 将节点存到主流stream info里
    auto &current_logical_main_stream_info = logical_stream_id_to_stream_info[logical_main_stream_id];
    current_logical_main_stream_info.impl_->AppendNode(node);

    // 收集逻辑主流对应的物理流id
    logical_stream_id_to_real_stream_ids[logical_main_stream_id].insert(real_main_stream_id);

    // 收集逻辑主流对应的usr_stream_label
    std::string usr_stream_label = "";
    if (AttrUtils::GetStr(op_desc, public_attr::USER_STREAM_LABEL, usr_stream_label)) {
      current_logical_main_stream_info.impl_->SetUsrStreamLabel(usr_stream_label);
    }

    // 收集逻辑主流对应的hccl从流数量
    int64_t hccl_followed_stream_num = 0;
    if (AttrUtils::GetInt(op_desc, "used_stream_num", hccl_followed_stream_num) && (hccl_followed_stream_num > 0)) {
      auto hccl_real_iter = hccl_real_stream_id_to_hccl_followed_stream_num.find(real_main_stream_id);
      if (hccl_real_iter == hccl_real_stream_id_to_hccl_followed_stream_num.end()) {
        hccl_real_stream_id_to_hccl_followed_stream_num[real_main_stream_id] = hccl_followed_stream_num;
      } else {
        if (hccl_followed_stream_num < hccl_real_iter->second) {
          hccl_real_stream_id_to_hccl_followed_stream_num[real_main_stream_id] = hccl_followed_stream_num;
        }
      }
    }

    auto real_attached_stream_ids = op_desc->GetAttachedStreamIds();
    for (const auto real_attach_stream_id : real_attached_stream_ids) {
      auto real_attached_iter = real_stream_id_to_logical_stream_id.find(real_attach_stream_id);
      GE_ASSERT_TRUE(real_attached_iter != real_stream_id_to_logical_stream_id.end(),
                     "real attached stream id %ld not found in map", real_attach_stream_id);
      auto logical_attached_stream_id = real_attached_iter->second;
      // 收集逻辑主流对应的逻辑从流id
      logical_main_stream_id_to_attached_logical_stream_ids[logical_main_stream_id].insert(logical_attached_stream_id);
      // 收集逻辑从流对应的物理流id
      logical_stream_id_to_real_stream_ids[logical_attached_stream_id].insert(real_attach_stream_id);

      // 创建逻辑从流的stream info
      auto logical_attached_iter = logical_stream_id_to_stream_info.find(logical_attached_stream_id);
      if (logical_attached_iter == logical_stream_id_to_stream_info.end()) {
        LogicalStreamAllocationInfo stream_info;
        GE_ASSERT_NOTNULL(stream_info.impl_);
        stream_info.impl_->SetLogicalStreamId(logical_attached_stream_id);
        logical_stream_id_to_stream_info[logical_attached_stream_id] = stream_info;
      }

      // 将节点存到从流stream info里
      auto &current_logical_attached_stream_info = logical_stream_id_to_stream_info[logical_attached_stream_id];
      current_logical_attached_stream_info.impl_->AppendNode(node);
    }
  }

  // 统计实际的物理流数量
  for (const auto &iter : logical_stream_id_to_real_stream_ids) {
    auto logical_stream_id = iter.first;
    auto stream_info_iter = logical_stream_id_to_stream_info.find(logical_stream_id);
    GE_ASSERT_TRUE(stream_info_iter != logical_stream_id_to_stream_info.end());
    stream_info_iter->second.impl_->SetPhysicalStreamNum(iter.second.size());
  }

  // 统计通信从流数量
  for (const auto &iter : hccl_real_stream_id_to_hccl_followed_stream_num) {
    auto hccl_real_stream_id = iter.first;
    auto hccl_followed_stream_num = iter.second;
    auto hccl_iter = real_stream_id_to_logical_stream_id.find(hccl_real_stream_id);
    GE_ASSERT_TRUE(hccl_iter != real_stream_id_to_logical_stream_id.end());
    auto hccl_logical_stream_id = hccl_iter->second;
    auto stream_info_iter = logical_stream_id_to_stream_info.find(hccl_logical_stream_id);
    GE_ASSERT_TRUE(stream_info_iter != logical_stream_id_to_stream_info.end());
    auto &stream_info = stream_info_iter->second;
    stream_info.impl_->SetHcclFollowedStreamNum(stream_info.GetHcclFollowedStreamNum() + hccl_followed_stream_num);
  }

  // 统计计算从流信息
  for (const auto &iter : logical_main_stream_id_to_attached_logical_stream_ids) {
    auto logical_stream_id = iter.first;
    auto stream_info_iter = logical_stream_id_to_stream_info.find(logical_stream_id);
    GE_ASSERT_TRUE(stream_info_iter != logical_stream_id_to_stream_info.end());
    std::vector<int64_t> attached_logical_stream_ids;
    for (auto attached_stream_id : iter.second) {
      attached_logical_stream_ids.emplace_back(attached_stream_id);
    }
    stream_info_iter->second.impl_->SetAttachedStreamIds(attached_logical_stream_ids);
  }

  // 统计用户自定义pass分流的信息
  GE_ASSERT_SUCCESS(CollectCustomStreamInfo(graph, logical_stream_id_to_stream_info));
  for (const auto &iter : logical_stream_id_to_stream_info) {
    logical_stream_infos.emplace_back(iter.second);
  }

  return SUCCESS;
}

Status StreamAllocationSummary::StreamAllocationSummaryImpl::CollectStreamInfosFromUnKnownGraph(
    const ComputeGraphPtr graph, std::vector<LogicalStreamAllocationInfo> &logical_stream_infos) {
  std::map<int64_t, LogicalStreamAllocationInfo> logical_stream_id_to_stream_info;
  std::map<int64_t, std::set<int64_t>> logical_main_stream_id_to_attached_logical_stream_ids;
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto logical_main_stream_id = op_desc->GetStreamId();
    if (logical_main_stream_id < 0) {
      continue;
    }
    auto logical_stream_iter = logical_stream_id_to_stream_info.find(logical_main_stream_id);
    if (logical_stream_iter == logical_stream_id_to_stream_info.end()) {
      // 创建逻辑主流的stream info
      LogicalStreamAllocationInfo stream_info;
      GE_ASSERT_NOTNULL(stream_info.impl_);
      stream_info.impl_->SetLogicalStreamId(logical_main_stream_id);
      stream_info.impl_->SetPhysicalStreamNum(1U);
      logical_stream_id_to_stream_info[logical_main_stream_id] = stream_info;
    }
    logical_stream_id_to_stream_info[logical_main_stream_id].impl_->AppendNode(node);
    // 收集逻辑主流对应的usr_stream_label
    std::string usr_stream_label = "";
    if (AttrUtils::GetStr(op_desc, public_attr::USER_STREAM_LABEL, usr_stream_label)) {
      logical_stream_id_to_stream_info[logical_main_stream_id].impl_->SetUsrStreamLabel(usr_stream_label);
    }
    // 收集逻辑从流的信息
    auto logical_attached_stream_ids = op_desc->GetAttachedStreamIds();
    for (const auto logical_attached_stream_id : logical_attached_stream_ids) {
      // 收集逻辑主流对应的逻辑从流id
      logical_main_stream_id_to_attached_logical_stream_ids[logical_main_stream_id].insert(logical_attached_stream_id);
      // 创建逻辑从流的stream info
      auto logical_attached_iter = logical_stream_id_to_stream_info.find(logical_attached_stream_id);
      if (logical_attached_iter == logical_stream_id_to_stream_info.end()) {
        LogicalStreamAllocationInfo stream_info;
        GE_ASSERT_NOTNULL(stream_info.impl_);
        stream_info.impl_->SetLogicalStreamId(logical_attached_stream_id);
        stream_info.impl_->SetPhysicalStreamNum(1U);
        logical_stream_id_to_stream_info[logical_attached_stream_id] = stream_info;
      }
      logical_stream_id_to_stream_info[logical_attached_stream_id].impl_->AppendNode(node);
    }
  }
  // 统计计算从流信息
  for (const auto &iter : logical_main_stream_id_to_attached_logical_stream_ids) {
    auto logical_stream_id = iter.first;
    auto stream_info_iter = logical_stream_id_to_stream_info.find(logical_stream_id);
    GE_ASSERT_TRUE(stream_info_iter != logical_stream_id_to_stream_info.end());
    std::vector<int64_t> attached_logical_stream_ids;
    for (auto attached_stream_id : iter.second) {
      attached_logical_stream_ids.emplace_back(attached_stream_id);
    }
    stream_info_iter->second.impl_->SetAttachedStreamIds(attached_logical_stream_ids);
  }

  // 统计用户自定义pass分流的信息
  GE_ASSERT_SUCCESS(CollectCustomStreamInfo(graph, logical_stream_id_to_stream_info));

  for (const auto &iter : logical_stream_id_to_stream_info) {
    logical_stream_infos.emplace_back(iter.second);
  }
  return SUCCESS;
}

Status StreamAllocationSummary::StreamAllocationSummaryImpl::CollectStreamInfos(const GeRootModelPtr &ge_root_model){
  for (const auto &iter : ge_root_model->GetSubgraphInstanceNameToModel()) {
    auto ge_model_ptr = iter.second;
    std::string model_stream_infos;
    auto model_graph = ge_model_ptr->GetGraph();
    GE_ASSERT_NOTNULL(model_graph);
    std::string graph_name = model_graph->GetName();
    std::vector<LogicalStreamAllocationInfo> stream_infos;
    if (model_graph->GetGraphUnknownFlag()) {
      GE_ASSERT_SUCCESS(CollectStreamInfosFromUnKnownGraph(model_graph, stream_infos));
    } else {
      GE_ASSERT_SUCCESS(CollectStreamInfosFromKnownGraph(model_graph, stream_infos));
    }
    graph_to_logical_stream_infos_[AscendString(graph_name.c_str())] = stream_infos;
    size_t stream_num = 0U;
    GE_ASSERT(AttrUtils::GetInt(ge_model_ptr, ATTR_MODEL_STREAM_NUM, stream_num));
    StreamGraphPtr graph = std::make_shared<StreamGraph>(model_graph);
    graph_to_stream_graph_[AscendString(graph_name.c_str())] = AscendString(graph->ToDotString().c_str());
  }
  return SUCCESS;
}

StreamAllocationSummary::StreamAllocationSummary() : impl_(std::make_unique<StreamAllocationSummaryImpl>()) {}

StreamAllocationSummary::~StreamAllocationSummary() = default;

const std::map<AscendString, AscendString> &StreamAllocationSummary::ToStreamGraph() const {
  return impl_->ToStreamGraph();
}

const std::map<AscendString, std::vector<LogicalStreamAllocationInfo>>
    &StreamAllocationSummary::GetAllLogicalStreamInfos() const {
  return impl_->GetAllLogicalStreamInfos();
}

Status CompiledGraphSummary::GetStreamAllocationSummary(std::shared_ptr<StreamAllocationSummary> &stream_allocation) const {
  stream_allocation = data_->GetStreamAllocationSummary();
  return SUCCESS;
}

LogicalStreamAllocationInfo::LogicalStreamAllocationInfo()
    : impl_(std::make_unique<LogicalStreamAllocationInfoImpl>()) {}

LogicalStreamAllocationInfo::~LogicalStreamAllocationInfo() = default;

LogicalStreamAllocationInfo::LogicalStreamAllocationInfo(const LogicalStreamAllocationInfo &stream_info)
    : impl_(std::make_unique<LogicalStreamAllocationInfoImpl>(*stream_info.impl_)) {}

LogicalStreamAllocationInfo &LogicalStreamAllocationInfo::operator=(const LogicalStreamAllocationInfo &stream_info) {
  if (this != &stream_info) {
    impl_ = std::make_unique<LogicalStreamAllocationInfoImpl>(*stream_info.impl_);
  }
  return *this;
}

AscendString LogicalStreamAllocationInfo::ToStringInfo() const {
  return impl_->ToStringInfo();
}

int64_t LogicalStreamAllocationInfo::GetLogicalStreamId() const {
  return impl_->GetLogicalStreamId();
}

AscendString LogicalStreamAllocationInfo::GetUsrStreamLabel() const {
  return impl_->GetUsrStreamLabel();
}

bool LogicalStreamAllocationInfo::IsAssignedByStreamPass() const {
  return impl_->IsAssignedByStreamPass();
}

std::vector<int64_t> LogicalStreamAllocationInfo::GetAttachedStreamIds() const {
  return impl_->GetAttachedStreamIds();
}

size_t LogicalStreamAllocationInfo::GetPhysicalStreamNum() const {
  return impl_->GetPhysicalStreamNum();
}

size_t LogicalStreamAllocationInfo::GetHcclFollowedStreamNum() const {
  return impl_->GetHcclFollowedStreamNum();
}

const std::vector<GNode> &LogicalStreamAllocationInfo::GetAllNodes() const {
  return impl_->GetAllNodes();
}

LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::LogicalStreamAllocationInfoImpl()
    : logical_stream_id_(-1),
      usr_stream_label_(""),
      is_assigned_by_usr_stream_pass_(false),
      attached_stream_ids_({}),
      physical_stream_num_(0),
      hccl_followed_stream_num_(0) {}

AscendString LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::ToStringInfo() const {
  std::string is_assigned_by_usr_stream_pass_str = is_assigned_by_usr_stream_pass_ ? "true" : "false";
  std::string attached_stream_ids_str;
  for (const auto attached_stream_id : attached_stream_ids_) {
    attached_stream_ids_str += std::to_string(attached_stream_id) + " ";
  }
  std::stringstream stream_infos;
  stream_infos << "logic_stream_id: " << std::to_string(logical_stream_id_)
               << ", user_stream_label: " << usr_stream_label_
               << ", is_assigned_by_user_stream_pass: " << is_assigned_by_usr_stream_pass_str
               << ", attached_stream_ids: " << attached_stream_ids_str
               << ", physical_model_stream_num: " << std::to_string(physical_stream_num_)
               << ", hccl_followed_stream_num: " << std::to_string(hccl_followed_stream_num_) << ".\n";
  return AscendString(stream_infos.str().c_str());
}

int64_t LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::GetLogicalStreamId() const {
  return logical_stream_id_;
}

AscendString LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::GetUsrStreamLabel() const {
  return AscendString(usr_stream_label_.c_str());
}

bool LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::IsAssignedByStreamPass() const {
  return is_assigned_by_usr_stream_pass_;
}

std::vector<int64_t> LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::GetAttachedStreamIds() const {
  return attached_stream_ids_;
}

size_t LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::GetPhysicalStreamNum() const {
  return physical_stream_num_;
}

size_t LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::GetHcclFollowedStreamNum() const {
  return hccl_followed_stream_num_;
}

const std::vector<GNode> &LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::GetAllNodes() const {
  return nodes_;
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::AppendNode(const NodePtr &node) {
  nodes_.emplace_back(NodeAdapter::Node2GNode(node));
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::SetLogicalStreamId(int64_t logical_stream_id) {
  logical_stream_id_ = logical_stream_id;
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::SetUsrStreamLabel(
    const std::string &usr_stream_label) {
  usr_stream_label_ = usr_stream_label;
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::SetIsAssignedByUsrStreamPass(
    bool is_assigned_by_usr_stream_pass) {
  is_assigned_by_usr_stream_pass_ = is_assigned_by_usr_stream_pass;
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::SetAttachedStreamIds(
    const std::vector<int64_t> &attached_stream_ids) {
  attached_stream_ids_ = attached_stream_ids;
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::SetPhysicalStreamNum(size_t physical_stream_num) {
  physical_stream_num_ = physical_stream_num;
}

void LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl::SetHcclFollowedStreamNum(
    size_t hccl_followed_stream_num) {
  hccl_followed_stream_num_ = hccl_followed_stream_num;
}

}  // namespace ge
