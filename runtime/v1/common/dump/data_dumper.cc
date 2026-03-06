/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump/data_dumper.h"

#include <cstdlib>
#include <nlohmann/json.hpp>

#include "mmpa/mmpa_api.h"
#include "adx_datadump_server.h"
#include "framework/common/types.h"
#include "common/plugin/datatype_util.h"
#include "common/sgt_slice_type.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "framework/common/runtime_tensor_desc.h"
#include "runtime/rt.h"
#include "runtime/rts/rts_device.h"

namespace {
constexpr uint32_t kAicpuLoadFlag = 1U;
constexpr uint32_t kAicpuUnloadFlag = 0U;
constexpr uint64_t kOpDebugSize = 2048U;
constexpr uint64_t kOpDebugShape = 2048U;
constexpr uint64_t kDumpL1FusionOpMByteSize = 2097152U;  // 2 * 1024 * 1024
constexpr int8_t kDecimalFormat = 10;
constexpr uint32_t k16BitsMask = 0x0000FFFFU;  // 16 bits, 1111,1111,1111,1111
constexpr int32_t k16BitWidth = 16;
constexpr uint32_t kAddrLength = static_cast<uint32_t>(sizeof(void *));
constexpr ge::char_t const *kDumpOutput = "output";
constexpr ge::char_t const *kDumpInput = "input";
constexpr ge::char_t const *kDumpAll = "all";
constexpr ge::char_t const *kDumpDataDefaultValue = "stats";
constexpr uint32_t kDataTypeOutput = 0U;

// parse for format like nodename:input:index
bool ParseNameIndex(const std::string &node_name_index, std::string &node_name, std::string &input_or_output,
                    size_t &index) {
  auto sep = node_name_index.rfind(':');
  if (sep == std::string::npos) {
    return false;
  }
  const auto &index_str = node_name_index.substr(sep + 1U);
  index = static_cast<size_t>(std::strtol(index_str.c_str(), nullptr, kDecimalFormat));
  const auto node_name_without_index = node_name_index.substr(0U, sep);
  sep = node_name_without_index.rfind(':');
  if (sep == std::string::npos) {
    return false;
  }
  node_name = node_name_without_index.substr(0U, sep);
  input_or_output = node_name_without_index.substr(sep + 1U);
  return !((input_or_output != kDumpInput) && (input_or_output != kDumpOutput));
}

static bool IsTensorDescWithSkipDumpAddrType(const bool has_mem_type_attr, const std::vector<int64_t> &v_memory_type,
                                             const size_t i) {
  return has_mem_type_attr && ((v_memory_type[i] == static_cast<int64_t>(RT_MEMORY_L1)) ||
         (v_memory_type[i] == static_cast<int64_t>(ge::kRtMemoryUB)));
}

int64_t GetShapeSizeByJsonInfo(const nlohmann::json &json, uint32_t thread_id, size_t tensor_idx) {
  if (json.size() == 0U || json.size() <= thread_id) {
    return 0;
  }
  const auto non_tail = json[static_cast<size_t>(thread_id)];
  if (non_tail.size() <= tensor_idx) {
    return 0;
  }
  const auto tensor_info = non_tail[tensor_idx];
  if (tensor_info.size() == 0U) {
    return 0;
  }
  int64_t shape = 1;
  for (size_t i = 0U; i < tensor_info.size(); ++i) {
    const int64_t lower_val = tensor_info[i].find("lower").value().get<int64_t>();
    const int64_t higher_val = tensor_info[i].find("higher").value().get<int64_t>();
    shape *= (higher_val - lower_val);
  }
  return shape;
}
}  // namespace

namespace ge {
DataDumper::~DataDumper() noexcept {
  GE_FREE_RT_LOG(dev_mem_load_);
  GE_FREE_RT_LOG(dev_mem_unload_);
  GE_FREE_RT_LOG(dev_mem_unload_for_model_);
}

void DataDumper::SetLoopAddr(const uintptr_t global_step, const uintptr_t loop_per_iter, const uintptr_t loop_cond) {
  global_step_ = global_step;
  loop_per_iter_ = loop_per_iter;
  loop_cond_ = loop_cond;
}

void DataDumper::SaveDumpInput(const std::shared_ptr<Node> &node) {
  if (node != nullptr) {
    const auto &input_op_desc = node->GetOpDesc();
    if (input_op_desc == nullptr) {
      GELOGE(PARAM_INVALID, "[Get][OpDesc] input op desc is null.");
      return;
    }

    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &dst_in_data_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
        const Node *dst_node = dst_in_data_anchor->GetOwnerNodeBarePtr();
        const auto &op_desc = dst_node->GetOpDesc();
        if (op_desc == nullptr) {
          GELOGE(PARAM_INVALID, "[Get][OpDesc] input op desc is null.");
          return;
        }

        (void)input_map_.insert({op_desc->GetName(),
                                {input_op_desc, dst_in_data_anchor->GetIdx(), out_data_anchor->GetIdx()}});
      }
    }
  }
}

void DataDumper::SaveEndGraphId(const uint32_t task_id, const uint32_t stream_id) {
  end_graph_task_id_ = task_id;
  end_graph_stream_id_ = stream_id;
}

Status DataDumper::SaveOpDebugId(const uint32_t task_id, const uint32_t stream_id, const void *const op_debug_addr,
                                const bool is_op_debug) {
  int32_t bit_width;
  int32_t device_id = 0;

  GE_CHK_RT(rtGetDevice(&device_id));
  GE_RETURN_WITH_LOG_IF_TRUE(device_id < 0, "Check device_id %d failed", device_id);

  GE_CHK_RT(rtsDeviceGetCapability(device_id, RT_FEATURE_SYSTEM_TASKID_BIT_WIDTH, &bit_width));
  if (bit_width == k16BitWidth) {
    op_debug_task_id_ = task_id & k16BitsMask;
  } else {
    op_debug_task_id_ = task_id;
  }

  op_debug_stream_id_ = stream_id;
  op_debug_addr_ = op_debug_addr;
  is_op_debug_ = is_op_debug;

  return SUCCESS;
}

void DataDumper::SetWorkSpaceAddr(const std::shared_ptr<OpDesc> &op_desc, const std::vector<uint64_t> &space_addr) {
  for (size_t i = 0U; i < op_list_.size(); ++i) {
    if (op_list_[i].op->GetId() == op_desc->GetId()) {
      for (size_t j = 0U; j < space_addr.size(); ++j) {
        GELOGI("workspace_info[%zu] addr[0x%llx]", j, space_addr[j]);
        op_list_[i].space_addr.emplace_back(space_addr[j]);
      }
    }
  }
}

void DataDumper::SetWorkSpaceAddrForPrint(const std::shared_ptr<OpDesc> &op_desc,
                                          const std::vector<uint64_t> &space_addr) {
  for (size_t i = 0U; i < op_print_list_.size(); ++i) {
    if (op_print_list_[i].op->GetId() == op_desc->GetId()) {
      for (size_t j = 0U; j < space_addr.size(); ++j) {
        GELOGI("workspace_info[%zu] addr[0x%llx]", j, space_addr[j]);
        op_print_list_[i].space_addr.emplace_back(space_addr[j]);
      }
    }
  }
}

void DataDumper::SaveDumpTask(const OpDescInfoId &id, const std::shared_ptr<OpDesc> &op_desc, const uintptr_t args,
                              const FirstLevelAddressInfo &first_level_address_info,
                              const std::map<uint64_t, uint64_t> &cust_to_relevant_offset,
                              const ModelTaskType task_type, bool is_op_debug) {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] Opdesc is nullptr");
    return;
  }

  uint32_t context_id = 0U;
  (void)AttrUtils::GetInt(op_desc, "current_context_id", context_id);
  GELOGI("Save dump task %s, task id: %u, stream id: %u, context id: %u, thread id: %u, is_op_debug: %d.",
         op_desc->GetName().c_str(), id.task_id, id.stream_id, context_id, id.thread_id,
         static_cast<int32_t>(is_op_debug));

  op_list_.push_back({id.task_id, id.stream_id, context_id, id.thread_id, op_desc, args, true, 0, 0, {}, 0,
                      first_level_address_info.address_type, first_level_address_info.address, {},
                      cust_to_relevant_offset, task_type, is_op_debug});

  for (auto iter = input_map_.equal_range(op_desc->GetName()); iter.first != iter.second; ++iter.first) {
    InnerInputMapping &inner_input_mapping = iter.first->second;
    auto &data_op = inner_input_mapping.data_op;
    if (data_op == nullptr) {
      GELOGE(PARAM_INVALID, "[Check][Param] data_op is null.");
      return;
    }

    const auto input_tensor = op_desc->GetInputDescPtr(static_cast<uint32_t>(inner_input_mapping.input_anchor_index));
    if (input_tensor == nullptr) {
      GELOGE(PARAM_INVALID, "[Get][InputDescPtr] input_tensor in op:%s is null, index:%d, size:%zu.",
             op_desc->GetName().c_str(), inner_input_mapping.input_anchor_index, op_desc->GetInputsSize());
      return;
    }

    int64_t data_size = 0;
    if (AttrUtils::GetInt(input_tensor, ATTR_NAME_INPUT_ORIGIN_SIZE, data_size)) {
      GELOGI("Get aipp data size according to attr is %" PRId64, data_size);
    } else {
      if (TensorUtils::GetTensorSizeInBytes(*input_tensor, data_size) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Get][InputSize] failed in %s, index:%u",
               op_desc->GetName().c_str(), inner_input_mapping.input_anchor_index);
        return;
      }
    }

    GELOGI("Save input dump task: %s, id: %u, stream id: %u, data size: %" PRId64 ", input index: %d, output index: %d",
           data_op->GetName().c_str(), id.task_id, id.stream_id, data_size, inner_input_mapping.input_anchor_index,
           inner_input_mapping.output_anchor_index);
    op_list_.push_back({id.task_id, id.stream_id, 0U, id.thread_id, data_op, args, false,
                        inner_input_mapping.input_anchor_index, inner_input_mapping.output_anchor_index,
                        input_tensor->GetShape().GetDims(), data_size,
                        first_level_address_info.address_type, first_level_address_info.address, {},
                        cust_to_relevant_offset, task_type, is_op_debug});
  }
}

void DataDumper::SavePrintDumpTask(const OpDescInfoId &id, const std::shared_ptr<OpDesc> &op_desc, const uintptr_t args,
                                   const FirstLevelAddressInfo &first_level_address_info,
                                   const ModelTaskType task_type) {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] Opdesc is nullptr");
    return;
  }

  uint32_t context_id = 0U;
  (void)AttrUtils::GetInt(op_desc, "current_context_id", context_id);
  if ((op_desc->GetType() != "SuperKernel") || (task_type == ModelTaskType::MODEL_TASK_SUPER_KERNEL)) {
    GELOGI("Save ascendc printf dump task %s, task id: %u, stream id: %u, context id: %u, thread id: %u.",
           op_desc->GetName().c_str(), id.task_id, id.stream_id, context_id, id.thread_id);
    op_print_list_.push_back({id.task_id, id.stream_id, context_id, id.thread_id, op_desc, args, true, 0, 0, {}, 0,
                              first_level_address_info.address_type, first_level_address_info.address, {}, {},
                              task_type, false});
  }
}

static void SetOpMappingLoopAddr(const uintptr_t step_id, const uintptr_t loop_per_iter, const uintptr_t loop_cond,
                                 toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (step_id != 0U) {
    GELOGI("step_id exists.");
    op_mapping_info.set_step_id_addr(static_cast<uint64_t>(step_id));
  }

  if (loop_per_iter != 0U) {
    GELOGI("loop_per_iter exists.");
    op_mapping_info.set_iterations_per_loop_addr(static_cast<uint64_t>(loop_per_iter));
  }

  if (loop_cond != 0U) {
    GELOGI("loop_cond exists.");
    op_mapping_info.set_loop_cond_addr(static_cast<uint64_t>(loop_cond));
  }
}

Status DataDumper::GenerateOutput(toolkit::aicpu::dump::Output &output,
                                  const OpDesc::Vistor<GeTensorDescPtr> &tensor_descs,
                                  const uintptr_t addr, const size_t index, const uint64_t offset) {
  output.set_data_type(DataTypeUtil::GetIrDataType(tensor_descs.at(index)->GetDataType()));
  output.set_format(static_cast<int32_t>(tensor_descs.at(index)->GetFormat()));

  for (const int64_t dim : tensor_descs.at(index)->GetShape().GetDims()) {
    output.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
  }
  for (const int64_t dim : tensor_descs.at(index)->GetOriginShape().GetDims()) {
    output.mutable_origin_shape()->add_dim(static_cast<uint64_t>(dim));
  }
  int64_t output_size = 0;
  if (TensorUtils::GetTensorSizeInBytes(*tensor_descs.at(index), output_size) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get tensor size fail");
    GELOGE(PARAM_INVALID, "[Get][OutputSize] failed");
    return PARAM_INVALID;
  }
  GELOGD("Get output size in dump is %" PRId64 ".", output_size);
  std::string origin_name;
  bool no_tiling_mem_type = false;
  int32_t origin_output_index = -1;
  const std::string* origin_name_ptr = AttrUtils::GetStr(tensor_descs.at(index).get(), ATTR_NAME_DATA_DUMP_ORIGIN_NAME);
  if (origin_name_ptr != nullptr) {
      origin_name = *origin_name_ptr;
  }
  (void)AttrUtils::GetInt(tensor_descs.at(index).get(), ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
  (void)AttrUtils::GetBool(tensor_descs.at(index).get(), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling_mem_type);
  output.set_size(static_cast<uint64_t>(output_size));
  output.set_original_name(origin_name);
  output.set_original_output_index(origin_output_index);
  output.set_original_output_format(static_cast<int32_t>(tensor_descs.at(index)->GetOriginFormat()));
  output.set_original_output_data_type(static_cast<int32_t>(tensor_descs.at(index)->GetOriginDataType()));
  output.set_address(static_cast<uint64_t>(addr));
  output.set_offset(offset);
  InnerRealAddressAndSize real_address_and_size;
  real_address_and_size.address = static_cast<uint64_t>(addr);
  real_address_and_size.size = (offset == 0U) ? static_cast<uint64_t>(output_size) : offset;
  context_.output.emplace_back(real_address_and_size);
  output.set_addr_type(no_tiling_mem_type ?
      toolkit::aicpu::dump::AddressType::NOTILING_ADDR : toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR);
  return SUCCESS;
}

Status DataDumper::DumpRefOutput(const DataDumper::InnerDumpInfo &inner_dump_info,
                                 toolkit::aicpu::dump::Output &output,
                                 const size_t i, const std::string &node_name_index) {
  std::string dump_op_name;
  std::string input_or_output;
  size_t index;
  // parser and find which node's input or output tensor desc is chosen for dump info
  if (!ParseNameIndex(node_name_index, dump_op_name, input_or_output, index)) {
    GELOGE(PARAM_INVALID, "[Check][Param] Op [%s] output desc[%zu] with invalid ATTR_DATA_DUMP_REF attr[%s].",
           inner_dump_info.op->GetName().c_str(), i, node_name_index.c_str());
    return PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(compute_graph_);
  const auto &replace_node = compute_graph_->FindNode(dump_op_name);
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(replace_node == nullptr,
                                       "[Check][Param] Op [%s] output desc[%zu] with invalid ATTR_DATA_DUMP_REF "
                                       "attr[%s], cannot find redirect node[%s].",
                                       inner_dump_info.op->GetName().c_str(), i, node_name_index.c_str(),
                                       dump_op_name.c_str());
  const auto &replace_opdesc = replace_node->GetOpDesc();
  GE_CHECK_NOTNULL(replace_opdesc);
  const auto iter = ref_info_.find(replace_opdesc);
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(iter == ref_info_.end(),
                                       "[Check][Param] Op [%s] output desc[%zu] cannot find "
                                       "any saved redirect node[%s]'s info.",
                                       inner_dump_info.op->GetName().c_str(), i, replace_opdesc->GetName().c_str());
  GE_CHECK_NOTNULL(iter->second);
  auto addr = PtrToValue(iter->second);
  if (input_or_output == kDumpInput) {
    const auto &replace_input_descs = replace_opdesc->GetAllInputsDescPtr();
    addr += static_cast<size_t>(kAddrLength) * index;
    GE_CHK_STATUS_RET(GenerateOutput(output, replace_input_descs, addr, index),
                      "[Generate][Output] failed for %s, index:%zu", inner_dump_info.op->GetName().c_str(), index);
  } else {
    const auto &replace_output_descs = replace_opdesc->GetAllOutputsDescPtr();
    const size_t replace_input_size = replace_opdesc->GetAllInputsDescPtr().size();
    addr += static_cast<uint64_t>((index + replace_input_size) * static_cast<size_t>(kAddrLength));
    GE_CHK_STATUS_RET(GenerateOutput(output, replace_output_descs, addr, index),
                      "[Generate][Output] failed for %s, index:%zu", inner_dump_info.op->GetName().c_str(), index);
  }
  GELOGD("Op [%s] output desc[%zu] dump info is replaced by node[%s] [%s] tensor_desc [%zu]",
         inner_dump_info.op->GetName().c_str(), i, dump_op_name.c_str(), input_or_output.c_str(), index);
  return SUCCESS;
}

Status DataDumper::DumpOutputWithRawAddress(const InnerDumpInfo &inner_dump_info,
                                            toolkit::aicpu::dump::Task &task) {
  const auto &output_descs = inner_dump_info.op->GetAllOutputsDescPtr();
  const size_t offset = inner_dump_info.op->GetAllInputsDescPtr().size();
  GE_CHECK_GE(inner_dump_info.address.size(), offset + output_descs.size());
  for (size_t i = 0U; i < output_descs.size(); ++i) {
    toolkit::aicpu::dump::Output output;
    const auto addr = inner_dump_info.address[offset + i];
    GE_CHK_STATUS_RET(GenerateOutput(output, output_descs, addr, i), "[Generate][Output] failed for %s, index:%zu",
                      inner_dump_info.op->GetName().c_str(), i);
    output.set_addr_type(toolkit::aicpu::dump::AddressType::RAW_ADDR);
    task.mutable_output()->Add(std::move(output));
  }
  return SUCCESS;
}

Status DataDumper::GetOffsetFromJson(const InnerDumpInfo &inner_dump_info, size_t tensor_idx, size_t input_size,
                                     uint64_t &offset) const {
  std::string ffts_str;
  const std::string* ffts_str_ptr = ge::AttrUtils::GetStr(inner_dump_info.op, ffts::kAttrSgtJsonInfo);
  if (ffts_str_ptr == nullptr || ffts_str_ptr->empty()) {
      return ge::FAILED;
  }
  ffts_str = *ffts_str_ptr;
  GE_ASSERT_TRUE(nlohmann::json::accept(ffts_str));
  nlohmann::json slice_info_json = nlohmann::json::parse(ffts_str);
  GE_ASSERT_TRUE(!slice_info_json.is_null());
  int64_t shape_size = -1;
  if (input_size == 0U) {
    const auto input_tensor_slice = slice_info_json.find("input_tensor_slice");
    GE_ASSERT_TRUE(input_tensor_slice != slice_info_json.end());
    shape_size = ge::GetSizeInBytes(GetShapeSizeByJsonInfo(input_tensor_slice.value(), inner_dump_info.thread_id,
                                                           tensor_idx),
                                    inner_dump_info.op->GetAllInputsDesc().at(tensor_idx).GetDataType());
  } else {
    const auto output_tensor_slice = slice_info_json.find("output_tensor_slice");
    GE_ASSERT_TRUE(output_tensor_slice != slice_info_json.end());
    shape_size = ge::GetSizeInBytes(GetShapeSizeByJsonInfo(output_tensor_slice.value(), inner_dump_info.thread_id,
                                                           tensor_idx),
                                    inner_dump_info.op->GetAllOutputsDesc().at(tensor_idx).GetDataType());
  }
  GE_ASSERT_TRUE(shape_size >= 0);
  offset = static_cast<uint64_t>(shape_size);
  return ge::SUCCESS;
}

uint64_t DataDumper::GetOffset(const InnerDumpInfo &inner_dump_info, const size_t i, const size_t input_size) const {
  uint64_t offset = 0UL;
  if (GetOffsetFromJson(inner_dump_info, i, input_size, offset) == ge::SUCCESS && offset != 0UL) {
    return offset;
  }

  vector<uint64_t> task_addr_offset;
  task_addr_offset = inner_dump_info.op->TryGetExtAttr("task_addr_offset", task_addr_offset);
  if (!task_addr_offset.empty() && (i + input_size < task_addr_offset.size())) {
    offset = task_addr_offset[i + input_size];
  }

  return offset;
}

Status DataDumper::DumpOutputWithTask(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task) {
  const auto &output_descs = inner_dump_info.op->GetAllOutputsDescPtr();
  const std::vector<void *> output_addrs = ModelUtils::GetOutputAddrs(*runtime_param_, inner_dump_info.op);
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr =
          ge::AttrUtils::GetListInt(inner_dump_info.op, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(has_mem_type_attr && (v_memory_type.size() != output_descs.size()),
                                       "[Check][Param] DumpOutputWithTask[%s], output size[%zu], "
                                       "output memory type size[%zu]", inner_dump_info.op->GetName().c_str(),
                                       output_descs.size(), v_memory_type.size());

  size_t no_need_dump_output_num = 0U;
  const std::string model_name = model_name_;
  const std::string om_name = om_name_;
  for (size_t i = 0U; i < output_descs.size(); ++i) {
    toolkit::aicpu::dump::Output output;
    const auto &output_desc = *output_descs.at(i);
    int32_t calc_type = 0;
    GELOGI("[Dumper] Model name %s, Node name %s, Node type %s, output index %zu.", model_name.c_str(), inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
    if (dump_properties_.IsOutputInOpNameBlacklist(model_name, inner_dump_info.op->GetName().c_str(), i) ||
      dump_properties_.IsOutputInOpNameBlacklist(om_name, inner_dump_info.op->GetName().c_str(), i) ||
      dump_properties_.IsOutputInOpNameBlacklist(DUMP_LAYER_OP_MODEL, inner_dump_info.op->GetName().c_str(), i)) {
      GELOGI("[Dumper] Node name %s, Node type: %s, output index %zu is in opname-blacklist, skip to dump this output.",
         inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
      continue;
    }
    if (dump_properties_.IsOutputInOpTypeBlacklist(model_name, inner_dump_info.op->GetType().c_str(), i) ||
      dump_properties_.IsOutputInOpTypeBlacklist(om_name, inner_dump_info.op->GetType().c_str(), i) ||
      dump_properties_.IsOutputInOpTypeBlacklist(DUMP_LAYER_OP_MODEL, inner_dump_info.op->GetType().c_str(), i)) {
      GELOGI("[Dumper] Node name %s, Node type: %s, output index %zu is in optype-blacklist, skip to dump this output.",
         inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
      continue;
    }
    const bool has_calc_type = ge::AttrUtils::GetInt(output_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (has_calc_type && (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY))) {
      GELOGD("Node[%s] output[index:%zu] [name:%s] is an optional output, don't need to dump this output.",
             inner_dump_info.op->GetName().c_str(), i, output_desc.GetName().c_str());
      ++no_need_dump_output_num;
      continue;
    }

    if ((output_descs.size() - no_need_dump_output_num) < output_addrs.size()) {
      REPORT_INNER_ERR_MSG("E19999", "The number of output does not match in op:%s(%s). The size[%zu] of output which is "
                         "no need to dump should not greater than the size[%zu] of output descs minus the size[%zu] of "
                         "output which is need to dump.", inner_dump_info.op->GetName().c_str(),
                         inner_dump_info.op->GetType().c_str(), no_need_dump_output_num, output_descs.size(),
                         output_addrs.size());
      GELOGE(PARAM_INVALID, "[Check][Param] The number of output does not match in op:%s(%s). The size[%zu] of output "
             "which is no need to dump should not greater than the size[%zu] of output descs minus the size[%zu] "
             "of output which is need to dump.", inner_dump_info.op->GetName().c_str(),
             inner_dump_info.op->GetType().c_str(), no_need_dump_output_num, output_descs.size(), output_addrs.size());
      return PARAM_INVALID;
    }

    // check dump output tensor desc is redirected by attr ATTR_DATA_DUMP_REF
    const std::string* node_name_index_ptr = AttrUtils::GetStr(&output_desc, ATTR_DATA_DUMP_REF);
    if (node_name_index_ptr != nullptr) {
      GE_CHK_STATUS_RET(DumpRefOutput(inner_dump_info, output, i, *node_name_index_ptr), "[Dump][RefOutput] failed");
      task.mutable_output()->Add(std::move(output));
      continue;
    }
    if (IsTensorDescWithSkipDumpAddrType(has_mem_type_attr, v_memory_type, i)) {
      GELOGI("[L1Fusion] DumpOutputWithTask[%s] output[%zu] is l1 addr.", inner_dump_info.op->GetName().c_str(), i);
      int64_t output_size = 0;
      if (TensorUtils::GetTensorSizeInBytes(*output_descs.at(i), output_size) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Get output tensor size fail in op:%s(%s), index:%zu",
                          inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
        GELOGE(PARAM_INVALID, "[Get][OutputSize] failed in %s, index:%zu", inner_dump_info.op->GetName().c_str(), i);
        return PARAM_INVALID;
      }
      GELOGI("Get output size of l1_fusion_dump is %" PRId64, output_size);
      need_generate_op_buffer_ = true;
    } else {
      const auto input_size = inner_dump_info.op->GetInputsSize();
      uint64_t cur_offset = i + input_size;
      auto &cust_to_relevant = inner_dump_info.cust_to_relevant_offset_;
      if (!cust_to_relevant.empty()) {
        auto iter = cust_to_relevant.find(cur_offset);
        if (iter == cust_to_relevant.end()) {
          GELOGD("Skip to dump op [%s] output idx [%zu]", inner_dump_info.op->GetNamePtr(), i);
          continue;
        }
        cur_offset = iter->second;
      }
      const uintptr_t addr = static_cast<uintptr_t>(inner_dump_info.args + (cur_offset * kAddrLength));
      GELOGD("Dump op [%s] output[%zu], args_begin:[%" PRIx64 "] offset_addr:[%" PRIx64 "]",
        inner_dump_info.op->GetNamePtr(), i, inner_dump_info.args, addr);
      GE_CHK_STATUS_RET(GenerateOutput(output, output_descs, addr, i, GetOffset(inner_dump_info, i, input_size)),
                        "[Generate][Output] failed for %s, index:%zu", inner_dump_info.op->GetName().c_str(), i);
      task.mutable_output()->Add(std::move(output));
    }
  }
  return SUCCESS;
}

Status DataDumper::DumpOutput(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump output.");
  if (inner_dump_info.is_task) {
    // tbe or aicpu op, these ops are with task
    return inner_dump_info.is_raw_address ? DumpOutputWithRawAddress(inner_dump_info, task) :
        DumpOutputWithTask(inner_dump_info, task);
  }
  // else data, const or variable op
  toolkit::aicpu::dump::Output output;
  const std::string model_name = model_name_;
  const std::string om_name = om_name_;
  // if optype-blacklist contains output0, skip to dump the input node of model.
  if (dump_properties_.IsOutputInOpTypeBlacklist(model_name, inner_dump_info.op->GetType().c_str(), kDataTypeOutput) ||
    dump_properties_.IsOutputInOpTypeBlacklist(om_name, inner_dump_info.op->GetType().c_str(), kDataTypeOutput) ||
    dump_properties_.IsOutputInOpTypeBlacklist(DUMP_LAYER_OP_MODEL, inner_dump_info.op->GetType().c_str(), kDataTypeOutput)) {
    GELOGI("[Dumper] Node name %s, Node type: %s, output index %zu is in optype-blacklist, skip to dump this output.",
        inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), kDataTypeOutput);
    return SUCCESS;
  }
  const auto output_tensor =
          inner_dump_info.op->GetOutputDescPtr(static_cast<uint32_t>(inner_dump_info.output_anchor_index));
  const std::vector<void *> output_addrs = ModelUtils::GetOutputAddrs(*runtime_param_, inner_dump_info.op);
  if (output_tensor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "output_desc tensor is nullptr in op:%s(%s), index:%d, check invalid",
                       inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(),
                       inner_dump_info.output_anchor_index);
    GELOGE(PARAM_INVALID, "[Get][OutputDescPtr] output_tensor is null in op:%s, index:%d, size:%zu.",
           inner_dump_info.op->GetName().c_str(), inner_dump_info.output_anchor_index,
           inner_dump_info.op->GetOutputsSize());
    return PARAM_INVALID;
  }

  output.set_data_type(DataTypeUtil::GetIrDataType(output_tensor->GetDataType()));
  output.set_format(static_cast<int32_t>(output_tensor->GetFormat()));

  for (const int64_t dim : inner_dump_info.dims) {
    output.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
  }

  std::string origin_name;
  int32_t origin_output_index = -1;
  bool no_tiling_mem_type = false;
  const std::string* origin_name_ptr = AttrUtils::GetStr(output_tensor, ATTR_NAME_DATA_DUMP_ORIGIN_NAME);
  if (origin_name_ptr != nullptr) {
      origin_name = *origin_name_ptr;
  }
  (void)AttrUtils::GetInt(output_tensor, ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
  (void)AttrUtils::GetBool(output_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling_mem_type);
  output.set_size(static_cast<uint64_t>(inner_dump_info.data_size));
  output.set_original_name(origin_name);
  output.set_original_output_index(origin_output_index);
  output.set_original_output_format(static_cast<int32_t>(output_tensor->GetOriginFormat()));
  output.set_original_output_data_type(static_cast<int32_t>(output_tensor->GetOriginDataType()));
  // due to lhisi virtual addr bug, cannot use args now
  if (inner_dump_info.output_anchor_index >= static_cast<int32_t>(output_addrs.size())) {
    REPORT_INNER_ERR_MSG("E19999", "output_anchor_index:%d >= output addr size:%zu in op:%s(%s), "
                       "check invalid", inner_dump_info.output_anchor_index, output_addrs.size(),
                       inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] output_anchor_index:%d >= output addr size:%zu in op:%s(%s)",
           inner_dump_info.output_anchor_index, output_addrs.size(),
           inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str());
    return FAILED;
  }
  uint64_t cur_offset = inner_dump_info.input_anchor_index;
  auto &cust_to_relevant = inner_dump_info.cust_to_relevant_offset_;
  if (!cust_to_relevant.empty()) {
    auto iter = cust_to_relevant.find(cur_offset);
    if (iter == cust_to_relevant.end()) {
      GELOGD("Skip to dump op [%s] output idx [%zu]", inner_dump_info.op->GetNamePtr(),
             inner_dump_info.output_anchor_index);
      return SUCCESS;
    }
    GELOGD("Output dump address offset: %u, op: %s, input index: %d", cur_offset, inner_dump_info.op->GetNamePtr(),
           inner_dump_info.input_anchor_index);
    cur_offset = iter->second;
  }
  const uintptr_t data_addr = inner_dump_info.args + static_cast<uintptr_t>(kAddrLength * cur_offset);
  GELOGD("Set Output dump address, op: %s, args: %p, input index: %d, addr: %p", inner_dump_info.op->GetNamePtr(),
         inner_dump_info.args, inner_dump_info.input_anchor_index, data_addr);
  output.set_address(static_cast<uint64_t>(data_addr));
  output.set_addr_type(no_tiling_mem_type ?
      toolkit::aicpu::dump::AddressType::NOTILING_ADDR : toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR);

  task.mutable_output()->Add(std::move(output));

  return SUCCESS;
}

Status DataDumper::GenerateInput(toolkit::aicpu::dump::Input &input, const OpDesc::Vistor<GeTensorDescPtr> &tensor_descs,
                                 const uintptr_t addr, const size_t index, const uint64_t offset) {
  input.set_data_type(DataTypeUtil::GetIrDataType(tensor_descs.at(index)->GetDataType()));
  input.set_format(static_cast<int32_t>(tensor_descs.at(index)->GetFormat()));

  for (const int64_t dim : tensor_descs.at(index)->GetShape().GetDims()) {
    input.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
  }
  for (const int64_t dim : tensor_descs.at(index)->GetOriginShape().GetDims()) {
    input.mutable_origin_shape()->add_dim(static_cast<uint64_t>(dim));
  }
  int64_t input_size = 0;
  if (AttrUtils::GetInt(*tensor_descs.at(index), ATTR_NAME_INPUT_ORIGIN_SIZE, input_size)) {
    GELOGI("Get aipp input size according to attr is %" PRId64, input_size);
  } else {
    if (TensorUtils::GetTensorSizeInBytes(*tensor_descs.at(index), input_size) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get tensor size fail");
      GELOGE(PARAM_INVALID, "[Get][TensorSize] failed");
      return PARAM_INVALID;
    }
  }
  GELOGD("Get input size in dump is %" PRId64, input_size);
  input.set_size(static_cast<uint64_t>(input_size));
  input.set_address(static_cast<uint64_t>(addr));
  input.set_offset(offset);
  InnerRealAddressAndSize real_address_and_size;
  real_address_and_size.address = static_cast<uint64_t>(addr);
  real_address_and_size.size = (offset == 0U) ? static_cast<uint64_t>(input_size) : offset;
  context_.input.emplace_back(real_address_and_size);
  bool no_tiling_mem_type = false;
  (void)AttrUtils::GetBool(*tensor_descs.at(index), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling_mem_type);
  input.set_addr_type(no_tiling_mem_type ?
      toolkit::aicpu::dump::AddressType::NOTILING_ADDR : toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR);
  return SUCCESS;
}

Status DataDumper::DumpRefInput(const DataDumper::InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Input &input,
                                const size_t i, const std::string &node_name_index) {
  std::string dump_op_name;
  std::string input_or_output;
  size_t index;
  // parser and find which node's input or output tensor desc is chosen for dump info
  if (!ParseNameIndex(node_name_index, dump_op_name, input_or_output, index)) {
    GELOGE(PARAM_INVALID, "[Call][ParseNameIndex] Op [%s] input desc[%zu] with invalid ATTR_DATA_DUMP_REF attr[%s].",
           inner_dump_info.op->GetName().c_str(), i, node_name_index.c_str());
    return PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(compute_graph_);
  const auto &replace_node = compute_graph_->FindNode(dump_op_name);
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(replace_node == nullptr,
                                       "[Check][Param] Op [%s] input desc[%zu] with invalid ATTR_DATA_DUMP_REF "
                                       "attr[%s], cannot find redirect node[%s].",
                                       inner_dump_info.op->GetName().c_str(), i, node_name_index.c_str(),
                                       dump_op_name.c_str());
  const auto &replace_opdesc = replace_node->GetOpDesc();
  GE_CHECK_NOTNULL(replace_opdesc);
  const auto iter = ref_info_.find(replace_opdesc);
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(iter == ref_info_.end(),
                                       "[Check][Param] Op [%s] input desc[%zu] cannot find "
                                       "any saved redirect node[%s]'s info.",
                                       inner_dump_info.op->GetName().c_str(), i, replace_opdesc->GetName().c_str());
  GE_CHECK_NOTNULL(iter->second);
  auto addr = PtrToValue(iter->second);
  if (input_or_output == kDumpInput) {
    const auto &replace_input_descs = replace_opdesc->GetAllInputsDescPtr();
    addr += static_cast<uint64_t>(kAddrLength * index);
    GE_CHK_STATUS_RET(GenerateInput(input, replace_input_descs, static_cast<uintptr_t>(addr), index),
                      "[Generate][Input] failed for %s, index:%zu", inner_dump_info.op->GetName().c_str(), index);
  } else {
    const auto &replace_output_descs = replace_opdesc->GetAllOutputsDescPtr();
    const size_t replace_input_size = replace_opdesc->GetAllInputsDescPtr().size();
    addr += static_cast<uint64_t>((index + replace_input_size) * static_cast<size_t>(kAddrLength));
    GE_CHK_STATUS_RET(GenerateInput(input, replace_output_descs, static_cast<uintptr_t>(addr), index),
                      "[Generate][Input] failed for %s, index:%zu", inner_dump_info.op->GetName().c_str(), index);
  }
  GELOGD("Op [%s] input desc[%zu] dump info is replaced by node[%s] [%s] tensor_desc [%zu]",
         inner_dump_info.op->GetName().c_str(), i, dump_op_name.c_str(), input_or_output.c_str(), index);
  return SUCCESS;
}

Status DataDumper::DumpInputWithRawAddress(const InnerDumpInfo &inner_dump_info,
                                           toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump input.");
  const auto &input_descs = inner_dump_info.op->GetAllInputsDescPtr();
  GE_CHECK_GE(inner_dump_info.address.size(), input_descs.size());
  for (size_t i = 0U; i < input_descs.size(); ++i) {
    toolkit::aicpu::dump::Input input;
    const auto addr = inner_dump_info.address[i];
    GE_CHK_STATUS_RET(GenerateInput(input, input_descs, addr, i), "[Generate][Input] failed for op:%s, index:%zu",
                      inner_dump_info.op->GetName().c_str(), i);
    input.set_addr_type(toolkit::aicpu::dump::AddressType::RAW_ADDR);
    task.mutable_input()->Add(std::move(input));
  }
  return SUCCESS;
}

Status DataDumper::DumpInput(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump input.");
  const auto &input_descs = inner_dump_info.op->GetAllInputsDescPtr();
  const std::vector<void *> input_addrs = ModelUtils::GetInputAddrs(*runtime_param_, inner_dump_info.op);
  if (input_descs.size() != input_addrs.size()) {
    REPORT_INNER_ERR_MSG("E19999", "input_desc size:%zu != input addr size:%zu in op:%s(%s)",
                       input_descs.size(), input_addrs.size(),
                       inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid input desc addrs size %zu, op %s has %zu input desc.",
           input_addrs.size(), inner_dump_info.op->GetName().c_str(), input_descs.size());
    return PARAM_INVALID;
  }
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr =
          ge::AttrUtils::GetListInt(inner_dump_info.op, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(has_mem_type_attr && (v_memory_type.size() != input_descs.size()),
                                       "[Check][Param] DumpInput[%s], input size[%zu], input memory type size[%zu]",
                                       inner_dump_info.op->GetName().c_str(), input_descs.size(), v_memory_type.size());

  const std::string model_name = model_name_;
  const std::string om_name = om_name_;
  for (size_t i = 0U; i < input_descs.size(); ++i) {
    toolkit::aicpu::dump::Input input;
    std::string node_name_index;
    GELOGI("[Dumper] Model name %s, Node name %s, Node type %s, input index %zu.", model_name.c_str(), inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
    if (dump_properties_.IsInputInOpNameBlacklist(model_name, inner_dump_info.op->GetName().c_str(), i) ||
      dump_properties_.IsInputInOpNameBlacklist(om_name, inner_dump_info.op->GetName().c_str(), i) ||
      dump_properties_.IsInputInOpNameBlacklist(DUMP_LAYER_OP_MODEL, inner_dump_info.op->GetName().c_str(), i)) {
      GELOGI("[Dumper] Node name %s, Node type: %s, input index %zu is in opname-blacklist, skip to dump this input.",
         inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
      continue;
    }
    if (dump_properties_.IsInputInOpTypeBlacklist(model_name, inner_dump_info.op->GetType().c_str(), i) ||
      dump_properties_.IsInputInOpTypeBlacklist(om_name, inner_dump_info.op->GetType().c_str(), i) ||
      dump_properties_.IsInputInOpTypeBlacklist(DUMP_LAYER_OP_MODEL, inner_dump_info.op->GetType().c_str(), i)) {
      GELOGI("[Dumper] Node name %s, Node type: %s, input index %zu is in optype-blacklist, skip to dump this input.",
         inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
      continue;
    }
    // check dump input tensor desc is redirected by attr ATTR_DATA_DUMP_REF
    const std::string* node_name_index_ptr = AttrUtils::GetStr(input_descs.at(i).get(), ATTR_DATA_DUMP_REF);
    if (node_name_index_ptr != nullptr) {
      node_name_index = *node_name_index_ptr;
      GE_CHK_STATUS_RET(DumpRefInput(inner_dump_info, input, i, node_name_index),
                        "[Dump][RefInput] failed, node name index:%s", node_name_index.c_str());
      task.mutable_input()->Add(std::move(input));
      continue;
      // normal dump without attr
    }
    if (IsTensorDescWithSkipDumpAddrType(has_mem_type_attr, v_memory_type, i)) {
      GELOGI("[L1Fusion] DumpInput[%s] input[%zu] is l1 addr", inner_dump_info.op->GetName().c_str(), i);
      int64_t input_size = 0;
      if (AttrUtils::GetInt(*input_descs.at(i), ATTR_NAME_INPUT_ORIGIN_SIZE, input_size)) {
        GELOGI("Get aipp input size according to attr is %" PRId64, input_size);
      } else {
        if (TensorUtils::GetTensorSizeInBytes(*input_descs.at(i), input_size) != SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Get input tensor size fail in op:%s(%s), index:%zu",
                            inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
          GELOGE(PARAM_INVALID, "[Get][InputTensorSize] fail in op:%s(%s), index:%zu",
                  inner_dump_info.op->GetName().c_str(), inner_dump_info.op->GetType().c_str(), i);
          return PARAM_INVALID;
        }
      }
      GELOGI("Get input size of l1_fusion_dump is %" PRId64, input_size);
      need_generate_op_buffer_ = true;
    } else {
      uint64_t cur_offset = i;
      auto &cust_to_relevant = inner_dump_info.cust_to_relevant_offset_;
      if (!cust_to_relevant.empty()) {
        auto iter = cust_to_relevant.find(cur_offset);
        if (iter == cust_to_relevant.end()) {
          GELOGD("Skip to dump op [%s] input idx [%zu]", inner_dump_info.op->GetNamePtr(), i);
          continue;
        }
        cur_offset = iter->second;
      }
      const uintptr_t addr = inner_dump_info.args + static_cast<uintptr_t>(cur_offset * kAddrLength);
      GELOGD("Dump op [%s] input[%zu], args_begin:[%" PRIx64 "] offset_addr:[%" PRIx64 "]",
        inner_dump_info.op->GetNamePtr(), i, inner_dump_info.args, addr);
      GE_CHK_STATUS_RET(GenerateInput(input, input_descs, addr, i, GetOffset(inner_dump_info, i, 0U)),
                        "[Generate][Input] failed for op:%s, index:%zu", inner_dump_info.op->GetName().c_str(), i);
      task.mutable_input()->Add(std::move(input));
    }
  }
  return SUCCESS;
}

void DataDumper::DumpContext(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task) {
  bool is_fftsplus_task = false;
  (void)AttrUtils::GetBool(inner_dump_info.op, "_is_fftsplus_task", is_fftsplus_task);
  if (inner_dump_info.op->HasAttr("current_context_id") || (is_fftsplus_task && inner_dump_info.is_op_debug)) {
    toolkit::aicpu::dump::Context ffts_context;
    ffts_context.set_context_id(inner_dump_info.context_id);
    ffts_context.set_thread_id(inner_dump_info.thread_id);
    std::stringstream dbg_ss;
    for (const auto &input : context_.input) {
      toolkit::aicpu::dump::RealAddressAndSize real_address_and_size;
      real_address_and_size.set_address(input.address);
      real_address_and_size.set_size(input.size);
      ffts_context.mutable_input()->Add(std::move(real_address_and_size));
      dbg_ss << "[input addr: 0x" << &(std::hex) << input.address << ", size: 0x" << input.size << "]";
    }
    for (const auto &output : context_.output) {
      toolkit::aicpu::dump::RealAddressAndSize real_address_and_size;
      real_address_and_size.set_address(output.address);
      real_address_and_size.set_size(output.size);
      ffts_context.mutable_output()->Add(std::move(real_address_and_size));
      dbg_ss << "[output addr: 0x" << &(std::hex) << output.address << ", size: 0x" << output.size << "]";
    }
    task.mutable_context()->Add(std::move(ffts_context));
    GELOGD(
        "Op %s add context with task id %u steam id %u context id %u thread id %u, input num %u, output num %u, "
        "address info %s, is_fftsplus_task %d",
        inner_dump_info.op->GetName().c_str(), inner_dump_info.task_id, inner_dump_info.stream_id,
        inner_dump_info.context_id, inner_dump_info.thread_id, context_.input.size(), context_.output.size(),
        dbg_ss.str().c_str(), static_cast<int32_t>(is_fftsplus_task));
  }
  context_.input.clear();
  context_.output.clear();
}

void DataDumper::GenerateOpBuffer(const uint64_t size, toolkit::aicpu::dump::Task &task) {
  if (need_generate_op_buffer_) {
    toolkit::aicpu::dump::OpBuffer op_buffer;
    op_buffer.set_buffer_type(toolkit::aicpu::dump::BufferType::L1);
    op_buffer.set_address(static_cast<uint64_t>(l1_fusion_addr_));
    op_buffer.set_size(size);
    task.mutable_buffer()->Add(std::move(op_buffer));
    GELOGI("Generate op buffer succsess.");
    need_generate_op_buffer_ = false;
  }
}

Status DataDumper::ExecuteLoadDumpInfo(const toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_str;
  const size_t proto_size = op_mapping_info.ByteSizeLong();
  const bool ret = op_mapping_info.SerializeToString(&proto_str);
  if ((!ret) || (proto_size == 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Serialize proto to std::string fail");
    GELOGE(PARAM_INVALID, "[Call][SerializeToString] failed, proto size %zu.", proto_size);
    return PARAM_INVALID;
  }

  if (dev_mem_load_ != nullptr) {
    GELOGW("dev_mem_load_ has been used.");
    GE_FREE_RT_LOG(dev_mem_load_);
  }

  rtError_t rt_ret = rtMalloc(&dev_mem_load_, proto_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMalloc failed, size:%zu, ret:%d", proto_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:%d", proto_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "load dump information.", proto_size);

  rt_ret = rtMemcpy(dev_mem_load_, proto_size, proto_str.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpy failed, size:%zu, ret:%d", proto_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:%d", proto_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtDatadumpInfoLoad(dev_mem_load_, static_cast<uint32_t>(proto_size));
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtDatadumpInfoLoad failed, length:%zu, ret:%d", proto_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtDatadumpInfoLoad] failed, length:%zu, ret:%d", proto_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  load_flag_ = true;
  GELOGI("LoadDumpInfo success, proto size is: %zu.", proto_size);
  return SUCCESS;
}

Status DataDumper::ExecuteUnLoadDumpInfo(const toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_str;
  const size_t proto_size = op_mapping_info.ByteSizeLong();
  const bool ret = op_mapping_info.SerializeToString(&proto_str);
  if ((!ret) || (proto_size == 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Serialize proto to std::string fail");
    GELOGE(PARAM_INVALID, "[Call][SerializeToString] failed, proto size %zu.", proto_size);
    return PARAM_INVALID;
  }

  if (dev_mem_unload_ != nullptr) {
    GELOGW("dev_mem_unload_ has been used.");
    GE_FREE_RT_LOG(dev_mem_unload_);
  }

  rtError_t rt_ret = rtMalloc(&dev_mem_unload_, proto_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMalloc failed, size:%zu, ret:%d", proto_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:%d", proto_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "unload dump information.", proto_size);

  rt_ret = rtMemcpy(dev_mem_unload_, proto_size, proto_str.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpy failed, size:%zu, ret:%d", proto_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:%d", proto_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtDatadumpInfoLoad(dev_mem_unload_, static_cast<uint32_t>(proto_size));
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtDatadumpInfoLoad failed, length:%zu, ret:%d", proto_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtDatadumpInfoLoad] failed, length:%zu, ret:%d", proto_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  load_flag_ = false;
  GELOGI("UnloadDumpInfo success, proto size is: %zu.", proto_size);
  return SUCCESS;
}

Status DataDumper::GetDumpPath(std::string &dump_path) {
  dump_path = dump_properties_.GetDumpPath();
  if (op_print_list_.empty()) {
    dump_path += std::to_string(device_id_) + "/";
    GELOGD("Set dump_path[%s]", dump_path.c_str());
    return SUCCESS;
  }

  constexpr int32_t kDumpStatus = 0;
  // printf使用了dump流程，如果算子使能了printf，说明需要使用工具的dump
  if(AdxDataDumpServerInit() != kDumpStatus) {
    GELOGE(PARAM_INVALID, "[GetDumpPath][AdxDataDumpServer] failed.");
    return PARAM_INVALID;
  }

  if (dump_path.empty()) {
    GE_ASSERT_SUCCESS(GetAscendWorkPath(dump_path));
    if (dump_path.empty()) {
      ge::char_t curr_path[MMPA_MAX_PATH] = {};
      if (mmGetCwd(&curr_path[0], MMPA_MAX_PATH) != EN_OK) {
        GELOGW("get current path failed");
        return FAILED;
      }
      dump_path = std::string(curr_path);
    }
    dump_path += "/printf/";
  }
  dump_path += std::to_string(device_id_) + "/";
  GELOGD("Set dump_path[%s]", dump_path.c_str());
  return SUCCESS;
}

Status DataDumper::LoadDumpInfo() {
  std::string model_name;
  PrintCheckLog(model_name);
  GELOGI("model name %s, is_single_op_debug %d, op_list size %zu, model id %u", model_name.c_str(),
         is_single_op_debug_, op_list_.size(), model_id_);

  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  std::string dump_path = "";
  auto ret = GetDumpPath(dump_path);
  if (ret != SUCCESS) {
    return ret;
  }
  op_mapping_info.set_dump_path(dump_path);
  if (!is_single_op_debug_) {
    op_mapping_info.set_model_name(model_name);
    op_mapping_info.set_model_id(model_id_);
    op_mapping_info.set_dump_step(dump_properties_.GetDumpStep());
  }
  op_mapping_info.set_flag(kAicpuLoadFlag);
  const toolkit::aicpu::dump::DumpData dump_data = (dump_properties_.GetDumpData() == kDumpDataDefaultValue)
                                                 ? toolkit::aicpu::dump::DumpData::STATS_DUMP_DATA
                                                 : toolkit::aicpu::dump::DumpData::TENSOR_DUMP_DATA;
  op_mapping_info.set_dump_data(dump_data);
  SetOpMappingLoopAddr(global_step_, loop_per_iter_, loop_cond_, op_mapping_info);
  ret = BuildTaskInfo(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][TaskInfo] failed, ret:%u, path:%s", ret, dump_path.c_str());
    return ret;
  }

  ret = BuildTaskInfoForPrint(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][TaskInfoForPrint] failed, ret:%u, path:%s", ret, dump_path.c_str());
    return ret;
  }

  SetEndGraphIdToAicpu(op_mapping_info);

  SetOpDebugIdToAicpu(op_debug_task_id_, op_debug_stream_id_, op_debug_addr_, op_mapping_info);

  if ((!op_list_.empty()) || !op_print_list_.empty() || is_op_debug_ || is_end_graph_) {
    ret = ExecuteLoadDumpInfo(op_mapping_info);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Execute][LoadDumpInfo] failed, ret:%u", ret);
      return ret;
    }
    op_mapping_info_ = std::move(op_mapping_info);
  }
  return SUCCESS;
}

Status DataDumper::ReLoadDumpInfo() {
  if (op_mapping_info_.ByteSizeLong() == 0U) {
    GELOGI("No information saved when loading model[%u].", model_id_);
    return SUCCESS;
  }
  GELOGI("ReLoadDumpInfo model id %u.", model_id_);

  if (load_flag_) {
    GELOGW("Load the task for the same model[%u] more than once.", model_id_);
    return SUCCESS;
  }

  const auto dump_path = dump_properties_.GetDumpPath() + std::to_string(device_id_) + "/";
  GELOGI("Reload Dump path is %s, dump step is %s", dump_path.c_str(), dump_properties_.GetDumpStep().c_str());
  op_mapping_info_.set_dump_path(dump_path);
  if (!is_single_op_debug_) {
    op_mapping_info_.set_dump_step(dump_properties_.GetDumpStep());
  }
  SetOpMappingLoopAddr(global_step_, loop_per_iter_, loop_cond_, op_mapping_info_);
  auto ret = UpdateOpMappingInfo();
  GE_ASSERT_SUCCESS(ret, "[Execute][UpdateOpMappingInfo] failed, ret:%u, model id %u", ret, model_id_);
  ret = ExecuteLoadDumpInfo(op_mapping_info_);
  GE_ASSERT_SUCCESS(ret, "[Execute][ReLoadDumpInfo] failed, ret:%u, model id %u", ret, model_id_);

  return SUCCESS;
}

Status DataDumper::UnloadDumpInfoByModel(uint32_t model_id) {
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.set_model_id(model_id);
  op_mapping_info.set_flag(kAicpuUnloadFlag);
  GELOGI("UnloadDumpInfo model id %u.", model_id);

  std::string proto_str;
  const size_t proto_size = op_mapping_info.ByteSizeLong();
  const bool ret = op_mapping_info.SerializeToString(&proto_str);
  GE_ASSERT_TRUE((ret) && (proto_size != 0U));

  if (dev_mem_unload_for_model_ != nullptr) {
    GELOGW("dev_mem_unload_for_model_ has been used.");
    GE_FREE_RT_LOG(dev_mem_unload_for_model_);
  }

  GE_ASSERT_TRUE(rtMalloc(&dev_mem_unload_for_model_, proto_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16) == RT_ERROR_NONE);
  GE_ASSERT_TRUE(rtMemcpy(dev_mem_unload_for_model_, proto_size, proto_str.c_str(),
                          proto_size, RT_MEMCPY_HOST_TO_DEVICE) == RT_ERROR_NONE);
  GE_ASSERT_TRUE(rtDatadumpInfoLoad(dev_mem_unload_for_model_, static_cast<uint32_t>(proto_size)) == RT_ERROR_NONE);
  return SUCCESS;
}

void DataDumper::DumpWorkspace(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task,
                               const std::shared_ptr<OpDesc> &op_desc) const {
  std::vector<int64_t> space_type;
  const bool has_space_type = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, space_type);
  if (has_space_type) {
    bool has_memory_log = false;
    const auto result = std::find(space_type.begin(), space_type.end(), ge::AicpuWorkSpaceType::CUST_LOG);
    if (result != space_type.end()) {
      has_memory_log = true;
    }
    const std::vector<int64_t> v_workspace_size = op_desc->GetWorkspaceBytes();
    GELOGD("op[%s], has_memory_log[%u], workspace size[%zu], space_addr size[%zu], space_type size[%zu]",
           op_desc->GetName().c_str(), static_cast<uint32_t>(has_memory_log), v_workspace_size.size(),
           inner_dump_info.space_addr.size(), space_type.size());
    for (size_t i = 0U; has_memory_log && (i < v_workspace_size.size()) && (i < inner_dump_info.space_addr.size()) &&
         (i < space_type.size()); ++i) {
      if (space_type[i] == static_cast<int64_t>(ge::AicpuWorkSpaceType::CUST_LOG)) {
        toolkit::aicpu::dump::Workspace space;
        space.set_type(toolkit::aicpu::dump::Workspace::LOG);
        space.set_size(static_cast<uint64_t>(v_workspace_size[i]));
        GELOGI("workspace_info addr_size is: %zu %" PRIu64, inner_dump_info.space_addr.size(),
               inner_dump_info.space_addr[i]);
        space.set_data_addr(inner_dump_info.space_addr[i]);
        task.mutable_space()->Add(std::move(space));
      }
    }
  }
}

Status DataDumper::BuildTaskInfoForDumpOutput(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info,
                                              const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task) {
  const Status ret = DumpOutput(dump_info, task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Dump][Output] failed, ret:%u, op:%s", ret, dump_info.op->GetName().c_str());
    return ret;
  }
  DumpContext(dump_info, task);
  GenerateOpBuffer(kDumpL1FusionOpMByteSize, task);

  const auto &op_desc = dump_info.op;
  if (dump_properties_.IsWatcherNode(op_desc->GetName())) {
    for(const auto &iter : layer_op_on_watcher_mode_list_) {
      toolkit::aicpu::dump::Task task_watcher = task;
      task_watcher.set_task_id(iter.task_id);
      task_watcher.set_stream_id(iter.stream_id);
      string watcher_op_name = iter.op_desc->GetName() + "_To_" + op_desc->GetName();
      GELOGI("WatcherOpName %s stream_id %u task_id %u.", watcher_op_name.c_str(), iter.stream_id, iter.task_id);
      task_watcher.mutable_op()->set_op_name(watcher_op_name);
      op_mapping_info.mutable_task()->Add(std::move(task_watcher));
    }
  }
  op_mapping_info.mutable_task()->Add(std::move(task));

  return SUCCESS;
}

Status DataDumper::BuildTaskInfoForDumpInput(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info,
                                             const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task) {
  if (dump_info.is_task) {
    const Status ret = dump_info.is_raw_address ? DumpInputWithRawAddress(dump_info, task) : DumpInput(dump_info, task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input] failed, ret:%u, op:%s", ret, dump_info.op->GetName().c_str());
      return ret;
    }
  }
  DumpContext(dump_info, task);
  GenerateOpBuffer(kDumpL1FusionOpMByteSize, task);
  op_mapping_info.mutable_task()->Add(std::move(task));
  return SUCCESS;
}

Status DataDumper::BuildTaskInfoForDumpAllorOpDebug(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info,
                                                    const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task) {
  DumpWorkspace(dump_info, task, dump_info.op);
  auto ret = DumpOutput(dump_info, task);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Dump][Output] failed when in dumping all, ret:%u, op:%s", ret, dump_info.op->GetName().c_str());
    return ret;
  }
  if (dump_info.is_task) {
    ret = dump_info.is_raw_address ? DumpInputWithRawAddress(dump_info, task) : DumpInput(dump_info, task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input] failed when in dumping all, ret:%u, op:%s", ret, dump_info.op->GetName().c_str());
      return ret;
    }
  }
  DumpContext(dump_info, task);
  GenerateOpBuffer(kDumpL1FusionOpMByteSize, task);
  op_mapping_info.mutable_task()->Add(std::move(task));
  return SUCCESS;
}

void DataDumper::InitAicpuDumpTask(const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task) const {
  const auto &op_desc = dump_info.op;
  task.set_end_graph(false);
  task.set_task_id(dump_info.task_id);
  task.set_stream_id(dump_info.stream_id);
  task.mutable_op()->set_op_name(op_desc->GetName());
  task.mutable_op()->set_op_type(op_desc->GetType());
  bool is_fftsplus_task = false;
  (void)AttrUtils::GetBool(op_desc, "_is_fftsplus_task", is_fftsplus_task);

  if ((op_desc->HasAttr("current_context_id") || (is_fftsplus_task && dump_info.is_op_debug)) &&
      (dump_info.task_type != ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL)) {
    task.set_task_type(toolkit::aicpu::dump::Task::FFTSPLUS);
    task.set_context_id(dump_info.context_id);
    GELOGI("op[%s] set task type[FFTSPLUS], context id[%u], is_fftsplus_task: %d", op_desc->GetName().c_str(),
           dump_info.context_id, static_cast<int32_t>(is_fftsplus_task));
    const std::string* ffts_str_ptr = AttrUtils::GetStr(op_desc, ffts::kAttrSgtJsonInfo);
    if (ffts_str_ptr != nullptr && (!ffts_str_ptr->empty())) {
      toolkit::aicpu::dump::OpAttr op_attr;
      op_attr.set_name(ffts::kAttrSgtJsonInfo);
      op_attr.set_value(*ffts_str_ptr);
      task.mutable_attr()->Add(std::move(op_attr));
    }
  }
}

Status DataDumper::BuildTaskInfo(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  for (const auto &op_iter : op_list_) {
    const auto &op_desc = op_iter.op;
    GELOGI("Op %s add task to op_mapping_info, stream_id: %u, task_id:%u.", op_desc->GetName().c_str(),
           op_iter.stream_id, op_iter.task_id);
    toolkit::aicpu::dump::Task task;
    InitAicpuDumpTask(op_iter, task);

    Status ret;
    if (op_iter.task_type == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL) {
      DumpWorkspace(op_iter, task, op_iter.op);
      op_mapping_info.mutable_task()->Add(std::move(task));
      GELOGD("DumpWorkspace for op[%s], task_type is ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL", op_iter.op->GetName().c_str());
      continue;
    }
    if (dump_properties_.GetDumpMode() == kDumpOutput) {
      ret = BuildTaskInfoForDumpOutput(op_mapping_info, op_iter, task);
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }
    if (dump_properties_.GetDumpMode() == kDumpInput) {
      ret = BuildTaskInfoForDumpInput(op_mapping_info, op_iter, task);
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }
    if ((dump_properties_.GetDumpMode() == kDumpAll) || is_op_debug_) {
      ret = BuildTaskInfoForDumpAllorOpDebug(op_mapping_info, op_iter, task);
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }
  }
  return SUCCESS;
}

Status DataDumper::BuildTaskInfoForPrint(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  for (const auto &op_iter : op_print_list_) {
    const auto &op_desc = op_iter.op;
    GELOGD("Op %s in model begin to add task for print in op_mapping_info.", op_desc->GetName().c_str());
    toolkit::aicpu::dump::Task task;
    InitAicpuDumpTask(op_iter, task);

    const std::vector<int64_t> v_workspace_size = op_desc->GetWorkspaceBytes();
    GELOGI("dump op[%s] workspace for print in static model scenario", op_desc->GetName().c_str());
    int64_t buffer_size = 0;
    const std::string kOpDfxBufferSize = "_op_dfx_buffer_size";
    if (ge::AttrUtils::GetInt(op_desc, kOpDfxBufferSize, buffer_size) && buffer_size > 0) {
      if (v_workspace_size.empty() || buffer_size > v_workspace_size[0]) {
        GELOGE(PARAM_INVALID, "[Check][Param] v_workspace_size is empty or buffer_size > v_workspace_size[0], "
               "v_workspace_size[%zu], buffer_size[%" PRId64 "]", v_workspace_size.size(), buffer_size);
        return PARAM_INVALID;
      }
      GELOGI("v_workspace_size[0] is [%" PRId64 "], buffer_size is [%" PRId64 "],", v_workspace_size[0], buffer_size);
      toolkit::aicpu::dump::Workspace space;
      space.set_type(toolkit::aicpu::dump::Workspace::LOG);
      space.set_size(static_cast<uint64_t>(buffer_size));
      if (!op_iter.space_addr.empty()) {
        GELOGI("the size of op_iter.space_addr is [%zu], op_iter.space_addr[0] is [0x%llx]",
               op_iter.space_addr.size(), op_iter.space_addr[0]);
        if (op_desc->HasAttr("current_context_id")) {
          toolkit::aicpu::dump::Context ffts_context;
          ffts_context.set_context_id(op_iter.context_id);
          ffts_context.set_thread_id(op_iter.thread_id);
          task.mutable_context()->Add(std::move(ffts_context));
        }
        space.set_data_addr(op_iter.space_addr[0]);
        task.mutable_space()->Add(std::move(space));
        op_mapping_info.mutable_task()->Add(std::move(task));
      }
    }
  }
  return SUCCESS;
}

void DataDumper::SetEndGraphIdToAicpu(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  if ((dump_properties_.GetDumpMode() == kDumpOutput) || (dump_properties_.GetDumpMode() == kDumpInput) ||
      (dump_properties_.GetDumpMode() == kDumpAll)) {
    toolkit::aicpu::dump::Task task;
    task.set_end_graph(true);
    task.set_task_id(end_graph_task_id_);
    task.set_stream_id(end_graph_stream_id_);
    task.mutable_op()->set_op_name(NODE_NAME_END_GRAPH);
    task.mutable_op()->set_op_type(ENDGRAPH);
    op_mapping_info.mutable_task()->Add(std::move(task));

    is_end_graph_ = true;
    if (op_mapping_info.model_name_param_case() == toolkit::aicpu::dump::OpMappingInfo::kModelName) {
      GELOGI("Add end_graph_info to aicpu, model_name is %s, task_id is %u, stream_id is %u",
          op_mapping_info.model_name().c_str(), end_graph_task_id_, end_graph_stream_id_);
      return;
    }
    GELOGI("Add end_graph_info to aicpu, task_id is %u, stream_id is %u", end_graph_task_id_, end_graph_stream_id_);
  }
}

void DataDumper::SetOpDebugIdToAicpu(const uint32_t task_id, const uint32_t stream_id, const void *const op_debug_addr,
                                     toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) const {
  if (is_op_debug_) {
    GELOGI("add op_debug_info to aicpu, task_id is %u, stream_id is %u.", task_id, stream_id);
    toolkit::aicpu::dump::Task task;
    task.set_end_graph(false);
    task.set_task_id(task_id);
    task.set_stream_id(stream_id);
    task.mutable_op()->set_op_name(NODE_NAME_OP_DEBUG);
    task.mutable_op()->set_op_type(OP_TYPE_OP_DEBUG);

    // set output
    toolkit::aicpu::dump::Output output;
    output.set_data_type(DT_UINT8);
    output.set_format(FORMAT_ND);

    output.mutable_shape()->add_dim(kOpDebugShape);

    output.set_original_name(NODE_NAME_OP_DEBUG);
    output.set_original_output_index(0);
    output.set_original_output_format(FORMAT_ND);
    output.set_original_output_data_type(DT_UINT8);
    // due to lhisi virtual addr bug, cannot use args now
    output.set_address(PtrToValue(op_debug_addr));
    output.set_size(kOpDebugSize);
    output.set_addr_type(toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR);

    task.mutable_output()->Add(std::move(output));
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
}

void DataDumper::UnloadDumpInfo() {
  if (!load_flag_) {
    return;
  }

  GELOGI("UnloadDumpInfo start, model_id: %u.", model_id_);
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.set_model_id(model_id_);
  op_mapping_info.set_flag(kAicpuUnloadFlag);

  for (const auto &op_iter : op_list_) {
    toolkit::aicpu::dump::Task task;
    task.set_task_id(op_iter.task_id);
    task.set_stream_id(op_iter.stream_id);
    op_mapping_info.mutable_task()->Add(std::move(task));
  }

  for (const auto &op_iter : op_print_list_) {
    toolkit::aicpu::dump::Task task;
    task.set_task_id(op_iter.task_id);
    task.set_stream_id(op_iter.stream_id);
    op_mapping_info.mutable_task()->Add(std::move(task));
  }

  const auto ret = ExecuteUnLoadDumpInfo(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][UnLoadDumpInfo] failed, ret:%d", ret);
  }
}

void DataDumper::DumpShrink() {
  compute_graph_.reset();
  input_map_.clear();
  ref_info_.clear();
}

void DataDumper::PrintCheckLog(std::string &dump_list_key) {
  std::set<std::string> model_list = dump_properties_.GetAllDumpModel();
  const bool not_find_by_omname = model_list.find(om_name_) == model_list.end();
  const bool not_find_by_modelname = model_list.find(model_name_) == model_list.cend();
  dump_list_key = not_find_by_omname ? model_name_ : om_name_;
  GELOGI("%zu op need dump in known shape model %s.", op_list_.size(), dump_list_key.c_str());

  if (model_list.find(DUMP_ALL_MODEL) == model_list.end()) {
    if (not_find_by_omname && not_find_by_modelname) {
      std::string model_list_str;
      for (auto &model : model_list) {
        model_list_str += "[" + model + "].";
      }

      GELOGW("Model %s will not be set to dump, dump list: %s", dump_list_key.c_str(), model_list_str.c_str());
      return;
    }
  }

  const std::set<std::string> &config_dump_op_list = dump_properties_.GetPropertyValue(dump_list_key);
  std::set<std::string> dump_op_list;
  for (auto &inner_dump_info : op_list_) {
    // oplist value OpDescPtr is not nullptr
    (void)dump_op_list.insert(inner_dump_info.op->GetName());
  }

  for (auto &dump_op : config_dump_op_list) {
    if (dump_op_list.find(dump_op) == dump_op_list.end()) {
      GELOGW("Op %s set to dump but does not exist in model %s or not a valid op.", dump_op.c_str(), dump_list_key.c_str());
    }
  }
}

Status DataDumper::UpdateOpMappingInfo() {
  if (!is_op_debug_) {
    return ge::SUCCESS;
  }
  GELOGI("update op_debug_info to aicpu, task_id is %u, stream_id is %u.", op_debug_task_id_, op_debug_stream_id_);
  for (int32_t i = 0; i < op_mapping_info_.task_size(); i++) {
    toolkit::aicpu::dump::Task *task = op_mapping_info_.mutable_task(i);
    if (task->op().op_name() == NODE_NAME_OP_DEBUG) {
      task->set_stream_id(op_debug_stream_id_);
      task->set_task_id(op_debug_task_id_);
      auto item = task->mutable_output(0);
      GE_ASSERT_NOTNULL(item);
      item->set_address(PtrToValue(op_debug_addr_));
      item->set_size(kOpDebugSize);
      return ge::SUCCESS;
    }
  }
  GELOGE(FAILED, "Node_OpDebug task is not found.");
  return ge::FAILED;
}
}  // namespace ge
