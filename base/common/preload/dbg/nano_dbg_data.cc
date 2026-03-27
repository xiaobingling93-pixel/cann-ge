/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/dbg/nano_dbg_data.h"
#include <utility>
#include <tuple>
#include <numeric>
#include "common/preload/model/pre_model_utils.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/tlv/nano_dbg_desc.h"
#include "graph/utils/node_utils.h"
#include "base/err_msg.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
const std::set<std::string> kIoNodeTypes{DATA, AIPPDATA, ANN_DATA, QUEUE_DATA, NETOUTPUT};
const std::set<std::string> kWeightNodeTypes{CONSTANT, CONSTANTOP};
static std::atomic<std::uint32_t> g_task_id(0U);
void GetOpDataKernel(const domi::TaskDef &task_def, uint32_t &id, toolkit::aicpu::dump::Task::TaskType &task_type) {
  id = task_def.kernel().context().op_index();
  task_type = toolkit::aicpu::dump::Task::AICORE;
}

void GetOpDataKernelEx(const domi::TaskDef &task_def, uint32_t &id, toolkit::aicpu::dump::Task::TaskType &task_type) {
  id = task_def.kernel_ex().op_index();
  task_type = toolkit::aicpu::dump::Task::AICPU;
}

void GetOpDataAllKernel(const domi::TaskDef &task_def, uint32_t &id, toolkit::aicpu::dump::Task::TaskType &task_type) {
  id = task_def.kernel_with_handle().context().op_index();
  task_type = toolkit::aicpu::dump::Task::AICORE;
}

void GetOpDataDsa(const domi::TaskDef &task_def, uint32_t &id, toolkit::aicpu::dump::Task::TaskType &task_type) {
  id = task_def.dsa_task().op_index();
  task_type = toolkit::aicpu::dump::Task::DSA;
}

using GetOpDataFunc = std::function<void(const domi::TaskDef &, uint32_t &, toolkit::aicpu::dump::Task::TaskType &)>;
static const std::map<ModelTaskType, GetOpDataFunc> task_map = {
    {ModelTaskType::MODEL_TASK_KERNEL, &GetOpDataKernel},
    {ModelTaskType::MODEL_TASK_KERNEL_EX, &GetOpDataKernelEx},
    {ModelTaskType::MODEL_TASK_ALL_KERNEL, &GetOpDataAllKernel},
    {ModelTaskType::MODEL_TASK_DSA, &GetOpDataDsa},
};
}  // namespace

NanoDbgData::NanoDbgData(const GeModelPtr &ge_model, const std::unordered_map<int64_t, uint32_t> zerocopy_info)
    : ge_model_(ge_model), zerocopy_info_(zerocopy_info) {
  input_mem_types_.clear();
  output_mem_types_.clear();
}

Status NanoDbgData::Init() {
  GE_CHECK_NOTNULL(ge_model_);
  GELOGD("Init Nano dbg in");
  InitNodes();
  GE_CHK_STATUS_RET(InitDbgData(), "[Call][InitDbgData] failed");
  GE_CHK_STATUS_RET(InitDbgTlv(), "[Call][InitDbgTlv] failed");
  GELOGD("Init Nano dbg out");
  return SUCCESS;
}

Status NanoDbgData::InitDbgData() {
  model_name_ = ge_model_->GetName();
  const auto &model_task_def_ptr = ge_model_->GetModelTaskDefPtr();
  GE_CHECK_NOTNULL(model_task_def_ptr);

  if (model_task_def_ptr->task_size() == 0) {
    GELOGW("model has no task info.");
    return SUCCESS;
  }
  domi::ModelTaskDef &model_task_def = *model_task_def_ptr.get();
  GELOGI("task size %d", model_task_def.task().size());

  for (int32_t i = 0U; i < model_task_def.task_size(); ++i) {
    const domi::TaskDef &task_def = model_task_def.task(i);
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    const auto iter = task_map.find(task_type);
    if (iter == task_map.end()) {
      GELOGD("Skip task type:%d", task_def.type());
      continue;
    }
    model_task_def_ptr->mutable_task(i)->set_id(g_task_id++);
    GE_CHK_STATUS_RET(AddDbgOp(task_def), "[Call][AddDbgOp] failed");
  }
  return SUCCESS;
}

Status NanoDbgData::AddDbgOp(const domi::TaskDef &task_def) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  const auto iter = task_map.find(task_type);
  if (iter == task_map.end()) {
    GELOGD("Skip task type:%d", task_def.type());
    return SUCCESS;
  }

  uint32_t op_index;
  NanoDbgOpDesc dbg_op = {};
  iter->second(task_def, op_index, dbg_op.task_type);
  GELOGD("task type:%d, op_index:%u", task_def.type(), op_index);
  const OpDescPtr &op_desc = GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);

  dbg_op.logic_stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
  dbg_op.task_id = task_def.id();
  dbg_op.op_name = op_desc->GetName();
  dbg_op.op_type = op_desc->GetType();
  dbg_op.block_dim = task_def.kernel().block_dim();

  (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, dbg_op.original_op_names);
  // no need to care return value of AttrUtils and use default value when return failed
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_DATA_DUMP_IS_MULTIOP, dbg_op.datadump_is_multiop);
  const std::string* L1_fusion_sub_graph_no_ptr = AttrUtils::GetStr(op_desc, "_L1_fusion_sub_graph_no");
  if (L1_fusion_sub_graph_no_ptr != nullptr) {
    dbg_op.L1_fusion_sub_graph_no = *L1_fusion_sub_graph_no_ptr;
  }

  GE_CHK_STATUS_RET(AddDbgInput(op_desc, dbg_op, op_index), "[Add][DbgInput] failed");
  GE_CHK_STATUS_RET(AddDbgBuffer(dbg_op), "[Add][DbgBuffer] for input failed");
  GE_CHK_STATUS_RET(AddDbgOutput(op_desc, dbg_op, op_index), "[Add][DbgOutput] failed");
  GE_CHK_STATUS_RET(AddDbgBuffer(dbg_op), "[Add][DbgBuffer] for output failed");
  GE_CHK_STATUS_RET(AddDbgWorkspace(op_desc, dbg_op), "[Add][DbgWorkspace] failed");
  GE_CHK_STATUS_RET(AddDbgMemInfo(op_desc, dbg_op), "[Add][DbgMemInfo] failed");
  (void)op_list_.emplace_back(dbg_op);
  return SUCCESS;
}

Status NanoDbgData::GenDbgInput(const GeTensorDesc &tensor_desc, NanoDbgInputDesc &dbg_input) const {
  int64_t input_size = 0;
  if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_size)) {
    GELOGI("Get aipp input size according to attr is %ld", input_size);
  } else {
    if (TensorUtils::GetTensorSizeInBytes(tensor_desc, input_size) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get tensor size failed");
      GELOGE(PARAM_INVALID, "[Get][TensorSize] failed");
      return PARAM_INVALID;
    }
  }
  GELOGD("Get input size in dump is %ld", input_size);
  dbg_input.size = static_cast<uint64_t>(input_size);
  dbg_input.data_type = tensor_desc.GetDataType();
  dbg_input.format = GetPrimaryFormat(static_cast<int32_t>(tensor_desc.GetFormat()));
  dbg_input.shape_dims = tensor_desc.GetShape().GetDims();
  dbg_input.original_shape_dims = tensor_desc.GetOriginShape().GetDims();

  if (dbg_input.addr_type == toolkit::aicpu::dump::AddressType::NANO_IO_ADDR) {
    const int64_t offset = static_cast<int64_t>(dbg_input.addr);
    auto const founded = std::find_if(zerocopy_info_.begin(), zerocopy_info_.end(), NanoZeroCopyValueToKey(offset));
    if (founded != zerocopy_info_.end()) {
      GELOGD("[DEBUG] offset[%ld] is a zerocopy io offset, type[%d], real offset[%lu].", offset, dbg_input.addr_type,
             founded->first);
    }
  } else {
    GELOGD("[DEBUG] offset[%lu] is a type[%d] offset.", dbg_input.addr, dbg_input.addr_type);
  }
  GELOGD("GenDbgInput out");
  return SUCCESS;
}

Status NanoDbgData::AddDbgInput(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op, const uint32_t &op_index) {
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  const auto &input_descs = op_desc->GetAllInputsDescPtr();
  auto const addrs_type = input_mem_types_.find(static_cast<int64_t>(op_index));
  GE_ASSERT_TRUE(addrs_type != input_mem_types_.end(),
                 "op[%s] lost input addr type info", op_desc->GetName().c_str());
  const std::vector<int64_t> v_input_offset = op_desc->GetInputOffset();
  GELOGD("op[%s] v_input_offset.size=%zu, input_descs.size=%zu, addrs_type size=%zu", op_desc->GetName().c_str(),
         v_input_offset.size(), input_descs.size(), addrs_type->second.size());
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE((has_mem_type_attr && (v_memory_type.size() != input_descs.size())) ||
                                       (addrs_type->second.size() < input_descs.size()) ||
                                       (v_input_offset.size() != input_descs.size()),
                                       "[Check][Param] AddDbgInput[%s], input size[%zu], input memory type size[%zu], "
                                       "input addr type size[%zu], v_input_offset size[%zu]",
                                       op_desc->GetName().c_str(), input_descs.size(), v_memory_type.size(),
                                       addrs_type->second.size(), v_input_offset.size());

  for (size_t i = 0U; i < input_descs.size(); ++i) {
    if (has_mem_type_attr && ((v_memory_type[i] == static_cast<int64_t>(RT_MEMORY_L1))
        || (v_memory_type[i] == static_cast<int64_t>(kRtMemoryUB)))) {
      need_generate_op_buffer_ = true;
      GELOGI("[L1Fusion] AddDbgInput[%s] input[%zu] is l1 addr, skip dump", op_desc->GetName().c_str(), i);
      continue;
    }
    if (addrs_type->second[i] == -1) {
      GELOGI("[AddDbgInput] AddDbgInput[%s] input[%zu] is suspended, skip dump", op_desc->GetName().c_str(), i);
      continue;
    }
    NanoDbgInputDesc dbg_input = {};
    dbg_input.addr_type = static_cast<toolkit::aicpu::dump::AddressType>(addrs_type->second[i]);
    dbg_input.addr = static_cast<uint64_t>(v_input_offset.at(i));
    GE_CHK_STATUS_RET(GenDbgInput(*input_descs.at(i), dbg_input), "[GenDbgInput] Gen failed.");

    std::vector<uint64_t> task_addr_offset;
    task_addr_offset = op_desc->TryGetExtAttr("task_addr_offset", task_addr_offset);
    if ((!task_addr_offset.empty()) && (i < task_addr_offset.size())) {
      dbg_input.offset = task_addr_offset[i];
    }
    GELOGI("set op[%s] input info, addr type[%u], offset[%lu]", op_desc->GetName().c_str(), dbg_input.addr_type,
           dbg_input.addr);
    (void)dbg_op.input_list.emplace_back(dbg_input);
  }
  return SUCCESS;
}

Status NanoDbgData::GenDbgOutput(const GeTensorDesc &tensor_desc, NanoDbgOutputDesc &dbg_output) const {
  dbg_output.data_type = tensor_desc.GetDataType();
  dbg_output.format = GetPrimaryFormat(static_cast<int32_t>(tensor_desc.GetFormat()));
  dbg_output.shape_dims = tensor_desc.GetShape().GetDims();

  int64_t output_size = 0;
  if (TensorUtils::GetTensorSizeInBytes(tensor_desc, output_size) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get tensor size failed");
    GELOGE(PARAM_INVALID, "[Get][TensorSize] failed");
    return PARAM_INVALID;
  }

  GELOGD("Get output size in dump is %ld", output_size);
  dbg_output.size = static_cast<uint64_t>(output_size);
  int32_t origin_output_index = -1;
  const std::string* origin_name_ptr = AttrUtils::GetStr(tensor_desc, ATTR_NAME_DATA_DUMP_ORIGIN_NAME);
  if (origin_name_ptr != nullptr) {
    dbg_output.original_name = *origin_name_ptr;
  }
  (void)AttrUtils::GetInt(tensor_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
  dbg_output.original_index = origin_output_index;
  dbg_output.original_data_type = static_cast<int32_t>(tensor_desc.GetOriginDataType());
  dbg_output.original_format = GetPrimaryFormat(static_cast<int32_t>(tensor_desc.GetOriginFormat()));
  dbg_output.original_shape_dims = tensor_desc.GetOriginShape().GetDims();
  if (dbg_output.addr_type == toolkit::aicpu::dump::AddressType::NANO_IO_ADDR) {
    const int64_t offset = static_cast<int64_t>(dbg_output.addr);
    auto const founded = std::find_if(zerocopy_info_.begin(), zerocopy_info_.end(), NanoZeroCopyValueToKey(offset));
    if (founded != zerocopy_info_.end()) {
      GELOGD("[DEBUG] offset[%ld] is a zerocopy offset, type[%d], real offset[%lu].", offset, dbg_output.addr_type,
             founded->first);
    }
  } else {
    GELOGD("[DEBUG] offset[%lu] is a type[%d] offset.", dbg_output.addr, dbg_output.addr_type);
  }
  return SUCCESS;
}

Status NanoDbgData::AddDbgOutput(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op, const uint32_t &op_index) {
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  const auto &output_descs = op_desc->GetAllOutputsDescPtr();
  auto const addrs_type = output_mem_types_.find(static_cast<int64_t>(op_index));
  GE_ASSERT_TRUE(addrs_type != output_mem_types_.end(),
                 "op[%s] lost output addr type info", op_desc->GetName().c_str());
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GELOGD("op[%s] v_output_offset.size=%zu, output_descs.size=%zu, addrs_type size=%zu", op_desc->GetName().c_str(),
         v_output_offset.size(), output_descs.size(), addrs_type->second.size());
  GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE((has_mem_type_attr && (v_memory_type.size() != output_descs.size())) ||
                                       (addrs_type->second.size() < output_descs.size()) ||
                                       (v_output_offset.size() != output_descs.size()),
                                       "[Check][Param] AddDbgOutput[%s], output size[%zu], output memory type "
                                       "size[%zu], output addr type size[%zu], v_output_offset size[%zu]",
                                       op_desc->GetName().c_str(), output_descs.size(), v_memory_type.size(),
                                       addrs_type->second.size(), v_output_offset.size());

  for (size_t i = 0U; i < output_descs.size(); ++i) {
    if (has_mem_type_attr && ((v_memory_type[i] == static_cast<int64_t>(RT_MEMORY_L1))
        || (v_memory_type[i] == static_cast<int64_t>(kRtMemoryUB)))) {
      need_generate_op_buffer_ = true;
      GELOGI("[L1Fusion] AddDbgOutput[%s] output[%zu] is l1 addr, skip dump", op_desc->GetName().c_str(), i);
      continue;
    }
    const auto &output_tensor = *output_descs.at(i);
    if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(output_tensor)) {
      GELOGD("Node[%s], output[index:%zu] [name:%s] is an optional output, don't need to dump this output.",
             op_desc->GetName().c_str(), i, output_tensor.GetName().c_str());
      continue;
    }
    if (addrs_type->second[i] == -1) {
      GELOGI("[AddDbgOutput] AddDbgOutput[%s] output[%zu] is suspended, skip dump", op_desc->GetName().c_str(), i);
      continue;
    }
    NanoDbgOutputDesc dbg_output = {};
    dbg_output.addr_type = static_cast<toolkit::aicpu::dump::AddressType>(addrs_type->second[i]);
    dbg_output.addr = static_cast<uint64_t>(v_output_offset.at(i));
    GE_CHK_STATUS_RET(GenDbgOutput(output_tensor, dbg_output), "[GenDbgOutput] Gen failed.");

    std::vector<uint64_t> task_addr_offset;
    task_addr_offset = op_desc->TryGetExtAttr("task_addr_offset", task_addr_offset);
    if ((!task_addr_offset.empty()) && (i < task_addr_offset.size())) {
      dbg_output.offset = task_addr_offset[i];
    }
    GELOGI("set op[%s] output info, addr type[%u], offset[%lu]", op_desc->GetName().c_str(), dbg_output.addr_type,
           dbg_output.addr);
    (void)dbg_op.output_list.emplace_back(dbg_output);
  }
  return SUCCESS;
}

Status NanoDbgData::AddDbgWorkspace(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op) const {
  std::vector<int64_t> space_type;
  const bool has_space_type = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, space_type);
  bool has_memory_log = false;
  if (has_space_type) {
    const auto result = std::find(space_type.begin(), space_type.end(), ge::AicpuWorkSpaceType::CUST_LOG);
    if (result != space_type.end()) {
      has_memory_log = true;
    }
  }
  const auto v_workspace_size = op_desc->GetWorkspaceBytes();
  for (size_t i = 0U; has_memory_log && (i < v_workspace_size.size()); ++i) {
    GELOGI("workspace_info[%zu] size:%zu", i, v_workspace_size[i]);
    NanoDbgWorkspaceDesc dbg_workspace = {};
    dbg_workspace.size = static_cast<uint64_t>(v_workspace_size[i]);
    dbg_workspace.type = toolkit::aicpu::dump::Workspace::LOG;
    (void)dbg_op.workspace_list.emplace_back(dbg_workspace);
  }
  return SUCCESS;
}

Status NanoDbgData::AddDbgBuffer(NanoDbgOpDesc &dbg_op) {
  if (need_generate_op_buffer_) {
    NanoDbgBufferDesc dbg_buffer = {};
    dbg_buffer.type = toolkit::aicpu::dump::BufferType::L1;
    dbg_buffer.size = kDumpL1FusionOpMByteSize;
    (void)dbg_op.buffer_list.emplace_back(dbg_buffer);
    GELOGI("Generate op buffer success");
    need_generate_op_buffer_ = false;
  }
  return SUCCESS;
}

Status NanoDbgData::AddDbgMemInfo(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op) const {
  const auto input_size = PreModelUtils::GetInputSize(op_desc);
  const auto output_size = PreModelUtils::GetOutputSize(op_desc);
  const auto workspace_size = PreModelUtils::GetWorkspaceSize(op_desc);
  const auto weight_size = PreModelUtils::GetWeightSize(op_desc);

  NanoDbgMemInfoDesc dbg_mem = {};
  dbg_mem.input_mem_size = static_cast<uint64_t>(std::accumulate(input_size.begin(), input_size.end(), 0LL));
  dbg_mem.output_mem_size = static_cast<uint64_t>(std::accumulate(output_size.begin(), output_size.end(), 0LL));
  dbg_mem.workspace_mem_size =
      static_cast<uint64_t>(std::accumulate(workspace_size.begin(), workspace_size.end(), 0LL));
  dbg_mem.weight_mem_size = static_cast<uint64_t>(std::accumulate(weight_size.begin(), weight_size.end(), 0LL));
  dbg_mem.total_mem_size =
      dbg_mem.input_mem_size + dbg_mem.output_mem_size + dbg_mem.workspace_mem_size + dbg_mem.weight_mem_size;
  (void)dbg_op.mem_info_list.emplace_back(dbg_mem);
  return SUCCESS;
}

void NanoDbgData::InitNodes() {
  const auto nodes = ge_model_->GetGraph()->GetAllNodes();
  for (size_t i = 0U; i < nodes.size(); ++i) {
    const auto &node = nodes.at(i);
    const auto &op_desc = node->GetOpDesc();
    GE_RT_VOID_CHECK_NOTNULL(op_desc);
    op_map_[static_cast<uint32_t>(op_desc->GetId())] = op_desc;
    GenMemType(op_desc->GetId(), node);
  }
}

void NanoDbgData::GenMemType(const int64_t id, const ge::NodePtr &node) {
  GELOGI("GenMemType for node[%s], id[%d]", node->GetName().c_str(), id);
  size_t size = node->GetAllInDataAnchorsSize();
  const auto &op_desc = node->GetOpDesc();
  for (size_t i = 0UL; i < size; i++) {
    const auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    const auto input_desc_ptr = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    if ((in_anchor != nullptr) && (input_desc_ptr != nullptr)) {
      if (in_anchor->GetPeerOutAnchor() == nullptr) {
        GELOGD("GenMemType for node[%s], id[%d], size[%zu] input index[%zu] suspended", node->GetName().c_str(), id, size, i);
        input_mem_types_[id].push_back(-1);
        continue;
      }
      const auto in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
      const auto true_node = NodeUtils::GetInNodeCrossSubgraph(in_node);
      SaveMemType(input_mem_types_, id, true_node);
    }
  }
  size = node->GetAllOutDataAnchorsSize();
  for (size_t i = 0UL; i < size; i++) {
    const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(i));
    const auto output_desc_ptr = op_desc->GetOutputDescPtr(static_cast<uint32_t>(i));
    if ((out_anchor != nullptr) && (output_desc_ptr != nullptr)) {
      if (out_anchor->GetPeerInDataAnchorsPtr().empty()) {
        GELOGD("GenMemType for node[%s], id[%d], size[%zu] output index[%zu] suspended", node->GetName().c_str(), id, size, i);
        output_mem_types_[id].push_back(-1);
        continue;
      }
      for (const auto &in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
        if (in_anchor == nullptr) {
          continue;
        }
        const auto out_node = in_anchor->GetOwnerNode();
        SaveMemType(output_mem_types_, id, out_node);
        break;
      }
    }
  }
}

void NanoDbgData::SaveMemType(std::map<int64_t, std::vector<int32_t>> &mem_types,
                              const int64_t id, const ge::NodePtr &node) const {
  if (kIoNodeTypes.count(node->GetType()) > 0U) {
    mem_types[id].push_back(toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);  // NANO_IO_ADDR
  } else if (kWeightNodeTypes.count(node->GetType()) > 0U) {
    mem_types[id].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);  // NANO_WEIGHT_ADDR
  } else {
    mem_types[id].push_back(toolkit::aicpu::dump::AddressType::NANO_WORK_ADDR);  // NANO_WORK_ADDR
  }
  GELOGI("[DbgDescPart] node_id[%" PRId64 "], peer_node[%s], anchor_id[%zu], peer_type[%s]", id,
         node->GetName().c_str(), mem_types[id].size() - 1UL, node->GetType().c_str());
}

Status NanoDbgData::InitDbgTlv() {
  GenDbgPartitionLen();

  buff_.resize(buff_size_);
  des_addr_ = buff_.data();
  des_size_ = buff_size_;
  GELOGD("[DbgDescPart]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  GE_CHK_STATUS_RET(SaveDbgPartition(), "[DbgData] save failed");
  return SUCCESS;
}

void NanoDbgData::GenOpOriNameLen(const std::vector<string> &name_list) {
  buff_size_ += sizeof(TlvHead);
  for (const auto &name : name_list) {
    // L3 tlv shape dims
    buff_size_ += name.size();
  }
  buff_size_ += name_list.size() * sizeof(uint32_t);
}

void NanoDbgData::GenInputDescLen(const std::vector<NanoDbgInputDesc> &input_list) {
  buff_size_ += sizeof(TlvHead);
  buff_size_ += sizeof(DbgInputDescTlv2);
  buff_size_ += sizeof(DbgInputDescParamTlv2) * input_list.size();
  for (const auto &tensor : input_list) {
    // L3 tlv shape dims
    buff_size_ += sizeof(TlvHead);
    buff_size_ += sizeof(int64_t) * tensor.shape_dims.size();
    // L3 tlv original shape dims
    buff_size_ += sizeof(TlvHead);
    buff_size_ += sizeof(int64_t) * tensor.original_shape_dims.size();
  }
}

void NanoDbgData::GenOutputDescLen(const std::vector<NanoDbgOutputDesc> &output_list) {
  buff_size_ += sizeof(TlvHead);
  buff_size_ += sizeof(DbgOutputDescTlv2);
  buff_size_ += sizeof(DbgOutputDescParamTlv2) * output_list.size();
  for (const auto &tensor : output_list) {
    // L3 tlv origin name
    buff_size_ += sizeof(TlvHead);
    buff_size_ += tensor.original_name.size();
    // L3 tlv shape dims
    buff_size_ += sizeof(TlvHead);
    buff_size_ += sizeof(int64_t) * tensor.shape_dims.size();
    // L3 tlv original shape dims
    buff_size_ += sizeof(TlvHead);
    buff_size_ += sizeof(int64_t) * tensor.original_shape_dims.size();
  }
}

void NanoDbgData::GenWorkspaceDescLen(const std::vector<NanoDbgWorkspaceDesc> &workspace_list) {
  buff_size_ += sizeof(TlvHead);
  buff_size_ += sizeof(DbgWorkspaceDescTlv2);
  buff_size_ += sizeof(DbgWorkspaceDescParamTlv2) * workspace_list.size();
}

void NanoDbgData::GenDbgPartitionLen() {
  buff_size_ += sizeof(DbgDataHead);

  // L1 tlv
  buff_size_ += sizeof(TlvHead);
  buff_size_ += model_name_.size();

  // L1 tlv
  buff_size_ += sizeof(TlvHead);
  buff_size_ += sizeof(DbgOpDescTlv1);

  for (size_t i = 0U; i < op_list_.size(); i++) {
    buff_size_ += sizeof(DbgOpDescParamTlv1);

    // L2 tlv op name
    buff_size_ += sizeof(TlvHead);
    buff_size_ += op_list_[i].op_name.size();

    // L2 tlv original op name
    GenOpOriNameLen(op_list_[i].original_op_names);

    // L2 tlv op type
    buff_size_ += sizeof(TlvHead);
    buff_size_ += op_list_[i].op_type.size();

    // L2 tlv l1 fusion sub graph no
    buff_size_ += sizeof(TlvHead);
    buff_size_ += op_list_[i].L1_fusion_sub_graph_no.size();

    // L2 tlv mem info
    buff_size_ += sizeof(TlvHead);
    buff_size_ += sizeof(DbgOpMemInfosTlv2);
    buff_size_ += op_list_[i].mem_info_list.size() * sizeof(DbgOpMemInfoTlv2);

    // L2 tlv mem buffer
    buff_size_ += sizeof(TlvHead);
    buff_size_ += sizeof(DbgOpBufTlv2);
    buff_size_ += op_list_[i].buffer_list.size() * sizeof(DbgOpBufParamTlv2);

    // L2 tlv input desc
    GenInputDescLen(op_list_[i].input_list);

    // L2 tlv output desc
    GenOutputDescLen(op_list_[i].output_list);

    // L2 tlv workspace desc
    GenWorkspaceDescLen(op_list_[i].workspace_list);
  }
  GELOGD("[DbgData]buff_size_:%u", buff_size_);
}

Status NanoDbgData::SaveDbgHead() {
  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  DbgDataHead header;
  header.version_id = 0U;
  header.magic = DBG_DATA_HEAD_MAGIC;
  const errno_t ret = memcpy_s(des_addr_, des_size_, &header, sizeof(header));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(header)));
  des_size_ -= sizeof(header);
  GELOGD("[DbgData]end save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgStrTlv(const string &str, const uint32_t type) {
  if (str.size() == 0U) {
    GELOGD("[DbgData] string size is zero, type:%u", type);
    return SUCCESS;
  }

  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = type;
  tlv_head.len = static_cast<uint32_t>(str.size());
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);
  ret = memcpy_s(des_addr_, des_size_, str.data(), str.size());
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + str.size()));
  des_size_ -= str.size();
  return SUCCESS;
}

Status NanoDbgData::SaveDbgVecTlv(const std::vector<string> &vec, const uint32_t type) {
  if (vec.size() == 0U) {
    GELOGD("[DbgData] string size is zero, type:%u", type);
    return SUCCESS;
  }

  GELOGD("[DbgData] begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = type;
  tlv_head.len = 0U;
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);
  const uint32_t begin_len = static_cast<uint32_t>(des_size_);

  for (const auto &str : vec) {
    const uint32_t str_len = static_cast<uint32_t>(str.size());
    ret = memcpy_s(des_addr_, des_size_, &str_len, sizeof(uint32_t));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(uint32_t)));
    des_size_ -= sizeof(uint32_t);
    ret = memcpy_s(des_addr_, des_size_, str.data(), str.size());
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + str.size()));
    des_size_ -= (des_size_ >= str.size()) ? str.size() : 0U;
  }

  tlv_head_ptr->len = static_cast<uint32_t>(begin_len - des_size_);

  return SUCCESS;
}

Status NanoDbgData::SaveDbgVecTlv(const std::vector<int64_t> &vec, const uint32_t type) {
  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = type;
  tlv_head.len = static_cast<uint32_t>(vec.size() * sizeof(int64_t));
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);
  if (vec.size() == 0U) {
    GELOGD("[DbgData] vec size is zero, type:%u", type);
    return SUCCESS;
  }
  ret = memcpy_s(des_addr_, des_size_, vec.data(), static_cast<size_t>(tlv_head.len));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + tlv_head.len));
  des_size_ -= tlv_head.len;
  return SUCCESS;
}

Status NanoDbgData::SaveDbgMemInfoTlv(const std::vector<NanoDbgMemInfoDesc> &mem_info_list) {
  if (mem_info_list.size() == 0U) {
    GELOGD("[DbgData] mem_info_list size is zero");
    return SUCCESS;
  }

  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = DBG_L2_TLV_TYPE_MEM_INFO;
  tlv_head.len = 0U;  // update later
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);

  const size_t len_begin = des_size_;
  DbgOpMemInfosTlv2 mem_infos;
  mem_infos.num = static_cast<uint32_t>(mem_info_list.size());
  ret = memcpy_s(des_addr_, des_size_, &mem_infos, sizeof(mem_infos));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(mem_infos)));
  des_size_ -= sizeof(mem_infos);

  for (const auto &mem_info : mem_info_list) {
    DbgOpMemInfoTlv2 mem_tlv;
    mem_tlv.l3_tlv_list_len = 0U;
    mem_tlv.input_mem_size = mem_info.input_mem_size;
    mem_tlv.output_mem_size = mem_info.output_mem_size;
    mem_tlv.weight_mem_size = mem_info.weight_mem_size;
    mem_tlv.workspace_mem_size = mem_info.workspace_mem_size;
    mem_tlv.total_mem_size = mem_info.total_mem_size;
    ret = memcpy_s(des_addr_, des_size_, &mem_tlv, sizeof(mem_tlv));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(mem_tlv)));
    des_size_ -= sizeof(mem_tlv);
  }
  tlv_head_ptr->len = static_cast<uint32_t>(len_begin - des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgBufTlv(const std::vector<NanoDbgBufferDesc> &buffer_list) {
  if (buffer_list.size() == 0U) {
    GELOGD("[DbgData] buffer_list size is zero");
    return SUCCESS;
  }

  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = DBG_L2_TLV_TYPE_OP_BUF;
  tlv_head.len = 0U;  // update later
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);

  const size_t len_begin = des_size_;
  DbgOpBufTlv2 buf_tlv;
  buf_tlv.num = static_cast<uint32_t>(buffer_list.size());
  ret = memcpy_s(des_addr_, des_size_, &buf_tlv, sizeof(buf_tlv));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(buf_tlv)));
  des_size_ -= sizeof(buf_tlv);

  for (const auto &buf : buffer_list) {
    DbgOpBufParamTlv2 buf_param;
    buf_param.l3_tlv_list_len = 0U;
    buf_param.type = static_cast<uint8_t>(buf.type);
    buf_param.addr = buf.addr;
    buf_param.size = buf.size;
    ret = memcpy_s(des_addr_, des_size_, &buf_param, sizeof(buf_param));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(buf_param)));
    des_size_ -= sizeof(buf_param);
  }
  tlv_head_ptr->len = static_cast<uint32_t>(len_begin - des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgInputDescTlv(const std::vector<NanoDbgInputDesc> &input_list) {
  if (input_list.size() == 0U) {
    GELOGD("[DbgData] input_list size is zero");
    return SUCCESS;
  }

  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = DBG_L2_TLV_TYPE_INPUT_DESC;
  tlv_head.len = 0U;  // update later
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);

  const size_t len_begin = des_size_;
  DbgInputDescTlv2 inputs;
  inputs.num = static_cast<uint32_t>(input_list.size());
  ret = memcpy_s(des_addr_, des_size_, &inputs, sizeof(inputs));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(inputs)));
  des_size_ -= sizeof(inputs);

  for (const auto &input : input_list) {
    DbgInputDescParamTlv2 input_tlv;
    input_tlv.data_type = input.data_type;
    input_tlv.format = input.format;
    input_tlv.addr_type = input.addr_type;
    input_tlv.addr = input.addr;
    input_tlv.offset = input.offset;
    input_tlv.size = input.size;
    input_tlv.l3_tlv_list_len = 0U;
    DbgInputDescParamTlv2 *input_tlv_ptr = PtrToPtr<uint8_t, DbgInputDescParamTlv2>(des_addr_);

    ret = memcpy_s(des_addr_, des_size_, &input_tlv, sizeof(input_tlv));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(input_tlv)));
    des_size_ -= sizeof(input_tlv);
    const uint32_t l3_tlv_len_begin = static_cast<uint32_t>(des_size_);
    // L3 tlv shape
    GE_CHK_STATUS_RET(SaveDbgVecTlv(input.shape_dims, DBG_INPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS),
                      "[DbgData]save shape dims failed");

    // L3 tlv original shape
    GE_CHK_STATUS_RET(SaveDbgVecTlv(input.original_shape_dims, DBG_INPUT_DESC_L3_TLV_TYPE_ORI_SHAPE_DIMS),
                      "[DbgData]save origin shape dims failed");
    input_tlv_ptr->l3_tlv_list_len = l3_tlv_len_begin - static_cast<uint32_t>(des_size_);
  }
  tlv_head_ptr->len = static_cast<uint32_t>(len_begin - des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgOutputDescTlv(const std::vector<NanoDbgOutputDesc> &output_list) {
  if (output_list.size() == 0U) {
    GELOGD("[DbgData] output_list size is zero");
    return SUCCESS;
  }

  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = DBG_L2_TLV_TYPE_OUTPUT_DESC;
  tlv_head.len = 0U;  // update later
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);

  const size_t len_begin = des_size_;
  DbgOutputDescTlv2 outputs;
  outputs.num = static_cast<uint32_t>(output_list.size());
  ret = memcpy_s(des_addr_, des_size_, &outputs, sizeof(outputs));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(outputs)));
  des_size_ -= sizeof(outputs);

  for (const auto &output : output_list) {
    DbgOutputDescParamTlv2 output_tlv;
    output_tlv.data_type = output.data_type;
    output_tlv.format = output.format;
    output_tlv.addr_type = output.addr_type;
    output_tlv.original_index = output.original_index;
    output_tlv.original_data_type = output.original_data_type;
    output_tlv.original_format = output.original_format;
    output_tlv.addr = output.addr;
    output_tlv.offset = output.offset;
    output_tlv.size = output.size;
    output_tlv.l3_tlv_list_len = 0U;
    DbgOutputDescParamTlv2 *output_tlv_ptr = PtrToPtr<uint8_t, DbgOutputDescParamTlv2>(des_addr_);
    ret = memcpy_s(des_addr_, des_size_, &output_tlv, sizeof(output_tlv));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(output_tlv)));
    des_size_ -= sizeof(output_tlv);
    const uint32_t l3_tlv_len_begin = static_cast<uint32_t>(des_size_);

    // L3 tlv shape
    GE_CHK_STATUS_RET(SaveDbgVecTlv(output.shape_dims, DBG_OUTPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS),
                      "[DbgData]save shape dims failed");

    // L3 tlv original shape
    GE_CHK_STATUS_RET(SaveDbgVecTlv(output.original_shape_dims, DBG_OUTPUT_DESC_L3_TLV_TYPE_ORI_SHAPE_DIMS),
                      "[DbgData]save origin shape dims failed");

    // L3 tlv original name
    GE_CHK_STATUS_RET(SaveDbgStrTlv(output.original_name, DBG_OUTPUT_DESC_L3_TLV_TYPE_ORI_NAME),
                      "[DbgData]save origin shape dims failed");
    output_tlv_ptr->l3_tlv_list_len = l3_tlv_len_begin - static_cast<uint32_t>(des_size_);
  }
  tlv_head_ptr->len = static_cast<uint32_t>(len_begin - des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgWorkspaceDescTlv(const std::vector<NanoDbgWorkspaceDesc> &workspace_list) {
  if (workspace_list.size() == 0U) {
    GELOGD("[DbgData] workspace_list size is zero");
    return SUCCESS;
  }

  GELOGD("[DbgData]begin save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);
  TlvHead tlv_head;
  tlv_head.type = DBG_L2_TLV_TYPE_WORKSPACE_DESC;
  tlv_head.len = 0U;  // update later
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  errno_t ret = memcpy_s(des_addr_, des_size_, &tlv_head, sizeof(tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(tlv_head)));
  des_size_ -= sizeof(tlv_head);

  const size_t len_begin = des_size_;
  DbgWorkspaceDescTlv2 workspaces;
  workspaces.num = static_cast<uint32_t>(workspace_list.size());
  ret = memcpy_s(des_addr_, des_size_, &workspaces, sizeof(workspaces));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(workspaces)));
  des_size_ -= sizeof(workspaces);

  for (const auto &workspace : workspace_list) {
    DbgWorkspaceDescParamTlv2 workspace_tlv;
    workspace_tlv.l3_tlv_list_len = 0U;
    workspace_tlv.type = workspace.type;
    workspace_tlv.data_addr = workspace.data_addr;
    workspace_tlv.size = workspace.size;
    ret = memcpy_s(des_addr_, des_size_, &workspace_tlv, sizeof(workspace_tlv));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(workspace_tlv)));
    des_size_ -= sizeof(workspace_tlv);
  }
  tlv_head_ptr->len = static_cast<uint32_t>(len_begin - des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgL1Tlv() {
  // L1 tlv model name
  TlvHead l1_tlv_head;
  l1_tlv_head.type = DBG_L1_TLV_TYPE_MODEL_NAME;
  l1_tlv_head.len = static_cast<uint32_t>(model_name_.size());
  errno_t ret = memcpy_s(des_addr_, des_size_, &l1_tlv_head, sizeof(l1_tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(l1_tlv_head)));
  des_size_ -= sizeof(l1_tlv_head);

  ret = memcpy_s(des_addr_, des_size_, model_name_.data(), static_cast<size_t>(l1_tlv_head.len));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + l1_tlv_head.len));
  des_size_ -= l1_tlv_head.len;

  // L1 tlv op desc
  l1_tlv_head.type = DBG_L1_TLV_TYPE_OP_DESC;
  l1_tlv_head.len = 0U;  // update later
  TlvHead *l1_tlv_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);
  ret = memcpy_s(des_addr_, des_size_, &l1_tlv_head, sizeof(l1_tlv_head));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(l1_tlv_head)));
  des_size_ -= sizeof(l1_tlv_head);

  const size_t len_begin = des_size_;
  DbgOpDescTlv1 op_desc;
  op_desc.num = static_cast<uint32_t>(op_list_.size());
  GELOGD("[DbgData]L1 op desc, num:%u", op_desc.num);
  ret = memcpy_s(des_addr_, des_size_, &op_desc, sizeof(op_desc));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(op_desc)));
  des_size_ -= sizeof(op_desc);

  for (const auto &op : op_list_) {
    // L1 value
    DbgOpDescParamTlv1 op_param;
    op_param.task_id = op.task_id;
    op_param.stream_id = op.stream_id;
    op_param.logic_stream_id = op.logic_stream_id;
    op_param.task_type = op.task_type;
    op_param.block_dim = op.block_dim;
    op_param.datadump_is_multiop = op.datadump_is_multiop;
    op_param.l2_tlv_list_len = 0U;
    DbgOpDescParamTlv1 *op_param_ptr = PtrToPtr<uint8_t, DbgOpDescParamTlv1>(des_addr_);

    GELOGD("[DbgData]L1 op param, op_name:%s, stream_id:%u, task_id:%u", op.op_name.c_str(), op.logic_stream_id,
           op.task_id);
    ret = memcpy_s(des_addr_, des_size_, &op_param, sizeof(op_param));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "memcpy_s ret:%d", ret);
    des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(op_param)));
    des_size_ -= sizeof(op_param);
    const uint32_t op_len_begin = static_cast<uint32_t>(des_size_);

    // L2 tlv op name
    GE_CHK_STATUS_RET(SaveDbgStrTlv(op.op_name, DBG_L2_TLV_TYPE_OP_NAME), "[DbgData]save op name failed");
    // L2 tlv op type
    GE_CHK_STATUS_RET(SaveDbgStrTlv(op.op_type, DBG_L2_TLV_TYPE_OP_TYPE), "[DbgData]save op type failed");
    // L2 tlv original op name
    GE_CHK_STATUS_RET(SaveDbgVecTlv(op.original_op_names, DBG_L2_TLV_TYPE_ORI_OP_NAME),
                      "[DbgData]save original op name failed");
    // L2 tlv l1 fusion sub graph no
    GE_CHK_STATUS_RET(SaveDbgStrTlv(op.L1_fusion_sub_graph_no, DBG_L2_TLV_TYPE_L1_SUB_GRAPH_NO),
                      "[DbgData]save l1 sub graph no failed");

    // L2 tlv input
    GE_CHK_STATUS_RET(SaveDbgInputDescTlv(op.input_list), "[DbgData]save input failed");
    // L2 tlv output
    GE_CHK_STATUS_RET(SaveDbgOutputDescTlv(op.output_list), "[DbgData]save output failed");
    // L2 tlv workspace
    GE_CHK_STATUS_RET(SaveDbgWorkspaceDescTlv(op.workspace_list), "[DbgData]save workspace failed");
    // L2 tlv buffer
    GE_CHK_STATUS_RET(SaveDbgBufTlv(op.buffer_list), "[DbgData]save buffer failed");
    // L2 tlv mem info
    GE_CHK_STATUS_RET(SaveDbgMemInfoTlv(op.mem_info_list), "[DbgData]save mem info failed");
    op_param_ptr->l2_tlv_list_len = static_cast<uint32_t>(op_len_begin - des_size_);
  }
  l1_tlv_ptr->len = static_cast<uint32_t>(len_begin - des_size_);
  return SUCCESS;
}

Status NanoDbgData::SaveDbgPartition() {
  GE_CHK_STATUS_RET(SaveDbgHead(), "[DbgData]save head failed");
  GE_CHK_STATUS_RET(SaveDbgL1Tlv(), "[DbgData]save l1 tlv failed");
  return SUCCESS;
}
}  // namespace ge
