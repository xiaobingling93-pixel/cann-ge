/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump/exception_dumper.h"

#ifdef __GNUC__
#include <sys/types.h>
#include <unistd.h>
#endif

#include "mmpa/mmpa_api.h"
#include "common/plugin/datatype_util.h"
#include "common/debug/memory_dumper.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "proto/dump_task.pb.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping.pb.h"
#include "graph/ge_context.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "runtime/mem.h"
#include "exception_dumper.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/sgt_slice_type.h"
#include "common/checker.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "framework/runtime/subscriber/global_dumper.h"

#include "adump_opinfo_builder.h"
#include "exe_graph/runtime/tensor.h"
#include "base/err_msg.h"

namespace ge {
namespace {
const std::string kExtraPath = "/extra-info/data-dump/";
constexpr ge::char_t const *kDumpModeOutput = "output";
constexpr ge::char_t const *kDumpModeInput = "input";
constexpr ge::char_t const *kDumpModeAll = "all";
constexpr size_t kMaxOpDescInfoNum = 2048UL * 2048UL;
constexpr size_t kMaxTilingDataLogLen = 750UL;

static uint64_t GetNowTime() {
  uint64_t ret = 0U;
  mmTimeval tv;
  if (mmGetTimeOfDay(&tv, nullptr) == 0) {
    ret = (static_cast<uint64_t>(tv.tv_sec) * 1000000UL) + static_cast<uint64_t>(tv.tv_usec);
  }

  return ret;
}

static void ReplaceStringElem(std::string &str) {
  (void)for_each(str.begin(), str.end(), [](ge::char_t &ch) {
    if ((ch == ' ') || (ch == '.') || (ch == '/') || (ch == '\\')) {
      ch = '_';
    }
  });
}

static ge::Status GetStrAttrFromAllAttr(const std::string &all_attrs, const std::string &key, std::string &value) {
  size_t start_idx = all_attrs.find(key);
  if ((start_idx >= all_attrs.length()) || (start_idx + key.length() + 1 >= all_attrs.length())) {
    return ge::FAILED;
  }
  start_idx += (key.length() + 1);
  const size_t end_idx = all_attrs.find(";", start_idx);
  GE_ASSERT_TRUE((end_idx >= start_idx) && (end_idx < all_attrs.length()));
  const std::string tmp = all_attrs.substr(start_idx, end_idx - start_idx);
  proto::AttrDef attr_def;
  GE_ASSERT_TRUE(attr_def.ParseFromString(tmp));
  const auto deserializer = ge::AttrSerializerRegistry::GetInstance().GetDeserializer(attr_def.value_case());
  GE_ASSERT_NOTNULL(deserializer);
  AnyValue str_val;
  GE_ASSERT_SUCCESS(deserializer->Deserialize(attr_def, str_val));
  const auto val_str = str_val.Get<std::string>();

  GE_ASSERT_NOTNULL(val_str);
  value = val_str->c_str();
  return ge::SUCCESS;
}

static void SetDumpDataForWorkspace(const ge::OpDescInfo &op_desc_info, toolkit::dump::DumpData &dump_data,
                                    const bool is_exception, const ge::DumpProperties &dump_properties) {
  if ((!op_desc_info.workspace_bytes.empty()) &&
      (op_desc_info.space_addrs.size() == op_desc_info.workspace_bytes.size()) &&
      ((is_exception) || (dump_properties.GetDumpMode() == kDumpModeAll))) {
    GELOGI("workspace_info size %zu %zu", op_desc_info.workspace_bytes.size(), op_desc_info.space_addrs.size());
    for (size_t i = 0UL; i < op_desc_info.space_addrs.size(); ++i) {
      if (op_desc_info.space_addrs[i] == nullptr) {
        continue;
      }
      toolkit::dump::Workspace space;
      GELOGI("workspace_info add to dump_data");
      space.set_size(static_cast<uint64_t>(op_desc_info.workspace_bytes[i]));
      space.set_type(toolkit::dump::Workspace::LOG);
      dump_data.mutable_space()->Add(std::move(space));
    }
  }
}

static void SetDumpDataForSgtInfo(const ge::OpDescInfo &op_desc_info, toolkit::dump::DumpData &dump_data) {
  std::string sgt_json;
  if ((GetStrAttrFromAllAttr(op_desc_info.all_attrs, ffts::kAttrSgtJsonInfo, sgt_json) == ge::SUCCESS) &&
      (!sgt_json.empty())) {
    toolkit::dump::OpAttr op_attr;
    op_attr.set_name(ffts::kAttrSgtJsonInfo);
    op_attr.set_value(sgt_json);
    dump_data.mutable_attr()->Add(std::move(op_attr));
  }
}

static void SetDumpData(const ge::OpDescInfo &op_desc_info, toolkit::dump::DumpData &dump_data,
                        const bool is_exception, const ge::DumpProperties &dump_properties) {
  dump_data.set_version("2.0");
  dump_data.set_dump_time(GetNowTime());
  dump_data.set_op_name(op_desc_info.op_name);
  if (is_exception || (dump_properties.GetDumpMode() != kDumpModeOutput)) {
    for (size_t i = 0U; i < op_desc_info.input_format.size(); ++i) {
      if ((i >= op_desc_info.input_addrs.size()) || (op_desc_info.input_addrs[i] == nullptr)) {
        continue;
      }
      toolkit::dump::OpInput input;
      input.set_data_type(
        static_cast<toolkit::dump::OutputDataType>(ge::DataTypeUtil::GetIrDataType(op_desc_info.input_data_type[i])));
      input.set_format(static_cast<toolkit::dump::OutputFormat>(op_desc_info.input_format[i]));
      for (const int64_t dim : op_desc_info.input_shape[i]) {
        input.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
      }
      input.set_size(static_cast<uint64_t>(op_desc_info.input_size[i]));
      dump_data.mutable_input()->Add(std::move(input));
    }
  }

  if (is_exception || (dump_properties.GetDumpMode() != kDumpModeInput)) {
    for (size_t j = 0U; j < op_desc_info.output_format.size(); ++j) {
      if ((j >= op_desc_info.output_addrs.size()) || (op_desc_info.output_addrs[j] == nullptr)) {
        continue;
      }
      toolkit::dump::OpOutput output;
      output.set_data_type(static_cast<toolkit::dump::OutputDataType>(
                           ge::DataTypeUtil::GetIrDataType(op_desc_info.output_data_type[j])));
      output.set_format(static_cast<toolkit::dump::OutputFormat>(op_desc_info.output_format[j]));
      for (const int64_t dim : op_desc_info.output_shape[j]) {
        output.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
      }
      output.set_size(static_cast<uint64_t>(op_desc_info.output_size[j]));
      dump_data.mutable_output()->Add(std::move(output));
    }
  }
  SetDumpDataForWorkspace(op_desc_info, dump_data, is_exception, dump_properties);
  SetDumpDataForSgtInfo(op_desc_info, dump_data);
}

static void *GetArgsAddrByIOIndex(const std::map<uint64_t, uint64_t> &relevant_offset,
                                     std::vector<void *> &host_addr, const size_t io_idx) {
  void *target_addr = host_addr[io_idx];
  if (!relevant_offset.empty()) {
    auto iter = relevant_offset.find(io_idx);
    if (iter == relevant_offset.end() || iter->second >= host_addr.size()) {
      GELOGD("relevant offset [%zu] is not found from args, will skip to do dump.", io_idx);
      target_addr = nullptr;
    } else {
      target_addr = host_addr[iter->second];
    }
  }
  return target_addr;
}

Status PrepareInputTensor(const OpDescPtr &op, const ExtraOpInfo &extra_op_info,
                          std::vector<Adx::TensorInfoV2> &tensor_infos, std::vector<gert::Tensor> &input_tensors) {
  const size_t size = extra_op_info.input_addrs.size();
  input_tensors.resize(size);
  size_t addr_index = 0U;
  const bool is_host_args = extra_op_info.is_host_args;
  for (size_t i = 0U; i < op->GetAllInputsSize(); ++i) {
    const auto tensor_desc = op->MutableInputDesc(static_cast<uint32_t>(i));
    if ((tensor_desc == nullptr) || (addr_index >= size)) {
      continue;
    }
    const size_t idx = addr_index;
    addr_index++;

    // In the scenarios of single op and rt2, is_host_args is set to true and cust_to_relevant_offset_ is not set
    // so init args_offset to invalid value and  adump use addr in tensor
    uint64_t args_offset = is_host_args ? std::numeric_limits<uint32_t>::max() : idx;
    const auto &iter = extra_op_info.cust_to_relevant_offset_.find(args_offset);
    if (iter != extra_op_info.cust_to_relevant_offset_.cend()) {
      args_offset = iter->second;
    }

    int64_t tensor_size;
    if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size) != SUCCESS) {
      GELOGW("calc tensor size failed, op %s, input %u", op->GetName().c_str(), i);
      continue;
    }

    Adx::TensorInfoV2 tensor_info;
    tensor_info.tensorSize = static_cast<size_t>(tensor_size);
    tensor_info.format = tensor_desc->GetFormat();
    tensor_info.dataType = tensor_desc->GetDataType();
    tensor_info.tensorAddr = static_cast<int64_t *>(extra_op_info.input_addrs[idx]);
    std::vector<int64_t> dims = tensor_desc->GetShape().GetDims();
    for (size_t j = 0U; j < dims.size(); j++) {
      tensor_info.shape.push_back(dims[j]);
    }
    tensor_info.placement = gert::TensorPlacement::kOnDeviceHbm;
    if (tensor_desc->IsOriginShapeInitialized()) {
      std::vector<int64_t> origin_dims = tensor_desc->GetOriginShape().GetDims();
      for (size_t j = 0U; j < origin_dims.size(); j++) {
        tensor_info.originShape.push_back(origin_dims[j]);
      }
    }
    tensor_info.type = Adx::TensorType::INPUT;
    tensor_info.addrType = Adx::AddressType::TRADITIONAL;
    tensor_info.argsOffSet = static_cast<uint32_t>(args_offset);
    (void)tensor_infos.emplace_back(tensor_info);

    GELOGI("input[%u] addr %p, size %zu, argsOffSet %lu", i, extra_op_info.input_addrs[idx], tensor_size, args_offset);
  }
  return ge::SUCCESS;
}

Status PrepareOutputTensor(const OpDescPtr &op, const ExtraOpInfo &extra_op_info,
                           std::vector<Adx::TensorInfoV2> &tensor_infos, std::vector<gert::Tensor> &output_tensors) {
  const auto input_num = extra_op_info.input_addrs.size();
  const size_t size = extra_op_info.output_addrs.size();
  output_tensors.resize(size);
  const bool is_host_args = extra_op_info.is_host_args;
  for (size_t i = 0U; i < size; ++i) {
    const auto tensor_desc = op->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      continue;
    }

    int64_t tensor_size;
    if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size) != SUCCESS) {
      GELOGW("calc tensor size failed, op %s, output %u", op->GetName().c_str(), i);
      continue;
    }

    uint64_t args_offset = is_host_args ? std::numeric_limits<uint64_t>::max() : input_num + i;
    const auto &iter = extra_op_info.cust_to_relevant_offset_.find(args_offset);
    if (iter != extra_op_info.cust_to_relevant_offset_.cend()) {
      args_offset = iter->second;
    }

    Adx::TensorInfoV2 tensor_info;
    tensor_info.tensorSize = static_cast<size_t>(tensor_size);
    tensor_info.format = tensor_desc->GetFormat();
    tensor_info.dataType = tensor_desc->GetDataType();
    tensor_info.tensorAddr = static_cast<int64_t *>(extra_op_info.output_addrs[i]);
    tensor_info.placement = gert::TensorPlacement::kOnDeviceHbm;
    if (tensor_desc->IsOriginShapeInitialized()) {
      std::vector<int64_t> origin_dims = tensor_desc->GetOriginShape().GetDims();
      for (size_t j = 0U; j < origin_dims.size(); j++) {
        tensor_info.originShape.push_back(origin_dims[j]);
      }
    }
    std::vector<int64_t> dims = tensor_desc->GetShape().GetDims();
    for (size_t j = 0U; j < dims.size(); j++) {
      tensor_info.shape.push_back(dims[j]);
    }
    tensor_info.type = Adx::TensorType::OUTPUT;
    tensor_info.addrType = Adx::AddressType::TRADITIONAL;
    tensor_info.argsOffSet = static_cast<uint32_t>(args_offset);
    (void)tensor_infos.emplace_back(tensor_info);
    GELOGI("output[%u] addr %p, size %ld, argsOffSet %lu", i, extra_op_info.output_addrs[i], tensor_size, args_offset);
  }
  return ge::SUCCESS;
}

Status PrepareWorkspaceTensor(const ExtraOpInfo &extra_op_info, std::vector<Adx::TensorInfoV2> &tensor_infos,
                              std::vector<gert::Tensor> &workspace_tensors) {
  const size_t size = extra_op_info.workspace_info.size();
  workspace_tensors.resize(size);
  for (size_t i = 0U; i < size; ++i) {
    const int64_t workspace_size = extra_op_info.workspace_info[i].second;
    void *workspace_addr = reinterpret_cast<void *>(extra_op_info.workspace_info[i].first);

    Adx::TensorInfoV2 tensor_info;
    tensor_info.tensorSize = static_cast<size_t>(workspace_size);
    tensor_info.format = ge::FORMAT_ND;
    tensor_info.dataType = DT_UINT8;
    tensor_info.tensorAddr = static_cast<int64_t *>(workspace_addr);
    tensor_info.placement = gert::TensorPlacement::kOnDeviceHbm;
    tensor_info.originShape.push_back(workspace_size);
    tensor_info.shape.push_back(workspace_size);
    tensor_info.type = Adx::TensorType::WORKSPACE;
    tensor_info.addrType = Adx::AddressType::TRADITIONAL;
    (void)tensor_infos.emplace_back(tensor_info);
    GELOGI("workspace[%u] addr %p, size %ld", i, workspace_addr, workspace_size);
  }
  return ge::SUCCESS;
}
}  // namespace

// auto_register: true, register to global dumper and the global dumper will call it when exception happens.
ExceptionDumper::ExceptionDumper(bool auto_register) {
  if (auto_register) {
    gert::GlobalDumper::GetInstance()->RegisterExceptionDumpers(this);
    is_registered_ = true;
  }
};

ExceptionDumper::~ExceptionDumper() {
  if (is_registered_) {
    is_registered_ = false;
    gert::GlobalDumper::GetInstance()->UnregisterExceptionDumpers(this);
  }
};

void ExceptionDumper::SaveDumpOpInfoLocal(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id,
                                          bool is_dynamic) {
  (void)is_dynamic;
  OpDescInfo op_desc_info;
  SaveOpDescInfo(op, id, op_desc_info);
  if ((!extra_op_info.input_sizes.empty()) && (op_desc_info.input_size.size() == extra_op_info.input_sizes.size())) {
    op_desc_info.input_size.assign(extra_op_info.input_sizes.begin(), extra_op_info.input_sizes.end());
  }
  if ((!extra_op_info.output_sizes.empty()) && (op_desc_info.output_size.size() == extra_op_info.output_sizes.size())) {
    op_desc_info.output_size.assign(extra_op_info.output_sizes.begin(), extra_op_info.output_sizes.end());
  }
  for (size_t i = 0U; i < extra_op_info.workspace_info.size(); ++i) {
    (void)op_desc_info.space_addrs.emplace_back(reinterpret_cast<void *>(extra_op_info.workspace_info[i].first));
    (void)op_desc_info.workspace_bytes.emplace_back(extra_op_info.workspace_info[i].second);
  }
  op_desc_info.is_host_args = extra_op_info.is_host_args;
  op_desc_info.args_before_execute = extra_op_info.args_before_execute;
  op_desc_info.args = extra_op_info.args;
  op_desc_info.args_size = extra_op_info.args_size;
  op_desc_info.input_addrs = extra_op_info.input_addrs;
  op_desc_info.output_addrs = extra_op_info.output_addrs;
  op_desc_info.tiling_key = extra_op_info.tiling_key;
  op_desc_info.tiling_data = extra_op_info.tiling_data;
  op_desc_info.node_info = extra_op_info.node_info;
  op_desc_info.cust_to_relevant_offset_ = extra_op_info.cust_to_relevant_offset_;

  GELOGI(
    "[Save][OpExceptionInfo] op[%s] dev_id: %u, stream_id: %u, task_id: %u, context_id: %u, thread_id: %u, "
    "args: %#lx, args_size: %zu, tiling_key: %u, is_host_args:%d.",
    op_desc_info.op_name.c_str(), op_desc_info.id.device_id, op_desc_info.id.stream_id, op_desc_info.id.task_id,
    op_desc_info.id.context_id, op_desc_info.id.thread_id, static_cast<uint64_t>(op_desc_info.args),
    op_desc_info.args_size, op_desc_info.tiling_key, static_cast<int32_t>(op_desc_info.is_host_args));

  const std::lock_guard<std::mutex> lock(mutex_);
  ++op_desc_info_idx_;
  if (op_desc_info_.size() < kMaxOpDescInfoNum) {
    (void)op_desc_info_.emplace_back(std::move(op_desc_info));
  } else {
    op_desc_info_[op_desc_info_idx_ % kMaxOpDescInfoNum] = op_desc_info;
  }
}

void ExceptionDumper::SaveOpInfoToAdump(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id,
                                        bool is_dynamic) {
  uint32_t block_dim = 0;
  std::string dev_func;
  std::string tvm_magic;
  uint32_t imply_type = 0;
  std::string tiling_key = std::to_string(extra_op_info.tiling_key);
  std::string kernal_info = extra_op_info.node_info + "/" + tiling_key;
  const auto op_file_path = op->TryGetExtAttr(ATTR_NAME_OP_FILE_PATH, std::string("./kernel_meta"));
  std::vector<void *> workspace_addrs;
  std::vector<int64_t> workspace_sizes;
  uint32_t context_id = id.context_id;

  // holders keep a tensor used by Adx::TensorInfoV2
  std::vector<gert::Tensor> input_holder;
  std::vector<gert::Tensor> output_holder;
  std::vector<gert::Tensor> workspace_holder;

  std::vector<Adx::TensorInfoV2> input_tensor_infos;
  std::vector<Adx::TensorInfoV2> output_tensor_infos;
  std::vector<Adx::TensorInfoV2> workspace_tensor_infos;

  (void)AttrUtils::GetInt(op, TVM_ATTR_NAME_BLOCKDIM, block_dim);
  (void)AttrUtils::GetStr(op, op->GetName() + "_kernelname", "_kernelname", dev_func);
  (void)AttrUtils::GetStr(op, TVM_ATTR_NAME_MAGIC, tvm_magic);
  (void)AttrUtils::GetInt(op, ATTR_NAME_IMPLY_TYPE, imply_type);

  for (size_t i = 0U; i < extra_op_info.workspace_info.size(); ++i) {
    (void)workspace_addrs.emplace_back(reinterpret_cast<void *>(extra_op_info.workspace_info[i].first));
    (void)workspace_sizes.emplace_back(extra_op_info.workspace_info[i].second);
  }

  if (context_id == UINT32_MAX) {
    (void)AttrUtils::GetInt(op, "current_context_id", context_id);
  }

  PrepareInputTensor(op, extra_op_info, input_tensor_infos, input_holder);
  PrepareOutputTensor(op, extra_op_info, output_tensor_infos, output_holder);
  PrepareWorkspaceTensor(extra_op_info, workspace_tensor_infos, workspace_holder);

  AdumpOpInfoBuilder builder(op->GetName(), op->GetType(), is_dynamic);
  builder.Task(static_cast<uint32_t>(id.device_id), id.stream_id, id.task_id, context_id)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_BLOCK_DIM, std::to_string(block_dim))
    .AdditionInfo(Adx::DUMP_ADDITIONAL_TILING_KEY, tiling_key)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_TILING_DATA, extra_op_info.tiling_data)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_IMPLY_TYPE, std::to_string(imply_type))
    .AdditionInfo(Adx::DUMP_ADDITIONAL_ALL_ATTRS, AttrUtils::GetAllAttrsStr(op))
    .AdditionInfo(Adx::DUMP_ADDITIONAL_IS_HOST_ARGS, extra_op_info.is_host_args ? "true" : "false")
    .AdditionInfo(Adx::DUMP_ADDITIONAL_NODE_INFO, extra_op_info.node_info)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_DEV_FUNC, dev_func)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_TVM_MAGIC, tvm_magic)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_OP_FILE_PATH, op_file_path)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_KERNEL_INFO, kernal_info)
    .AdditionInfo(Adx::DUMP_ADDITIONAL_WORKSPACE_BYTES, ToString(workspace_sizes).c_str())
    .AdditionInfo(Adx::DUMP_ADDITIONAL_WORKSPACE_ADDRS, ToString(workspace_addrs).c_str())
    .TersorInfo(input_tensor_infos)
    .TersorInfo(output_tensor_infos)
    .TersorInfo(workspace_tensor_infos)
    .DeviceInfo(Adx::DEVICE_INFO_NAME_ARGS, reinterpret_cast<void *>(extra_op_info.args), extra_op_info.args_size);

  const Adx::OperatorInfoV2 &info = builder.Build();
  GELOGI(
    "[Add][OpExceptionInfo] op[%s] dev_id: %u, stream_id: %u, task_id: %u, context_id: %u, thread_id: %u, "
    "args: %#lx, args_size: %zu, tiling_key: %u, is_host_args:%d, tensor num: %zu, dynamic %d.",
    op->GetName().c_str(), id.device_id, id.stream_id, id.task_id, id.context_id, id.thread_id, extra_op_info.args,
    extra_op_info.args_size, extra_op_info.tiling_key, extra_op_info.is_host_args, info.tensorInfos.size(),
    is_dynamic);

  const auto adx_ret = Adx::AdumpAddExceptionOperatorInfoV2(info);
  if (adx_ret != Adx::ADUMP_SUCCESS) {
    GELOGW("call AdumpAddExceptionOperatorInfoV2 failed, op[%s], dev_id %d, stream_id %u, task_id %u, is_dynamic: %s",
           op->GetName().c_str(), id.device_id, id.stream_id, id.task_id, is_dynamic ? "true" : "false");
    return;
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  devid_stream_saved_.insert({id.device_id, id.stream_id});
}

void ExceptionDumper::SaveDumpOpInfo(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id,
                                     bool is_dynamic) {
  // always save op info for static graph in ge local
  if (!is_dynamic) {
    SaveDumpOpInfoLocal(op, extra_op_info, id, is_dynamic);
  }
  if (gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) {
    SaveOpInfoToAdump(op, extra_op_info, id, is_dynamic);
  }
}

void ExceptionDumper::SaveInputOutputInfo(const bool is_input, const OpDescPtr &op, OpDescInfo &op_desc_info) const {
  size_t size;
  if (is_input) {
    size = op->GetAllInputsSize();
  } else {
    size = op->GetOutputsSize();
  }

  GeTensorDescPtr tensor_desc;
  for (size_t i = 0U; i < size; ++i) {
    if (is_input) {
      tensor_desc = op->MutableInputDesc(static_cast<uint32_t>(i));
    } else {
      tensor_desc = op->MutableOutputDesc(static_cast<uint32_t>(i));
    }
    if (tensor_desc == nullptr) {
      continue;
    }
    int64_t tensor_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size) == SUCCESS) {
      if (is_input) {
        (void)op_desc_info.input_format.emplace_back(tensor_desc->GetFormat());
        (void)op_desc_info.input_shape.emplace_back(tensor_desc->GetShape().GetDims());
        (void)op_desc_info.input_data_type.emplace_back(tensor_desc->GetDataType());
        (void)op_desc_info.input_size.emplace_back(tensor_size);
      } else {
        (void)op_desc_info.output_format.emplace_back(tensor_desc->GetFormat());
        (void)op_desc_info.output_shape.emplace_back(tensor_desc->GetShape().GetDims());
        (void)op_desc_info.output_data_type.emplace_back(tensor_desc->GetDataType());
        (void)op_desc_info.output_size.emplace_back(tensor_size);
      }
      GELOGD("[Save][SaveInputOutputInfo] Save dump op info, index %zu, tensor size %ld, input flag %u.", i,
             tensor_size, static_cast<uint32_t>(is_input));
    }
  }
}

void ExceptionDumper::SaveOpDescInfo(const OpDescPtr &op, const OpDescInfoId &id, OpDescInfo &op_desc_info) const {
  if (op == nullptr) {
    GELOGW("[Save][OpExceptionInfo] op desc ptr is null.");
    return;
  }

  op_desc_info.op_name = op->GetName();
  op_desc_info.op_type = op->GetType();
  op_desc_info.id.task_id = id.task_id;
  op_desc_info.id.stream_id = id.stream_id;
  op_desc_info.id.device_id = id.device_id;
  op_desc_info.all_attrs = AttrUtils::GetAllAttrsStr(op);

  // Maybe can not get current_context_id from attribute, default value should be a INVALID value
  op_desc_info.id.context_id = id.context_id;
  if (id.context_id == UINT32_MAX) {
    (void)AttrUtils::GetInt(op, "current_context_id", op_desc_info.id.context_id);
  }

  op_desc_info.id.thread_id = id.thread_id;
  (void)AttrUtils::GetInt(op, ATTR_NAME_IMPLY_TYPE, op_desc_info.imply_type);
  (void)AttrUtils::GetInt(op, TVM_ATTR_NAME_BLOCKDIM, op_desc_info.block_dim);
  (void)AttrUtils::GetStr(op, op->GetName() + "_kernelname", "_kernelname", op_desc_info.dev_func);
  (void)AttrUtils::GetStr(op, TVM_ATTR_NAME_MAGIC, op_desc_info.tvm_magic);

  op_desc_info.op_file_path = op->TryGetExtAttr(ATTR_NAME_OP_FILE_PATH, std::string("./kernel_meta"));
  SaveInputOutputInfo(true, op, op_desc_info);
  SaveInputOutputInfo(false, op, op_desc_info);
}

void ExceptionDumper::LogExceptionArgs(const OpDescInfo &op_desc_info) const {
  if (!op_desc_info.args_before_execute.empty()) {
    GEEVENT("[AIC_INFO] %s, addr:%p", op_desc_info.args_before_execute.c_str(), op_desc_info.args);
  }
  if (op_desc_info.is_host_args) {
    GELOGI("Args is host, skip log args");
    return;
  }
  uint8_t *host_addr = nullptr;
  aclError ret = aclrtMallocHost(PtrToPtr<uint8_t *, void *>(&host_addr), op_desc_info.args_size);
  if (ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMallocHost failed, size:%zu, ret:%d", op_desc_info.args_size,
                      ret);
    GELOGE(FAILED, "[Call][RtMallocHost] failed, size:%zu, ret:%d", op_desc_info.args_size, ret);
    return;
  }
  GE_MAKE_GUARD_RTMEM(host_addr);
  ret = aclrtMemcpy(host_addr, static_cast<uint64_t>(op_desc_info.args_size),
      reinterpret_cast<void *>(op_desc_info.args), static_cast<uint64_t>(op_desc_info.args_size),
      ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, size:%zu, ret:%d", op_desc_info.args_size, ret);
    GELOGE(FAILED, "[Call][aclrtMemcpy] failed, size:%zu, ret:%d", op_desc_info.args_size, ret);
    return;
  }

  std::stringstream ss;
  ss << "args after execute: ";
  gert::PrintHex(reinterpret_cast<void **>(host_addr), op_desc_info.args_size / sizeof(void *), ss);
  GEEVENT("[AIC_INFO] %s, addr:%p", ss.str().c_str(), op_desc_info.args);
}

void ExceptionDumper::LogExceptionTvmOpInfo(const OpDescInfo &op_desc_info) const {
  if (static_cast<domi::ImplyType>(op_desc_info.imply_type) != domi::ImplyType::TVM) {
    GELOGI("exception op:%s(%s) imply_type:%s not tvm", op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(),
           TypeUtilsInner::ImplyTypeToSerialString(static_cast<domi::ImplyType>(op_desc_info.imply_type)).c_str());
    return;
  }

  if ((op_desc_info.input_format.size() != op_desc_info.input_shape.size()) ||
      (op_desc_info.input_format.size() != op_desc_info.input_data_type.size())) {
    GELOGW("exception op:%s(%s) input format size:%zu, shape size:%zu, dtype size:%zu not equal, skip log op info",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.input_format.size(),
           op_desc_info.input_shape.size(), op_desc_info.input_data_type.size());
    return;
  }

  if ((op_desc_info.output_format.size() != op_desc_info.output_shape.size()) ||
      (op_desc_info.output_format.size() != op_desc_info.output_data_type.size())) {
    GELOGW("exception op:%s(%s) output format size:%zu, shape size:%zu, dtype size:%zu not equal, skip log op info",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.output_format.size(),
           op_desc_info.output_shape.size(), op_desc_info.output_data_type.size());
    return;
  }

  GEEVENT("[AIC_INFO] node_name:%s, node_type:%s, stream_id:%u, task_id:%u, context_id:%u, thread_id:%u",
          op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.id.stream_id,
          op_desc_info.id.task_id, op_desc_info.id.context_id, op_desc_info.id.thread_id);
  for (size_t i = 0U; i < op_desc_info.input_format.size(); i++) {
    const std::string content = "input:" + std::to_string(i) + ";shape:" + ToString(op_desc_info.input_shape[i]) +
                                ";format:" + TypeUtils::FormatToSerialString(op_desc_info.input_format[i]) +
                                ";dtype:" + TypeUtils::DataTypeToSerialString(op_desc_info.input_data_type[i]);
    GEEVENT("[AIC_INFO] %s;addr:%p", content.c_str(), op_desc_info.input_addrs[i]);
  }

  for (size_t i = 0U; i < op_desc_info.output_format.size(); i++) {
    const std::string content = "output:" + std::to_string(i) + ";shape:" + ToString(op_desc_info.output_shape[i]) +
                                ";format:" + TypeUtils::FormatToSerialString(op_desc_info.output_format[i]) +
                                ";dtype:" + TypeUtils::DataTypeToSerialString(op_desc_info.output_data_type[i]);
    GEEVENT("[AIC_INFO] %s;addr:%p", content.c_str(), op_desc_info.output_addrs[i]);
  }

  GEEVENT("[AIC_INFO] block_dim:%u", op_desc_info.block_dim);
  GEEVENT("[AIC_INFO] workspace_bytes:%s", ToString(op_desc_info.workspace_bytes).c_str());
  GEEVENT("[AIC_INFO] workspace_addrs:%s", ToString(op_desc_info.space_addrs).c_str());
  GEEVENT("[AIC_INFO] all attrs:%s", op_desc_info.all_attrs.c_str());
  GEEVENT("[AIC_INFO] dev_func:%s", op_desc_info.dev_func.c_str());
  GEEVENT("[AIC_INFO] tvm_magic:%s", op_desc_info.tvm_magic.c_str());
  GEEVENT("[AIC_INFO] kernel_info:%s/%u", op_desc_info.node_info.c_str(), op_desc_info.tiling_key);
  GEEVENT("[AIC_INFO] tiling_key:%u", op_desc_info.tiling_key);
  std::string log_tiling_data = "";
  if (!op_desc_info.tiling_data.empty()) {
    log_tiling_data = google::protobuf::CEscape(op_desc_info.tiling_data);
  }
  size_t index = 0UL;
  while (index < log_tiling_data.length()) {
    GEEVENT("[AIC_INFO] tiling_data:%s", log_tiling_data.substr(index, kMaxTilingDataLogLen).c_str());
    index += kMaxTilingDataLogLen;
  }
  LogExceptionArgs(op_desc_info);
  ge::char_t curr_path[MMPA_MAX_PATH] = {};
  if (mmGetCwd(&curr_path[0], MMPA_MAX_PATH) != EN_OK) {
    GELOGW("get current path failed when do aicerror info record");
    return;
  }

  ge::char_t real_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(op_desc_info.op_file_path.c_str(), &real_path[0], MMPA_MAX_PATH) != EN_OK) {
    GELOGW("real path for %s failed when do aicerror info record", op_desc_info.op_file_path.c_str());
    return;
  }
  const std::string file_prefix = op_desc_info.dev_func.substr(0U, op_desc_info.dev_func.rfind("__"));
  const std::string src_file = std::string(real_path) + "/" + file_prefix + ".o";
  const std::string dst_path = std::string(curr_path);

#ifdef __GNUC__
  const uint32_t pid = static_cast<uint32_t>(fork());
  if (pid == 0U) {
    (void)execlp("cp", "cp", src_file.c_str(), dst_path.c_str(), nullptr);
  }
#endif

  GEEVENT("[AIC_INFO] op_file_path:%s", dst_path.c_str());
}

Status ExceptionDumper::DumpNodeInfo(const OpDescInfo &op_desc_info, const std::string &file_path,
                                     const bool is_exception, const bool is_ffts_plus,
                                     const ge::DumpProperties &dump_properties) const {
  toolkit::dump::DumpData dump_data;
  SetDumpData(op_desc_info, dump_data, is_exception, dump_properties);
  const uint64_t now_time = GetNowTime();
  std::string op_name = op_desc_info.op_name;
  std::string op_type = op_desc_info.op_type;
  ReplaceStringElem(op_name);
  ReplaceStringElem(op_type);
  std::string dump_file_path = file_path + op_type + "." + op_name + "." +
    std::to_string(op_desc_info.id.task_id) + "." + std::to_string(now_time);
  if (is_ffts_plus) {
    const std::string ffts_suffix = "." + std::to_string(toolkit::aicpu::dump::Task::FFTSPLUS) +
                                    "." + std::to_string(op_desc_info.id.context_id) +
                                    "." + std::to_string(op_desc_info.id.thread_id);
    (void)dump_file_path.append(ffts_suffix);
  }

  if (is_exception) {
    GELOGE(FAILED, "[Dump][Exception] dump exception to file, file: %s", dump_file_path.c_str());  // for tools analysis
  } else {
    GELOGI("[Dump][NodeInfo] dump node info to file, file: %s", dump_file_path.c_str());
  }

  const uint64_t proto_size = dump_data.ByteSizeLong();
  const std::unique_ptr<char[]> proto_msg = MakeUnique<char[]>(proto_size);
  GE_CHECK_NOTNULL(proto_msg);
  GE_CHECK_LE(proto_size, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
  const bool ret = dump_data.SerializeToArray(proto_msg.get(), static_cast<int32_t>(proto_size));
  if ((!ret) || (proto_size == 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Serialize proto to std::string fail");
    GELOGE(PARAM_INVALID, "[Dump][Exception] Dump data proto serialize failed");
    return PARAM_INVALID;
  }

  GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(dump_file_path.c_str(), &proto_size, sizeof(uint64_t)),
                    "Failed to dump proto size");
  GELOGI("Dump_size: %" PRIu64 "", proto_size);
  GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(dump_file_path.c_str(), proto_msg.get(), proto_size),
                    "Failed to dump proto msg");
  GE_CHK_STATUS_RET(DumpExceptionInput(op_desc_info, dump_file_path, is_exception, dump_properties),
                    "Dump exception input failed");
  GE_CHK_STATUS_RET(DumpExceptionOutput(op_desc_info, dump_file_path, is_exception, dump_properties),
                    "Dump exception output failed");
  GE_CHK_STATUS_RET(DumpExceptionWorkspace(op_desc_info, dump_file_path, is_exception, dump_properties),
                    "Dump exception workspace failed");
  GELOGI("[Dump][NodeInfo] Dump op[%s] info SUCCESS", op_name.c_str());
  return SUCCESS;
}

void ExceptionDumper::RefreshAddrs(OpDescInfo &op_desc_info) const {
  if (op_desc_info.args == 0U) {
    GELOGI("op:%s(%s) store args is empty, skip refresh addr",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str());
    return;
  }

  if (op_desc_info.is_host_args) {
    GELOGI("op:%s(%s) store args is on host, skip refresh addr", op_desc_info.op_name.c_str(),
           op_desc_info.op_type.c_str());
    return;
  }

  const size_t input_num = op_desc_info.input_addrs.size();
  const size_t output_num = op_desc_info.output_addrs.size();
  const size_t target_size = (input_num + output_num) * sizeof(void *);
  std::vector<void *> host_addr(input_num + output_num);
  const auto rt_ret = aclrtMemcpy(host_addr.data(), target_size,
      ValueToPtr(static_cast<uint64_t>(op_desc_info.args)), target_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (rt_ret != ACL_SUCCESS) {
    GELOGI("op:%s(%s) can't aclrtMemcpy to host, store args:%zu, memcpy size:%zu, skip refresh addr",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.args, target_size);
    return;
  }

  for (size_t i = 0U; i < input_num; i++) {
    void *target_addr = GetArgsAddrByIOIndex(op_desc_info.cust_to_relevant_offset_, host_addr, i);
    GELOGI("op:%s(%s) input index:%zu addr:%p refresh to addr:%p", op_desc_info.op_name.c_str(),
           op_desc_info.op_type.c_str(), i, op_desc_info.input_addrs[i], target_addr);
    op_desc_info.input_addrs[i] = target_addr;
  }

  for (size_t i = 0U; i < output_num; i++) {
    void *target_addr = GetArgsAddrByIOIndex(op_desc_info.cust_to_relevant_offset_, host_addr, i + input_num);
    GELOGI("op:%s(%s) output index:%zu addr:%p refresh to addr:%p", op_desc_info.op_name.c_str(),
           op_desc_info.op_type.c_str(), i, op_desc_info.output_addrs[i], target_addr);
    op_desc_info.output_addrs[i] = target_addr;
  }
}

bool ExceptionDumper::GetOpDescInfo(const OpDescInfoId &op_id, OpDescInfo &op_desc_info) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GELOGI("[Get][OpDescInfo] There are %zu op info saved, target stream_id:%u, task_id:%u, context_id:%u, thread_id:%u, "
         "dev_id: %u.", op_desc_info_.size(), op_id.stream_id, op_id.task_id, op_id.context_id, op_id.thread_id,
         op_id.device_id);
  for (auto &dump_op_info : op_desc_info_) {
    if (((dump_op_info.id.task_id == op_id.task_id) || (dump_op_info.id.task_id == UINT32_MAX)) &&
        ((dump_op_info.id.stream_id == op_id.stream_id) || (dump_op_info.id.stream_id == UINT32_MAX)) &&
        (dump_op_info.id.context_id == op_id.context_id) && (dump_op_info.id.thread_id == op_id.thread_id) &&
        (dump_op_info.id.device_id == op_id.device_id)) {
      GELOGI(
          "[Get][OpDescInfo] Find exception op [%s] of task_id: %u, stream_id: %u, context_id: %u, thread_id: %u, "
          "dev_id: %u.",
          dump_op_info.op_name.c_str(), op_id.task_id, op_id.stream_id, op_id.context_id, op_id.thread_id,
          op_id.device_id);
      op_desc_info = dump_op_info;
      RefreshAddrs(op_desc_info);
      LogExceptionTvmOpInfo(op_desc_info);
      return true;
    }
  }
  return false;
}

Status ExceptionDumper::DumpDevMem(const ge::char_t *const file, const void *const addr, const uint64_t size) {
  if (size == 0) {
    GELOGI("No need to dump data, because the size is 0.");
    return SUCCESS;
  }
  uint8_t *host_addr = nullptr;
  aclError ret = aclrtMallocHost(PtrToPtr<uint8_t *, void *>(&host_addr), size);
  if (ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMallocHost failed, size:%" PRIu64 ", ret:%d", size, ret);
    GELOGE(FAILED, "[Call][RtMallocHost] failed, size:%zu, ret:%d", size, ret);
    return FAILED;
  }
  GE_MAKE_GUARD_RTMEM(host_addr);
  ret = aclrtMemcpy(host_addr, size, addr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, size:%" PRIu64 ", ret:%d", size, ret);
    GELOGE(FAILED, "[Call][aclrtMemcpy] failed, size:%zu, ret:%d", size, ret);
    return FAILED;
  }

  GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(file, host_addr, size));
  return SUCCESS;
}

Status ExceptionDumper::DumpExceptionInput(const OpDescInfo &op_desc_info, const std::string &dump_file,
                                           const bool is_exception, const ge::DumpProperties &dump_properties) const {
  if ((!is_exception) && (dump_properties.GetDumpMode() == kDumpModeOutput)) {
    return SUCCESS;
  }

  const size_t all_size = op_desc_info.input_addrs.size();
  const size_t valid_size = op_desc_info.input_size.size();
  size_t cur_idx = 0U;
  GELOGI("[Dump][Input] op[%s] num %zu-%zu", op_desc_info.op_name.c_str(), all_size, valid_size);
  for (size_t i = 0U; i < all_size; ++i) {
    void *addr = op_desc_info.input_addrs[i];
    if (addr == nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(cur_idx < valid_size);
    const uint64_t size = static_cast<uint64_t>(op_desc_info.input_size[cur_idx++]);
    Status ret;
    GELOGI("[Dump][Input][%zu] addr %p, size %llu.", i, addr, size);
    if (!is_exception) {
      // host dump no need to copy from device to host
      ret = MemoryDumper::DumpToFile(dump_file.data(), addr, size);
    } else {
      ret = DumpDevMem(dump_file.data(), addr, size);
    }
    GE_CHK_STATUS_RET(ret, "[Dump][Input][%zu] failed, op[%s].", op_desc_info.op_name.c_str(), i);
  }
  return SUCCESS;
}

Status ExceptionDumper::DumpExceptionOutput(const OpDescInfo &op_desc_info, const std::string &dump_file,
                                            const bool is_exception, const ge::DumpProperties &dump_properties) const {
  if ((!is_exception) && (dump_properties.GetDumpMode() == kDumpModeInput)) {
    return SUCCESS;
  }
  const size_t all_size = op_desc_info.output_addrs.size();
  const size_t valid_size = op_desc_info.output_size.size();
  size_t cur_idx = 0U;
  GELOGI("[Dump][Output] op[%s] num %zu-%zu", op_desc_info.op_name.c_str(), all_size, valid_size);
  for (size_t i = 0U; i < all_size; ++i) {
    void *addr = op_desc_info.output_addrs[i];
    if (addr == nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(cur_idx < valid_size);
    const uint64_t size = static_cast<uint64_t>(op_desc_info.output_size[cur_idx++]);
    Status ret;
    GELOGI("[Dump][output][%zu] addr %p, size %llu.", i, addr, size);
    if (!is_exception) {
      // host dump no need to copy from device to host
      ret = MemoryDumper::DumpToFile(dump_file.data(), addr, size);
    } else {
      ret = DumpDevMem(dump_file.data(), addr, size);
    }
    GE_CHK_STATUS_RET(ret, "[Dump][output][%zu] failed, op[%s].", op_desc_info.op_name.c_str(), i);
  }
  return SUCCESS;
}

Status ExceptionDumper::DumpExceptionWorkspace(const OpDescInfo &op_desc_info, const std::string &dump_file,
                                               const bool is_exception,
                                               const ge::DumpProperties &dump_properties) const {
  if ((!is_exception) && (dump_properties.GetDumpMode() != kDumpModeAll)) {
    return SUCCESS;
  }

  GELOGI("[Dump][Workspace] op[%s] num %zu", op_desc_info.op_name.c_str(), op_desc_info.space_addrs.size());
  for (size_t i = 0U; i < op_desc_info.space_addrs.size(); ++i) {
    void *addr = op_desc_info.space_addrs[i];
    if (addr == nullptr) {
      continue;
    }
    const uint64_t size = static_cast<uint64_t>(op_desc_info.workspace_bytes[i]);
    Status ret;
    GELOGI("[Dump][Workspace][%zu] addr %p, size %llu.", i, addr, size);
    if (!is_exception) {
      // host dump no need to copy from device to host
      ret = MemoryDumper::DumpToFile(dump_file.data(), addr, size);
    } else {
      ret = DumpDevMem(dump_file.data(), addr, size);
    }
    GE_CHK_STATUS_RET(ret, "[Dump][Workspace][%zu] failed, op[%s].", op_desc_info.op_name.c_str(), i);
  }
  return SUCCESS;
}

OpDescInfo *ExceptionDumper::MutableOpDescInfo(const uint32_t task_id, const uint32_t stream_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (OpDescInfo &op_desc_info : op_desc_info_) {
    if ((op_desc_info.id.task_id == task_id) && (op_desc_info.id.stream_id == stream_id)) {
      return &op_desc_info;
    }
  }
  return nullptr;
}

void ExceptionDumper::Reset(ExtraOpInfo &extra_op_info) {
  extra_op_info.input_addrs.clear();
  extra_op_info.output_addrs.clear();
}

void ExceptionDumper::Clear() {
  const std::lock_guard<std::mutex> lock(mutex_);
  op_desc_info_.clear();
  for (const auto &iter : devid_stream_saved_) {
    auto adx_ret = Adx::AdumpDelExceptionOperatorInfo(iter.first, iter.second);
    GELOGI("[Del][OpExceptionInfo] dev_id: %d, stream_id: %u, adx ret: %d.", iter.first, iter.second, adx_ret);
  }
  devid_stream_saved_.clear();
}
}  // namespace ge
