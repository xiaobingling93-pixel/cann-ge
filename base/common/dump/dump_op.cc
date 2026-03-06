/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump/dump_op.h"
#include "acl/acl_rt.h"
#include <array>
#include "common/dump/dump_manager.h"
#include "common/plugin/datatype_util.h"
#include "framework/common/debug/ge_log.h"
#include "common/sgt_slice_type.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "framework/common/debug/log.h"
#include "graph/anchor.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping.pb.h"
#include "runtime/rt.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"
#include "runtime/rts/rts_device.h"
#include "aicpu_task_struct.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "ge/ge_error_codes.h"
#include "common/checker.h"
#include "base/err_msg.h"

namespace {
constexpr uint32_t kAiCpuLoadFlag = 1U;
const std::string kDumpModeOutput = "output";
const std::string kDumpModeInput = "input";
const std::string kDumpModeAll = "all";
const std::string kDumpKernelsDumpOp = "DumpDataInfo";
constexpr uint32_t k16BitsMask = 0x0000FFFFU;  // 16 bits, 1111,1111,1111,1111
constexpr int32_t k16BitWidth = 16;
const std::string kDumpDataDefaultValue = "stats";
constexpr uint32_t kInputBitsMask = 0x01U;
constexpr uint32_t kOutputBitsMask = 0x02U;
}  // namespace

namespace ge {
DumpOp::~DumpOp() {
  if (proto_dev_mem_ != nullptr) {
    (void)aclrtFree(proto_dev_mem_);
    proto_dev_mem_ = nullptr;
  }

  if (proto_size_dev_mem_ != nullptr) {
    (void)aclrtFree(proto_size_dev_mem_);
    proto_size_dev_mem_ = nullptr;
  }

  if (dev_mem_unload_ !=nullptr) {
    (void)aclrtFree(dev_mem_unload_);
    dev_mem_unload_ = nullptr;
  }

  if (launch_kernel_args_dev_mem_ != nullptr) {
    GE_CHK_RT(aclrtFree(launch_kernel_args_dev_mem_));
    launch_kernel_args_dev_mem_ = nullptr;
  }
}

void DumpOp::SetLoopAddr(const uintptr_t global_step, const uintptr_t loop_per_iter, const uintptr_t loop_cond) {
  global_step_ = global_step;
  loop_per_iter_ = loop_per_iter;
  loop_cond_ = loop_cond;
}

void DumpOp::SetDynamicModelInfo(const std::string &dynamic_model_name, const std::string &dynamic_om_name,
                                 const uint32_t dynamic_model_id) {
  dynamic_model_name_ = dynamic_model_name;
  dynamic_om_name_ = dynamic_om_name;
  dynamic_model_id_ = dynamic_model_id;
  GELOGD("Model name [%s], om_name [%s], model id [%u].", dynamic_model_name.c_str(), dynamic_om_name.c_str(),
         dynamic_model_id);
}

static void SetLoopAddrToOpMapping(const uintptr_t step_id, const uintptr_t loop_per_iter,
                                   const uintptr_t loop_cond,
                                   toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  GELOGI("step_id: %lu, loop_per_iter:%lu, loop_cond: %lu.", static_cast<uint64_t>(step_id),
         static_cast<uint64_t>(loop_per_iter), static_cast<uint64_t>(loop_cond));

  if (step_id != 0U) {
    op_mapping_info.set_step_id_addr(static_cast<uint64_t>(step_id));
  }

  if (loop_per_iter != 0U) {
    op_mapping_info.set_iterations_per_loop_addr(static_cast<uint64_t>(loop_per_iter));
  }

  if (loop_cond != 0U) {
    op_mapping_info.set_loop_cond_addr(static_cast<uint64_t>(loop_cond));
  }
}

void DumpOp::DumpWorkspace(toolkit::aicpu::dump::Task &task) {
  for (size_t i = 0UL; i < space_addrs_.size(); ++i) {
    const uint64_t addr = static_cast<uint64_t>(space_addrs_[i].first);
    const uint64_t size = static_cast<uint64_t>(space_addrs_[i].second);
    GELOGI("workspace_info: %p %zu", addr, size);
    toolkit::aicpu::dump::Workspace space;
    space.set_type(toolkit::aicpu::dump::Workspace::LOG);
    space.set_data_addr(addr);
    space.set_size(size);
    task.mutable_space()->Add(std::move(space));
  }
}

toolkit::aicpu::dump::AddressType DumpOp::GetAddrType(const toolkit::aicpu::dump::Task &task,
                                                      const GeTensorDesc &desc) const {
  if (task.context_size() != 0) {
    return toolkit::aicpu::dump::AddressType::RAW_ADDR;
  }

  bool no_tiling = false;
  if (AttrUtils::GetBool(desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling) && no_tiling) {
    return toolkit::aicpu::dump::AddressType::NOTILING_ADDR;
  }

  return toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR;
}

Status DumpOp::DumpOutput(toolkit::aicpu::dump::Task &task, const OpDescPtr &op_desc,
                          const std::vector<uintptr_t> &addrs, bool ffts_flag) const {
  const auto &output_descs = op_desc->GetAllOutputsDescPtr();
  const std::string dump_model_name = dynamic_model_name_;
  const std::string dump_om_name = dynamic_om_name_;
  GELOGI("Start to dump output in Launch dump op, model name %s, size %u, ffts flag %d.", dump_model_name.c_str(), output_descs.size(),
         static_cast<int32_t>(ffts_flag));
  for (size_t i = 0UL; i < output_descs.size(); ++i) {
    const std::string op_name = op_desc->GetName();
    const std::string op_type = op_desc->GetType();
    if (dump_properties_.IsOutputInOpNameBlacklist(dump_model_name, op_name, static_cast<uint32_t>(i)) ||
      dump_properties_.IsOutputInOpNameBlacklist(dump_om_name, op_name, static_cast<uint32_t>(i)) ||
      dump_properties_.IsOutputInOpNameBlacklist(DUMP_LAYER_OP_MODEL, op_name, static_cast<uint32_t>(i))) {
      GELOGI("[Dumper] Node name %s, Node type: %s, output index %zu is in opname-blacklist, skip to dump this output.",
         op_name.c_str(), op_type.c_str(), i);
      continue;
    }
    if (dump_properties_.IsOutputInOpTypeBlacklist(dump_model_name, op_type, static_cast<uint32_t>(i)) ||
      dump_properties_.IsOutputInOpTypeBlacklist(dump_om_name, op_type, static_cast<uint32_t>(i)) ||
      dump_properties_.IsOutputInOpTypeBlacklist(DUMP_LAYER_OP_MODEL, op_type, static_cast<uint32_t>(i))) {
      GELOGI("[Dumper] Node name %s, Node type: %s, output index %zu is in optype-blacklist, skip to dump this output.",
         op_name.c_str(), op_type.c_str(), i);
      continue;
    }
    if ((i >= addrs.size()) || (!ffts_flag && addrs[i] == reinterpret_cast<uintptr_t>(nullptr))) {
      GELOGW("[Dumper] Node name %s, i is %zu, output addrs size is %zu", op_desc->GetName().c_str(), i,
             addrs.size());
      continue;
    }
    GELOGD("Get op[%s:%s] output_desc[shape:%s, original shape:%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           output_descs.at(i)->GetShape().ToString().c_str(), output_descs.at(i)->GetOriginShape().ToString().c_str());
    toolkit::aicpu::dump::Output output;
    output.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(output_descs.at(i)->GetDataType())));
    output.set_format(static_cast<int32_t>(output_descs.at(i)->GetFormat()));
    for (const int64_t dim : output_descs.at(i)->GetShape().GetDims()) {
      output.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    for (const int64_t dim : output_descs.at(i)->GetOriginShape().GetDims()) {
      output.mutable_origin_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    int64_t output_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(*output_descs.at(i), output_size) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TensorSize]Failed, output %zu, node %s(%s),",
             i, op_desc->GetName().c_str(), op_desc->GetType().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Get output %zu tensor size of node %s(%s) failed",
                        i, op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }

    GELOGI("[Dumper] Node [%s] output[%zu] size %ld addr is %p.", op_desc->GetName().c_str(), i, output_size, addrs[i]);
    output.set_size(static_cast<uint64_t>(output_size));
    output.set_address(static_cast<uint64_t>(addrs[i]));
    output.set_offset(std::numeric_limits<uint64_t>::max());
    output.set_addr_type(GetAddrType(task, *output_descs.at(i)));
    task.mutable_output()->Add(std::move(output));
  }
  return SUCCESS;
}

Status DumpOp::DumpInput(toolkit::aicpu::dump::Task &task, const OpDescPtr &op_desc,
                         const std::vector<uintptr_t> &addrs, bool ffts_flag) const {
  GeTensorDescPtr input_descs;
  const std::string dump_model_name = dynamic_model_name_;
  const std::string dump_om_name = dynamic_om_name_;
  GELOGI("Start dump input in Launch dump op %s, model_name %s, input_descs size %zu, addr size %zu.", dump_model_name.c_str(), op_desc->GetName().c_str(),
    op_desc->GetAllInputsSize(), addrs.size());
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    const std::string op_name = op_desc->GetName();
    const std::string op_type = op_desc->GetType();
    GELOGI("[Dumper] Node name %s, node type %s input_descs idx %zu", op_name.c_str(), op_type.c_str(), i);
    if (dump_properties_.IsInputInOpNameBlacklist(dump_model_name, op_name, static_cast<uint32_t>(i)) ||
      dump_properties_.IsInputInOpNameBlacklist(dump_om_name, op_name, static_cast<uint32_t>(i)) ||
      dump_properties_.IsInputInOpNameBlacklist(DUMP_LAYER_OP_MODEL, op_name, static_cast<uint32_t>(i))) {
      GELOGI("[Dumper] Node name %s, Node type: %s, input index %zu is in opname-blacklist, skip to dump this input.",
         op_name.c_str(), op_type.c_str(), i);
      continue;
    }
    if (dump_properties_.IsInputInOpTypeBlacklist(dump_model_name, op_type, static_cast<uint32_t>(i)) ||
      dump_properties_.IsInputInOpTypeBlacklist(dump_om_name, op_type, static_cast<uint32_t>(i)) ||
      dump_properties_.IsInputInOpTypeBlacklist(DUMP_LAYER_OP_MODEL, op_type, static_cast<uint32_t>(i))) {
      GELOGI("[Dumper] Node name %s, Node type: %s, input index %zu is in optype-blacklist, skip to dump this input.",
         op_name.c_str(), op_type.c_str(), i);
      continue;
    }
    input_descs = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if ((input_descs == nullptr) || (input_descs->GetShape().IsUnknownShape())) {
      continue;
    }
    if ((i > addrs.size()) || (!ffts_flag && addrs[i] == reinterpret_cast<uintptr_t>(nullptr))) {
      GELOGW("[Dumper] Node name %s, addr_id is %zu, input addrs size is %zu", op_desc->GetName().c_str(), i,
             addrs.size());
      continue;
    }
    toolkit::aicpu::dump::Input input;
    input.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(input_descs->GetDataType())));
    input.set_format(static_cast<int32_t>(input_descs->GetFormat()));

    GELOGD("Get op[%s:%s] input_desc[shape:%s, original shape:%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           input_descs->GetShape().ToString().c_str(), input_descs->GetOriginShape().ToString().c_str());
    for (const int64_t dim : input_descs->GetShape().GetDims()) {
      input.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    for (const int64_t dim : input_descs->GetOriginShape().GetDims()) {
      input.mutable_origin_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    int64_t input_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(*input_descs, input_size) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TensorSize]Failed, input %zu, node %s(%s)",
             i, op_desc->GetName().c_str(), op_desc->GetType().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Get input %zu tensor size of node %s(%s) failed",
                        i, op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }

    GELOGI("[Dumper] Node [%s] input[%zu] size %ld addr is %p.", op_desc->GetName().c_str(), i, input_size, addrs[i]);
    input.set_size(static_cast<uint64_t>(input_size));
    input.set_address(static_cast<uint64_t>(addrs[i]));
    input.set_offset(std::numeric_limits<uint64_t>::max());
    input.set_addr_type(GetAddrType(task, *input_descs));
    task.mutable_input()->Add(std::move(input));
  }
  return SUCCESS;
}

void DumpOp::SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc,
                         const std::vector<uintptr_t> &input_addrs, const std::vector<uintptr_t> &output_addrs,
                         rtStream_t const stream) {
  dump_properties_ = dump_properties;
  op_desc_ = op_desc;
  input_addrs_ = input_addrs;
  output_addrs_ = output_addrs;
  stream_ = stream;
}

Status DumpOp::ProtoMallocAndMemcpy(const size_t proto_size, const std::string &proto_msg) {
  GE_FREE_RT_LOG(proto_dev_mem_);
  aclError rt_ret = aclrtMalloc(&proto_dev_mem_, proto_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMalloc]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMalloc failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = aclrtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMemcpy]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GE_FREE_RT_LOG(proto_size_dev_mem_);
  rt_ret = aclrtMalloc(&proto_size_dev_mem_, sizeof(size_t), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMalloc]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMalloc failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = aclrtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMemcpy]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status DumpOp::ExecutorDumpOp(bool need_device_args) {
  std::string proto_msg;
  const size_t proto_size = op_mapping_info_.ByteSizeLong();
  const bool ret = op_mapping_info_.SerializeToString(&proto_msg);
  if ((!ret) || (proto_size == 0U)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Serialize][Protobuf]Failed, proto_size is %zu",
           proto_size);
    REPORT_INNER_ERR_MSG("E19999", "[Serialize][Protobuf]Failed, proto_size is %zu", proto_size);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  const Status status = ProtoMallocAndMemcpy(proto_size, proto_msg);
  if (status != SUCCESS) {
    return status;
  }

  constexpr uint32_t io_addr_num = 2U;
  constexpr uint32_t args_size =
      static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead)) +
      (io_addr_num * static_cast<uint32_t>(sizeof(uint64_t)));
  std::array<uint8_t, args_size> args = {};
  size_t args_pos = 0UL;
  aicpu::AicpuParamHead &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(&args[args_pos])));
  args_pos += sizeof(aicpu::AicpuParamHead);
  param_head.length = args_size;
  param_head.ioAddrNum = io_addr_num;
  *(static_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) = PtrToValue(proto_dev_mem_);
  args_pos += sizeof(uint64_t);
  *(reinterpret_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) = PtrToValue(proto_size_dev_mem_);
  rtArgsEx_t args_for_launch = {};
  if (need_device_args) {
    GE_ASSERT_TRUE(launch_kernel_args_dev_mem_ == nullptr);
    GE_CHK_RT_RET(aclrtMalloc(&launch_kernel_args_dev_mem_, args_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    GE_CHK_RT_RET(aclrtMemcpy(launch_kernel_args_dev_mem_, args_size, &args[0U], args_size, ACL_MEMCPY_HOST_TO_DEVICE));
    args_for_launch.args = launch_kernel_args_dev_mem_;
    args_for_launch.isNoNeedH2DCopy = 1U;
  } else {
    args_for_launch.args = &args[0U];
    args_for_launch.isNoNeedH2DCopy = 0U;
  }
  args_for_launch.argsSize = args_size;
  const rtError_t rt_ret = rtCpuKernelLaunchWithFlag(nullptr, kDumpKernelsDumpOp.c_str(), 1U,
                                                     &args_for_launch, nullptr, stream_, RT_KERNEL_DEFAULT);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtCpuKernelLaunch]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtCpuKernelLaunch failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGI("Kernel launch dump op %s success", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status DumpOp::SetDumpModelName() {
  if (dynamic_model_name_.empty() && dynamic_om_name_.empty()) {
    GELOGI("Single op dump, no need set model name");
    return SUCCESS;
  }
  op_mapping_info_.set_model_id(dynamic_model_id_);
  std::set<std::string> model_list = dump_properties_.GetAllDumpModel();
  const bool not_find_by_omname = model_list.find(dynamic_om_name_) == model_list.end();
  const bool not_find_by_modelname = model_list.find(dynamic_model_name_) == model_list.cend();
  const std::string dump_model_name = not_find_by_omname ? dynamic_model_name_ : dynamic_om_name_;
  if ((!dump_model_name.empty()) && (dump_properties_.IsOpDebugOpen())) {
    GELOGI("Dump model name is %s", dump_model_name.c_str());
    op_mapping_info_.set_model_name(dump_model_name);
    return SUCCESS;
  }
  if ((model_list.find(DUMP_ALL_MODEL) == model_list.end()) &&
      (model_list.find(DUMP_LAYER_OP_MODEL) == model_list.end())) {
    if (not_find_by_omname && not_find_by_modelname) {
      std::string model_list_str;
      for (auto &model : model_list) {
        model_list_str += "[" + model + "].";
      }
      GELOGW("Model %s will not be set to dump, dump list: %s", dump_model_name.c_str(), model_list_str.c_str());
      return FAILED;
    }
  }
  if ((!dump_model_name.empty()) && dump_properties_.IsDumpOpen()) {
    GELOGI("Dump model name is %s", dump_model_name.c_str());
    op_mapping_info_.set_model_name(dump_model_name);
  }
  return SUCCESS;
}

Status DumpOp::UpdateAddrs(const std::vector<uintptr_t> &input_addrs,
                           const std::vector<uintptr_t> &output_addrs) {
  for (auto &task : *op_mapping_info_.mutable_task()) {
    if (dump_properties_.GetDumpMode() == kDumpModeInput) {
      task.clear_input();
      const auto ret = DumpInput(task, op_desc_, input_addrs);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Dump][Input]Update dump input Failed, node %s(%s), ret 0x%X",
              op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
        REPORT_INNER_ERR_MSG("E19999", "Update dump input failed, node %s(%s), ret 0x%X",
                          op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
        return ret;
      }
    }
    if (dump_properties_.GetDumpMode() == kDumpModeOutput) {
      task.clear_output();
      const auto ret = DumpOutput(task, op_desc_, output_addrs);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Dump][Input]Update dump output Failed, node %s(%s), ret 0x%X",
              op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
        return ret;
      }
    }
  }
  std::string proto_msg;
  const size_t proto_size = op_mapping_info_.ByteSizeLong();
  if (proto_size == 0U) {
    GELOGW("[Dump][Update] proto_size is zero");
    return SUCCESS;
  }
  const bool ret = op_mapping_info_.SerializeToString(&proto_msg);
  if (!ret) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Serialize][Protobuf]Failed, proto_size is %zu",
           proto_size);
    REPORT_INNER_ERR_MSG("E19999", "[Serialize][Protobuf]Failed, proto_size is %zu", proto_size);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  auto rt_ret = aclrtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMemcpy]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = aclrtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMemcpy]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

void DumpOp::DumpTask(toolkit::aicpu::dump::Task &task, const uint32_t task_id) {
  GELOGW("Task id is %u, stream id is %u", task_id, stream_id_);
  task.set_task_id(task_id);
  task.set_stream_id(stream_id_);
  task.mutable_op()->set_op_name(op_desc_->GetName());
  task.mutable_op()->set_op_type(op_desc_->GetType());
}

void DumpOp::SaveFftsSubOpInfo(const OpDescPtr &op_desc, const std::vector<Context> &context) {
  ffts_sub_op_list_.push_back({op_desc, context});
}

Status DumpOp::BuildFftsSubOpTask(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  const auto mode = dump_properties_.GetDumpMode();
  uint32_t dump_mode_bits;
  if (mode == kDumpModeAll || dump_properties_.IsOpDebugOpen()) {
    dump_mode_bits = kInputBitsMask | kOutputBitsMask;
  } else if (mode == kDumpModeInput) {
    dump_mode_bits = kInputBitsMask;
  } else if (mode == kDumpModeOutput) {
    dump_mode_bits = kOutputBitsMask;
  } else {
    return SUCCESS;
  }

  for (const auto &op_iter : ffts_sub_op_list_) {
    const auto &op_desc = op_iter.op;
    GELOGD("Op %s in model begin to add ffts task in op_mapping_info", op_desc->GetName().c_str());
    toolkit::aicpu::dump::Task task;
    task.set_end_graph(false);
    task.set_task_id(static_cast<uint32_t>(UINT16_MAX));
    task.set_stream_id(static_cast<uint32_t>(UINT16_MAX));
    task.mutable_op()->set_op_name(op_desc->GetName());
    task.mutable_op()->set_op_type(op_desc->GetType());
    task.set_task_type(toolkit::aicpu::dump::Task::FFTSPLUS);
    for (const auto &context : op_iter.context) {
      toolkit::aicpu::dump::Context ffts_context;
      ffts_context.set_context_id(context.context_id);
      ffts_context.set_thread_id(context.thread_id);
      std::stringstream dbg_ss;
      if ((dump_mode_bits & kInputBitsMask) != 0U) {
        for (const auto &input : context.input) {
          toolkit::aicpu::dump::RealAddressAndSize real_address_and_size;
          real_address_and_size.set_address(input.address);
          real_address_and_size.set_size(input.size);
          ffts_context.mutable_input()->Add(std::move(real_address_and_size));
          dbg_ss << "[input addr: 0x" << &(std::hex) << input.address << ", size: 0x" << input.size << "]";
        }
      }
      if ((dump_mode_bits & kOutputBitsMask) != 0U) {
        for (const auto &output : context.output) {
          toolkit::aicpu::dump::RealAddressAndSize real_address_and_size;
          real_address_and_size.set_address(output.address);
          real_address_and_size.set_size(output.size);
          ffts_context.mutable_output()->Add(std::move(real_address_and_size));
          dbg_ss << "[output addr: 0x" << &(std::hex) << output.address << ", size: 0x" << output.size << "]";
        }
      }
      task.mutable_context()->Add(std::move(ffts_context));
      GELOGD("Op %s add context with context id %u thread id %u, input num %u, output num %u, address info %s",
             op_desc->GetName().c_str(), context.context_id, context.thread_id, context.input.size(),
             context.output.size(), dbg_ss.str().c_str());
    }
    const std::string* ffts_str = AttrUtils::GetStr(*op_desc, ffts::kAttrSgtJsonInfo);
    if (ffts_str != nullptr && !ffts_str->empty()) {
      toolkit::aicpu::dump::OpAttr op_attr;
      op_attr.set_name(ffts::kAttrSgtJsonInfo);
      op_attr.set_value(*ffts_str);
      task.mutable_attr()->Add(std::move(op_attr));
      GELOGI("Add sgt json attr %s in op %s.", ffts_str->c_str(), op_desc->GetName().c_str());
    }

    op_desc_ = op_desc;
    const std::vector<uintptr_t> input_addrs(op_desc->GetAllInputsSize());
    const std::vector<uintptr_t> output_addrs(op_desc->GetAllOutputsDescPtr().size());
    if ((dump_mode_bits & kInputBitsMask) != 0U) {
      GE_CHK_STATUS_RET(DumpInput(task, op_desc, input_addrs, true), "Dump Input failed, node %s(%s)",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    }
    if ((dump_mode_bits & kOutputBitsMask) != 0U) {
      GE_CHK_STATUS_RET(DumpOutput(task, op_desc, output_addrs, true), "Dump Output failed, node %s(%s)",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  return SUCCESS;
}

Status DumpOp::GenerateFftsDump(const DumpProperties &dump_properties, void *&load_dump_info, uint32_t &load_dump_len,
                                void *&unload_dump_info, uint32_t &unload_dump_len, const bool is_single_op_dump) {
  int32_t device_id = 0;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));
  GE_RETURN_WITH_LOG_IF_TRUE(device_id < 0, "Check device_id %d failed", device_id);
  dump_properties_ = dump_properties;

  const auto dump_path = dump_properties_.GetDumpPath() + std::to_string(device_id) + "/";
  if (!is_single_op_dump) {
    op_mapping_info_.set_dump_step(dump_properties_.GetDumpStep());
  }
  const auto dump_data = (dump_properties_.GetDumpData() == kDumpDataDefaultValue)
                             ? toolkit::aicpu::dump::DumpData::STATS_DUMP_DATA
                             : toolkit::aicpu::dump::DumpData::TENSOR_DUMP_DATA;
  op_mapping_info_.set_dump_data(dump_data);
  op_mapping_info_.set_dump_path(dump_path);
  op_mapping_info_.set_flag(kAiCpuLoadFlag);

  if ((!is_single_op_dump) && (SetDumpModelName() != SUCCESS)) {
    return SUCCESS;
  }
  SetLoopAddrToOpMapping(global_step_, loop_per_iter_, loop_cond_, op_mapping_info_);
  GELOGI("Dump step is %s, dump path is %s in Generate ffts plus dump op", dump_properties_.GetDumpStep().c_str(),
         dump_path.c_str());

  GE_CHK_RT_RET(BuildFftsSubOpTask(op_mapping_info_));

  std::string proto_msg;
  const size_t proto_size = op_mapping_info_.ByteSizeLong();
  GE_CHK_BOOL_RET_STATUS(op_mapping_info_.SerializeToString(&proto_msg), FAILED,
                         "op_mapping_info serialize to string failed.");

  if (proto_dev_mem_ != nullptr) {
    GELOGW("proto_dev_mem_ has been used.");
    GE_FREE_RT_LOG(proto_dev_mem_);
  }

  GE_CHK_RT_RET(aclrtMalloc(&proto_dev_mem_, proto_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_CHK_RT_RET(aclrtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE));

  load_dump_info = proto_dev_mem_;
  load_dump_len = static_cast<uint32_t>(proto_size);
  GE_CHK_BOOL_RET_STATUS((load_dump_len == proto_size), FAILED, "load_dump_len != proto_size");

  GE_ASSERT_SUCCESS(BuildUnLoadFftsDumpInfo(unload_dump_info, unload_dump_len));
  ffts_sub_op_list_.clear();
  op_mapping_info_.clear_task(); // clear current task to avoid repetitive dump in next iteration
  return SUCCESS;
}

Status DumpOp::BuildUnLoadFftsDumpInfo(void *&unload_dump_info, uint32_t &unload_dump_len) {
  GELOGI("UnloadDumpInfo start.");
  op_mapping_info_.set_flag(0);
  op_mapping_info_.clear_model_id();

  for (const auto &op_iter : ffts_sub_op_list_) {
    toolkit::aicpu::dump::Task task;
    task.set_task_id(static_cast<uint32_t>(UINT16_MAX));
    task.set_stream_id(static_cast<uint32_t>(UINT16_MAX));
    for (const auto &context : op_iter.context) {
      toolkit::aicpu::dump::Context ffts_context;
      ffts_context.set_context_id(context.context_id);
      ffts_context.set_thread_id(context.thread_id);
      task.mutable_context()->Add(std::move(ffts_context));
    }
    op_mapping_info_.mutable_task()->Add(std::move(task));
  }

  std::string proto_str;
  const size_t proto_size = op_mapping_info_.ByteSizeLong();
  GE_CHK_BOOL_RET_STATUS(op_mapping_info_.SerializeToString(&proto_str), FAILED,
                         "op_mapping_info serialize to string failed.");

  if (dev_mem_unload_ != nullptr) {
    GELOGW("dev_mem_unload_ has been used.");
    GE_FREE_RT_LOG(dev_mem_unload_);
  }

  GE_CHK_RT_RET(aclrtMalloc(&dev_mem_unload_, proto_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "unload dump information.", proto_size);
  GE_CHK_RT_RET(aclrtMemcpy(dev_mem_unload_, proto_size, proto_str.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE));

  unload_dump_info = dev_mem_unload_;
  unload_dump_len = static_cast<uint32_t>(proto_size);
  GE_CHK_BOOL_RET_STATUS((unload_dump_len == proto_size), FAILED, "unload_dump_len != proto_size");
  return SUCCESS;
}

Status DumpOp::LaunchDumpOp(const bool is_single_op_dump, bool need_device_args) {
  GELOGI("Start to launch dump op %s, is single op dump %d, device args flag %d.",
         op_desc_->GetName().c_str(), static_cast<int32_t>(is_single_op_dump), static_cast<int32_t>(need_device_args));

  int32_t device_id = 0;
  const rtError_t rt_ret = aclrtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtGetDevice]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "[Call][aclrtGetDevice]Failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (device_id < 0) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][DeviceId]Failed, device_id %d", device_id);
    REPORT_INNER_ERR_MSG("E19999", "Check device_id %d failed", device_id);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  const auto dump_path = dump_properties_.GetDumpPath() + std::to_string(device_id) + "/";
  op_mapping_info_.clear_task();
  op_mapping_info_.set_dump_path(dump_path);
  op_mapping_info_.set_flag(kAiCpuLoadFlag);
  if (!is_single_op_dump) {
    op_mapping_info_.set_dump_step(dump_properties_.GetDumpStep());
  }
  const auto dump_data = (dump_properties_.GetDumpData() == kDumpDataDefaultValue)
                             ? toolkit::aicpu::dump::DumpData::STATS_DUMP_DATA
                             : toolkit::aicpu::dump::DumpData::TENSOR_DUMP_DATA;
  op_mapping_info_.set_dump_data(dump_data);

  if ((!is_single_op_dump) && (SetDumpModelName() != SUCCESS)) {
    return SUCCESS;
  }
  SetLoopAddrToOpMapping(global_step_, loop_per_iter_, loop_cond_, op_mapping_info_);
  GELOGI("Dump step is %s, dump path is %s in Launch dump op", dump_properties_.GetDumpStep().c_str(),
         dump_path.c_str());
  if ((task_id_ == 0U) || (stream_id_ == 0U)) {
    GE_CHK_RT(rtsGetThreadLastTaskId(&task_id_));
    int32_t temp_stream_id;
    GE_CHK_RT(rtsStreamGetId(stream_, &temp_stream_id));
    stream_id_ = static_cast<uint32_t>(temp_stream_id);
  }
  int32_t bit_width;
  GE_CHK_RT(rtsDeviceGetCapability(device_id, RT_FEATURE_SYSTEM_TASKID_BIT_WIDTH, &bit_width));
  if (bit_width == k16BitWidth) {
    task_id_ = task_id_ & k16BitsMask;
  }
  toolkit::aicpu::dump::Task task;
  DumpTask(task, task_id_);
  return ExecuteDump(task, need_device_args);
}

Status DumpOp::ExecuteDump(toolkit::aicpu::dump::Task &task, bool need_device_args) {
  const Status status = LaunchDump(task);
  if (status != SUCCESS) {
    return status;
  }
  const auto ret = ExecutorDumpOp(need_device_args);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Dump][Op]Failed, ret 0x%X", ret);
    return ret;
  }
  GELOGI("Dump %s success", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status DumpOp::LaunchDump(toolkit::aicpu::dump::Task &task) {
  if (dump_properties_.GetDumpMode() == kDumpModeOutput) {
    const auto ret = DumpOutput(task, op_desc_, output_addrs_);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Output]Failed, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
  } else if (dump_properties_.GetDumpMode() == kDumpModeInput) {
    const auto ret = DumpInput(task, op_desc_, input_addrs_);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input]Failed, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_INNER_ERR_MSG("E19999", "Dump Input failed, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
  } else if ((dump_properties_.GetDumpMode() == kDumpModeAll) || dump_properties_.IsOpDebugOpen()) {
    DumpWorkspace(task);
    auto ret = DumpOutput(task, op_desc_, output_addrs_);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Output]Failed when in dumping all, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    ret = DumpInput(task, op_desc_, input_addrs_);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input]Failed when in dumping all, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_INNER_ERR_MSG("E19999", "Dump Input failed when in dumping all, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
  } else {
    // for future use of dump mode
  }
  op_mapping_info_.mutable_task()->Add(std::move(task));
  return SUCCESS;
}
}  // namespace ge
