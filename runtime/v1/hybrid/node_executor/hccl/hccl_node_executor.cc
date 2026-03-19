/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/hccl/hccl_node_executor.h"

#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/math/math_util.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/util/hcom_ome_util.h"
#include "graph/utils/type_utils.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "graph/def_types.h"

namespace ge {
namespace {
constexpr size_t kVarTableDims = 2UL;
constexpr size_t kVarTableRowCnt = 3UL;
constexpr size_t kVarTableIdxAddr = 1UL;
constexpr size_t kVarTableIdxLen = 2UL;
// input anchor nums according to IR
constexpr size_t kAllToAllVInputNums = 5UL;
constexpr size_t kGatherAllToAllVInputNums = 4UL;
constexpr size_t kAllToAllVCInputNums = 2UL;
const std::string kHcomOpAttr = "group";

const std::set<std::string> kRdmaReadTypes = { HCOMREMOTEREAD, HCOMREMOTEREFREAD };
const std::set<std::string> kRdmaWriteTypes = { HCOMREMOTEWRITE, HCOMREMOTESCATTERWRITE };
const std::set<std::string> kRdmaScatterTypes = { HCOMREMOTEREFREAD, HCOMREMOTESCATTERWRITE };
const std::set<std::string> kAllToAllTypes = {HCOMALLTOALLV, HCOMGATHERALLTOALLV, HCOMALLTOALLVC};
}  // namespace
namespace hybrid {

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HCCL, HcclNodeExecutor);

Status HcclNodeTask::FillHcomOpInfo(const TaskContext &context, const OpDescPtr op_desc,
                                    const std::vector<void *> &inputs,
                                    const std::vector<void *> &outputs,
                                    HcomOperation &hcom_op_info) {
  hcom_op_info.hcclType = op_desc->GetType();
  hcom_op_info.inputPtr = inputs.empty() ? nullptr : inputs[0UL];
  hcom_op_info.outputPtr = outputs.empty() ? nullptr : outputs[0UL];
  const auto input_desc = context.GetNodeItem().MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  const ge::DataType src_data_type = input_desc->GetDataType();
  const auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) inputdesc0 datatype:%s not support.", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType] %s(%s) inputdesc0 datatype:%s not support.", op_desc->GetName().c_str(),
           op_desc->GetType().c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return PARAM_INVALID;
  }
  hcom_op_info.dataType = iter->second;
  const std::set<std::string> hccl_types = { HCOMALLREDUCE, HCOMREDUCESCATTER, HVDCALLBACKALLREDUCE, HCOMREDUCE };
  if (hccl_types.count(op_desc->GetType()) > 0UL) {
    HcclReduceOp op_type = HCCL_REDUCE_SUM;
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclOperationType(op_desc, op_type),
                      "[Get][HcclOperationType] failed for %s type:%s", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str());
    hcom_op_info.opType = op_type;
  }
  int64_t root_id = 0;
  if ((op_desc->GetType() == HCOMBROADCAST) || (op_desc->GetType() == HCOMGATHER)) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclRootId(op_desc, root_id), "[Get][HcclRootId] failed for %s type:%s",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }
  hcom_op_info.root = static_cast<uint32_t>(root_id);
  int64_t count = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcomCount(op_desc, static_cast<HcclDataType>(hcom_op_info.dataType),
                                              op_desc->GetType() == HCOMALLGATHER, count),
                    "[Get][HcomCount] failed for %s type:%s", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  GELOGI("[%s] HcclNodeTask::ExecuteAsync hccl_type %s, count %d, data_type %d, op_type %d, root %u.",
         context.GetNodeName(), hcom_op_info.hcclType.c_str(), count, hcom_op_info.dataType, hcom_op_info.opType,
         hcom_op_info.root);
  hcom_op_info.count = static_cast<uint64_t>(count);
  return SUCCESS;
}

Status HcclNodeTask::GetInputsOutPuts(const TaskContext &context, std::vector<void *> &inputs,
                                      std::vector<void *> &outputs) {
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    TensorValue *const tv = context.MutableInput(i);
    GE_CHECK_NOTNULL(tv);
    inputs.emplace_back(tv->MutableData());
  }

  for (int32_t i = 0; i < context.NumOutputs(); ++i) {
    TensorValue *const tv = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tv);
    outputs.emplace_back(tv->MutableData());
  }
  return SUCCESS;
}

Status HcclNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGI("[%s] HcclNodeTask::ExecuteAsync in.", context.GetNodeName());
  if (context.GetContextHandle() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", " %s(%s) invalid, hccl handle is nullptr!",
                       context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    GELOGE(FAILED, "[Check][Param:context] %s(%s) hccl handle is nullptr!",
           context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    return FAILED;
  }

  std::vector<void *> inputs;
  std::vector<void *> outputs;
  auto ret = GetInputsOutPuts(context, inputs, outputs);
  GE_CHK_STATUS_RET(ret, "GetInputsOutPuts failed, node name[%s].", context.GetNodeName());
  const NodeItem &node_item = context.GetNodeItem();
  const OpDescPtr op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  const auto hcom_exec_enqueue_op =
      reinterpret_cast<HcclResult(*)(HcomOperation, std::function<void(const HcclResult status)>)>(mmDlsym(
          context.GetContextHandle(), "HcomExecEnqueueOperation"));
  if (hcom_exec_enqueue_op == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecEnqueueOperation] failed for %s(%s) hcom unknown node function.",
           context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    if (mmDlclose(context.GetContextHandle()) != 0) {
      GELOGW("Failed to close handle %s", mmDlerror());
    }
    return FAILED;
  }

  HcomOperation hcom_op_info;
  ret = FillHcomOpInfo(context, op_desc, inputs, outputs, hcom_op_info);
  GE_CHK_STATUS_RET(ret, "FillHcomOpInfo failed, node name[%s].", context.GetNodeName());
  const auto callback = [op_desc, done_callback](const HcclResult stat) {
    if (stat != HCCL_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "call HcomExecEnqueueOperation failed for node %s(%s), ret: %d",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), stat);
      GELOGE(HCCL_E_INTERNAL, "[Call][HcomExecEnqueueOperation] failed for node %s(%s), ret: %d",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), stat);
    }
    if (done_callback != nullptr) {
      done_callback();
    }
    GELOGI("node %s hccl callback success.", op_desc->GetName().c_str());
  };

  const HcclResult hccl_ret = hcom_exec_enqueue_op(hcom_op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call HcomExecEnqueueOperation failed for node:%s(%s), ret: %d",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), hccl_ret);
    GELOGE(HCCL_E_INTERNAL, "[Call][HcomExecEnqueueOperation] failed for node:%s(%s), ret: %d",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), hccl_ret);
    return HCCL_E_INTERNAL;
  }

  GELOGI("[%s] HcclNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status RdmaNodeTask::UpdateArgs(TaskContext &context) {
  (void)context;
  return SUCCESS;
}

Status RdmaNodeTask::Init(TaskContext &context) {
  GELOGI("[%s] RdmaNodeTask::Init in.", context.GetNodeName());
  const NodeItem &node_item_ref = context.GetNodeItem();
  const auto op_desc_p = node_item_ref.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_p);
  const auto remote_idx = op_desc_p->GetInputIndexByName("remote");
  const auto in_data_anchor = node_item_ref.node->GetInDataAnchor(remote_idx);
  GE_CHECK_NOTNULL(in_data_anchor);
  const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(out_data_anchor);
  const auto peer_node = out_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_node->GetOpDesc());

  remote_index_ = {peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx()};
  if (kRdmaReadTypes.count(node_item_ref.node->GetType()) != 0UL) {
    local_index_ = 0;
  } else {
    local_index_ = op_desc_p->GetInputIndexByName("local");
  }
  const int32_t offset_idx = node_item_ref.op_desc->GetInputIndexByName("local_offset");
  if ((offset_idx != -1) && (node_item_ref.op_desc->GetInputDescPtr(static_cast<uint32_t>(offset_idx)) != nullptr)) {
    skip_flag_ = true;
    GE_CHECK_NOTNULL(node_item_ref.node->GetInDataAnchor(offset_idx));
    GE_CHECK_NOTNULL(node_item_ref.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor());
    GE_CHECK_NOTNULL(node_item_ref.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetOwnerNode());
    GE_CHECK_NOTNULL(node_item_ref.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc());
    offset_index_ = {
        node_item_ref.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetId(),
        node_item_ref.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetIdx() };
  }
  return SUCCESS;
}

Status RdmaNodeTask::GetOffsetTensor(const TaskContext &context, const RuntimeInferenceContext &rt_ctx,
                                     const size_t row_num, GeTensorPtr &offset_tensor) const {
  const int32_t offset_idx = context.GetNodeItem().op_desc->GetInputIndexByName("local_offset");
  GE_CHECK_NOTNULL(context.GetNodeItem().op_desc->GetInputDescPtr(static_cast<uint32_t>(offset_idx)));
  const auto in_data_type =
      context.GetNodeItem().op_desc->GetInputDesc(static_cast<uint32_t>(offset_idx)).GetDataType();

  GE_CHK_STATUS_RET(rt_ctx.GetTensor(offset_index_.first, static_cast<int32_t>(offset_index_.second), offset_tensor));
  GE_CHECK_NOTNULL(offset_tensor);
  const size_t tensor_size = offset_tensor->GetData().GetSize();
  if ((tensor_size / static_cast<size_t>(GetSizeByDataType(in_data_type))) != row_num) {
    REPORT_INNER_ERR_MSG("E19999", "num of offset and remote addr mismatch, check invalid"
                                 "offset size=%zu, remote_addr size=%zu, dtype=%s", tensor_size, row_num,
                       TypeUtils::DataTypeToSerialString(in_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Check][Size]num of offset and remote addr mismatch,"
                          "offset size=%zu, remote_addr size=%zu, dtype=%s",
           tensor_size, row_num, TypeUtils::DataTypeToSerialString(in_data_type).c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status RdmaNodeTask::SetAddrInfo(const TaskContext &context, const RuntimeInferenceContext &rt_ctx,
                                 const uint64_t *const data,
                                 const size_t row_num,
                                 std::vector<HcomRemoteAccessAddrInfo> &addr_infos) const {
  TensorValue *tv = nullptr;
  GeTensorDescPtr tensor_desc = nullptr;
  if (kRdmaReadTypes.count(context.GetNodeItem().NodeType()) != 0UL) {
    tv = context.MutableOutput(local_index_);
    tensor_desc = context.MutableOutputDesc(local_index_);
  } else {
    tv = context.MutableInput(local_index_);
    tensor_desc = context.MutableInputDesc(local_index_);
  }
  GE_CHECK_NOTNULL(tv);
  addr_infos.resize(row_num);
  if (skip_flag_) {
    GeTensorPtr offset_tensor;
    GE_CHK_STATUS_RET(GetOffsetTensor(context, rt_ctx, row_num, offset_tensor),
                      "GetOffsetTensor failed, node name[%s].", context.GetNodeName());
    const auto addr_offset = PtrToPtr<const uint8_t, const uint64_t>(offset_tensor->GetData().GetData());
    GE_CHECK_NOTNULL(addr_offset);
    const auto base_addr = PtrToPtr<void, float32_t>(tv->MutableData());
    GE_CHECK_NOTNULL(base_addr);

    for (auto idx = 0UL; idx < row_num; idx++) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt)
      const auto line_idx = idx * kVarTableRowCnt;
      addr_infos[idx] = { static_cast<uint32_t>(data[line_idx]),
                          data[line_idx + kVarTableIdxAddr],
                          PtrToValue(static_cast<const void *>(base_addr + addr_offset[idx])),
                          data[line_idx + kVarTableIdxLen] };
    }
  } else {
    auto local_addr = PtrToValue(tv->MutableData());
    const auto device_len = (row_num == 0UL) ? 0UL :
        static_cast<uint64_t>(tensor_desc->GetShape().GetShapeSize()) / row_num * sizeof(float32_t);
    const auto data_len_ptr = PtrAdd<const uint64_t>(data, row_num, kVarTableIdxLen);
    GE_CHECK_NOTNULL(data_len_ptr);
    const uint64_t data_len = *data_len_ptr;
    if ((device_len <= 0UL) || (device_len > data_len)) {
      REPORT_INNER_ERR_MSG("E19999", "Local embedding length is out of range, expect "
		         "%" PRIu64 ", but %" PRIu64 " exactly.", data_len, static_cast<uint64_t>(device_len));
      GELOGE(FAILED, "[Check][Size]Local embedding length is out of range, expect %lu, but %lu exactly.",
             data_len, device_len);
      return FAILED;
    }

    for (size_t idx = 0UL; idx < row_num; ++idx) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt)
      const auto line_idx = idx * kVarTableRowCnt;
      addr_infos[idx] = { static_cast<uint32_t>(data[line_idx]), data[line_idx + kVarTableIdxAddr], local_addr,
                          device_len };
      local_addr += device_len;
    }
  }

  return SUCCESS;
}

Status RdmaNodeTask::ExtractTensor(const TaskContext &context,
                                   std::vector<HcomRemoteAccessAddrInfo> &addr_infos) const {
  RuntimeInferenceContext &ctx = context.GetExecutionContext()->runtime_context_;
  GeTensorPtr remote_tensor;
  GE_CHK_STATUS_RET(ctx.GetTensor(remote_index_.first, static_cast<int32_t>(remote_index_.second), remote_tensor));
  const auto data = PtrToPtr<const uint8_t, const uint64_t>(remote_tensor->GetData().GetData());
  if (data == nullptr) {
    if (kRdmaScatterTypes.count(context.GetNodeItem().NodeType()) != 0UL) {
      GELOGD("data is null, no need to do rdma read/write, node=%s", context.GetNodeName());
      return SUCCESS;
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Tensor data is nullptr. and kRdmaScatterTypes not contain %s",
                         context.GetNodeItem().NodeType().c_str());
      GELOGE(FAILED, "[Find][NodeType]Tensor data is nullptr. and kRdmaScatterTypes not contain %s",
             context.GetNodeItem().NodeType().c_str());
      return FAILED;
    }
  }
  const auto dims = remote_tensor->GetTensorDesc().GetShape().GetDims();
  if ((dims.size() != kVarTableDims) && (static_cast<size_t>(dims.back()) != kVarTableRowCnt)) {
    REPORT_INNER_ERR_MSG("E19999", "Variable table shape check failed, number of shape dims:%zu not equal expect:%zu "
        "and shape dims back:%" PRId64 " not equal expect:%zu, node:%s(%s)", dims.size(), kVarTableDims,
        dims.back(), kVarTableRowCnt, context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Variable table shape check failed, number of shape dims:%zu not equal "
        "expect:%zu and shape dims back:%" PRId64 " not equal expect:%zu, node:%s(%s)", dims.size(), kVarTableDims,
        dims.back(), kVarTableRowCnt, context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    return PARAM_INVALID;
  }

  AllocationAttr attr;
  if (context.GetNodeItem().NodeType() == HCOMREMOTEREAD) {
    size_t remote_size = 0UL;
    for (size_t idx = 0UL; idx < static_cast<size_t>(dims.front()); ++idx) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt);
      const auto size_ptr = PtrAdd<const uint64_t>(data, remote_tensor->GetData().GetSize(),
                                                   (idx * kVarTableRowCnt) + kVarTableIdxLen);
      GE_CHECK_NOTNULL(size_ptr);
      remote_size += *size_ptr;
    }
    GE_CHECK_NOTNULL(NpuMemoryAllocator::GetAllocator());
    attr.SetMemType(MemStorageType::RDMA_HBM);
    for (int32_t i = 0; i < context.NumOutputs(); ++i) {
      GELOGD("Allocate rdma memory for node %s, size: %zu", context.GetNodeName(), remote_size);
      auto tensor_buffer = TensorBuffer::Create(NpuMemoryAllocator::GetAllocator(), remote_size, &attr);
      GE_CHK_STATUS_RET(context.SetOutput(i, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer.release()))));
    }
  } else if (context.GetNodeItem().NodeType() == HCOMREMOTEREFREAD) {
    attr.SetMemType(MemStorageType::RDMA_HBM);
    GE_CHK_STATUS_RET(context.AllocateOutputs(&attr));
  } else {} // no operation
  return SetAddrInfo(context, ctx, data, static_cast<size_t>(dims.front()), addr_infos);
}

Status RdmaNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGI("[%s] RdmaNodeTask::ExecuteAsync in.", context.GetNodeName());
  const auto hcom_exec_enqueue_remote_access =
      reinterpret_cast<HcclResult(*)(const std::string &, const std::vector<HcomRemoteAccessAddrInfo> &,
          std::function<void(const HcclResult status)>)>(
          mmDlsym(context.GetContextHandle(), "HcomExecEnqueueRemoteAccess"));
  if (hcom_exec_enqueue_remote_access == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecEnqueueRemoteAccess] failed for node:%s(%s) hcom unknown node function.",
           context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    if (mmDlclose(context.GetContextHandle()) != 0) {
      GELOGW("Failed to close handle %s", mmDlerror());
    }
    return FAILED;
  }
  std::vector<HcomRemoteAccessAddrInfo> addr_infos;
  GE_CHK_STATUS_RET(ExtractTensor(context, addr_infos));
  if (addr_infos.empty()) {
    if (done_callback != nullptr) {
      done_callback();
    }
    return SUCCESS;
  }

  aclrtEvent evt = nullptr;
  if (context.GetExecutionContext()->hccl_stream != nullptr) {
    GE_CHK_RT_RET(aclrtCreateEventWithFlag(
      &evt, ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS | ACL_EVENT_TIME_LINE));
    GE_CHK_RT_RET(rtStreamWaitEvent(context.GetExecutionContext()->hccl_stream, evt));
  }
  TaskContext *const p_ctx = &context;
  const auto callback = [p_ctx, done_callback, evt](const HcclResult stat) {
    if (stat != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "[Call][HcomExcutorInitialize] failed for node:%s(%s), ret: 0x%X",
             p_ctx->GetNodeName(), p_ctx->GetNodeItem().NodeType().c_str(), stat);
      p_ctx->SetStatus(FAILED);
    }
    if (done_callback != nullptr) {
      done_callback();
    }
    if (evt != nullptr) {
      GE_CHK_RT_RET(aclrtRecordEvent(evt, nullptr));
      GE_CHK_RT_RET(aclrtDestroyEvent(evt));
    }
    GELOGI("rdma callback success.");
    return SUCCESS;
  };

  const HcclResult hccl_ret = hcom_exec_enqueue_remote_access(context.GetNodeItem().NodeType(), addr_infos, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(HCCL_E_INTERNAL, "[Call][HcomExecEnqueueRemoteAccess] failed for node:%s(%s), ret: 0x%X",
           context.GetNodeName(), context.GetNodeItem().NodeType().c_str(), hccl_ret);
    return HCCL_E_INTERNAL;
  }

  GELOGI("[%s] RdmaNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

static Status BuildAllToAllVparams(const TaskContext &context, HcomAllToAllVParams &params,
                                   std::string &alltoallv_group) {
  void **input_addrs[kAllToAllVInputNums] = {&params.sendbuf, &params.sendcounts, &params.sdispls, &params.recvcounts,
                                             &params.rdispls};
  for (size_t i = 0UL; i < kAllToAllVInputNums; ++i) {
    const auto addr = context.MutableInput(static_cast<int32_t>(i));
    GE_CHECK_NOTNULL(addr);
    *input_addrs[i] = addr->MutableData();
  }
  const auto recv_tv = context.MutableOutput(0);
  GE_CHECK_NOTNULL(recv_tv);
  params.recvbuf = recv_tv->MutableData();

  const NodeItem &node_item_ref = context.GetNodeItem();
  const OpDescPtr op_desc_p = node_item_ref.GetOpDesc();
  const auto input_desc = node_item_ref.MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  const ge::DataType src_data_type = input_desc->GetDataType();
  const auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) alltoallv datatype:%s not support.", op_desc_p->GetName().c_str(),
                       op_desc_p->GetType().c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType] %s(%s) alltoallv datatype:%s not support.", op_desc_p->GetName().c_str(),
           op_desc_p->GetType().c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return PARAM_INVALID;
  }
  params.sendtype = iter->second;
  params.recvtype = iter->second;

  (void)ge::AttrUtils::GetStr(op_desc_p, kHcomOpAttr, alltoallv_group);
  params.group = alltoallv_group.c_str();

  return SUCCESS;
}

static Status BuildGatherAllToAllParams(const TaskContext &context, HcomGatherAllToAllVParams &params,
                                        std::string &gatheralltoallv_group) {
  void **input_addrs[kGatherAllToAllVInputNums] = {&params.addrInfo, &params.addrInfoCountPerRank, &params.recvcounts,
                                                   &params.rdispls};
  for (size_t i = 0UL; i < kGatherAllToAllVInputNums; ++i) {
    const auto addr = context.MutableInput(static_cast<int32_t>(i));
    GE_CHECK_NOTNULL(addr);
    *input_addrs[i] = addr->MutableData();
  }
  const auto recv_tv = context.MutableOutput(0);
  GE_CHECK_NOTNULL(recv_tv);
  params.recvbuf = recv_tv->MutableData();
  const auto gathered_tv = context.MutableOutput(1);
  GE_CHECK_NOTNULL(gathered_tv);
  params.gatheredbuf = gathered_tv->MutableData();

  const NodeItem &node_item = context.GetNodeItem();
  const OpDescPtr op_desc = node_item.GetOpDesc();

  ge::DataType hccl_data_type = ge::DT_FLOAT;
  (void)ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, hccl_data_type);
  const auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(hccl_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) received datatype:%s not support.", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str(), TypeUtils::DataTypeToSerialString(hccl_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType] %s(%s) received datatype:%s not support.", op_desc->GetName().c_str(),
           op_desc->GetType().c_str(), TypeUtils::DataTypeToSerialString(hccl_data_type).c_str());
    return PARAM_INVALID;
  }
  params.recvtype = iter->second;

  int64_t addr_len = 0;
  (void)ge::AttrUtils::GetInt(op_desc, "addr_length", addr_len);
  params.addrLength = static_cast<int32_t>(addr_len);
  (void)ge::AttrUtils::GetStr(op_desc, kHcomOpAttr, gatheralltoallv_group);
  params.group = gatheralltoallv_group.c_str();

  return SUCCESS;
}

static Status BuildAllToAllVCParams(const TaskContext &context, HcomAllToAllVCParams &params,
                                    std::string &alltoallvc_group) {
  void **input_addrs[kAllToAllVCInputNums] = {&params.sendbuf, &params.sendcountmatrix};
  for (size_t i = 0UL; i < kAllToAllVCInputNums; ++i) {
    const auto addr = context.MutableInput(static_cast<int32_t>(i));
    GE_CHECK_NOTNULL(addr);
    *input_addrs[i] = addr->MutableData();
  }
  const auto recv_tv = context.MutableOutput(0);
  GE_CHECK_NOTNULL(recv_tv);
  params.recvbuf = recv_tv->MutableData();
  const NodeItem &node_item_ref = context.GetNodeItem();
  const OpDescPtr op_desc_p = node_item_ref.GetOpDesc();
  const auto input_desc = node_item_ref.MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  const ge::DataType src_data_type = input_desc->GetDataType();
  const auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) alltoallv datatype:%s not support.", op_desc_p->GetName().c_str(),
                       op_desc_p->GetType().c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType] %s(%s) alltoallv datatype:%s not support.", op_desc_p->GetName().c_str(),
           op_desc_p->GetType().c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return PARAM_INVALID;
  }
  params.sendtype = iter->second;
  params.recvtype = iter->second;

  (void)ge::AttrUtils::GetStr(op_desc_p, kHcomOpAttr, alltoallvc_group);
  params.group = alltoallvc_group.c_str();
  return SUCCESS;
}
Status AllToAllNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGI("[%s] AllToAllNodeTask::ExecuteAsync in.", context.GetNodeName());

  TaskContext *const p_ctx = &context;
  const auto callback = [p_ctx, done_callback](const HcclResult stat) {
    if (stat != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "[Run][CallBack] [%s(%s)] AllToAllNodeTask execute failed.",
             p_ctx->GetNodeName(), p_ctx->GetNodeItem().NodeType().c_str());
      p_ctx->SetStatus(FAILED);
    }
    if (done_callback != nullptr) {
      done_callback();
    }
    GELOGI("[%s] AllToAllNodeTask callback successfully.", p_ctx->GetNodeName());
  };

  if (context.GetNodeItem().NodeType() == HCOMALLTOALLV) {
    const auto hcom_exec_enqueue_all_to_allv =
        reinterpret_cast<HcclResult(*)(HcomAllToAllVParams, std::function<void(const HcclResult status)>)>(mmDlsym(
            context.GetContextHandle(), "HcomExecEnqueueAllToAllV"));
    if (hcom_exec_enqueue_all_to_allv == nullptr) {
      GELOGE(FAILED, "[Invoke][Function] [HcomExecEnqueueAllToAllV] for node:%s(%s) failed.",
             context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
      return FAILED;
    }
    HcomAllToAllVParams params;
    std::string alltoallv_group;
    GE_CHK_STATUS_RET(BuildAllToAllVparams(context, params, alltoallv_group));
    const HcclResult hccl_ret = hcom_exec_enqueue_all_to_allv(params, callback);
    if (hccl_ret != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "[Process][HcomExecEnqueueAllToAllV] AllToAllV teak enqueue failed for node [%s(%s)].",
             context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
      return HCCL_E_INTERNAL;
    }
  } else if (context.GetNodeItem().NodeType() == HCOMGATHERALLTOALLV) {
    const auto hcom_enqueue_gather_all_to_allv =
        reinterpret_cast<HcclResult(*)(HcomGatherAllToAllVParams, std::function<void(const HcclResult status)>)>(
            mmDlsym(context.GetContextHandle(), "HcomExecEnqueueGatherAllToAllV"));
    if (hcom_enqueue_gather_all_to_allv == nullptr) {
      GELOGE(FAILED, "[Invoke][Function] [HcomExecEnqueueGatherAllToAllV] for node:%s(%s) failed.",
             context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
      return FAILED;
    }
    HcomGatherAllToAllVParams params;
    std::string gatheralltoallv_group;
    GE_CHK_STATUS_RET(BuildGatherAllToAllParams(context, params, gatheralltoallv_group));
    const HcclResult hccl_ret = hcom_enqueue_gather_all_to_allv(params, callback);
    if (hccl_ret != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL,
             "[Process][HcomExecEnqueueGatherAllToAllV] GatherAllToAllV teak enqueue failed for node [%s(%s)].",
             context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
      return HCCL_E_INTERNAL;
    }
  } else {
    const auto hcom_enqueue_all_to_allv_c =
        reinterpret_cast<HcclResult (*)(HcomAllToAllVCParams, std::function<void(const HcclResult status)>)>(
            mmDlsym(context.GetContextHandle(), "HcomExecEnqueueAllToAllVC"));
    if (hcom_enqueue_all_to_allv_c == nullptr) {
      GELOGE(FAILED, "[Invoke][Function] [HcomExecEnqueueAllToAllVC] for node:%s(%s) failed.", context.GetNodeName(),
             context.GetNodeItem().NodeType().c_str());
      return FAILED;
    }
    HcomAllToAllVCParams params;
    std::string alltoallvc_group;
    GE_CHK_STATUS_RET(BuildAllToAllVCParams(context, params, alltoallvc_group));
    const HcclResult hccl_ret = hcom_enqueue_all_to_allv_c(params, callback);
    if (hccl_ret != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL,
             "[Process][HcomAllToAllVCParams] AllToAllVC teak enqueue failed for node [%s(%s)].",
             context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
      return HCCL_E_INTERNAL;
    }
  }
  GELOGI("[%s] AllToAllNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeTask::UpdateArgs(TaskContext &context) {
  (void)context;
  return SUCCESS;
}

Status HcclNodeTask::Init(TaskContext &context) {
  GELOGI("[%s] HcclNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGI("[%s] HcclNodeExecutor::PrepareTask in.", context.GetNodeName());

  GE_CHK_STATUS_RET(task.Init(context), "[Invoke][Init]hccl node %s(%s) load hccl so failed.",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  // allocate output mem, output mem or remote read will be calculated when node execute.
  if (kRdmaReadTypes.count(context.GetNodeItem().NodeType()) == 0UL) {
    GE_CHK_STATUS_RET(context.AllocateOutputs(),
                      "[Invoke][AllocateOutputs]hccl node %s(%s) task allocate output failed.",
                      context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  }

  GE_CHK_STATUS_RET(task.UpdateArgs(context), "[Update][Args] failed for hccl node %s(%s).",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  GELOGI("[%s] HcclNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  (void)model;
  GELOGI("[%s] HcclNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);
  if ((kRdmaReadTypes.count(node->GetType()) != 0UL) || (kRdmaWriteTypes.count(node->GetType()) != 0UL)) {
    task = MakeShared<RdmaNodeTask>();
  } else if (kAllToAllTypes.count(node->GetType()) != 0UL) {
    task = MakeShared<AllToAllNodeTask>();
  } else {
    task = MakeShared<HcclNodeTask>();
  }
  GE_CHECK_NOTNULL(task);
  GELOGI("[%s] HcclNodeExecutor::LoadTask success.", node->GetName().c_str());
  return SUCCESS;
}

Status HcclNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                     const std::function<void()> &callback) const {
  context.SetContextHandle(handle_);
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback),
                    "[Invoke][ExecuteAsync] failed to execute task. node:%s(%s)",
                    context.GetNodeItem().NodeName().c_str(), context.GetNodeItem().NodeType().c_str());
  return SUCCESS;
}

Status HcclNodeExecutor::Initialize() {
  const std::string file_name = "libhcom_graph_adaptor.so";
  std::string path = GetModelPath();
  (void)path.append(file_name);
  const std::string canonical_path = RealPath(path.c_str());
  if (canonical_path.empty()) {
    GELOGW("failed to get realpath of %s", path.c_str());
    return FAILED;
  }

  GELOGI("FileName:%s, Path:%s.", file_name.c_str(), canonical_path.c_str());
  handle_ = mmDlopen(canonical_path.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                     static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  if (handle_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Open SoFile %s failed, error:%s! ", canonical_path.c_str(), mmDlerror());
    GELOGE(GE_PLGMGR_SO_NOT_EXIST, "[Open][SoFile] %s failed, error:%s! ", canonical_path.c_str(), mmDlerror());
    return FAILED;
  }
  const auto HcomExecInitialize =
      reinterpret_cast<HcclResult(*)()>(mmDlsym(handle_, "HcomExecInitialize"));
  if (HcomExecInitialize == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecInitialize] Failed for hcom unknown node function.");
    return FAILED;
  }
  const HcclResult hccl_ret = HcomExecInitialize();
  if (hccl_ret == HCCL_E_PTR) {
    GELOGI("Hccl comm is null, hcom executor initialize is not required.");
  } else if (hccl_ret == HCCL_SUCCESS) {
    GELOGI("Hcom executor initialize success.");
  } else {
    GELOGE(FAILED, "[Call][HcomExecInitialize] failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  return SUCCESS;
}

Status HcclNodeExecutor::Finalize() {
  const auto HcomExecFinalize = reinterpret_cast<HcclResult(*)()>(mmDlsym(handle_, "HcomExecFinalize"));
  if (HcomExecFinalize == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecFinalize] failed for hcom unknown node function.");
    return FAILED;
  }
  const HcclResult hccl_ret = HcomExecFinalize();
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(FAILED, "[Call][HcomExecFinalize] failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  // mmDlclose file handle
  if (mmDlclose(handle_) != 0) {
    GELOGW("Failed to close handle %s", mmDlerror());
  }
  GELOGI("Hcom executor finalize success.");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
