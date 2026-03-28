/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rts_ops_kernel_builder.h"
#include "register/ops_kernel_builder_registry.h"
#include <string>
#include "framework/common/ge_inner_error_codes.h"
#include "common/constant/constant.h"
#include "common/util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "op/op_factory.h"
#include "proto/task.pb.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/args_format_desc.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

constexpr uint64_t RT_MEMCPYASYNC_SPLIT_SIZE = 67108864UL;  // 64*1024*1024
constexpr uint32_t RT_GENERAL_SQE_NUM = 3U;
constexpr uint8_t MEM_WAIT_SQE_NUM = 4U;

#ifdef WIN32
#define __THREAD_LOCAL__ __declspec(thread)
#else
#define __THREAD_LOCAL__ __thread
#endif

REGISTER_OPS_KERNEL_BUILDER(RTS_OP_KERNEL_LIB_NAME, RtsOpsKernelBuilder);

Status RtsOpsKernelBuilder::Initialize(const map<string, string> &options) {
  (void)options;
  RTS_LOGD("RtsOpsKernelBuilder init start.");

  OpInfo defaultOpInfo = {};
  defaultOpInfo.engine = RTS_ENGINE_NAME;
  defaultOpInfo.opKernelLib = RTS_OP_KERNEL_LIB_NAME;
  defaultOpInfo.computeCost = 0;
  defaultOpInfo.flagPartial = false;
  defaultOpInfo.flagAsync = false;
  defaultOpInfo.isAtomic = false;

  // init op_info_map_
  auto allOps = OpFactory::Instance().GetAllOps();
  for (auto &op : allOps) {
    op_info_map_[op] = defaultOpInfo;
  }

  RTS_LOGI("RtsOpsKernelBuilder inited success. op num=%zu.", op_info_map_.size());
  return SUCCESS;
}

Status RtsOpsKernelBuilder::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

Status RtsOpsKernelBuilder::CalcOpRunningParam(Node &geNode) {
  bool isNeedCalc = false;
  auto ret = IsNeedCalcOpRunningParam(geNode, isNeedCalc);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!isNeedCalc) {
    return SUCCESS;
  }
  OpDescPtr opDesc = geNode.GetOpDesc();
  const string nodeName = geNode.GetName();
  const string nodeType = geNode.GetType();
  size_t outputSize = opDesc->GetOutputsSize();
  RTS_LOGI("[%s: %s] output size=%zu.", nodeName.c_str(), nodeType.c_str(), outputSize);

  int64_t nodeSqeNum = 0;
  for (size_t i = 0; i < outputSize; ++i) {
    int64_t outputMemSize = 0;
    GeTensorDesc outputTensor = opDesc->GetOutputDesc(i);
    Format format = outputTensor.GetFormat();
    DataType dataType = outputTensor.GetDataType();
    graphStatus grhStatus = ge::TensorUtils::GetTensorMemorySizeInBytes(outputTensor, outputMemSize);
    nodeSqeNum += outputMemSize / RT_MEMCPYASYNC_SPLIT_SIZE;
    if (grhStatus != GRAPH_SUCCESS) {
      RTS_REPORT_CALL_ERROR(
          "Get op[%s: %s] out[%zu] memory size failed, "
          "format=%s, dataType=%s, retCode=%#x.",
          nodeName.c_str(), nodeType.c_str(), i, TypeUtils::FormatToAscendString(format).GetString(),
          TypeUtils::DataTypeToAscendString(dataType).GetString(), grhStatus);
      return FAILED;
    }
    if (outputMemSize < 0) {
      RTS_REPORT_CALL_ERROR(
          "Got op[%s:%s] out[%zu] memory size is negative(not support), "
          "format=%s, dataType=%s, outputMemSize=%" PRId64,
          nodeName.c_str(), nodeType.c_str(), i, TypeUtils::FormatToAscendString(format).GetString(),
          TypeUtils::DataTypeToAscendString(dataType).GetString(), outputMemSize);
      return FAILED;
    }
    TensorUtils::SetSize(outputTensor, outputMemSize);
    RTS_LOGI("Calc op[%s: %s] out[%zu] mem size is %" PRId64 " format=%s, dataType=%s.", nodeName.c_str(),
             nodeType.c_str(), i, outputMemSize, TypeUtils::FormatToAscendString(format).GetString(),
             TypeUtils::DataTypeToAscendString(dataType).GetString());
    TensorUtils::SetSize(outputTensor, outputMemSize);
    grhStatus = opDesc->UpdateOutputDesc(i, outputTensor);
    if (grhStatus != GRAPH_SUCCESS) {
      RTS_REPORT_CALL_ERROR(
          "Update op[%s: %s] out[%zu] description failed, "
          "format=%s, dataType=%s, retCode=%#x.",
          nodeName.c_str(), nodeType.c_str(), i, TypeUtils::FormatToAscendString(format).GetString(),
          TypeUtils::DataTypeToAscendString(dataType).GetString(), grhStatus);
      return FAILED;
    }
  }

  if (nodeType == "MemcpyAsync") {
    size_t setSqeNumRes = AttrUtils::SetInt(opDesc, ATTR_NAME_NODE_SQE_NUM,
                                            (nodeSqeNum > RT_GENERAL_SQE_NUM) ? nodeSqeNum : RT_GENERAL_SQE_NUM);
    RTS_LOGD("setSqeNumRes is %zu, nodeSqeNum is %lld.", setSqeNumRes, nodeSqeNum);
  } else if (nodeType == "RecvMem") {
    size_t setSqeNumRes = AttrUtils::SetInt(opDesc, ATTR_NAME_NODE_SQE_NUM, MEM_WAIT_SQE_NUM);
    RTS_LOGD("setSqeNumRes is %zu, nodeSqeNum is %u.", setSqeNumRes, MEM_WAIT_SQE_NUM);
  }
  RTS_LOGD("Calc op[%s: %s] running param success.", nodeName.c_str(), nodeType.c_str());
  return SUCCESS;
}

void RtsOpsKernelBuilder::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const {
  infos = op_info_map_;
}

Status RtsOpsKernelBuilder::GenerateTask(const Node &geNode, RunContext &context, vector<TaskDef> &tasks) {
  bool unknownShape = geNode.GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (unknownShape) {
    RTS_LOGI("op:%s is in unknownShape graph, does not need to generate task.", geNode.GetName().c_str());
    return GRAPH_SUCCESS;
  }

  string name = geNode.GetName();
  string type = geNode.GetType();

  RTS_LOGI("Generate task start, node:%s, node type:%s, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());

  auto op = OpFactory::Instance().CreateOp(geNode, context);
  if (op == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op for node:%s, node type:%s failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  Status ret = op->Init();
  if (ret != SUCCESS) {
    RTS_REPORT_CALL_ERROR("Node:%s, node type:%s op init failed, retCode=%#x", name.c_str(), type.c_str(), ret);
    return ret;
  }

  bool hasSgtSliceInfo = geNode.GetOpDesc()->HasAttr("_ffts_plus");
  if (hasSgtSliceInfo) {
    RTS_LOGI("Generate ffts+ context def start.");
    ret = op->GenerateCtxDef(geNode);
    if (ret != SUCCESS) {
      RTS_LOGI("Generate ffts+ context def failed.");
      return ret;
    }
    return SUCCESS;
  }

  ret = op->Run(tasks);
  if (ret != SUCCESS) {
    RTS_REPORT_CALL_ERROR("Node:%s, node type:%s op run failed, retCode=%#x", name.c_str(), type.c_str(), ret);
    return ret;
  }
  RTS_LOGD("Generate task end, node:%s, node type:%s, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());
  return ret;
}

Status RtsOpsKernelBuilder::UpdateTask(const Node &geNode, vector<TaskDef> &tasks) {
  string name = geNode.GetName();
  string type = geNode.GetType();
  RunContext runCtx;  // not used
  auto op = OpFactory::Instance().CreateOp(geNode, runCtx);
  if (op == nullptr) {
    RTS_REPORT_CALL_ERROR("Update Create op for node:%s, node type:%s failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  return op->UpdateTaskDef(tasks);
}

bool RtsOpsKernelBuilder::CheckSupported(const OpDescPtr &opDesc, std::string &) const {
  return op_info_map_.count(opDesc->GetType()) > 0;
}

Status RtsOpsKernelBuilder::CreateSession(const map<string, string> &session_options) {
  (void)session_options;
  // do nothing
  return SUCCESS;
}

Status RtsOpsKernelBuilder::DestroySession(const map<string, string> &session_options) {
  (void)session_options;
  // do nothing
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce
