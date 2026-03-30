/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_build_graph.h"
#include "hcom_op_utils.h"
#include <map>
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include <unordered_map>

using namespace std;

namespace hccl {
const std::string HCOM_ATTR_ADDR_LENGTH = "addr_length";
std::map<const std::string, HcomOpType> HcomOpTypeTransform = {
    {HCCL_KERNEL_OP_TYPE_ALLGATHER, HcomOpType::HCOM_ALL_GATHER},
    {HCCL_KERNEL_OP_TYPE_ALLGATHERV, HcomOpType::HCOM_ALL_GATHER_V},
    {HCCL_KERNEL_OP_TYPE_ALLREDUCE, HcomOpType::HCOM_ALL_REDUCE},
    {HCCL_KERNEL_OP_TYPE_BROADCAST, HcomOpType::HCOM_BROADCAST},
    {HCCL_KERNEL_OP_TYPE_REDUCESCATTER, HcomOpType::HCOM_REDUCE_SCATTER},
    {HCCL_KERNEL_OP_TYPE_REDUCESCATTERV, HcomOpType::HCOM_REDUCE_SCATTER_V},
    {HCCL_KERNEL_OP_TYPE_ALLTOALLV, HcomOpType::HCOM_ALL_TO_ALL_V},
    {HCCL_KERNEL_OP_TYPE_ALLTOALLVC, HcomOpType::HCOM_ALL_TO_ALL_VC},
    {HCCL_KERNEL_OP_TYPE_ALLTOALL, HcomOpType::HCOM_ALL_TO_ALL},
    {HCCL_KERNEL_OP_TYPE_SEND, HcomOpType::HCOM_SEND},
    {HCCL_KERNEL_OP_TYPE_RECEIVE, HcomOpType::HCOM_RECEIVE},
    {HCCL_KERNEL_OP_TYPE_REDUCE, HcomOpType::HCOM_REDUCE},
};

/*
 * **********************************************************************
 * 读取node中的属性，输出到属性的结构体中
 * 返回各算子的非公共属性
 * **********************************************************************
 */
HcclResult HcomAllGatherGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  return HcomOpUtils::GetRankSize(node->GetOpDesc(), opAttr.op.allgather.rankSize);
}

HcclResult HcomAllGatherVGetOpAttr([[maybe_unused]] const ge::NodePtr &node, [[maybe_unused]] struct HcomOpAttr &opAttr) {
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  return HcomOpUtils::GetReduction(node->GetOpDesc(), opAttr.op.allreduce.reduction);
}

HcclResult HcomBroadcastGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  return HcomOpUtils::GetRoot(node->GetOpDesc(), opAttr.op.broadcast.root);
}

HcclResult HcomReduceScatterGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  CHK_RET(HcomOpUtils::GetRankSize(node->GetOpDesc(), opAttr.op.reducescatter.rankSize));
  return HcomOpUtils::GetReduction(node->GetOpDesc(), opAttr.op.reducescatter.reduction);
}

HcclResult HcomReduceScatterVGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  CHK_RET(HcomOpUtils::GetReduction(node->GetOpDesc(), opAttr.op.reducescatterv.reduction));
  return HCCL_SUCCESS;
}

HcclResult HcomAllToAllVGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  ge::GeTensorDesc outputTensor = node->GetOpDesc()->GetOutputDesc(0);
  ge::DataType geDataType = outputTensor.GetDataType();
  CHK_RET(HcomOpUtils::TransformDataType(geDataType, opAttr.op.alltoallv.recvType));
  return HCCL_SUCCESS;
}

HcclResult HcomAllToAllVCGetOpAttr([[maybe_unused]] const ge::NodePtr &node,
                                   [[maybe_unused]] struct HcomOpAttr &opAttr) {
  return HCCL_SUCCESS;
}

HcclResult HcomAllToAllGetOpAttr([[maybe_unused]] const ge::NodePtr &node, [[maybe_unused]] struct HcomOpAttr &opAttr) {
  // 暂时作为桩函数
  return HCCL_SUCCESS;
}

HcclResult HcomRecvGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  CHK_RET(HcomOpUtils::GetSrcRank(node->GetOpDesc(), opAttr.op.recv.srcRank));
  return HcomOpUtils::GetSrTag(node->GetOpDesc(), opAttr.op.recv.srTag);
}

HcclResult HcomSendGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  CHK_RET(HcomOpUtils::GetDestRank(node->GetOpDesc(), opAttr.op.send.destRank));
  return HcomOpUtils::GetSrTag(node->GetOpDesc(), opAttr.op.send.srTag);
}

HcclResult HcomReduceGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  CHK_RET(HcomOpUtils::GetReduction(node->GetOpDesc(), opAttr.op.reduce.reduction));
  CHK_RET(HcomOpUtils::GetRoot(node->GetOpDesc(), opAttr.op.reduce.root));
  return HCCL_SUCCESS;
}

std::vector<std::function<HcclResult(const ge::NodePtr &, struct HcomOpAttr &)>> HcomOpGetAttrFuncs = {
    HcomAllGatherGetOpAttr,     HcomAllGatherVGetOpAttr,     HcomAllReduceGetOpAttr, HcomBroadcastGetOpAttr,
    HcomReduceScatterGetOpAttr, HcomReduceScatterVGetOpAttr, HcomAllToAllVGetOpAttr, HcomAllToAllVCGetOpAttr,
    HcomAllToAllGetOpAttr,      HcomSendGetOpAttr,           HcomReduceGetOpAttr,    HcomRecvGetOpAttr};

/*
 * **********************************************************************
 * 读取node中的属性，输出到属性的结构体中
 * 属性包含：opType，dataType，group，算子的非公共属性
 * **********************************************************************
 */
HcclResult GenerateHcomOpArg(const ge::NodePtr &node, struct HcomOpAttr &opAttr) {
  auto iter = HcomOpTypeTransform.find(node->GetOpDesc()->GetType());
  if (iter == HcomOpTypeTransform.end()) {
    HCCL_ERROR("[Generate][HcomOpAttr] %s is not supported.", node->GetOpDesc()->GetType().c_str());
    return HCCL_E_PARA;
  }
  opAttr.opType = iter->second;
  auto opDesc = node->GetOpDesc();
  CHK_RET(HcomOpUtils::GetDataType(opDesc, opAttr.dataType));
  CHK_RET(HcomOpUtils::GetAivCoreLimit(opDesc, opDesc->GetType(), opAttr.aivCoreLimit));

  std::string sGroup;
  CHK_RET(HcomOpUtils::GetGroup(opDesc, sGroup));
  int32_t sret = memcpy_s(&opAttr.group[0], sizeof(opAttr.group), sGroup.c_str(), (sGroup.length() + 1));
  CHK_PRT_RET(sret != EOK,
              HCCL_ERROR("[Generate][HcomOpAttr]memcpy failed. ret[%d],"
                         "params:destMaxSize[%zu],count[%zu]",
                         sret, sizeof(opAttr.group), (sGroup.length() + 1)),
              HCCL_E_SYSCALL);

  CHK_RET(HcomOpGetAttrFuncs[static_cast<int32_t>(opAttr.opType)](node, opAttr));
  return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 执行图构图，集合通信算子node在lowering过程中，const args节点的构造
 * 包含算子的const options
 * **********************************************************************
 */
gert::bg::ValueHolderPtr GenerateHcomOpArgs(const ge::NodePtr &node) {
  struct HcomOpAttr opAttr;
  HcclResult ret = GenerateHcomOpArg(node, opAttr);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("Node[%s]: Generate ConstOpArgs failed.", node->GetName().c_str());
    return nullptr;
  }
  return gert::bg::ValueHolder::CreateConst(&opAttr, sizeof(struct HcomOpAttr), false);
}

/*
 * **********************************************************************
 * 执行图构图，集合通信算子node在lowering过程中，launch kernel节点的构造
 * 包含算子的inputs，outputs，options
 * **********************************************************************
 */
std::vector<gert::bg::ValueHolderPtr> LaunchHcomOpKernel(const HcomLaunchArg &args,
                                                         const std::vector<gert::bg::DevMemValueHolderPtr> &inputAddrs,
                                                         const std::vector<gert::bg::DevMemValueHolderPtr> &outputAddrs,
                                                         const std::vector<gert::bg::ValueHolderPtr> &inputShapes,
                                                         const std::vector<gert::bg::ValueHolderPtr> &outputShapes) {
  std::vector<gert::bg::ValueHolderPtr> inputs;
  uint32_t inputNum = inputAddrs.size();
  uint32_t outputNum = outputAddrs.size();
  inputs.emplace_back(args.opArgs);
  inputs.emplace_back(args.stream);
  inputs.emplace_back(gert::bg::ValueHolder::CreateConst(&inputNum, sizeof(inputNum)));
  inputs.emplace_back(gert::bg::ValueHolder::CreateConst(&outputNum, sizeof(outputNum)));
  inputs.insert(inputs.end(), inputShapes.begin(), inputShapes.end());
  inputs.insert(inputs.end(), outputShapes.begin(), outputShapes.end());
  auto prepareHcomKernel = gert::bg::ValueHolder::CreateDataOutput("PrepareHcomKernel", inputs, 1);
  if (prepareHcomKernel.size() == 0) {
    HCCL_ERROR("[LaunchHcomOpKernel] prepareHcomKernel size is 0, please check input param");
    return prepareHcomKernel;
  }
  std::vector<gert::bg::ValueHolderPtr> launchInputs;
  launchInputs.emplace_back(prepareHcomKernel[0]);
  launchInputs.insert(launchInputs.end(), inputAddrs.begin(), inputAddrs.end());
  launchInputs.insert(launchInputs.end(), outputAddrs.begin(), outputAddrs.end());
  return gert::bg::ValueHolder::CreateDataOutput("LaunchHcomKernel", launchInputs, outputAddrs.size());
}

std::vector<gert::bg::DevMemValueHolderPtr> LaunchRecvOpKernel(const HcomLaunchArg &args, const ge::NodePtr &node,
                                                               const gert::LowerInput &lowerInput) {
  std::vector<gert::bg::ValueHolderPtr> inputs;

  ge::DataType geDataType = node->GetOpDesc()->GetOutputDesc(0U).GetDataType();

  inputs.emplace_back(args.opArgs);
  inputs.emplace_back(args.stream);
  inputs.emplace_back(
      lowerInput.global_data->GetOrCreateAllocator({gert::kOnDeviceHbm, gert::AllocatorUsage::kAllocNodeOutput}));
  inputs.emplace_back(gert::bg::ValueHolder::CreateConst(&geDataType, sizeof(geDataType)));
  return gert::bg::DevMemValueHolder::CreateDataOutput("LaunchRecvKernel", inputs, LAUNCH_RECV_KERNEL_OUTPUT_SIZE,
                                                       node->GetOpDesc()->GetStreamId());
}

}  // namespace hccl