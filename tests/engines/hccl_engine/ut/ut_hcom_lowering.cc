/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include <mockcpp/mockcpp.hpp>

#include "hccl_stub.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <runtime/rt.h>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>
#include "hcom_launch_kernel.h"
#include "common/ge_common/ge_types.h"
#include "hcom_build_graph.h"
#include "hcom_node_converter.h"
#include "hcom_acl_adapter.h"
#include "exe_graph/lowering/lowering_global_data.h"

#include "exe_graph/runtime/kernel_context.h"

#include "exe_graph/runtime/kernel_run_context.h"

#include "llt_hccl_stub_kernel_run_ctx_faker.h"

#include "v80_rank_table.h"

#include "llt_hccl_stub_ge.h"
#include "adapter_dlhcclfunc.h"

using namespace std;
using namespace hccl;

namespace hccl {
extern HcclResult GetCountByShape(const gert::Shape &shape, HcclDataType dataType, uint64_t &count);
extern HcclResult HcomAllToAllGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr);
}  // namespace hccl

class HcomLoweringTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HcomLoweringTest SetUP" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "HcomLoweringTest TearDown" << std::endl;
  }
  virtual void SetUp() {
    std::cout << "A Test SetUP" << std::endl;
  }
  virtual void TearDown() {
    std::cout << "A Test TearDown" << std::endl;
  }
};

class NodeTest : public ge::Node {
 public:
  NodeTest() {
    ;
  };
  ~NodeTest() {
    ;
  };
};

TEST_F(HcomLoweringTest, ut_LaunchHcomOpKernel) {
  gert::bg::ValueHolderPtr valuePtr = nullptr;
  gert::bg::DevMemValueHolderPtr memValuePtr = nullptr;

  HcomLaunchArg launchArg;
  launchArg.opArgs = valuePtr;
  launchArg.stream = valuePtr;

  std::vector<gert::bg::DevMemValueHolderPtr> inputAddrs;
  inputAddrs.push_back(memValuePtr);
  std::vector<gert::bg::DevMemValueHolderPtr> outputAddrs;
  outputAddrs.push_back(memValuePtr);
  std::vector<gert::bg::ValueHolderPtr> inputShapes;
  inputShapes.push_back(valuePtr);
  std::vector<gert::bg::ValueHolderPtr> outputShapes;
  outputShapes.push_back(valuePtr);

  LaunchHcomOpKernel(launchArg, inputAddrs, outputAddrs, inputShapes, outputShapes);
}

TEST_F(HcomLoweringTest, ut_node_emptyNode) {
  ge::NodePtr node;
  gert::bg::ValueHolderPtr valuePtr = nullptr;
  gert::bg::DevMemValueHolderPtr memValuePtr = nullptr;
  gert::LowerInput lowerInput;
  lowerInput.input_addrs.push_back(memValuePtr);
  lowerInput.input_shapes.push_back(valuePtr);
  lowerInput.global_data = nullptr;
  LoweringHcomNode(node, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_broadcast) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_BROADCAST);
  ge::AttrUtils::SetInt(op, "root_rank", 0);
  auto bcastNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr memValuePtr = nullptr;
  lowerInput.input_addrs.push_back(memValuePtr);
  lowerInput.input_shapes.push_back(valuePtr);
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringHcomNode(bcastNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_reduce) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_REDUCE);
  ge::AttrUtils::SetInt(op, "root_rank", 0);
  ge::AttrUtils::SetStr(op, "reduction", "sum");
  auto reduceNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr memValuePtr = nullptr;
  lowerInput.input_addrs.push_back(memValuePtr);
  lowerInput.input_shapes.push_back(valuePtr);
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringHcomNode(reduceNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_send) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_SEND);
  ge::AttrUtils::SetInt(op, "dest_rank", 1);
  ge::AttrUtils::SetInt(op, "sr_tag", 111);
  auto sendNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr AddrsPtr;
  lowerInput.input_addrs.push_back(AddrsPtr);
  lowerInput.input_shapes.push_back(valuePtr);
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringHcomNode(sendNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_receive) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_RECEIVE);
  ge::AttrUtils::SetInt(op, "src_rank", 0);
  ge::AttrUtils::SetInt(op, "sr_tag", 111);
  auto receiveNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr AddrsPtr;
  lowerInput.input_addrs.push_back(AddrsPtr);
  lowerInput.input_shapes.push_back(valuePtr);
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringRecvNode(receiveNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_recv_emptyNode) {
  ge::NodePtr node;
  gert::bg::ValueHolderPtr valuePtr = nullptr;
  gert::bg::DevMemValueHolderPtr memValuePtr = nullptr;
  gert::LowerInput lowerInput;
  lowerInput.input_addrs.push_back(memValuePtr);
  lowerInput.input_shapes.push_back(valuePtr);
  lowerInput.global_data = nullptr;
  LoweringRecvNode(node, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_allreduce) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_ALLREDUCE);
  ge::AttrUtils::SetStr(op, "reduction", "sum");
  auto allreduceNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::DevMemValueHolderPtr memValuePtr;
  gert::bg::ValueHolderPtr valuePtr;
  lowerInput.input_addrs.push_back(memValuePtr);
  lowerInput.input_shapes.push_back(valuePtr);
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringHcomNode(allreduceNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_reducescatterv) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_REDUCESCATTERV);
  ge::AttrUtils::SetStr(op, "reduction", "sum");
  auto reducescattervNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr memValuePtr;
  for (int i = 0; i < 5; i++) {
    lowerInput.input_addrs.push_back(memValuePtr);
    lowerInput.input_shapes.push_back(valuePtr);
  }
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringAlltoAllNode(reducescattervNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_alltoallv) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_ALLTOALLV);
  auto alltoallvNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr memValuePtr;
  for (int i = 0; i < 5; i++) {
    lowerInput.input_addrs.push_back(memValuePtr);
    lowerInput.input_shapes.push_back(valuePtr);
  }
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringAlltoAllNode(alltoallvNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_allgatherv) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_ALLGATHERV);
  auto allgathervNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr memValuePtr;
  for (int i = 0; i < 4; i++) {
    lowerInput.input_addrs.push_back(memValuePtr);
    lowerInput.input_shapes.push_back(valuePtr);
  }
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringAlltoAllNode(allgathervNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_node_alltoallvc) {
  ge::ComputeGraph graph("test_graph");
  ge::OpDescPtr op;
  std::string tempStr = "test_group";
  ge::AttrUtils::SetStr(op, "group", tempStr);
  HcclDataType dataType = HCCL_DATA_TYPE_INT8;
  ge::AttrUtils::SetInt(op, "dataType", dataType);
  op->SetType(HCCL_KERNEL_OP_TYPE_ALLTOALLVC);
  auto alltoallvcNode = graph.AddNode(op);

  gert::LowerInput lowerInput;
  gert::bg::ValueHolderPtr valuePtr;
  gert::bg::DevMemValueHolderPtr memValuePtr;
  for (int i = 0; i < 2; i++) {
    lowerInput.input_addrs.push_back(memValuePtr);
    lowerInput.input_shapes.push_back(valuePtr);
  }
  gert::LoweringGlobalData global_data;
  lowerInput.global_data = &global_data;
  LoweringAlltoAllNode(alltoallvcNode, lowerInput);
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_allgatherv_count0) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./st_hcomLaunchKernel_allGatherv.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpInputStruct inputStruct;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_GATHER_V;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 4;
  launchArgs.outputNum = 1;

  launchArgs.inputAddrs.resize(4);
  launchArgs.inputAddrs[0] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[0], 1024, 0, 1024);
  launchArgs.inputAddrs[1] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[1], 1024, 0, 1024);
  launchArgs.inputAddrs[2] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[2], 1024, 0, 1024);
  launchArgs.inputAddrs[3] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[3], 1024, 0, 1024);

  gert::Shape inferShape({0});
  vector<gert::Shape> inShape;
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  launchArgs.inputShapes = inShape;

  launchArgs.outputAddrs.resize(1);
  launchArgs.outputAddrs[0] = (char *)malloc(1024);
  memset_s(launchArgs.outputAddrs[0], 1024, 0, 1024);

  gert::Shape outshape({0});
  launchArgs.outputShapes.push_back(outshape);

  MOCKER(HcceAllGatherV).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
  HcomAllGatherVKernel(launchArgs, &inputStruct);

  free(launchArgs.inputAddrs[0]);
  free(launchArgs.inputAddrs[1]);
  free(launchArgs.inputAddrs[2]);
  free(launchArgs.inputAddrs[3]);
  free(launchArgs.outputAddrs[0]);
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_broadcast) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_broadcast.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_BROADCAST;
  strcpy(opAttr.group, "hccl_world_group");
  opAttr.op.broadcast.root = 0;
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceBroadcast).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *bcastContext = new gert::KernelContext();
  LaunchHcomKernel(bcastContext);
  delete bcastContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_send) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_send.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_SEND;
  strcpy(opAttr.group, "hccl_world_group");
  opAttr.op.send.destRank = 1;
  opAttr.op.send.srTag = 111;
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(hcclStreamSynchronize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *sendContext = new gert::KernelContext();
  LaunchHcomKernel(sendContext);
  delete sendContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

HcclResult hrtMemSyncCopy_stub(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind) {
  s64 *recvShape = static_cast<s64 *>(dst);
  recvShape[0] = 1;
  recvShape[1] = 1;
  return HCCL_SUCCESS;
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_reduce) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_reduce.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_REDUCE;
  strcpy(opAttr.group, "hccl_world_group");
  opAttr.op.reduce.root = 0;
  opAttr.op.reduce.reduction = HCCL_REDUCE_SUM;
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceReduce).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *reduceContext = new gert::KernelContext();
  LaunchHcomKernel(reduceContext);
  delete reduceContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_reducescatter) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_reducescatter.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  gert::KernelContext *allGatherContext = new gert::KernelContext();
  KernelRunContext context;
  AsyncAnyValue any_value;

  HcomOpLaunchArgs launchArgs0;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_REDUCE_SCATTER;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs0.opAttr = opAttr;
  any_value.data.pointer = &launchArgs0;

  *context.values = &any_value;
  context.input_size = 1;
  *(allGatherContext->GetContext()) = context;

  MOCKER(HcomLaunchReduceScatterKernel).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
  LaunchHcomKernel(allGatherContext);

  delete allGatherContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_reducescatterv_count0) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./st_hcomLaunchKernel_reducescatterv.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpInputStruct inputStruct;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_REDUCE_SCATTER_V;
  opAttr.op.reducescatterv.reduction = HCCL_REDUCE_SUM;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 4;
  launchArgs.outputNum = 1;

  launchArgs.inputAddrs.resize(4);
  launchArgs.inputAddrs[0] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[0], 1024, 0, 1024);
  launchArgs.inputAddrs[1] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[1], 1024, 0, 1024);
  launchArgs.inputAddrs[2] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[2], 1024, 0, 1024);
  launchArgs.inputAddrs[3] = (char *)malloc(1024);
  memset_s(launchArgs.inputAddrs[3], 1024, 0, 1024);

  gert::Shape inferShape({0});
  vector<gert::Shape> inShape;
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  launchArgs.inputShapes = inShape;

  launchArgs.outputAddrs.resize(1);
  launchArgs.outputAddrs[0] = (char *)malloc(1024);
  memset_s(launchArgs.outputAddrs[0], 1024, 0, 1024);

  gert::Shape outshape({0});
  launchArgs.outputShapes.push_back(outshape);

  MOCKER(HcceReduceScatterV).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
  HcomReduceScatterVKernel(launchArgs, &inputStruct);

  free(launchArgs.inputAddrs[0]);
  free(launchArgs.inputAddrs[1]);
  free(launchArgs.inputAddrs[2]);
  free(launchArgs.inputAddrs[3]);
  free(launchArgs.outputAddrs[0]);
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_alltoallv) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_alltoallv.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_TO_ALL_V;
  strcpy(opAttr.group, "hccl_world_group");
  opAttr.op.broadcast.root = 0;
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> inShape;
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);
  launchArgs.inputShapes = inShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  inputAddrs.push_back(pInputAddr);
  inputAddrs.push_back(pInputAddr);
  inputAddrs.push_back(pInputAddr);
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceAlltoAllV).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *a2avContext = new gert::KernelContext();
  LaunchHcomKernel(a2avContext);
  delete a2avContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_alltoallvc) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_alltoallvc.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_TO_ALL_VC;
  strcpy(opAttr.group, "hccl_world_group");
  opAttr.op.broadcast.root = 0;
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> inShape;
  inShape.push_back(inferShape);
  inShape.push_back(inferShape);

  launchArgs.inputShapes = inShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  inputAddrs.push_back(pInputAddr);

  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceAlltoAllVC).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *a2avcContext = new gert::KernelContext();
  LaunchHcomKernel(a2avcContext);
  delete a2avcContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_allgather_new) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_allgather_new.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_GATHER;
  strcpy(opAttr.group, "hccl_world_group");
  opAttr.op.broadcast.root = 0;
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceAllGather).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *gatherContext = new gert::KernelContext();
  LaunchHcomKernel(gatherContext);
  delete gatherContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

ge::graphStatus GetOption2(ge::GEThreadLocalContext *that, const std::string &optionExec, std::string &dumpDebugValue) {
  nlohmann::json rankTable = {{"collective_id", "192.168.3.3-9527-0001"},
                              {"master_ip", "172.16.16.82"},
                              {"master_port", "18000"},
                              {"status", "completed"},
                              {"version", "1.1"},
                              {"node_list",
                               {{{"node_addr", "172.16.16.82"}, {"ranks", {{{"rank_id", "0"}}}}},
                                {{"node_addr", "172.16.16.39"}, {"ranks", {{{"rank_id", "1"}}}}}}}};
  static std::string rankTableString = rankTable.dump();
  if (optionExec == ge::OPTION_EXEC_RANK_TABLE_ADDR) {
    dumpDebugValue = std::to_string(ge::PtrToValue(rankTableString.data()));
    return ge::GRAPH_SUCCESS;
  } else if (optionExec == ge::OPTION_EXEC_RANK_TABLE_LEN) {
    dumpDebugValue = std::to_string(rankTableString.length());
    return ge::GRAPH_SUCCESS;
  } else if (optionExec == ge::OPTION_EXEC_RANK_ID) {
    dumpDebugValue = "0";
    return ge::GRAPH_SUCCESS;
  } else if (optionExec == ge::OPTION_EXEC_ROLE_TABLE_LEN) {
    dumpDebugValue = "0";
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_SUCCESS;
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_errortest1) {
  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(GetOption2));
  gert::KernelContext *context = new gert::KernelContext();
  LaunchHcomKernel(context);
  delete context;
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_allreduce_sess) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_allreduce_sess.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_REDUCE;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceAllReduce).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *gatherContext = new gert::KernelContext();
  LaunchHcomKernel(gatherContext);

  LaunchHcomKernel(gatherContext);
  delete gatherContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_allreduce_large_sess) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_allreduce_large_sess.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_REDUCE;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 1;
  launchArgs.outputNum = 1;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  uint64_t count = 300 * 1024 * 1024;
  MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceAllReduce).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *gatherContext = new gert::KernelContext();
  LaunchHcomKernel(gatherContext);
  delete gatherContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_allreduce_multi_input_sess) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_allreduce_multi_input_sess.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  HcomOpLaunchArgs launchArgs;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_REDUCE;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs.opAttr = opAttr;

  std::string stream = "stream";
  string *pStream = &stream;
  launchArgs.stream = pStream;

  launchArgs.inputNum = 2;
  launchArgs.outputNum = 2;

  gert::Shape inferShape;
  vector<gert::Shape> shape;
  shape.push_back(inferShape);
  launchArgs.inputShapes = shape;
  launchArgs.outputShapes = shape;

  std::string inputAddr = "inputAddr";
  string *pInputAddr = &inputAddr;
  vector<void *> inputAddrs;
  inputAddrs.push_back(pInputAddr);
  inputAddrs.push_back(pInputAddr);
  launchArgs.inputAddrs = inputAddrs;

  std::string outputAddr = "outputAddr";
  string *pOutputAddr = &outputAddr;
  vector<void *> outputAddrs;
  outputAddrs.push_back(pOutputAddr);
  outputAddrs.push_back(pOutputAddr);
  launchArgs.outputAddrs = outputAddrs;

  MOCKER(GetHcomOpLaunchArgs).stubs().with(mockcpp::any(), outBound(launchArgs)).will(returnValue(HCCL_SUCCESS));

  uint64_t count = 1024;
  MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcceAllReduce).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  gert::KernelContext *gatherContext = new gert::KernelContext();
  LaunchHcomKernel(gatherContext);
  delete gatherContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_makeSureInput) {
  ge::NodePtr node;
  gert::LowerInput lower_input;

  std::vector<gert::bg::ValueHolderPtr> input_shapes;
  gert::bg::ValueHolderPtr inputShape;
  input_shapes.push_back(inputShape);
  input_shapes.push_back(inputShape);
  input_shapes.push_back(inputShape);
  input_shapes.push_back(inputShape);
  input_shapes.push_back(inputShape);
  std::vector<gert::bg::DevMemValueHolderPtr> input_addrs;
  gert::bg::DevMemValueHolderPtr inputAddrs;
  input_addrs.push_back(inputAddrs);
  input_addrs.push_back(inputAddrs);
  input_addrs.push_back(inputAddrs);
  input_addrs.push_back(inputAddrs);
  input_addrs.push_back(inputAddrs);
  lower_input.input_addrs = input_addrs;
  lower_input.input_shapes = input_shapes;
  gert::LoweringGlobalData global_data;
  lower_input.global_data = &global_data;

  MakeSureCommAlltoAllInput(node, lower_input);
}

TEST_F(HcomLoweringTest, ut_launchHcomKernelInitComm_test) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_launchHcomKernelInitComm_test.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  gert::KernelContext *allGatherContext = new gert::KernelContext();
  KernelRunContext context;
  AsyncAnyValue any_value;

  HcomOpLaunchArgs launchArgs0;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_GATHER;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs0.opAttr = opAttr;
  any_value.data.pointer = &launchArgs0;

  *context.values = &any_value;
  context.input_size = 1;
  *(allGatherContext->GetContext()) = context;

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(GetOption2));

  LaunchHcomKernelInitComm(allGatherContext);
  delete allGatherContext;
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, ut_HcomAllToAllGetOpAttr_test) {
  ge::NodePtr node;
  HcomOpAttr opAttr;
  EXPECT_EQ(HcomAllToAllGetOpAttr(node, opAttr), HCCL_SUCCESS);
}

TEST_F(HcomLoweringTest, ut_hcomLaunchKernel_allGatherv2_When_Normal_Expect_ReturnlsHCCL_SUCCESS) {
  MOCKER(HcceAllGather).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
#ifdef MACRO_DEV_TYPE_NEW
  MOCKER(HcomGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(DevType::DEV_TYPE_950));
#else
  MOCKER(HcomGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(DevType::DEV_TYPE_910_95));
#endif
  MOCKER(HcomAllGatherKernel).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_hcomLaunchKernel_allGather.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  u32 ret1 = hrtSetDevice(0);
  EXPECT_EQ(ret1, HCCL_SUCCESS);

  gert::KernelContext *allGatherContext = new gert::KernelContext();
  KernelRunContext context;
  AsyncAnyValue any_value;

  HcomOpLaunchArgs launchArgs0;
  HcomOpAttr opAttr;
  opAttr.dataType = HCCL_DATA_TYPE_INT8;
  opAttr.opType = HcomOpType::HCOM_ALL_GATHER;
  strcpy(opAttr.group, "hccl_world_group");
  launchArgs0.opAttr = opAttr;
  any_value.data.pointer = &launchArgs0;

  *context.values = &any_value;
  context.input_size = 1;
  *(allGatherContext->GetContext()) = context;

  MOCKER(HcomLaunchAllGatherKernelV2).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  LaunchHcomKernel(allGatherContext);
  delete allGatherContext;
  HcomDestroy();
  remove(file_name_t);
  GlobalMockObject::verify();
}

TEST_F(HcomLoweringTest, Ut_HcomReduceScatterKernelV2) {
  MOCKER(GetCountByShape).stubs().with().will(returnValue(HCCL_SUCCESS));
  HcomOpLaunchArgs launchArgs;
  HcomOpInputStruct inputStruct;
  launchArgs.inputNum = 1;
  launchArgs.inputAddrs = {reinterpret_cast<void *>(0x1000)};
  launchArgs.outputAddrs = {reinterpret_cast<void *>(0x2000)};
  launchArgs.opAttr.dataType = HCCL_DATA_TYPE_INT32;
  launchArgs.opAttr.op.reducescatter.reduction = HCCL_REDUCE_SUM;
  launchArgs.stream = nullptr;
  HcclResult result = HcomReduceScatterKernelV2(launchArgs, &inputStruct);
  EXPECT_EQ(result, HCCL_SUCCESS);

  result = HcomAllToAllVKernelV2(launchArgs, &inputStruct);
  EXPECT_EQ(result, HCCL_SUCCESS);

  result = HcomAllToAllVCKernelV2(launchArgs, &inputStruct);
  EXPECT_EQ(result, HCCL_SUCCESS);

  result = HcomReduceKernelV2(launchArgs, &inputStruct);
  EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomLoweringTest, Ut_HcomCopyInputsToCCLbuff) {
  MOCKER(hrtMemAsyncCopy).stubs().with().will(returnValue(HCCL_SUCCESS));
  HcomOpLaunchArgs launchArgs;
  launchArgs.opAttr.dataType = HCCL_DATA_TYPE_INT32;  // 假设数据类型为int32
  launchArgs.inputAddrs = {reinterpret_cast<void *>(0x1000), reinterpret_cast<void *>(0x2000)};
  launchArgs.outputAddrs = {reinterpret_cast<void *>(0x2000), reinterpret_cast<void *>(0x3000)};
  launchArgs.stream = nullptr;

  uint32_t inputsNum = 2;
  uint32_t inputsOffset = 0;
  std::vector<uint64_t> inputsCount = {10, 20};
  uint64_t cclBuffSize = 2;  // 足够大的缓冲区
  uint64_t commCount = 0;
  std::vector<void *> inputAddrs(3, nullptr);

  void *cclBuff = malloc(cclBuffSize);
  HcclResult result = HcomCopyInputsToCCLbuff(launchArgs, inputsNum, inputsOffset, inputsCount, cclBuff, cclBuffSize,
                                              commCount, inputAddrs);
  EXPECT_EQ(result, HCCL_SUCCESS);
  result = HcomCopyCCLbuffToOutnputs(launchArgs, inputsNum, inputsOffset, inputsCount, cclBuff, inputAddrs);
  EXPECT_EQ(result, HCCL_SUCCESS);

  free(cclBuff);
}