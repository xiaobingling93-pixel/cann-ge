/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/node_executor/task_context.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/executor/node_state.h"
#include "common/dump/exception_dumper.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/global_variables/diagnose_switch.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "common/debug/ge_log.h"
#include "common/sgt_slice_type.h"
#include "common/dump/dump_utils.h"
#include "depends/profiler/src/dump_stub.h"
#include "common/dump/dump_manager.h"
#include "common/dump/adump_opinfo_builder.h"

namespace ge {
class UTEST_dump_exception : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {
    remove("valid_path");
  }
};
namespace {
bool CheckLogExpected(std::vector<gert::OneLog> &logs, const std::string &expect_log) {
  for (auto &onelog : logs) {
    std::string content = onelog.content;
    if (content.find(expect_log) != std::string::npos) {
      return true;
    }
  }
  return false;
}
}  // namespace
TEST_F(UTEST_dump_exception, SaveProfilingTaskDescInfo) {
  hybrid::GraphItem graph_item;
  hybrid::GraphExecutionContext exec_ctx;
  hybrid::SubgraphContext subgraph_context(&graph_item, &exec_ctx);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  NodePtr node = std::make_shared<Node>(op_desc, graph);
  hybrid::NodeItem node_item = hybrid::NodeItem(node);
  node_item.input_start = 0;
  node_item.output_start = 0;
  hybrid::FrameState frame_state = hybrid::FrameState();
  hybrid::NodeState node_state(node_item, &subgraph_context, frame_state);
  auto task_context = hybrid::TaskContext::Create(&node_state, &subgraph_context);
  task_context->task_id_ = 1;
  EXPECT_EQ(task_context->SaveProfilingTaskDescInfo("test", 1, "test"), SUCCESS);
}

TEST_F(UTEST_dump_exception, SaveProfilingTaskDescInfo_fail) {
  hybrid::GraphItem graph_item;
  hybrid::GraphExecutionContext exec_ctx;
  hybrid::SubgraphContext subgraph_context(&graph_item, &exec_ctx);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  NodePtr node = std::make_shared<Node>(op_desc, graph);
  hybrid::NodeItem node_item = hybrid::NodeItem(node);
  node_item.input_start = 0;
  node_item.output_start = 0;
  hybrid::FrameState frame_state = hybrid::FrameState();
  hybrid::NodeState node_state(node_item, &subgraph_context, frame_state);
  auto task_context = hybrid::TaskContext::Create(&node_state, &subgraph_context);
  DumpManager::GetInstance().Init({{"ge.exec.enable_exception_dump", "1"}});
  task_context->task_id_ = 999;
  EXPECT_EQ(task_context->SaveProfilingTaskDescInfo("test", 1, "test"), -1);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_exception, save_dump_op_info_success) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  uint32_t task_id = 1;
  uint32_t stream_id = 233;
  ExtraOpInfo extra_op_info;
  ExceptionDumper exception_dumper;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  GeTensorDesc invalid_tensor(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  TensorUtils::SetSize(invalid_tensor, 512);
  op_desc->AddInputDesc(invalid_tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(invalid_tensor);
  op_desc->AddOutputDesc(tensor);
  ge::OpDescInfoId id(task_id, stream_id, 0);
  EXPECT_NO_THROW(exception_dumper.SaveDumpOpInfo(op_desc, extra_op_info, id, false));
}

TEST_F(UTEST_dump_exception, save_dump_op_info_success_with_ctx) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  OpDescInfoId id(1, 233, 1, 1);
  ExtraOpInfo extra_op_info;
  ExceptionDumper exception_dumper;
  EXPECT_NO_THROW(exception_dumper.SaveDumpOpInfo(op_desc, extra_op_info, id, false));
}

TEST_F(UTEST_dump_exception, SaveDumpInfo_Ok_CountReachesUpperBound) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  ExtraOpInfo extra_op_info;
  ExceptionDumper exception_dumper;
  exception_dumper.op_desc_info_.resize(2048UL * 2048UL);
  EXPECT_EQ(exception_dumper.op_desc_info_.size(), 2048UL * 2048UL);
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  uint32_t task_id_1 = 3;
  uint32_t stream_id_1 = 999;
  ge::OpDescInfoId id(task_id_1, stream_id_1, 0);
  exception_dumper.SaveDumpOpInfo(op_desc_1, extra_op_info, id, false);
  auto actual_task_id = exception_dumper.op_desc_info_[1].id.task_id;
  auto actual_stream_id = exception_dumper.op_desc_info_[1].id.stream_id;
  EXPECT_EQ(actual_stream_id, stream_id_1);
  EXPECT_EQ(actual_task_id, task_id_1);
  EXPECT_EQ(exception_dumper.op_desc_info_.size(), 2048UL * 2048UL);
}

TEST_F(UTEST_dump_exception, LogArgs_WithKeyWord_ValidArgs) {
  OpDescInfo op_desc_info;
  auto args_holder = std::unique_ptr<uint8_t[]>(new uint8_t[32]);
  op_desc_info.args = reinterpret_cast<uintptr_t>(args_holder.get());
  op_desc_info.args_size = 32;
  op_desc_info.is_host_args = false;
  ExceptionDumper exception_dumper{};
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  exception_dumper.LogExceptionArgs(op_desc_info);
  EXPECT_TRUE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[AIC_INFO] args after execute:"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UTEST_dump_exception, LogTilingData_WithKeyWord) {
  OpDescInfo op_desc_info;
  op_desc_info.imply_type = static_cast<uint32_t>(domi::ImplyType::TVM);
  op_desc_info.tiling_data = "testtesttesttesttesttesttesttest";
  ExceptionDumper exception_dumper{};
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  exception_dumper.LogExceptionTvmOpInfo(op_desc_info);
  EXPECT_TRUE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[AIC_INFO] tiling_data:"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UTEST_dump_exception, dump_no_input_with_output_set) {
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("output");

  std::vector<uint8_t> input_stub(8);
  std::vector<uint8_t> output_stub(8);
  std::vector<void *> mock_arg{input_stub.data(), output_stub.data()};

  OpDescInfo op_desc_info;
  op_desc_info.op_name = "Save";
  op_desc_info.op_type = "Save";
  op_desc_info.id.task_id = 1;
  op_desc_info.id.stream_id = 2;
  op_desc_info.args = reinterpret_cast<uintptr_t>(mock_arg.data());
  op_desc_info.imply_type = static_cast<uint32_t>(domi::ImplyType::TVM);
  op_desc_info.input_format = {FORMAT_NCHW};
  op_desc_info.input_shape = {{1}};
  op_desc_info.input_data_type = {DT_FLOAT};
  op_desc_info.input_addrs = {nullptr};
  op_desc_info.input_size = {2};
  op_desc_info.output_format = {FORMAT_NCHW};
  op_desc_info.output_shape = {{1}};
  op_desc_info.output_data_type = {DT_FLOAT};
  op_desc_info.output_addrs = {nullptr};
  op_desc_info.output_size = {2};
  ExceptionDumper exception_dumper;
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);

  // case1: dump mode: output, input will not dump
  exception_dumper.DumpExceptionInput(op_desc_info, "/", false, dump_properties);
  EXPECT_FALSE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[Dump][Input] op"));
  runtime_stub.GetSlogStub().Clear();

  // case2: dump mode: output, workspace will not dump
  exception_dumper.DumpExceptionWorkspace(op_desc_info, "/", false, dump_properties);
  EXPECT_FALSE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[Dump][Workspace] op"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UTEST_dump_exception, dump_no_output_with_input_set) {
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("input");

  std::vector<uint8_t> input_stub(8);
  std::vector<uint8_t> output_stub(8);
  std::vector<void *> mock_arg{input_stub.data(), output_stub.data()};

  OpDescInfo op_desc_info;
  op_desc_info.op_name = "Save";
  op_desc_info.op_type = "Save";
  op_desc_info.id.task_id = 1;
  op_desc_info.id.stream_id = 2;
  op_desc_info.args = reinterpret_cast<uintptr_t>(mock_arg.data());
  op_desc_info.imply_type = static_cast<uint32_t>(domi::ImplyType::TVM);
  op_desc_info.input_format = {FORMAT_NCHW};
  op_desc_info.input_shape = {{1}};
  op_desc_info.input_data_type = {DT_FLOAT};
  op_desc_info.input_addrs = {nullptr};
  op_desc_info.input_size = {2};
  op_desc_info.output_format = {FORMAT_NCHW};
  op_desc_info.output_shape = {{1}};
  op_desc_info.output_data_type = {DT_FLOAT};
  op_desc_info.output_addrs = {nullptr};
  op_desc_info.output_size = {2};
  ExceptionDumper exception_dumper;
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);

  // case1: dump mode: input, Output will not dump
  exception_dumper.DumpExceptionOutput(op_desc_info, "/", false, dump_properties);
  EXPECT_FALSE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[Dump][Output] op"));
  runtime_stub.GetSlogStub().Clear();

  // case1: dump mode: input, workspace will not dump
  exception_dumper.DumpExceptionWorkspace(op_desc_info, "/", false, dump_properties);
  EXPECT_FALSE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[Dump][Workspace] op"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UTEST_dump_exception, host_dump_all) {
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");

  std::vector<uint8_t> input_stub(8);
  std::vector<uint8_t> output_stub(8);
  std::vector<void *> mock_arg{input_stub.data(), output_stub.data()};

  OpDescInfo op_desc_info;
  op_desc_info.op_name = "Save";
  op_desc_info.op_type = "Save";
  op_desc_info.id.task_id = 1;
  op_desc_info.id.stream_id = 2;
  op_desc_info.args = reinterpret_cast<uintptr_t>(mock_arg.data());
  op_desc_info.imply_type = static_cast<uint32_t>(domi::ImplyType::TVM);
  op_desc_info.input_format = {FORMAT_NCHW};
  op_desc_info.input_shape = {{1}};
  op_desc_info.input_data_type = {DT_FLOAT};
  op_desc_info.input_addrs = {reinterpret_cast<void *>(5000)};
  op_desc_info.input_size = {2};
  op_desc_info.output_format = {FORMAT_NCHW};
  op_desc_info.output_shape = {{1}};
  op_desc_info.output_data_type = {DT_FLOAT};
  op_desc_info.output_addrs = {reinterpret_cast<void *>(6000)};
  op_desc_info.output_size = {2};
  ExceptionDumper exception_dumper;
  gert::GertRuntimeStub runtime_stub;
  std::string dump_path = "./dump/";
  ASSERT_TRUE(ge::CreateDirectory(dump_path) == 0);
  std::string dump_file = dump_path + "dumpfile";
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  exception_dumper.DumpExceptionInput(op_desc_info, dump_file, false, dump_properties);
  exception_dumper.DumpExceptionOutput(op_desc_info, dump_file, false, dump_properties);
  exception_dumper.DumpExceptionWorkspace(op_desc_info, dump_file, false, dump_properties);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  ASSERT_TRUE(mmRmdir(dump_path.c_str()) == 0);
}

TEST_F(UTEST_dump_exception, dump_node_info_wrong) {
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");

  std::vector<uint8_t> input_stub(8);
  std::vector<uint8_t> output_stub(8);
  std::vector<void *> mock_arg{input_stub.data(), output_stub.data()};

  OpDescInfo op_desc_info;
  op_desc_info.op_name = "Save";
  op_desc_info.op_type = "Save";
  op_desc_info.id.task_id = 1;
  op_desc_info.id.stream_id = 2;
  op_desc_info.args = reinterpret_cast<uintptr_t>(mock_arg.data());
  op_desc_info.imply_type = static_cast<uint32_t>(domi::ImplyType::TVM);
  op_desc_info.input_format = {FORMAT_NCHW};
  op_desc_info.input_shape = {{1}};
  op_desc_info.input_data_type = {DT_FLOAT};
  op_desc_info.input_addrs = {nullptr};
  op_desc_info.input_size = {2};
  op_desc_info.output_format = {FORMAT_NCHW};
  op_desc_info.output_shape = {{1}};
  op_desc_info.output_data_type = {DT_FLOAT};
  op_desc_info.output_addrs = {nullptr};
  op_desc_info.output_size = {2};
  op_desc_info.space_addrs = {reinterpret_cast<void *>(5000)};
  op_desc_info.workspace_bytes = {0};
  ExceptionDumper exception_dumper;
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(exception_dumper.DumpNodeInfo(op_desc_info, "/var", false, false, dump_properties), ge::SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);

  // exception dump logs for tools analysis, don't modify!!!
  EXPECT_EQ(exception_dumper.DumpNodeInfo(op_desc_info, "/var", true, false, dump_properties), ge::SUCCESS);
  EXPECT_TRUE(CheckLogExpected(runtime_stub.GetSlogStub().GetLogs(), "[Dump][Exception] dump exception to file, file:"));
}

TEST_F(UTEST_dump_exception, SaveLiteException_mc2) {
  DumpStub::GetInstance().Clear();
  auto op_desc = std::make_shared<OpDesc>("MatmulAllreduce", "MatmulAllreduce");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_INT32);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape, FORMAT_NCHW, DT_INT32);
  EXPECT_TRUE(scalar_shape.IsScalar());
  std::vector<int64_t> dyn_dim{-1, 1024};
  GeShape dyn_shape(dyn_dim);
  GeTensorDesc dyn_desc(dyn_shape, FORMAT_NCHW, DT_INT32);
  GeTensorDesc invalid_desc({}, FORMAT_RESERVED, DT_UNDEFINED);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(invalid_desc);
  op_desc->AddInputDesc(scalar_desc);
  op_desc->AddInputDesc(invalid_desc);
  op_desc->AddOutputDesc(dyn_desc);
  op_desc->AddOutputDesc(scalar_desc);
  op_desc->SetWorkspaceBytes(dim);
  ge::diagnoseSwitch::EnableLiteExceptionDump();
  const uint64_t pld = std::numeric_limits<uint64_t>::max();
  std::vector<size_t> l0_dump_list_{0, 1, pld, pld, 2, 3, 4};
  ReportL0ExceptionDumpInfo(op_desc, l0_dump_list_);
  const auto &units = DumpStub::GetInstance().GetUnits();
  EXPECT_TRUE(!units.empty());
  auto size_list = *units.rbegin();
  EXPECT_TRUE(size_list.size() >= 9);
  EXPECT_EQ(size_list[1], 7);
  EXPECT_EQ(size_list[2], 1024);
  EXPECT_EQ(size_list[3], 4);  // scalar
  EXPECT_EQ(size_list[4], 0);
  EXPECT_EQ(size_list[5], 0);
  EXPECT_EQ(size_list[6], 0);  // -1
  EXPECT_EQ(size_list[7], 4);  // scalar
  EXPECT_EQ(size_list[8], 4);
}

TEST_F(UTEST_dump_exception, SaveLiteException_ifa) {
  DumpStub::GetInstance().Clear();
  auto op_desc = std::make_shared<OpDesc>("IncreFlashAttention", "IncreFlashAttention");
  for (int i = 1; i <= 5; ++i) {
    std::vector<int64_t> dim(2, i);
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_INT32);
    op_desc->AddInputDesc(tensor_desc);
  }

  std::vector<int64_t> dim(2, 2);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_INT32);
  op_desc->AddOutputDesc(tensor_desc);  // out
  op_desc->SetWorkspaceBytes({1000});   // ws
  ge::diagnoseSwitch::EnableLiteExceptionDump();

  std::vector<size_t> l0_dump_list_{0, 0x200000000000002, 1, 2, 0x200000000000002, 3, 4, 5, 6};
  ReportL0ExceptionDumpInfo(op_desc, l0_dump_list_);
  const auto &units = DumpStub::GetInstance().GetUnits();
  EXPECT_TRUE(!units.empty());
  auto size_list = *units.rbegin();
  EXPECT_TRUE(size_list.size() >= 11);

  EXPECT_EQ(size_list[1], 9);
  EXPECT_EQ(size_list[4], 16);
  EXPECT_EQ(size_list[5], 36);
  EXPECT_EQ(size_list[10], 1000);
}

TEST_F(UTEST_dump_exception, SaveLiteException_need_assert) {
  DumpStub::GetInstance().Clear();
  auto op_desc = std::make_shared<OpDesc>("IncreFlashAttention", "IncreFlashAttention");
  std::vector<int64_t> dim1(2, 1);
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1, FORMAT_NCHW, DT_INT32);
  op_desc->AddInputDesc(tensor_desc1);

  std::vector<int64_t> dim2(2, 2);
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2, FORMAT_NCHW, DT_INT32);
  op_desc->AddOutputDesc(tensor_desc2);  // out
  op_desc->SetWorkspaceBytes({1000});   // ws

  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxAssert = "assert";
  std::vector<std::string> dfx_opts{kOpDfxAssert};
  ge::AttrUtils::SetListStr(op_desc, kOpDfxOptions, dfx_opts);

  std::vector<size_t> l0_dump_list_{0, 1, 2};
  ReportL0ExceptionDumpInfo(op_desc, l0_dump_list_);
  const auto &units = DumpStub::GetInstance().GetUnits();
  EXPECT_TRUE(!units.empty());
  auto size_list = *units.rbegin();
  EXPECT_TRUE(size_list.size() >= 5);
  EXPECT_EQ(size_list[4], 1000 + (kAssertWorkFlag << kDumpTypeBitNum));
}

TEST_F(UTEST_dump_exception, SaveLiteException_need_print) {
  DumpStub::GetInstance().Clear();
  auto op_desc = std::make_shared<OpDesc>("IncreFlashAttention", "IncreFlashAttention");
  std::vector<int64_t> dim1(2, 1);
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1, FORMAT_NCHW, DT_INT32);
  op_desc->AddInputDesc(tensor_desc1);

  std::vector<int64_t> dim2(2, 2);
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2, FORMAT_NCHW, DT_INT32);
  op_desc->AddOutputDesc(tensor_desc2);  // out
  op_desc->SetWorkspaceBytes({1000});   // ws

  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxPrintf= "printf";
  std::vector<std::string> dfx_opts{kOpDfxPrintf};
  ge::AttrUtils::SetListStr(op_desc, kOpDfxOptions, dfx_opts);

  std::vector<size_t> l0_dump_list_{0, 1, 2};
  ReportL0ExceptionDumpInfo(op_desc, l0_dump_list_);
  const auto &units = DumpStub::GetInstance().GetUnits();
  EXPECT_TRUE(!units.empty());
  auto size_list = *units.rbegin();
  EXPECT_TRUE(size_list.size() >= 5);
  EXPECT_EQ(size_list[4], 1000 + (kAssertWorkFlag << kDumpTypeBitNum));
}

TEST_F(UTEST_dump_exception, AdumpOpInfoBuilder_test) {
  AdumpOpInfoBuilder builder("Test", "Add", true);
  builder.DeviceInfo("ARGS_BEFOR_EXECUTE", nullptr, 0);
  EXPECT_EQ(builder.Build().deviceInfos.size(), 0);

  builder.DeviceInfo("ARGS_BEFOR_EXECUTE", nullptr, 100);
  EXPECT_EQ(builder.Build().deviceInfos.size(), 0);

  builder.DeviceInfo("ARGS_BEFOR_EXECUTE", &builder, 0);
  EXPECT_EQ(builder.Build().deviceInfos.size(), 0);

  builder.DeviceInfo("ARGS_BEFOR_EXECUTE", &builder, 100);
  EXPECT_EQ(builder.Build().deviceInfos.size(), 1);
}
}  // namespace ge
