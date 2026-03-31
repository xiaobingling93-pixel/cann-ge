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
#include "common/dump/data_dumper.h"
#include "framework/common/types.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/sgt_slice_type.h"
#include "macro_utils/dt_public_unscope.h"
#include "runtime_stub.h"

using namespace std;

namespace {
struct AdumpCallInfo {
  int call_count = 0;
  std::string op_type;
  std::string op_name;
  std::vector<Adx::TensorInfo> tensors;
  rtStream_t stream;
  AdxStub::DumpCfg cfg;
} g_adump_call;

void ResetAdumpCall() {
  g_adump_call = AdumpCallInfo();
}
}  // namespace

int32_t AdxStub::AdumpDumpTensorWithCfg(const std::string& op_type, const std::string& op_name,
                                        const std::vector<Adx::TensorInfo>& tensors,
                                        rtStream_t stream, const AdxStub::DumpCfg& cfg) {
  g_adump_call.call_count++;
  g_adump_call.op_type = op_type;
  g_adump_call.op_name = op_name;
  g_adump_call.tensors = tensors;
  g_adump_call.stream = stream;
  g_adump_call.cfg = cfg;
  return 0;
}

namespace ge {
class UtestDataDumper : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

std::vector<void *> stub_get_output_addrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  std::vector<void *> res;
  res.emplace_back(reinterpret_cast<void *>(23333));
  return res;
}

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});
  return op_desc;
}

TEST_F(UtestDataDumper, LoadDumpInfo_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  std::shared_ptr<OpDesc> op_desc_1(new OpDesc());
  op_desc_1->AddOutputDesc("test", GeTensorDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc_1, 0);
  string dump_mode = "output";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.ReLoadDumpInfo();
  data_dumper.UnloadDumpInfo();
  data_dumper.ReLoadDumpInfo();
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, ExecuteDumpInfo_failure) {
  RuntimeParam param;
  DataDumper dumper(&param);
  toolkit::aicpu::dump::OpMappingInfo mapping;

  // Trying to dump empty OpMappingInfo.
  EXPECT_EQ(dumper.ExecuteLoadDumpInfo(mapping), PARAM_INVALID);
  EXPECT_EQ(dumper.ExecuteUnLoadDumpInfo(mapping), PARAM_INVALID);

  // Initialize something in OpMappingInfo.
  mapping.set_dump_path("/tmp/dummy/");

  RTS_STUB_RETURN_VALUE(rtMalloc, rtError_t, -1);
  EXPECT_NE(dumper.ExecuteLoadDumpInfo(mapping), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtMalloc, rtError_t, -1);
  EXPECT_NE(dumper.ExecuteUnLoadDumpInfo(mapping), SUCCESS);

  RTS_STUB_RETURN_VALUE(rtMemcpy, rtError_t, -1);
  EXPECT_NE(dumper.ExecuteLoadDumpInfo(mapping), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtMemcpy, rtError_t, -1);
  EXPECT_NE(dumper.ExecuteUnLoadDumpInfo(mapping), SUCCESS);

  RTS_STUB_RETURN_VALUE(rtDatadumpInfoLoad, rtError_t, -1);
  EXPECT_NE(dumper.ExecuteLoadDumpInfo(mapping), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtDatadumpInfoLoad, rtError_t, -1);
  EXPECT_NE(dumper.ExecuteUnLoadDumpInfo(mapping), SUCCESS);
}

TEST_F(UtestDataDumper, buildtask_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  std::shared_ptr<OpDesc> op_desc_1(new OpDesc());
  op_desc_1->AddOutputDesc("test", GeTensorDesc());
  (void)AttrUtils::SetInt(op_desc_1, "current_context_id", 1);
  (void)AttrUtils::SetStr(op_desc_1, "_sgt_json_info", "123");
  vector<uint64_t> task_addr_offset{10, 10};
  (void)op_desc_1->SetExtAttr("task_addr_offset", task_addr_offset);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc_1, 0);
  string dump_mode = "all";
  data_dumper.is_op_debug_ = true;
  DataDumper::InnerRealAddressAndSize real_address_and_size;
  data_dumper.context_.input.emplace_back(real_address_and_size);
  data_dumper.context_.output.emplace_back(real_address_and_size);
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, BuildtaskInfo_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  std::shared_ptr<OpDesc> op_desc_1(new OpDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc_1, 0);
  string dump_mode = "input";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, BuildtaskInfoForPreprocessKernel_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  std::shared_ptr<OpDesc> op_desc_1(new OpDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc_1, 0, {}, {}, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL);
  string dump_mode = "input";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, SetWorkSpaceAddrForPrint_OneOpDescMultipleOpListElements) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  OpDescPtr op_desc = CreateOpDesc("mc2", "MC2");
  (void)AttrUtils::SetInt(op_desc, "current_context_id", 1);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  data_dumper.SavePrintDumpTask({0, 0, 0, 0}, op_desc, 0);
  data_dumper.SavePrintDumpTask({1, 0, 0, 0}, op_desc, 0);
  data_dumper.SavePrintDumpTask({2, 1, 0, 0}, op_desc, 0);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  std::string dump_path = "/tmp/";
  DumpProperties dump_properties;
  dump_properties.SetDumpPath(dump_path);
  data_dumper.SetDumpProperties(dump_properties);

  int64_t buffer_size = 23;
  const std::string kOpDfxBufferSize = "_op_dfx_buffer_size";
  ge::AttrUtils::SetInt(op_desc, kOpDfxBufferSize, buffer_size);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32, 64, 128});
  std::vector<uint64_t> space_addrs = {0x1000U, 0x2000U, 0x3000U};
  data_dumper.SetWorkSpaceAddrForPrint(op_desc, space_addrs);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  int match_count = 0;
  for (const auto& print_task : data_dumper.op_print_list_) {
    if (print_task.op != nullptr && print_task.op->GetId() == op_desc->GetId()) {
      EXPECT_EQ(print_task.space_addr.size(), space_addrs.size());
      for (size_t j = 0; j < space_addrs.size(); ++j) {
        EXPECT_EQ(print_task.space_addr[j], space_addrs[j]);
      }
      match_count++;
    }
  }

  EXPECT_EQ(match_count, 3);
}

TEST_F(UtestDataDumper, SetWorkSpaceAddr_OneOpDescMultipleOpListElements) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  OpDescPtr op_desc = CreateOpDesc("mc2", "MC2");
  (void)AttrUtils::SetInt(op_desc, "current_context_id", 1);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  data_dumper.SaveDumpTask({1, 0, 0, 0}, op_desc, 0);
  data_dumper.SaveDumpTask({2, 1, 0, 0}, op_desc, 0);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  std::string dump_path = "/tmp/";
  DumpProperties dump_properties;
  dump_properties.SetDumpPath(dump_path);
  data_dumper.SetDumpProperties(dump_properties);

  int64_t buffer_size = 23;
  const std::string kOpDfxBufferSize = "_op_dfx_buffer_size";
  ge::AttrUtils::SetInt(op_desc, kOpDfxBufferSize, buffer_size);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32, 64, 128});
  std::vector<uint64_t> space_addrs = {0x1000U, 0x2000U, 0x3000U};
  data_dumper.SetWorkSpaceAddr(op_desc, space_addrs);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  int match_count = 0;
  for (const auto& task : data_dumper.op_list_) {
    if (task.op != nullptr && task.op->GetId() == op_desc->GetId()) {
      EXPECT_EQ(task.space_addr.size(), space_addrs.size());
      for (size_t j = 0; j < space_addrs.size(); ++j) {
        EXPECT_EQ(task.space_addr[j], space_addrs[j]);
      }
      match_count++;
    }
  }

  EXPECT_EQ(match_count, 3);
}

TEST_F(UtestDataDumper, BuildTaskInfoForPrint_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  OpDescPtr op_desc = CreateOpDesc("conv", CONVOLUTION);
  (void)AttrUtils::SetInt(op_desc, "current_context_id", 1);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  data_dumper.SavePrintDumpTask({0, 0, 0, 0}, nullptr, 0);
  data_dumper.SavePrintDumpTask({0, 0, 0, 0}, op_desc, 0);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  std::string dump_path = "/tmp/";
  DumpProperties dump_properties;
  dump_properties.SetDumpPath(dump_path);
  data_dumper.SetDumpProperties(dump_properties);

  int64_t buffer_size = 23;
  const std::string kOpDfxBufferSize = "_op_dfx_buffer_size";
  ge::AttrUtils::SetInt(op_desc, kOpDfxBufferSize, buffer_size);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  std::vector<uint64_t> space_addr = {23333U};
  data_dumper.SetWorkSpaceAddrForPrint(op_desc, space_addr);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  op_desc->SetWorkspaceBytes(vector<int64_t>{10});
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, DumpWorkspace_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  data_dumper.is_op_debug_ = true;

  toolkit::aicpu::dump::Task task;
  OpDescPtr op_desc = CreateOpDesc("conv", CONVOLUTION);

  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);

  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  DataDumper::InnerDumpInfo inner_dump_info;
  inner_dump_info.op = op_desc;
  (void)data_dumper.DumpOutputWithTask(inner_dump_info, task);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  std::vector<uint64_t> space_addr = {23333U};
  data_dumper.SetWorkSpaceAddr(op_desc, space_addr);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, DumpOutputWithTask_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);

  toolkit::aicpu::dump::Task task;
  OpDescPtr op_desc = CreateOpDesc("conv", CONVOLUTION);
  GeTensorDesc tensor_0(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_1(GeShape(), FORMAT_NCHW, DT_FLOAT);
  int32_t calc_type = 1;
  ge::AttrUtils::SetInt(tensor_1, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  op_desc->AddOutputDesc(tensor_0);
  op_desc->AddOutputDesc(tensor_1);
  DataDumper::InnerDumpInfo inner_dump_info;
  inner_dump_info.op = op_desc;
  data_dumper.need_generate_op_buffer_ = true;
  Status ret = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(ret, SUCCESS);
  int64_t task_size = 1;
  data_dumper.GenerateOpBuffer(task_size, task);
}

TEST_F(UtestDataDumper, DumpOutputWithoutTask_success) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);

  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor_0(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_1(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc = CreateOpDesc("conv", CONVOLUTION);
  int32_t calc_type = 1;
  ge::AttrUtils::SetInt(tensor_1, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddOutputDesc(tensor_0);
  op_desc->AddOutputDesc(tensor_1);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  EXPECT_EQ(data_dumper.DumpOutput(inner_dump_info, task), SUCCESS);
}

TEST_F(UtestDataDumper, DumpOutput_test) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  RuntimeParam rts_param;

  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(11);
  data_dumper.SetDeviceId(11);
  data_dumper.SaveEndGraphId(0U, 0U);
  data_dumper.SetComputeGraph(graph);
  data_dumper.need_generate_op_buffer_ = true;
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  DataDumper::InnerDumpInfo inner_dump_info{};

  OpDescPtr op_desc1 = CreateOpDesc("conv", CONVOLUTION);
  GeTensorDesc tensor1(GeShape(), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(tensor1, ATTR_DATA_DUMP_REF, "a");
  op_desc1->AddOutputDesc(tensor1);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc1, 0);
  graph->AddNode(op_desc1);
  inner_dump_info.op = op_desc1;
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, PARAM_INVALID);

  data_dumper.dump_properties_.SetDumpMode(std::string("all"));
  data_dumper.is_op_debug_ = true;
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  data_dumper.is_op_debug_ = false;
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  OpDescPtr op_desc2 = CreateOpDesc("conv", CONVOLUTION);
  GeTensorDesc tensor2(GeShape(), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(op_desc2, ATTR_DATA_DUMP_REF, "a:b");
  op_desc2->AddOutputDesc(tensor2);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc2, 0);
  graph->AddNode(op_desc2);
  inner_dump_info.op = op_desc2;
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);

  OpDescPtr op_desc3 = CreateOpDesc("conv", CONVOLUTION);
  GeTensorDesc tensor3(GeShape(), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(tensor3, ATTR_DATA_DUMP_REF, "conv:output:1");
  op_desc3->AddOutputDesc(tensor3);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc3, 0);
  graph->AddNode(op_desc3);
  inner_dump_info.op = op_desc3;
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, PARAM_INVALID);

  OpDescPtr op_desc4 = CreateOpDesc("conv", CONVOLUTION);
  std::string json_str =
      "{\"dependencies\":[],\"thread_scopeId\":200,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
      "\"oriOutputTensorShape\":["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{\"splitCutIndex\":1,\"reduceCutIndex\":2,\"cutId\":3}],"
      "\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"output_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  ge::AttrUtils::SetStr(op_desc4, ffts::kAttrSgtJsonInfo, json_str);
  GeTensorDesc tensor4(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc4->AddInputDesc(tensor4);
  op_desc4->AddOutputDesc(tensor4);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc4, 0);
  graph->AddNode(op_desc4);
  inner_dump_info.op = op_desc4;
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestDataDumper, DumpOutput_with_memlist) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  RuntimeParam rts_param;

  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(11);
  data_dumper.SaveEndGraphId(0U, 0U);
  data_dumper.SetComputeGraph(graph);
  data_dumper.need_generate_op_buffer_ = true;
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  DataDumper::InnerDumpInfo inner_dump_info;

  ge::OpDescPtr op_desc = CreateOpDesc("conv2", "conv2");
  std::vector<int64_t> in_memory_type_list = {static_cast<int64_t>(RT_MEMORY_L1)};
  std::vector<int64_t> out_memory_type_list = {static_cast<int64_t>(RT_MEMORY_L1)};
  (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type_list);
  (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type_list);
  GeTensorDesc tensor1(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor1);

  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  graph->AddNode(op_desc);
  inner_dump_info.op = op_desc;
  int64_t *addr = (int64_t *)malloc(1024);
  inner_dump_info.args = reinterpret_cast<uintptr_t>(addr);
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);

  free(addr);
}

TEST_F(UtestDataDumper, DumpInput_test) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  RuntimeParam rts_param;

  DataDumper data_dumper(&rts_param);
  data_dumper.DumpShrink();
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(11);
  data_dumper.SaveEndGraphId(0U, 0U);
  data_dumper.SetComputeGraph(graph);
  data_dumper.need_generate_op_buffer_ = true;

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  DataDumper::InnerDumpInfo inner_dump_info{};

  OpDescPtr op_desc = CreateOpDesc("conv", CONVOLUTION);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(tensor, ATTR_DATA_DUMP_REF, "conv:input:1");
  op_desc->AddInputDesc(tensor);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  graph->AddNode(op_desc);
  inner_dump_info.op = op_desc;
  data_dumper.dump_properties_.SetDumpMode(std::string("all"));
  data_dumper.is_op_debug_ = true;
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
  data_dumper.dump_properties_.SetDumpMode(std::string("input"));
  retStatus = data_dumper.DumpInput(inner_dump_info, task);
  EXPECT_EQ(retStatus, PARAM_INVALID);
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  OpDescPtr op_desc2 = CreateOpDesc("conv", CONVOLUTION);
  std::string json_str =
      "{\"dependencies\":[],\"thread_scopeId\":200,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
      "\"oriOutputTensorShape\":["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{\"splitCutIndex\":1,\"reduceCutIndex\":2,\"cutId\":3}],"
      "\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"output_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  ge::AttrUtils::SetStr(op_desc2, ffts::kAttrSgtJsonInfo, json_str);
  GeTensorDesc tensor2(GeShape(), FORMAT_NCHW, DT_FLOAT);
  data_dumper.runtime_param_->mem_size = 101;
  op_desc2->AddInputDesc(tensor2);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc2, 0);
  graph->AddNode(op_desc2);
  inner_dump_info.op = op_desc2;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  retStatus = data_dumper.DumpInput(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);

  toolkit::aicpu::dump::Input input;
  std::string node_name_index = "a";
  retStatus = data_dumper.DumpRefInput(inner_dump_info, input, 1, node_name_index);
  EXPECT_EQ(retStatus, PARAM_INVALID);

  node_name_index = "conv:input:1";
  retStatus = data_dumper.DumpRefInput(inner_dump_info, input, 1, node_name_index);
  EXPECT_EQ(retStatus, PARAM_INVALID);
}

TEST_F(UtestDataDumper, PrintCheckLog_test) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);

  DumpProperties dump_properties1;
  std::string dump_model(DUMP_ALL_MODEL);
  std::string dump_path("/");
  std::string dump_mode("output");
  std::set<std::string> dump_layers;

  std::set<std::string> model_list = data_dumper.GetDumpProperties().GetAllDumpModel();
  data_dumper.SetDumpProperties(dump_properties1);
  std::string key("all");
  EXPECT_NO_THROW(data_dumper.PrintCheckLog(key));

  DumpProperties dump_properties2;
  dump_properties2.SetDumpMode(dump_mode);
  dump_properties2.AddPropertyValue(dump_model, dump_layers);
  dump_properties2.SetDumpPath(dump_path);
  data_dumper.SetDumpProperties(dump_properties2);
  EXPECT_NO_THROW(data_dumper.PrintCheckLog(dump_model));
}

TEST_F(UtestDataDumper, SaveDumpInput_invalid) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);

  EXPECT_NO_THROW(data_dumper.SaveDumpInput(nullptr));

  NodePtr op_node = std::make_shared<Node>(nullptr, nullptr);
  EXPECT_NO_THROW(data_dumper.SaveDumpInput(op_node));
}

TEST_F(UtestDataDumper, ReloadOpdebugInfoFailed) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelId(0);
  data_dumper.SetModelName("opDebug");
  std::shared_ptr<OpDesc> op_desc(new OpDesc());
  op_desc->AddOutputDesc("opDebug", GeTensorDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  string dump_mode = "output";
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
  data_dumper.is_op_debug_ = true;
  EXPECT_NE(data_dumper.ReLoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(UtestDataDumper, DumpOutput_ModelName_opblacklist_test) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["conv"].output_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].output_indices = {1};
  new_blacklist_map["test"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestDataDumper, DumpOutput_OmName_opblacklist_test) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetOmName("test1");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["conv"].output_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].output_indices = {1};
  new_blacklist_map["test1"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  retStatus = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestDataDumper, DumpOutput_DataType_ModelInputNode_opblacklist_test) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetOmName("test1");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["data"].output_indices = {1};
  model1_bl.dump_optype_blacklist["data"].output_indices = {0};
  new_blacklist_map["test1"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc = CreateOpDesc("data", "data");
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  retStatus = data_dumper.DumpOutput(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestDataDumper, DumpInput_ModelName_opblacklist_test) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("input"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["conv"].input_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].input_indices = {1};
  new_blacklist_map["test"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  EXPECT_EQ(data_dumper.DumpInput(inner_dump_info, task), SUCCESS);
}

TEST_F(UtestDataDumper, DumpIntput_OmName_opblacklist_test) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetOmName("test1");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("input"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["conv"].input_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].input_indices = {1};
  new_blacklist_map["test1"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  EXPECT_EQ(data_dumper.DumpInput(inner_dump_info, task), SUCCESS);
}
TEST_F(UtestDataDumper, IsDumpOpWithAdump_AllTrue) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  EXPECT_TRUE(dumper.IsDumpOpWithAdump());
}

TEST_F(UtestDataDumper, IsDumpOpWithAdump_OverflowFalse) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = false;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  EXPECT_FALSE(dumper.IsDumpOpWithAdump());
}

TEST_F(UtestDataDumper, IsDumpOpWithAdump_PersistentFalse) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = false;
  dumper.adump_interface_available_ = true;
  EXPECT_FALSE(dumper.IsDumpOpWithAdump());
}

TEST_F(UtestDataDumper, IsDumpOpWithAdump_AdumpUnavailable) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = false;
  EXPECT_FALSE(dumper.IsDumpOpWithAdump());
}

TEST_F(UtestDataDumper, IsDumpOpWithAdump_WatcherModelEnable) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.dump_properties_.AddPropertyValue(DUMP_WATCHER_MODEL, {"square", "allreduce"});
  EXPECT_FALSE(dumper.IsDumpOpWithAdump());
}

TEST_F(UtestDataDumper, SaveDumpTask_CallsAdump) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  // 强制启用所有条件
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.SetOmName("test_om");
  dumper.dump_properties_.SetDumpMode("all");
  ResetAdumpCall();

  OpDescPtr op_desc = CreateOpDesc("conv", "Conv2D");
  // 使用有效 shape 确保大小非零
  GeTensorDesc tensor(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  // 显式设置大小，避免 GetTensorSizeInBytes 失败
  TensorUtils::SetSize(tensor, 1*2*3*4*sizeof(float));
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);

  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0x12345678);
  dumper.SaveDumpTask({10, 20, 0, 0}, op_desc, 0x2000, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, false, fake_stream);

  EXPECT_FALSE(dumper.op_list_.empty());
  const auto &dump_info = dumper.op_list_.back();
  // 确保 stream 已保存
  EXPECT_EQ(dump_info.stream, fake_stream);
  Status ret = dumper.DumpOpWithAdump(dump_info);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(g_adump_call.call_count, 2);
  EXPECT_EQ(g_adump_call.op_name, "conv");
  EXPECT_EQ(g_adump_call.stream, fake_stream);
  EXPECT_GE(g_adump_call.tensors.size(), 2);
}

TEST_F(UtestDataDumper, SavePrintDumpTask_CallsAdump) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.dump_properties_.SetDumpMode("all");
  ResetAdumpCall();

  OpDescPtr op_desc = CreateOpDesc("print_op", "Print");
  GeTensorDesc tensor(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 1*2*3*4*sizeof(float));
  op_desc->AddInputDesc(tensor);
  op_desc->SetWorkspaceBytes({64});

  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0x87654321);
  dumper.SavePrintDumpTask({10, 20, 0, 0}, op_desc, 0x2000, {}, ModelTaskType::MODEL_TASK_KERNEL, fake_stream);

  // 设置工作空间地址（必须在 SavePrintDumpTask 之后，因为需要找到 op_print_list_ 中的任务）
  std::vector<uint64_t> space_addr = {0x2000U};
  dumper.SetWorkSpaceAddrForPrint(op_desc, space_addr);

  EXPECT_FALSE(dumper.op_print_list_.empty());
  const auto &dump_info = dumper.op_print_list_.back();
  EXPECT_EQ(dump_info.stream, fake_stream);
  Status ret = dumper.DumpOpWithAdump(dump_info);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(g_adump_call.call_count, 2);
  EXPECT_EQ(g_adump_call.op_name, "print_op");
  EXPECT_EQ(g_adump_call.stream, fake_stream);
}
// 修正 DumpOpWithAdump_NormalMode
TEST_F(UtestDataDumper, DumpOpWithAdump_NormalMode) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.dump_properties_.SetDumpMode("all");

  OpDescPtr op_desc = CreateOpDesc("normal_op", "Normal");
  GeTensorDesc tensor(GeShape({2,3}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetWorkspaceBytes({64});

  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0xaaaabbbb);

  // 临时禁用 Adump 条件，避免 SaveDumpTask 自动调用
  bool saved_overflow = dumper.overflow_enabled_;
  dumper.overflow_enabled_ = false;
  dumper.SaveDumpTask({0,0,0,0}, op_desc, 0x1000, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, true, fake_stream);
  dumper.overflow_enabled_ = saved_overflow;

  // 设置工作空间地址
  std::vector<uint64_t> space_addrs = {0x3000U};
  dumper.SetWorkSpaceAddr(op_desc, space_addrs);

  // 重置记录，然后手动调用 DumpOpWithAdump
  ResetAdumpCall();
  const auto &dump_info = dumper.op_list_.back();
  Status ret = dumper.DumpOpWithAdump(dump_info);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(g_adump_call.call_count, 1);
  const auto& tensors = g_adump_call.tensors;
  EXPECT_EQ(tensors.size(), 3);
  bool has_workspace = false;
  for (const auto& t : tensors) {
    if (t.type == Adx::TensorType::WORKSPACE) has_workspace = true;
  }
  EXPECT_TRUE(has_workspace);
}
// 测试输入黑名单在 FillInputTensorInfos 中生效
TEST_F(UtestDataDumper, DumpOpWithAdump_InputBlacklist) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.SetModelName("test_model");
  dumper.SetOmName("test_om");
  dumper.dump_properties_.SetDumpMode("input");

  std::map<std::string, ModelOpBlacklist> blacklist;
  ModelOpBlacklist bl;
  bl.dump_opname_blacklist["conv"].input_indices = {0};
  blacklist["test_model"] = bl;
  dumper.dump_properties_.SetModelDumpBlacklistMap(blacklist);

  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);  // 两个输入

  ResetAdumpCall();
  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0x12345678);
  dumper.SaveDumpTask({0,0,0,0}, op_desc, 0x1000, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, false, fake_stream);

  EXPECT_EQ(g_adump_call.call_count, 1);
  const auto& tensors = g_adump_call.tensors;
  // 只有第二个输入（索引1）被保留
  EXPECT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].type, Adx::TensorType::INPUT);
  EXPECT_EQ(tensors[0].argsOffSet, 1);
}

// 测试输出黑名单在 FillOutputTensorInfos 中生效
TEST_F(UtestDataDumper, DumpOpWithAdump_OutputBlacklist) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.SetModelName("test_model");
  dumper.SetOmName("test_om");
  dumper.dump_properties_.SetDumpMode("output");

  std::map<std::string, ModelOpBlacklist> blacklist;
  ModelOpBlacklist bl;
  bl.dump_opname_blacklist["conv"].output_indices = {0};
  blacklist["test_model"] = bl;
  dumper.dump_properties_.SetModelDumpBlacklistMap(blacklist);

  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);  // 两个输出

  ResetAdumpCall();
  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0x12345678);
  dumper.SaveDumpTask({0,0,0,0}, op_desc, 0x1000, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, false, fake_stream);

  EXPECT_EQ(g_adump_call.call_count, 1);
  const auto& tensors = g_adump_call.tensors;
  // 只有第二个输出（索引1）被保留
  EXPECT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].type, Adx::TensorType::OUTPUT);
  EXPECT_EQ(tensors[0].argsOffSet, 1);  // input_count=0, output_index=1 => offset=1
}

// 测试 stream 为空时返回失败
TEST_F(UtestDataDumper, DumpOpWithAdump_StreamNull) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  ResetAdumpCall();

  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);

  // 传入 stream = nullptr
  rtStream_t null_stream = nullptr;
  dumper.SaveDumpTask({0,0,0,0}, op_desc, 0x1000, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, false, null_stream);

  // DumpOpWithAdump 应返回失败，所以 Adump 不会被调用
  EXPECT_EQ(g_adump_call.call_count, 0);
  // 但 SaveDumpTask 会记录调用结果日志，不会导致异常
}

// 测试在 op_debug 模式下跳过 Data 节点
TEST_F(UtestDataDumper, DumpOpWithAdump_SkipDataNode) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.SetModelName("test_model");
  dumper.dump_properties_.is_train_op_debug_ = true;
  ResetAdumpCall();

  OpDescPtr op_desc = CreateOpDesc("data", "Data");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);

  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0x12345678);
  dumper.SaveDumpTask({0,0,0,0}, op_desc, 0x1000, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, false, fake_stream);

  // 由于是 Data 节点且 op_debug 开启，应跳过，Adump 不被调用
  EXPECT_EQ(g_adump_call.call_count, 0);
}

// 测试 LoadDumpInfo 在 Adump 启用时直接返回
TEST_F(UtestDataDumper, LoadDumpInfo_AdumpEnabled) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.dump_properties_.SetDumpMode("output");

  // 添加一个 dummy op，但 LoadDumpInfo 应提前返回，不会构建任务
  OpDescPtr op_desc = CreateOpDesc("conv", "conv");
  dumper.SaveDumpTask({0,0,0,0}, op_desc, 0);

  Status ret = dumper.LoadDumpInfo();
  EXPECT_EQ(ret, SUCCESS);
}

// 测试 SaveDumpTask 中 input_map_ 分支
TEST_F(UtestDataDumper, Adump_InputNode) {
  // 创建 DataDumper 并启用条件
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.SetOmName("test_om");
  dumper.dump_properties_.SetDumpMode("input");

  // 创建输入节点（data_op）和主节点（consumer_op）
  OpDescPtr data_op = CreateOpDesc("data", "Data");
  GeTensorDesc tensor(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 1*2*3*4*sizeof(float));
  data_op->AddOutputDesc(tensor);   // data 有一个输出

  OpDescPtr consumer_op = CreateOpDesc("conv", "conv");
  consumer_op->AddInputDesc(tensor);  // consumer 有一个输入

  // 直接插入 input_map_ 映射：表示 consumer 的输入索引 0 来自 data_op 的输出索引 0
  dumper.input_map_.insert({consumer_op->GetName(),
                            {data_op, 0, 0}});  // data_op, input_anchor_index=0, output_anchor_index=0

  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0x2222);
  ResetAdumpCall();

  // 保存主任务（consumer_op），此时内部会遍历 input_map_ 中匹配 consumer 的条目，并处理输入节点
  dumper.SaveDumpTask({0, 0, 0, 0}, consumer_op, 0x1000, {}, {},
                      ModelTaskType::MODEL_TASK_KERNEL, false, fake_stream);

  // 验证 Adump 被调用两次（主算子一次，输入节点一次）
  EXPECT_EQ(g_adump_call.call_count, 1);
}

// 覆盖 FillInputTensorInfos 中 originShape 分支
TEST_F(UtestDataDumper, FillInputTensorInfos_WithOriginShape) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);

  OpDescPtr op_desc = CreateOpDesc("test_op", "Test");
  GeTensorDesc tensor(GeShape({1,2,3}), FORMAT_NCHW, DT_FLOAT);
  // 设置 origin shape
  GeShape origin_shape({4,5,6});
  tensor.SetOriginShape(origin_shape);
  op_desc->AddInputDesc(tensor);

  std::vector<Adx::TensorInfo> tensors;
  dumper.FillInputTensorInfos(op_desc, 0x1000, {}, tensors);
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].originShape.size(), 3);
  EXPECT_EQ(tensors[0].originShape[0], 4);
  EXPECT_EQ(tensors[0].originShape[1], 5);
  EXPECT_EQ(tensors[0].originShape[2], 6);
}

// 覆盖 FillOutputTensorInfos 中 originShape 分支
TEST_F(UtestDataDumper, FillOutputTensorInfos_WithOriginShape) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);

  OpDescPtr op_desc = CreateOpDesc("test_op", "Test");
  GeTensorDesc tensor(GeShape({1,2,3}), FORMAT_NCHW, DT_FLOAT);
  GeShape origin_shape({4,5,6});
  tensor.SetOriginShape(origin_shape);
  op_desc->AddOutputDesc(tensor);

  std::vector<Adx::TensorInfo> tensors;
  dumper.FillOutputTensorInfos(op_desc, 0x1000, 0, {}, tensors);
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].originShape.size(), 3);
  EXPECT_EQ(tensors[0].originShape[0], 4);
  EXPECT_EQ(tensors[0].originShape[1], 5);
  EXPECT_EQ(tensors[0].originShape[2], 6);
}

// 覆盖 FillWorkspaceTensorInfos 中地址为0的跳过
TEST_F(UtestDataDumper, FillWorkspaceTensorInfos_SkipZeroAddr) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);

  OpDescPtr op_desc = CreateOpDesc("test_op", "Test");
  op_desc->SetWorkspaceBytes({32, 64, 128});
  std::vector<uint64_t> space_addrs = {0x1000, 0x0, 0x2000};

  DataDumper::InnerDumpInfo dump_info;
  dump_info.op = op_desc;
  dump_info.space_addr = space_addrs;

  std::vector<Adx::TensorInfo> tensors;
  dumper.FillWorkspaceTensorInfos(dump_info, tensors);
  // 只有地址非零的两个 workspace 被添加
  EXPECT_EQ(tensors.size(), 2);
  EXPECT_EQ(reinterpret_cast<uint64_t>(tensors[0].tensorAddr), 0x1000);
  EXPECT_EQ(reinterpret_cast<uint64_t>(tensors[1].tensorAddr), 0x2000);
}

// 覆盖 FillRawTensorInfos 中地址列表不足、地址为0的跳过
TEST_F(UtestDataDumper, FillRawTensorInfos_EdgeCases) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);

  OpDescPtr op_desc = CreateOpDesc("test_op", "Test");
  GeTensorDesc tensor(GeShape({1,2,3}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 24);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);

  DataDumper::InnerDumpInfo dump_info;
  dump_info.op = op_desc;
  dump_info.is_raw_address = true;

  // 地址列表不足
  dump_info.address = {0x1000, 0x2000};  // 只有两个地址，需要 input(1)+output(2)=3
  std::vector<Adx::TensorInfo> tensors;
  Status ret = dumper.FillRawTensorInfos(dump_info, tensors);
  EXPECT_EQ(ret, PARAM_INVALID);

  // 地址列表足够但有0地址
  dump_info.address = {0x1000, 0x0, 0x3000};
  tensors.clear();
  ret = dumper.FillRawTensorInfos(dump_info, tensors);
  EXPECT_EQ(ret, SUCCESS);
  // 输入1个，输出2个，但第二个输出地址为0（索引2），实际只有输入和第一个输出
  EXPECT_EQ(tensors.size(), 2);
  EXPECT_EQ(reinterpret_cast<uint64_t>(tensors[0].tensorAddr), 0x1000);
  EXPECT_EQ(reinterpret_cast<uint64_t>(tensors[1].tensorAddr), 0x3000);
}

// 覆盖 DumpOpWithAdump 中 tensors.empty() 返回 SUCCESS 分支
TEST_F(UtestDataDumper, DumpOpWithAdump_EmptyTensors) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.dump_properties_.SetDumpMode("output");
  // 黑名单过滤所有输出
  std::map<std::string, ModelOpBlacklist> blacklist;
  ModelOpBlacklist bl;
  bl.dump_opname_blacklist["test_op"].output_indices = {0};
  blacklist["test_model"] = bl;
  dumper.dump_properties_.SetModelDumpBlacklistMap(blacklist);

  OpDescPtr op_desc = CreateOpDesc("test_op", "Test");
  GeTensorDesc tensor(GeShape({1,2,3}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor);
  // 没有输入，输出被过滤，所以 tensors 为空
  ResetAdumpCall();
  rtStream_t fake_stream = reinterpret_cast<rtStream_t>(0xdeadbeef);
  DataDumper::InnerDumpInfo dump_info;
  dump_info.op = op_desc;
  dump_info.stream = fake_stream;
  dump_info.is_raw_address = false;
  dump_info.args = 0x1000;
  Status ret = dumper.DumpOpWithAdump(dump_info);
  EXPECT_EQ(ret, SUCCESS);
  // 由于 tensors.empty()，返回 SUCCESS 但未调用 Adump
  EXPECT_EQ(g_adump_call.call_count, 0);
}

// 覆盖 DumpOpWithAdump 中 stream == nullptr 失败分支
TEST_F(UtestDataDumper, DumpOpWithAdump_StreamNull_Fail) {
  RuntimeParam rts_param;
  DataDumper dumper(&rts_param);
  dumper.overflow_enabled_ = true;
  dumper.persistent_unlimited_enabled_ = true;
  dumper.adump_interface_available_ = true;
  dumper.SetModelName("test_model");
  dumper.SetModelId(123);
  dumper.dump_properties_.SetDumpMode("output");

  OpDescPtr op_desc = CreateOpDesc("test_op", "Test");
  GeTensorDesc tensor(GeShape({1,2,3}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor);

  ResetAdumpCall();
  DataDumper::InnerDumpInfo dump_info;
  dump_info.op = op_desc;
  dump_info.stream = nullptr;  // 故意为空
  dump_info.is_raw_address = false;
  dump_info.args = 0x1000;
  Status ret = dumper.DumpOpWithAdump(dump_info);
  EXPECT_EQ(ret, FAILED);
  // 不应调用 Adump
  EXPECT_EQ(g_adump_call.call_count, 0);
}

}  // namespace ge
