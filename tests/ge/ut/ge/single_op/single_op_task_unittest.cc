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
#include <vector>
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/op_desc.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "framework/common/profiling_definitions.h"
#include "runtime/rt.h"

#include "macro_utils/dt_public_scope.h"
#include "single_op/single_op_model.h"
#include "aicpu_task_struct.h"
#include "single_op/task/tbe_task_builder.h"
#include "single_op/task/op_task.h"
#include "single_op/task/tbe_task_builder.h"
#include "common/profiling/profiling_manager.h"
#include "register/op_tiling_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/global_variables/diagnose_switch.h"
#include "depends/profiler/src/dump_stub.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace optiling;

namespace {
  struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[3];
}__attribute__((packed));
}  // namespace

class UtestSingleOpTask : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

void LaunchKernelTask() {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  model.input_offset_list_.push_back(0);
  model.input_sizes_.push_back(16);

  model.output_offset_list_.push_back(0);
  model.output_sizes_.push_back(16);

  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  vector<int64_t> shape{16, 16};
  GeShape ge_shape(shape);
  GeTensorDesc desc(ge_shape);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);

  std::mutex stream_mu_;
  rtStream_t stream_ = nullptr;
  StreamResource stream_resource(0);
  SingleOp single_op(&stream_resource, &stream_mu_, stream_);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_with_handle = task_def.mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  model.op_list_[1] = node;

  TbeOpTask task_tmp;
  TbeOpTask *task = &task_tmp;
  StreamResource resource(0U);
  ASSERT_EQ(model.BuildKernelTask(task_def, &task, resource), SUCCESS);
  ge::DataBuffer data_buffer;
  vector<GeTensorDesc> input_desc;
  vector<DataBuffer> input_buffers = { data_buffer };
  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers = { data_buffer };
  task->node_ = node;
  OpTilingFuncV2 op_tiling_func = [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &) -> bool {return true;};
  REGISTER_OP_TILING_UNIQ_V2(Add, op_tiling_func, 1);
  OpTilingRegistryInterf_V2("Add", op_tiling_func);
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "op_compile_info_json");
  task->max_tiling_size_ = 64;
  task->run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task->run_info_, nullptr);
  task->arg_size_ = 64;
  task->args_.reset(new (std::nothrow) uint8_t[task->arg_size_]);

  ASSERT_EQ(task->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream_), SUCCESS);
  char handle = '0';
  task->SetHandle(&handle);
  ASSERT_EQ(task->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream_), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOpTask, test_build_kernel_task) {
  LaunchKernelTask();
}

TEST_F(UtestSingleOpTask, KerenlTaskProf_RecordInfo_WithProfilingOn) {
  ProfilingProperties::Instance().SetLoadProfiling(true);
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len)->int32_t {
    if (type == InfoType::kInfo) {
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_NE(info->dataLen, 0);
      auto prof_tensor_info = reinterpret_cast<MsprofTensorInfo *>(info->data);
      std::hash<std::string> hs;
      EXPECT_EQ(prof_tensor_info->opName, hs("Add"));
    }
    if (type == InfoType::kApi) {
      std::hash<std::string> hs;
      auto info = reinterpret_cast<MsprofApi *>(data);
      EXPECT_EQ(info->itemId, hs("Add"));
    }
    return 0;
  };
  AutoProfilingTestWithExpectedFunc(LaunchKernelTask, check_func);
}

TEST_F(UtestSingleOpTask, test_update_ioaddr) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");

  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  vector<bool> is_input_const = { true, false };
  op_desc->SetIsInputConst(is_input_const);
  auto node = graph->AddNode(op_desc);

  TbeOpTask task;
  ASSERT_EQ(task.GetOpType(), "");
  task.op_desc_ = op_desc;
  task.node_ = node;
  ASSERT_EQ(task.SetArgIndex(), SUCCESS);
  task.arg_size_ = sizeof(void *) * 4;
  task.args_.reset(new (std::nothrow) uint8_t[task.arg_size_]);
  task.arg_index_ = {0};
  task.input_num_ = 2;
  task.output_num_ = 1;
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.arg_size_;

  vector<void *> args;
  vector<DataBuffer> inputs;
  vector<DataBuffer> outputs;
  ASSERT_EQ(task.UpdateIoAddr(inputs, outputs), ACL_ERROR_GE_PARAM_INVALID);

  ge::DataBuffer data_buffer;
  inputs = { data_buffer };
  outputs = { data_buffer };
  ASSERT_EQ(task.UpdateIoAddr(inputs, outputs), SUCCESS);

  task.workspaces_ = { (void *)0x0002 };
  ASSERT_EQ(task.UpdateTilingArgs(), SUCCESS);
}

TEST_F(UtestSingleOpTask, test_atomic_exec) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);

  auto node = graph->AddNode(op_desc);
  AtomicAddrCleanOpTask task;
  ASSERT_EQ(task.GetOpType(), "DynamicAtomicAddrClean");
  task.op_desc_ = op_desc;
  task.node_ = node;

  vector<DataBuffer> inputs;
  vector<DataBuffer> outputs;
  std::vector<int64_t> atomic_output_indices;
  ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ASSERT_EQ(task.InitAtomicAddrCleanIndices(), INTERNAL_ERROR);
  atomic_output_indices = { 0 };
  ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  ASSERT_EQ(task.InitAtomicAddrCleanIndices(), INTERNAL_ERROR);
  task.arg_size_ = sizeof(void *) * 2;
  task.args_.reset(new (std::nothrow) uint8_t[task.arg_size_]);
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.arg_size_;
  ASSERT_NE(task.UpdateIoAddr(inputs, outputs), SUCCESS);

  ge::DataBuffer data_buffer;
  outputs = { data_buffer };
  ASSERT_EQ(task.UpdateIoAddr(inputs, outputs), SUCCESS);

  ASSERT_EQ(task.UpdateTilingArgs(), SUCCESS);

  task.run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task.run_info_, nullptr);
  task.CalcTilingInfo();
}


TEST_F(UtestSingleOpTask, test_atomic_init) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);

  map<string, map<int64_t, int64_t>> workspace_info;
  map<int64_t, int64_t> workspace_info_pair;
  workspace_info_pair.insert(std::make_pair(INT32_MAX, 1));
  workspace_info.insert(std::make_pair("1", workspace_info_pair));
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO,workspace_info);

  auto node = graph->AddNode(op_desc);
  AtomicAddrCleanOpTask task;
  task.op_desc_ = op_desc;
  task.node_ = node;

  vector<DataBuffer> inputs;
  vector<DataBuffer> outputs;
  std::vector<int64_t> atomic_output_indices;
  ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ASSERT_NE(task.InitAtomicAddrCleanIndices(), SUCCESS);

}

UINT32 StubTilingNeedAtomic(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  size_t *workspace_size = context->GetWorkspaceSizes(2);
  *workspace_size = 66U;
  *(workspace_size + 1) = 88U;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParse(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreator() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}

NodePtr CreateNodeWithSoftSyncOp(ComputeGraphPtr graph, const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  auto funcs = space_registry->CreateOrGetOpImpl(RELU6);

  funcs->tiling = StubTilingNeedAtomic;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;

  auto op_desc = make_shared<OpDesc>("relu6", RELU6);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);
  return node;
}

TEST_F(UtestSingleOpTask, test_soft_sync_op) {
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto graph = make_shared<ComputeGraph>("graph");
  auto node = CreateNodeWithSoftSyncOp(graph, space_registry);
  TbeOpTask tbe_task(node);
  tbe_task.op_desc_ = node->GetOpDesc();
  tbe_task.node_ = node;
  tbe_task.space_registries_ = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  tbe_task.space_registries_->at(static_cast<size_t>(OppImplVersion::kOpp)) = space_registry;
  tbe_task.run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  tbe_task.args_ = MakeUnique<uint8_t[]>(512 + kMaxHostMemInputLen);
  tbe_task.arg_size_ = 512 + kMaxHostMemInputLen;
  tbe_task.args_ex_.args = tbe_task.args_.get();
  tbe_task.args_ex_.argsSize = tbe_task.arg_size_;
  auto stream_resource = MakeUnique<StreamResource>(0);
  stream_resource->allocator_ = &stream_resource->internal_allocator_;
  tbe_task.stream_resource_ = stream_resource.get();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);

  EXPECT_EQ(tbe_task.LaunchKernel(stream), 0);
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOpTask, TestSoftSyncExecute_SaveExceptionDumper_WithExceptionDumperOn) {
  ge::diagnoseSwitch::EnableExceptionDump();
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  const auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  auto graph = make_shared<ComputeGraph>("graph");
  auto node = CreateNodeWithSoftSyncOp(graph, space_registry);
  TbeOpTask tbe_task(node);
  tbe_task.op_desc_ = node->GetOpDesc();
  tbe_task.node_ = node;
  tbe_task.space_registries_ = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  tbe_task.space_registries_->at(static_cast<size_t>(OppImplVersion::kOpp)) = space_registry;
  tbe_task.run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  tbe_task.args_ = MakeUnique<uint8_t[]>(512 + kMaxHostMemInputLen);
  tbe_task.arg_size_ = 512 + kMaxHostMemInputLen;
  tbe_task.args_ex_.args = tbe_task.args_.get();
  tbe_task.args_ex_.argsSize = tbe_task.arg_size_;
  std::cout<<"test size"<<tbe_task.args_ex_.argsSize<<std::endl;
  auto stream_resource = MakeUnique<StreamResource>(0);
  stream_resource->allocator_ = &stream_resource->internal_allocator_;
  tbe_task.stream_resource_ = stream_resource.get();

  ge::DumpStub::GetInstance().ClearOpInfos();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);

  EXPECT_EQ(tbe_task.LaunchKernel(stream), 0);
  tbe_task.PostProcess(stream);
  EXPECT_FALSE(ge::DumpStub::GetInstance().GetOpInfos().empty());
  const auto &op_info = ge::DumpStub::GetInstance().GetOpInfos()[0];
  std::string tiling_key = AdxGetTilingKey(op_info);
  EXPECT_EQ(tiling_key, "666");
  EXPECT_EQ(AdxGetAdditionalInfo(op_info, "is_host_args"), "true");

  ge::DumpStub::GetInstance().ClearOpInfos();
  rtStreamDestroy(stream);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(UtestSingleOpTask, test_aicpu_task_launch_kernel) {
  AiCpuCCTask task;
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);
  task.num_inputs_ = 2;
  task.num_outputs_ = 1;
  task.input_is_const_ = {1, 0};
  int total_addr = 3;
  uint32_t* addrs[total_addr] = {nullptr, nullptr, nullptr};
  task.io_addr_ = reinterpret_cast<uintptr_t*>(addrs);
  task.io_addr_num_ = total_addr;
  vector<DataBuffer> outputs(1, DataBuffer());
  outputs[0].data = 0;
  task.unknown_type_ = ge::DEPEND_COMPUTE;
  ASSERT_EQ(task.InitForSummaryAndCopy(), SUCCESS);
  auto &summary = task.output_summary_host_[0];
  summary.shape_data_ptr = 0;
  summary.shape_data_size = 1;
  summary.raw_data_ptr = 0;
  summary.raw_data_size = 1;
  void *shape_buffer = nullptr;
  rtMalloc(&shape_buffer, 1, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  task.out_shape_hbm_.emplace_back(shape_buffer);
  task.memcpy_so_name_ = "libcpu_kernel.so";
  task.memcpy_kernel_name_ = "RunCpuKernel";
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 3;
  domi::TaskDef task_def;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);
  auto &memcpy_args = kernel_def->args();
  task.memcpy_args_size_ = kernel_def->args_size();
  task.memcpy_args_.reset(new(std::nothrow) uint8_t[task.memcpy_args_size_]());
  memcpy_s(task.memcpy_args_.get(), task.memcpy_args_size_, memcpy_args.c_str(), memcpy_args.size());
  ASSERT_EQ(task.CopyDataToHbm(outputs, stream), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOpTask, test_aicpu_task_update_io_addr) {
  AiCpuCCTask task;
  task.num_inputs_ = 2;
  task.num_outputs_ = 1;
  task.input_is_const_ = {1, 0};
  int total_addr = 3;
  uint32_t* addrs[total_addr] = {nullptr, nullptr, nullptr};
  task.io_addr_ = reinterpret_cast<uintptr_t*>(addrs);
  task.io_addr_num_ = total_addr;

  {
    vector<DataBuffer> inputs(1, DataBuffer());
    vector<DataBuffer> outputs(1, DataBuffer());
    auto ret = task.UpdateIoAddr(inputs, outputs);
    ASSERT_EQ(ret, SUCCESS);
    ASSERT_EQ(addrs[0], nullptr);
    ASSERT_EQ(addrs[1], nullptr);
    ASSERT_EQ(addrs[2], nullptr);
  }

  {
    uint32_t data_buf[2];
    vector<DataBuffer> inputs{DataBuffer(&data_buf[0], 4, false)};
    vector<DataBuffer> outputs{DataBuffer(&data_buf[1], 4, false)};
    auto ret = task.UpdateIoAddr(inputs, outputs);
    ASSERT_EQ(ret, SUCCESS);
    ASSERT_EQ(addrs[0], nullptr);
    ASSERT_EQ(addrs[1], &data_buf[0]);
    ASSERT_EQ(addrs[2], &data_buf[1]);
  }

  {
    uint32_t data_buf[2];
    vector<DataBuffer> inputs{DataBuffer(nullptr, 4, false)};
    vector<DataBuffer> outputs{DataBuffer(&data_buf[1], 4, false)};
    auto ret = task.UpdateIoAddr(inputs, outputs);
    ASSERT_EQ(ret, PARAM_INVALID);
  }

  {
    uint32_t data_buf[2];
    vector<DataBuffer> inputs{DataBuffer(&data_buf[0], 4, false)};
    vector<DataBuffer> outputs{DataBuffer(nullptr, 4, false)};
    auto ret = task.UpdateIoAddr(inputs, outputs);
    ASSERT_EQ(ret, PARAM_INVALID);
  }
}

TEST_F(UtestSingleOpTask, test_blocking_aicpu_op_01) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  auto op_desc = make_shared<OpDesc>("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  AiCpuCCTask aicpu_task;
  aicpu_task.SetOpDesc(op_desc);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);

  ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), SUCCESS);
  ASSERT_EQ(aicpu_task.LaunchKernel(stream), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOpTask, test_blocking_aicpu_op_02) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  auto op_desc = make_shared<OpDesc>("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  AiCpuTask aicpu_task;
  aicpu_task.SetOpDesc(op_desc);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);

  ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), SUCCESS);
  ASSERT_EQ(aicpu_task.LaunchKernel(stream), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOpTask, test_blocking_aicpu_op_fail) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  auto op_desc = make_shared<OpDesc>("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 0), RT_ERROR_NONE);

  {
    AiCpuTask aicpu_task;
    aicpu_task.SetOpDesc(op_desc);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), SUCCESS);
    ASSERT_EQ(aicpu_task.LaunchKernel(stream), SUCCESS);

    RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_task.LaunchKernel(stream), FAILED);
  }

  {
    AiCpuTask aicpu_task;
    aicpu_task.SetOpDesc(op_desc);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtStreamWaitEventWithTimeout, rtError_t, 0x78000001);
    RTS_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_task.LaunchKernel(stream), FAILED);
  }

  {
    AiCpuTask aicpu_task;
    aicpu_task.SetOpDesc(op_desc);
    ASSERT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtEventReset, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_task.LaunchKernel(stream), FAILED);
  }

  {
    AiCpuTask aicpu_task;
    aicpu_task.SetOpDesc(op_desc);
    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
    EXPECT_EQ(aicpu_task.SetExtInfoAndType(kernel_def.kernel_ext_info(), 0), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
    EXPECT_EQ(aicpu_task.LaunchKernel(stream), SUCCESS);
  }

  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOpTask, test_update_node_by_shape) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  vector<int64_t> shape{16, 16};
  GeShape ge_shape(shape);
  GeTensorDesc in_desc(ge_shape);
  GeTensorDesc out_desc(ge_shape);
  vector<ge::GeTensorDesc> input_desc;
  vector<ge::GeTensorDesc> output_desc;

  input_desc.emplace_back(in_desc);
  output_desc.emplace_back(out_desc);

  op_desc->AddInputDesc(in_desc);
  op_desc->AddOutputDesc(out_desc);
  auto node = graph->AddNode(op_desc);

  TbeOpTask tbe_task;
  tbe_task.node_ = node;
  tbe_task.arg_size_ = sizeof(void *) * 1;
  tbe_task.args_.reset(new (std::nothrow) uint8_t[tbe_task.arg_size_]);
  ASSERT_EQ(tbe_task.UpdateNodeByShape(input_desc, output_desc), SUCCESS);
}
TEST_F(UtestSingleOpTask, test_update_ioaddr_with_host_input) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");

  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  vector<bool> is_input_const = { true, false };
  op_desc->SetIsInputConst(is_input_const);
  auto node = graph->AddNode(op_desc);

  TbeOpTask task;
  task.op_desc_ = op_desc;
  task.node_ = node;

  vector<DataBuffer> inputs;
  vector<DataBuffer> outputs;

  ge::DataBuffer data_buffer;
  data_buffer.placement = 1;
  data_buffer.length = 65;
  data_buffer.data = new uint8_t[data_buffer.length];

  inputs = { data_buffer };
  outputs = { data_buffer };

  uint32_t arg_size_without_tiling = 32;
  uint32_t host_mem_ext_size = 64;
  task.max_tiling_size_ = 32;
  uint32_t arg_size = arg_size_without_tiling + task.max_tiling_size_ + host_mem_ext_size;
  task.arg_size_ = arg_size;

  task.args_.reset(new (std::nothrow) uint8_t[task.arg_size_]);
  task.arg_index_ = {0};

  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = arg_size;

  task.args_ex_.tilingAddrOffset = arg_size_without_tiling - sizeof(void *);
  task.args_ex_.tilingDataOffset = arg_size_without_tiling;
  task.need_host_mem_opt_ = true;

  ASSERT_EQ(task.UpdateIoAddr(inputs, outputs), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(data_buffer.data);

  data_buffer.length = 64;
  data_buffer.data = new uint8_t[data_buffer.length];
  inputs[0] = data_buffer;
  ASSERT_EQ(task.UpdateIoAddr(inputs, outputs), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(data_buffer.data);
}
