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

#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/fe/kernel_task_info.h"
#include "graph/load/model_manager/task_info/fe/super_kernel_task_info.h"
#include "graph/load/model_manager/task_info/hccl/hccl_task_info.h"
#include "graph/load/model_manager/task_info/hccl/hccl_util.h"
#include "graph/load/model_manager/task_info/fe/fusion_task_info.h"
#include "depends/runtime/src/runtime_stub.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "framework/common/types.h"
#include "graph/manager/graph_var_manager.h"
#include "depends/profiler/src/dump_stub.h"
#include "graph/manager/graph_mem_allocator.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "graph/utils/op_desc_utils.h"
#include "faker/space_registry_faker.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "aicpu_task_struct.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "register/hidden_inputs_func_registry.h"
#include "common/share_graph.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_utils.h"
#include "graph/manager/mem_manager.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
const std::string kAttrNameAtomicWspMode = "wspMode";
const std::string kWspFoldedMode = "folded";
constexpr uint32_t kUBAlignedLen = 32UL;

HcclResult InitializeHeterogeneousRuntime(const std::string &group, void *tilingData, void *ccuTaskGroup) {
  return HCCL_SUCCESS;
}
}
class UtestKernelTaskInfo : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

// test KernelTaskInfo Init.
TEST_F(UtestKernelTaskInfo, success_kernel_taskInfo_not_te) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint64_t input_addr = (uint64_t)malloc(1024);
  uint64_t output_addr = (uint64_t)malloc(1024);
  uint64_t workspace_addr = (uint64_t)malloc(1024);
  iow_addrs.input_logic_addrs = {{input_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  iow_addrs.workspace_logic_addrs = {{workspace_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task->type()));

  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  domi::KernelDef *kernel_def = task->mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  const auto op_desc = CreateOpDesc("relu", RELU);
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  ctx->set_op_index(op_desc->GetId());

  EXPECT_EQ(task_info->Init(*task, &model, args, persistant_workspace, iow_addrs), FAILED);

  kernel_def->set_block_dim(10);
  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);

  ctx->set_kernel_type(0);
  EXPECT_EQ(task_info->Init(*task, &model, args, persistant_workspace, iow_addrs), FAILED);

  task_info->Release();
  free((void*)(args[0].dev_addr));
  free((void*)(input_addr));
  free((void*)(output_addr));
  free((void*)(workspace_addr));
}

TEST_F(UtestKernelTaskInfo, success_init_kernel_task_info_fail) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint64_t input_addr = (uint64_t)malloc(1024);
  uint64_t output_addr = (uint64_t)malloc(1024);
  uint64_t workspace_addr = (uint64_t)malloc(1024);
  iow_addrs.input_logic_addrs = {{input_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  iow_addrs.workspace_logic_addrs = {{workspace_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  KernelTaskInfo kernel_task_info;
  domi::TaskDef task_def;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();

  const auto op_desc = CreateOpDesc("relu", RELU);
  model.op_list_[op_desc->GetId()] = op_desc;
  ctx->set_op_index(op_desc->GetId());
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  // Failed by rtGetFunctionByName.
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), FAILED);
  free((void*)(args[0].dev_addr));
  free((void*)(input_addr));
  free((void*)(output_addr));
  free((void*)(workspace_addr));
}

// test kernel_ex_task_Release
TEST_F(UtestKernelTaskInfo, init_task_tvm) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint64_t input_addr = (uint64_t)malloc(1024);
  uint64_t fm_base_addr = (uint64_t)malloc(1024 + 1024);
  uint64_t output_addr = fm_base_addr;
  uint64_t workspace_addr = fm_base_addr + 1024;
  iow_addrs.input_logic_addrs = {{input_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  iow_addrs.workspace_logic_addrs = {{workspace_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, fm_base_addr,
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  const auto op_desc = CreateOpDesc("relu", RELU);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetInputOffset({8});
  op_desc->SetOutputOffset({8});
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");
  {
    KernelTaskInfo kernel_task_info;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    int64_t op_index = kernel_task_info.ParseOpIndex(task_def);
    EXPECT_EQ(op_index, op_desc->GetId());

    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    EXPECT_EQ(kernel_task_info.input_data_addrs_[0], input_addr);
    EXPECT_EQ(kernel_task_info.output_data_addrs_[0], output_addr);
    EXPECT_EQ(kernel_task_info.workspace_addrs_[0], workspace_addr);
    EXPECT_EQ(kernel_task_info.input_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFix);
    EXPECT_EQ(kernel_task_info.output_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap);
    EXPECT_EQ(kernel_task_info.workspace_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap);
  }

  task_def.clear_kernel();
  model.runtime_param_.mem_base = 0U;
  free((void*)(args[0].dev_addr));
  free((void*)(input_addr));
  free((void*)(fm_base_addr));
}

TEST_F(UtestKernelTaskInfo, init_task_tvm_zero_copy_var_input) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint64_t input_addr = (uint64_t)malloc(1024);
  uint64_t fm_base_addr = (uint64_t)malloc(1024 + 1024);
  uint64_t output_addr = fm_base_addr;
  uint64_t workspace_addr = fm_base_addr + 1024;
  iow_addrs.input_logic_addrs = {{input_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  iow_addrs.workspace_logic_addrs = {{workspace_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, fm_base_addr,
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  const auto op_desc = CreateOpDesc("relu", RELU);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetInputOffset({8});
  op_desc->SetOutputOffset({8});
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");
  {
    KernelTaskInfo kernel_task_info;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    int64_t op_index = kernel_task_info.ParseOpIndex(task_def);
    EXPECT_EQ(op_index, op_desc->GetId());

    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    model.runtime_param_.session_id = 0;
    ZeroCopyOffset zero_copy_offset;
    std::map<uintptr_t, std::vector<uintptr_t>> addr_map;
    addr_map[input_addr].emplace_back(0);
    zero_copy_offset.outside_addrs_.emplace_back(addr_map);
    model.input_data_info_[0] = zero_copy_offset;
    model.runtime_param_.var_size = 1024;
    VarManager::Instance(0)->var_resource_ = make_shared<ge::VarResource>(0);
    VarManager::Instance(0)->var_resource_->var_offset_map_[8] = 1;
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    EXPECT_EQ(kernel_task_info.input_data_addrs_[0], input_addr);
    EXPECT_EQ(kernel_task_info.output_data_addrs_[0], output_addr);
    EXPECT_EQ(kernel_task_info.workspace_addrs_[0], workspace_addr);
    EXPECT_EQ(kernel_task_info.input_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFix);
    EXPECT_EQ(kernel_task_info.output_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap);
    EXPECT_EQ(kernel_task_info.workspace_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap);
  }

  task_def.clear_kernel();
  model.runtime_param_.mem_base = 0U;
  free((void*)(args[0].dev_addr));
  free((void*)(input_addr));
  free((void*)(fm_base_addr));
}

TEST_F(UtestKernelTaskInfo, init_task_tvm_and_memset) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  model.SetFeatureBaseRefreshable(true);
  const auto op_desc = CreateOpDesc("relu", RELU);

  domi::ModelTaskDef model_task_def;
  // add task1
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length
  model.op_list_[op_desc->GetId()] = op_desc;
  model.operator_list_[op_desc->GetId()] = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");

  // add task2
  auto memset_op = CreateOpDesc("memset", MEMSET);
  domi::TaskDef &memset_task_def = *model_task_def.add_task();
  domi::KernelDef *memset_kernel_def = memset_task_def.mutable_kernel();
  memset_kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  memset_kernel_def->mutable_context()->set_op_index(memset_op->GetId());
  memset_kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t memset_offset = 24U;
  memset_kernel_def->mutable_context()->set_args_offset(&memset_offset, sizeof(uint16_t));
  std::vector<char> args_info_memset(32U, '0');
  memset_kernel_def->set_args_size(args_info_memset.size());
  memset_kernel_def->set_args(args_info_memset.data(), args_info_memset.size());
  memset_op->SetWorkspace({1458});   // offset
  memset_op->SetWorkspaceBytes({150});    // length
  EXPECT_EQ(ge::AttrUtils::SetStr(memset_op, kAttrNameAtomicWspMode, kWspFoldedMode), true);
  model.op_list_[memset_op->GetId()] = memset_op;
  model.operator_list_[memset_op->GetId()] = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(memset_op));
  TBEKernelPtr memset_kernel_handle = MakeShared<OpKernelBin>(memset_op->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(memset_op->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, memset_kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(memset_op, memset_op->GetName() + "_kernelname", memset_op->GetName()));
  AttrUtils::SetStr(memset_op, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(memset_op, ATTR_NAME_KERNEL_BIN_ID, "te_memset_123");

  {
    KernelTaskInfo kernel_task_info;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    EXPECT_EQ(kernel_task_info.args_size_, args_info.size());
    free(ValueToPtr(args[0].dev_addr));
  }
  {
    KernelTaskInfo kernel_task_info;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(memset_task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(memset_task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    // folded mode should add 8 bytes of tail + 32U aligned.
    EXPECT_EQ(kernel_task_info.args_size_, args_info_memset.size() + sizeof(uint64_t) + 32U);
    free(ValueToPtr(args[0].dev_addr));
  }

  task_def.clear_kernel();
  model.runtime_param_.mem_base = 0U;
}

// test kernel_ex_task_Release
TEST_F(UtestKernelTaskInfo, init_task_tvm_known) {
  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  const auto op_desc = CreateOpDesc("relu", RELU);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  free(ValueToPtr(args[0].dev_addr));
  task_def.clear_kernel();
  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestKernelTaskInfo, init_tvm_task_info_with_te_kernel_type) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint64_t input_addr = (uint64_t)malloc(1024);
  uint64_t output_addr = (uint64_t)malloc(1024);
  uint64_t workspace_addr = (uint64_t)malloc(1024);
  iow_addrs.input_logic_addrs = {{input_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  iow_addrs.workspace_logic_addrs = {{workspace_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  // DavinciModel is nullptr
  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.Init(task_def, nullptr, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;

  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);
  kernel_def->set_sm_desc(&l2CtrlInfo, sizeof(rtSmDesc_t));
  kernel_def->set_flowtable("fl", 2);
  kernel_def->set_block_dim(10);

  // Failed: GetOpByIndex
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  ctx->set_op_index(op_desc->GetId() + 4);
  ctx->set_args_offset("\0\0"); // args_offset = 0
  EXPECT_NE(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  // Failed: if ((context.args_offset().size() / sizeof(uint16_t)) < 1U)
  ctx->clear_op_index();
  ctx->set_op_index(op_desc->GetId());
  ctx->clear_args_offset();
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), FAILED);

  // Failed: args_size_ <= io_addr_offset_
  ctx->set_args_offset("args111111", 10);
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();

  free((void*)(args[0].dev_addr));
  free((void*)(input_addr));
  free((void*)(output_addr));
  free((void*)(workspace_addr));
}

// test InitAICPUCustomTask with kernel_type is CUSTOMIZED
TEST_F(UtestKernelTaskInfo, init_kernel_task_info_with_customized_kernel_type) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;

  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);
  kernel_def->set_sm_desc(&l2CtrlInfo, sizeof(rtSmDesc_t));
  kernel_def->set_flowtable("fl", 2);
  kernel_def->set_block_dim(10);

  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(3);
  ctx->set_op_index(4);
  ctx->set_args_offset("\0\0"); // args_offset = 0
  EXPECT_NE(kernel_task_info.Init(task_def, &model), SUCCESS);

  ctx->clear_args_offset();
  ctx->set_args_offset("args111111", 10);
  EXPECT_NE(kernel_task_info.Init(task_def, &model), SUCCESS);

  ctx->clear_args_offset();
  ctx->set_op_index(op_desc->GetId());

  const char task[] = "opattr";
  AttrUtils::SetBytes(op_desc, ATTR_NAME_OPATTR, Buffer::CopyFrom((uint8_t *)task, sizeof(task)));
  EXPECT_NE(kernel_task_info.Init(task_def, &model), SUCCESS);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  model.op_list_[op_desc->GetId()] = op_desc;

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_args_offset("\0\0");
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.args_ = (void*)malloc(256);

  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(op_desc, *kernel_def), PARAM_INVALID);
  free(kernel_task_info.args_);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  context->clear_args_offset();
  context->set_args_offset("args111111", 10);
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(op_desc, *kernel_def), FAILED);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed2) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  model.op_list_[op_desc->GetId()] = op_desc;

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.args_ = (void*)malloc(256);

  context->set_args_offset("\0\0");
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  // AttrUtils::GetBytes  -> true
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(op_desc, *kernel_def), PARAM_INVALID);
  free(kernel_task_info.args_);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed3) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  model.op_list_[op_desc->GetId()] = op_desc;

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.args_ = (void*)malloc(256);

  context->set_args_offset("\0\0");
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(op_desc, *kernel_def), PARAM_INVALID);
  free(kernel_task_info.args_);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();

  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  string args;
  args.append(100, '1');
  kernel_def.set_so_name("libDvpp.so");
  kernel_def.set_kernel_name("DvppResize");
  kernel_def.set_args(args.data(), 100);
  kernel_def.set_args_size(100);

  domi::KernelContext &ctx = *kernel_def.mutable_context();
  ctx.set_kernel_type(6);
  ctx.set_op_index(op_desc->GetId());

  // No need Call ParseTaskRunParam for Test task, Just use default value of args_offset_.
  {
    // ModelUtils::GetInputDataAddrs  -> ok
    // ModelUtils::GetOutputDataAddrs -> ok
    // rtMalloc -> RT_ERROR_NONE
    // rtMemcpy -> RT_ERROR_NONE
    KernelTaskInfo kernel_task_info;
    kernel_task_info.task_type_ = static_cast<ModelTaskType>(task_def.type());
    kernel_task_info.args_size_ = 120;
    kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);

    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    int64_t host_data[1024] = {0};
    args[0].host_addr = host_data;
    args[0].len = 1024;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    domi::GetContext().is_online_model = true;
    EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
    EXPECT_TRUE(kernel_task_info.IsSupportReDistribute());
    EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
    EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
    domi::GetContext().is_online_model = false;
    free(ValueToPtr(args[0].dev_addr));
  }

  kernel_def.clear_context();
  task_def.clear_kernel();
}


TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type_host_only) {
  DavinciModel model(0, nullptr);
//  model.runtime_param_.mem_size = 2048U;
//  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
//  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemInfo host_svm_mem_info{};
  host_svm_mem_info.memory_size = 2048U;
  host_svm_mem_info.logic_memory_base = kMemoryHostSVMFeatureMapLogicBase;
  host_svm_mem_info.memory_type = RT_MEMORY_HOST_SVM;
  host_svm_mem_info.memory_key = "_svm";
  host_svm_mem_info.memory_base =
      PtrToPtr<void, uint8_t>(MemoryAllocator(RT_MEMORY_HOST_SVM).MallocMemory(host_svm_mem_info.memory_key, host_svm_mem_info.memory_size));
  model.runtime_param_.host_mem_base = PtrToValue(host_svm_mem_info.memory_base);
  model.runtime_param_.memory_infos[RT_MEMORY_HOST_SVM] = std::move(host_svm_mem_info);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp", 1, 1);
  op_desc->SetId(0);
  EXPECT_TRUE(ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST_SVM}));
  EXPECT_TRUE(ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST_SVM}));
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();

  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  string args;
  args.append(100, '1');
  kernel_def.set_so_name("libDvpp.so");
  kernel_def.set_kernel_name("DvppResize");
  kernel_def.set_args(args.data(), 100);
  kernel_def.set_args_size(100);

  domi::KernelContext &ctx = *kernel_def.mutable_context();
  ctx.set_kernel_type(6);
  ctx.set_op_index(op_desc->GetId());

  // No need Call ParseTaskRunParam for Test task, Just use default value of args_offset_.
  MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  {
    // ModelUtils::GetInputDataAddrs  -> ok
    // ModelUtils::GetOutputDataAddrs -> ok
    // rtMalloc -> RT_ERROR_NONE
    // rtMemcpy -> RT_ERROR_NONE
    KernelTaskInfo kernel_task_info;
    kernel_task_info.deploy_type_flag_ = RT_KERNEL_HOST_ONLY;
    kernel_task_info.task_type_ = static_cast<ModelTaskType>(task_def.type());
    kernel_task_info.args_size_ = 120;
    kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    int64_t host_data[1024] = {0};
    args[0].len = 1024;
    args[0].host_addr = host_data;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
    EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
    free(ValueToPtr(args[0].dev_addr));
  }

  kernel_def.clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type_fail) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();

  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  string args;
  args.append(100, '1');
  kernel_def.set_so_name("libDvpp.so");
  kernel_def.set_kernel_name("DvppResize");
  kernel_def.set_args(args.data(), 100);
  kernel_def.set_args_size(100);

  domi::KernelContext &ctx = *kernel_def.mutable_context();
  ctx.set_kernel_type(6);
  ctx.set_op_index(op_desc->GetId());

  {
    KernelTaskInfo kernel_task_info;
    kernel_task_info.args_size_ = 120;
    kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    int64_t host_data[1024] = {0};
    args[0].len = 1024;
    args[0].host_addr = host_data;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    // ModelUtils::GetInputDataAddrs  -> ok
    // ModelUtils::GetOutputDataAddrs -> ok
    // rtMalloc -> RT_ERROR_NONE
    // rtMemcpy -> RT_ERROR_INVALID_VALUE
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    const string ext_info = {1, 1, 1, 1, 0, 0, 0, 0};
    EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(ext_info), SUCCESS);

    EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
    EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
    free(ValueToPtr(args[0].dev_addr));
  }

  kernel_def.clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type_fail2) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();

  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  string args;
  args.append(100, '1');
  kernel_def.set_so_name("libDvpp.so");
  kernel_def.set_kernel_name("DvppResize");
  kernel_def.set_args(args.data(), 100);
  kernel_def.set_args_size(100);

  domi::KernelContext *ctx = kernel_def.mutable_context();
  ctx->set_kernel_type(6);
  ctx->set_op_index(op_desc->GetId());

  {
    KernelTaskInfo kernel_task_info;
    kernel_task_info.args_size_ = 120;
    kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    int64_t host_data[1024] = {0};
    args[0].len = 1024;
    args[0].host_addr = host_data;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    // ModelUtils::GetInputDataAddrs  -> ok
    // ModelUtils::GetOutputDataAddrs -> ok
    // rtMalloc -> RT_ERROR_INVALID_VALUE
    // rtMemcpy -> RT_ERROR_NONE
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
    EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
    free(ValueToPtr(args[0].dev_addr));
  }

  kernel_def.clear_context();
  task_def.clear_kernel();
}

// test StoreInputOutputTensor failed
TEST_F(UtestKernelTaskInfo, store_input_output_tensor_fail) {
  std::vector<uint64_t> input_data_addrs;
  std::vector<uint64_t> output_data_addrs;
  std::vector<ccAICPUTensor> input_descs;
  std::vector<ccAICPUTensor> output_descs;

  KernelTaskInfo kernel_task_info;
  kernel_task_info.args_ = (void*)malloc(256);
  // rtMalloc -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);
  free(kernel_task_info.args_);
}


TEST_F(UtestKernelTaskInfo, store_input_output_tensor_fail2) {
  std::vector<uint64_t> input_data_addrs;
  std::vector<uint64_t> output_data_addrs;
  std::vector<ccAICPUTensor> input_descs;
  std::vector<ccAICPUTensor> output_descs;

  KernelTaskInfo kernel_task_info;
  kernel_task_info.args_ = (void*)malloc(256);
  // rtMalloc -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);
  free(kernel_task_info.args_);
}

TEST_F(UtestKernelTaskInfo, distribute_success) {
  KernelTaskInfo kernel_task_info;
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  kernel_task_info.operator_ = operator_info;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.func_handle_ = (void *)0x12000;
  domi::TaskDef task_def;
  // rtModelGetTaskId -> RT_ERROR_INVALID_VALUE
  rtModel_t rt_model_handle = (rtModel_t *)0x12345678;
  model.rt_model_handle_ = rt_model_handle;

  // Failed for SetStream
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), FAILED);

  // rtKernelLaunchWithFlag -> RT_ERROR_INVALID_VALUE
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(kernel_task_info.IsSupportReDistribute());
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  model.rt_model_handle_ = nullptr;
}

// test success DistributeDumpTask
TEST_F(UtestKernelTaskInfo, success_distribute_dump_task) {
  DavinciModel model(0, nullptr);
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  domi::TaskDef task_def;
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();
  domi::KernelContext &context_def = *kernel_def.mutable_context();
  context_def.set_op_index(op_desc->GetId());

  kernel_def.set_stub_func("kerneltaskinfo");
  kernel_def.set_block_dim(10);
  kernel_def.set_args("args111111", 10);
  kernel_def.set_args_size(10);
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;
  kernel_def.set_sm_desc((void *)&l2CtrlInfo, sizeof(rtSmDesc_t));

  // for SetStream
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  // Failed for origin_op_index_size
  for (uint32_t i = 0U; i <= CC_FUSION_OP_MAX; ++i) {
    context_def.add_origin_op_index(i);
  }
  EXPECT_NE(kernel_task_info.Init(task_def, &model), SUCCESS);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  task_def.clear_kernel();
}

// test success GetTaskID
TEST_F(UtestKernelTaskInfo, success_get_task_id) {
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task->type()));

  EXPECT_EQ(task_info->GetTaskID(), 0);

  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.GetTaskID(), 0);

  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.GetTaskID(), 0);
}

// test StoreInputOutputTensor success
TEST_F(UtestKernelTaskInfo, success_store_input_output_tensor) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  kernel_task_info.args_ = (void*)malloc(256);

  std::vector<uint64_t> input_data_addrs;
  std::vector<uint64_t> output_data_addrs;
  std::vector<ccAICPUTensor> input_descs;
  std::vector<ccAICPUTensor> output_descs;

  int test = 1;
  int *addr = &test;
  void *input = addr;
  void *output = addr;
  input_data_addrs.push_back(PtrToValue(input));
  output_data_addrs.push_back(PtrToValue(output));

  ccAICPUTensor input_desc;
  ccAICPUTensor output_desc;
  input_descs.push_back(input_desc);
  output_descs.push_back(output_desc);

  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);
  free(kernel_task_info.args_);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
}

// test KernelTaskInfo release fail
TEST_F(UtestKernelTaskInfo, fail_release) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  kernel_task_info.args_ = (void*)malloc(256);

  std::vector<uint64_t> input_data_addrs;
  std::vector<uint64_t> output_data_addrs;
  std::vector<ccAICPUTensor> input_descs;
  std::vector<ccAICPUTensor> output_descs;

  int test = 1;
  int *addr = &test;
  void *input = addr;
  void *output = addr;
  input_data_addrs.push_back(PtrToValue(input));
  output_data_addrs.push_back(PtrToValue(output));

  ccAICPUTensor input_desc;
  ccAICPUTensor output_desc;
  input_descs.push_back(input_desc);
  output_descs.push_back(output_desc);

  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);

  // rtMemFreeManaged -> RT_ERROR_INVALID_VALUE
  free(kernel_task_info.args_);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
}

// test fusion_end_task Init
TEST_F(UtestKernelTaskInfo, kernel_task_info_init_success) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint64_t input_addr = (uint64_t)malloc(1024);
  uint64_t output_addr = (uint64_t)malloc(1024);
  uint64_t workspace_addr = (uint64_t)malloc(1024);
  iow_addrs.input_logic_addrs = {{input_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  iow_addrs.workspace_logic_addrs = {{workspace_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  auto model_def = MakeShared<domi::ModelTaskDef>();

  model.model_id_ = 1;
  model.name_ = "test";
  model.version_ = 0x01;

  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetModelTaskDef(model_def);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  const auto op_desc = CreateOpDesc("data", DATA);
  op_desc->SetId(0);
  op_desc->SetInputOffset({1});
  op_desc->SetOutputOffset({100});

  GeTensorDesc descin(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(descin, 4);
  op_desc->AddInputDesc(descin);
  GeTensorDesc descout(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  TensorUtils::SetSize(descout, 32);
  op_desc->AddOutputDesc(descout);
  op_desc->SetId(0);

  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_op_index(op_desc->GetId());
  vector<string> original_op_names = { "conv", "add" };
  AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);

  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), FAILED);

  free((void*)(args[0].dev_addr));
  free((void*)(input_addr));
  free((void*)(output_addr));
  free((void*)(workspace_addr));
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_unfolded_te) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  const auto test_size = "test_size";
  kernel_def->set_args(test_size);
  kernel_def->set_sm_desc("hello");
  auto op_desc = std::make_shared<OpDesc>("add", "Add");
  model.op_list_[op_desc->GetId()] = op_desc;
  model.SetFeatureBaseRefreshable(true);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_no_sm_desc_unfolded_te) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  const auto test_size = "test_size";
  kernel_def->set_args(test_size);
  auto op_desc = std::make_shared<OpDesc>("add", "Add");
  model.op_list_[op_desc->GetId()] = op_desc;
  model.SetFeatureBaseRefreshable(true);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_folded_te) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  const auto test_size = "test_size";
  kernel_def->set_args(test_size);
  kernel_def->set_sm_desc("hello");
  auto op_desc = std::make_shared<OpDesc>("add", MEMSET);
  EXPECT_EQ(ge::AttrUtils::SetStr(op_desc, kAttrNameAtomicWspMode, kWspFoldedMode), true);
  model.op_list_[op_desc->GetId()] = op_desc;
  model.SetFeatureBaseRefreshable(true);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_no_sm_desc_folded_te) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  const auto test_size = "test_size";
  kernel_def->set_args(test_size);
  auto op_desc = std::make_shared<OpDesc>("add", MEMSET);
  EXPECT_EQ(ge::AttrUtils::SetStr(op_desc, kAttrNameAtomicWspMode, kWspFoldedMode), true);
  model.op_list_[op_desc->GetId()] = op_desc;
  model.SetFeatureBaseRefreshable(true);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_aicpu) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(6);

  auto op_desc = std::make_shared<OpDesc>("concat", "TensorArrayWrite");
  model.op_list_[op_desc->GetId()] = op_desc;
  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, int_task_cust_aicpu) {
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp", 2, 2);
  const char cust_aicpu_bin[] = "cust_framework_kernel_bin_001";
  vector<char> buffer(cust_aicpu_bin, cust_aicpu_bin + strlen(cust_aicpu_bin));
  const auto kernel_handle = std::make_shared<OpKernelBin>(op_desc->GetName(), std::move(buffer));
  op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_handle);
  op_desc->SetId(0);
  const auto model_task_def = MakeShared<domi::ModelTaskDef>();

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetModelTaskDef(model_task_def);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_kernel = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_kernel);
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  domi::TaskDef &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def.set_stream_id(op_desc->GetStreamId());

  std::vector<char> args_info(64U, '0');
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();
  kernel_def.set_args_size(args_info.size());
  kernel_def.set_args(args_info.data(), args_info.size());
  kernel_def.set_so_name("libfeatures.so");
  kernel_def.set_kernel_name("features");

  domi::KernelContext &context = *kernel_def.mutable_context();
  context.set_kernel_type(static_cast<uint32_t>(ccKernelType::CUST_AI_CPU));
  context.set_op_index(op_desc->GetId());

  {
    // Get OP_EXTATTR_CUSTAICPU_KERNEL from CustAICPUKernelStore
    KernelTaskInfo kernel_task_info;
    kernel_task_info.args_size_ = 120;
    kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
    kernel_task_info.task_type_ = static_cast<ModelTaskType>(task_def.type());
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    int64_t host_data[1024] = {0};
    args[0].len = 1024;
    args[0].host_addr = host_data;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    free(ValueToPtr(args[0].dev_addr));
  }

  {
    // Get OP_EXTATTR_CUSTAICPU_KERNEL from OpDesc
    KernelTaskInfo kernel_task_info;
    kernel_task_info.args_size_ = 120;
    kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
    kernel_task_info.task_type_ = static_cast<ModelTaskType>(task_def.type());
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(1024);
    int64_t host_data[1024] = {0};
    args[0].len = 1024;
    args[0].host_addr = host_data;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, aicpu_kernel);
    EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    free(ValueToPtr(args[0].dev_addr));
  }
}

TEST_F(UtestKernelTaskInfo, int_task_cust_aicpu_known) {
  const auto op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp", 2, 2);
  const char cust_aicpu_bin[] = "cust_framework_kernel_bin_001";
  vector<char> buffer(cust_aicpu_bin, cust_aicpu_bin + strlen(cust_aicpu_bin));
  const auto kernel_handle = std::make_shared<OpKernelBin>(op_desc->GetName(), std::move(buffer));
  op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_handle);
  op_desc->SetId(0);
  AttrUtils::SetInt(op_desc, "op_para_size", 16);
  const auto model_task_def = MakeShared<domi::ModelTaskDef>();

  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetModelTaskDef(model_task_def);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_kernel = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_kernel);
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  domi::TaskDef &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def.set_stream_id(op_desc->GetStreamId());

  std::vector<char> args_info(64U, '0');
  domi::KernelDef &kernel_def = *task_def.mutable_kernel();
  kernel_def.set_args_size(args_info.size());
  kernel_def.set_args(args_info.data(), args_info.size());
  kernel_def.set_so_name("libfeatures.so");
  kernel_def.set_kernel_name("features");

  domi::KernelContext &context = *kernel_def.mutable_context();
  context.set_kernel_type(static_cast<uint32_t>(ccKernelType::CUST_AI_CPU));
  context.set_op_index(op_desc->GetId());
  model.op_list_[op_desc->GetId()] = op_desc;

  KernelTaskInfo kernel_task_info;
  kernel_task_info.args_size_ = 120;
  kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  int64_t host_data[1024] = {0};
  args[0].len = 1024;
  args[0].host_addr = host_data;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  // Get OP_EXTATTR_CUSTAICPU_KERNEL from CustAICPUKernelStore
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  free(ValueToPtr(args[0].dev_addr));
  // Get OP_EXTATTR_CUSTAICPU_KERNEL from OpDesc
  // op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, aicpu_kernel);
  // EXPECT_EQ(kernel_task_info.Init(task_def, &model), SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(kernel_task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_EQ(infos.size(), 0UL);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_custom_aicpu) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(7);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.task_type_ = ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.op_desc_ = std::make_shared<OpDesc>("concat", "TensorArrayWrite");
  model.op_list_[kernel_task_info.op_desc_->GetId()] = kernel_task_info.op_desc_;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_InitDumpArgs) {
  DavinciModel model(0, nullptr);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;

  DumpProperties properties;
  properties.model_dump_properties_map_ = { {DUMP_ALL_MODEL, {}} };
  model.SetDumpProperties(properties);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.args_ = static_cast<void *>(&kernel_task_info);
  kernel_task_info.args_size_ = 1024;
  // Default kernel_type is CCE_AI_CORE, not set dump
  kernel_task_info.InitDumpArgs(0U);
  EXPECT_EQ(kernel_task_info.dump_args_, nullptr);

  const void *dump_args_001 = reinterpret_cast<uint8_t *>(&kernel_task_info) + 32U;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.InitDumpArgs(32U);
  EXPECT_EQ(kernel_task_info.dump_args_, dump_args_001);

  const void *dump_args_002 = reinterpret_cast<uint8_t *>(&kernel_task_info) + 64U;
  model.is_op_debug_reg_ = true;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.InitDumpArgs(64U);
  EXPECT_EQ(kernel_task_info.dump_args_, dump_args_002);
}

TEST_F(UtestKernelTaskInfo, cust_aicpu_workspace) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::WorkSpaceInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_WORKSPACE_INFO;
  ext_info->infoLen = sizeof(hybrid::WorkSpaceInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::WorkSpaceInfo *space_info = reinterpret_cast<hybrid::WorkSpaceInfo*>(buf + offset);
  space_info->size = 0U;
  space_info->addr = 0U;

  domi::TaskDef task_def;
  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  const auto op_desc = CreateOpDesc("deque", "Deque");
  op_desc->SetId(0);
  op_desc->SetWorkspace({0x55});
  op_desc->SetWorkspaceBytes({0x10});

  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
  DavinciModel davinci_mdl(0, nullptr);
  davinci_mdl.op_list_.emplace(0, op_desc);

  KernelTaskInfo kernel_task;
  kernel_task.op_desc_ = op_desc;
  kernel_task.workspace_addrs_.push_back(0x55);
  kernel_task.davinci_model_ = &davinci_mdl;
  EXPECT_EQ(kernel_task.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  kernel_task.Release();
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_update_args_te) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  std::vector<uint64_t> active_base_addr;
  active_base_addr.resize(model.logical_mem_allocations_.size());
  for (size_t i = 0; i < model.logical_mem_allocations_.size(); i++) {
    active_base_addr.emplace_back(model.allocation_ids_to_active_base_addr_[i]);
  }
  EXPECT_EQ(kernel_task_info.UpdateHostArgs(active_base_addr, nullptr, 0), SUCCESS);
  model.SetFeatureBaseRefreshable(true);
  EXPECT_EQ(kernel_task_info.UpdateHostArgs(active_base_addr, nullptr, 0), SUCCESS);
  kernel_task_info.kernel_type_ = ccKernelType::AI_CPU;
  kernel_task_info.args_size_ = 8;
  EXPECT_NE(kernel_task_info.UpdateHostArgs(active_base_addr, nullptr, 0), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_update_args_aicpu) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  kernel_task_info.args_size_ = 120;
  kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
  kernel_task_info.io_addrs_ = { PtrToValue((void*)0x12345678), PtrToValue((void*)0x22345678) };
  rtMalloc(&kernel_task_info.args_, kernel_task_info.args_size_, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  kernel_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back({0,0});
  kernel_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back({1,1});
  std::vector<uint64_t> active_base_addr;
  active_base_addr.resize(model.logical_mem_allocations_.size());
  for (size_t i = 0; i < model.logical_mem_allocations_.size(); i++) {
    active_base_addr.emplace_back(model.allocation_ids_to_active_base_addr_[i]);
  }
  active_base_addr.push_back(100);
  active_base_addr.push_back(200);
  std::vector<uint64_t> args(2U, 0U);
  EXPECT_EQ(kernel_task_info.UpdateHostArgs(active_base_addr, args.data(), 128), SUCCESS);
  model.SetFeatureBaseRefreshable(true);
  EXPECT_EQ(kernel_task_info.UpdateHostArgs(active_base_addr, args.data(), 128), SUCCESS);
  rtFree((void*)kernel_task_info.args_);
}

TEST_F(UtestKernelTaskInfo, blocking_aicpu_op) {
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

  domi::TaskDef task_def;
  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  const auto op_desc = CreateOpDesc("deque", "Deque");
  op_desc->SetId(0);
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_.emplace(0, op_desc);
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  davinci_model.operator_list_[op_desc->GetId()] = operator_info;

  KernelTaskInfo kernel_task_info;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.davinci_model_ = &davinci_model;
  kernel_task_info.operator_ = operator_info;
  kernel_task_info.func_handle_ = (void *)0x12000;
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  kernel_task_info.Release();
  kernel_task_info.op_desc_ = op_desc;
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  kernel_task_info.Release();
}

TEST_F(UtestKernelTaskInfo, blocking_aicpu_op_fail_01) {
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

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  op_desc->SetId(0);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_.emplace(0, op_desc);
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  davinci_model.operator_list_[op_desc->GetId()] = operator_info;

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &davinci_model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.operator_ = operator_info;
  kernel_task_info.func_handle_ = (void *)0x12000;
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);

  kernel_task_info.is_blocking_aicpu_op_ = true;
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);
  kernel_task_info.Release();
}

TEST_F(UtestKernelTaskInfo, blocking_aicpu_op_fail_02) {
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

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  op_desc->SetId(0);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_.emplace(0, op_desc);
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  davinci_model.operator_list_[op_desc->GetId()] = operator_info;

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &davinci_model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.operator_ = operator_info;
  kernel_task_info.func_handle_ = (void *)0x12000;
  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);

  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtStreamWaitEventWithTimeout, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);
  kernel_task_info.Release();

  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtEventReset, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);
  kernel_task_info.Release();

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(kernel_task_info.IsSupportReDistribute());
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  kernel_task_info.Release();
}

TEST_F(UtestKernelTaskInfo, CopyNoncontinuousArgs_Invalid) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  kernel_task_info.args_size_ = 2 * sizeof(int64_t);
  kernel_task_info.args_addr_.resize(kernel_task_info.args_size_);
  std::vector<uint64_t> active_base_addr;
  active_base_addr.resize(model.logical_mem_allocations_.size());
  for (size_t i = 0; i < model.logical_mem_allocations_.size(); i++) {
    active_base_addr.emplace_back(model.allocation_ids_to_active_base_addr_[i]);
  }
  active_base_addr.push_back(12405233);
  active_base_addr.push_back(12406543);
  active_base_addr.push_back(12405409);
  MemAllocationAndOffset v1 = {0, 120};
  MemAllocationAndOffset v2 = {2, 0};
  kernel_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v1);
  kernel_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v2);
  std::vector<int64_t> host_args({0,0});
  uint16_t offset = 0;
  auto ret = kernel_task_info.UpdateNoncontinuousArgs(offset, active_base_addr,
      static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestKernelTaskInfo, IsL1OrUBFusionOp_Invalid) {
  DavinciModel model(0, nullptr);
  OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;

  auto ret = kernel_task_info.IsL1OrUBFusionOp(op_desc);
  EXPECT_FALSE(ret);
}

TEST_F(UtestKernelTaskInfo, IsL1OrUBFusionOp_valid) {
  DavinciModel model(0, nullptr);
  OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");

  KernelTaskInfo kernel_task_info;

  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
  EXPECT_TRUE(kernel_task_info.IsL1OrUBFusionOp(op_desc));

  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, {});
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
  EXPECT_TRUE(kernel_task_info.IsL1OrUBFusionOp(op_desc));
}

TEST_F(UtestKernelTaskInfo, test_UpdateAtomicArgs) {
  KernelTaskInfo kernel_task_info;
  OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  std::vector<int64_t> atomic_output_indices = {0};
  (void)AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  std::map<int64_t, int64_t> work_spaces;
  work_spaces[0] = 0;
  workspace_info["test"] = work_spaces;
  GeAttrValue::NAMED_ATTRS workspaces;
  for (const auto &iter : workspace_info) {
    const std::string &op_name = iter.first;
    const auto &index_offset_map = iter.second;
    std::vector<int64_t> value;
    for (const auto &iter2 : index_offset_map) {
      value.emplace_back(iter2.first);
      value.emplace_back(iter2.second);
    }
    workspaces.SetAttr(op_name, GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>(value));
  }
  (void)AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);
  kernel_task_info.op_desc_ = op_desc;
  int a = 10;
  void *p = &a;
  std::vector<uint64_t> input_data_addrs;
  std::vector<uint64_t> output_data_addrs;
  std::vector<uint64_t> workspace_data_addrs;
  output_data_addrs.push_back(PtrToValue(p));
  workspace_data_addrs.push_back(PtrToValue(p));

  kernel_task_info.UpdateAtomicCleanArgs(input_data_addrs, output_data_addrs, workspace_data_addrs);
  EXPECT_EQ(input_data_addrs.size(), 0);
  EXPECT_EQ(output_data_addrs.size(), 1);
  EXPECT_EQ(workspace_data_addrs.size(), 0);
}

TEST_F(UtestKernelTaskInfo, test_SetIoAddr) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000000U;
  OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp", 0, 1);
  op_desc->SetWorkspace({0x55});
  op_desc->SetWorkspaceBytes({0x10});
  std::vector<int64_t> atomic_output_indices = {0};
  (void)AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  std::map<int64_t, int64_t> work_spaces;
  work_spaces[0] = 0x100000;
  workspace_info[op_desc->GetName()] = work_spaces;
  GeAttrValue::NAMED_ATTRS workspaces;
  for (const auto &iter : workspace_info) {
    const std::string &op_name = iter.first;
    const auto &index_offset_map = iter.second;
    std::vector<int64_t> value;
    for (const auto &iter2 : index_offset_map) {
      value.emplace_back(iter2.first);
      value.emplace_back(iter2.second);
    }
    workspaces.SetAttr(op_name, GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>(value));
  }
  (void)AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_op_index(op_desc->GetId());
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  ctx->mutable_origin_op_index()->Clear();
  model.op_list_[op_desc->GetId()] = op_desc;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.is_separately_clean_task_ = true;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.args_size_ = 16;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.args_descs.size(), 1);
  EXPECT_EQ(task_run_param.args_descs[0].args_len, 0);
  EXPECT_EQ(task_run_param.args_descs[0].placement, ArgsPlacement::kArgsPlacementHbm);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 0);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].support_refresh, true);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs[0].logic_addr, model.runtime_param_.mem_base + 0x55);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs[0].support_refresh, true);
  kernel_task_info.SetIoAddrs();
  // output index + workspace index
  EXPECT_EQ(kernel_task_info.io_addrs_.size(), 2);
}


TEST_F(UtestKernelTaskInfo, test_SetIoAddrWithTilingData) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000000U;
  ASSERT_EQ(rtMalloc(&model.globalworkspace_overflow_addr_, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16),
            SUCCESS);
  OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp", 0, 1);
  op_desc->SetWorkspace({0x55});
  op_desc->SetWorkspaceBytes({0x10});
  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);
  std::vector<int64_t> atomic_output_indices = {0};
  (void)AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  std::map<int64_t, int64_t> work_spaces;
  work_spaces[0] = 0x100000;
  workspace_info[op_desc->GetName()] = work_spaces;
  GeAttrValue::NAMED_ATTRS workspaces;
  for (const auto &iter : workspace_info) {
    const std::string &op_name = iter.first;
    const auto &index_offset_map = iter.second;
    std::vector<int64_t> value;
    for (const auto &iter2 : index_offset_map) {
      value.emplace_back(iter2.first);
      value.emplace_back(iter2.second);
    }
    workspaces.SetAttr(op_name, GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>(value));
  }
  (void)AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_op_index(op_desc->GetId());
  ctx->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  ctx->mutable_origin_op_index()->Clear();
  model.op_list_[op_desc->GetId()] = op_desc;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.is_separately_clean_task_ = false;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.args_size_ = 16;
  kernel_task_info.task_type_ = ModelTaskType::MODEL_TASK_ALL_KERNEL;
  kernel_task_info.is_addrs_folded_ = true;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.args_descs.size(), 1);
  EXPECT_EQ(task_run_param.args_descs[0].placement, ArgsPlacement::kArgsPlacementHbm);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 0);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].support_refresh, true);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs[0].logic_addr, model.runtime_param_.mem_base + 0x55);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs[0].support_refresh, true);

  kernel_task_info.SetIoAddrs();
  // output index + workspace index
  EXPECT_EQ(kernel_task_info.io_addrs_.size(), 4);
  rtFree((void*)model.globalworkspace_overflow_addr_);
}

UINT32 StubTiling(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  size_t *workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = 64U;
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

TEST_F(UtestKernelTaskInfo, soft_sync_op_tiling) {
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto space_registry_array = gert::OpImplSpaceRegistryV2Array();
  space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp)) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp));
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(std::make_shared<gert::OpImplSpaceRegistryV2Array>(space_registry_array));
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, 0, UINT64_MAX, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  auto op_desc = CreateOpDesc("relu", RELU);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetBool(op_desc, "globalworkspace_type", true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 16);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetIsInputConst({false});
  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({0, 128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 32);
  op_desc->SetId(0);
  op_desc->SetWorkspace({32});
  op_desc->SetWorkspaceBytes({32});
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->tiling = StubTiling;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->tiling_parse = StubTilingParse;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->compile_info_creator = CompileInfoCreator;

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{0, 1};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(op_desc, ""), SUCCESS);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(48U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  KernelTaskInfo kernel_task_info;
  int64_t op_index = kernel_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, op_desc->GetId());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length

  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  {
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }

  {
    model.feature_base_refreshable_ = true;
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }
}

TEST_F(UtestKernelTaskInfo, mc2_static_bin_reuse) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto space_registry_array = gert::OpImplSpaceRegistryV2Array();
  space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp)) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();

  auto space_registry = space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp));
  auto funcs = space_registry->CreateOrGetOpImpl("MatmulAllReduce");
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(ge::MakeShared<gert::OpImplSpaceRegistryV2Array>(space_registry_array));
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ASSERT_EQ(rtMalloc(&model.globalworkspace_overflow_addr_, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16),
            SUCCESS);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  model.platform_infos_.core_num_ = 0U;
  const auto op_desc = CreateOpDesc("mc2", "MatmulAllReduce", 3, 3);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetId(0);

  // ir_def
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"z", 1}, {"gather_out", 2}};
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("z", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000});
  op_desc->SetOutputOffset({5000, 5100, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("11111111");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(op_desc->GetId());
  aicpu_context.set_op_index(op_desc->GetId());
  aicpu_context.set_args_format("{i0}{i1}{}{o0}{}{o2}{hi.hcom0*}{ws*}{overflow_addr}{ws0}{t}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  EXPECT_NE(operator_info, nullptr);
  model.operator_list_[op_desc->GetId()] = operator_info;

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(aicpu_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  // io_addr校验
  EXPECT_EQ(kernel_task_info.io_addrs_.size(), 11);
  uint64_t fm_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memory_holder.data()));

  EXPECT_EQ(kernel_task_info.io_addrs_[0], fm_base + 1000);
  EXPECT_EQ(kernel_task_info.io_addrs_[1], 0); // 可选输入
  EXPECT_EQ(kernel_task_info.io_addrs_[2], 0);
  EXPECT_EQ(kernel_task_info.io_addrs_[3], fm_base + 5000);
  EXPECT_EQ(kernel_task_info.io_addrs_[4], 0);
  EXPECT_EQ(kernel_task_info.io_addrs_[5], fm_base + 6000);
  EXPECT_EQ(kernel_task_info.io_addrs_[6], 0xf1);
  EXPECT_EQ(kernel_task_info.io_addrs_[7], fm_base + 7000);
  EXPECT_EQ(kernel_task_info.io_addrs_[9], fm_base + 7000);

  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  free((void*)args[0].dev_addr);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  rtFree((void*)model.globalworkspace_overflow_addr_);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestKernelTaskInfo, mc2_fusion_task_static_bin_reuse_with_sub_aicore_ccu) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  HcclDllHcomMgr mgr = HcclDllHcomMgr::GetInstance();
  HcclDllHcomMgr::GetInstance().hccl_HcomGetCcuTaskInfo_func = &InitializeHeterogeneousRuntime;

  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto space_registry_array = gert::OpImplSpaceRegistryV2Array();
  space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp)) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp));
  auto funcs = space_registry->CreateOrGetOpImpl("MatmulAllReduce");
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(ge::MakeShared<gert::OpImplSpaceRegistryV2Array>(space_registry_array));
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     6300, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation io_mem_allocation = {1, static_cast<uint64_t>(model.runtime_param_.mem_base) + 6300,
                                     model.runtime_param_.mem_size - 6300, ge::MemAllocation::Type::OUTPUT, 0U};

  model.logical_mem_allocations_.emplace_back(io_mem_allocation);

  ASSERT_EQ(rtMalloc(&model.globalworkspace_overflow_addr_, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16),
            SUCCESS);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  model.platform_infos_.core_num_ = 0U;
  const auto op_desc = CreateOpDesc("mc2", "MatmulAllReduce", 5, 5);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetId(0);

  std::string kernel_handle_name = model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, nullptr, nullptr);

  GeShape shape0({8});
  GeTensorDesc desc0(shape0);
  TensorUtils::SetSize(desc0, 32);
  op_desc->UpdateInputDesc(0, desc0);
  op_desc->UpdateInputDesc(1, desc0);
  op_desc->UpdateOutputDesc(0, desc0);
  op_desc->UpdateOutputDesc(1, desc0);

  GeShape shape1({4, 4, 4, 4});
  GeTensorDesc desc1(shape1);
  TensorUtils::SetSize(desc1, 1024);
  op_desc->UpdateInputDesc(2, desc1);
  op_desc->UpdateInputDesc(3, desc1);
  op_desc->UpdateInputDesc(4, desc1);
  op_desc->UpdateOutputDesc(2, desc1);
  op_desc->UpdateOutputDesc(3, desc1);
  op_desc->UpdateOutputDesc(4, desc1);

  // ir_def
  // 实际连边的顺序
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}, {"k0", 2}, {"k1", 3}, {"a", 4}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}, {"z0", 2}, {"z1", 3}, {"m", 4}};

  // ir的顺序为添加顺序
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("a", IrInputType::kIrInputRequired);

  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("z", IrOutputType::kIrOutputDynamic);
  op_desc->AppendIrOutput("m", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000, 4100, 4200, 6400});
  op_desc->SetOutputOffset({5000, 6000, 6100, 6200, 6300});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);
  (void)AttrUtils::SetStr(op_desc, HCOM_ATTR_GROUP, "test");

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("11111111");
  run_info->SetTilingKey(0x1234);
  run_info->AddWorkspace(512);
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetBool(op_desc, "_memcheck", true);

  auto &fusion_task = *model_task_def->add_task();
  fusion_task.set_type(static_cast<int32_t>(ge::ModelTaskType::MODEL_TASK_FUSION_KERNEL));
  auto fusion = fusion_task.mutable_fusion_task();
  fusion->set_args_format("{ffts_addr}{i0}{i2}{}{i_desc3}{i_instance4}{o0}{o1}{o_desc2}{o_instance4}{hi.hcom0*}{ws*}{overflow_addr}{ws0}{t}{#123}{tiling_context}{*op_type}");
  fusion->set_op_index(op_desc->GetId());;
  fusion->set_kfc_args_format_offset(13);;

  // 1.1 AICORE 子任务
  auto* sub1 = fusion->add_fusion_sub_task_info();
  sub1->set_type(domi::FusionSubTaskInfo::AICORE);
  auto* aicore = sub1->mutable_task()->mutable_aicore_fusion_task_info();

  // 1.1.1 KernelContext
  auto* ctx = aicore->mutable_context();
  ctx->set_kernel_type(11);

  // 1.1.2 args 二进制
  aicore->set_is_all_kernel(true);

  // 1.1.3 LaunchConfig → LaunchAttribute
  auto* cfg = aicore->mutable_config();
  auto* attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM);
  attr->mutable_value()->set_block_dim(256);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM_OFFSET);
  attr->mutable_value()->set_block_dim_offset(1);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::SCHEMMODE);
  attr->mutable_value()->set_schem_model(1);

  // 1.2 CCU 子任务
  auto* sub2 = fusion->add_fusion_sub_task_info();
  sub2->set_type(domi::FusionSubTaskInfo::CCU);
  sub2->mutable_task()->mutable_ccu_task_group()->add_group("group");

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  EXPECT_NE(operator_info, nullptr);
  model.operator_list_[op_desc->GetId()] = operator_info;

  // for SetTvmTaskZeroCopy
  ZeroCopyOffset zero_copy_offset;
  std::vector<uint64_t> tensor_addrs;
  EXPECT_EQ(zero_copy_offset.SetOutputOutsideAddrs(6300, false, reinterpret_cast<uintptr_t>(memory_holder.data()) + 6300, tensor_addrs), SUCCESS);
  model.output_data_info_[4] = zero_copy_offset;

  ZeroCopyOffset zero_copy_offset1;
  EXPECT_EQ(zero_copy_offset1.SetInputOutsideAddrs(6400, reinterpret_cast<uintptr_t>(memory_holder.data()) + 6400, false, model.real_virtual_addrs_), SUCCESS);
  model.input_data_info_[4] = zero_copy_offset1;

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  // model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  DumpStub::GetInstance().Clear();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));

  model.is_op_debug_reg_ = true;

  FusionTaskInfo fusion_task_info;
  fusion_task_info.davinci_model_ = &model;
  fusion_task_info.op_desc_ = op_desc;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(fusion_task_info.ParseTaskRunParam(fusion_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(fusion_task_info.Init(fusion_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  // rt_sub_task_ 校验
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[0].type, 2);
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[0].task.aicoreInfo.tilingKey, 0x1234);
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[0].task.aicoreInfo.config->numAttrs, 4); // 包含dump
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[1].type, 3);

  // io_addr校验
  EXPECT_EQ(fusion_task_info.io_addrs_.size(), 41);
  uint64_t fm_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memory_holder.data()));

  EXPECT_EQ(fusion_task_info.io_addrs_[0], fm_base + 1000);
  EXPECT_EQ(fusion_task_info.io_addrs_[1], fm_base + 3000);
  EXPECT_EQ(fusion_task_info.io_addrs_[3], args[0].dev_addr + 15 * sizeof(uint64_t));
  EXPECT_EQ(fusion_task_info.io_addrs_[4], fm_base + 6400);
  EXPECT_EQ(fusion_task_info.io_addrs_[5], fm_base + 5000);
  EXPECT_EQ(fusion_task_info.io_addrs_[6], fm_base + 6000);
  EXPECT_EQ(fusion_task_info.io_addrs_[7], args[0].dev_addr + 28 * sizeof(uint64_t));
  EXPECT_EQ(fusion_task_info.io_addrs_[8], fm_base + 6300);
  EXPECT_EQ(fusion_task_info.io_addrs_[9], 0xf1);
  EXPECT_EQ(fusion_task_info.io_addrs_[10], fm_base + 7000);
  EXPECT_EQ(fusion_task_info.io_addrs_[12], fm_base + 7000);
  EXPECT_EQ(fusion_task_info.io_addrs_[14], 123);
  EXPECT_EQ(fusion_task_info.io_addrs_[15], 0x58);
  EXPECT_EQ(fusion_task_info.io_addrs_[16], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[17], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[18], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[19], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[20], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[21], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[22], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[23], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[24], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[25], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[26], fm_base + 4100);
  EXPECT_EQ(fusion_task_info.io_addrs_[27], fm_base + 4200);
  EXPECT_EQ(fusion_task_info.io_addrs_[28], 0x58);
  EXPECT_EQ(fusion_task_info.io_addrs_[29], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[30], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[31], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[32], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[33], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[34], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[35], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[36], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[37], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[38], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[39], fm_base + 6100);
  EXPECT_EQ(fusion_task_info.io_addrs_[40], fm_base + 6200);

  // l0 excetion dump校验； 旧的l0 exception dump 流程未校验，待补充
  // iow的长度
  auto units = ge::DumpStub::GetInstance().GetStaticUnits();
  ASSERT_EQ(units.size(), 1);
  ASSERT_EQ(units[0].size(), 23);
  EXPECT_EQ(units[0][0], 32);    // i0
  EXPECT_EQ(units[0][1], 32);    // i1
  EXPECT_EQ(units[0][2], 0);     // hold
  EXPECT_EQ(units[0][3], 112);   // idesc1
  EXPECT_EQ(units[0][4], 1024);    // i4
  EXPECT_EQ(units[0][5], 32);    // o0
  EXPECT_EQ(units[0][6], 32);    // o1
  EXPECT_EQ(units[0][7], 112);   // 0desc1
  EXPECT_EQ(units[0][8], 1024);    // o4
  EXPECT_EQ(units[0][9], 0);     // hcom
  EXPECT_EQ(units[0][10], 512);   // ws
  EXPECT_EQ(units[0][11], 1);     // i1 dim
  EXPECT_EQ(units[0][12], 8);    //
  EXPECT_EQ(units[0][13], 1);    // i2 dim
  EXPECT_EQ(units[0][14], 8);    //
  EXPECT_EQ(units[0][16], 4);    //
  EXPECT_EQ(units[0][17], 4);    //
  EXPECT_EQ(units[0][18], 4);    //
  EXPECT_EQ(units[0][19], 4);    //
  EXPECT_EQ(units[0][20], 0);    // o1
  EXPECT_EQ(units[0][21], 0);    // o2
  EXPECT_EQ(units[0][22], 0);    // o3

  ge::DumpStub::GetInstance().Clear();

  // data dump 输入输出在args table表中的偏移
  auto cust_to_relevant = fusion_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = { {0, 0}, {1, 1}, {2, 26}, {3, 27}, {4, 4}, {5, 5}, {6, 6}, {7, 39}, {8, 40}, {9, 8}};
  EXPECT_EQ(golden, cust_to_relevant);

  EXPECT_EQ(fusion_task_info.Distribute(), SUCCESS);
  // rt_args_ex_ 校验
  EXPECT_EQ(fusion_task_info.rt_args_ex_.argsSize, 44 * sizeof(uint64_t) + 15 * sizeof(uint64_t)); // 按照args table 表最大来赋值，ws* 16 uint32_t

  EXPECT_EQ(fusion_task_info.Release(), SUCCESS);

  ExtraOpInfo extra_dump_info{};
  fusion_task_info.GetTilingKeyAndData(extra_dump_info.tiling_key, extra_dump_info.tiling_data);
  EXPECT_EQ(extra_dump_info.tiling_key, 0x1234);
  // EXPECT_EQ(extra_dump_info.tiling_data, "11111111");

  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(fusion_task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);

  fusion_task_info.PostProcess(fusion_task);
  EXPECT_EQ(fusion_task_info.ParseOpIndex(fusion_task), 0);

  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  free((void*)args[0].dev_addr);
  rtFree((void*)model.globalworkspace_overflow_addr_);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestKernelTaskInfo, mc2_fusion_task_static_bin_reuse_with_sub_aicore_aicpu) {
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  HcclDllHcomMgr mgr = HcclDllHcomMgr::GetInstance();
  HcclDllHcomMgr::GetInstance().hccl_HcomGetCcuTaskInfo_func = &InitializeHeterogeneousRuntime;

  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto space_registry_array = gert::OpImplSpaceRegistryV2Array();
  space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp)) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp));
  auto funcs = space_registry->CreateOrGetOpImpl("MatmulAllReduce");
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(ge::MakeShared<gert::OpImplSpaceRegistryV2Array>(space_registry_array));
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     6300, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation io_mem_allocation = {1, static_cast<uint64_t>(model.runtime_param_.mem_base) + 6300,
                                     model.runtime_param_.mem_size - 6300, ge::MemAllocation::Type::OUTPUT, 0U};

  model.logical_mem_allocations_.emplace_back(io_mem_allocation);

  ASSERT_EQ(rtMalloc(&model.globalworkspace_overflow_addr_, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16),
            SUCCESS);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  model.platform_infos_.core_num_ = 0U;
  const auto op_desc = CreateOpDesc("mc2", "MatmulAllReduce", 5, 5);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetId(0);

  std::string kernel_handle_name = model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, nullptr, nullptr);

  GeShape shape0({8});
  GeTensorDesc desc0(shape0);
  TensorUtils::SetSize(desc0, 32);
  op_desc->UpdateInputDesc(0, desc0);
  op_desc->UpdateInputDesc(1, desc0);
  op_desc->UpdateOutputDesc(0, desc0);
  op_desc->UpdateOutputDesc(1, desc0);

  GeShape shape1({4, 4, 4, 4});
  GeTensorDesc desc1(shape1);
  TensorUtils::SetSize(desc1, 1024);
  op_desc->UpdateInputDesc(2, desc1);
  op_desc->UpdateInputDesc(3, desc1);
  op_desc->UpdateInputDesc(4, desc1);
  op_desc->UpdateOutputDesc(2, desc1);
  op_desc->UpdateOutputDesc(3, desc1);
  op_desc->UpdateOutputDesc(4, desc1);

  // ir_def
  // 实际连边的顺序
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}, {"k0", 2}, {"k1", 3}, {"a", 4}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}, {"z0", 2}, {"z1", 3}, {"m", 4}};

  // ir的顺序为添加顺序
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("a", IrInputType::kIrInputRequired);

  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("z", IrOutputType::kIrOutputDynamic);
  op_desc->AppendIrOutput("m", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000, 4100, 4200, 6400});
  op_desc->SetOutputOffset({5000, 6000, 6100, 6200, 6300});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);
  (void)AttrUtils::SetStr(op_desc, HCOM_ATTR_GROUP, "test");

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("11111111");
  run_info->SetTilingKey(0x1234);
  run_info->AddWorkspace(512);
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetBool(op_desc, "_memcheck", true);

  auto &fusion_task = *model_task_def->add_task();
  fusion_task.set_type(static_cast<int32_t>(ge::ModelTaskType::MODEL_TASK_FUSION_KERNEL));
  auto fusion = fusion_task.mutable_fusion_task();
  fusion->set_args_format("{ffts_addr}{i0}{i2}{}{i_desc3}{i_instance4}{o0}{o1}{o_desc2}{o_instance4}{hi.hcom0*}{ws*}{overflow_addr}{ws0}{t}{#123}{tiling_context}{*op_type}");
  fusion->set_op_index(op_desc->GetId());;
  fusion->set_kfc_args_format_offset(13);;

  // 1.1 AICORE 子任务
  auto* sub1 = fusion->add_fusion_sub_task_info();
  sub1->set_type(domi::FusionSubTaskInfo::AICORE);
  auto* aicore = sub1->mutable_task()->mutable_aicore_fusion_task_info();

  // 1.1.1 KernelContext
  auto* ctx = aicore->mutable_context();
  ctx->set_kernel_type(11);

  // 1.1.2 args 二进制
  aicore->set_is_all_kernel(true);

  // 1.1.3 LaunchConfig → LaunchAttribute
  auto* cfg = aicore->mutable_config();
  auto* attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM);
  attr->mutable_value()->set_block_dim(256);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM_OFFSET);
  attr->mutable_value()->set_block_dim_offset(1);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::SCHEMMODE);
  attr->mutable_value()->set_schem_model(1);

  // 1.3 aicpu 子任务 覆盖异常分支
  auto* sub3 = fusion->add_fusion_sub_task_info();
  sub3->set_type(domi::FusionSubTaskInfo::AICPU);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  EXPECT_NE(operator_info, nullptr);
  model.operator_list_[op_desc->GetId()] = operator_info;

  // for SetTvmTaskZeroCopy
  ZeroCopyOffset zero_copy_offset;
  std::vector<uint64_t> tensor_addrs;
  EXPECT_EQ(zero_copy_offset.SetOutputOutsideAddrs(6300, false, reinterpret_cast<uintptr_t>(memory_holder.data()) + 6300, tensor_addrs), SUCCESS);
  model.output_data_info_[4] = zero_copy_offset;

  ZeroCopyOffset zero_copy_offset1;
  EXPECT_EQ(zero_copy_offset1.SetInputOutsideAddrs(6400, reinterpret_cast<uintptr_t>(memory_holder.data()) + 6400, false, model.real_virtual_addrs_), SUCCESS);
  model.input_data_info_[4] = zero_copy_offset1;

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  // model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  DumpStub::GetInstance().Clear();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));

  model.is_op_debug_reg_ = true;

  FusionTaskInfo fusion_task_info;
  fusion_task_info.davinci_model_ = &model;
  fusion_task_info.op_desc_ = op_desc;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(fusion_task_info.ParseTaskRunParam(fusion_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(fusion_task_info.Init(fusion_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  free((void*)args[0].dev_addr);
  rtFree((void*)model.globalworkspace_overflow_addr_);
}

TEST_F(UtestKernelTaskInfo, mc2_fusion_task_stubfunc_with_sub_aicore_ccu) {
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  HcclDllHcomMgr mgr = HcclDllHcomMgr::GetInstance();
  HcclDllHcomMgr::GetInstance().hccl_HcomGetCcuTaskInfo_func = &InitializeHeterogeneousRuntime;

  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto space_registry_array = gert::OpImplSpaceRegistryV2Array();
  space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp)) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp));
  auto funcs = space_registry->CreateOrGetOpImpl("MatmulAllReduce");
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(ge::MakeShared<gert::OpImplSpaceRegistryV2Array>(space_registry_array));
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ASSERT_EQ(rtMalloc(&model.globalworkspace_overflow_addr_, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16),
            SUCCESS);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  model.platform_infos_.core_num_ = 0U;
  const auto op_desc = CreateOpDesc("mc2", "MatmulAllReduce", 5, 5);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetId(0);

  std::string kernel_handle_name = model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, nullptr, nullptr);

  GeShape shape0({8});
  GeTensorDesc desc0(shape0);
  TensorUtils::SetSize(desc0, 32);
  op_desc->UpdateInputDesc(0, desc0);
  op_desc->UpdateInputDesc(1, desc0);
  op_desc->UpdateOutputDesc(0, desc0);
  op_desc->UpdateOutputDesc(1, desc0);

  GeShape shape1({4, 4, 4, 4});
  GeTensorDesc desc1(shape1);
  TensorUtils::SetSize(desc1, 1024);
  op_desc->UpdateInputDesc(2, desc1);
  op_desc->UpdateInputDesc(3, desc1);
  op_desc->UpdateInputDesc(4, desc1);
  op_desc->UpdateOutputDesc(2, desc1);
  op_desc->UpdateOutputDesc(3, desc1);
  op_desc->UpdateOutputDesc(4, desc1);

  // ir_def
  // 实际连边的顺序
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}, {"k0", 2}, {"k1", 3}, {"a", 4}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}, {"z0", 2}, {"z1", 3}, {"m", 4}};

  // ir的顺序为添加顺序
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("a", IrInputType::kIrInputRequired);

  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("z", IrOutputType::kIrOutputDynamic);
  op_desc->AppendIrOutput("m", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000, 4100, 4200, 4300});
  op_desc->SetOutputOffset({5000, 6000, 6100, 6200, 6300});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);
  (void)AttrUtils::SetStr(op_desc, HCOM_ATTR_GROUP, "test");

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("11111111");
  run_info->SetTilingKey(0x1234);
  run_info->AddWorkspace(512);
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetBool(op_desc, "_memcheck", true);

  auto &fusion_task = *model_task_def->add_task();
  fusion_task.set_type(static_cast<int32_t>(ge::ModelTaskType::MODEL_TASK_FUSION_KERNEL));
  auto fusion = fusion_task.mutable_fusion_task();
  fusion->set_args_format("{i0}{i2}{}{i_desc3}{i_instance4}{o0}{o1}{o_desc2}{o_instance4}{hi.hcom0*}{ws*}{overflow_addr}{ws0}{t}{#123}");
  fusion->set_op_index(op_desc->GetId());;
  fusion->set_kfc_args_format_offset(12);;

  // 1.1 AICORE 子任务
  auto* sub1 = fusion->add_fusion_sub_task_info();
  sub1->set_type(domi::FusionSubTaskInfo::AICORE);
  auto* aicore = sub1->mutable_task()->mutable_aicore_fusion_task_info();

  // 1.1.1 KernelContext
  auto* ctx = aicore->mutable_context();
  ctx->set_kernel_type(11);

  // 1.1.2 args 二进制
  aicore->set_is_all_kernel(false);

  // 1.1.3 LaunchConfig → LaunchAttribute
  auto* cfg = aicore->mutable_config();
  auto* attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM);
  attr->mutable_value()->set_block_dim(256);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM_OFFSET);
  attr->mutable_value()->set_block_dim_offset(1);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::SCHEMMODE);
  attr->mutable_value()->set_schem_model(1);

  // 1.2 CCU 子任务
  auto* sub3 = fusion->add_fusion_sub_task_info();
  sub3->set_type(domi::FusionSubTaskInfo::CCU);
  sub3->mutable_task()->mutable_ccu_task_group()->add_group("group");

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  EXPECT_NE(operator_info, nullptr);
  model.operator_list_[op_desc->GetId()] = operator_info;

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  // model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  DumpStub::GetInstance().Clear();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));

  FusionTaskInfo fusion_task_info;
  fusion_task_info.davinci_model_ = &model;
  fusion_task_info.op_desc_ = op_desc;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(fusion_task_info.ParseTaskRunParam(fusion_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(fusion_task_info.Init(fusion_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  // rt_sub_task_ 校验
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[0].type, 2);
  EXPECT_NE(fusion_task_info.rt_fusion_task_.subTask[0].task.aicoreInfo.stubFunc, nullptr);
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[0].task.aicoreInfo.config->numAttrs, 4); // 包含dump
  EXPECT_EQ(fusion_task_info.rt_fusion_task_.subTask[1].type, 3);

  // io_addr校验
  EXPECT_EQ(fusion_task_info.io_addrs_.size(), 41);
  uint64_t fm_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memory_holder.data()));

  EXPECT_EQ(fusion_task_info.io_addrs_[0], fm_base + 1000);
  EXPECT_EQ(fusion_task_info.io_addrs_[1], fm_base + 3000);
  EXPECT_EQ(fusion_task_info.io_addrs_[3], args[0].dev_addr + 15 * sizeof(uint64_t));
  EXPECT_EQ(fusion_task_info.io_addrs_[4], fm_base + 4300);
  EXPECT_EQ(fusion_task_info.io_addrs_[5], fm_base + 5000);
  EXPECT_EQ(fusion_task_info.io_addrs_[6], fm_base + 6000);
  EXPECT_EQ(fusion_task_info.io_addrs_[7], args[0].dev_addr + 28 * sizeof(uint64_t));
  EXPECT_EQ(fusion_task_info.io_addrs_[8], fm_base + 6300);
  EXPECT_EQ(fusion_task_info.io_addrs_[9], 0xf1);
  EXPECT_EQ(fusion_task_info.io_addrs_[10], fm_base + 7000);
  EXPECT_EQ(fusion_task_info.io_addrs_[12], fm_base + 7000);
  EXPECT_EQ(fusion_task_info.io_addrs_[14], 123);
  EXPECT_EQ(fusion_task_info.io_addrs_[15], 0x58);
  EXPECT_EQ(fusion_task_info.io_addrs_[16], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[17], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[18], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[19], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[20], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[21], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[22], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[23], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[24], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[25], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[26], fm_base + 4100);
  EXPECT_EQ(fusion_task_info.io_addrs_[27], fm_base + 4200);
  EXPECT_EQ(fusion_task_info.io_addrs_[28], 0x58);
  EXPECT_EQ(fusion_task_info.io_addrs_[29], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[30], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[31], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[32], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[33], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[34], 0x100000004);
  EXPECT_EQ(fusion_task_info.io_addrs_[35], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[36], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[37], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[38], 0x4);
  EXPECT_EQ(fusion_task_info.io_addrs_[39], fm_base + 6100);
  EXPECT_EQ(fusion_task_info.io_addrs_[40], fm_base + 6200);

  // l0 excetion dump校验； 旧的l0 exception dump 流程未校验，待补充
  // iow的长度
  auto units = ge::DumpStub::GetInstance().GetStaticUnits();
  ASSERT_EQ(units.size(), 1);
  ASSERT_EQ(units[0].size(), 23);
  EXPECT_EQ(units[0][0], 32);    // i0
  EXPECT_EQ(units[0][1], 32);    // i1
  EXPECT_EQ(units[0][2], 0);     // hold
  EXPECT_EQ(units[0][3], 112);   // idesc1
  EXPECT_EQ(units[0][4], 1024);    // i4
  EXPECT_EQ(units[0][5], 32);    // o0
  EXPECT_EQ(units[0][6], 32);    // o1
  EXPECT_EQ(units[0][7], 112);   // 0desc1
  EXPECT_EQ(units[0][8], 1024);    // o4
  EXPECT_EQ(units[0][9], 0);     // hcom
  EXPECT_EQ(units[0][10], 512);   // ws
  EXPECT_EQ(units[0][11], 1);     // i1 dim
  EXPECT_EQ(units[0][12], 8);    //
  EXPECT_EQ(units[0][13], 1);    // i2 dim
  EXPECT_EQ(units[0][14], 8);    //
  EXPECT_EQ(units[0][16], 4);    //
  EXPECT_EQ(units[0][17], 4);    //
  EXPECT_EQ(units[0][18], 4);    //
  EXPECT_EQ(units[0][19], 4);    //
  EXPECT_EQ(units[0][20], 0);    // o1
  EXPECT_EQ(units[0][21], 0);    // o2
  EXPECT_EQ(units[0][22], 0);    // o3

  ge::DumpStub::GetInstance().Clear();

  // data dump 输入输出在args table表中的偏移
  auto cust_to_relevant = fusion_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = { {0, 0}, {1, 1}, {2, 26}, {3, 27}, {4, 4}, {5, 5}, {6, 6}, {7, 39}, {8, 40}, {9, 8}};
  EXPECT_EQ(golden, cust_to_relevant);


  EXPECT_EQ(fusion_task_info.Distribute(), SUCCESS);
  // rt_args_ex_ 校验
  EXPECT_EQ(fusion_task_info.rt_args_ex_.argsSize, 41 * sizeof(uint64_t) + 15 * sizeof(uint64_t)); // ws* 16 uint32_t

  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  free((void*)args[0].dev_addr);
  rtFree((void*)model.globalworkspace_overflow_addr_);
}


TEST_F(UtestKernelTaskInfo, No_soft_sync_op_with_2_tasks) {
  DavinciModel model(0, nullptr);
  auto op_desc = CreateOpDesc("relu", RELU);
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");

  op_desc->SetId(0);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  MemAllocation fm_mem_allocation = {0, 0U, 2048U, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &atomic_task = *model_task_def.add_task();
  atomic_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto atomic_kernel = atomic_task.mutable_kernel();
  std::vector<uint8_t> atomic_args(1, 0);
  atomic_kernel->set_args(atomic_args.data(), atomic_args.size());
  atomic_kernel->set_args_size(atomic_args.size());
  auto atomic_context = atomic_kernel->mutable_context();
  atomic_context->set_op_index(0);
  atomic_context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t atomic_offset = 0U;
  atomic_context->set_args_offset(&atomic_offset, sizeof(uint16_t));

  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  auto node = graph->AddNode(op_desc);
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;

  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
  EXPECT_EQ(model.task_list_.size(), 2);


  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  PisToArgs args0 = {};
  args0[0] = {reinterpret_cast<uint64_t>(atomic_args.data()), nullptr, (int64_t)atomic_args.size()};
  EXPECT_EQ(kernel_task_info.Init(atomic_task, &model, args0, persistant_workspace, iow_addrs), SUCCESS);
  PisToArgs args1 = {};
  args1[0] = {reinterpret_cast<uint64_t>(args_info.data()), nullptr, (int64_t)args_info.size()};
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args1, persistant_workspace, iow_addrs), SUCCESS);
  string kernel_name = "kernel";
  const std::string attr_key_kernel_name = op_desc->GetName() + "_atomic_kernelname";
  AttrUtils::SetStr(op_desc, attr_key_kernel_name, kernel_name);
  EXPECT_EQ(kernel_task_info.is_separately_clean_task_, false);
  free((void*)args[0].dev_addr);
}

TEST_F(UtestKernelTaskInfo, static_shape_reuse_binary) {
  DavinciModel model(0, nullptr);
  model.SetKnownNode(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, 0, UINT64_MAX, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  auto op_desc = CreateOpDesc("relu", RELU);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 256);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetIsInputConst({false});
  op_desc->SetIsInputConst({false});
  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({0, 128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 32);
  op_desc->SetId(0);
  op_desc->SetWorkspace({32});
  op_desc->SetWorkspaceBytes({32});

  std::map<string, std::map<int64_t, int64_t>> workspaces;
  std::map<int64_t, int64_t> value;
  value[0] = 0;
  workspaces.insert(std::make_pair("relu", value));
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("1");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetBool(op_desc, "_memcheck", true);

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{0, 1};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(op_desc, ""), SUCCESS);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(56U, '0'); // set + 一个input + 两个output + workspace + tilling
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  auto node = graph->AddNode(op_desc);
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;

  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  {
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }

  {
    model.feature_base_refreshable_ = true;
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }
}


TEST_F(UtestKernelTaskInfo, static_shape_reuse_binary_with_ori_op_para_size) {
  DavinciModel model(0, nullptr);
  model.SetKnownNode(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, 0, UINT64_MAX, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  auto op_desc = CreateOpDesc("relu", RELU);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 256);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetIsInputConst({false});
  op_desc->SetIsInputConst({false});
  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({0, 128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 32);
  op_desc->SetId(0);
  op_desc->SetWorkspace({32});
  op_desc->SetWorkspaceBytes({32});

  std::map<string, std::map<int64_t, int64_t>> workspaces;
  std::map<int64_t, int64_t> value;
  value[0] = 0;
  workspaces.insert(std::make_pair("relu", value));
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("1");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  (void)ge::AttrUtils::SetInt(op_desc, "ori_op_para_size", 24);

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{0, 1};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(op_desc, ""), SUCCESS);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(56U, '0'); // set + 一个input + 两个output + workspace + tilling
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  auto node = graph->AddNode(op_desc);
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;

  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  {
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }

  {
    model.feature_base_refreshable_ = true;
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }
}

TEST_F(UtestKernelTaskInfo, static_shape_reuse_binary_vectorcore) {
  DavinciModel model(0, nullptr);
  model.SetKnownNode(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, 0, UINT64_MAX, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  auto op_desc = CreateOpDesc("relu", RELU);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 16);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetIsInputConst({false});
  op_desc->SetIsInputConst({false});
  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({0, 128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 32);
  op_desc->SetId(0);
  op_desc->SetWorkspace({32});
  op_desc->SetWorkspaceBytes({32});

  std::map<string, std::map<int64_t, int64_t>> workspaces;
  std::map<int64_t, int64_t> value;
  value[0] = 0;
  workspaces.insert(std::make_pair("relu", value));
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("1");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{0, 1};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(op_desc, ""), SUCCESS);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::MIX_VECTOR_CORE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(56U, '0'); // set + 一个input + 两个output + workspace + tilling
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  domi::TaskDef &task_def1 = *model_task_def.add_task();
  task_def1.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_VECTOR_KERNEL));
  domi::KernelDef *kernel_def1 = task_def1.mutable_kernel();
  kernel_def1->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::MIX_VECTOR_CORE));
  kernel_def1->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def1->mutable_context()->mutable_origin_op_index()->Clear();
  kernel_def1->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  kernel_def1->set_args_size(args_info.size());
  kernel_def1->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  auto node = graph->AddNode(op_desc);
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;

  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  {
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }

  {
    model.feature_base_refreshable_ = true;
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }
}

TEST_F(UtestKernelTaskInfo, static_shape_reuse_binary_atomic) {
  DavinciModel model(0, nullptr);
  model.SetKnownNode(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, 0, UINT64_MAX, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  auto op_desc = CreateOpDesc("relu", RELU);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 16);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetIsInputConst({false});
  op_desc->SetIsInputConst({false});
  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({0, 128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 32);
  op_desc->SetId(0);
  op_desc->SetWorkspace({32});
  op_desc->SetWorkspaceBytes({32});

  std::map<string, std::map<int64_t, int64_t>> workspaces;
  std::map<int64_t, int64_t> value;
  value[0] = 0;
  workspaces.insert(std::make_pair("relu", value));
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("1");
  op_desc->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, run_info);

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{0, 1};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);

  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(op_desc, ""), SUCCESS);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(56U, '0'); // set + 一个input + 两个output + workspace + tilling
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  auto node = graph->AddNode(op_desc);
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.is_separately_clean_task_ = true;
  kernel_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_task_info.CopyTilingDataIfNeeded(), SUCCESS);
  EXPECT_TRUE(kernel_task_info.has_tiling_);
  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  {
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }
}

TEST_F(UtestKernelTaskInfo, KernelTaskInit_SaveExceptionInfo_EnableExceptionDump) {
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kExceptionDump));
  ge::DumpStub::GetInstance().ClearOpInfos();
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  const auto op_desc = CreateOpDesc("relu", RELU, 2, 1);
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");

  op_desc->SetInputOffset({10,10});
  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length
  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  kernel_task_info.args_ex_.args = kernel_task_info.args_;
  kernel_task_info.args_ex_.argsSize = kernel_task_info.args_size_;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxPrintf = "printf";
  std::vector<std::string> dfx_opts{kOpDfxPrintf};
  ge::AttrUtils::SetListStr(op_desc, kOpDfxOptions, dfx_opts);
  model.SaveDfxInfo(op_desc->GetId(), task_def, kernel_task_info);
  auto &exception_dump = *(gert::GlobalDumper::GetInstance()->GetInnerExceptionDumpers().begin());
  EXPECT_EQ(exception_dump, &model.exception_dumper_);

  const auto &op_info = ge::DumpStub::GetInstance().GetOpInfos()[0];
  auto input_addrs = ModelUtils::GetInputAddrs(model.runtime_param_, op_desc);
  EXPECT_EQ(op_info.tensorInfos[0].tensorAddr, input_addrs[0]);
  EXPECT_EQ(op_info.tensorInfos[1].tensorAddr, input_addrs[1]);

  EXPECT_EQ(exception_dump->op_desc_info_[0].input_addrs[0], input_addrs[0]);
  EXPECT_EQ(exception_dump->op_desc_info_[0].input_addrs[1], input_addrs[1]);

  // check workspace
  void *addr;
  uint64_t size;
  EXPECT_TRUE(ge::AdxGetWorkspaceInfo(op_info, 0, addr, size));
  EXPECT_EQ(size, ModelUtils::GetWorkspaceSize(op_desc)[0]);
  EXPECT_EQ(addr, ModelUtils::GetWorkspaceDataAddrs(model.runtime_param_, op_desc)[0]);

  // check args
  EXPECT_TRUE(AdxGetArgsInfo(op_info, addr, size));

  task_def.clear_kernel();
  model.runtime_param_.mem_base = 0U;
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  ge::DumpStub::GetInstance().ClearOpInfos();
  free(ValueToPtr(args[0].dev_addr));
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
}

// exception dump is disbale, and op info will saved by ge and don't send to adump
TEST_F(UtestKernelTaskInfo, KernelTaskInit_SaveExceptionInfo_WithoutEnableExceptionDump) {
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  ge::DumpStub::GetInstance().ClearOpInfos();

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  const auto op_desc = CreateOpDesc("relu", RELU, 2, 1);
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");

  op_desc->SetInputOffset({10,10});
  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(64U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->SetWorkspace({1308});   // offset
  op_desc->SetWorkspaceBytes({150});    // length
  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = (uint64_t)malloc(1024);
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  kernel_task_info.args_ex_.args = kernel_task_info.args_;
  kernel_task_info.args_ex_.argsSize = kernel_task_info.args_size_;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxPrintf = "printf";
  std::vector<std::string> dfx_opts{kOpDfxPrintf};
  ge::AttrUtils::SetListStr(op_desc, kOpDfxOptions, dfx_opts);
  model.SaveDfxInfo(op_desc->GetId(), task_def, kernel_task_info);
  auto &exception_dump = *(gert::GlobalDumper::GetInstance()->GetInnerExceptionDumpers().begin());
  EXPECT_EQ(exception_dump, &model.exception_dumper_);

  EXPECT_FALSE(exception_dump->op_desc_info_.empty());
  EXPECT_TRUE(ge::DumpStub::GetInstance().GetOpInfos().empty());

  // get op info form ge local
  auto input_addrs = ModelUtils::GetInputAddrs(model.runtime_param_, op_desc);
  EXPECT_EQ(exception_dump->op_desc_info_[0].input_addrs[0], input_addrs[0]);
  EXPECT_EQ(exception_dump->op_desc_info_[0].input_addrs[1], input_addrs[1]);

  EXPECT_EQ(exception_dump->op_desc_info_[0].workspace_bytes[0], ModelUtils::GetWorkspaceSize(op_desc)[0]);
  EXPECT_EQ(exception_dump->op_desc_info_[0].space_addrs[0],
            ModelUtils::GetWorkspaceDataAddrs(model.runtime_param_, op_desc)[0]);

  task_def.clear_kernel();
  model.runtime_param_.mem_base = 0U;
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);

  free(ValueToPtr(args[0].dev_addr));
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
}

TEST_F(UtestKernelTaskInfo, CopyNoncontinuousArgs_Succes) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  kernel_task_info.args_addr_.resize(2 * sizeof(uint64_t));
  kernel_task_info.args_size_ = 2 * sizeof(uint64_t);
  std::vector<uint64_t> active_base_addr;
  active_base_addr.resize(model.logical_mem_allocations_.size());
  for (size_t i = 0; i < model.logical_mem_allocations_.size(); i++) {
    active_base_addr.emplace_back(model.allocation_ids_to_active_base_addr_[i]);
  }
  active_base_addr.push_back(12405233);
  active_base_addr.push_back(12406543);
  active_base_addr.push_back(12405409);
  MemAllocationAndOffset v1 = {0, 120};
  MemAllocationAndOffset v2 = {2, 0};
  kernel_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v1);
  kernel_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v2);
  std::vector<int64_t> host_args({0,0});
  uint16_t offset = 0;
  auto ret = kernel_task_info.UpdateNoncontinuousArgs(offset, active_base_addr,
      static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestKernelTaskInfo, init_mc2_cust_aicpu_success) {
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    addrs.push_back(reinterpret_cast<void *>(0xf2));
    return ge::GRAPH_SUCCESS;
  };

  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  const auto op_desc = CreateOpDesc("mc2", "MatmulAllGather", 3, 2);
  EXPECT_NE(op_desc, nullptr);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  // ir_def
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}};
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  // tiling_data
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(op_desc->GetId());
  aicpu_context.set_op_index(op_desc->GetId());
  aicpu_context.set_args_format("{#123}{i0}{i1}{}{o0}{o1}{hi.hcom0*}{hi.hcom1*}{ws*}{overflow_addr}{ws0}{t}");
  aicpu_context.set_args_count(12);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(aicpu_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  uint8_t device_args[1024];
  args[0].dev_addr = reinterpret_cast<uint64_t>(device_args);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(kernel_task_info.io_addrs_.size(), 11UL);
  EXPECT_EQ(kernel_task_info.io_addrs_[2], 0UL);   // input1 optional
  EXPECT_EQ(kernel_task_info.io_addrs_[3], 0UL);  // placeholder
  EXPECT_EQ(kernel_task_info.io_addrs_[6], 0xf1);  // hidden_input
  EXPECT_EQ(kernel_task_info.io_addrs_[7], 0xf2);  // hidden_input

  auto resource_list = op_desc->GetExtAttr<shared_ptr<std::vector<void *>>>("_rt_resource_list");
  EXPECT_NE(resource_list, nullptr);
  EXPECT_EQ(resource_list->get()->size(), 1UL);
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
}

bool has_hcom_attr = false;

TEST_F(UtestKernelTaskInfo, super_kernel_with_args_format_graph_load_and_success) {
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    has_hcom_attr = AttrUtils::HasAttr(op_desc, "_skn_hcom_input_addr") &&
        AttrUtils::HasAttr(op_desc, "_skn_hcom_output_addr") && AttrUtils::HasAttr(op_desc, "_skn_hcom_ws_addr");
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  DEF_GRAPH(g0) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // sub node 1 input 1
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // sub node 1 input 2
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // sub node 1 input 3
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // sub node 1 input 4
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // sub node 1 input 5

    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // sub node 2 input 1
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);  // sub node 2 input 2
    auto data_7 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 7);  // sub node 2 input 3

    auto node_1 = OP_CFG("node_1")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto node_2 = OP_CFG("HcomAllReduce")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 0)->NODE("node_2", node_2));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 1)->NODE("node_2", node_2));
    CHAIN(NODE("_arg_7", data_6)->EDGE(0, 3)->NODE("node_2", node_2));
    CHAIN(NODE("node_1", node_1)->EDGE(0, 2)->NODE("node_2", node_2));
    CHAIN(NODE("node_2", node_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("node_2", node_2)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("node_2", node_2)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
  };

  auto sub_graph = ToComputeGraph(g0);
  EXPECT_NE(sub_graph, nullptr);

  {
    // 设置data的shape和offset
    for (auto i = 0; i < 8; ++i) {
      GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
      const auto &data = sub_graph->FindNode("_arg_" + std::to_string(i));
      EXPECT_NE(data, nullptr);
      data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
      data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
      data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

      // 设置父节点属性
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, i);
    }

    // 设置输出的shape和offset
    const auto &out_node = sub_graph->FindNode("Node_Output");
    EXPECT_NE(out_node, nullptr);
    out_node->GetOpDesc()->SetSrcName({"node_2", "node_2", "node_3"});
    out_node->GetOpDesc()->SetSrcIndex({0, 1, 2});
    out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    GeTensorDesc input_desc(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(2, input_desc);
    out_node->GetOpDesc()->SetInputOffset({41000, 42000, 43000});
    AttrUtils::SetInt(out_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(out_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(out_node->GetOpDesc()->MutableInputDesc(2), ATTR_NAME_PARENT_NODE_INDEX, 2);

    GeShape shape({4, 4, 4, 4});
    GeTensorDesc desc(shape);
    GeShape scalar_shape;
    GeTensorDesc scalar_desc(scalar_shape);

    // 初始化sub node1
    auto skt_sub_node_1 = sub_graph->FindNode("node_1");
    EXPECT_NE(skt_sub_node_1, nullptr);

    const auto op_desc_1 = skt_sub_node_1->GetOpDescBarePtr();
    op_desc_1->AddDynamicInputDescByIndex("x", 2, 0);
    op_desc_1->UpdateInputDesc(0, desc);
    op_desc_1->UpdateInputDesc(1, desc);
    op_desc_1->AddDynamicInputDescByIndex("y", 3, 2);
    op_desc_1->UpdateInputDesc(2, scalar_desc);
    op_desc_1->UpdateInputDesc(3, scalar_desc);
    op_desc_1->UpdateInputDesc(4, scalar_desc);
    op_desc_1->UpdateOutputDesc(0, desc);

    op_desc_1->SetInputOffset({11000, 12000, 13000, 14000, 15000});
    op_desc_1->SetOutputOffset({21000});
    op_desc_1->SetWorkspace({7000});
    op_desc_1->SetWorkspaceBytes({512});

    op_desc_1->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"y0", 2}, {"y1", 3}, {"y2", 4}};
    op_desc_1->MutableAllOutputName() = {{"z", 0}};

    op_desc_1->AppendIrInput("x", IrInputType::kIrInputDynamic);
    op_desc_1->AppendIrInput("y", IrInputType::kIrInputDynamic);
    op_desc_1->AppendIrOutput("z", IrOutputType::kIrOutputRequired);

    op_desc_1->SetId(19);

    // 初始化subnode2
    auto skt_sub_node_2 = sub_graph->FindNode("node_2");
    EXPECT_NE(skt_sub_node_2, nullptr);

    const auto op_desc_2 = skt_sub_node_2->GetOpDescBarePtr();
    op_desc_2->AddDynamicInputDescByIndex("x", 2, 0);
    op_desc_2->UpdateInputDesc(0, desc);
    op_desc_2->UpdateInputDesc(1, desc);

    op_desc_2->SetSrcName({"node_1", "node_1"});
    op_desc_2->SetSrcIndex({0}); // 内部连边的配置么？
    op_desc_2->UpdateInputDesc(2, desc);

    op_desc_2->AddDynamicInputDescByIndex("z", 1, 3);
    op_desc_2->UpdateInputDesc(3, desc);
    op_desc_2->UpdateOutputDesc(0, desc);
    op_desc_2->UpdateOutputDesc(1, desc);
    op_desc_2->UpdateOutputDesc(2, desc);

    op_desc_2->SetInputOffset({31000, 32000, 21000, 33000});
    op_desc_2->SetOutputOffset({41000, 42000, 43000});
    op_desc_2->SetWorkspace({8000});
    op_desc_2->SetWorkspaceBytes({512});

    op_desc_2->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"y0", 2}, {"z0", 3}};
    op_desc_2->MutableAllOutputName() = {{"j0", 0}, {"j1", 1}, {"k0", 2}};

    op_desc_2->AppendIrInput("x", IrInputType::kIrInputDynamic);
    op_desc_2->AppendIrInput("y", IrInputType::kIrInputRequired);
    op_desc_2->AppendIrInput("z", IrInputType::kIrInputDynamic);
    op_desc_2->AppendIrOutput("j", IrOutputType::kIrOutputDynamic);
    op_desc_2->AppendIrOutput("k", IrOutputType::kIrOutputDynamic);

    op_desc_2->SetId(20);
    AttrUtils::SetInt(op_desc_2, ATTR_NAME_ATTACHED_STREAM_ID, 0);
    AttrUtils::SetInt(op_desc_2, "op_para_size", 256);

    gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
    auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto funcs = space_registry->CreateOrGetOpImpl("HcomAllReduce");
    funcs->tiling = StubTiling;
    funcs->tiling_parse = StubTilingParse;
    funcs->compile_info_creator = CompileInfoCreator;
    funcs->compile_info_deleter = nullptr;
    EXPECT_EQ(funcs->SetTilingInputDataDependency(0), GRAPH_SUCCESS);
  }

  // 构造super kernel所在的图
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);
    auto data_7 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);

    auto skt_node = OP_CFG("super_kernel")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr(ATTR_NAME_KERNEL_BIN_ID, "te_superkernel_123");

    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 5)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 6)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_7", data_7)->EDGE(0, 7)->NODE("super_kernel", skt_node));
    CHAIN(NODE("super_kernel", skt_node)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("super_kernel", skt_node)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("super_kernel", skt_node)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);
  {
    // 设置data的shape和offset
    for (auto i = 0; i < 8; ++i) {
      GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
      const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
      EXPECT_NE(data, nullptr);
      data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
      data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
      data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    }

    // 设置输出的shape和offset
    const auto &out_node = root_graph->FindNode("Node_Output");
    EXPECT_NE(out_node, nullptr);
    out_node->GetOpDesc()->SetSrcName({"super_kernel", "super_kernel", "super_kernel"});
    out_node->GetOpDesc()->SetSrcIndex({0, 1, 2});
    out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    GeTensorDesc input_desc(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(2, input_desc);
    out_node->GetOpDesc()->SetInputOffset({41000, 42000, 43000});

    // 设置super_kernel
    auto skt_node = root_graph->FindNode("super_kernel");
    EXPECT_NE(skt_node, nullptr);

    GeShape shape({4, 4, 4, 4});
    GeTensorDesc desc(shape);
    const auto op_desc = skt_node->GetOpDescBarePtr();
    op_desc->UpdateInputDesc(0, desc);
    op_desc->UpdateInputDesc(1, desc);
    op_desc->UpdateInputDesc(2, desc);
    op_desc->UpdateInputDesc(3, desc);
    op_desc->UpdateInputDesc(4, desc);
    op_desc->UpdateInputDesc(5, desc);
    op_desc->UpdateInputDesc(6, desc);
    op_desc->UpdateInputDesc(7, desc);
    op_desc->UpdateOutputDesc(0, desc);
    op_desc->UpdateOutputDesc(1, desc);
    op_desc->UpdateOutputDesc(2, desc);

    op_desc->SetInputOffset({11000, 12000, 13000, 14000, 15000, 31000, 32000, 33000});
    op_desc->SetOutputOffset({41000, 42000, 43000});
    op_desc->SetId(19);
  }

  auto super_node = root_graph->FindNode("super_kernel");
  const auto op_desc = super_node->GetOpDescBarePtr();
  op_desc->SetId(21);
  op_desc->SetWorkspace({9000});
  op_desc->SetWorkspaceBytes({512});

  // 将子图填入父节点属性
  super_node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(super_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  (void)super_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &skt_task = *model_task_def->add_task();

  skt_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_SUPER_KERNEL));
  skt_task.set_stream_id(0);

  auto aicore_kernel = skt_task.mutable_kernel();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ge::ccKernelType::TE));
  aicore_context.set_op_id(super_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(super_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{skn19ffts_addr}{skn19i_desc0}{skn19i_desc1}{skn19o0}{skn19ws0}{skn20ffts_addr}{skn20event_addr123*}"
      "{skn20i_desc0}{skn20i1}{skn20i_instance3}{skn20o_desc0}{skn20o_instance2}{skn20ws*}{skn20hi.hcom0*}"
      "{skn20tiling_context}{skn20tiling_context.tiling_data}{skn20*op_type}{skn20tiling_context.tiling_key}{skn20tiling_context.block_dim}{ws0*}{overflow_addr}");
  aicore_context.set_args_count(14);
  uint16_t args_offset = 0;
  aicore_context.set_args_offset(&args_offset, sizeof(uint16_t));
  size_t args_size = 128UL;
  const std::vector<uint8_t> args_info(args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 50480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = super_node->GetOpDesc();
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(super_node->GetOpDesc()));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.runtime_param_.mem_size = 50000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size); // 基址
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  DumpProperties properties;
  properties.model_dump_properties_map_ = { {DUMP_ALL_MODEL, {}} };
  model.SetDumpProperties(properties);

  SuperKernelV2TaskInfo super_kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(super_kernel_task_info.ParseTaskRunParam(skt_task, &model, task_run_param), SUCCESS);
  EXPECT_EQ(super_kernel_task_info.GetTaskID(), 0);
  EXPECT_EQ(super_kernel_task_info.GetStreamId(), 0);

  PisToArgs args;
  int64_t host_data[640] = {0};
  args[0].dev_addr = (uint64_t)malloc(640);
  args[0].host_addr = host_data;
  args[0].len = 640;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[640] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 640;

  // 转换为MemoryAppType::kMemoryTypeFeatureMap
  for (auto &addr : task_run_param.parsed_input_addrs) {
    addr.memory_type = 1;
  }

  for (auto &addr : task_run_param.parsed_output_addrs) {
    addr.memory_type = 1;
  }

  for (auto &addr : task_run_param.parsed_workspace_addrs) {
    addr.memory_type = 1;
  }

  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(super_kernel_task_info.ParseOpIndex(skt_task), 21);
  MemManager::Instance().Initialize({ RT_MEMORY_HBM });
  EXPECT_EQ(super_kernel_task_info.Init(skt_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(super_kernel_task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);

  // 地址信息校验
  EXPECT_EQ(infos.size(), 69);
  EXPECT_EQ(infos[0].offset, 0);         //  op_desc_1 ffts_addr
  EXPECT_EQ(infos[3].offset, 21000);     //  op_desc_1 output 0
  EXPECT_EQ(infos[4].offset, 7000);      //  op_desc_1 workspace 0
  EXPECT_EQ(infos[5].offset, 0);         //  op_desc_2 ffts_addr
  EXPECT_EQ(infos[8].offset, 21000);     //  op_desc_2 input 2
  EXPECT_EQ(infos[9].offset, 33000);     //  op_desc_2 input 3
  EXPECT_EQ(infos[11].offset, 43000);    //  op_desc_2 output 2
  EXPECT_EQ(infos[12].offset, 8000);     //  op_desc_2 workspace 0
  EXPECT_EQ(infos[19].offset, 9000);     //  sk workspace 0
  EXPECT_EQ(infos[31].offset, 11000);    //  op_desc_1 input 0
  EXPECT_EQ(infos[32].offset, 12000);    //  op_desc_1 input 1
  EXPECT_EQ(infos[40].offset, 13000);    //  op_desc_1 input 2
  EXPECT_EQ(infos[41].offset, 14000);    //  op_desc_1 input 3
  EXPECT_EQ(infos[42].offset, 15000);    //  op_desc_1 input 4
  EXPECT_EQ(infos[54].offset, 31000);    //  op_desc_2 input 0
  EXPECT_EQ(infos[55].offset, 32000);    //  op_desc_2 input 1
  EXPECT_EQ(infos[67].offset, 41000);    //  op_desc_2 output 0
  EXPECT_EQ(infos[68].offset, 42000);    //  op_desc_2 output 1

  // dump信息校验
  auto cust_to_relevant = super_kernel_task_info.cust_to_relevant_offset_;
  // op_desc_1 input 0; op_desc_1 input 1; op_desc_1 input 2; op_desc_1 input 3; op_desc_1 input 4;op_desc_2 input 0; op_desc_2 input 1; op_desc_2 input 3; op_desc_2 output 0; op_desc_2 output 1; op_desc_2 output 2
  std::map<uint64_t, uint64_t> golden = {{0, 31}, {1, 32}, {2, 40}, {3, 41}, {4, 42}, {5, 54}, {6, 55}, {7, 9}, {8, 67}, {9, 68}, {10, 11}};
  EXPECT_EQ(golden, cust_to_relevant);

  // lo exception dump校验， desc展开的shape不占位，其余都占位
  auto l0_dump_list = super_kernel_task_info.l0_dump_list_;
  EXPECT_EQ(l0_dump_list.size(), 29);

  EXPECT_EQ(l0_dump_list[1], 0x0200000000000002); // skn19i_desc0
  EXPECT_EQ(l0_dump_list[2], 0); // op_desc_1 input 0
  EXPECT_EQ(l0_dump_list[3], 1); // op_desc_1 input 1
  EXPECT_EQ(l0_dump_list[4], 0x0200000000000003); // skn19i_desc1
  EXPECT_EQ(l0_dump_list[5], 2); // op_desc_1 input 2
  EXPECT_EQ(l0_dump_list[6], 3); // op_desc_1 input 3
  EXPECT_EQ(l0_dump_list[7], 4);  // op_desc_1 input 4
  EXPECT_EQ(l0_dump_list[12], 0x0200000000000002); // skn20i_desc0
  EXPECT_EQ(l0_dump_list[13], 5); // op_desc_2 input 0
  EXPECT_EQ(l0_dump_list[14], 6); // op_desc_2 input 1
  EXPECT_EQ(l0_dump_list[16], 7); // op_desc_2 input 3
  EXPECT_EQ(l0_dump_list[17], 0x0200000000000002); // skn20o_desc0
  EXPECT_EQ(l0_dump_list[18], 8); // op_desc_2 output 0
  EXPECT_EQ(l0_dump_list[19], 9); // op_desc_2 output 1
  EXPECT_EQ(l0_dump_list[20], 10); // op_desc_2 output 2

  EXPECT_EQ(super_kernel_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(super_kernel_task_info.Release(), SUCCESS);
  EXPECT_TRUE(has_hcom_attr);
  super_kernel_task_info.PostProcess(skt_task);

  free((void*)args[0].dev_addr);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UtestKernelTaskInfo, super_kernel_findSkSubNode_failed) {
  auto root_graph = gert::ShareGraph::BuildAtomicAicoreGraph();
  auto trans1 = root_graph->FindNode("trans1");
  const auto op_desc = trans1->GetOpDesc();

  auto sub_graph = gert::ShareGraph::BuildAtomicAicoreGraph();
  op_desc->SetExtAttr("_sk_sub_graph", sub_graph);

  SuperKernelV2TaskInfo super_node;
  NodePtr subnode;
  EXPECT_EQ(super_node.FindSkSubNode(op_desc, 100, subnode), FAILED);
}

TEST_F(UtestKernelTaskInfo, ifa_with_args_format_graph_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);  // fake0
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 7)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  const auto op_desc = ifa_node->GetOpDescBarePtr();
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  op_desc->AddDynamicInputDescByIndex("fake", 1, 6);
  op_desc->UpdateInputDesc(3, scalar_desc);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 7000, 8000});
  op_desc->SetOutputOffset({11000, 12000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  op_desc->MutableAllInputName() = {{"query", 0},          {"k0", 1},   {"k1", 2}, {"value0", 3}, {"value1", 4},
                                    {"attention_mask", 5}, {"fake0", 6}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("fake", IrInputType::kIrInputDynamic);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &ifa_task = *model_task_def->add_task();

  ifa_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  ifa_task.set_stream_id(0);
  auto aicore_kernel = ifa_task.mutable_kernel();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{i_desc6}{o_desc0}{ws0}{t}");
  aicore_context.set_args_count(8);

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(aicpu_args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = ifa_node->GetOpDesc();
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(ifa_node->GetOpDesc()));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.runtime_param_.mem_size = 20000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(ifa_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  int64_t host_data[640] = {0};
  args[0].dev_addr = (uint64_t)malloc(640);
  args[0].host_addr = host_data;
  args[0].len = 640;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[640] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 640;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(kernel_task_info.Init(ifa_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);

  auto cust_to_relevant = kernel_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 0}, {1, 19}, {2, 20}, {3, 26}, {4, 27}, {5, 3}, {6, 31}, {7, 37}, {8, 38}};
  EXPECT_EQ(golden, cust_to_relevant);

  free((void*)args[0].dev_addr);
}

TEST_F(UtestKernelTaskInfo, sk_sub_task_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);  // fake0
    auto ifa = OP_CFG("IncreFlashAttention_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 7)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  DEF_GRAPH(g1_parent) {
     CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("sk", "SuperKernel")->EDGE(0, 0)->NODE("net_output", "NetOutput"));
   };
  auto sk_parent_graph = ToComputeGraph(g1_parent);
  NodePtr sk_node = nullptr;
  for (auto node : sk_parent_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "SuperKernel") {
      sk_node = node;
      op_desc->SetExtAttr("_sk_sub_graph", root_graph);
    }
  }
  EXPECT_NE(sk_node, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  const auto op_desc = ifa_node->GetOpDescBarePtr();
  op_desc->SetId(1000);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  op_desc->AddDynamicInputDescByIndex("fake", 1, 6);
  op_desc->UpdateInputDesc(3, scalar_desc);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 7000, 8000});
  op_desc->SetOutputOffset({11000, 12000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  op_desc->MutableAllInputName() = {{"query", 0},          {"k0", 1},   {"k1", 2}, {"value0", 3}, {"value1", 4},
                                    {"attention_mask", 5}, {"fake0", 6}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("fake", IrInputType::kIrInputDynamic);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &ifa_task = *model_task_def->add_task();

  ifa_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  ifa_task.set_stream_id(0);
  auto aicore_kernel = ifa_task.mutable_kernel();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicore_context.set_op_id(sk_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(sk_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{skn1000i0}{skn1000i_desc1}{skn1000i_desc2}{skn1000i4}{skn1000i_desc6}{skn1000o_desc0}{skn1000ws0}{skn1000t}{skn1000event_addr123*}");
  aicore_context.set_args_count(9);

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(aicpu_args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(sk_parent_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  DavinciModel model(0, nullptr);
  model.op_list_[sk_node->GetOpDesc()->GetId()] = sk_node->GetOpDesc();
  model.runtime_param_.mem_size = 20000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};
  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(ifa_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  int64_t host_data[640] = {0};
  args[0].dev_addr = (uint64_t)malloc(640);
  args[0].host_addr = host_data;
  args[0].len = 640;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[640] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 640;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(kernel_task_info.Init(ifa_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);

  auto cust_to_relevant = kernel_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 0}, {1, 20}, {2, 21}, {3, 27}, {4, 28}, {5, 3}, {6, 32}, {7, 38}, {8, 39}};
  EXPECT_EQ(golden, cust_to_relevant);

  free((void*)args[0].dev_addr);
}

TEST_F(UtestKernelTaskInfo, mix_ifa_with_args_format_graph_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  const auto op_desc = ifa_node->GetOpDesc();
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({11000, 12000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_ifa_123");

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &ifa_task = *model_task_def->add_task();

  ifa_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  ifa_task.set_stream_id(0);
  auto aicore_kernel = ifa_task.mutable_kernel();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{ffts_addr}{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{t}");
  aicore_context.set_args_count(8);
  uint16_t offset = 0U;
  aicore_context.set_args_offset(&offset, sizeof(uint16_t));

  size_t aicpu_args_size = 136UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(aicpu_args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = ifa_node->GetOpDesc();
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(ifa_node->GetOpDesc()));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.runtime_param_.mem_size = 20000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(ifa_task, &model, task_run_param), SUCCESS);

  ZeroCopyOffset in_offset;
  model.input_data_info_[0] = in_offset;
  std::vector<uint64_t> in_tensors{};
  model.input_data_info_[0].SetOutputOutsideAddrs(1000, false, task_run_param.parsed_input_addrs[0].logic_addr,
                                                  in_tensors);
  ZeroCopyOffset out_offset;
  model.output_data_info_[0] = out_offset;
  std::vector<uint64_t> out_tensors{};
  model.output_data_info_[0].SetOutputOutsideAddrs(12000, false, task_run_param.parsed_output_addrs[0].logic_addr,
                                                   out_tensors);
  PisToArgs args;
  int64_t host_data[520] = {0};
  args[0].dev_addr = (uint64_t)malloc(520);
  args[0].host_addr = host_data;
  args[0].len = 520;
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(kernel_task_info.Init(ifa_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);

  auto cust_to_relevant = kernel_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 1}, {1, 19}, {2, 20}, {3, 26}, {4, 27}, {5, 4}, {6, 33}, {7, 34}};
  EXPECT_EQ(golden, cust_to_relevant);

  EXPECT_EQ(kernel_task_info.l0_dump_list_.size(), 14UL);

  auto in_res = model.input_data_info_[0].GetOutsideAddrs();
  auto out_res = model.output_data_info_[0].GetOutsideAddrs();
  ASSERT_TRUE(!in_res.empty());
  ASSERT_TRUE(!out_res.empty());
  EXPECT_TRUE(!in_res[0][iow_addrs.input_logic_addrs[0].logic_addr].empty());
  EXPECT_TRUE(!out_res[0][iow_addrs.output_logic_addrs[0].logic_addr].empty());
  free((void*)args[0].dev_addr);
}

TEST_F(UtestKernelTaskInfo, init_mc2_cust_aicpu_with_tilefwk_hiddeninput_success) {
  auto tilefwk_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    addrs.push_back(reinterpret_cast<void *>(0xf2));
    return ge::GRAPH_SUCCESS;
  };

  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::TILEFWK, tilefwk_hidden_funcs);

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  const auto op_desc = CreateOpDesc("mc2", "MatmulAllGather", 3, 2);
  EXPECT_NE(op_desc, nullptr);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  // ir_def
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}};
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  // tiling_data
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(op_desc->GetId());
  aicpu_context.set_op_index(op_desc->GetId());
  aicpu_context.set_args_format("{#123}{i0}{i1}{}{o0}{o1}{hi.tilefwk0*}{hi.tilefwk1*}{ws*}{overflow_addr}{ws0}{t}");
  aicpu_context.set_args_count(12);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(aicpu_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  uint8_t device_args[1024];
  args[0].dev_addr = reinterpret_cast<uint64_t>(device_args);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, tiling_sink_success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 1);
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("MatmulAllReduce");
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(0), GRAPH_SUCCESS);

  DavinciModel model(0, nullptr);
  //  model.SetSpaceRegistry(space_registry);
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ASSERT_EQ(
      rtMalloc(&model.globalworkspace_overflow_addr_, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16),
      SUCCESS);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  model.platform_infos_.core_num_ = 0U;
  const auto op_desc = CreateOpDesc("mc2", "MatmulAllReduce", 3, 2);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetId(0);

  // ir_def
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}};
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);
  AttrUtils::SetInt(op_desc, "op_para_size", 16);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MatmulAllReduce")->tiling = StubTiling;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MatmulAllReduce")->tiling_parse = StubTilingParse;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MatmulAllReduce")->compile_info_creator = CompileInfoCreator;

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(op_desc->GetId());
  aicpu_context.set_op_index(op_desc->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.block_dim}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  EXPECT_NE(operator_info, nullptr);
  model.operator_list_[op_desc->GetId()] = operator_info;

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  {
    KernelTaskInfo kernel_task_info_dump;
    kernel_task_info_dump.davinci_model_ = &model;
    kernel_task_info_dump.op_desc_ = op_desc;
    TaskRunParam task_run_param_dump = {};
    EXPECT_EQ(kernel_task_info_dump.ParseTaskRunParam(aicpu_task, &model, task_run_param_dump), SUCCESS);

    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(2048);
    int64_t host_data[2048] = {0};
    args[0].len = 2048;
    args[0].host_addr = host_data;
    GELOGD("ARGS SIZE:%u", kernel_task_info_dump.args_size_);
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param_dump.parsed_input_addrs), std::move(task_run_param_dump.parsed_output_addrs),
                          std::move(task_run_param_dump.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info_dump.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    free((void *)args[0].dev_addr);
  }

  {
    DumpStub::GetInstance().SetEnableFlag(false);
    std::shared_ptr<TilingContextAddr> default_ctx_ptr;
    EXPECT_TRUE(op_desc->SetExtAttr(kTilingContextAddrs, default_ctx_ptr));
    KernelTaskInfo kernel_task_info;
    kernel_task_info.davinci_model_ = &model;
    kernel_task_info.op_desc_ = op_desc;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(aicpu_task, &model, task_run_param), SUCCESS);

    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(2048);
    int64_t host_data[2048] = {0};
    args[0].len = 2048;
    args[0].host_addr = host_data;
    GELOGD("ARGS SIZE:%u", kernel_task_info.args_size_);
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    // no need to du dump
    EXPECT_TRUE(kernel_task_info.cust_to_relevant_offset_.empty());
    EXPECT_TRUE(!kernel_task_info.dump_flag_);
    EXPECT_TRUE(kernel_task_info.l0_dump_list_.empty());

    free((void *)args[0].dev_addr);
    DumpStub::GetInstance().SetEnableFlag(true);
  }

  {
    std::vector<ArgDesc> desc;
    ArgsFormatDescUtils::Append(desc, AddrType::FFTS_ADDR);
    model.tiling_sink_task_arg_descs_list_[0] =  desc;

    std::shared_ptr<TilingContextAddr> default_ctx_ptr;
    EXPECT_TRUE(op_desc->SetExtAttr(kTilingContextAddrs, default_ctx_ptr));
    KernelTaskInfo kernel_task_info;
    kernel_task_info.davinci_model_ = &model;
    kernel_task_info.op_desc_ = op_desc;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_task_info.ParseTaskRunParam(aicpu_task, &model, task_run_param), SUCCESS);

    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(2048);
    int64_t host_data[2048] = {0};
    args[0].len = 2048;
    args[0].host_addr = host_data;
    GELOGD("ARGS SIZE:%u", kernel_task_info.args_size_);
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    // no need to du dump
    EXPECT_TRUE(kernel_task_info.cust_to_relevant_offset_.empty());
    EXPECT_TRUE(!kernel_task_info.dump_flag_);
    EXPECT_TRUE(kernel_task_info.l0_dump_list_.empty());

    free((void *)args[0].dev_addr);
  }

  rtFree((void *)model.globalworkspace_overflow_addr_);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 1);
}

TEST_F(UtestKernelTaskInfo, ifa_with_tiling_sink_graph_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  const auto op_desc = ifa_node->GetOpDescBarePtr();
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({11000, 12000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &ifa_task = *model_task_def->add_task();

  ifa_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  ifa_task.set_stream_id(0);
  auto aicore_kernel = ifa_task.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{}{tiling_context.tiling_data}");
  aicore_context.set_args_count(8);
  uint16_t offset = 16U;
  aicore_context.set_args_offset(&offset, sizeof(uint16_t));

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(aicpu_args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  std::shared_ptr<TilingContextAddr> tiling_context_addr = MakeShared<TilingContextAddr>();
  tiling_context_addr->tiling_context_addr = 0U;
  tiling_context_addr->op_type_addr = 0U;
  tiling_context_addr->tiling_key_addr = 0U;
  tiling_context_addr->block_dim_addr = 0U;
  tiling_context_addr->tiling_data_addr = 0U;
  EXPECT_TRUE(op_desc->SetExtAttr(kTilingContextAddrs, tiling_context_addr));

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = ifa_node->GetOpDesc();
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(ifa_node->GetOpDesc()));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.runtime_param_.mem_size = 20000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(ifa_node->GetOpDesc(), ""), SUCCESS);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(ifa_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  int64_t host_data[256] = {0};
  args[0].dev_addr = (uint64_t)malloc(256);
  args[0].host_addr = host_data;
  args[0].len = 256;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(kernel_task_info.Init(ifa_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);

  auto cust_to_relevant = kernel_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 0}, {1, 19}, {2, 20}, {3, 26}, {4, 27}, {5, 3}, {6, 33}, {7, 34}};
  EXPECT_EQ(golden, cust_to_relevant);

  EXPECT_EQ(kernel_task_info.l0_dump_list_.size(), 14UL);
  EXPECT_EQ(kernel_task_info.l0_dump_list_[12U], std::numeric_limits<uint64_t>::max());  // placeholder
  EXPECT_EQ(static_cast<uint8_t>((kernel_task_info.l0_dump_list_[13U] >> 56U)) & 0xFFU,
            0x3U);  // tiling_context.tiling_data
  EXPECT_EQ(kernel_task_info.tiling_key_, 0);
  free((void *)args[0].dev_addr);
}

TEST_F(UtestKernelTaskInfo, ifa_with_tiling_sink_graph_load_and_success_with_dfx_shape) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});

  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  const auto op_desc = ifa_node->GetOpDescBarePtr();
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  TensorUtils::SetSize(desc, 1024);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  TensorUtils::SetSize(scalar_desc, 64);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({11000, 12000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &ifa_task = *model_task_def->add_task();

  ifa_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  ifa_task.set_stream_id(0);
  auto aicore_kernel = ifa_task.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{}{tiling_context.tiling_data}");
  aicore_context.set_args_count(8);
  uint16_t offset = 16U;
  aicore_context.set_args_offset(&offset, sizeof(uint16_t));

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(aicpu_args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = ifa_node->GetOpDesc();
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(ifa_node));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.runtime_param_.mem_size = 20000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(ifa_node->GetOpDesc(), ""), SUCCESS);

  // 增加 aicpu task
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("IncreFlashAttention_T");
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(0), GRAPH_SUCCESS);

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(op_desc->GetId());
  aicpu_context.set_op_index(op_desc->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.block_dim}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  // size_t aicpu_args_size = 128UL;
  // const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  // auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(root_graph);
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  DumpManager::GetInstance().Init({{"ge.exec.enable_exception_dump", "2"}});
  ge::DumpStub::GetInstance().Clear();

  // ParseTaskRunParam
  KernelTaskInfo kernel_aicpu_task_info;
  kernel_aicpu_task_info.davinci_model_ = &model;

  TaskRunParam task_aicpu_run_param = {};
  EXPECT_EQ(kernel_aicpu_task_info.ParseTaskRunParam(aicpu_task, &model, task_aicpu_run_param), SUCCESS);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(ifa_task, &model, task_run_param), SUCCESS);

  // aicpu init
  {
    PisToArgs args;
    args[0].dev_addr = (uint64_t)malloc(2048);
    int64_t host_data[2048] = {0};
    args[0].len = 2048;
    args[0].host_addr = host_data;
    GELOGD("ARGS SIZE:%u", kernel_aicpu_task_info.args_size_);
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};
    EXPECT_EQ(kernel_aicpu_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
    std::shared_ptr<TilingContextAddr> tiling_context_addr =
      op_desc->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
    EXPECT_NE(tiling_context_addr, nullptr);
    // max tiling data size:16*1024,
    uint64_t atomic_index_addr = tiling_context_addr->tiling_data_addr + 16 * 1024 - 8;
    // atomic index, runtime stub 里device使用host地址，直接取值
    EXPECT_EQ(*reinterpret_cast<uint64_t *>(static_cast<uintptr_t>(atomic_index_addr)), 1);
    auto units = ge::DumpStub::GetInstance().GetStaticUnits();

    ASSERT_EQ(units.size(), 1);
    ASSERT_EQ(units[0].size(), 12); // input 1
    EXPECT_EQ(units[0][0], 1024);   // i_desc1
    EXPECT_EQ(units[0][1], 112);    // i_desc2
    EXPECT_EQ(units[0][2], 64);     // i_desc2
    EXPECT_EQ(units[0][3], 0);      // input 4
    EXPECT_EQ(units[0][4], 64);     // o_desc0
    EXPECT_EQ(units[0][5], 512);    // ws
    EXPECT_EQ(units[0][6], 4);      // input 1 dim
    EXPECT_EQ(units[0][7], 4);      // 4
    EXPECT_EQ(units[0][8], 4);      // 4
    EXPECT_EQ(units[0][9], 4);      // 4
    EXPECT_EQ(units[0][10], 4);     // 4
    EXPECT_EQ(units[0][11], 0);     // input 4 dim
    ge::DumpStub::GetInstance().Clear();
    free((void *)args[0].dev_addr);
  }
  // aicore init
  {
    PisToArgs args;
    int64_t host_data[256] = {0};
    args[0].dev_addr = (uint64_t)malloc(256);
    args[0].host_addr = host_data;
    args[0].len = 256;
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};

    EXPECT_EQ(kernel_task_info.Init(ifa_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);

    auto cust_to_relevant = kernel_task_info.cust_to_relevant_offset_;
    std::map<uint64_t, uint64_t> golden = {{0, 0}, {1, 19}, {2, 20}, {3, 26}, {4, 27}, {5, 3}, {6, 33}, {7, 34}};
    EXPECT_EQ(golden, cust_to_relevant);

    EXPECT_EQ(kernel_task_info.l0_dump_list_.size(), 14UL);
    EXPECT_EQ(kernel_task_info.l0_dump_list_[12U], std::numeric_limits<uint64_t>::max());  // placeholder
    EXPECT_EQ(static_cast<uint8_t>((kernel_task_info.l0_dump_list_[13U] >> 56U)) & 0xFFU,
              0x3U);  // tiling_context.tiling_data
    EXPECT_EQ(kernel_task_info.tiling_key_, 0);
    free((void *)args[0].dev_addr);
  }
}

TEST_F(UtestKernelTaskInfo, ifa_with_tiling_sink_graph_load_and_success_by_instance_format) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  const auto op_desc = ifa_node->GetOpDescBarePtr();
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({11000, 12000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &ifa_task = *model_task_def->add_task();

  ifa_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  ifa_task.set_stream_id(0);
  auto aicore_kernel = ifa_task.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{}{i_instance0}{i_instance1}{i_instance2}{i_desc2}{o_instance1}");
  aicore_context.set_args_count(6);
  uint16_t offset = 16U;
  aicore_context.set_args_offset(&offset, sizeof(uint16_t));

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(aicpu_args_size);

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  std::shared_ptr<TilingContextAddr> tiling_context_addr = MakeShared<TilingContextAddr>();
  tiling_context_addr->tiling_context_addr = 0U;
  tiling_context_addr->op_type_addr = 0U;
  tiling_context_addr->tiling_key_addr = 0U;
  tiling_context_addr->block_dim_addr = 0U;
  tiling_context_addr->tiling_data_addr = 0U;
  EXPECT_TRUE(op_desc->SetExtAttr(kTilingContextAddrs, tiling_context_addr));

  DavinciModel model(0, nullptr);
  model.op_list_[op_desc->GetId()] = ifa_node->GetOpDesc();
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(ifa_node->GetOpDesc()));
  model.operator_list_[op_desc->GetId()] = operator_info;
  model.runtime_param_.mem_size = 20000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(ifa_node->GetOpDesc(), ""), SUCCESS);

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(ifa_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  int64_t host_data[256] = {0};
  args[0].dev_addr = (uint64_t)malloc(256);
  args[0].host_addr = host_data;
  args[0].len = 256;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(kernel_task_info.Init(ifa_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);

  auto cust_to_relevant = kernel_task_info.cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 1}, {1, 2}, {2, 3}, {3, 11}, {4, 12}, {7, 5}};
  EXPECT_EQ(golden, cust_to_relevant);

  std::vector<uint64_t> golden_l0_dump_list{std::numeric_limits<uint64_t>::max(), 0, 1, 2, 0x200000000000002, 3, 4, 7};

  EXPECT_EQ(kernel_task_info.l0_dump_list_.size(), 8UL);
  EXPECT_EQ(kernel_task_info.l0_dump_list_, golden_l0_dump_list);
  free((void *)args[0].dev_addr);
}

TEST_F(UtestKernelTaskInfo, InitPreprocessTask_dump_flag_Success) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(11);
  data_dumper.SetDeviceId(11);
  data_dumper.SaveEndGraphId(0U, 0U);
  data_dumper.SetComputeGraph(graph);
  DavinciModel model(0, nullptr);
  KernelTaskInfo kernel_task_info;
  kernel_task_info.task_type_ = ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.op_desc_ = CreateOpDesc("relu", RELU);
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.davinci_model_->data_dumper_ = data_dumper;
  kernel_task_info.args_format_holder_.arg_descs.emplace_back(ArgDesc());
  kernel_task_info.InitDumpArgs(0U);
  EXPECT_EQ(kernel_task_info.dump_flag_, static_cast<uint32_t>(RT_KERNEL_CUSTOM_AICPU));
  EXPECT_EQ(kernel_task_info.dump_args_, nullptr);

  kernel_task_info.kernel_type_ = ccKernelType::AI_CPU_KFC;
  kernel_task_info.InitDumpArgs(0U);
  EXPECT_EQ(kernel_task_info.dump_flag_, static_cast<uint32_t>(RT_KERNEL_CUSTOM_AICPU));
  EXPECT_EQ(kernel_task_info.dump_args_, nullptr);

}

TEST_F(UtestKernelTaskInfo, AssembleTilingSinkTensors_PreprocessKernel_Success) {
  KernelTaskInfo kernel_task_info;
  kernel_task_info.task_type_ = ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.args_format_holder_.tiling_depends_input_idx.emplace_back(0);
  kernel_task_info.op_desc_ = CreateOpDesc("relu", RELU);
  kernel_task_info.args_ = 0x0;
  std::map<size_t, gert::AddrRefreshedTensor> index_to_tensor;
  EXPECT_NE(kernel_task_info.AssembleTilingSinkTensors(index_to_tensor), SUCCESS);
  EXPECT_EQ(index_to_tensor[0].device_addr, 660);
}

TEST_F(UtestKernelTaskInfo, AssembleTilingSinkTensors_Sk_PreprocessKernel_Success) {
  SuperKernelV2TaskInfo sk_task_info;
  sk_task_info.task_type_ = ModelTaskType::MODEL_TASK_SUPER_KERNEL;
  sk_task_info.kernel_type_ = ccKernelType::MIX_VECTOR_CORE;
  sk_task_info.sub_node_op_index_list_.emplace_back(0);

  sk_task_info.args_format_holder_.tiling_depends_input_idx.emplace_back(0);
  sk_task_info.sub_node_op_desc_list_.emplace_back(CreateOpDesc("relu", RELU));
  SuperKernelV2TaskInfo::ArgsFormatInfo sub_node_args_format_holder = {};
  sub_node_args_format_holder.tiling_depends_input_idx.emplace_back(0);
  sk_task_info.sub_node_args_format_holder_list_.push_back(sub_node_args_format_holder);
  std::vector<uint64_t> fake_addr{1};
  sk_task_info.sub_node_input_addrs_list_.emplace_back(fake_addr);
  sk_task_info.sub_node_output_addrs_list_.emplace_back(fake_addr);
  sk_task_info.sub_node_workspace_addrs_list_.emplace_back(fake_addr);
  sk_task_info.sub_node_input_mem_types_list_.emplace_back(fake_addr);
  sk_task_info.sub_node_output_mem_types_list_.emplace_back(fake_addr);
  sk_task_info.sub_node_workspace_mem_types_list_.emplace_back(fake_addr);
  sk_task_info.op_desc_ = CreateOpDesc("sk", "SuperKernel");
  sk_task_info.args_ = 0x0;

  std::map<int32_t ,std::map<size_t, gert::AddrRefreshedTensor>> index_to_tensor;
  EXPECT_EQ(sk_task_info.AssembleTilingSinkTensors(index_to_tensor), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, AppendWorkspaceAddr_PreprocessKernel_Success) {
  KernelTaskInfo kernel_task_info;
  kernel_task_info.task_type_ = ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.op_desc_ = CreateOpDesc("relu", RELU);
  kernel_task_info.op_desc_->SetWorkspaceBytes({100, 100});
  kernel_task_info.workspace_addrs_ = {0x1, 0x2};
  kernel_task_info.workspace_mem_types_ = {0x1, 0x2};
  EXPECT_EQ(kernel_task_info.AppendWorkspaceAddr(1), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, InitPreprocessTask_Success) {
  auto &mm = ModelManager::GetInstance();
  std::string uniq_so_name = "1_vendors_test_libcust_opmaster.so";
  std::string so_name;
  so_name.append("/opp/vendors/test/op_impl/ai_core/tbe/op_master_device/lib/libcust_opmaster.so");
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, "", std::move(so_bin), so_name.length());
  mm.cust_op_master_so_names_to_unique_name_[uniq_so_name] = uniq_so_name;
  mm.cust_op_master_so_names_to_bin_[uniq_so_name] = op_so_bin;
  mm.cust_aicpu_so_.clear();
  DavinciModel model(0, nullptr);
  model.SetId(1U);
  KernelTaskInfo kernel_task_info;
  kernel_task_info.so_name_ = so_name;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.task_type_ = ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.op_desc_ = CreateOpDesc("relu", RELU);
  EXPECT_EQ(kernel_task_info.InitPreprocessTask(kernel_task_info.op_desc_), SUCCESS);
}


void StubExceptionFunc(aclrtExceptionInfo *exception_info, void *reserved) {
  (void)exception_info;
  (void)reserved;
}

TEST_F(UtestKernelTaskInfo, SetExceptionCallback_Success) {
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto space_registry_array = gert::OpImplSpaceRegistryV2Array();
  space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp)) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = space_registry_array.at(static_cast<size_t>(OppImplVersion::kOpp));
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  funcs->compile_info_deleter = nullptr;
  funcs->exception_func = StubExceptionFunc;

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(std::make_shared<gert::OpImplSpaceRegistryV2Array>(space_registry_array));
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, 0, UINT64_MAX, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  auto op_desc = CreateOpDesc("relu", RELU);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetBool(op_desc, "globalworkspace_type", true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 16);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetIsInputConst({false});
  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({0, 128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 32);
  op_desc->SetId(0);
  op_desc->SetWorkspace({32});
  op_desc->SetWorkspaceBytes({32});
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->tiling = StubTiling;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->tiling_parse = StubTilingParse;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->compile_info_creator = CompileInfoCreator;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU)->exception_func = StubExceptionFunc;

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{0, 1};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  EXPECT_EQ(model.bin_kernel_handle_.RegisterDynamicKernel(op_desc, ""), SUCCESS);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->mutable_origin_op_index()->Clear();
  uint16_t offset = 16U;
  kernel_def->mutable_context()->set_args_offset(&offset, sizeof(uint16_t));
  std::vector<char> args_info(48U, '0');
  kernel_def->set_args_size(args_info.size());
  kernel_def->set_args(args_info.data(), args_info.size());

  KernelTaskInfo kernel_task_info;
  int64_t op_index = kernel_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, op_desc->GetId());

  model.op_list_[op_desc->GetId()] = op_desc;
  auto graph = std::make_shared<ComputeGraph>("tmp");
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetGraph(graph);
  auto node = graph->AddNode(op_desc);
  auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromNode(node));
  model.operator_list_[op_desc->GetId()] = operator_info;
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});

  model.args_manager_.AllocKernelLaunchArgsHostMem(model.logical_mem_allocations_.size());
  {
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }

  {
    model.feature_base_refreshable_ = true;
    EXPECT_EQ(model.InitTaskInfo(model_task_def), SUCCESS);
    EXPECT_EQ(model.DistributeTask(model_task_def), SUCCESS);
  }
}
}  // namespace ge
