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

#include "graph/load/model_manager/davinci_model.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/graph_mem_allocator.h"

#include "graph/load/model_manager/task_info/aicpu/kernel_ex_task_info.h"
#include "aicpu_engine_struct.h"
#include "depends/runtime/src/runtime_stub.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "graph/load/model_manager/model_utils.h"

namespace ge {
class UtestKernelExTaskInfo : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_init) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = PtrToValue(malloc(1024));
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{0xc212, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.input_data_addrs_.resize(1);
  kernel_ex_task_info.output_data_addrs_.resize(1);
  kernel_ex_task_info.input_addr_mem_types_.resize(1);
  kernel_ex_task_info.output_addr_mem_types_.resize(1);

  EXPECT_EQ(kernel_ex_task_info.Init(task_def, nullptr, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  model.logical_mem_allocations_.clear();
  MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX,
                                      ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), FAILED);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_op_index(1);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  kernel_ex_def->clear_op_index();
  kernel_ex_def->set_op_index(0);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  kernel_ex_task_info.Release();

  std::vector<uint8_t> task_info(150U, 0U);
  kernel_ex_def->set_task_info(task_info.data(), task_info.size());
  kernel_ex_def->set_task_info_size(task_info.size());
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  kernel_ex_task_info.Release();

  constexpr uint32_t arg_size = sizeof(STR_FWK_OP_KERNEL);
  string value1(arg_size, 'a');
  kernel_ex_def->set_args_size(arg_size);
  kernel_ex_def->set_args(value1);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  kernel_ex_task_info.Release();

  ZeroCopyOffset zero_copy_offset_input0;
  kernel_ex_task_info.io_addrs_= {ValueToPtr(0x01), ValueToPtr(0x02)};
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data;
  virtual_addr_out_data[0x01].emplace_back(0x11111);
  virtual_addr_out_data[0x02].emplace_back(0x22222);
  zero_copy_offset_input0.outside_addrs_.emplace_back(virtual_addr_out_data);

  model.input_data_info_[0] = zero_copy_offset_input0;
  kernel_ex_task_info.addrs_size_ = sizeof(uint64_t) * kernel_ex_task_info.io_addrs_.size();
  EXPECT_EQ(kernel_ex_task_info.AssembleInputOutputAddr(), SUCCESS);

  task_def.clear_kernel_ex();
  free(ValueToPtr(args[0].dev_addr));
  kernel_ex_task_info.Release();
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_release) {
  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  void *ptr = nullptr;
  kernel_ex_task_info.kernel_buf_ = nullptr;
  rtMalloc(&kernel_ex_task_info.input_output_addr_, 64, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  ptr = kernel_ex_task_info.input_output_addr_;
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);
  rtFree(ptr);

  kernel_ex_task_info.input_output_addr_ = nullptr;
  rtMalloc(&kernel_ex_task_info.kernel_buf_, 64, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  rtFree(kernel_ex_task_info.kernel_buf_);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  void *addr = nullptr;
  rtMalloc(&addr, 64, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  kernel_ex_task_info.ext_args_.emplace_back(addr);
  rtMalloc(&kernel_ex_task_info.kernel_buf_, 64, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  rtMalloc(&kernel_ex_task_info.input_output_addr_, 64, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  ptr = kernel_ex_task_info.input_output_addr_;
  rtFree(addr);
  rtFree(kernel_ex_task_info.kernel_buf_);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);
  rtFree(ptr);
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_info_copy) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;

  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  std::vector<uint8_t> task_info(150U, 0U);
  kernel_ex_def->set_task_info(task_info.data(), task_info.size());
  kernel_ex_def->set_task_info_size(task_info.size());
  kernel_ex_def->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  std::vector<uint8_t> args_info(sizeof(STR_FWK_OP_KERNEL) + 10, 0);
  kernel_ex_def->set_args(args_info.data(), args_info.size());
  kernel_ex_def->set_args_size(args_info.size());
  // if (sizeof(STR_FWK_OP_KERNEL) < args_size)

  PisToArgs args;
  args[0].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), FAILED);
  kernel_ex_def->set_args(args_info.data(), sizeof(STR_FWK_OP_KERNEL));
  kernel_ex_def->set_args_size(sizeof(STR_FWK_OP_KERNEL));

  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);


  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();

  kernel_ex_def->set_kernel_ext_info(buf, len);
  kernel_ex_def->set_kernel_ext_info_size(len);
  EXPECT_NE(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  free(ValueToPtr(args[0].dev_addr));
  task_def.clear_kernel_ex();
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0;
  kernel_ex_task_info.Release();
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, init_with_zero_copy) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = PtrToValue(malloc(1024));

  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(false);
  model.runtime_param_.mem_size = 2048U;
  model.runtime_param_.zero_copy_size = 2048U; // 全部都是io mem allocation

  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());

  MemAllocation input_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base), 8U,
                                        ge::MemAllocation::Type::INPUT, 0, 0};
  model.logical_mem_allocations_.emplace_back(input_mem_allocation);

  MemAllocation output_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base) + 8U, 8U,
                                         ge::MemAllocation::Type::OUTPUT, 0, 0};
  model.logical_mem_allocations_.emplace_back(output_mem_allocation);

  iow_addrs.input_logic_addrs = {{static_cast<uint64_t>(model.runtime_param_.mem_base),
    (uint64_t)ge::MemoryAppType::kMemoryTypeModelIo}};

  iow_addrs.output_logic_addrs = {{static_cast<uint64_t>(model.runtime_param_.mem_base) + 8U,
    (uint64_t)ge::MemoryAppType::kMemoryTypeModelIo}};

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelExDef &kernel_ex_def = *task_def.mutable_kernel_ex();

  std::vector<uint8_t> task_info(150U, 0U);
  kernel_ex_def.set_task_info(task_info.data(), task_info.size());
  kernel_ex_def.set_task_info_size(task_info.size());
  kernel_ex_def.set_op_index(0U);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length
  GeTensorDesc input_desc;
  EXPECT_TRUE(ge::AttrUtils::SetInt(input_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0));
  EXPECT_EQ(model.op_list_[0]->AddInputDesc(input_desc), GRAPH_SUCCESS);
  EXPECT_EQ(model.op_list_[0]->AddOutputDesc(input_desc), GRAPH_SUCCESS);
  model.op_list_[0]->SetInputOffset({0});
  model.op_list_[0]->SetOutputOffset({8});
  EXPECT_FALSE(ModelUtils::GetInputAddrs(model.GetRuntimeParam(), model.GetOpByIndex(0U)).empty());

  {
    KernelExTaskInfo kernel_ex_task_info;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    int64_t op_index = kernel_ex_task_info.ParseOpIndex(task_def);
    EXPECT_EQ(op_index, 0);
    kernel_ex_task_info.deploy_type_flag_ = RT_KERNEL_DEVICE_FIRST;
    EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    EXPECT_EQ(PtrToValue(kernel_ex_task_info.input_data_addrs_[0]),
      static_cast<uint64_t>(model.runtime_param_.mem_base));
    EXPECT_EQ(PtrToValue(kernel_ex_task_info.output_data_addrs_[0]),
      static_cast<uint64_t>(model.runtime_param_.mem_base) + 8U);

    EXPECT_EQ(kernel_ex_task_info.input_addr_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeModelIo);
    EXPECT_EQ(kernel_ex_task_info.output_addr_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeModelIo);
    kernel_ex_task_info.Release();
  }
  free(ValueToPtr(args[0].dev_addr));
  task_def.clear_kernel_ex();
  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestKernelExTaskInfo, init_for_known_node) {
  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  uint64_t allocation_size = model.logical_mem_allocations_.size();
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

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelExDef &kernel_ex_def = *task_def.mutable_kernel_ex();

  std::vector<uint8_t> task_info(150U, 0U);
  kernel_ex_def.set_task_info(task_info.data(), task_info.size());
  kernel_ex_def.set_task_info_size(task_info.size());
  kernel_ex_def.set_op_index(0U);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length
  GeTensorDesc input_desc;
  EXPECT_TRUE(ge::AttrUtils::SetInt(input_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0));
  EXPECT_EQ(model.op_list_[0]->AddInputDesc(input_desc), GRAPH_SUCCESS);
  model.op_list_[0]->SetInputOffset({0});
  EXPECT_FALSE(ModelUtils::GetInputAddrs(model.GetRuntimeParam(), model.GetOpByIndex(0U)).empty());

  {
    model.logical_mem_allocations_.clear();
    MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX,
                                       ge::MemAllocation::Type::ABSOLUTE, 0U};
    model.logical_mem_allocations_.emplace_back(not_change_mem_item);

    KernelExTaskInfo kernel_ex_task_info;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

    PisToArgs args;
    args[0].dev_addr = PtrToValue(malloc(1024));
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};

    EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

    uint8_t host_args[100] = {};
    std::vector<uint64_t> active_base_addr;
    active_base_addr.resize(allocation_size);
    for (size_t i = 0; i < allocation_size; i++) {
      active_base_addr.emplace_back(model.allocation_ids_to_active_base_addr_[i]);
    }
    active_base_addr.emplace_back(0);
    EXPECT_EQ(kernel_ex_task_info.UpdateHostArgs(active_base_addr, &host_args, 100), SUCCESS);

    EXPECT_EQ(kernel_ex_task_info.UpdateDumpInfos(host_args, 100), SUCCESS);
    std::vector<TaskArgsRefreshInfo> infos;
    EXPECT_EQ(kernel_ex_task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
    EXPECT_EQ(infos.size(), 1UL);

    free(ValueToPtr(args[0].dev_addr));
    kernel_ex_task_info.Release();
  }

  task_def.clear_kernel_ex();
  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestKernelExTaskInfo, init_for_known_node_host_only) {
  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  model.runtime_param_.mem_size = 2048U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemInfo host_svm_mem_info{};
  host_svm_mem_info.memory_size = 2048U;
  host_svm_mem_info.logic_memory_base = kMemoryHostSVMFeatureMapLogicBase;
  host_svm_mem_info.memory_type = RT_MEMORY_HOST_SVM;
  host_svm_mem_info.memory_key = "_svm";
  host_svm_mem_info.memory_base =
      PtrToPtr<void, uint8_t>(MemoryAllocator(RT_MEMORY_HOST_SVM).MallocMemory(host_svm_mem_info.memory_key, host_svm_mem_info.memory_size));
  model.runtime_param_.host_mem_base = PtrToValue(host_svm_mem_info.memory_base);
  MemAllocation host_mem_allocation = {1, static_cast<uint64_t>(model.runtime_param_.host_mem_base),
                                     (uint64_t)host_svm_mem_info.memory_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(host_mem_allocation);
  model.runtime_param_.memory_infos[RT_MEMORY_HOST_SVM] = std::move(host_svm_mem_info);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef &task_def = *model_task_def.add_task();
  domi::KernelExDef &kernel_ex_def = *task_def.mutable_kernel_ex();

  std::vector<uint8_t> task_info(150U, 0U);
  kernel_ex_def.set_task_info(task_info.data(), task_info.size());
  kernel_ex_def.set_task_info_size(task_info.size());
  kernel_ex_def.set_op_index(0U);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp", 1, 1);
  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length
//  GeTensorDesc input_desc;
//  EXPECT_TRUE(ge::AttrUtils::SetInt(input_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0));

  model.op_list_[0]->SetInputOffset({kMemoryHostSVMFeatureMapLogicBase + 0U});
  model.op_list_[0]->SetOutputOffset({kMemoryHostSVMFeatureMapLogicBase + 1024U});
  EXPECT_TRUE(ge::AttrUtils::SetListInt(model.op_list_[0], ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST_SVM}));
  EXPECT_TRUE(ge::AttrUtils::SetListInt(model.op_list_[0], ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST_SVM}));

  {
    KernelExTaskInfo kernel_ex_task_info;
    kernel_ex_task_info.deploy_type_flag_ = RT_KERNEL_HOST_ONLY;
    TaskRunParam task_run_param = {};
    EXPECT_EQ(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
    EXPECT_EQ(task_run_param.args_descs.size(), 1);
    EXPECT_EQ(task_run_param.args_descs[0].args_len, sizeof(uint64_t) * 3); // 刷新归一流程多增加了8字节
    EXPECT_EQ(task_run_param.args_descs[0].placement, ArgsPlacement::kArgsPlacementHbm);
    EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1U);
    EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
    EXPECT_EQ(task_run_param.parsed_input_addrs[0].logic_addr, PtrToValue(PtrToPtr<uint8_t, void>(host_svm_mem_info.memory_base)));
    EXPECT_EQ(task_run_param.parsed_output_addrs[0].logic_addr,  PtrToValue(PtrToPtr<uint8_t, void>(host_svm_mem_info.memory_base))+ 1024);
    EXPECT_EQ(task_run_param.parsed_input_addrs[0].memory_type, RT_MEMORY_HOST_SVM);
    EXPECT_EQ(task_run_param.parsed_output_addrs[0].memory_type, RT_MEMORY_HOST_SVM);
    EXPECT_EQ(task_run_param.parsed_input_addrs[0].support_refresh, true);
    EXPECT_EQ(task_run_param.parsed_output_addrs[0].support_refresh, true);
    EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0U);

    PisToArgs args;
    args[0].dev_addr = PtrToValue(malloc(1024));
    args[static_cast<uint32_t>(ArgsPlacement::kArgsPlacementHostSvm)].dev_addr = PtrToValue(malloc(1024));
    const PisToPersistentWorkspace persistant_workspace = {};
    IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                          std::move(task_run_param.parsed_workspace_addrs)};

    EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
    free(ValueToPtr(args[0].dev_addr));
    free(ValueToPtr(args[static_cast<uint32_t>(ArgsPlacement::kArgsPlacementHostSvm)].dev_addr));
    kernel_ex_task_info.Release();
  }

  task_def.clear_kernel_ex();
  model.runtime_param_.mem_base = 0U;
  model.runtime_param_.host_mem_base = 0U;
}

TEST_F(UtestKernelExTaskInfo, kernel_ex_task_info_calculate_args) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  AttrUtils::SetStr(model.op_list_[0], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");

  KernelExTaskInfo kernel_ex_task_info;
  TaskRunParam task_run_param = {};
  GeTensorDesc descout(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  TensorUtils::SetSize(descout, 32);
  EXPECT_EQ(model.op_list_[0]->AddOutputDesc(descout), GRAPH_SUCCESS);
  model.op_list_[0]->SetOutputOffset({24});// offset
  model.runtime_param_.mem_size = 1024;
  EXPECT_EQ(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, kernel_ex_task_ext_info_without_topic_type) {
  const string ext_info = {1, 1, 1, 1, 0, 0, 0, 0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  EXPECT_EQ(kernel_ex_task_info.deploy_type_flag_, 0); // if not set by extra info, default is 0
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, parse_update_addr) {
  const string ext_info = {3,0,0,0,4,0,0,0,4,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_1) {
  const string ext_info = {7,0,0,0,4,0,0,0,0,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_2) {
  const string ext_info = {7,0,0,0,4,0,0,0,1*16,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_3) {
  const string ext_info = {7,0,0,0,4,0,0,0,2*16,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_4) {
  const string ext_info = {7,0,0,0,4,0,0,0,3*16,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  kernel_ex_task_info.Release();
}
// Use the 5th and 6h bits indicate the value of topic_type.
// xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
// xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
// xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
// xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
// Use the 9th-11th bits indicate the value of qos. 12th indicate qos on/off
// xxxxxxxx xxxxxxxx xxxx0000 xxxxxxxx: qos off
// xxxxxxxx xxxxxxxx xxxx1000 xxxxxxxx: qos on, level=0(min level)
// xxxxxxxx xxxxxxxx xxxx1111 xxxxxxxx: qos on, level=7(max level)
TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_5) {
  const string ext_info = {7,0,0,0,4,0,0,0,4*16,9,0,0}; // little enddian
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
  EXPECT_EQ(kernel_ex_task_info.deploy_type_flag_, 0x00); // 0x40&0x30 >> 4
  EXPECT_EQ(kernel_ex_task_info.qos_level_flag_, 0x900);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_failed_1) {
  const string ext_info = {7,0,0,0,2,0,0,0,2,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  DavinciModel model(0, nullptr);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_NE(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, blocking_aicpu_op) {
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
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.op_desc_ = op_desc;
  DavinciModel davinci_model(0, nullptr);
  kernel_ex_task_info.davinci_model_ = &davinci_model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  domi::GetContext().is_online_model = true;
  kernel_ex_task_info.func_handle_ = (void *)0x12000;
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(kernel_ex_task_info.IsSupportReDistribute());
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  kernel_ex_task_info.Release();
  kernel_ex_task_info.op_desc_ = op_desc;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, blocking_aicpu_op_fail_01) {
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
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.op_desc_ = op_desc;
  DavinciModel davinci_model(0, nullptr);
  kernel_ex_task_info.davinci_model_ = &davinci_model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);

  kernel_ex_task_info.is_blocking_aicpu_op_ = true;
  kernel_ex_task_info.func_handle_ = (void *)0x12000;
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, blocking_aicpu_op_fail_02) {
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
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.op_desc_ = op_desc;
  DavinciModel davinci_model(0, nullptr);
  kernel_ex_task_info.davinci_model_ = &davinci_model;
  kernel_ex_task_info.func_handle_ = (void *)0x12000;
  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);

  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtStreamWaitEventWithTimeout, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);
  kernel_ex_task_info.Release();

  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtEventReset, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);
  kernel_ex_task_info.Release();

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, testUpdateArgs) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;

  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  std::vector<uint8_t> task_info(150U, 0U);
  kernel_ex_def->set_task_info(task_info.data(), task_info.size());
  kernel_ex_def->set_task_info_size(task_info.size());
  kernel_ex_def->set_op_index(0);

  std::vector<uint8_t> args_info(sizeof(STR_FWK_OP_KERNEL) + 10, 0);
  kernel_ex_def->set_args(args_info.data(), sizeof(STR_FWK_OP_KERNEL));
  kernel_ex_def->set_args_size(sizeof(STR_FWK_OP_KERNEL));

  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length

  PisToArgs args;
  args[0].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_ex_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs);

  auto ret = kernel_ex_task_info.UpdateArgs();
  EXPECT_EQ(ret, SUCCESS);
  free(ValueToPtr(args[0].dev_addr));
  delete[] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  kernel_ex_task_info.Release();
}

TEST_F(UtestKernelExTaskInfo, testHeadFile) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = PtrToValue(malloc(1024));
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{0xc212, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs);

  auto ret = kernel_ex_task_info.CallSaveDumpInfo();
  EXPECT_EQ(ret, true);
  free(ValueToPtr(args[0].dev_addr));
}

}  // namespace ge
