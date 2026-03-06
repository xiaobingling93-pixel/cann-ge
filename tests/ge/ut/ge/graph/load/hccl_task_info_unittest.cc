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
#include <gmock/gmock.h>
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/ge_local_context.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "stub/gert_runtime_stub.h"
#include "depends/runtime/src/runtime_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/hccl/hccl_task_info.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include <iostream>

namespace ge {
namespace {
int32_t g_unload_called_count = 0;
class MockMemcpy : public RuntimeStub {
 public:
  MOCK_METHOD5(rtMemcpy, int32_t(void * , uint64_t, const void *, uint64_t, rtMemcpyKind_t));
};

class AclMockMemcpy : public AclRuntimeStub {
 public:
  MOCK_METHOD5(aclrtMemcpy, int32_t(void *, size_t, const void *, size_t, aclrtMemcpyKind));
};

class HcclOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  HcclOpsKernelInfoStore() = default;
  Status Initialize(const std::map<std::string, std::string> &options) override { return SUCCESS; }
  // close opsKernelInfoStore
  Status Finalize() override { return SUCCESS; }
  // get all opsKernelInfo
  void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const override {}
  // whether the opsKernelInfoStore is supported based on the operator attribute
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const override { return true; }
  Status UnloadTask(GETaskInfo &task) {
    g_unload_called_count++;
    return SUCCESS;
  }
};
class FailHcclOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  FailHcclOpsKernelInfoStore() = default;
  Status Initialize(const std::map<std::string, std::string> &options) override { return SUCCESS; }
  // close opsKernelInfoStore
  Status Finalize() override { return SUCCESS; }
  // get all opsKernelInfo
  void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const override {}
  // whether the opsKernelInfoStore is supported based on the operator attribute
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const override { return true; }
  Status LoadTask(GETaskInfo &task) { return FAILED; }
  Status UnloadTask(GETaskInfo &task) {
    g_unload_called_count++;
    return FAILED;
  }
};
}  // namespace

class UtestHcclTaskInfo : public testing::Test {
 protected:
  void SetUp() {
    g_unload_called_count = 0;
    auto acl_mock_memcpy = [](void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind) -> int {
      std::cout << "dst: " << dst << std::endl;
      if (count == 0) {
        return -1;
      }
      if (dst == nullptr || src == nullptr) {
        return -1;
      }
      if (dst != nullptr && src != nullptr) {
        memcpy_s(dst, dest_max, src, count);
      }
      return RT_ERROR_NONE;
    };
    auto acl_runtime_stub = std::make_shared<AclMockMemcpy>();
    AclRuntimeStub::SetInstance(acl_runtime_stub);
    EXPECT_CALL(*acl_runtime_stub, aclrtMemcpy).WillRepeatedly(testing::Invoke(acl_mock_memcpy));
  }
  void TearDown() {
    AclRuntimeStub::Reset();
  }
};

// test success GetTaskID
TEST_F(UtestHcclTaskInfo, success_get_task_id) {
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task->type()));

  EXPECT_EQ(task_info->GetTaskID(), 0);

  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.GetTaskID(), 0);
}

TEST_F(UtestHcclTaskInfo, test_SetFollowStream) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.main_follow_stream_mapping_[0].emplace_back(stream);

  std::map<std::string, std::string> session_option{{"ge.exec.overflow", "1"}};
  GetThreadLocalContext().SetSessionOption(session_option);

  HcclTaskInfo hccl_task_info;
  auto op_desc = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  AttrUtils::SetInt(op_desc, "used_stream_num", 1);
  hccl_task_info.davinci_model_ = &model;
  EXPECT_EQ(hccl_task_info.SetFollowStream(op_desc), SUCCESS);

  AttrUtils::SetInt(op_desc, "used_stream_num", 2);
  EXPECT_EQ(hccl_task_info.SetFollowStream(op_desc), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestHcclTaskInfo, success_task_init_args) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[1].dev_addr = 3;
  iow_addrs.input_logic_addrs = {{1, 0}};
  iow_addrs.output_logic_addrs = {{2, 0}};
  iow_addrs.workspace_logic_addrs = {{1, 0}};

  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  OpsKernelExecutorManager::GetInstance().Initialize({});
  OpsKernelExecutorManager::GetInstance().executors_[kEngineNameHccl] = std::make_shared<HcclOpsKernelInfoStore>();

  domi::TaskDef task_def;
  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomAllReduce");
  GeTensorDesc desc;
  auto op_desc = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, 0);
  AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "min");
  AttrUtils::SetBool(op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, true);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->SetInputOffset({8});
  op_desc->SetWorkspace({800});
  op_desc->SetWorkspaceBytes({150});
  op_desc->SetOutputOffset({8});
  op_desc->SetOpKernelLibName(kEngineNameHccl);

  model.feature_base_refreshable_ = true;
  model.runtime_param_.mem_size = 1024;
  MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  model.op_list_[op_desc->GetId()] = op_desc;
  std::set<std::string> temp;
  model.data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  HcclTaskInfo hccl_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(hccl_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  EXPECT_EQ(hccl_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(PtrToValue(hccl_task_info.input_data_addrs_[0]), 1);
  EXPECT_EQ(hccl_task_info.input_mem_types_[0], 0);
  EXPECT_EQ(PtrToValue(hccl_task_info.output_data_addrs_[0]), 2);
  EXPECT_EQ(hccl_task_info.output_mem_types_[0], 0);
  EXPECT_EQ(PtrToValue(hccl_task_info.workspace_addrs_[0]), 1);
  EXPECT_EQ(hccl_task_info.workspace_mem_types_[0], 0);
  EXPECT_EQ(PtrToValue(hccl_task_info.args_), 3);

  uint8_t host_args[1024] = {};
  EXPECT_EQ(hccl_task_info.UpdateDumpInfos(host_args, 1024), SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(hccl_task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_EQ(infos.size(), 3UL);

  task_def.clear_kernel_hccl();
}

// test hccl_init
TEST_F(UtestHcclTaskInfo, success_task_init) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  model.stream_flag_list_ = {0, 0};
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  OpsKernelExecutorManager::GetInstance().Initialize({});
  OpsKernelExecutorManager::GetInstance().executors_[kEngineNameHccl] = std::make_shared<HcclOpsKernelInfoStore>();

  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation absolut_mem_allocation = {1, 0, 0xffffffffffff, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(absolut_mem_allocation);

  domi::TaskDef task_def;
  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomAllReduce");
  GeTensorDesc desc;
  auto op_desc = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, 0);
  AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "min");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::TensorUtils::SetSize(*(op_desc->MutableInputDesc(0)), 8);
  ge::TensorUtils::SetSize(*(op_desc->MutableOutputDesc(0)), 8);
  op_desc->SetInputOffset({8});
  op_desc->SetOutputOffset({16});
  op_desc->SetWorkspace({24});
  op_desc->SetWorkspaceBytes({150});
  op_desc->SetOpKernelLibName(kEngineNameHccl);
  model.op_list_[op_desc->GetId()] = op_desc;
  HcclTaskInfo hccl_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(hccl_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  task_run_param.parsed_input_addrs[0].memory_type = 1;
  task_run_param.parsed_output_addrs[0].memory_type = 1;
  task_run_param.parsed_workspace_addrs[0].memory_type = 1;
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(hccl_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  std::vector<IowPaRemapInfo> remap_infos;
  EXPECT_EQ(hccl_task_info.GetTaskIowPaRemapInfos(remap_infos), SUCCESS);
  EXPECT_EQ(remap_infos.size(), 3U);
  EXPECT_EQ(remap_infos[0].policy, PaRemapPolicy::KNoSupport);
  EXPECT_EQ(remap_infos[0].allocation_id, 0U);
  EXPECT_EQ(remap_infos[0].allocation_offset, 8U);
  EXPECT_EQ(remap_infos[0].tensor_size, 8U);
  EXPECT_EQ(remap_infos[1].policy, PaRemapPolicy::KNoSupport);
  EXPECT_EQ(remap_infos[1].allocation_id, 0U);
  EXPECT_EQ(remap_infos[1].allocation_offset, 16U);
  EXPECT_EQ(remap_infos[1].tensor_size, 8U);
  EXPECT_EQ(remap_infos[2].policy, PaRemapPolicy::KNoSupport);
  EXPECT_EQ(remap_infos[2].allocation_id, 0U);
  EXPECT_EQ(remap_infos[2].allocation_offset, 24U);
  EXPECT_EQ(remap_infos[2].tensor_size, 150U);

  domi::TaskDef task_def1;
  domi::KernelHcclDef *kernel_hccl_def1 = task_def1.mutable_kernel_hccl();
  kernel_hccl_def1->set_op_index(1);
  auto op_desc1 = std::make_shared<OpDesc>("hvd_wait", HVDWAIT);
  op_desc1->SetStreamId(0);
  op_desc1->SetId(1);
  op_desc1->SetOpKernelLibName(kEngineNameHccl);
  model.op_list_[op_desc1->GetId()] = op_desc1;
  HcclTaskInfo hccl_task_info1;
  EXPECT_EQ(hccl_task_info1.Init(task_def1, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  model.feature_base_refreshable_ = true;
  HcclTaskInfo hccl_task_info2;
  EXPECT_EQ(hccl_task_info2.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  task_def.clear_kernel_hccl();
  task_def1.clear_kernel_hccl();
  OpsKernelExecutorManager::GetInstance().executors_.clear();
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
}

TEST_F(UtestHcclTaskInfo, hccl_task_overflow_dump) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[1].dev_addr = 3;
  iow_addrs.input_logic_addrs = {{1, 0}};
  iow_addrs.output_logic_addrs = {{2, 0}};
  iow_addrs.workspace_logic_addrs = {{1, 0}};

  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  model.stream_flag_list_ = {0};
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  OpsKernelExecutorManager::GetInstance().Initialize({});
  OpsKernelExecutorManager::GetInstance().executors_[kEngineNameHccl] = std::make_shared<HcclOpsKernelInfoStore>();

  domi::TaskDef task_def;
  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomAllReduce");
  auto op_desc = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, 0);
  AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "min");
  AttrUtils::SetBool(op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, true);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  GeTensorDesc desc(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(desc, 512);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->SetInputOffset({8});
  op_desc->SetWorkspaceBytes({150});
  op_desc->SetOpKernelLibName(kEngineNameHccl);
  model.op_list_[op_desc->GetId()] = op_desc;
  HcclTaskInfo hccl_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(hccl_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(hccl_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  model.is_op_debug_reg_ = true; // open overflow dump
  GeTensorDesc input_desc(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_STRING);
  op_desc->UpdateInputDesc(0, input_desc);
  hccl_task_info.PostProcess(task_def);
  // unsupported hccl op data_type
  EXPECT_TRUE(model.data_dumper_.op_list_.empty());
  task_def.clear_kernel_hccl();
  OpsKernelExecutorManager::GetInstance().executors_.clear();
}

// test hccl_init
TEST_F(UtestHcclTaskInfo, fail_task_init) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.Init(task_def, &model), FAILED);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  model.stream_flag_list_ = {0};

  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomAllReduce");
  auto op_desc = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetInputOffset({8});
  op_desc->SetWorkspaceBytes({150});
  model.op_list_[op_desc->GetId()] = op_desc;

  // fail for GetHcclDataType
  HcclTaskInfo hccl_task_info1;
  EXPECT_EQ(hccl_task_info1.Init(task_def, &model), PARAM_INVALID);

  // fail for GetAllRootId
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  HcclTaskInfo hccl_task_info2;
  EXPECT_EQ(hccl_task_info2.Init(task_def, &model), FAILED);

  // fail for SetAddrs
  AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, 0);
  HcclTaskInfo hccl_task_info3;
  EXPECT_EQ(hccl_task_info3.Init(task_def, &model), PARAM_INVALID);

  task_def.clear_kernel_hccl();
}

// test hccl_GetPrivateDefByTaskDef
TEST_F(UtestHcclTaskInfo, success_hccl_get_private_def_by_task_def) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = 3;
  iow_addrs.input_logic_addrs = {{1, 0}};
  iow_addrs.output_logic_addrs = {{2, 0}};
  iow_addrs.workspace_logic_addrs = {{1, 0}};

  DavinciModel model(0, nullptr);
  model.op_list_[0] = std::make_shared<OpDesc>("FrameworkOp", "FrameworkOp");
  model.op_list_[0]->SetOpKernelLibName(kEngineNameHccl);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task7 = model_task_def.add_task();
  domi::KernelHcclDef *kernel_hccl_def = task7->mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  task7->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
  // for SetStream
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  model.stream_flag_list_.push_back(0);
  MemAllocation not_change_mem_item = {0U, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  // for GetPrivateDefByTaskDef
  std::string value = "hccl_task";
  task7->set_private_def(value);

  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task7->type()));

  OpsKernelExecutorManager::GetInstance().Initialize({});
  OpsKernelExecutor *p_executor = nullptr;
  OpsKernelExecutorManager::GetInstance().executors_[kEngineNameHccl] = std::make_shared<HcclOpsKernelInfoStore>();
  OpsKernelExecutorManager::GetInstance().GetExecutor(kEngineNameHccl, p_executor);
  EXPECT_TRUE(p_executor != nullptr);
  // for Distribute
  EXPECT_EQ(task_info7->Init(task7[0], &model, args, persistant_workspace, iow_addrs), SUCCESS);
  OpsKernelExecutorManager::GetInstance().executors_.clear();
}

TEST_F(UtestHcclTaskInfo, test_hccl_task_distribute_with_attached_stream) {
  auto kernel_info_store = std::make_shared<HcclOpsKernelInfoStore>();
  HcclTaskInfo hccl_task_info;
  DavinciModel model(0, nullptr);
  model.SetOmName("test_hccl_om");
  model.SetDumpModelName("test_hccl_model");
  model.stream_list_.push_back(reinterpret_cast<void *>(&model));
  hccl_task_info.davinci_model_ = &model;
  hccl_task_info.hccl_op_desc_ = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  std::vector<NamedAttrs> attached_stream_infos;
  NamedAttrs attrs0;
  NamedAttrs attrs1;
  attached_stream_infos.push_back(attrs0);
  attached_stream_infos.push_back(attrs1);
  AttrUtils::SetListNamedAttrs(hccl_task_info.hccl_op_desc_, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
  hccl_task_info.hccl_op_desc_->SetAttachedStreamIds({-1, 0});
  // ops_kernel_info_store return success
  hccl_task_info.ops_kernel_store_ = kernel_info_store.get();
  EXPECT_EQ(hccl_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(hccl_task_info.Release(), SUCCESS);
  model.stream_list_.clear();
}

TEST_F(UtestHcclTaskInfo, test_hccl_task_distribute_release) {
  auto kernel_info_store = std::make_shared<HcclOpsKernelInfoStore>();
  HcclTaskInfo hccl_task_info;
  DavinciModel model(0, nullptr);
  DumpProperties dump_properties;
  // static hccl no dump
  model.SetDumpProperties(dump_properties);
  model.SetOmName("test_hccl_om");
  model.SetDumpModelName("test_hccl_model");
  hccl_task_info.davinci_model_ = &model;
  OpDesc hccl_op_desc;
  hccl_op_desc.SetName("Test_hccl");
  hccl_op_desc.SetType("HcomAllGather");
  hccl_task_info.hccl_op_desc_ = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  std::cout<<"start distribute" <<std::endl;
  // without ops_kernel_info_store, fail
  EXPECT_EQ(hccl_task_info.Distribute(), INTERNAL_ERROR);
  EXPECT_EQ(hccl_task_info.Release(), INTERNAL_ERROR);

  // ops_kernel_info_store return success
  hccl_task_info.ops_kernel_store_ = kernel_info_store.get();
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(hccl_task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(hccl_task_info.IsSupportReDistribute());
  EXPECT_EQ(hccl_task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  EXPECT_EQ(hccl_task_info.Release(), SUCCESS);
  ASSERT_EQ(g_unload_called_count, 1);

  // ops_kernel_info_store return failed
  auto fail_kernel_info_store = std::make_shared<FailHcclOpsKernelInfoStore>();
  hccl_task_info.ops_kernel_store_ = fail_kernel_info_store.get();
  EXPECT_EQ(hccl_task_info.Distribute(), INTERNAL_ERROR);
  EXPECT_EQ(hccl_task_info.Release(), INTERNAL_ERROR);
  ASSERT_EQ(g_unload_called_count, 2);
}

TEST_F(UtestHcclTaskInfo, test_hccl_task_dump_all) {
  auto kernel_info_store = std::make_shared<HcclOpsKernelInfoStore>();
  HcclTaskInfo hccl_task_info;
  DavinciModel model(0, nullptr);
  DumpProperties dump_properties;
  // static hccl no dump
  model.SetDumpProperties(dump_properties);
  model.SetOmName("test_hccl_om");
  model.SetDumpModelName("test_hccl_model");
  hccl_task_info.davinci_model_ = &model;
  OpDesc hccl_op_desc;
  hccl_op_desc.SetName("Test_hccl");
  hccl_op_desc.SetType("HcomAllGather");
  hccl_task_info.hccl_op_desc_ = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  std::cout << "start distribute" << std::endl;
  hccl_task_info.ops_kernel_store_ = kernel_info_store.get();

  // static hccl dump all
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  model.SetDumpProperties(dump_properties);
  EXPECT_EQ(hccl_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(hccl_task_info.Release(), SUCCESS);
  ASSERT_EQ(g_unload_called_count, 1);
}

TEST_F(UtestHcclTaskInfo, test_hccl_task_dump_input) {
  auto kernel_info_store = std::make_shared<HcclOpsKernelInfoStore>();
  HcclTaskInfo hccl_task_info;
  DavinciModel model(0, nullptr);
  DumpProperties dump_properties;
  // static hccl no dump
  model.SetDumpProperties(dump_properties);
  model.SetOmName("test_hccl_om");
  model.SetDumpModelName("test_hccl_model");
  hccl_task_info.davinci_model_ = &model;
  OpDesc hccl_op_desc;
  hccl_op_desc.SetName("Test_hccl");
  hccl_op_desc.SetType("HcomAllGather");
  hccl_task_info.hccl_op_desc_ = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  std::cout<<"start distribute" <<std::endl;
  hccl_task_info.ops_kernel_store_ = kernel_info_store.get();
  // static hccl dump input
  dump_properties.SetDumpMode("input");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  model.SetDumpProperties(dump_properties);
  EXPECT_EQ(hccl_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(hccl_task_info.Release(), SUCCESS);
  ASSERT_EQ(g_unload_called_count, 1);
}

TEST_F(UtestHcclTaskInfo, test_hccl_task_dump_output) {
  auto kernel_info_store = std::make_shared<HcclOpsKernelInfoStore>();
  HcclTaskInfo hccl_task_info;
  DavinciModel model(0, nullptr);
  DumpProperties dump_properties;
  // static hccl no dump
  model.SetDumpProperties(dump_properties);
  model.SetOmName("test_hccl_om");
  model.SetDumpModelName("test_hccl_model");
  hccl_task_info.davinci_model_ = &model;
  OpDesc hccl_op_desc;
  hccl_op_desc.SetName("Test_hccl");
  hccl_op_desc.SetType("HcomAllGather");
  hccl_task_info.hccl_op_desc_ = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  std::cout<<"start distribute" <<std::endl;
  hccl_task_info.ops_kernel_store_ = kernel_info_store.get();
  // static hccl dump output
  dump_properties.SetDumpMode("output");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  model.SetDumpProperties(dump_properties);
  EXPECT_EQ(hccl_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(hccl_task_info.Release(), SUCCESS);
  ASSERT_EQ(g_unload_called_count, 1);
}

TEST_F(UtestHcclTaskInfo, Calculate_Update_Args) {
  dlog_setlevel(-1, 0, 1);
  auto mock_memcpy = [](void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) -> int {
    std::cout << "dst: " << dst << std::endl;
    if (count == 0) {
      return -1;
    }
    if (dst == nullptr || src == nullptr) {
      return -1;
    }
    if (dst != nullptr && src != nullptr) {
      memcpy_s(dst, dest_max, src, count);
    }
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockMemcpy>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(mock_memcpy));

  auto acl_mock_memcpy = [](void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind) -> int {
    std::cout << "dst: " << dst << std::endl;
    if (count == 0) {
      return -1;
    }
    if (dst == nullptr || src == nullptr) {
      return -1;
    }
    if (dst != nullptr && src != nullptr) {
      memcpy_s(dst, dest_max, src, count);
    }
    return RT_ERROR_NONE;
  };
  auto acl_runtime_stub = std::make_shared<AclMockMemcpy>();
  AclRuntimeStub::SetInstance(acl_runtime_stub);
  EXPECT_CALL(*acl_runtime_stub, aclrtMemcpy).WillRepeatedly(testing::Invoke(acl_mock_memcpy));

  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  model.stream_flag_list_ = {0};
  model.op_list_[0] = std::make_shared<OpDesc>("hccl_op", "HcomBroadcast");
  GeTensorDesc ge_tensor;
  model.op_list_[0]->AddInputDesc(ge_tensor);
  model.op_list_[0]->AddOutputDesc(ge_tensor);
  domi::TaskDef task_def;
  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomBroadcast");

  HcclTaskInfo hccl_task_info;
  TaskRunParam task_run_param = {};
  auto ret = hccl_task_info.ParseTaskRunParam(task_def, &model, task_run_param);
  EXPECT_EQ(ret, SUCCESS);
  hccl_task_info.Init(task_def, &model);
  hccl_task_info.io_addrs_.push_back(12405000);
  hccl_task_info.io_addrs_.push_back(12405100);
  int64_t op_index = hccl_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  model.SetDumpProperties(dump_properties);
  std::vector<int64_t> host_args({0,0});
  std::vector<uint64_t> active_base_addr;
  active_base_addr.resize(model.logical_mem_allocations_.size());
  for (size_t i = 0; i < model.logical_mem_allocations_.size(); i++) {
    active_base_addr.emplace_back(model.allocation_ids_to_active_base_addr_[i]);
  }
  ret = hccl_task_info.UpdateHostArgs(active_base_addr, static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_EQ(ret, SUCCESS);
  hccl_task_info.InsertDumpOp("input");
  hccl_task_info.InsertDumpOp("output");
  active_base_addr.push_back(12405233);
  active_base_addr.push_back(12406543);
  active_base_addr.push_back(12405409);
  MemAllocationAndOffset v1 = {0, 120};
  MemAllocationAndOffset v2 = {2, 0};
  hccl_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v1);
  hccl_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v2);
  ret = hccl_task_info.UpdateHostArgs(active_base_addr, static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(host_args[0], 12405353);
  EXPECT_EQ(host_args[1], 12405409);
  (void)rtFree(hccl_task_info.input_hccl_dump_.proto_size_dev_mem_);
  hccl_task_info.input_hccl_dump_.proto_size_dev_mem_ = nullptr;
  ret = hccl_task_info.UpdateHostArgs(active_base_addr, static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_NE(ret, SUCCESS);
  (void)rtFree(hccl_task_info.input_hccl_dump_.proto_dev_mem_);
  hccl_task_info.input_hccl_dump_.proto_dev_mem_ = nullptr;
  ret = hccl_task_info.UpdateHostArgs(active_base_addr, static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_NE(ret, SUCCESS);
  task_def.clear_kernel_hccl();
  RuntimeStub::SetInstance(nullptr);
  AclRuntimeStub::SetInstance(nullptr);
  dlog_setlevel(-1, 3, 0);
}

TEST_F(UtestHcclTaskInfo, AllToAll_GetCount_Success) {
  auto op_desc = std::make_shared<OpDesc>("hcom_alltoall", HCOMALLTOALL);
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos;
  GETaskKernelHcclInfo kernel_hccl_info;
  kernel_hccl_infos.emplace_back(kernel_hccl_info);
  kernel_hccl_infos[0U].dataType = 1;
  auto ret = HcomOmeUtil::GetHcclCount(op_desc, kernel_hccl_infos);
  EXPECT_EQ(ret, SUCCESS);
}
}  // namespace ge
