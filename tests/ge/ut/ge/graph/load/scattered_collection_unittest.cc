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
#include <string.h>

#include "macro_utils/dt_public_scope.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "graph/def_types.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/zero_copy_offset.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.h"
#include "graph/load/model_manager/task_info/fe/fusion_start_task_info.h"
#include "graph/load/model_manager/task_info/fe/fusion_stop_task_info.h"
#include "graph/load/model_manager/task_info/rts/label_switch_by_index_task_info.h"
#include "graph/load/model_manager/task_info/rts/stream_active_task_info.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/reusable_stream_allocator.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;

namespace ge {
class UtestScatteredCollection : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// zero_copy_offset.cc
TEST_F(UtestScatteredCollection, ZeroCopyOffset_Invalid) {
  ZeroCopyOffset zero_copy_offset;
  int64_t data_size = 0;
  void *virtual_addr = nullptr;

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  (void)AttrUtils::SetListInt(op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, {1024});
  (void)AttrUtils::SetListInt(op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, {0});

  bool fusion_flag = true;
  auto ret = zero_copy_offset.InitInputDataInfo(data_size, virtual_addr, op_desc, fusion_flag);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<int64_t> input_size_list;
  std::vector<uint64_t> virtual_addr_list;
  size_t idx = 0;
  input_size_list.emplace_back(5);
  virtual_addr_list.emplace_back(PtrToValue(virtual_addr));
  ret = zero_copy_offset.InitOutputDataInfo(input_size_list, virtual_addr_list, op_desc, idx, fusion_flag);
  EXPECT_EQ(ret, SUCCESS);

  int64_t output_offset = 1024;
  uintptr_t addr_val = 0U;
  std::set<uint64_t> real_virtual_addrs;
  zero_copy_offset.SetInputOutsideAddrs(output_offset, addr_val, fusion_flag, real_virtual_addrs);

  int64_t input_offset = 1024;
  std::vector<uint64_t> tensor_addrs;
  zero_copy_offset.SetOutputOutsideAddrs(input_offset, fusion_flag, addr_val, tensor_addrs);

  uintptr_t outside_addr = 0U;
  uintptr_t args_base = 0U;
  size_t offset = 0;
  zero_copy_offset.valid_relative_offset_ = false;
  zero_copy_offset.SetOutsideAddrsValue(outside_addr, false, args_base, offset);

  uintptr_t logical_addr = 0U;
  uintptr_t device_addr = 0U;
  zero_copy_offset.SetLogicalOutsideAddrs(logical_addr, false, device_addr);
}

// ffts_plus_proto_transfer.cc
TEST_F(UtestScatteredCollection, FftsPlusProtoTransfer_InitManualAicAivCtx_Failed) {
  std::vector<uintptr_t> io_addrs;
  std::vector<void *> ext_args;
  std::set<size_t> mode_addr_idx;
  const RuntimeParam runtime_param;
  FftsPlusArgsHelper helper(runtime_param);
  FftsPlusProtoTransfer ffpt(0U, &helper, runtime_param, ext_args);
  ffpt.op_desc_ = std::make_shared<ge::OpDesc>("name", "type");

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  // InitFftsPlusTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICORE));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  rtFftsPlusAicAivCtx_t ctx;

  // Test: ctx_def.kernel_name_size() == 0
  EXPECT_EQ(ffpt.InitManualAicAivCtx(*aicaivctx, ctx), FAILED);

  // Test: ctx_def.kernel_name_size() != kAutoMixAicAivCtxPcNum
  aicaivdef->add_kernel_name("aictest1");
  aicaivdef->add_kernel_name("aictest2");
  EXPECT_EQ(ffpt.InitManualAicAivCtx(*aicaivctx, ctx), FAILED);

  // head file
  FftsRunAddrHandle handle1;
  FftsAddrPrefHandle handle2;
  FftsFindNodeHandle handle3;
  FftsSaveCtxArgsHandle handle4;

  ffpt.SetRunAddrHandle(handle1);
  ffpt.SetAddrPrefHandle(handle2);
  ffpt.SetFindNodeHandle(handle3);
  ffpt.SetSaveCtxArgsHandle(handle4);

  // TEST for default function
  const OpDescPtr op_desc;
  STR_FWK_OP_KERNEL fwk_op_kernel;
  const domi::FftsPlusAicpuCtxDef aicpu_ctx_def;
  const domi::aicpuKernelDef aicpu_kernle_def;
  EXPECT_EQ(ffpt.aicpu_get_session_id_(), 0U);
  EXPECT_EQ(ffpt.create_aicpu_session_(fwk_op_kernel), SUCCESS);
  EXPECT_EQ(ffpt.load_cust_aicpu_so_(op_desc, aicpu_ctx_def), SUCCESS);
  EXPECT_EQ(ffpt.save_aicpu_ctx_handle_(op_desc, aicpu_kernle_def), SUCCESS);
}

TEST_F(UtestScatteredCollection, FftsPlusProtoTransfer_InitAutoMixAicAivCtx) {
  std::vector<void *> ext_args;
  RuntimeParam runtime_param;
  runtime_param.logic_mem_base = 0x10;
  runtime_param.mem_size = 1000;
  runtime_param.mem_base = 0x2000;
  FftsPlusArgsHelper helper(runtime_param);
  FftsPlusProtoTransfer ffpt(0U, &helper, runtime_param, ext_args);
  ffpt.op_desc_ = std::make_shared<ge::OpDesc>("name", "type");
  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  ffpt.op_desc_->AddInputDesc(desc_temp);
  auto desc_out = *desc_temp_ptr;
  std::vector<int64_t> sub_offsets = {0,80};
  (void)ge::AttrUtils::SetListInt(desc_out, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_OFFSETS, sub_offsets);
  std::vector<uint32_t> ctx_ids = {20, 21};
  (void)ge::AttrUtils::SetListInt(desc_out, "_tensor_ctx_id", ctx_ids);
  ffpt.op_desc_->AddOutputDesc(desc_out);
  std::vector<int64_t> workspace;
  workspace.push_back(1024);
  sub_offsets[0] = 90;
  sub_offsets[1] = 90;
  (void)ge::AttrUtils::SetListInt(ffpt.op_desc_, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_OFFSETS, sub_offsets);
  ffpt.op_desc_->SetWorkspace(workspace);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(3);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef &mixctxdef = *mixaicaivctx->mutable_mix_aic_aiv_ctx();

  const auto ffts_run_addr_handle = [&runtime_param](const uintptr_t logic_addr, uint8_t *&mem_addr,
                                                     uint64_t &mem_type) -> Status {
    return ModelUtils::GetRtAddress(runtime_param, logic_addr, mem_addr, mem_type);
  };
  ffpt.SetRunAddrHandle(ffts_run_addr_handle);
  domi::FftsPlusCtxDef *data_ctx1 = ffts_plus_task_def->add_ffts_plus_ctx();
  data_ctx1->set_op_index(0);
  data_ctx1->set_context_id(20);
  data_ctx1->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  auto data1 = data_ctx1->mutable_data_ctx();
  data1->set_addr_base(0x10);
  data1->set_addr_offset(20);

  domi::FftsPlusCtxDef *data_ctx2 = ffts_plus_task_def->add_ffts_plus_ctx();
  data_ctx2->set_op_index(0);
  data_ctx2->set_context_id(21);
  data_ctx2->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  auto data2 = data_ctx2->mutable_data_ctx();
  data2->set_addr_base(0x10);
  data2->set_addr_offset(120);

  rtFftsPlusMixAicAivCtx_t ctx;
  uint32_t start_idx = 0;

  // Test: ctx_def.kernel_name_size() == 0
  mixctxdef.set_save_task_addr(1);
  mixctxdef.set_thread_dim(2);
  ctx.threadDim = mixctxdef.thread_dim();
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), FAILED);

  InitMixAicAivCtx(&mixctxdef, true);
  mixctxdef.set_thread_dim(2);
  ctx.threadDim = mixctxdef.thread_dim();
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), SUCCESS);
  std::vector<uint64_t> task_addr_offset;
  task_addr_offset = ffpt.op_desc_->TryGetExtAttr("task_addr_offset", task_addr_offset);
  EXPECT_EQ(task_addr_offset.size(), 3U);
  EXPECT_EQ(task_addr_offset[0], 32U);
  EXPECT_EQ(task_addr_offset[1], 80U);
  EXPECT_EQ(task_addr_offset[2], 90U);

  // auto thread memory_reuse
  rtFftsPlusComCtx_t res1;
  EXPECT_EQ(ffpt.InitDataCtx(*data_ctx1, &res1), SUCCESS);
  rtFftsPlusDataCtx_t *const rts_data_ctx1 = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(&res1);
  EXPECT_EQ(rts_data_ctx1->addressOffset, 0U);
  EXPECT_EQ(rts_data_ctx1->addressBaseL, 0x2000);
  EXPECT_EQ(rts_data_ctx1->addressBaseH, 0U);

  rtFftsPlusComCtx_t res2;
  EXPECT_EQ(ffpt.InitDataCtx(*data_ctx2, &res2), SUCCESS);
  rtFftsPlusDataCtx_t *const rts_data_ctx2 = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(&res2);
  EXPECT_EQ(rts_data_ctx2->addressOffset, 0U);
  EXPECT_EQ(rts_data_ctx2->addressBaseL, 0x2050);
  EXPECT_EQ(rts_data_ctx2->addressBaseH, 0U);


  start_idx = 7;
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), FAILED);
}

TEST_F(UtestScatteredCollection, FftsPlusProtoTransfer_InitManualMixAicAivCtx) {
  std::vector<void *> ext_args;
  RuntimeParam runtime_param;
  runtime_param.logic_mem_base = 0x10;
  runtime_param.mem_size = 1000;
  runtime_param.mem_base = 0x2000;
  FftsPlusArgsHelper helper(runtime_param);
  FftsPlusProtoTransfer ffpt(0U, &helper, runtime_param, ext_args);
  ffpt.op_desc_ = std::make_shared<ge::OpDesc>("name", "type");

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();

  rtFftsPlusMixAicAivCtx_t ctx;
  uint32_t start_idx = 0;
  vector<std::string> name_prefixes{"_mix_aic", "_mic_aiv"};
  // Test: ctx_def.kernel_name_size() == 0
  EXPECT_EQ(ffpt.InitManualMixAicAivCtx(*mixctxdef, name_prefixes, ctx, start_idx), FAILED);

  // Test: ctx_def.kernel_name_size() != kAutoMixAicAivCtxPcNum
  mixctxdef->add_kernel_name("aictest");
  EXPECT_EQ(ffpt.InitManualMixAicAivCtx(*mixctxdef, name_prefixes, ctx, start_idx), FAILED);

  mixctxdef->clear_kernel_name();
  InitMixAicAivCtx(mixctxdef, false);
  EXPECT_EQ(ffpt.InitManualMixAicAivCtx(*mixctxdef, name_prefixes, ctx, start_idx), SUCCESS);

//  vector<std::string> mix_name_prefixes {"_mix_enhanced"};
//  EXPECT_EQ(ffpt.InitManualMixAicAivCtx(*mixctxdef, mix_name_prefixes, ctx, start_idx), SUCCESS);
}

// fusion_start_task_info.cc
TEST_F(UtestScatteredCollection, FusionStartTaskInfo_Invalid) {
  FusionStartTaskInfo fstart;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  auto davinci_model = new DavinciModel(0, nullptr);

  auto ret = fstart.Init(task_def, davinci_model);
  EXPECT_NE(ret, SUCCESS);
  domi::GetContext().is_online_model = true;
  ret = fstart.Distribute();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(fstart.IsSupportReDistribute());
  EXPECT_EQ(fstart.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  delete davinci_model;
}

TEST_F(UtestScatteredCollection, FusionStartTaskInfo_Success) {
  FusionStartTaskInfo fstart;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  auto davinci_model = new DavinciModel(0, nullptr);

  rtStream_t stream = nullptr;
  davinci_model->reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model->reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model->stream_list_ = { stream };

  auto ret = fstart.Init(task_def, davinci_model);
  EXPECT_EQ(ret, SUCCESS);
  delete davinci_model;
}

// fusion_stop_task_info.cc
TEST_F(UtestScatteredCollection, FusionStopTaskInfo_Init_Invalid) {
  FusionStopTaskInfo fstop;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  auto davinci_model = new DavinciModel(0, nullptr);

  auto ret = fstop.Init(task_def, davinci_model);
  EXPECT_NE(ret, SUCCESS);
  domi::GetContext().is_online_model = true;
  ret = fstop.Distribute();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(fstop.IsSupportReDistribute());
  EXPECT_EQ(fstop.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  delete davinci_model;
}

TEST_F(UtestScatteredCollection, FusionStopTaskInfo_Init_Success) {
  FusionStopTaskInfo fstop;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  auto davinci_model = new DavinciModel(0, nullptr);

  rtStream_t stream = nullptr;
  davinci_model->reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model->reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model->stream_list_ = { stream };

  auto ret = fstop.Init(task_def, davinci_model);
  EXPECT_EQ(ret, SUCCESS);
  delete davinci_model;
}

TEST_F(UtestScatteredCollection, init_label_switch_by_index_task_info) {
  const PisToArgs args = {};
  PisToPersistentWorkspace persistent_workspace;
  persistent_workspace[static_cast<uint32_t>(ArgsPlacement::kArgsPlacementTs)].dev_addr = (uint64_t)malloc(1024);
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};

  domi::TaskDef task_def;
  domi::LabelSwitchByIndexDef *label_switch_by_index_def = task_def.mutable_label_switch_by_index();
  label_switch_by_index_def->set_op_index(0);
  label_switch_by_index_def->set_label_max(1);
  LabelSwitchByIndexTaskInfo task_info;

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  rtLabel_t label = nullptr;
  rtLabelCreate(&label);
  model.label_list_ = {label};

  GeTensorDesc desc;
  auto op_desc = std::make_shared<OpDesc>("LabelSwitchByIndex", LABELSWITCHBYINDEX);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset({8});
  model.op_list_[op_desc->GetId()] = op_desc;
  model.SetFeatureBaseRefreshable(true);

  AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, {0});
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistent_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(PtrToValue(task_info.index_value_), 0x1a23);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;

  free((void *)(persistent_workspace[static_cast<uint32_t>(ArgsPlacement::kArgsPlacementTs)].dev_addr));
}

TEST_F(UtestScatteredCollection, LabelSwitchByIndexTaskInf_HBM_test) {
  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  OpDescPtr op_desc = CreateOpDesc("label_switch", LABELSWITCHBYINDEX);
  op_desc->SetId(0);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::LabelSwitchByIndexDef *label_task_def = task_def.mutable_label_switch_by_index();
  label_task_def->set_op_index(op_desc->GetId());
  label_task_def->set_label_max(2);
  label_task_def->set_op_index(0);

  LabelSwitchByIndexTaskInfo task_info;
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 0x40000;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  model.op_list_[op_desc->GetId()] = op_desc;

  TaskRunParam task_run_param = {};
  (void)task_info.ParseOpIndex(task_def);
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset({1024});

  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.persistent_workspace_descs[0].placement, ArgsPlacement::kArgsPlacementHbm);

  mmSetEnv(kEnvRecordPath, "", 1);
}

// label_switch_by_index_task_info.cc
TEST_F(UtestScatteredCollection, LabelSwitchByIndexTaskInfo_test) {
  const PisToArgs args = {};
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};

  OpDescPtr op_desc = CreateOpDesc("label_switch", LABELSWITCHBYINDEX);
  op_desc->SetId(0);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::LabelSwitchByIndexDef *label_task_def = task_def.mutable_label_switch_by_index();
  label_task_def->set_op_index(op_desc->GetId());
  label_task_def->set_label_max(2);
  label_task_def->set_op_index(0);

  LabelSwitchByIndexTaskInfo task_info;
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 0x40000;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  // fail for OpDesc not found.
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  // fail for input num
  model.op_list_[op_desc->GetId()] = op_desc;
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), FAILED);
  int64_t op_index = task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);

  model.SetFeatureBaseRefreshable(true);
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), FAILED);

  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset({1024});

  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  // check task_run_param
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1);
  EXPECT_NE(task_run_param.parsed_input_addrs[0].logic_addr, 0);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].support_refresh, false);
  EXPECT_EQ(task_run_param.persistent_workspace_descs[0].placement, ArgsPlacement::kArgsPlacementTs);

  // fail for LABEL_SWITCH_LIST
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, {});
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, {0, 1});
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);
}

// stream_active_task_info.cc
TEST_F(UtestScatteredCollection, testStreamActiveTaskInfo) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  model.op_list_[0] = CreateOpDesc("data", DATA);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  StreamActiveTaskInfo sati;
  auto ret = sati.Init(task_def, &model);
  EXPECT_EQ(ret, INTERNAL_ERROR);
  domi::GetContext().is_online_model = true;
  ret = sati.Distribute();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(sati.IsSupportReDistribute());
  EXPECT_EQ(sati.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}
} // end of namespace ge
