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
#include "base/registry/op_impl_space_registry_v2.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/common_kernel_impl/op_tiling_kernel.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "common/tiling_fwk_data_helper.h"
// v1 need
#include "register/op_tiling/op_tiling_constants.h"
#include "register/op_tiling/op_compile_info_manager.h"
#include "register/op_tiling_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "stub/gert_runtime_stub.h"
#include "common/op_tiling/tiling_dfx.h"
#include "depends/profiler/src/dump_stub.h"

namespace gert {
namespace {
ge::graphStatus SuccessTilingFunc(TilingContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus FailedTilingFunc(TilingContext *context) {
  return ge::GRAPH_FAILED;
}
ge::graphStatus SuccessTilingFuncWithWorkspace(TilingContext *context) {
  auto ws = context->GetWorkspaceSizes(3);
  ws[0] = 100;
  ws[1] = 600;
  ws[2] = 512;
  return ge::GRAPH_SUCCESS;
}
}  // namespace
struct TilingUT : public testing::Test {
  TilingUT() {
    tiling = KernelRegistry::GetInstance().FindKernelFuncs("Tiling");
    tilingParse = KernelRegistry::GetInstance().FindKernelFuncs("TilingParse");
    findTilingFunc = KernelRegistry::GetInstance().FindKernelFuncs("FindTilingFunc");
    BuildOpTilingUnmanagedTensorData =
        KernelRegistry::GetInstance().FindKernelFuncs("BuildOpTilingUnmanagedTensorData");
    BuildOpTilingOutputShape = KernelRegistry::GetInstance().FindKernelFuncs("BuildOpTilingOutputShape");
    fake_launch_arg_holder.reset(new (std::nothrow) uint8_t[64]);
    fake_launch_arg = reinterpret_cast<RtKernelLaunchArgsEx *>(fake_launch_arg_holder.get());
  }
  const KernelRegistry::KernelFuncs *tiling;
  const KernelRegistry::KernelFuncs *tilingParse;
  const KernelRegistry::KernelFuncs *findTilingFunc;
  const KernelRegistry::KernelFuncs *BuildOpTilingUnmanagedTensorData;
  const KernelRegistry::KernelFuncs *BuildOpTilingOutputShape;
  std::unique_ptr<uint8_t[]> fake_launch_arg_holder;
  RtKernelLaunchArgsEx *fake_launch_arg;
};

TEST_F(TilingUT, BuildTilingOutputs_Success) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto run_context = gert::TilingContextFaker()
                         .NodeIoNum(1, 1)
                         .IrInputNum(1)
                         .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                         .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                         .InputShapes({&in_shape})
                         .OutputShapes({&out_shape})
                         .TilingData(param.get())
                         .Build();

  ASSERT_EQ(tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
}

TEST_F(TilingUT, BuildTilingOutputs_Failed_WrongOutputNum) {
  auto run_context = BuildKernelRunContext(1, 4);
  ASSERT_NE(tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
}

TEST_F(TilingUT, Tiling_Failed_FuncNullptr) {
  auto run_context = BuildKernelRunContext(2, 6);
  ASSERT_EQ(tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(kernel::Tiling(run_context), ge::GRAPH_SUCCESS);
  run_context.FreeAll();
}

UINT32 StubFailedTiling(KernelContext *) {
  return 0x01;
}

TEST_F(TilingUT, Tiling_Failed_WhenTilingFuncFailed) {
  auto run_context = BuildKernelRunContext(3, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  auto tiling_data = TilingData::CreateCap(10);
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(StubFailedTiling),
                                    .launch_arg = fake_launch_arg};
  run_context.value_holder[0].Set(&fwk_data, nullptr);
  run_context.value_holder[5].Set(tiling_data.get(), nullptr);
  ASSERT_EQ(tiling->run_func(run_context), 0x01);
}

TEST_F(TilingUT, Tiling_PrintTilingData_Success) {
  auto run_context = BuildKernelRunContext(2, 5);
  auto tiling_data = TilingData::CreateCap(10);
  run_context.value_holder[5].Set(tiling_data.get(), nullptr);
  auto msgs = tiling->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1U);
  ASSERT_TRUE(msgs[0].find("TilingData: ") != string::npos);
}

TEST_F(TilingUT, Tiling_PrintTilingCacheStatus_Success) {
  auto run_context = BuildKernelRunContext(2, gert::TilingContext::kOutputNum + 1);
  auto holder = CreateLaunchArg(2, 8);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  run_context.value_holder[5].Set(&(args->GetTilingData()), nullptr);
  run_context.value_holder[gert::TilingContext::kOutputNum + 2].Set(args, nullptr);
  auto msgs = tiling->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1U);
  ASSERT_TRUE(msgs[0].find("Tiling cache status: disabled, ") != string::npos);
  args->SetTilingCacheStatus(RtKernelLaunchArgsEx::TilingCacheStatus::kHit);
  msgs = tiling->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1U);
  ASSERT_TRUE(msgs[0].find("Tiling cache status: hit, ") != string::npos);
}

TEST_F(TilingUT, Tiling_Data_Null_PrintTilingData_Failed) {
  auto run_context = BuildKernelRunContext(2, 5);
  run_context.value_holder[5].Set(nullptr, nullptr);
  auto msgs = tiling->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 0U);
}

TEST_F(TilingUT, FindTilingFunc_Failed_NullNodeType) {
  auto run_context = BuildKernelRunContext(1, 1);
  ASSERT_NE(findTilingFunc->run_func(run_context), ge::GRAPH_SUCCESS);
}

UINT32 StubTiling(TilingContext *) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(TilingUT, FindTilingFunc_Success) {
  auto run_context = BuildKernelRunContext(2, 1);
  const char *node_type = "null_tiling";
  run_context.value_holder[0].Set(const_cast<char *>(node_type), nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)->tiling
      = StubTiling;

  auto space_registry = std::make_shared<OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  space_registry->CreateOrGetOpImpl(node_type)->tiling = StubTiling;
  run_context.value_holder[1].Set(space_registry.get(), nullptr);

  ASSERT_EQ(findTilingFunc->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*run_context.GetContext<KernelContext>()->GetOutputPointer<OpImplKernelRegistry::TilingKernelFunc>(0),
            &StubTiling);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)->tiling = nullptr;
}

TEST_F(TilingUT, test_tiling_parse_create_output_failed_on_empty_node_type) {
  auto run_context = BuildKernelRunContext(1, 1);
  ASSERT_NE(tilingParse->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
}

void *StubCreateCompileInfo() {
  return nullptr;
}

TEST_F(TilingUT, test_tiling_parse_create_output_success) {
  auto run_context = BuildKernelRunContext(3, 1);
  const char *node_type = "null_tiling";
  run_context.value_holder[1].Set(const_cast<char *>(node_type), nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)
      ->compile_info_creator = StubCreateCompileInfo;

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  space_registry->CreateOrGetOpImpl(node_type)->compile_info_creator = StubCreateCompileInfo;
  run_context.value_holder[2].Set(space_registry.get(), nullptr);

  ASSERT_EQ(tilingParse->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)
      ->compile_info_creator = nullptr;
}

TEST_F(TilingUT, test_tiling_parse_create_output_is_null_follow_default_impl) {
  auto run_context = BuildKernelRunContext(3, 1);
  const char *node_type = "null_tiling";
  run_context.value_holder[1].Set(const_cast<char *>(node_type), nullptr);
  // set compile_info_creator is null, make it follow default impl of compile_info_creator
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)
      ->compile_info_creator = nullptr;

  auto space_registry = SpaceRegistryFaker().Build();
  ASSERT_NE(space_registry, nullptr);
  space_registry->CreateOrGetOpImpl(node_type)->compile_info_creator = nullptr;
  run_context.value_holder[2].Set(space_registry.get(), nullptr);

  ASSERT_EQ(tilingParse->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)
      ->compile_info_creator = nullptr;
  run_context.FreeAll();
}

TEST_F(TilingUT, test_tiling_parse_run_failed) {
  auto run_context = BuildKernelRunContext(1, 1);
  ASSERT_NE(tilingParse->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(TilingUT, test_tiling_parse_func_not_exsited) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "null_tiling";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_NE(tilingParse->run_func(run_context), ge::GRAPH_SUCCESS);
}

UINT32 StubTilingParse(KernelContext *context) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(TilingUT, test_tiling_parse_func_run_success) {
  auto run_context = BuildKernelRunContext(3, 1);
  const char *node_type = "null_tiling_parse";
  run_context.value_holder[1].Set(const_cast<char *>(node_type), nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)
      ->tiling_parse = StubTilingParse;

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  space_registry->CreateOrGetOpImpl(node_type)->tiling_parse = StubTilingParse;
  run_context.value_holder[2].Set(space_registry.get(), nullptr);

  ASSERT_EQ(tilingParse->run_func(run_context), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)
      ->tiling_parse = nullptr;
}

TEST_F(TilingUT, test_build_unmanage_tensor_data_success) {
  uint64_t test_tilling_key = 1000UL;
  uint32_t block_dim = 16U;
  std::vector<int64_t> work_space = {10, 1};
  bool clear_atomic = false;
  int32_t tilling_cond = 100;
  std::string buff = "buff test";
  std::vector<int64_t> tmp_buff1{1};
  int64_t logic_stream_id = 0;

  auto run_context = BuildKernelRunContext(static_cast<size_t>(gert::OpTilingExtendInputs::kOpTilingExtendInputsEnds),
                                           kOpTilingOutputSize + 1);
  run_context.value_holder[0].Set(&test_tilling_key, nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(block_dim), nullptr);
  run_context.value_holder[2].Set(&clear_atomic, nullptr);
  run_context.value_holder[3].Set(const_cast<char *>(buff.c_str()), nullptr);
  run_context.value_holder[4].Set(&work_space[0], nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(tilling_cond), nullptr);
  run_context.value_holder[7].Set(&logic_stream_id, nullptr);
  run_context.value_holder[static_cast<uint32_t>(gert::OpTilingExtendInputs::kStreamId)].Set(&logic_stream_id, nullptr);

  ASSERT_EQ(BuildOpTilingUnmanagedTensorData->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(BuildOpTilingUnmanagedTensorData->run_func(run_context), ge::GRAPH_SUCCESS);
  auto context = reinterpret_cast<KernelContext *>(run_context.context);
  auto gtd1 = context->GetOutputPointer<GertTensorData>(0U);
  auto gtd2 = context->GetOutputPointer<GertTensorData>(1U);
  auto gtd3 = context->GetOutputPointer<GertTensorData>(2U);
  auto gtd4 = context->GetOutputPointer<GertTensorData>(3U);
  char *tilling_data_out_addr = (static_cast<char *>(gtd1->GetAddr()));
  ASSERT_EQ(tilling_data_out_addr, const_cast<char *>(buff.c_str()));
  ASSERT_EQ((static_cast<uint64_t *>(gtd2->GetAddr())), &test_tilling_key);
  ASSERT_EQ(*(static_cast<uint32_t *>(gtd3->GetAddr())), block_dim);
  ASSERT_EQ(*(static_cast<uint32_t *>(gtd4->GetAddr())), tilling_cond);
  ASSERT_EQ(gtd1->GetPlacement(), kOnHost);
  ASSERT_EQ(gtd2->GetPlacement(), kOnHost);
  ASSERT_EQ(gtd3->GetPlacement(), kOnHost);
  ASSERT_EQ(gtd4->GetPlacement(), kOnHost);
}

TEST_F(TilingUT, test_build_unmanage_tensor_data_input_null_fail) {
  auto run_context = BuildKernelRunContext(8, kOpTilingOutputSize + 1);

  ASSERT_NE(BuildOpTilingUnmanagedTensorData->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(TilingUT, test_build_unmanage_tensor_data_output_null_failed) {
  uint64_t test_tilling_key = 1000UL;
  uint32_t block_dim = 16U;
  std::vector<int64_t> work_space = {10, 1};
  bool clear_atomic = false;
  int32_t tilling_cond = 1;
  std::string buff = "buff test";
  int64_t logic_stream_id = 0;
  auto run_context = BuildKernelRunContext(8, kOpTilingOutputSize + 1);

  run_context.value_holder[0].Set(&test_tilling_key, nullptr);
  run_context.value_holder[1].Set(&block_dim, nullptr);
  run_context.value_holder[2].Set(&clear_atomic, nullptr);
  run_context.value_holder[3].Set(const_cast<char *>(buff.c_str()), nullptr);
  run_context.value_holder[4].Set(&work_space[0], nullptr);
  run_context.value_holder[5].Set(&tilling_cond, nullptr);
  run_context.value_holder[7].Set(&logic_stream_id, nullptr);
  ASSERT_NE(BuildOpTilingUnmanagedTensorData->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(TilingUT, test_build_unmanage_tensor_data_tiling_cond_invalid) {
  uint64_t test_tilling_key = 1000UL;
  uint32_t block_dim = 16U;
  std::vector<int64_t> work_space = {10, 1};
  bool clear_atomic = false;
  int32_t tilling_cond = -10;
  std::string buff = "buff test";
  int64_t logic_stream_id = 0;
  auto run_context = BuildKernelRunContext(8, kOpTilingOutputSize + 1);

  run_context.value_holder[0].Set(&test_tilling_key, nullptr);
  run_context.value_holder[1].Set(&block_dim, nullptr);
  run_context.value_holder[2].Set(&clear_atomic, nullptr);
  run_context.value_holder[3].Set(const_cast<char *>(buff.c_str()), nullptr);
  run_context.value_holder[4].Set(&work_space[0], nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(tilling_cond), nullptr);
  run_context.value_holder[7].Set(&logic_stream_id, nullptr);

  GertTensorData td1;
  GertTensorData td2;
  GertTensorData td3;
  GertTensorData td4;
  run_context.value_holder[8].Set(const_cast<GertTensorData *>(&td1), nullptr);
  run_context.value_holder[9].Set(const_cast<GertTensorData *>(&td2), nullptr);
  run_context.value_holder[10].Set(const_cast<GertTensorData *>(&td3), nullptr);
  run_context.value_holder[11].Set(const_cast<GertTensorData *>(&td4), nullptr);

  ASSERT_NE(BuildOpTilingUnmanagedTensorData->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(TilingUT, test_build_unmanage_tensor_data_creator_success) {
  // tiling data
  auto run_context = BuildKernelRunContext(8, kOpTilingOutputSize + 1);
  ASSERT_EQ(BuildOpTilingUnmanagedTensorData->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  run_context.FreeAll();
}

TEST_F(TilingUT, test_build_tiling_output_shape_creator_success) {
  auto run_context = BuildKernelRunContext(6, 4);
  ASSERT_EQ(BuildOpTilingOutputShape->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  run_context.FreeAll();
}

TEST_F(TilingUT, test_build_outshape_data_success) {
  auto param = gert::TilingData::CreateCap(200);
  auto tiling_data = reinterpret_cast<TilingData *>(param.get());
  ASSERT_NE(tiling_data, nullptr);
  const uint32_t data_size = 10U;
  for (uint32_t i = 0; i < data_size; ++i) {
    tiling_data->Append(i);
  }
  auto run_context = BuildKernelRunContext(6, 4);
  run_context.value_holder[3].Set(param.get(), nullptr);

  StorageShape shape1({1, 2}, {1, 2});
  StorageShape shape2({1, 2}, {1, 2});
  StorageShape shape3({1, 2}, {1, 2});
  StorageShape shape4({1, 2}, {1, 2});
  run_context.value_holder[6].Set(const_cast<StorageShape *>(&shape1), nullptr);
  run_context.value_holder[7].Set(const_cast<StorageShape *>(&shape2), nullptr);
  run_context.value_holder[8].Set(const_cast<StorageShape *>(&shape3), nullptr);
  run_context.value_holder[9].Set(const_cast<StorageShape *>(&shape4), nullptr);
  ASSERT_EQ(BuildOpTilingOutputShape->run_func(run_context), ge::GRAPH_SUCCESS);
  run_context.FreeValue(6);
  run_context.FreeValue(7);
  run_context.FreeValue(8);
  run_context.FreeValue(9);
}
TEST_F(TilingUT, FallibleTiling_ReturnSuccess_WhenTilingFuncFailed) {
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(FailedTilingFunc),
                                    .launch_arg = fake_launch_arg};
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum))
          .Inputs({&fwk_data, nullptr, nullptr})
          .Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(kernel::FallibleTiling(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      *context->GetOutputPointer<uint32_t>(static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kTilingStatus)),
      1U);
}
TEST_F(TilingUT, FallibleTiling_ReturnSuccess_WhenTilingFuncSuccess) {
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(SuccessTilingFunc),
                                    .launch_arg = fake_launch_arg};
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum))
          .Inputs({&fwk_data, nullptr, nullptr})
          .Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(kernel::FallibleTiling(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      *context->GetOutputPointer<uint32_t>(static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kTilingStatus)),
      0U);
}
TEST_F(TilingUT, FallibleTiling_Failed_WhenInputNumWrong) {
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(SuccessTilingFunc),
                                    .launch_arg = fake_launch_arg};
  auto context_holder = KernelRunContextFaker().KernelIONum(1, TilingContext::kOutputNum).Inputs({&fwk_data}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(kernel::FallibleTiling(context), ge::GRAPH_SUCCESS);
}
TEST_F(TilingUT, Workspace_AlignTo512_Tiling) {
  auto workspace_holder = ContinuousVector::Create<size_t>(16);
  auto workspace = reinterpret_cast<TypedContinuousVector<size_t> *>(workspace_holder.get());
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(SuccessTilingFuncWithWorkspace),
                                    .launch_arg = fake_launch_arg};
  auto context_holder = TilingContextFaker()
                            .NodeIoNum(2, 1)
                            .TilingFwkData(&fwk_data)
                            .Workspace(reinterpret_cast<ContinuousVector *>(workspace))
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(kernel::Tiling(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(workspace->GetSize(), 3U);
  EXPECT_EQ(workspace->GetData()[0], 512U);
  EXPECT_EQ(workspace->GetData()[1], 1024U);
  EXPECT_EQ(workspace->GetData()[2], 512U);
}

TEST_F(TilingUT, TilingAppendWorkspace_succ) {
  auto tiling_ws = ContinuousVector::Create<int64_t>(2);
  auto tiling_ws_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(tiling_ws.get());
  tiling_ws_vc->SetSize(2);
  *tiling_ws_vc->MutableData() = 1;
  *(tiling_ws_vc->MutableData() + 1) = 2;
  auto known_ws = ContinuousVector::Create<int64_t>(3);
  auto known_ws_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(known_ws.get());
  known_ws_vc->SetSize(3);
  *known_ws_vc->MutableData() = 3;
  *(known_ws_vc->MutableData() + 1) = 4;
  *(known_ws_vc->MutableData() + 2) = 5;

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)tiling_ws.get(), nullptr);
  run_context.value_holder[1].Set((void *)known_ws.get(), nullptr);
  auto op_desc = std::make_shared<ge::OpDesc>("add", "Add");
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({1, 2, 3});
  ASSERT_EQ(
      KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendWorkspace")->outputs_creator(nullptr, run_context),
      ge::GRAPH_SUCCESS);
  auto context = run_context.GetContext<KernelContext>();
  EXPECT_EQ(kernel::TilingAppendWorkspace(context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(
      KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendWorkspace")->trace_printer(run_context).empty());
  auto res = run_context.value_holder[2].GetPointer<TypedContinuousVector<int64_t>>();
  EXPECT_EQ(res->GetSize(), 3);
  EXPECT_EQ(res->MutableData()[0], 1);
  EXPECT_EQ(res->MutableData()[1], 2);
  EXPECT_EQ(res->MutableData()[2], 5);
}

TEST_F(TilingUT, TilingMemCheck_succ) {
  ge::DumpStub::GetInstance().Clear();
  auto launch_arg = CreateLaunchArg(1, 1);
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  int64_t shape_sizes[2] = {8224, 4128};
  auto tiling_ws = ContinuousVector::Create<int64_t>(2);
  auto tiling_ws_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(tiling_ws.get());
  tiling_ws_vc->SetSize(2);
  *tiling_ws_vc->MutableData() = 1;
  *(tiling_ws_vc->MutableData() + 1) = 2;
  uint64_t ori_param_size = 20;
  bool is_memcheck_enable = true;
  bool is_args_exception_enable = true;

  auto args_sizes = ContinuousVector::Create<int64_t>(2);
  auto args_sizes_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(args_sizes.get());
  args_sizes_vc->SetSize(2);
  *args_sizes_vc->MutableData() = 0;
  *(args_sizes_vc->MutableData() + 1) = 0;

  auto args_idx_to_io_idx = ContinuousVector::Create<optiling::ArgsIndexToIoIndex>(2);
  auto args_idx_to_io_idx_vc =
      reinterpret_cast<TypedContinuousVector<optiling::ArgsIndexToIoIndex> *>(args_idx_to_io_idx.get());
  args_idx_to_io_idx_vc->SetSize(2);
  *args_idx_to_io_idx_vc->MutableData() = {optiling::ArgsRole::kInput, 0, 0};
  *(args_idx_to_io_idx_vc->MutableData() + 1) = {optiling::ArgsRole::kOutput, 1, 0};
  auto tmp_context_holder =
      KernelRunContextFaker()
          .NodeIoNum(1, 1)
          .IrInputNum(1)
          .Inputs({&in_shape, &out_shape, (void *)shape_sizes[0], launch_arg.get(), (void *)tiling_ws.get(),
                   (void *)ori_param_size, (void *)is_memcheck_enable, (void *)is_args_exception_enable,
                   (void *)args_sizes.get(), (void *)args_idx_to_io_idx.get()})
          .Outputs({&out_shape})
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .Build();

  auto context = tmp_context_holder.GetContext<KernelContext>();
  const KernelRegistry::KernelFuncs *tiling_append_data =
      KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendDfxInfo");
  ASSERT_EQ(tiling_append_data->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_append_data->run_func(context), ge::GRAPH_SUCCESS);
  auto &tiling_data = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg.get())->GetTilingData();

  // ori_param_size + input(1) + output(1) + workspace(2) + atomic(index)
  int64_t total_size = ori_param_size + sizeof(uint64_t) + sizeof(uint64_t) + 2 * sizeof(uint64_t) + sizeof(uint64_t);
  EXPECT_EQ(tiling_data.GetDataSize(), total_size);
  auto inner_data = (uint8_t *)tiling_data.GetData() + ori_param_size;
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data)), 8224);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + sizeof(int64_t)), 8224);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 2 * sizeof(int64_t)), 1);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 3 * sizeof(int64_t)), 2);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 4 * sizeof(int64_t)),
            1);  // atomic index
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0].size(), 9);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][0], 8224);  // input dim num
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][1], 8224);  // dim[0]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][2], 1);     // dim[1]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][3], 2);     // dim[2]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][4], 3);     // input dim num
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][5], 1);     // dim[0]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][6], 16);    // dim[1]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][7], 256);   // dim[2]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][8], 0);     // output
  ge::DumpStub::GetInstance().Clear();
}

TEST_F(TilingUT, TilingMemCheck2_succ) {
  ge::DumpStub::GetInstance().Clear();
  auto launch_arg = CreateLaunchArg(1, 1);
  ASSERT_NE(launch_arg, nullptr);
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  int64_t shape_sizes[2] = {8224, 4128};
  auto tiling_ws = ContinuousVector::Create<int64_t>(2);
  auto tiling_ws_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(tiling_ws.get());
  tiling_ws_vc->SetSize(2);
  *tiling_ws_vc->MutableData() = 1;
  *(tiling_ws_vc->MutableData() + 1) = 2;
  uint64_t ori_param_size = 20;
  bool is_memcheck_enable = true;
  bool is_args_exception_enable = true;

  auto args_sizes = ContinuousVector::Create<int64_t>(4);
  auto args_sizes_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(args_sizes.get());
  args_sizes_vc->SetSize(4);
  *args_sizes_vc->MutableData() = 0;
  *(args_sizes_vc->MutableData() + 1) = 4;
  *(args_sizes_vc->MutableData() + 2) = 0;
  *(args_sizes_vc->MutableData() + 3) = 8;

  auto args_idx_to_io_idx = ContinuousVector::Create<optiling::ArgsIndexToIoIndex>(2);
  auto args_idx_to_io_idx_vc =
      reinterpret_cast<TypedContinuousVector<optiling::ArgsIndexToIoIndex> *>(args_idx_to_io_idx.get());
  args_idx_to_io_idx_vc->SetSize(2);
  *args_idx_to_io_idx_vc->MutableData() = {optiling::ArgsRole::kInput, 0, 0};
  *(args_idx_to_io_idx_vc->MutableData() + 1) = {optiling::ArgsRole::kOutput, 2, 0};

  auto tmp_context_holder =
      KernelRunContextFaker()
          .NodeIoNum(2, 2)
          .IrInputNum(2)
          .Inputs({&in_shape, &in_shape, &out_shape, &out_shape, (void *)shape_sizes[0], (void *)shape_sizes[1],
                   launch_arg.get(), (void *)tiling_ws.get(), (void *)ori_param_size, (void *)is_memcheck_enable,
                   (void *)is_args_exception_enable, (void *)args_sizes.get(), (void *)args_idx_to_io_idx.get()})
          .Outputs({&out_shape, &out_shape})
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .Build();

  auto context = tmp_context_holder.GetContext<KernelContext>();
  const KernelRegistry::KernelFuncs *tiling_append_data =
      KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendDfxInfo");
  ASSERT_EQ(tiling_append_data->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_append_data->run_func(context), ge::GRAPH_SUCCESS);
  auto &tiling_data = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg.get())->GetTilingData();
  // ori_param_size + input(1) + inputdesc(1) + output(1) + outputdesc(1) + workspace(2) + atomic(index)
  uint64_t total_size =
      ori_param_size + 2 * sizeof(uint64_t) + 2 * sizeof(uint64_t) + 2 * sizeof(uint64_t) + sizeof(uint64_t);
  EXPECT_EQ(tiling_data.GetDataSize(), total_size);
  auto inner_data = (uint8_t *)tiling_data.GetData() + ori_param_size;
  // 1*16*256*2(DT_FLOAT16) + 32
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data)), 8224);
  // args_sizes_vc[1] = 4
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + sizeof(int64_t)), 4);
  // 1*16*256*2(DT_FLOAT16) + 32
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 2 * sizeof(int64_t)), 8224);
  // args_sizes_vc[3] = 8
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 3 * sizeof(int64_t)), 8);
  // tiling_ws_vc[0] = 1  tiling_ws_vc[1] = 2
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 4 * sizeof(int64_t)), 1);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 5 * sizeof(int64_t)), 2);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 6 * sizeof(int64_t)),
            1);  // atomic index
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0].size(), 11);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][0], 8224);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][1], 4);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][2], 8224);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][3], 8);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][4], 1);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][5], 2);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][6], 3);    // input1 dim num
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][7], 1);    // dim[0]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][8], 16);   // dim[1]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][9], 256);  // dim[2]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][10], 0);   // output2
  ge::DumpStub::GetInstance().Clear();
}

TEST_F(TilingUT, TilingMemCheck3_succ) {
  ge::DumpStub::GetInstance().Clear();
  auto launch_arg = CreateLaunchArg(1, 1);
  ASSERT_NE(launch_arg, nullptr);
  gert::StorageShape in_shape = {{}, {}};
  gert::StorageShape out_shape = {{}, {}};
  int64_t shape_sizes[2] = {32, 32};
  auto tiling_ws = ContinuousVector::Create<int64_t>(2);
  auto tiling_ws_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(tiling_ws.get());
  tiling_ws_vc->SetSize(2);
  *tiling_ws_vc->MutableData() = 1;
  *(tiling_ws_vc->MutableData() + 1) = 2;
  uint64_t ori_param_size = 20;
  bool is_memcheck_enable = true;
  bool is_args_exception_enable = true;

  auto args_sizes = ContinuousVector::Create<int64_t>(4);
  auto args_sizes_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(args_sizes.get());
  args_sizes_vc->SetSize(4);
  *args_sizes_vc->MutableData() = 0;
  *(args_sizes_vc->MutableData() + 1) = 0;
  *(args_sizes_vc->MutableData() + 2) = 0;
  *(args_sizes_vc->MutableData() + 3) = 0;

  auto args_idx_to_io_idx = ContinuousVector::Create<optiling::ArgsIndexToIoIndex>(3);
  auto args_idx_to_io_idx_vc =
      reinterpret_cast<TypedContinuousVector<optiling::ArgsIndexToIoIndex> *>(args_idx_to_io_idx.get());
  args_idx_to_io_idx_vc->SetSize(2);
  *args_idx_to_io_idx_vc->MutableData() = {optiling::ArgsRole::kInput, 1, 0};
  *(args_idx_to_io_idx_vc->MutableData() + 1) = {optiling::ArgsRole::kOutput, 2, 0};

  auto tmp_context_holder =
      KernelRunContextFaker()
          .NodeIoNum(2, 2)
          .IrInputNum(2)
          .Inputs({&in_shape, &in_shape, &out_shape, &out_shape, (void *)shape_sizes[0], (void *)shape_sizes[1],
                   launch_arg.get(), (void *)tiling_ws.get(), (void *)ori_param_size, (void *)is_memcheck_enable,
                   (void *)is_args_exception_enable, (void *)args_sizes.get(), (void *)args_idx_to_io_idx.get()})
          .Outputs({&out_shape, &out_shape})
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .Build();

  auto context = tmp_context_holder.GetContext<KernelContext>();
  const KernelRegistry::KernelFuncs *tiling_append_data =
      KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendDfxInfo");
  ASSERT_EQ(tiling_append_data->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_append_data->run_func(context), ge::GRAPH_SUCCESS);
  auto &tiling_data = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg.get())->GetTilingData();
  // ori_param_size + input(1) + input(1) + output(1) + output(2) + workspace(2) + atomic(index)
  uint64_t total_size =
      ori_param_size + 2 * sizeof(uint64_t) + 2 * sizeof(uint64_t) + 2 * sizeof(uint64_t) + sizeof(uint64_t);
  EXPECT_EQ(tiling_data.GetDataSize(), total_size);
  auto inner_data = (uint8_t *)tiling_data.GetData() + ori_param_size;
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data)), 0);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + sizeof(int64_t)), 64);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 2 * sizeof(int64_t)), 32);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 3 * sizeof(int64_t)), 0);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 4 * sizeof(int64_t)), 1);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 5 * sizeof(int64_t)), 2);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 6 * sizeof(int64_t)),
            1);  // atomic index
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0].size(), 8);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][0], 0);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][1], 64);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][2], 32);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][3], 0);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][4], 1);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][5], 2);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][6], 0);  // iutput0 dim num
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][7], 0);  // output0 dim num
  ge::DumpStub::GetInstance().Clear();
}

TEST_F(TilingUT, TilingMemCheck_succ_aligned) {
  ge::DumpStub::GetInstance().Clear();
  auto launch_arg = CreateLaunchArg(1, 1);
  ASSERT_NE(launch_arg, nullptr);
  auto &temp_tiling_data = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg.get())->GetTilingData();
  int32_t tiling_data_num = 3;
  temp_tiling_data.Append(tiling_data_num);

  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  int64_t shape_sizes[2] = {8224, 4128};
  auto tiling_ws = ContinuousVector::Create<int64_t>(2);
  auto tiling_ws_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(tiling_ws.get());
  tiling_ws_vc->SetSize(2);
  *tiling_ws_vc->MutableData() = 1;
  *(tiling_ws_vc->MutableData() + 1) = 2;
  int64_t ori_param_size = 0;
  bool is_memcheck_enable = true;
  bool is_args_exception_enable = true;

  auto args_sizes = ContinuousVector::Create<int64_t>(2);
  auto args_sizes_vc = reinterpret_cast<TypedContinuousVector<int64_t> *>(args_sizes.get());
  args_sizes_vc->SetSize(2);
  *args_sizes_vc->MutableData() = 0;
  *(args_sizes_vc->MutableData() + 1) = 0;

  auto args_idx_to_io_idx = ContinuousVector::Create<optiling::ArgsIndexToIoIndex>(2);
  auto args_idx_to_io_idx_vc =
      reinterpret_cast<TypedContinuousVector<optiling::ArgsIndexToIoIndex> *>(args_idx_to_io_idx.get());
  args_idx_to_io_idx_vc->SetSize(2);
  *args_idx_to_io_idx_vc->MutableData() = {optiling::ArgsRole::kInput, 0, 0};
  *(args_idx_to_io_idx_vc->MutableData() + 1) = {optiling::ArgsRole::kOutput, 1, 0};

  auto tmp_context_holder =
      KernelRunContextFaker()
          .NodeIoNum(1, 1)
          .IrInputNum(1)
          .Inputs({&in_shape, &out_shape, (void *)shape_sizes[0], launch_arg.get(), (void *)tiling_ws.get(),
                   (void *)ori_param_size, (void *)is_memcheck_enable, (void *)is_args_exception_enable,
                   (void *)args_sizes.get(), (void *)args_idx_to_io_idx.get()})
          .Outputs({&out_shape})
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .Build();

  auto context = tmp_context_holder.GetContext<KernelContext>();
  const KernelRegistry::KernelFuncs *tiling_append_data =
      KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendDfxInfo");
  ASSERT_EQ(tiling_append_data->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_append_data->run_func(context), ge::GRAPH_SUCCESS);
  auto &tiling_data = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg.get())->GetTilingData();

  // tiling data align + input(1) + output(1) + workspace(2) + atomic(index)
  int64_t total_size = 8 + sizeof(uint64_t) + sizeof(uint64_t) + 2 * sizeof(uint64_t) + sizeof(uint64_t);

  EXPECT_EQ(tiling_data.GetDataSize(), total_size);
  auto inner_data = reinterpret_cast<int8_t *>(tiling_data.GetData()) + 8;
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data)), 8224);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + sizeof(int64_t)), 8224);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 2 * sizeof(int64_t)), 1);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 3 * sizeof(int64_t)), 2);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(reinterpret_cast<int8_t *>(inner_data) + 4 * sizeof(int64_t)),
            1);  // atomic index
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0].size(), 9);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][0], 8224);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][1], 8224);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][2], 1);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][3], 2);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][4], 3);    // input dim num
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][5], 1);    // dim[0]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][6], 16);   // dim[1]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][7], 256);  // dim[2]
  EXPECT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits()[0][8], 0);    // output
  ge::DumpStub::GetInstance().Clear();
}

}  // namespace gert