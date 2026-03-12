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

#include "kernel/common_kernel_impl/tiling.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "kernel/tiling_cache.h"
#include "graph/utils/math_util.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "common/tiling_fwk_data_helper.h"
#include "common/op_tiling/tiling_dfx.h"
#include "depends/profiler/src/dump_stub.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kCacheSizeUt = 24UL;
constexpr size_t kEvictNumUt = 8UL;

std::unique_ptr<uint8_t[]> CreateWorkspaceSizesWithDummyData(const size_t capacity) {
  auto ws_holder = ContinuousVector::Create<size_t>(capacity);
  if (ws_holder == nullptr) {
    return nullptr;
  }
  const auto ws = reinterpret_cast<TypedContinuousVector<size_t> *>(ws_holder.get());
  for (int i = 0; i < 4; i++) {
    ws->MutableData()[i] = i;
    ws->SetSize(i + 1UL);
  }
  return ws_holder;
}

std::unique_ptr<uint8_t[]> CreateTilingDataWithDummyData(const size_t capacity) {
  auto td_holder = TilingData::CreateCap(capacity);
  if (td_holder == nullptr) {
    return nullptr;
  }
  const auto td = reinterpret_cast<TilingData *>(td_holder.get());
  for (size_t i = 0; i < capacity; i++) {
    td->Append<uint8_t>(static_cast<uint8_t>(i % 10));
  }
  return td_holder;
}

const TilingCacheValue &GetExpectTilingValue() {
  // 准备Tiling的输出, 此处借用TilingCacheValue数据结构
  static const TilingCacheValue expect_value = {.atomic_clean_flag = false,
                                                .tiling_cond = 1,
                                                .local_mem_size = 1024UL,
                                                .block_dim = 16UL,
                                                .tiling_key = 1234UL,
                                                .ori_tiling_data_size = 256UL,
                                                .dfx_dump_data_num = 0,
                                                .workspace_sizes_holder = CreateWorkspaceSizesWithDummyData(4UL),
                                                .launch_arg_holder = CreateLaunchArg(),
                                                .dfx_dump_data_holder = nullptr};
  const auto dummy_td_holder = CreateTilingDataWithDummyData(256UL);
  const auto dummy_td = reinterpret_cast<TilingData *>(dummy_td_holder.get());
  auto &expect_td = reinterpret_cast<RtKernelLaunchArgsEx *>(expect_value.launch_arg_holder.get())->GetTilingData();
  std::memcpy(expect_td.GetData(), dummy_td->GetData(), dummy_td->GetDataSize());
  expect_td.SetDataSize(dummy_td->GetDataSize());
  return expect_value;
}

void CheckTilingValue(const TilingCacheValue &tiling_value) {
  const auto &expect_value = GetExpectTilingValue();
  EXPECT_EQ(tiling_value.atomic_clean_flag, expect_value.atomic_clean_flag);
  EXPECT_EQ(tiling_value.tiling_cond, expect_value.tiling_cond);
  EXPECT_EQ(tiling_value.local_mem_size, expect_value.local_mem_size);
  EXPECT_EQ(tiling_value.block_dim, expect_value.block_dim);
  EXPECT_EQ(tiling_value.tiling_key, expect_value.tiling_key);
  // 校验点1: 缓存中的workspace_sizes是按照kAiCoreWorkspaceAlignment对齐的
  const auto ws = reinterpret_cast<TypedContinuousVector<size_t> *>(tiling_value.workspace_sizes_holder.get());
  const auto expect_ws = reinterpret_cast<TypedContinuousVector<size_t> *>(expect_value.workspace_sizes_holder.get());
  ASSERT_EQ(ws->GetSize(), expect_ws->GetSize());
  for (size_t i = 0U; i < expect_ws->GetSize(); ++i) {
    size_t aligned_workspace_size = 0;
    ge::RoundUpOverflow(expect_ws->MutableData()[i], kAiCoreWorkspaceAlignment, aligned_workspace_size);
    EXPECT_EQ(ws->GetData()[i], aligned_workspace_size);
  }
  // 校验点2: 缓存中的tilingData必须大小和内容一致
  auto arg = reinterpret_cast<RtKernelLaunchArgsEx *>(expect_value.launch_arg_holder.get());
  ASSERT_NE(arg, nullptr);
  const auto &td = arg->GetTilingData();
  const auto &expect_td = reinterpret_cast<RtKernelLaunchArgsEx *>(expect_value.launch_arg_holder.get())->GetTilingData();
  ASSERT_EQ(td.GetDataSize(), expect_td.GetDataSize());
  ASSERT_EQ(td.GetCapacity(), expect_td.GetCapacity());
  EXPECT_EQ(std::memcmp(td.GetData(), expect_td.GetData(), expect_td.GetDataSize()), 0);
}

UINT32 StubTilingFuncFail(KernelContext *context) {
  (void)context;
  return ge::GRAPH_FAILED;
}

UINT32 StubTilingFuncSuccEmpty(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingFuncSucc(KernelContext *context) {
  const auto &expect_tiling_value = GetExpectTilingValue();
  auto tiling_context = reinterpret_cast<TilingContext *>(context);
  tiling_context->SetNeedAtomic(expect_tiling_value.atomic_clean_flag);
  tiling_context->SetBlockDim(expect_tiling_value.block_dim);
  tiling_context->SetTilingCond(expect_tiling_value.tiling_cond);
  tiling_context->SetLocalMemorySize(expect_tiling_value.local_mem_size);
  tiling_context->SetTilingKey(expect_tiling_value.tiling_key);
  GE_ASSERT_NOTNULL(expect_tiling_value.workspace_sizes_holder);
  GE_ASSERT_NOTNULL(expect_tiling_value.launch_arg_holder);
  // 拷贝workspace sizes
  const auto expect_ws =
      reinterpret_cast<TypedContinuousVector<size_t> *>(expect_tiling_value.workspace_sizes_holder.get());
  auto ws_data = tiling_context->GetWorkspaceSizes(expect_ws->GetSize());
  GE_ASSERT_NOTNULL(ws_data);
  std::memcpy(ws_data, expect_ws->GetData(), sizeof(size_t) * 4);
  // 拷贝tilingData
  auto td_data = tiling_context->GetRawTilingData();
  GE_ASSERT_NOTNULL(td_data);
  const auto &expect_td = reinterpret_cast<RtKernelLaunchArgsEx *>(expect_tiling_value.launch_arg_holder.get())->GetTilingData();
  std::memcpy(td_data->GetData(), expect_td.GetData(), expect_td.GetDataSize());
  td_data->SetDataSize(expect_td.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

class CacheableTilingUt : public testing::Test {
 public:
  void SetUp() override {
    fake_launch_arg_holder = CreateLaunchArg();
    fake_launch_arg = reinterpret_cast<RtKernelLaunchArgsEx *>(fake_launch_arg_holder.get());
  }
  void TearDown() override {
    fake_launch_arg_holder.reset();
    fake_launch_arg = nullptr;
  }

 public:
  static const KernelRegistry::KernelFuncs *kf_cacheable_tiling;
  static const KernelRegistry::KernelFuncs *kf_cacheable_fallible_tiling;
  static const KernelRegistry::KernelFuncs *kf_prepare_fwk_data;
  static const KernelRegistry::KernelFuncs *kf_prepare_cacheable_fwk_data;
  std::unique_ptr<uint8_t[]> fake_launch_arg_holder;
  RtKernelLaunchArgsEx *fake_launch_arg{nullptr};
};
const KernelRegistry::KernelFuncs *CacheableTilingUt::kf_cacheable_tiling =
    KernelRegistry::GetInstance().FindKernelFuncs("CacheableTiling");
const KernelRegistry::KernelFuncs *CacheableTilingUt::kf_cacheable_fallible_tiling =
    KernelRegistry::GetInstance().FindKernelFuncs("CacheableFallibleTiling");
const KernelRegistry::KernelFuncs *CacheableTilingUt::kf_prepare_fwk_data =
    KernelRegistry::GetInstance().FindKernelFuncs("PrepareTilingFwkData");
const KernelRegistry::KernelFuncs *CacheableTilingUt::kf_prepare_cacheable_fwk_data =
    KernelRegistry::GetInstance().FindKernelFuncs("PrepareCacheableTilingFwkData");

TEST_F(CacheableTilingUt, CacheableTiling_Fail_FwkDataIsNull) {
  auto run_context = BuildKernelRunContext(2UL, 6UL);
  ASSERT_EQ(kf_cacheable_tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(kf_cacheable_tiling->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CacheableTilingUt, CacheableTiling_Fail_InvalidKernelInputs) {
  auto run_context = BuildKernelRunContext(0UL, 6UL);
  ASSERT_EQ(kf_cacheable_tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(kf_cacheable_tiling->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CacheableTilingUt, CacheableTiling_Fail_CallTilingFuncFailed) {
  // 准备输入输出Shape
  gert::StorageShape in_shape1({2, 3}, {2, 3});
  gert::StorageShape in_shape2({3, 4}, {3, 4});
  gert::StorageShape out_shape({2, 4}, {2, 4});
  // 准备CacheableFwkData
  std::unique_ptr<TilingCacheStrategy> cache_strategy(new (std::nothrow)
                                                          TilingCacheLruStrategy(kCacheSizeUt, kEvictNumUt));
  ASSERT_NE(cache_strategy, nullptr);
  TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(StubTilingFuncFail), .launch_arg = fake_launch_arg};
  std::string func_name = "BuildGeneralTilingCacheKey";
  CacheableTilingFwkData cacheable_fwk_data = {.fwk_data = fwk_data,
                                               .tiling_cache_mgr = TilingCacheManager(std::move(cache_strategy)),
                                               .build_tiling_cache_key_func_name = func_name.data()};
  // 准备Tiling的输出
  auto workspace_sizes_holder = CreateWorkspaceSizesWithDummyData(16UL);
  ASSERT_NE(workspace_sizes_holder, nullptr);
  auto run_context = KernelRunContextFaker()
                         .NodeIoNum(2UL, 1UL)
                         .IrInputNum(2UL)
                         .KernelIONum(6UL, static_cast<size_t>(TilingExOutputIndex::kNum))
                         .Inputs({&in_shape1, &in_shape2, &out_shape, nullptr, nullptr, &cacheable_fwk_data, nullptr, nullptr})
                         .Outputs({nullptr, nullptr, nullptr, nullptr, workspace_sizes_holder.get(),
                                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
                         .Build();
  ASSERT_EQ(kf_cacheable_tiling->run_func(run_context), ge::GRAPH_FAILED);
  // tiling_func调用失败不做缓存
  HashBuffer hash_buf;
  hash_buf.AddParamToBuf(in_shape1.GetStorageShape());
  hash_buf.AddParamToBuf(in_shape2.GetStorageShape());
  EXPECT_EQ(cacheable_fwk_data.tiling_cache_mgr.Exist(hash_buf.GetTilingCacheKey()), false);
}

TEST_F(CacheableTilingUt, PrintTilingData_Ok) {
  auto run_context = BuildKernelRunContext(2UL, 5UL);
  auto tiling_data = CreateTilingDataWithDummyData(8UL);
  run_context.value_holder[5].Set(tiling_data.get(), nullptr);
  auto msgs = kf_cacheable_tiling->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1U);
  ASSERT_TRUE(msgs[0].find("TilingData: ") != string::npos);
}

TEST_F(CacheableTilingUt, PrintTilingData_Ok_TilingDataIsNull) {
  auto run_context = BuildKernelRunContext(2UL, 5UL);
  run_context.value_holder[5].Set(nullptr, nullptr);
  auto msgs = kf_cacheable_tiling->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 0U);
}

TEST_F(CacheableTilingUt, CacheableTiling_Ok_TilingResultAddedAndFetched) {
  // 准备输入输出Shape
  gert::StorageShape in_shape1({2, 3}, {2, 3});
  gert::StorageShape in_shape2({3, 4}, {3, 4});
  gert::StorageShape out_shape({2, 4}, {2, 4});
  // 准备TilingCacheManager, 两次执行过程共享manager
  std::unique_ptr<TilingCacheStrategy> cache_strategy(new (std::nothrow)
                                                          TilingCacheLruStrategy(kCacheSizeUt, kEvictNumUt));
  ASSERT_NE(cache_strategy, nullptr);
  TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(StubTilingFuncFail), .launch_arg = fake_launch_arg};
  std::string func_name = "BuildGeneralTilingCacheKey";
  CacheableTilingFwkData cacheable_fwk_data = {.fwk_data = fwk_data,
      .tiling_cache_mgr = TilingCacheManager(std::move(cache_strategy)),
      .build_tiling_cache_key_func_name = func_name.data()};
  const auto execute_kernel_and_check = [&](KernelRegistry::KernelFunc stub_func) {
    const TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(stub_func), .launch_arg = fake_launch_arg};
    cacheable_fwk_data.fwk_data = fwk_data;
    auto run_context = KernelRunContextFaker()
                           .NodeIoNum(2UL, 1UL)
                           .IrInputNum(2UL)
                           .KernelIONum(6UL, static_cast<size_t>(TilingExOutputIndex::kNum))
                           .Inputs({&in_shape1, &in_shape2, &out_shape, nullptr, nullptr, &cacheable_fwk_data, nullptr,
                                    nullptr})
                           .Build();
    ASSERT_EQ(kf_cacheable_tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(kf_cacheable_tiling->run_func(run_context), ge::GRAPH_SUCCESS);
    // tiling_func调用成功,缓存Tiling结果,其中workspace中缓存的是调用AlignWorkspaceSizes对齐后的结果
    HashBuffer hash_buf;
    hash_buf.AddParamToBuf(in_shape1.GetStorageShape());
    hash_buf.AddParamToBuf(in_shape2.GetStorageShape());
    const auto tiling_cache = cacheable_fwk_data.tiling_cache_mgr.TryFetchCache(hash_buf.GetTilingCacheKey());
    ASSERT_NE(tiling_cache, nullptr);
    CheckTilingValue(tiling_cache->GetTilingCacheValue());
  };

  // 第一次执行, 没有缓存, 调用TilingFunc
  execute_kernel_and_check(StubTilingFuncSucc);
  // 重置context中的TilingData, tiling_func无动作, 第二次执行命中缓存
  execute_kernel_and_check(StubTilingFuncSuccEmpty);
}

TEST_F(CacheableTilingUt, PrepareTilingFwkData_Ok) {
  auto run_context = KernelRunContextFaker()
                         .KernelIONum(1UL, 1UL)
                         .Inputs({reinterpret_cast<void *>(StubTilingFuncSuccEmpty), fake_launch_arg})
                         .Build();
  EXPECT_EQ(kf_prepare_fwk_data->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(kf_prepare_fwk_data->run_func(run_context), ge::GRAPH_SUCCESS);
  auto context = run_context.GetContext<KernelContext>();
  const auto fwk_data = context->GetOutputPointer<TilingFwkData>(0UL);
  ASSERT_NE(fwk_data, nullptr);
  EXPECT_EQ(fwk_data->tiling_func, reinterpret_cast<void *>(StubTilingFuncSuccEmpty));
}

TEST_F(CacheableTilingUt, PrepareCacheableTilingFwkData_Ok) {
  auto run_context = KernelRunContextFaker()
                         .KernelIONum(2UL, 1UL)
                         .Inputs({reinterpret_cast<void *>(StubTilingFuncSuccEmpty), fake_launch_arg})
                         .Build();
  EXPECT_EQ(kf_prepare_cacheable_fwk_data->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(kf_prepare_cacheable_fwk_data->run_func(run_context), ge::GRAPH_SUCCESS);
  auto context = run_context.GetContext<KernelContext>();
  const auto cacheable_fwk_data = context->GetOutputPointer<CacheableTilingFwkData>(0UL);
  ASSERT_NE(cacheable_fwk_data, nullptr);
  EXPECT_EQ(cacheable_fwk_data->fwk_data.tiling_func, reinterpret_cast<void *>(StubTilingFuncSuccEmpty));
}

TEST_F(CacheableTilingUt, PrepareCacheableTilingFwkDataForDataDependency_Ok) {
  uint64_t data_dependency = 10UL;
  std::string func_name = "BuildGeneralTilingCacheKey";
  auto run_context = KernelRunContextFaker()
      .KernelIONum(2UL, 1UL)
      .Inputs({reinterpret_cast<void *>(StubTilingFuncSuccEmpty),
               reinterpret_cast<void *>(fake_launch_arg), reinterpret_cast<void *>(data_dependency),
               reinterpret_cast<void *>(func_name.data())})
      .Build();
  EXPECT_EQ(kf_prepare_cacheable_fwk_data->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(kf_prepare_cacheable_fwk_data->run_func(run_context), ge::GRAPH_SUCCESS);
  auto context = run_context.GetContext<KernelContext>();
  const auto cacheable_fwk_data = context->GetOutputPointer<CacheableTilingFwkData>(0UL);
  ASSERT_NE(cacheable_fwk_data, nullptr);
  EXPECT_EQ(cacheable_fwk_data->data_dependency, data_dependency);
  EXPECT_EQ(cacheable_fwk_data->build_tiling_cache_key_func_name, func_name.data());
}

TEST_F(CacheableTilingUt, CacheableTiling_Ok_DataDependentTilingResultAddedAndFetched) {
  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block1 = allocator->Malloc(2 * 3 * 2);
  auto mem_block2 = allocator->Malloc(3 * 4 * 2);
  auto mem_block3 = allocator->Malloc(2 * 4 * 2);
  // 准备输入输出Shape
  gert::Tensor in_tensor1 = {{{2, 3}, {2, 3}}, {}, kOnHost, {}, const_cast<void *>(mem_block1->GetAddr())};
  gert::Tensor in_tensor2 = {{{3, 4}, {3, 4}}, {}, kOnHost, {}, const_cast<void *>(mem_block2->GetAddr())};
  gert::Tensor out_tensor = {{{2, 4}, {2, 4}}, {}, kOnHost, {}, const_cast<void *>(mem_block3->GetAddr())};
  // 准备TilingCacheManager, 两次执行过程共享manager
  std::unique_ptr<TilingCacheStrategy> cache_strategy(new (std::nothrow)
                                                          TilingCacheLruStrategy(kCacheSizeUt, kEvictNumUt));
  ASSERT_NE(cache_strategy, nullptr);
  TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(StubTilingFuncFail), .launch_arg = fake_launch_arg};
  std::string func_name = "BuildGeneralTilingCacheKey";
  CacheableTilingFwkData cacheable_fwk_data = {.fwk_data = fwk_data,
                                               .tiling_cache_mgr = TilingCacheManager(std::move(cache_strategy)),
                                               .data_dependency = 3UL,
                                               .build_tiling_cache_key_func_name = func_name.data()};
  const auto execute_kernel_and_check = [&](KernelRegistry::KernelFunc stub_func) {
    cacheable_fwk_data.fwk_data.tiling_func = reinterpret_cast<void *>(stub_func);
    auto td_output_holder = TilingData::CreateCap(1024UL);
    auto run_context =
        KernelRunContextFaker()
            .NodeIoNum(2UL, 1UL)
            .IrInputNum(2UL)
            .KernelIONum(7UL, static_cast<size_t>(TilingExOutputIndex::kNum))
            .Inputs({&in_tensor1, &in_tensor2, &out_tensor, nullptr, nullptr, &cacheable_fwk_data, nullptr, nullptr})
            .Build();

    ASSERT_EQ(kf_cacheable_tiling->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(kf_cacheable_tiling->run_func(run_context), ge::GRAPH_SUCCESS);
    // tiling_func调用成功,缓存Tiling结果,其中workspace中缓存的是调用AlignWorkspaceSizes对齐后的结果
    HashBuffer hash_buf;
    hash_buf.AddParamToBuf(in_tensor1);
    hash_buf.AddParamToBuf(in_tensor2);
    const auto tiling_cache = cacheable_fwk_data.tiling_cache_mgr.TryFetchCache(hash_buf.GetTilingCacheKey());
    ASSERT_NE(tiling_cache, nullptr);
    CheckTilingValue(tiling_cache->GetTilingCacheValue());
  };

  // 第一次执行, 没有缓存, 调用TilingFunc
  execute_kernel_and_check(StubTilingFuncSucc);
  // 重置context中的TilingData, tiling_func无动作, 第二次执行命中缓存
  execute_kernel_and_check(StubTilingFuncSuccEmpty);

  mem_block1->Free();
  mem_block2->Free();
  mem_block3->Free();
}

TEST_F(CacheableTilingUt, CacheableFallibleTiling_ok_TilingFuncSucc) {
  std::unique_ptr<TilingCacheStrategy> cache_strategy(
      new (std::nothrow)TilingCacheLruStrategy(kCacheSizeUt, kEvictNumUt));
  ASSERT_NE(cache_strategy, nullptr);
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(StubTilingFuncSuccEmpty),
                                    .launch_arg = fake_launch_arg};
  std::string func_name = "BuildGeneralTilingCacheKey";
  CacheableTilingFwkData cacheable_fwk_data = {.fwk_data = fwk_data,
      .tiling_cache_mgr = TilingCacheManager(std::move(cache_strategy)),
      .build_tiling_cache_key_func_name = func_name.data()};
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum))
          .Inputs({&cacheable_fwk_data, nullptr, nullptr})
          .Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(kf_cacheable_fallible_tiling->run_func(context), ge::GRAPH_SUCCESS);
  const auto status = context->GetOutputPointer<uint32_t>(static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kTilingStatus));
  ASSERT_EQ(*status, 0U);
}

TEST_F(CacheableTilingUt, TilingAppendDfxInfo_Ok_ArgsExceptionWithTilingCacheEnabled) {
  ge::DumpStub::GetInstance().Clear();
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  int64_t shape_sizes[1] = {8224};
  // 首次执行, args标记为missed
  auto launch_arg_holder = CreateLaunchArg(1, 1);
  auto launch_arg = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg_holder.get());
  TilingCacheKey key = {nullptr, 0};
  TilingCache tiling_cache(key, {});
  launch_arg->SetTilingCache(&tiling_cache);
  launch_arg->SetTilingCacheStatus(RtKernelLaunchArgsEx::TilingCacheStatus::kMissed);
  auto tiling_ws = ContinuousVector::Create<int64_t>(2);
  auto tiling_ws_vec = reinterpret_cast<TypedContinuousVector<int64_t> *>(tiling_ws.get());
  tiling_ws_vec->SetSize(2);
  *tiling_ws_vec->MutableData() = 1;
  *(tiling_ws_vec->MutableData() + 1) = 2;
  // 看护tik场景, memcheck从ori_param_size偏移处开始
  uint64_t ori_param_size = 20;
  // 使能memcheck和args exception
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

  auto dfx_info_kernel_run = [&]() {
    auto tmp_context_holder =
        KernelRunContextFaker()
            .NodeIoNum(1, 1)
            .IrInputNum(1)
            .Inputs({&in_shape, &out_shape, reinterpret_cast<void *>(shape_sizes[0]), launch_arg_holder.get(),
                     tiling_ws.get(), reinterpret_cast<void *>(ori_param_size),
                     reinterpret_cast<void *>(is_memcheck_enable), reinterpret_cast<void *>(is_args_exception_enable),
                     args_sizes.get(), args_idx_to_io_idx.get()})
            .Outputs({&out_shape})
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
            .Build();

    auto context = tmp_context_holder.GetContext<KernelContext>();
    const KernelRegistry::KernelFuncs *tiling_append_data =
        KernelRegistry::GetInstance().FindKernelFuncs("TilingAppendDfxInfo");
    ASSERT_EQ(tiling_append_data->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_append_data->run_func(context), ge::GRAPH_SUCCESS);
    auto &tiling_data = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_arg_holder.get())->GetTilingData();

    // ori_param_size + input(1) + output(1) + workspace(2) + atomic(index)
    const uint64_t expected_tiling_data_size =
        ori_param_size + sizeof(uint64_t) + sizeof(uint64_t) + 2 * sizeof(uint64_t) + sizeof(uint64_t);
    ASSERT_EQ(tiling_data.GetDataSize(), expected_tiling_data_size);
    const auto dfx_data_ptr = static_cast<int8_t *>(tiling_data.GetData()) + ori_param_size;
    // 校验memcheck内存
    EXPECT_EQ(*reinterpret_cast<int64_t *>(dfx_data_ptr), 8224);
    EXPECT_EQ(*reinterpret_cast<int64_t *>(dfx_data_ptr + sizeof(int64_t)), 8224);
    EXPECT_EQ(*reinterpret_cast<int64_t *>(dfx_data_ptr + 2 * sizeof(int64_t)), 1);
    EXPECT_EQ(*reinterpret_cast<int64_t *>(dfx_data_ptr + 3 * sizeof(int64_t)), 2);
    EXPECT_EQ(*reinterpret_cast<int64_t *>(dfx_data_ptr + 4 * sizeof(int64_t)), 1);  // atomic index
  };

  dfx_info_kernel_run();
  ASSERT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits().size(), 1);
  auto dump_data = ge::DumpStub::GetInstance().GetDynamicUnits()[0];
  ASSERT_EQ(dump_data.size(), 9);  // 第一次无缓存, 包含size和shape信息
  EXPECT_EQ(dump_data[0], 8224);   // input_1 size
  EXPECT_EQ(dump_data[1], 8224);   // output_2 size
  EXPECT_EQ(dump_data[2], 1);      // workspace_size_1
  EXPECT_EQ(dump_data[3], 2);      // workspace_size_2
  EXPECT_EQ(dump_data[4], 3);      // input_1 dim_num
  EXPECT_EQ(dump_data[5], 1);      // input_1 dim[0]
  EXPECT_EQ(dump_data[6], 16);     // input_1 dim[1]
  EXPECT_EQ(dump_data[7], 256);    // input_1 dim[2]
  EXPECT_EQ(dump_data[8], 0);      // output_1 shape, use 0 as default placeholder
  // 校验DfxInfo的缓存
  ASSERT_EQ(tiling_cache.GetTilingCacheValue().dfx_dump_data_num, 4);
  auto cached_dump_data = tiling_cache.GetTilingCacheValue().dfx_dump_data_holder.get();
  std::vector<uint64_t> dump_data_vec(cached_dump_data,
                                      cached_dump_data + tiling_cache.GetTilingCacheValue().dfx_dump_data_num);
  std::vector<uint64_t> expected_dump_data_vec{8224, 8224, 1, 2};
  ASSERT_NE(cached_dump_data, nullptr);
  EXPECT_EQ(dump_data_vec, expected_dump_data_vec);
  ge::DumpStub::GetInstance().Clear();

  // 第二次执行, 命中缓存, 只有size
  launch_arg->GetTilingData().SetDataSize(0);
  launch_arg->SetTilingCacheStatus(RtKernelLaunchArgsEx::TilingCacheStatus::kHit);
  dfx_info_kernel_run();
  ASSERT_EQ(ge::DumpStub::GetInstance().GetDynamicUnits().size(), 1);
  dump_data = ge::DumpStub::GetInstance().GetDynamicUnits()[0];
  ASSERT_EQ(dump_data.size(), 4);  // 第二次命中缓存, 只包含size
  EXPECT_EQ(dump_data[0], 8224);   // input_1 size
  EXPECT_EQ(dump_data[1], 8224);   // output_1 size
  EXPECT_EQ(dump_data[2], 1);      // workspace_size_1
  EXPECT_EQ(dump_data[3], 2);      // workspace_size_2
  ge::DumpStub::GetInstance().Clear();
}
}  // namespace kernel
}  // namespace gert