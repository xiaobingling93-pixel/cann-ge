/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/common_kernel_impl/memory_copy.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/host_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/tensor_attr.h"
#include "faker/fake_value.h"
#include "faker/kernel_outputs_faker.h"
#include "graph/utils/math_util.h"
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"
#include "core/utils/tensor_utils.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "tests/depends/runtime/src/runtime_stub.h"

namespace gert {
namespace {
class MyMockRuntime : public ge::RuntimeStub {
 public:
  MOCK_METHOD6(rtMemcpyAsync, int32_t(void *dst, uint64_t dest_max, const void *src, uint64_t count,
                                      rtMemcpyKind_t kind, rtStream_t stream));
};
class MyMockAclRuntime : public ge::AclRuntimeStub {
 public:
  MOCK_METHOD6(aclrtMemcpyAsync, int32_t(void *dst, size_t dest_max, const void *src, size_t src_count,
                                         aclrtMemcpyKind kind, aclrtStream stream));
};
}
namespace kernel {
ge::graphStatus CopyD2H(KernelContext *context);
}
struct MemCopyKernelTest : public testing::Test {
  MemCopyKernelTest() {
    copyD2h = KernelRegistry::GetInstance().FindKernelFuncs("CopyD2H");
  }
  const KernelRegistry::KernelFuncs *copyD2h;
  KernelRegistry &registry = KernelRegistry::GetInstance();
  memory::CachingMemAllocator caching_mem_allocator_{0};
  memory::SingleStreamL2Allocator single_stream_l2_allocator_{&caching_mem_allocator_};
};

TEST_F(MemCopyKernelTest, copyd2h_input_error) {
  auto run_context = BuildKernelRunContext(2, 1);
  ASSERT_NE(kernel::CopyD2H(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MemCopyKernelTest, copyd2h_create_output_success) {
  auto run_context = BuildKernelRunContext(1, 1);
  ASSERT_EQ(copyD2h->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(run_context.value_holder[1].GetValue<void *>(), nullptr);
  run_context.FreeValue(1);
}

TEST_F(MemCopyKernelTest, copyd2h_run_success) {
  auto run_context = BuildKernelRunContext(3, 1);
  auto context = run_context.GetContext<KernelContext>();
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  run_context.value_holder[0].Set((void *)(&tensor_data), nullptr);
  run_context.value_holder[1].Set((void *)device_data.size(), nullptr);

  gert::memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator host_gert_mem_allocator(&host_mem_allocator);
  run_context.value_holder[2].Set((void *)(&host_gert_mem_allocator), nullptr);

  ASSERT_EQ(copyD2h->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(copyD2h->run_func(run_context), ge::GRAPH_SUCCESS);

  auto host_data = context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(host_data, nullptr);
  ASSERT_NE(host_data->GetAddr(), nullptr);
  ASSERT_EQ(host_data->GetStreamId(), 0);
  ASSERT_EQ(memcmp(host_data->GetAddr(), device_data.data(), device_data.size() * sizeof(uint8_t)), 0);
  ASSERT_EQ(host_data->Free(), ge::GRAPH_SUCCESS);
  run_context.FreeValue(3);
}

TEST_F(MemCopyKernelTest, copyd2h_zero_size_skip_memcpy) {
  auto run_context = BuildKernelRunContext(3, 1);
  auto context = run_context.GetContext<KernelContext>();
  GertTensorData tensor_data;
  run_context.value_holder[0].Set(&tensor_data, nullptr);
  run_context.value_holder[1].Set((void *)0, nullptr);

  gert::memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator host_gert_mem_allocator(&host_mem_allocator);
  run_context.value_holder[2].Set((void *)(&host_gert_mem_allocator), nullptr);

  ASSERT_EQ(copyD2h->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(copyD2h->run_func(run_context), ge::GRAPH_SUCCESS);

  auto host_data = context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(host_data, nullptr);
  ASSERT_NE(host_data->GetAddr(), nullptr);
  ASSERT_EQ(host_data->Free(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(host_data->GetSize(), 0);
  run_context.FreeValue(3);
}

TEST_F(MemCopyKernelTest, CopyH2D) {
  auto tensor_holder = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto size = ge::GetSizeInBytes(tensor_holder.GetTensor()->GetStorageShape().GetShapeSize(),
                                 tensor_holder.GetTensor()->GetDataType());
  auto shape = tensor_holder.GetTensor()->GetShape();
  auto datatype = tensor_holder.GetTensor()->GetDataType();

  auto outputs = OutputsHolder::Fake(static_cast<size_t>(kernel::MemoryCopyOutputs::kNum));
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(2 + 4, static_cast<size_t>(kernel::MemoryCopyOutputs::kNum))
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(0x11), &single_stream_l2_allocator_, &input_tensor_data,
                                       reinterpret_cast<void *>(size), &shape, reinterpret_cast<void *>(datatype)}))
          .Outputs(outputs.pointer)
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CopyH2D");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  auto tensor_data =
      run_context->GetOutputPointer<GertTensorData>(static_cast<size_t>(kernel::MemoryCopyOutputs::kAddress));
  ASSERT_NE(tensor_data, nullptr);
  EXPECT_NE(tensor_data->GetAddr(), tensor_holder.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data->GetAddr(), tensor_holder.GetTensor()->GetAddr(), size), 0);

  context_holder.FreeAll();
}

TEST_F(MemCopyKernelTest, CopyH2D_multi_inputs) {
  size_t output_num = 2u;
  auto tensor_holder = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto size = ge::GetSizeInBytes(tensor_holder.GetTensor()->GetStorageShape().GetShapeSize(),
                                 tensor_holder.GetTensor()->GetDataType());
  auto shape = tensor_holder.GetTensor()->GetShape();
  auto datatype = tensor_holder.GetTensor()->GetDataType();

  auto outputs = OutputsHolder::Fake(output_num);
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(2 + 4 + 4, output_num)
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(0x11), &single_stream_l2_allocator_, &input_tensor_data,
                                       reinterpret_cast<void *>(size), &shape, reinterpret_cast<void *>(datatype),
                                       &input_tensor_data, reinterpret_cast<void *>(size), &shape,
                                       reinterpret_cast<void *>(datatype)}))
          .Outputs(outputs.pointer)
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CopyH2D");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  auto tensor_data =
      run_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  EXPECT_NE(tensor_data->GetAddr(), tensor_holder.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data->GetAddr(), tensor_holder.GetTensor()->GetAddr(), size), 0);

  auto tensor_data1 =
      run_context->GetOutputPointer<GertTensorData>(1);
  ASSERT_NE(tensor_data1, nullptr);
  EXPECT_NE(tensor_data1->GetAddr(), tensor_holder.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data1->GetAddr(), tensor_holder.GetTensor()->GetAddr(), size), 0);

  context_holder.FreeAll();
}


TEST_F(MemCopyKernelTest, CopyH2D_Zero_Input) {
  // 校验当用户输入的tensor大小为0时，不应该执行异步拷贝，因为rtMemcpyAsync异步拷贝中memory
  // type为RT_MEMCPY_HOST_TO_DEVICE_EX时，src 的size长度为0会出现校验失败。
  auto tensor_holder = TensorFaker().Placement(kOnHost).Shape({}).Build();
  tensor_holder.GetTensor()->MutableTensorData().SetAddr(nullptr, nullptr);
  tensor_holder.GetTensor()->MutableOriginShape().AppendDim(0U);
  tensor_holder.GetTensor()->MutableStorageShape().AppendDim(0U);
  tensor_holder.GetTensor()->SetSize(0U);
  auto shape = tensor_holder.GetTensor()->GetShape();
  auto data_type = tensor_holder.GetTensor()->GetDataType();
  auto size = ge::RoundUp(ge::GetSizeInBytes(shape.GetStorageShape().GetShapeSize(), data_type), 512);

  auto outputs = OutputsHolder::Fake(static_cast<size_t>(kernel::MemoryCopyOutputs::kNum));
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(2 + 4, static_cast<size_t>(kernel::MemoryCopyOutputs::kNum))
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(0x11), &single_stream_l2_allocator_, &input_tensor_data,
                                       reinterpret_cast<void *>(size), &shape, reinterpret_cast<void *>(data_type)}))
          .Outputs(outputs.pointer)
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CopyH2D");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  auto tensor_data = run_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  // tensor_holder的Addr为null，tensor_data的Addr由malloc申请，不为空，这个地方由于tensor_holder的Size为0就不会执行拷贝，
  // 所以这个地方需要校验不相等
  EXPECT_NE(tensor_data->GetAddr(), tensor_holder.GetTensor()->GetAddr());
  EXPECT_NE(tensor_data->GetSize(), tensor_holder.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data->GetPlacement(), kOnDeviceHbm);

  context_holder.FreeAll();
}

TEST_F(MemCopyKernelTest, MakeSureTensorAtHost) {
  auto run_context = BuildKernelRunContext(4, 1);
  auto context = run_context.GetContext<KernelContext>();
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  run_context.value_holder[0].Set(reinterpret_cast<void *>(0x11), nullptr);  // stream

  gert::memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator host_gert_mem_allocator(&host_mem_allocator);
  run_context.value_holder[1].Set((void *)(&host_gert_mem_allocator), nullptr);

  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};

  run_context.value_holder[2].Set((void *)(&tensor_data), nullptr);
  run_context.value_holder[3].Set((void *)device_data.size(), nullptr);

  auto makeSureTensorAtHost = registry.FindKernelFuncs("MakeSureTensorAtHost");

  ASSERT_EQ(makeSureTensorAtHost->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);

  tensor_data.SetPlacement(kTensorPlacementEnd);
  ASSERT_EQ(makeSureTensorAtHost->run_func(run_context), ge::GRAPH_FAILED);

  tensor_data.SetPlacement(kOnDeviceHbm);
  ASSERT_EQ(makeSureTensorAtHost->run_func(run_context), ge::GRAPH_SUCCESS);

  tensor_data.SetPlacement(kOnHost);
  ASSERT_EQ(makeSureTensorAtHost->run_func(run_context), ge::GRAPH_SUCCESS);

  auto host_data = context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(host_data, nullptr);
  ASSERT_NE(host_data->GetAddr(), nullptr);
  ASSERT_EQ(memcmp(host_data->GetAddr(), device_data.data(), device_data.size() * sizeof(uint8_t)), 0);
  EXPECT_EQ(tensor_data.GetPlacement(), kOnHost);
  ASSERT_EQ(host_data->Free(), ge::GRAPH_SUCCESS);
  run_context.FreeValue(4);
}
TEST_F(MemCopyKernelTest, MakeSureTensorAtHost_ZeroSizeSkipMemcpy) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableOriginShape().AppendDim(0U);
  input_tensor->MutableStorageShape().AppendDim(0U);
  input_tensor->SetSize(0U);
  input_tensor->MutableTensorData().SetAddr((void *)0x01, nullptr);
  rtStream_t stream = (void *)1;
  size_t input_tensor_size = 0;

  gert::memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator host_gert_mem_allocator(&host_mem_allocator);

  GertTensorData tensor_data;
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(4, 1)
                            .Inputs({stream, &host_gert_mem_allocator, &input_tensor_data, (void *)input_tensor_size})
                            .Outputs({&tensor_data})
                            .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  auto kernel_func = gert::KernelRegistry::GetInstance().FindKernelFuncs("MakeSureTensorAtHost")->run_func;
  ASSERT_NE(kernel_func, nullptr);
  ASSERT_EQ(kernel_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(tensor_data.GetSize(), 0);
  ASSERT_EQ(tensor_data.GetPlacement(), kOnHost);
  ASSERT_EQ(tensor_data.GetStreamId(), 0);
}

TEST_F(MemCopyKernelTest, MakeSureTensorAtDevice) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);

  // 校验当用户输入的tensor大小为0时，不应该执行异步拷贝，因为rtMemcpyAsync异步拷贝中memory
  // type为RT_MEMCPY_HOST_TO_DEVICE_EX时，src 的size长度为0会出现校验失败。
  auto tensor_holder3 = TensorFaker().Placement(kOnHost).Shape({}).Build();
  tensor_holder3.GetTensor()->MutableTensorData().SetAddr(nullptr, nullptr);
  tensor_holder3.GetTensor()->MutableOriginShape().AppendDim(0U);
  tensor_holder3.GetTensor()->MutableStorageShape().AppendDim(0U);
  tensor_holder3.GetTensor()->SetSize(0U);
  auto shape3 = tensor_holder3.GetTensor()->GetShape();
  auto data_type3 = tensor_holder3.GetTensor()->GetDataType();
  auto size3 = ge::RoundUp(ge::GetSizeInBytes(shape3.GetStorageShape().GetShapeSize(), data_type3), 512);

  auto outputs = OutputsHolder::Fake(3);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  GertTensorData input_tensor_data3;
  TensorUtils::RefTdToGtd(tensor_holder3.GetTensor()->GetTensorData(), -1, input_tensor_data3);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(14, 3)
          .Inputs(std::vector<void *>(
              {reinterpret_cast<void *>(0x11), &single_stream_l2_allocator_, &input_tensor_data1,
               reinterpret_cast<void *>(size1), &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
               reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2), &input_tensor_data3,
               reinterpret_cast<void *>(size3), &shape3, reinterpret_cast<void *>(data_type3)}))
          .Outputs(outputs.pointer)
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("MakeSureTensorAtDevice");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto tensor_data1 = run_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data1, nullptr);
  EXPECT_NE(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr(),
                   ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1)),
            0);
  EXPECT_EQ(tensor_data1->GetPlacement(), kOnDeviceHbm);

  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_EQ(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data2->GetSize(), tensor_holder2.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto tensor_data3 = run_context->GetOutputPointer<GertTensorData>(2);
  ASSERT_NE(tensor_data3, nullptr);
  // tensor_holder3的Addr为null，tensor_data3的Addr由malloc申请，不为空，这个地方由于tensor_holder3的Size为0就不会执行拷贝，
  // 所以这个地方需要校验不相等
  EXPECT_NE(tensor_data3->GetAddr(), tensor_holder3.GetTensor()->GetAddr());
  EXPECT_NE(tensor_data3->GetSize(), tensor_holder3.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data3->GetPlacement(), kOnDeviceHbm);

  context_holder.FreeAll();
}
TEST_F(MemCopyKernelTest, CopyH2H) {
  auto src_tensor_holder = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto size = ge::GetSizeInBytes(src_tensor_holder.GetTensor()->GetStorageShape().GetShapeSize(),
                                 src_tensor_holder.GetTensor()->GetDataType());
  auto outputs = OutputsHolder::Fake(static_cast<size_t>(kernel::MemoryCopyOutputs::kNum));
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(src_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);

  gert::memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator host_gert_mem_allocator(&host_mem_allocator);

  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::MemoryCopyH2HInputs::kNum),
                       static_cast<size_t>(kernel::MemoryCopyOutputs::kNum))
          .Inputs(std::vector<void *>({&input_tensor_data, &src_tensor_holder.GetTensor()->GetShape(),
                                       reinterpret_cast<void *>(src_tensor_holder.GetTensor()->GetDataType()),
                                       &host_gert_mem_allocator}))
          .Outputs(outputs.pointer)
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CopyTensorDataH2H");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto dst_tensor_data =
      run_context->GetOutputPointer<GertTensorData>(static_cast<size_t>(kernel::MemoryCopyOutputs::kAddress));
  ASSERT_NE(dst_tensor_data, nullptr);
  EXPECT_NE(dst_tensor_data->GetAddr(), src_tensor_holder.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(dst_tensor_data->GetAddr(), src_tensor_holder.GetTensor()->GetAddr(), size), 0);
  EXPECT_EQ(dst_tensor_data->GetStreamId(), 0);
  context_holder.FreeAll();
}

TEST_F(MemCopyKernelTest, CopyD2D) {
  auto src_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto dst_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto size = ge::GetSizeInBytes(src_tensor_holder.GetTensor()->GetStorageShape().GetShapeSize(),
                                 src_tensor_holder.GetTensor()->GetDataType());
  auto stream = reinterpret_cast<void *>(0x11);
  GertTensorData src_tensor_data;
  TensorUtils::RefTdToGtd(src_tensor_holder.GetTensor()->GetTensorData(), -1, src_tensor_data);
  GertTensorData dst_tensor_data;
  TensorUtils::RefTdToGtd(dst_tensor_holder.GetTensor()->GetTensorData(), -1, dst_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::MemoryCopyInputs::kNum), 0U)
          .Inputs(std::vector<void *>({&src_tensor_data, &dst_tensor_data, reinterpret_cast<void *>(size), stream}))
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CopyD2D");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(dst_tensor_holder.GetTensor()->GetAddr(), src_tensor_holder.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(dst_tensor_holder.GetTensor()->GetAddr(), src_tensor_holder.GetTensor()->GetAddr(), size), 0);
  context_holder.FreeAll();
}
TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_TensorDescSetCorrectly_WhenCopyData) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({2, 3, 16}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableOriginShape() = Shape({2, 3, 4});
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}};
  rtStream_t stream = (void *)1;
  auto output_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({100}).Build();
  auto output_tensor = output_tensor_holder.GetTensor();
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
          .Inputs({&input_tensor->MutableOriginShape(), &input_tensor_data, &attr, stream, output_tensor})
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(output_tensor->GetStorageShape(), Shape({2, 3, 16}));
  ASSERT_EQ(output_tensor->GetOriginShape(), Shape({2, 3, 4}));
  ASSERT_EQ(output_tensor->GetDataType(), ge::DT_FLOAT);
  ASSERT_EQ(output_tensor->GetPlacement(), kOnDeviceHbm);
  ASSERT_EQ(output_tensor->GetStorageFormat(), ge::FORMAT_FRACTAL_NZ);
  ASSERT_EQ(output_tensor->GetOriginFormat(), ge::FORMAT_ND);
}
TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_Failed_WhenCopySizeCheckFail) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({2, 3, 4}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableStorageShape() = Shape({2, 3, 16});
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}};
  rtStream_t stream = (void *)1;
  auto output_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({100}).Build();
  auto output_tensor = output_tensor_holder.GetTensor();
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
          .Inputs({&input_tensor->MutableOriginShape(), &input_tensor_data, &attr, stream, output_tensor})
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_FAILED);
  EXPECT_EQ(
      runtime_stub.GetSlogStub().FindErrorLogEndsWith("Failed to copy output tensor data to the given buffer, output "
                                                      "tensor data size 128 is less than copy size size 384"),
      0);
}
TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_Failed_WhenZeroCopySizeCheckFail) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({2, 3, 4}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableStorageShape() = Shape({2, 3, 16});
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}};
  rtStream_t stream = (void *)1;
  auto output_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({1}).Build();
  auto output_tensor = output_tensor_holder.GetTensor();
  GertTensorData input_gert_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_gert_tensor_data);
  auto &out_tensor_data = const_cast<gert::TensorData &>(output_tensor->GetTensorData());
  out_tensor_data.SetAddr(input_tensor->GetTensorData().GetAddr(), nullptr);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0U)
          .Inputs({&input_tensor->MutableOriginShape(), &input_gert_tensor_data, &attr, stream, output_tensor})
          .Build();
  auto kernel_context = context_holder.GetContext<KernelContext>();

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(kernel_context), ge::GRAPH_FAILED);
  EXPECT_EQ(
      runtime_stub.GetSlogStub().FindErrorLogEndsWith("Failed to copy output tensor data to the given buffer, given "
                                                      "tensor data size 64 is less than copy size 128"),
      0);
}
TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_CopyDataOk_WhenCopyData) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({2, 3, 16}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableOriginShape() = Shape({2, 3, 4});
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}};
  rtStream_t stream = (void *)1;
  auto output_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({100}).Build();
  auto output_tensor = output_tensor_holder.GetTensor();
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
          .Inputs({&input_tensor->MutableOriginShape(), &input_tensor_data, &attr, stream, output_tensor})
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(output_tensor->GetAddr(), input_tensor_holder.GetTensor()->GetAddr());
  ASSERT_EQ(
      memcmp(output_tensor->GetAddr(), input_tensor_holder.GetTensor()->GetAddr(), input_tensor->GetShapeSize() * 4),
      0);
}

TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_p2p_CopyDataOk_WhenCopyData) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({2, 3, 16}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableOriginShape() = Shape({2, 3, 4});
  kernel::BuildTensorAttr attr = {kOnDeviceP2p, ge::DT_FLOAT, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}};
  rtStream_t stream = (void *)1;
  auto output_tensor_holder = TensorFaker().Placement(kOnDeviceP2p).Shape({100}).Build();
  auto output_tensor = output_tensor_holder.GetTensor();
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
          .Inputs({&input_tensor->MutableOriginShape(), &input_tensor_data, &attr, stream, output_tensor})
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(output_tensor->GetAddr(), input_tensor_holder.GetTensor()->GetAddr());
  ASSERT_EQ(
      memcmp(output_tensor->GetAddr(), input_tensor_holder.GetTensor()->GetAddr(), input_tensor->GetShapeSize() * 4),
      0);
  EXPECT_EQ(output_tensor->GetPlacement(), kOnDeviceP2p);
}

TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_NoNeedCopyData_WhenCopyDataSrcIsZreo) {
  auto input_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({}).Build();
  auto input_tensor = input_tensor_holder.GetTensor();
  input_tensor->MutableOriginShape().AppendDim(0U);
  input_tensor->MutableStorageShape().AppendDim(0U);
  input_tensor->SetSize(0U);
  input_tensor->MutableTensorData().SetAddr((void *)0x01, nullptr);
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}};
  rtStream_t stream = (void *)1;
  auto output_tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({100}).Build();
  auto output_tensor = output_tensor_holder.GetTensor();
  output_tensor->MutableTensorData().SetAddr((void *)0x02, nullptr);
  GertTensorData input_tensor_data;
  TensorUtils::RefTdToGtd(input_tensor_holder.GetTensor()->GetTensorData(), -1, input_tensor_data);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
          .Inputs({&input_tensor->MutableOriginShape(), &input_tensor_data, &attr, stream, output_tensor})
          .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_SUCCESS);
  // 由于输入input tensor为0,不做拷贝，因此输出的size仍不变, 同时输出的地址也不会发生改变
  ASSERT_EQ(output_tensor->GetSize(), 448U);
  ASSERT_EQ(output_tensor->GetAddr(), (void *)0x02);
}
#if 0
// todo MultiStreamMemBlock 暂时不支持 MoveL2ToL1，在支持后，放开这两个用例
TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_AddressRefOut_WhenNoOutAddress) {
  auto holder = GertTensorDataFaker()
                                 .Placement(kOnDeviceHbm)
                                 .OriginShape({8, 3, 224, 224})
                                 .StorageShape({8, 1, 224, 224, 16})
                                 .DataType(ge::DT_FLOAT16)
                                 .Build();
  // 这里周一看一下，确认kernel输入后修改用例为GertTensorData
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}}};
  rtStream_t stream = (void *)1;
  Tensor output_tensor;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
                            .Inputs({&(holder.ss.MutableOriginShape()), holder.gtd.get(),
                                     &attr, stream, &output_tensor})
                            .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_SUCCESS);

  EXPECT_EQ(output_tensor.GetStorageShape(), Shape({8, 1, 224, 224, 16}));
  EXPECT_EQ(output_tensor.GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output_tensor.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor.GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(output_tensor.GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(output_tensor.GetOriginFormat(), ge::FORMAT_NCHW);
  // out和input中TensorData设置的manager不同，因此直接比较out中的addr_
  EXPECT_EQ(output_tensor.GetAddr(), holder.gtd->GetAddr());
}
TEST_F(MemCopyKernelTest, EnsureTensorAtOutMemory_CeateOutTensor_WhenOutputIsNull) {
  auto holder = GertTensorDataFaker()
                                 .Placement(kOnDeviceHbm)
                                 .OriginShape({8, 3, 224, 224})
                                 .StorageShape({8, 1, 224, 224, 16})
                                 .DataType(ge::DT_FLOAT16)
                                 .Build();
  kernel::BuildTensorAttr attr = {kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}}};
  rtStream_t stream = (void *)1;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kNum), 0)
                            .Inputs({&(holder.ss.MutableOriginShape()), holder.gtd.get(), &attr, stream, nullptr})
                            .Build();
  auto run_context = context_holder.GetContext<KernelContext>();

  ASSERT_EQ(kernel::EnsureTensorAtOutMemory(run_context), ge::GRAPH_SUCCESS);

  auto output_tensor =
      run_context->GetInputPointer<Tensor>(static_cast<size_t>(kernel::EnsureTensorAtOutMemoryInputs::kOutputData));
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(output_tensor->GetStorageShape(), Shape({8, 1, 224, 224, 16}));
  EXPECT_EQ(output_tensor->GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output_tensor->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor->GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(output_tensor->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(output_tensor->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(output_tensor->GetAddr(), holder.gtd->GetAddr());
}
#endif
void ConstructStringTensor(Tensor *input) {
  auto elem_cnt = input->GetShapeSize();
  uint64_t single_str_size = 64U;
  uint64_t total_size = elem_cnt * (single_str_size + sizeof(ge::StringHead) + 1U);
  uint8_t *addr = reinterpret_cast<uint8_t *>(input->GetAddr());
  uint64_t offset = elem_cnt * sizeof(ge::StringHead);

  for (int64_t i = 0; i < elem_cnt; i++) {
    ge::StringHead *string_head = reinterpret_cast<ge::StringHead *>(addr);
    string_head->len = single_str_size;
    string_head->addr = offset;
    offset += single_str_size + 1U;
    addr += sizeof(ge::StringHead);
  }
  input->SetSize(total_size);
  return;
}

TEST_F(MemCopyKernelTest, CalcStringTensorSize_FromDevice) {
  // 内存中包含\0，len不包含\0，所以计算内存长度需要额外+1
  const size_t tensor_size = 2 * 64 * (16 + 64 + 1);
  auto mem_block = single_stream_l2_allocator_.Malloc(tensor_size);

  auto i0 = FakeValue<Tensor>(Tensor{
      {{64, 2}, {64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_STRING, mem_block->GetAddr()});
  GertTensorData tensor_data = {tensor_size, kOnDeviceHbm, single_stream_l2_allocator_.GetStreamId(), mem_block};
  auto shape = i0.holder.get()->GetShape();
  auto data_type = ge::DT_STRING;
  ConstructStringTensor(i0.holder.get());
  rtStream_t stream = (void *)1;

  auto run_context = BuildKernelRunContext(4, 1);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(data_type), nullptr);
  run_context.value_holder[1].Set(&shape, nullptr);
  run_context.value_holder[2].Set(&tensor_data, nullptr);
  run_context.value_holder[3].Set(stream, nullptr);

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CalcStringTensorSize");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto return_size = run_context.value_holder[4].GetValue<uint64_t>();
  ASSERT_EQ(return_size, tensor_size);
}

TEST_F(MemCopyKernelTest, CalcStringTensorSize_FromHost) {
  // 内存中包含\0，len不包含\0，所以计算内存长度需要额外+1
  const size_t tensor_size = 2 * 64 * (16 + 64 + 1);
  auto mem_block = single_stream_l2_allocator_.Malloc(tensor_size);

  auto i0 = FakeValue<Tensor>(Tensor{
      {{64, 2}, {64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnHost, ge::DT_STRING, mem_block->GetAddr()});
  GertTensorData tensor_data = {tensor_size, kOnHost, single_stream_l2_allocator_.GetStreamId(), mem_block};
  auto shape = i0.holder.get()->GetShape();
  auto data_type = ge::DT_STRING;
  ConstructStringTensor(i0.holder.get());
  rtStream_t stream = (void *)1;

  auto run_context = BuildKernelRunContext(4, 1);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(data_type), nullptr);
  run_context.value_holder[1].Set(&shape, nullptr);
  run_context.value_holder[2].Set(&tensor_data, nullptr);
  run_context.value_holder[3].Set(stream, nullptr);

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CalcStringTensorSize");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto return_size = run_context.value_holder[4].GetValue<uint64_t>();
  ASSERT_EQ(return_size, tensor_size);
}

TEST_F(MemCopyKernelTest, CalcStringTensorSize_DtypeNotStringError) {
  // 内存中包含\0，len不包含\0，所以计算内存长度需要额外+1
  const size_t tensor_size = 2 * 64 * (16 + 64 + 1);
  auto mem_block = single_stream_l2_allocator_.Malloc(tensor_size);

  auto i0 = FakeValue<Tensor>(Tensor{
      {{64, 2}, {64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_STRING, mem_block->GetAddr()});
  GertTensorData tensor_data = {tensor_size, kOnDeviceHbm, single_stream_l2_allocator_.GetStreamId(), mem_block};
  auto shape = i0.holder.get()->GetShape();
  auto data_type = ge::DT_FLOAT;
  ConstructStringTensor(i0.holder.get());
  rtStream_t stream = (void *)1;

  auto run_context = BuildKernelRunContext(4, 1);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(data_type), nullptr);
  run_context.value_holder[1].Set(&shape, nullptr);
  run_context.value_holder[2].Set(&tensor_data, nullptr);
  run_context.value_holder[3].Set(stream, nullptr);

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CalcStringTensorSize");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  EXPECT_NE(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MemCopyKernelTest, CalcStringTensorSize_SizeExceedError) {
  // 内存中包含\0，len不包含\0，所以计算内存长度需要额外+1
  const size_t tensor_size = 2 * 64 * (16 + 64 + 1);
  auto mem_block = single_stream_l2_allocator_.Malloc(tensor_size);

  auto i0 = FakeValue<Tensor>(Tensor{
      {{64, 8}, {64, 8}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_STRING, mem_block->GetAddr()});
  GertTensorData tensor_data = {tensor_size, kOnDeviceHbm, single_stream_l2_allocator_.GetStreamId(), mem_block};
  auto shape = i0.holder.get()->GetShape();
  auto data_type = ge::DT_FLOAT;
  ConstructStringTensor(i0.holder.get());
  rtStream_t stream = (void *)1;

  auto run_context = BuildKernelRunContext(4, 1);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(data_type), nullptr);
  run_context.value_holder[1].Set(&shape, nullptr);
  run_context.value_holder[2].Set(&tensor_data, nullptr);
  run_context.value_holder[3].Set(stream, nullptr);

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("CalcStringTensorSize");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  EXPECT_NE(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
}


TEST_F(MemCopyKernelTest, SinkWeightDataTestFail) {
  ASSERT_NE(registry.FindKernelFuncs("SinkWeightData"), nullptr);
  int64_t weight_size = 110;
  std::vector<uint8_t> weight_data(weight_size, 1);
  GertTensorData weight_info = {weight_data.data(), static_cast<size_t>(weight_size), kOnDeviceHbm, -1};
  size_t smaill_size = 100;
  void *device_mem1 = malloc(smaill_size);
  // weight_size > smaill_size ,fail

  GertTensorData tensor_data1 = {device_mem1, smaill_size, kOnDeviceHbm, -1};

  rtStream_t stream;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SinkWeightDataInputs::kNum),
                                         static_cast<size_t>(kernel::SinkWeightDataOutputs::kNum))
                            .Inputs({&weight_info, &tensor_data1, &single_stream_l2_allocator_, &stream})
                            .Build();
  auto invalid_context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SinkWeightData")->outputs_creator(nullptr, invalid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkWeightData")->run_func(invalid_context), ge::GRAPH_FAILED);
  free(device_mem1);
}

TEST_F(MemCopyKernelTest, SinkWeightDataTestSuccess) {
  ASSERT_NE(registry.FindKernelFuncs("SinkWeightData"), nullptr);
  size_t big_size = 200;
  int64_t weight_size = 100;
  std::vector<uint8_t> weight_data(weight_size, 1);
  GertTensorData weight_info = {weight_data.data(), static_cast<size_t>(weight_size), kOnDeviceHbm, -1};
  void *device_mem2 = malloc(big_size);
  // weight_size < big_size ,success
  GertTensorData tensor_data1 = {device_mem2, big_size, kOnDeviceHbm, -1};

  bool memcpy_async_has_been_called = false;
  auto MockRtMemcpyAsync = [&memcpy_async_has_been_called] (void *dst, uint64_t dest_max, const void *src,
                                                             uint64_t count, rtMemcpyKind_t kind,
                                                             rtStream_t stream) {
    if (kind == RT_MEMCPY_HOST_TO_DEVICE) {
      memcpy_async_has_been_called = true;
    }
    return 0;
  };
  auto MockAclrtMemcpyAsync = [&memcpy_async_has_been_called] (void *dst,
                                                              size_t dest_max,
                                                              const void *src,
                                                              size_t src_count,
                                                              aclrtMemcpyKind kind,
                                                              aclrtStream stream) {
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
      memcpy_async_has_been_called = true;
    }
    return 0;
  };
  auto runtime_stub = std::make_shared<MyMockRuntime>();
  auto acl_runtime_stub = std::make_shared<MyMockAclRuntime>();
  ge::RuntimeStub::SetInstance(runtime_stub);
  ge::AclRuntimeStub::SetInstance(acl_runtime_stub);
  EXPECT_CALL(*runtime_stub, rtMemcpyAsync).WillRepeatedly(testing::Invoke(MockRtMemcpyAsync));
  EXPECT_CALL(*acl_runtime_stub, aclrtMemcpyAsync).WillRepeatedly(testing::Invoke(MockAclrtMemcpyAsync));

  rtStream_t stream;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SinkWeightDataInputs::kNum),
                                         static_cast<size_t>(kernel::SinkWeightDataOutputs::kNum))
                            .Inputs({&weight_info, &tensor_data1, &single_stream_l2_allocator_, &stream})
                            .Build();
  auto valid_context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SinkWeightData")->outputs_creator(nullptr, valid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkWeightData")->run_func(valid_context), ge::GRAPH_SUCCESS);
  free(device_mem2);
  ASSERT_TRUE(memcpy_async_has_been_called);
  ge::RuntimeStub::Reset();
  ge::AclRuntimeStub::Reset();
}

TEST_F(MemCopyKernelTest, SinkWeightDataZeroStillSuccess) {
  ASSERT_NE(registry.FindKernelFuncs("SinkWeightData"), nullptr);
  size_t big_size = 10;
  int64_t weight_size = 0;
  std::vector<uint8_t> weight_data(weight_size, 1);
  GertTensorData weight_info = {weight_data.data(), static_cast<size_t>(weight_size), kOnDeviceHbm, -1};
  void *device_mem2 = malloc(big_size);
  // weight_size(0) < big_size , skip memcopy
  GertTensorData tensor_data1 = {device_mem2, big_size, kOnDeviceHbm, -1};
  rtStream_t stream;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::SinkWeightDataInputs::kNum),
                                         static_cast<size_t>(kernel::SinkWeightDataOutputs::kNum))
                            .Inputs({&weight_info, &tensor_data1, &single_stream_l2_allocator_, &stream})
                            .Build();
  auto valid_context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SinkWeightData")->outputs_creator(nullptr, valid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkWeightData")->run_func(valid_context), ge::GRAPH_SUCCESS);
  free(device_mem2);
}
}  // namespace gert