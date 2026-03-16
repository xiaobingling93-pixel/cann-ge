/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "graph/ge_error_codes.h"
#include <gtest/gtest.h>
#include "faker/allocator_faker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/kernel_registry.h"
#include "runtime/rt.h"
#include "faker/kernel_run_context_faker.h"
#include "operator_reg.h"

//#include "faker/kernel_run_context_facker.h"
//#include "stub/gert_runtime_stub.h"

namespace ge {
REG_OP(node)
.INPUT(x, TensorType::ALL())
.INPUT(y, TensorType::ALL())
.OUTPUT(x, TensorType::ALL())
.OUTPUT(z, TensorType::ALL())
.OP_END_FACTORY_REG(node);
}

namespace gert {
class EagerOpExecutionContextUT : public testing::Test {
 public:
  void SetUp() override {
    BuildIn2Out2Case();
    BuildDynamicInputCase();

  }
  void TearDown() override {
    in_2_out_2_case_.Finalize();
    dynamic_input_case_.Finalize();
  }
  struct ContextTestCaseHolder {
    // allocator 最后释放
    AllocatorFaker gert_allocator;
    std::vector<gert::Tensor> input_tensors;
    std::vector<gert::Tensor> output_tensors;
    std::shared_ptr<std::vector<gert::GertMemBlock *>> workspace_mems;
    int32_t stream_int;
    FakeKernelContextHolder context_holder;
    void Finalize() {
      // release workspace
      for (auto mem_block : *workspace_mems) {
        mem_block->Free(0);
      }
    }
  };
 protected:
  ContextTestCaseHolder in_2_out_2_case_;
  ContextTestCaseHolder dynamic_input_case_;
 private:
  void BuildIn2Out2Case() {
    in_2_out_2_case_.input_tensors.resize(2);
    in_2_out_2_case_.input_tensors[0] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                                         {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                                         kOnDeviceHbm,                                // placement
                                         ge::DT_FLOAT16,                              // data type
                                         (void *)0x12345};
    in_2_out_2_case_.input_tensors[1] = {{{2, 2, 3, 8}, {2, 2, 3, 8}},    // shape
                                         {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                         kOnDeviceHbm,                                // placement
                                         ge::DT_FLOAT16,                              // data type
                                         (void *)0x234565};
    in_2_out_2_case_.output_tensors.resize(2);
    in_2_out_2_case_.output_tensors[0] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                          kOnDeviceHbm,                                // placement
                                          ge::DT_FLOAT16,                              // data type
                                          nullptr};
    in_2_out_2_case_.output_tensors[1] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                          kOnDeviceHbm,                                // placement
                                          ge::DT_FLOAT16,                              // data type
                                          nullptr};
    in_2_out_2_case_.stream_int = 1;
    in_2_out_2_case_.workspace_mems = std::make_shared<std::vector<gert::GertMemBlock *>>();
    in_2_out_2_case_.workspace_mems->reserve(1UL);

    in_2_out_2_case_.context_holder = EagerOpExecutionContextFaker()
                                          .IrInstanceNum({1, 1})
                                          //.IrOutputInstanceNum({1, 1})
                                          .NodeIoNum(2, 2)
                                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                                          .InputTensor({&in_2_out_2_case_.input_tensors[0], &in_2_out_2_case_.input_tensors[1]})
                                          .OutputTensor({&in_2_out_2_case_.output_tensors[0], &in_2_out_2_case_.output_tensors[1]})
                                          .OutputMem(in_2_out_2_case_.workspace_mems)
                                          .Allocator(&in_2_out_2_case_.gert_allocator)
                                          .Stream(static_cast<void *>(&in_2_out_2_case_.stream_int))
                                          .Build();
  }
  void BuildDynamicInputCase() {
    dynamic_input_case_.input_tensors.resize(2);
    dynamic_input_case_.input_tensors[0] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                                         {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                                         kOnDeviceHbm,                                // placement
                                         ge::DT_FLOAT16,                              // data type
                                         (void *)0x12345};
    dynamic_input_case_.input_tensors[1] = {{{2, 2, 3, 8}, {2, 2, 3, 8}},    // shape
                                         {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                         kOnDeviceHbm,                                // placement
                                         ge::DT_FLOAT16,                              // data type
                                         (void *)0x234565};
    dynamic_input_case_.output_tensors.resize(2);
    dynamic_input_case_.output_tensors[0] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                          kOnDeviceHbm,                                // placement
                                          ge::DT_FLOAT16,                              // data type
                                          nullptr};
    dynamic_input_case_.output_tensors[1] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                          kOnDeviceHbm,                                // placement
                                          ge::DT_FLOAT16,                              // data type
                                          nullptr};
    dynamic_input_case_.stream_int = 1;
    dynamic_input_case_.workspace_mems = std::make_shared<std::vector<gert::GertMemBlock *>>();
    dynamic_input_case_.workspace_mems->reserve(1UL);

    dynamic_input_case_.context_holder = EagerOpExecutionContextFaker()
                                          .IrInstanceNum({0, 2})
                                          //.IrOutputInstanceNum({1, 1})
                                          .NodeIoNum(2, 2)
                                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                                          .InputTensor({&dynamic_input_case_.input_tensors[0], &dynamic_input_case_.input_tensors[1]})
                                          .OutputTensor({&dynamic_input_case_.output_tensors[0], &dynamic_input_case_.output_tensors[1]})
                                          .OutputMem(dynamic_input_case_.workspace_mems)
                                          .Allocator(&dynamic_input_case_.gert_allocator)
                                          .Stream(static_cast<void *>(&dynamic_input_case_.stream_int))
                                          .Build();
  }
};

TEST_F(EagerOpExecutionContextUT, GetInputTensor) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  auto tensor_1_in_context = context->GetInputTensor(0);
  ASSERT_NE(tensor_1_in_context, nullptr);
  EXPECT_EQ(tensor_1_in_context->GetOriginShape(), in_2_out_2_case_.input_tensors[0].GetOriginShape());
  EXPECT_EQ(tensor_1_in_context->GetStorageFormat(), in_2_out_2_case_.input_tensors[0].GetStorageFormat());
  EXPECT_EQ(tensor_1_in_context->GetAddr(), in_2_out_2_case_.input_tensors[0].GetAddr());

  auto tensor_2_in_context = context->GetInputTensor(1);
  ASSERT_NE(tensor_2_in_context, nullptr);
  EXPECT_EQ(tensor_2_in_context->GetOriginShape(), in_2_out_2_case_.input_tensors[1].GetOriginShape());
  EXPECT_EQ(tensor_2_in_context->GetStorageFormat(), in_2_out_2_case_.input_tensors[1].GetStorageFormat());
  EXPECT_EQ(tensor_2_in_context->GetAddr(), in_2_out_2_case_.input_tensors[1].GetAddr());

  EXPECT_EQ(context->GetStream(), &in_2_out_2_case_.stream_int);

  auto tensor_index_out_range_in_context = context->GetInputTensor(2);
  EXPECT_EQ(tensor_index_out_range_in_context, nullptr);
}

TEST_F(EagerOpExecutionContextUT, GetRequiredInputTensor) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  auto tensor_1_in_context = context->GetRequiredInputTensor(0);
  ASSERT_NE(tensor_1_in_context, nullptr);
  EXPECT_EQ(tensor_1_in_context->GetOriginShape(), in_2_out_2_case_.input_tensors[0].GetOriginShape());
  EXPECT_EQ(tensor_1_in_context->GetStorageFormat(), in_2_out_2_case_.input_tensors[0].GetStorageFormat());
  EXPECT_EQ(tensor_1_in_context->GetAddr(), in_2_out_2_case_.input_tensors[0].GetAddr());
}

TEST_F(EagerOpExecutionContextUT, GetOptionalInputTensor) {
  auto dynamic_input_context = dynamic_input_case_.context_holder.GetContext<EagerOpExecutionContext>();
  ASSERT_NE(dynamic_input_context, nullptr);
  auto tensor_0_in_context = dynamic_input_context->GetOptionalInputTensor(0);
  ASSERT_EQ(tensor_0_in_context, nullptr);

  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();
  ASSERT_NE(context, nullptr);
  auto tensor_1_in_context = context->GetOptionalInputTensor(0);
  ASSERT_NE(tensor_1_in_context, nullptr);
  EXPECT_EQ(tensor_1_in_context->GetOriginShape(), in_2_out_2_case_.input_tensors[0].GetOriginShape());
  EXPECT_EQ(tensor_1_in_context->GetStorageFormat(), in_2_out_2_case_.input_tensors[0].GetStorageFormat());
  EXPECT_EQ(tensor_1_in_context->GetAddr(), in_2_out_2_case_.input_tensors[0].GetAddr());
}

TEST_F(EagerOpExecutionContextUT, GetDynamicInputTensor) {
  auto context = dynamic_input_case_.context_holder.GetContext<EagerOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  auto tensor_1_in_context = context->GetDynamicInputTensor(1, 1);
  ASSERT_NE(tensor_1_in_context, nullptr);
  EXPECT_EQ(tensor_1_in_context->GetOriginShape(), dynamic_input_case_.input_tensors[1].GetOriginShape());
  EXPECT_EQ(tensor_1_in_context->GetStorageFormat(), dynamic_input_case_.input_tensors[1].GetStorageFormat());
  EXPECT_EQ(tensor_1_in_context->GetAddr(), dynamic_input_case_.input_tensors[1].GetAddr());
}

TEST_F(EagerOpExecutionContextUT, MallocOutputTensorOk) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  std::initializer_list<int64_t> origin_shape = {2, 1, 3, 4};
  std::initializer_list<int64_t> storage_shape = {1, 2, 3, 4};
  auto output_tensor0 =
      context->MallocOutputTensor(1, {origin_shape, storage_shape}, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()},
                                  ge::DT_FLOAT16);
  ASSERT_NE(output_tensor0, nullptr);
  EXPECT_EQ(output_tensor0->GetOriginShape(), origin_shape);
  EXPECT_EQ(output_tensor0->GetStorageShape(), storage_shape);
  EXPECT_EQ(output_tensor0->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output_tensor0->GetStorageFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output_tensor0->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor0->GetSize(), 512);
  EXPECT_NE(output_tensor0->GetAddr(), nullptr);

  auto output_tensor_0_in_context = context->GetOutputTensor(1);
  ASSERT_NE(output_tensor_0_in_context, nullptr);
  EXPECT_EQ(output_tensor_0_in_context->GetOriginShape(), origin_shape);
  EXPECT_EQ(output_tensor_0_in_context->GetStorageShape(), storage_shape);
  EXPECT_EQ(output_tensor_0_in_context->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output_tensor_0_in_context->GetStorageFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output_tensor_0_in_context->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor_0_in_context->GetSize(), 512);
  EXPECT_NE(output_tensor_0_in_context->GetAddr(), nullptr);
}

TEST_F(EagerOpExecutionContextUT, MallocOutputError) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  std::initializer_list<int64_t> origin_shape = {2, 1, 3, 4};
  std::initializer_list<int64_t> storage_shape = {1, 2, 3, 4};
  auto output_tensor =
      context->MallocOutputTensor(0, {origin_shape, storage_shape}, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()},
                                  ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor, nullptr);
}

TEST_F(EagerOpExecutionContextUT, MallocFreeWorkSpaceOk) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();
  ASSERT_NE(context, nullptr);

  auto block0 = context->MallocWorkSpace(1024);
  ASSERT_NE(block0, nullptr);

  auto block1 = context->MallocWorkSpace(1024);
  ASSERT_NE(block1, nullptr);

  auto kernel_context = reinterpret_cast<KernelContext *>(context);
  auto memory_vec = kernel_context->GetOutputPointer<std::vector<GertMemBlock *>>(context->GetComputeNodeOutputNum());
  ASSERT_NE(memory_vec, nullptr);
  EXPECT_EQ(memory_vec->size(), 2UL);
}

TEST_F(EagerOpExecutionContextUT, MakeOutputRefInput) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();
  ASSERT_NE(context, nullptr);
  auto output_tensor = context->MakeOutputRefInput(0, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(output_tensor->GetOriginShape(), in_2_out_2_case_.input_tensors[0].GetOriginShape());
  EXPECT_EQ(output_tensor->GetStorageFormat(), in_2_out_2_case_.input_tensors[0].GetStorageFormat());
  EXPECT_EQ(output_tensor->GetAddr(), in_2_out_2_case_.input_tensors[0].GetAddr());
}

TEST_F(EagerOpExecutionContextUT, MakeOutputRefInputError) {
  auto context = in_2_out_2_case_.context_holder.GetContext<EagerOpExecutionContext>();
  ASSERT_NE(context, nullptr);
  auto output_tensor = context->MakeOutputRefInput(1, 0);
  ASSERT_EQ(output_tensor, nullptr);
}


}


