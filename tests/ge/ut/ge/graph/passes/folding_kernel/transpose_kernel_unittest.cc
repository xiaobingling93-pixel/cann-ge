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
#include "host_kernels/transformation_ops/transpose_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/fp16_t/fp16_t.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelTransposeKernel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelTransposeKernel, ValidateInput1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("TRANSPOSE", TRANSPOSE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  std::string name2 = "name2";
  GeTensorDesc output_desc2;
  op_desc_ptr->AddOutputDesc(name2, output_desc2);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  TransposeKernel transpose_kernel;
  Status status = transpose_kernel.ValidateInput(op_desc_ptr, input);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelTransposeKernel, ValidateInput2) {
  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  TransposeKernel transpose_kernel;
  Status status = transpose_kernel.ValidateInput(nullptr, input);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelTransposeKernel, ValidateInput3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("TRANSPOSE", TRANSPOSE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  GeTensorDesc input_desc1;
  op_desc_ptr->AddInputDesc(0, input_desc1);

  GeTensorDesc input_desc2;
  op_desc_ptr->AddInputDesc(1, input_desc2);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0};

  TransposeKernel transpose_kernel;
  Status status = transpose_kernel.ValidateInput(op_desc_ptr, input);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelTransposeKernel, ValidateInput4) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("TRANSPOSE", TRANSPOSE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  GeTensorDesc input_desc1;
  op_desc_ptr->AddInputDesc(0, input_desc1);

  GeTensorDesc input_desc2;
  op_desc_ptr->AddInputDesc(1, input_desc2);

  int32_t start = 1, limit = 20;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};

  TransposeKernel transpose_kernel;
  Status status = transpose_kernel.ValidateInput(op_desc_ptr, input);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelTransposeKernel, ComputeParam1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("TRANSPOSE", TRANSPOSE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  GeTensorDesc input_desc1;
  op_desc_ptr->AddInputDesc(0, input_desc1);

  GeTensorDesc input_desc2;
  op_desc_ptr->AddInputDesc(1, input_desc2);

  int32_t start = 1, limit = 20;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  TransposeKernel transpose_kernel;
  Status status = transpose_kernel.Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelTransposeKernel, ComputeComplex64) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("TRANSPOSE", TRANSPOSE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_COMPLEX64);

  std::string name1 = "name1";
  vector<int64_t> output_dims_vec_0 = {2, 2, 2};
  GeTensorDesc output_desc1(GeShape(output_dims_vec_0), FORMAT_NCHW, DT_COMPLEX64);
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  vector<int64_t> dims_vec_0 = {2, 2, 2};
  vector<float> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc input_desc1(GeShape(dims_vec_0), FORMAT_NCHW, DT_COMPLEX64);
  op_desc_ptr->AddInputDesc(0, input_desc1);

  vector<int64_t> dims_vec_1 = {3};
  vector<int32_t> data_vec_1 = {0, 2, 1};
  GeTensorDesc input_desc2(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(1, input_desc2);

  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_COMPLEX64);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  TransposeKernel transpose_kernel;
  Status status = transpose_kernel.Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(SUCCESS, status);
  float *outputs_data = (float *)outputs[0]->GetData().data();
  std::vector<float> expect_output = {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16};
  for (size_t i = 0U; i < 16; ++i) {
    EXPECT_EQ(outputs_data[i], expect_output[i]);
  }
}
