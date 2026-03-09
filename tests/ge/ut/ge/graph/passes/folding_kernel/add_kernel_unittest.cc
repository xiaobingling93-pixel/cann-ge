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
#include "host_kernels/elewise_calculation_ops/add_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/fp16_t/fp16_t.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestFoldingKernelAddKernel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFoldingKernelAddKernel, AddOptimizeInitSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, static_cast<int64_t>(DT_INT32));

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, AddOptimizerInt32Scalar) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {1};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, AddOptimizerFloatSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  vector<int64_t> dims_vec_0 = {4};
  vector<float> data_vec_0 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1;
  vector<float> data_vec_1 = {1.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

// optimize op of slice success
TEST_F(UtestFoldingKernelAddKernel, OptimizeOpOfSliceSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UNDEFINED);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, AddCheckNullptr) {
  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(nullptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, InputDataSizeCheck1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UNDEFINED);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {nullptr, nullptr};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, InputDataSizeCheck2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UNDEFINED);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  vector<int64_t> dims_vec_0 = {1};
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<float> data_vec_1 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, ComputeComplex32Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);
  vector<int64_t> dims_vec_0 = {2};
  vector<fp16_t> data_vec_0 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_COMPLEX32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(fp16_t));

  vector<int64_t> dims_vec_1 = {2};
  vector<fp16_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_COMPLEX32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(fp16_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  fp16_t *out_data = (fp16_t *)v_output[0]->GetData().data();
  EXPECT_EQ(status, SUCCESS);
  for (size_t i = 0; i < data_vec_0.size(); ++i) {
    EXPECT_EQ(out_data[i], data_vec_0[i] + data_vec_1[i]);
  }
}

TEST_F(UtestFoldingKernelAddKernel, ComputeComplex64Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);
  vector<int64_t> dims_vec_0 = {2};
  vector<float> data_vec_0 = {0.1, 0.2, 0.3, 0.4};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_COMPLEX64);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1 = {2};
  vector<float> data_vec_1 = {0.1, 0.2, 0.3, 0.4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_COMPLEX64);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  float *out_data = (float *)v_output[0]->GetData().data();
  EXPECT_EQ(status, SUCCESS);
  for (size_t i = 0; i < data_vec_0.size(); ++i) {
    EXPECT_EQ(out_data[i], data_vec_0[i] + data_vec_1[i]);
  }
}

TEST_F(UtestFoldingKernelAddKernel, ComputeComplex128Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);
  vector<int64_t> dims_vec_0 = {3};
  vector<double> data_vec_0 = {0.1, 0.2, 0.3, 0.4, 1.5, 1.8};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_COMPLEX128);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(double));

  vector<int64_t> dims_vec_1 = {3};
  vector<double> data_vec_1 = {0.1, 0.2, 0.3, 0.4, 1.3, 2.1};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_COMPLEX128);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(double));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  double *out_data = (double *)v_output[0]->GetData().data();
  EXPECT_EQ(status, SUCCESS);
  for (size_t i = 0; i < data_vec_0.size(); ++i) {
    EXPECT_EQ(out_data[i], data_vec_0[i] + data_vec_1[i]);
  }
}

TEST_F(UtestFoldingKernelAddKernel, ComputeComplex64OverflowFailed) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);
  vector<int64_t> dims_vec_0 = {2};
  vector<float> data_vec_0 = {std::numeric_limits<float>::max(), 0.2, 0.3, 0.4};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_COMPLEX64);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1 = {2};
  vector<float> data_vec_1 = {std::numeric_limits<float>::max(), 0.2, 0.3, 0.4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_COMPLEX64);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_NE(status, SUCCESS);
}
