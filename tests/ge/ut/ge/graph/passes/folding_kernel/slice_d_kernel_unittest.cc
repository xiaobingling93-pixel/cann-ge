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
#include "host_kernels/selection_ops/slice_d_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
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
using ge::SUCCESS;

class UtestGraphPassesFoldingKernelSliceDKernel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, SliceDCheck1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("SLICED", SLICED);
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

  SliceDKernel slice_d_kernel;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> size_list;
  Status status = slice_d_kernel.SliceDCheck(op_desc_ptr, input, begin_list, size_list);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, SliceDCheck2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("SLICED", SLICED);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  std::string name2 = "name2";
  GeTensorDesc output_desc2;
  op_desc_ptr->AddOutputDesc(name2, output_desc2);

  GeTensorDesc input_desc1;
  op_desc_ptr->AddInputDesc(0, input_desc1);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0};

  SliceDKernel slice_d_kernel;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> size_list;
  Status status = slice_d_kernel.SliceDCheck(op_desc_ptr, input, begin_list, size_list);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, SliceDCheck3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("SLICED", SLICED);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  GeTensorDesc input_desc1;
  op_desc_ptr->AddInputDesc(0, input_desc1);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0};

  SliceDKernel slice_d_kernel;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> size_list;
  Status status = slice_d_kernel.SliceDCheck(op_desc_ptr, input, begin_list, size_list);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, ComputeParam1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("SLICED", SLICED);
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> outputs;

  SliceDKernel slice_d_kernel;
  Status status = slice_d_kernel.Compute(op_desc_ptr, input, outputs);
  EXPECT_NE(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, ComputeParam2) {
  OpDescPtr op_desc_ptr = nullptr;

  int32_t start = 1, limit = 20, delta = 2;

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

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  SliceDKernel slice_d_kernel;
  Status status = slice_d_kernel.Compute(op_desc_ptr, input, outputs);
  EXPECT_NE(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, OutputSliceDataBool) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("SLICED", SLICED);
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));

  std::vector<int64_t> x_dims;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> size_list;
  std::vector<int64_t> stride_list;
  stride_list.push_back(1);
  size_list.push_back(1);
  begin_list.push_back(1);
  x_dims.push_back(1);
  int32_t start = 1;
  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr x_tensor = 
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  void *data = reinterpret_cast<void *>(const_cast<uint8_t *>(x_tensor->GetData().data()));
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  Status ret = OpUtils::SetOutputSliceData(data, 32, DT_BOOL, x_dims, begin_list, size_list,
                                    output_ptr.get(), stride_list);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestGraphPassesFoldingKernelSliceDKernel, CheckOutputDims1) {
  OpDescPtr op_desc_ptr = nullptr;
  std::vector<int64_t> output_dims = {1,1,1};

  SliceDKernel slice_d_kernel;
  Status status = slice_d_kernel.CheckOutputDims(output_dims, op_desc_ptr);
  EXPECT_EQ(SUCCESS, status);
}
