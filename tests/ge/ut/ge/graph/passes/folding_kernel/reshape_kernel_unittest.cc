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
#include "host_kernels/array_ops/reshape_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelReshapeKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type, Format format>
  void EXPECT_TestInvalidReshape(vector<int64_t> &data_vec, vector<inner_dim_type> &dim_value_vec, vector<int64_t> &result) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

    ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    vector<inner_data_type> data_value_vec(dims_size, inner_data_type(1));
    GeTensorDesc data_tensor_desc(GeShape(data_vec), format, data_type);
    GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
    vector<int64_t> dim_vec;
    dim_vec.push_back(dim_value_vec.size());
    GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), format, dim_type);
    GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                        dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add expanddims node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    expanddim_op_desc->AddInputDesc(dim_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
    vector<ConstGeTensorPtr> input = {data_tensor};
    vector<GeTensorPtr> outputs;
    Status status = kernel->Compute(op_node->GetOpDesc(), input, outputs);
    EXPECT_EQ(NOT_CHANGED, status);
  }
};

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, InvalidFormat) {
  vector<int64_t> data_vec = {2, 3};
  vector<int64_t> dim_value_vec = {-1};
  vector<int64_t> result = {0};

  EXPECT_TestInvalidReshape<int32_t, DT_INT32, int64_t, DT_INT64, FORMAT_FRACTAL_Z>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, FoldingInt64Success) {
  vector<int64_t> data_vec = {3, 4, 2, 2, 8};
  vector<int64_t> dim_value_vec = {12, -1};
  vector<int64_t> result = {12, 32};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int64_t> dim_vec;
  dim_vec.push_back(dim_value_vec.size());
  GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  expanddim_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_node->GetOpDesc(), input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, OpdescIsNullFailed) {
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  ge::OpDescPtr null_op_desc = nullptr;
  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(null_op_desc, input, outputs);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, InputOutputShapeSizeNotMatch) {
  vector<int64_t> input_shape = {3, 4, 2, 2, 8};
  vector<int64_t> invalid_output_shape = {3, 4, 2, 8};
  vector<int64_t> dim_value_vec = {12, -1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(input_shape.begin(), input_shape.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(input_shape), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add shape const node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int64_t> dim_vec;
  dim_vec.push_back(dim_value_vec.size());
  GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add reshape node
  OpDescPtr reshape_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  reshape_op_desc->AddInputDesc(data_tensor_desc);
  reshape_op_desc->AddInputDesc(dim_tensor_desc);
  GeTensorDesc invalid_output_tensor_desc(GeShape(invalid_output_shape), FORMAT_NCHW, DT_BOOL);
  reshape_op_desc->AddOutputDesc(invalid_output_tensor_desc);
  NodePtr op_node = graph->AddNode(reshape_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_node->GetOpDesc(), input, outputs);
  EXPECT_EQ(ge::NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, InputOutputShapeSizeNotMatchButScalar) {
  vector<int64_t> input_shape = {};
  vector<int64_t> output_shape = {1};
  vector<int64_t> dim_value_vec = {1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(input_shape.begin(), input_shape.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(input_shape), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add shape const node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int64_t> dim_vec;
  dim_vec.push_back(dim_value_vec.size());
  GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add reshape node
  OpDescPtr reshape_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  reshape_op_desc->AddInputDesc(data_tensor_desc);
  reshape_op_desc->AddInputDesc(dim_tensor_desc);
  GeTensorDesc output_tensor_desc(GeShape(output_shape), FORMAT_NCHW, DT_BOOL);
  reshape_op_desc->AddOutputDesc(output_tensor_desc);
  NodePtr op_node = graph->AddNode(reshape_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_node->GetOpDesc(), input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
}

