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
#include <vector>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "slice/data_slice_helper.h"
#include "slice/data_slice_toolkit.h"
#include "slice/data_slice_factory.h"
#include "slice/data_slice_elementwise_impl.h"
#include "register/infer_axis_slice_registry.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory_impl.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "slice/data_slice_adapter.h"

using namespace std;
using namespace testing;
namespace ge {
class DataSlice : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(Temp) {
    AxisTypeInfo info1;
    info1.SetAxisType(ge::AxisType::ELEMENTWISE);
    std::vector<CutInfo> relate_inputs1 = {{0, {0}}};
    std::vector<CutInfo> relate_outputs1 = {{0, {0}}};
    info1.SetRelateInputs(relate_inputs1);
    info1.SetRelateOutputs(relate_outputs1);
    AxisTypeInfo info2;
    info2.SetAxisType(ge::AxisType::ELEMENTWISE);
    std::vector<CutInfo> relate_inputs2 = {{0, {1}}};
    std::vector<CutInfo> relate_outputs2 = {{0, {1}}};
    info2.SetRelateInputs(relate_inputs2);
    info2.SetRelateOutputs(relate_outputs2);
    AxisTypeInfo info3;
    info3.SetAxisType(ge::AxisType::ELEMENTWISE);
    std::vector<CutInfo> relate_inputs3 = {{0, {2}}};
    std::vector<CutInfo> relate_outputs3 = {{0, {2}}};
    info3.SetRelateInputs(relate_inputs3);
    info3.SetRelateOutputs(relate_outputs3);
    AxisTypeInfo info4;
    info4.SetAxisType(ge::AxisType::ELEMENTWISE);
    std::vector<CutInfo> relate_inputs4 = {{0, {3}}, {1, {0}}};
    std::vector<CutInfo> relate_outputs4 = {{0, {3}}};
    info4.SetRelateInputs(relate_inputs4);
    info4.SetRelateOutputs(relate_outputs4);
    AxisTypeInfo info5;
    info5.SetAxisType(ge::AxisType::ELEMENTWISE);
    std::vector<CutInfo> relate_inputs5 = {{0, {3}}};
    std::vector<CutInfo> relate_outputs5 = {{0, {3}}};
    info5.SetRelateInputs(relate_inputs5);
    info5.SetRelateOutputs(relate_outputs5);

    axis_type = {info1, info2, info3, info4, info5};
    return GRAPH_SUCCESS;
}
INFER_AXIS_TYPE_INFO_REG(Add, Temp);
INFER_AXIS_TYPE_INFO_REG(Cast, Temp);
IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(Func) {
    return GRAPH_FAILED;
}
INFER_AXIS_TYPE_INFO_REG(Softmax, Func);
IMPLEMT_COMMON_INFER_AXIS_SLICE(Temp1) {
  input_param = {{{},{},{},{0,31}}};
  return GRAPH_SUCCESS;
}
INFER_AXIS_SLICE_FUNC_REG(Add, Temp1);
TEST_F(DataSlice, data_slice_helper_1) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_2) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_3) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::UNSPLIT);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_4) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::SLIDINGWINDOW);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_5) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetSliceInfo(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_6) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_7) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_8) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Softmax", "Softmax");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_9) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Softmax", "Softmax");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NC1HWC0_reshape) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  ge::AttrUtils::SetStr(output_desc, ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NH");
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NC1HWC0) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NZ) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_FRACTAL_NZ);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_FRACTAL_NZ);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NoSplit) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  output_desc.SetOriginShape(ge::GeShape({10, 40, 40, 3}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  input_desc0.SetOriginShape(ge::GeShape({10, 40, 40, 3}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NoShape) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  input_desc0.SetOriginShape(ge::GeShape({10, 40, 40, 3}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_elementwise) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_elementwise_addn) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_elementwise_addn_noshape) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_reducemax) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::REDUCEMAX);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_reducegether) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::REDUCEGATHER);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_slidingwindow) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::SLIDINGWINDOW);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_elementwise_impl_failed) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc(ge::GeShape({20, 20, 20, 20}), ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc(ge::GeShape({5, 5, 5, 5}), ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  DataSliceType out_data_slice;
  DataSliceType in_data_slice;
  Operator op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  DataSliceElementwiseImpl dataElementwiseImpl;
  Status ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice, in_data_slice);
  EXPECT_EQ(ret, FAILED);

  DataSliceType in_data_slice_wrong = {{{0}}};
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice, in_data_slice_wrong);
  EXPECT_EQ(ret, FAILED);

  AxisTypeInfo axis_type_info_wrong;
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info_wrong, out_data_slice, in_data_slice);
  EXPECT_EQ(ret, FAILED);

  DataSliceType out_data_slice_wrong = {{{0, 60}}};
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice_wrong, in_data_slice);
  EXPECT_EQ(ret, FAILED);

  out_data_slice_wrong = {{{0, 10}}};
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice_wrong, in_data_slice);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, ValidateAxisIndex_failed) {
  int64_t from_axis = 1;
  const std::vector<std::vector<int64_t>> slice_info;
  int64_t to_axis = 0;
  const std::vector<std::vector<int64_t>> cur_tensor_range = {{1}};
  EXPECT_EQ(false, DataSliceAdapter::ValidateAxisIndex(from_axis, slice_info, to_axis, cur_tensor_range));
  int64_t from_axis_new = 0;
  const std::vector<std::vector<int64_t>> slice_info_new = {{1}};
  EXPECT_EQ(false, DataSliceAdapter::ValidateAxisIndex(from_axis_new, slice_info_new, to_axis, cur_tensor_range)); 
}
}
