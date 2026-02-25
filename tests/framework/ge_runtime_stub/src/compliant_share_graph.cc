/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/compliant_share_graph.h"
#include "graph/graph.h"
#include "common/checker.h"
#include "graph/utils/graph_utils_ex.h"
#include "es_ge_test_ops_c.h"
namespace ge {
namespace cg {
ge::ComputeGraphPtr BuildSoftmaxGraph(const std::vector<int64_t> &shape) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);
const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
std::vector<int64_t> axes = {1};
auto softmax = EsSoftmaxV2(data0, axes.data(), static_cast<int64_t>(axes.size()), false);
EsSetGraphOutput(softmax, 0);
auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
ComputeGraphPtr BuildAddGraph(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto add = EsAdd(data0, data1);
EsSetGraphOutput(add, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
ComputeGraphPtr BuildAddNGraph(const vector<std::vector<int64_t>> &shapes) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);
std::vector<EsCTensorHolder *> datas;
auto input_num = shapes.size();
for (int32_t i = 0; static_cast<size_t>(i) < input_num; ++i) {
  const auto data = EsCreateGraphInput(graph.get(), i);
  EsSetShape(data, shapes[i].data(), static_cast<int64_t>(shapes[i].size()));
  datas.push_back(data);
}
auto addn = EsAddN(datas.data(), input_num, input_num);
EsSetGraphOutput(addn, 0);

auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
ComputeGraphPtr BuildAddFillGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto add = EsAdd(data0, data1);
int32_t value = 1;
const auto fill = EsFill(add, EsCreateScalarInt64(graph.get(), value));
EsSetGraphOutput(fill, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
ComputeGraphPtr BuildSoftmaxAddNGraph(const vector<int64_t> &shape, const vector<std::vector<int64_t>> &shapes) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);
const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
std::vector<int64_t> axes = {1};
auto softmax = EsSoftmaxV2(data0, axes.data(), static_cast<int64_t>(axes.size()), false);
std::vector<EsCTensorHolder *> inputs_of_addn;
auto input_num = shapes.size();
for (int32_t i = 0; static_cast<size_t>(i) < input_num; ++i) {
  const auto data = EsCreateGraphInput(graph.get(), i + 1);
  EsSetShape(data, shapes[i].data(), static_cast<int64_t>(shapes[i].size()));
  inputs_of_addn.push_back(data);
}
inputs_of_addn.push_back(softmax);
auto addn = EsAddN(inputs_of_addn.data(), inputs_of_addn.size(), inputs_of_addn.size());
EsSetGraphOutput(addn, 0);

auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildSoftmaxAddGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);
const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
std::vector<int64_t> axes = {1};
auto softmax = EsSoftmaxV2(data0, axes.data(), static_cast<int64_t>(axes.size()), false);
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto add = EsAdd(softmax, data1);
EsSetGraphOutput(add, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
ComputeGraphPtr BuildAddReluReluGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
  const auto data1 = EsCreateGraphInput(graph.get(), 1);
  EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
  const auto add = EsAdd(data0, data1);
  EsSetShape(add, shape1.data(), static_cast<int64_t>(shape1.size()));
  const auto relu0 = EsRelu(add);
  EsSetShape(relu0, shape1.data(), static_cast<int64_t>(shape1.size()));
  const auto abs0 = EsAbs(relu0);
  EsSetShape(abs0, shape1.data(), static_cast<int64_t>(shape1.size()));
  const auto relu1 = EsRelu(add);
  EsSetShape(relu1, shape1.data(), static_cast<int64_t>(shape1.size()));
  const auto abs1 = EsAbs(relu1);
  EsSetShape(abs1, shape1.data(), static_cast<int64_t>(shape1.size()));
  EsSetGraphOutput(abs0, 0);
  EsSetGraphOutput(abs1, 1);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildGraphWithUnConsistantType(const vector<int64_t> &shape) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  const auto abs0 = EsAbs(data0);
  const auto add = EsAdd(abs0, abs0);
  const auto relu0 = EsRelu(add);
  const auto abs1 = EsAbs(relu0);
  const auto relu1 = EsRelu(add);
  const auto abs2 = EsAbs(relu1);
  EsSetGraphOutput(abs1, 0);
  EsSetGraphOutput(abs2, 1);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  auto data0_node = compute_graph->FindNode("input_0");
  auto data0_opdesc = data0_node->GetOpDesc();
  data0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  auto abs0_node = compute_graph->FindNode("Abs_0");
  auto abs0_opdesc = abs0_node->GetOpDesc();
  abs0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  abs0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto add_node = compute_graph->FindNode("Add_1");
  auto add_opdesc = add_node->GetOpDesc();
  add_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  add_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT16);
  add_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto relu0_node = compute_graph->FindNode("Relu_2");
  auto relu0_opdesc = relu0_node->GetOpDesc();
  relu0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  relu0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto abs1_node = compute_graph->FindNode("Abs_3");
  auto abs1_opdesc = abs1_node->GetOpDesc();
  abs1_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  abs1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto relu1_node = compute_graph->FindNode("Relu_4");
  auto relu1_opdesc = relu1_node->GetOpDesc();
  relu1_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  relu1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto abs2_node = compute_graph->FindNode("Abs_5");
  auto abs2_opdesc = abs2_node->GetOpDesc();
  abs2_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  abs2_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto output_node = compute_graph->GetOrUpdateNetOutputNode();
  auto output_opdesc = output_node->GetOpDesc();
  output_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  output_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT);
  output_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  return compute_graph;
}

ComputeGraphPtr BuildGraphWithUnConsistantTypeWithConstInput(const vector<int64_t> &shape) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  std::vector<float> data{0.5, 1.0, 2.0, 0.2, 5.4, 7.1};
  std::vector<int64_t> dims{2, 1, 3};
  const auto const0 = EsCreateConstFloat(graph.get(), data.data(), dims.data(), dims.size());
  const auto abs0 = EsAbs(data0);
  const auto add = EsAdd(abs0, const0);
  const auto relu0 = EsRelu(add);
  const auto abs1 = EsAbs(relu0);
  const auto relu1 = EsRelu(add);
  const auto abs2 = EsAbs(relu1);
  EsSetGraphOutput(abs1, 0);
  EsSetGraphOutput(abs2, 1);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  auto data0_node = compute_graph->FindNode("input_0");
  auto data0_opdesc = data0_node->GetOpDesc();
  data0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  auto const0_node = compute_graph->FindNode("Const_0");
  auto const0_opdesc = const0_node->GetOpDesc();
  const0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  auto abs0_node = compute_graph->FindNode("Abs_1");
  auto abs0_opdesc = abs0_node->GetOpDesc();
  abs0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  abs0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto add_node = compute_graph->FindNode("Add_2");
  auto add_opdesc = add_node->GetOpDesc();
  add_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  add_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT16);
  add_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto relu0_node = compute_graph->FindNode("Relu_3");
  auto relu0_opdesc = relu0_node->GetOpDesc();
  relu0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  relu0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto abs1_node = compute_graph->FindNode("Abs_4");
  auto abs1_opdesc = abs1_node->GetOpDesc();
  abs1_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  abs1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto relu1_node = compute_graph->FindNode("Relu_5");
  auto relu1_opdesc = relu1_node->GetOpDesc();
  relu1_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  relu1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto abs2_node = compute_graph->FindNode("Abs_6");
  auto abs2_opdesc = abs2_node->GetOpDesc();
  abs2_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  abs2_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto output_node = compute_graph->GetOrUpdateNetOutputNode();
  auto output_opdesc = output_node->GetOpDesc();
  output_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  output_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT);
  output_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  return compute_graph;
}

ComputeGraphPtr BuildAbsAddReluReluGraph(const vector<int64_t> &shape) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  const auto abs0 = EsAbs(data0);
  const auto add = EsAdd(abs0, abs0);
  const auto relu0 = EsRelu(add);
  const auto abs1 = EsAbs(relu0);
  const auto relu1 = EsRelu(add);
  const auto abs2 = EsAbs(relu1);
  EsSetGraphOutput(abs1, 0);
  EsSetGraphOutput(abs2, 1);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  auto data0_node = compute_graph->FindNode("input_0");
  auto data0_opdesc = data0_node->GetOpDesc();
  data0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data0_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  data0_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  data0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data0_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  data0_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto abs0_node = compute_graph->FindNode("Abs_0");
  auto abs0_opdesc = abs0_node->GetOpDesc();
  abs0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  abs0_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  abs0_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  abs0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  abs0_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  abs0_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto add_node = compute_graph->FindNode("Add_1");
  auto add_opdesc = add_node->GetOpDesc();
  add_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  add_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  add_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  add_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT);
  add_opdesc->MutableInputDesc(1)->SetShape(GeShape(shape));
  add_opdesc->MutableInputDesc(1)->SetOriginShape(GeShape(shape));
  add_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  add_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  add_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto relu0_node = compute_graph->FindNode("Relu_2");
  auto relu0_opdesc = relu0_node->GetOpDesc();
  relu0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  relu0_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  relu0_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));

  relu0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  relu0_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  relu0_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto abs1_node = compute_graph->FindNode("Abs_3");
  auto abs1_opdesc = abs1_node->GetOpDesc();
  abs1_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  abs1_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  abs1_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  abs1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  abs1_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  abs1_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto relu1_node = compute_graph->FindNode("Relu_4");
  auto relu1_opdesc = relu1_node->GetOpDesc();
  relu1_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  relu1_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  relu1_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  relu1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  relu1_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  relu1_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto abs2_node = compute_graph->FindNode("Abs_5");
  auto abs2_opdesc = abs2_node->GetOpDesc();
  abs2_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  abs2_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  abs2_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  abs2_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  abs2_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  abs2_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto output_node = compute_graph->GetOrUpdateNetOutputNode();
  auto output_opdesc = output_node->GetOpDesc();
  output_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  output_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT);
  output_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  output_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  output_opdesc->MutableInputDesc(1)->SetShape(GeShape(shape));
  output_opdesc->MutableInputDesc(1)->SetOriginShape(GeShape(shape));
  return compute_graph;
}

ComputeGraphPtr BuildNeedInsertCastGraph(const vector<int64_t> &shape) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> end = {-1};
  std::vector<int64_t> strides = {1};
  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  const auto stridedslice0 = EsStridedSliceD(data0, begin.data(), begin.size(),
      end.data(), end.size(), strides.data(), strides.size(), 0, 0, 0, 0, 0);
  const auto stridedslice1 = EsStridedSliceD(data0, begin.data(), begin.size(),
      end.data(), end.size(), strides.data(), strides.size(), 0, 0, 0, 0, 0);
  const auto stridedslice2 = EsStridedSliceD(data0, begin.data(), begin.size(),
      end.data(), end.size(), strides.data(), strides.size(), 0, 0, 0, 0, 0);
  const auto stridedslice3 = EsStridedSliceD(data0, begin.data(), begin.size(),
      end.data(), end.size(), strides.data(), strides.size(), 0, 0, 0, 0, 0);
  EsSetGraphOutput(stridedslice0, 0);
  EsSetGraphOutput(stridedslice1, 1);
  EsSetGraphOutput(stridedslice2, 2);
  EsSetGraphOutput(stridedslice3, 3);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  auto data0_node = compute_graph->FindNode("input_0");
  auto data0_opdesc = data0_node->GetOpDesc();
  data0_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data0_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  data0_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  data0_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data0_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  data0_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto stridedslice0_node = compute_graph->FindNode("StridedSliceD_0");
  auto stridedslice0_node_opdesc = stridedslice0_node->GetOpDesc();
  stridedslice0_node_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice0_node_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  stridedslice0_node_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  stridedslice0_node_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice0_node_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  stridedslice0_node_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto stridedslice1_node = compute_graph->FindNode("StridedSliceD_1");
  auto stridedslice1_node_opdesc = stridedslice1_node->GetOpDesc();
  stridedslice1_node_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice1_node_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  stridedslice1_node_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  stridedslice1_node_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice1_node_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  stridedslice1_node_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto stridedslice2_node = compute_graph->FindNode("StridedSliceD_2");
  auto stridedslice2_node_opdesc = stridedslice2_node->GetOpDesc();
  stridedslice2_node_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice2_node_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  stridedslice2_node_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  stridedslice2_node_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice2_node_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  stridedslice2_node_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto stridedslice3_node = compute_graph->FindNode("StridedSliceD_3");
  auto stridedslice3_node_opdesc = stridedslice3_node->GetOpDesc();
  stridedslice3_node_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice3_node_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  stridedslice3_node_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  stridedslice3_node_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  stridedslice3_node_opdesc->MutableOutputDesc(0)->SetShape(GeShape(shape));
  stridedslice3_node_opdesc->MutableOutputDesc(0)->SetOriginShape(GeShape(shape));

  auto output_node = compute_graph->GetOrUpdateNetOutputNode();
  auto output_opdesc = output_node->GetOpDesc();
  output_opdesc->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  output_opdesc->MutableInputDesc(1)->SetDataType(DT_FLOAT16);
  output_opdesc->MutableInputDesc(2)->SetDataType(DT_FLOAT16);
  output_opdesc->MutableInputDesc(3)->SetDataType(DT_FLOAT16);
  output_opdesc->MutableInputDesc(0)->SetShape(GeShape(shape));
  output_opdesc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  output_opdesc->MutableInputDesc(1)->SetShape(GeShape(shape));
  output_opdesc->MutableInputDesc(1)->SetOriginShape(GeShape(shape));
  output_opdesc->MutableInputDesc(2)->SetShape(GeShape(shape));
  output_opdesc->MutableInputDesc(2)->SetOriginShape(GeShape(shape));
  output_opdesc->MutableInputDesc(3)->SetShape(GeShape(shape));
  output_opdesc->MutableInputDesc(3)->SetOriginShape(GeShape(shape));
  return compute_graph;
}

ComputeGraphPtr BuildAddReluTransposeGraph(const vector<int64_t> &shape1) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));

std::vector<int64_t> perm0 = {0, 1, 2};
const auto trans0 = EsTransposeD(data0, perm0.data(), static_cast<int64_t>(perm0.size()));
std::vector<int64_t> perm1 = {0, 1, 2};
const auto trans1 = EsTransposeD(trans0, perm1.data(), static_cast<int64_t>(perm1.size()));
const auto relu0 = EsRelu(data0);
const auto relu1 = EsRelu(trans1);
const auto add = EsAdd(relu0, relu1);
EsSetGraphOutput(add, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildReduceCase1Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
auto relu0 = EsRelu(data0);
std::vector<int64_t> axes0 = {2};
auto reduce0 = EsReduceSumD(relu0, axes0.data(), static_cast<int64_t>(axes0.size()), true);
auto add0 = EsAdd(reduce0, data1);

EsSetGraphOutput(add0, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildReduceCase2Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
auto sigmoid0 = EsSigmoid(data0);
std::vector<int64_t> axes0 = {2};
auto reduce0 = EsReduceSumD(sigmoid0, axes0.data(), static_cast<int64_t>(axes0.size()), true);
auto add0 = EsAdd(reduce0, data1);
auto abs1 = EsAbs(add0);

EsSetGraphOutput(abs1, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildReduceCase3Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
std::vector<int64_t> axes0 = {2};
auto reduce0 = EsReduceSumD(data0, axes0.data(), static_cast<int64_t>(axes0.size()), true);
auto abs1 = EsAbs(reduce0);

EsSetGraphOutput(abs1, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildAddReluRsqrtMatMulGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                            const vector<int64_t> &shape3) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto add = EsAdd(data0, data1);
const auto relu0 = EsRelu(add);
const auto rsqrt0 = EsRsqrt(relu0);
const auto data2 = EsCreateGraphInput(graph.get(), 2);
EsSetShape(data2, shape3.data(), static_cast<int64_t>(shape3.size()));
const auto matmul0 = EsMatMul(add, data2, nullptr, false, false);

EsSetGraphOutput(rsqrt0, 0);
EsSetGraphOutput(matmul0, 1);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
*   data0  data1
*       \    /
*        add        data2
*       |      \     /
* transposed     matmul
*      |           |
*      |         relu
*      |            |
*      |          abs
*      |        /   |
*      |      abs  relu
*       \     /    /
*         netoutput
*/
ComputeGraphPtr BuildOnlyOneNodeSkipRealizeGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                                const vector<int64_t> &shape3) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto add = EsAdd(data0, data1);
std::vector<int64_t> perm0 = {0, 1};
const auto trans0 = EsTransposeD(add, perm0.data(), static_cast<int64_t>(perm0.size()));
const auto data2 = EsCreateGraphInput(graph.get(), 2);
EsSetShape(data2, shape3.data(), static_cast<int64_t>(shape3.size()));
const auto matmul0 = EsMatMul(add, data2, nullptr, false, false);
const auto relu0 = EsRelu(matmul0);
const auto abs0 = EsAbs(relu0);

const auto relu1 = EsRelu(abs0);
const auto sig1 = EsSigmoid(abs0);

EsSetGraphOutput(relu1, 0);
EsSetGraphOutput(sig1, 1);
EsSetGraphOutput(trans0, 2);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
* data0  data1
*   \    /
*   MatMul
*    |
* sigmoid  date2
*      |  /
*     mul
*    |
*   netoutput
*/
ComputeGraphPtr BuildMatMulSquareMulGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2, const vector<int64_t> &shape3) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));

const auto data2 = EsCreateGraphInput(graph.get(), 2);
EsSetShape(data2, shape3.data(), static_cast<int64_t>(shape3.size()));

const auto matmul0 = EsMatMul(data0, data1, nullptr, false, false);
const auto square = EsSquare(matmul0);

const auto mul = EsMul(square, data2);
EsSetGraphOutput(mul, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
* data0  data1
*   \    /
*    add0
*     |  data2
*     |  /
*    add1
*     |
*     mul
*     |
*   netoutput
*/
ComputeGraphPtr BuildAddAddGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2, const vector<int64_t> &shape3) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));

const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));

const auto data2 = EsCreateGraphInput(graph.get(), 2);
EsSetShape(data2, shape3.data(), static_cast<int64_t>(shape3.size()));

const auto add0 = EsAdd(data0, data1);
const auto add1 = EsAdd(add0, data2);

EsSetGraphOutput(add1, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
*   data0
*     |
*   Square
*     |
*    abs
*     |
*   Square
*     |
*  netoutput
*/
ComputeGraphPtr BuildSquareAbsSquareGraph(const vector<int64_t> &shape1) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));

const auto square0 = EsSquare(data0);
const auto abs0 = EsAbs(square0);
const auto square1 = EsSquare(abs0);

EsSetGraphOutput(square1, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
*   data0
*     |
*    abs
*     |
*    cast
*     |
*  netoutput
*/
ComputeGraphPtr BuildAbsCastGraph(const vector<int64_t> &shape1, int32_t dst_type) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));

const auto abs = EsAbs(data0);
const auto cast = EsCast(abs, dst_type);

EsSetGraphOutput(cast, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildReluReluConcatGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
auto relu0 = EsRelu(data0);
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
auto relu1 = EsRelu(data1);

std::vector<EsCTensorHolder *> concat_input;
concat_input.push_back(relu0);
concat_input.push_back(relu1);
const auto concat0 = EsConcatV2D(concat_input.data(), 2, 0, 2);

EsSetGraphOutput(concat0, 0);
const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

ComputeGraphPtr BuildReluSquareConcatV2Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
auto relu0 = EsRelu(data0);
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
auto square0 = EsSquare(data1);

std::vector<EsCTensorHolder *> concat_input;
concat_input.push_back(relu0);
concat_input.push_back(square0);

auto concat_dim_tensor = EsCreateScalarInt64(graph.get(), 0);
const auto concat0 = EsConcatV2(concat_input.data(), 2, concat_dim_tensor, 2);

EsSetGraphOutput(concat0, 0);
const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
* data0  data1
*   \    /
*   MatMul
*    |
* sigmoid  date2
*      |  /
*     mul
*    |
*   reduce  date3
*      |  /
*     add
*      |
*     abs
*    |
*   netoutput
*/
ComputeGraphPtr BuildMatMulSquareMulReduceGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                              const vector<int64_t> &shape3, const vector<int64_t> &shape4) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto data2 = EsCreateGraphInput(graph.get(), 2);
EsSetShape(data2, shape3.data(), static_cast<int64_t>(shape3.size()));
const auto data3 = EsCreateGraphInput(graph.get(), 3);
EsSetShape(data3, shape4.data(), static_cast<int64_t>(shape4.size()));

const auto matmul0 = EsMatMul(data0, data1, nullptr, false, false);
const auto square = EsSquare(matmul0);

const auto mul = EsMul(square, data2);
std::vector<int64_t> axes = {1};
auto reduce = EsReduceSumD(mul, axes.data(), static_cast<int64_t>(axes.size()), true);
const auto add = EsAdd(reduce, data3);
const auto abs = EsAbs(add);
EsSetGraphOutput(abs, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
* data0  data1
*   \    /
*   MatMul
*    |
*   reduce date2
*      |  /
*     add
*    |
* sigmoid  date3
*      |  /
*     mul
*    |
*   netoutput
*/
ComputeGraphPtr BuildMatMulReduceSquareMulGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                              const vector<int64_t> &shape3, const vector<int64_t> &shape4) {
auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

const auto data0 = EsCreateGraphInput(graph.get(), 0);
EsSetShape(data0, shape1.data(), static_cast<int64_t>(shape1.size()));
const auto data1 = EsCreateGraphInput(graph.get(), 1);
EsSetShape(data1, shape2.data(), static_cast<int64_t>(shape2.size()));
const auto data2 = EsCreateGraphInput(graph.get(), 2);
EsSetShape(data2, shape3.data(), static_cast<int64_t>(shape3.size()));
const auto data3 = EsCreateGraphInput(graph.get(), 3);
EsSetShape(data3, shape4.data(), static_cast<int64_t>(shape4.size()));

const auto matmul0 = EsMatMul(data0, data1, nullptr, false, false);
std::vector<int64_t> axes = {1};
auto reduce = EsReduceSumD(matmul0, axes.data(), static_cast<int64_t>(axes.size()), true);
const auto add = EsAdd(reduce, data2);
const auto square = EsSquare(add);

const auto mul = EsMul(square, data3);
EsSetGraphOutput(mul, 0);

const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
GE_ASSERT_NOTNULL(ge_graph);
return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
}  // namespace cg
}  // namespace ge