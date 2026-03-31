/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/share_graph.h"
#include "common/checker.h"
#include "common/ge_inner_attrs.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "array_ops.h"
#include "runtime/mem.h"
#include "faker/fake_value.h"
#include "common/tbe_handle_store/kernel_store.h"

#include <stdbool.h>
#include <omg/parser/parser_types.h>
#include <symengine/add.h>

namespace ge {
REG_OP(ConditionCalc)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .OUTPUT(cond, TensorType({DT_INT32}))
    .REQUIRED_ATTR(cond_func, String)
    .REQUIRED_ATTR(x_dependency, ListInt)
    .OP_END_FACTORY_REG(ConditionCalc);

REG_OP(Case)
    .INPUT(branch_index, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(Case);

REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If);

REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSum);

REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMax);

REG_OP(Abs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Abs);

REG_OP(Relu)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Relu);

REG_OP(Exp)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Exp);

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add);

REG_OP(MinimumGrad)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(grad_x, Bool, true)
    .ATTR(grad_y, Bool, true)
    .OP_END_FACTORY_REG(MinimumGrad);

REG_OP(TestNoInferShapeRange)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                          DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(TestNoInferShapeRange);

REG_OP(CustomOp)
    .INPUT(x1, TensorType::ALL())
    .INPUT(x2, TensorType::ALL())
    .INPUT(x3, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(CustomOp);

IMPLEMT_INFERFUNC(TestNoInferShapeRange, TestNoInferShapeRangeInfer) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  const auto &input_desc = op_desc->MutableInputDesc(0);
  op_desc->UpdateOutputDesc(0, input_desc->Clone());
  return GRAPH_SUCCESS;
}
}  // namespace ge

using namespace ge;
namespace gert {
namespace {
void SetNoStorage(const ge::OpDescPtr &op_desc, Format format, DataType dt, std::initializer_list<int64_t> shape) {
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(format);
    op_desc->MutableInputDesc(i)->SetOriginFormat(format);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetDataType(dt);
    op_desc->MutableInputDesc(i)->SetOriginDataType(dt);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(format);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(format);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetDataType(dt);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(dt);
  }
}
void SetShapeRangeNoStorage(const ge::OpDescPtr &op_desc, std::initializer_list<int64_t> min_shape,
                            std::initializer_list<int64_t> max_shape) {
  std::vector<std::pair<int64_t, int64_t>> range;
  for (size_t i = 0; i < min_shape.size(); ++i) {
    range.emplace_back(min_shape.begin()[i], max_shape.begin()[i]);
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetOriginShapeRange(range);
    op_desc->MutableInputDesc(i)->SetShapeRange(range);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetOriginShapeRange(range);
    op_desc->MutableOutputDesc(i)->SetShapeRange(range);
  }
}
template <typename T, ge::DataType DT>
void SetConstValue(const ge::NodePtr &const_node, std::vector<T> value) {
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT);
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({static_cast<int64_t>(value.size())}));
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({static_cast<int64_t>(value.size())}));
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  GeTensor tensor(const_node->GetOpDesc()->GetOutputDesc(0));
  std::vector<T> tensor_data(value);
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data.data()), tensor_data.size() * sizeof(T));
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", tensor);
}
void SetSubGraph(ComputeGraphPtr &parent_graph, NodePtr &parent_node, ComputeGraphPtr &sub_graph) {
  parent_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);

  size_t index = parent_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  parent_node->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
  parent_node->GetOpDesc()->SetSubgraphInstanceName(index, sub_graph->GetName());

  sub_graph->SetParentNode(parent_node);
  sub_graph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
}
void SetOffsetForDataNetoutput(ge::ComputeGraphPtr &graph) {
  for (const auto node : graph->GetAllNodes()) {
    if ((node->GetType() == DATA) || (node->GetType() == REFDATA)) {
      static int64_t offset = 0;
      node->GetOpDescBarePtr()->SetOutputOffset({offset});
      offset += 1024;
    }
    if (node->GetType() == NETOUTPUT) {
      static int64_t offset = 102400;
      std::vector<int64_t> offsets;
      for (size_t i = 0u; i < node->GetOpDescBarePtr()->GetInputsSize(); ++i) {
        offsets.emplace_back(offset);
      }
      node->GetOpDescBarePtr()->SetInputOffset(offsets);
    }
  }
}
GeTensor CreateVecTorGeTensor(std::vector<int64_t> shape, DataType type) {
  uint32_t shape_size = 1;
  for (auto dim : shape) {
    shape_size *= dim;
  }
  const auto data_size = ge::GetSizeInBytes(shape_size, type);
  return GeTensor(GeTensorDesc(GeShape(shape), FORMAT_ND, type), reinterpret_cast<uint8_t *>(shape.data()), data_size);
}
GeTensor CreateScalarGeTensor(int32_t v) {
  return GeTensor(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32), reinterpret_cast<uint8_t *>(&v), sizeof(v));
}
}  // namespace
void SetGraphOutShapeRange(const ge::ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    for (auto &output_tensor : node->GetOpDesc()->GetAllOutputsDescPtr()) {
      output_tensor->SetShapeRange({{1, 100}});
    }
  }
}
ge::NodePtr NodeBuilder::Build(ge::ComputeGraphPtr &parent) {
  auto node = parent->AddNode(desc_);

  for (auto &ctrl : control_inputs_) {
    ctrl->GetOutControlAnchor()->LinkTo(node->GetInControlAnchor());
  }
  std::vector<std::string> input_names;
  std::vector<int64_t> input_indexes;
  for (auto &item : inputs_) {
    for (auto &i2o : item.second) {  // self input : src output
      auto self_input = i2o.first;
      auto src_output = i2o.second;
      GE_ASSERT_GRAPH_SUCCESS(
          ge::GraphUtils::AddEdge(item.first->GetOutDataAnchor(src_output), node->GetInDataAnchor(self_input)));
      input_names.resize(self_input + 1U);
      input_indexes.resize(self_input + 1U);
      input_names[self_input] = item.first->GetName();
      input_indexes[self_input] = src_output;
    }
  }

  node->GetOpDesc()->SetSrcName(input_names);
  node->GetOpDesc()->SetSrcIndex(input_indexes);

  for (auto &item : subgraphs_) {
    auto &graph = item.second;
    auto &attr_name = item.first;
    auto index = node->GetOpDesc()->GetSubgraphInstanceNames().size();
    node->GetOpDesc()->AddSubgraphName(attr_name);
    node->GetOpDesc()->SetSubgraphInstanceName(index, graph->GetName());
    graph->SetParentNode(node);
    graph->SetParentGraph(parent);
    auto root_graph = ge::GraphUtils::FindRootGraph(parent);
    root_graph->AddSubGraph(graph);
    for (auto &subgraph : graph->GetAllSubgraphs()) {
      graph->RemoveSubGraph(subgraph);
      root_graph->AddSubGraph(subgraph);
    }
  }

  return node;
}

void AddCompileResult(const ge::NodePtr &node, bool atomic, const char *compile_info_json) {
  AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", compile_info_json);
  AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 2048);
  auto bin = std::make_shared<OpKernelBin>("name", std::vector<char>({'F', 'a', 'k', 'e', 'b', 'i', 'n'}));
  node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, bin);
  AttrUtils::SetStr(node->GetOpDesc(), TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  AttrUtils::SetStr(node->GetOpDesc(), TVM_ATTR_NAME_METADATA, "FakeMeta");
  AttrUtils::SetStr(node->GetOpDesc(), node->GetName() + "_kernelname", "FakeKernelName");

  if (atomic) {
    AttrUtils::SetStr(node->GetOpDesc(), "_atomic_compile_info_json", "{}");
    AttrUtils::SetInt(node->GetOpDesc(), "atomic_op_para_size", 2048);

    auto atomic_bin = std::make_shared<OpKernelBin>(
        "name", std::vector<char>({'F', 'a', 'k', 'e', 'A', 't', 'o', 'm', 'i', 'c', 'B', 'i', 'n'}));
    node->GetOpDesc()->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_bin);
    AttrUtils::SetStr(node->GetOpDesc(), ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    AttrUtils::SetStr(node->GetOpDesc(), ATOMIC_ATTR_TVM_METADATA, "FakeAtomicMeta");
    AttrUtils::SetStr(node->GetOpDesc(), node->GetName() + "_atomic_kernelname", "FakeAtomicKernelName");
  }
}

/*
 *
 *  transdata1
 *     |
 *   data1
 */
ComputeGraphPtr ShareGraph::BuildAtomicAicoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("trans1", "TransData")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  auto trans1 = graph->FindNode("trans1");
  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"trans1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  AddCompileResult(trans1, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(trans1->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(trans1->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(trans1->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");

  return graph;
}

/*
  ┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
  │ data1 │ ───────> │ trans1 │ ───────> │ memset │ ───────> │ NetOutput │
  └───────┘          └────────┘          └────────┘          └───────────┘
 */
ComputeGraphPtr ShareGraph::BuildMemSetAicoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("trans1", "TransData")->NODE("memset", "MemSet")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  auto trans1 = graph->FindNode("trans1");
  auto memset = graph->FindNode("memset");
  memset->GetOpDesc()->SetSrcName({"trans1"});
  memset->GetOpDesc()->SetSrcIndex({0});
  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"memset"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  AddCompileResult(trans1, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(trans1->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetListDataType(trans1->GetOpDesc(), "tbe_op_atomic_dtypes", {ge::DT_FLOAT});
  AttrUtils::SetStr(trans1->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(trans1->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 2}}");

  AddCompileResult(memset, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(memset->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetListDataType(memset->GetOpDesc(), "tbe_op_atomic_dtypes", {ge::DT_FLOAT});
  AttrUtils::SetStr(memset->GetOpDesc(), "compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(memset->GetOpDesc(), "compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 2}}");

  return graph;
}

/*
 *
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::AicoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  AttrUtils::SetStr(data1->GetOpDesc(), "_op_aicore_num", "2");
  AttrUtils::SetStr(data1->GetOpDesc(), "_op_vectorcore_num", "4");
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  AttrUtils::SetStr(data2->GetOpDesc(), "_op_aicore_num", "2");
  AttrUtils::SetStr(data2->GetOpDesc(), "_op_vectorcore_num", "4");
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1});
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  add1->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  add1->GetOpDesc()->AppendIrInput("x1", kIrInputRequired);
  add1->GetOpDesc()->AppendIrInput("x2", kIrInputRequired);
  ge::AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  AttrUtils::SetStr(add1->GetOpDesc(), "_op_aicore_num", "2");
  AttrUtils::SetStr(add1->GetOpDesc(), "_op_vectorcore_num", "4");
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  SetGraphOutShapeRange(graph);
  return graph;
}

 /*
  *  customop
  *  |  \     \
  *  |   \     \
  * data1 data2 data3
  */
 ge::ComputeGraphPtr ShareGraph::BuildCustomOpGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("custom_op", "CustomOp"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("custom_op")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 2)->NODE("custom_op"));
  };
  auto graph = ToComputeGraph(g1);
  auto data0 = graph->FindNode("data0");
  SetNoStorage(data0->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data0->GetOpDesc(), "index", 0);
  data0->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data0->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 2);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto custom_op = graph->FindNode("custom_op");
  SetNoStorage(custom_op->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  custom_op->GetOpDesc()->SetOpKernelLibName("DNN_VM_CUSTOM_OP_STORE");
  custom_op->GetOpDesc()->SetOpEngineName("DNN_VM_CUSTOM");

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"custom_op"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  SetGraphOutShapeRange(graph);
  return graph;
}

 /*
  *  data0  data1  data2
  *     \    |      /
  *     \    |     /
  *       customop
  *          |
  *          |
  *       netoutput
  */
ge::ComputeGraphPtr ShareGraph::BuildOnlyCustomOpKnowShapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("custom_op", "CustomOp"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("custom_op")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 2)->NODE("custom_op"));
  };
  auto graph = ToComputeGraph(g1);
  auto data0 = graph->FindNode("data0");
  SetNoStorage(data0->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data0->GetOpDesc(), "index", 0);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);


  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 2);

  auto custom_op = graph->FindNode("custom_op");
  SetNoStorage(custom_op->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  custom_op->GetOpDesc()->SetOpKernelLibName("DNN_VM_CUSTOM_OP_STORE");
  custom_op->GetOpDesc()->SetOpEngineName("DNN_VM_CUSTOM");

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"custom_op"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetGraphOutShapeRange(graph);
  return graph;
}

/*
  *  data0  data1    data2  data3    data4  data5
  *     \    |         \     /          /    /
  *     \    |         \   /          /     /
  *         add0       add1          add2
  *              \       |         /
  *                \     |       /
  *                  customop           data6
  *                    |             /
  *                     |           /
  *                        add3
  *                        |
  *                      netoutput
  */
ge::ComputeGraphPtr ShareGraph::BuildCustomOpWithAddKnowShapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("add0", "Add")->EDGE(0, 0)->NODE("custom_op", "CustomOp"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("add0", "Add"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("add1", "Add")->EDGE(1, 1)->NODE("custom_op", "CustomOp"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 0)->NODE("add2", "Add")->EDGE(2, 2)->NODE("custom_op", "CustomOp")->EDGE(0, 0)->NODE("add3", "Add"));
    CHAIN(NODE("data5", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("data6", "Data")->EDGE(0, 1)->NODE("add3", "Add")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data0 = graph->FindNode("data0");
  SetNoStorage(data0->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data0->GetOpDesc(), "index", 0);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);


  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 2);

  auto data3 = graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 3);

  auto data4 = graph->FindNode("data4");
  SetNoStorage(data4->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 4);

  auto data5 = graph->FindNode("data5");
  SetNoStorage(data5->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data5->GetOpDesc(), "index", 5);

  auto data6 = graph->FindNode("data6");
  SetNoStorage(data6->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AttrUtils::SetInt(data6->GetOpDesc(), "index", 6);

  auto custom_op = graph->FindNode("custom_op");
  SetNoStorage(custom_op->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  custom_op->GetOpDesc()->SetOpKernelLibName("DNN_VM_CUSTOM_OP_STORE");
  custom_op->GetOpDesc()->SetOpEngineName("DNN_VM_CUSTOM");

  auto add0 = graph->FindNode("add0");
  AddCompileResult(add0, false);
  SetNoStorage(add0->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  add0->GetOpDesc()->SetOpKernelLibName("AiCoreLib");
  add0->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  add1->GetOpDesc()->SetOpKernelLibName("AiCoreLib");
  add1->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);

  auto add2 = graph->FindNode("add2");
  AddCompileResult(add2, false);
  SetNoStorage(add2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  add2->GetOpDesc()->SetOpKernelLibName("AiCoreLib");
  add2->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);

  auto add3 = graph->FindNode("add3");
  AddCompileResult(add3, false);
  SetNoStorage(add3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  add2->GetOpDesc()->SetOpKernelLibName("AiCoreLib");
  add2->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add3"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetGraphOutShapeRange(graph);
  return graph;
}

/*
 *  add2
 *  |  \
 *  |   add1
 *  |  /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::AicoreGraphTwoAdd() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add2", "Add"));
                  CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add1", "Add")->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  add1->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);

  auto add2 = graph->FindNode("add2");
  AddCompileResult(add2, false);
  SetNoStorage(add2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  add2->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  add2->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  SetGraphOutShapeRange(graph);
  return graph;
}

/*           add3
 *          /    \
 *        add2    \
 *       /    \    \
 *     add1    \    \
 *    /   \     \    \
 * data1 data2 data3 data4
 */
ComputeGraphPtr ShareGraph::AtcNanoGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->NODE("add2", "Add")
              ->NODE("add3", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 1)->NODE("add3", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto data3 = graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 1);

  auto data4 = graph->FindNode("data4");
  SetNoStorage(data4->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 1);

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  AttrUtils::SetBool(add1->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add1->GetOpDesc(), "globalworkspace_size", 32U);
  std::vector<int64_t> in_memory_type_list = {2, 2};
  std::vector<int64_t> out_memory_type_list = {2};
  (void)ge::AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type_list);
  (void)ge::AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type_list);
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);

  auto add2 = graph->FindNode("add2");
  AddCompileResult(add2, false);
  SetNoStorage(add2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  AttrUtils::SetBool(add2->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add2->GetOpDesc(), "globalworkspace_size", 32U);
  (void)ge::AttrUtils::SetListInt(add2->GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type_list);
  (void)ge::AttrUtils::SetListInt(add2->GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type_list);
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);

  auto add3 = graph->FindNode("add3");
  AddCompileResult(add3, false);
  SetNoStorage(add3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8});
  (void)ge::AttrUtils::SetListInt(add3->GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type_list);
  (void)ge::AttrUtils::SetListInt(add3->GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type_list);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add3"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*           add3
 *          /    \
 *        add2    \
 *       /    \    \
 *     add1    \    \
 *    /   \     \    \
 * bed   cat   dad   ear
 */
ge::Graph ShareGraph::MultiBatchGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("bed", "Data")
              ->NODE("add1", "Add")
              ->NODE("add2", "Add")
              ->NODE("add3", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("cat", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("dad", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("ear", "RefData")->EDGE(0, 1)->NODE("add3", "Add"));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto data1 = compute_graph->FindNode("bed");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {1,1,1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = compute_graph->FindNode("cat");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2,2,2});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto data3 = compute_graph->FindNode("dad");
  SetNoStorage(data3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {1,1,1});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);

  auto data4 = compute_graph->FindNode("ear");
  SetNoStorage(data4->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {1,1,1});
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data4->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  ge::GeTensorDesc data_tensor(GeShape({1,1,1}), FORMAT_ND, DT_FLOAT);
  data4->GetOpDesc()->AddInputDesc(data_tensor);

  auto add1 = compute_graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2,2,2});
  AttrUtils::SetBool(add1->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add1->GetOpDesc(), "globalworkspace_size", 32U);

  auto add2 = compute_graph->FindNode("add2");
  AddCompileResult(add2, false);
  SetNoStorage(add2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2,2,2});
  AttrUtils::SetBool(add2->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add2->GetOpDesc(), "globalworkspace_size", 32U);

  auto add3 = compute_graph->FindNode("add3");
  AddCompileResult(add3, false);
  SetNoStorage(add3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {2,2,2});

  auto net_output = compute_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add3"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *          add1
 *      __/ |  | \__
 *    /    /   \    \
 * data1 data2 data3 data4
 */
ComputeGraphPtr ShareGraph::AddWith4InputsAicoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 2)->NODE("add1", "Add"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 3)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto data3 = graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);

  auto data4 = graph->FindNode("data4");
  SetNoStorage(data4->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetBool(add1->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add1->GetOpDesc(), "globalworkspace_size", 32U);
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(add1->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(add1->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  AttrUtils::SetInt(add1->GetOpDesc(), "atomic_op_para_size", 1000);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  auto name = add1->GetName() + "_faked_atomic_kernel";
  auto atomic_bin = std::make_shared<ge::OpKernelBin>(
      name, std::vector<char>({'F', 'a', 'k', 'e', 'A', 't', 'o', 'm', 'i', 'c', 'B', 'i', 'n'}));
  add1->GetOpDesc()->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_bin);
  AttrUtils::SetStr(add1->GetOpDesc(), ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  AttrUtils::SetStr(add1->GetOpDesc(), ATOMIC_ATTR_TVM_METADATA, "FakeAtomicMeta");
  AttrUtils::SetStr(add1->GetOpDesc(), add1->GetName() + "_atomic_kernelname", "FakeAtomicKernelName");
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetGraphOutShapeRange(graph);
  return graph;
}

/*
 *                                 NetOutput
 *                   /               |      \
 *    cmo           /                |       \
 *       \         /                 |        \
 *        reducesum1                 |         \
 *        /       \                  |          \
 *     data1    data2       GetFloatStatus       GetFloatStatusv2
 *     /                         \                    \
 * ClearFloatStatusv2          data3                data4
 */
ComputeGraphPtr ShareGraph::AicoreWithRtsOverflowGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("ClearFloatStatusv2", "NPUClearFloatStatusV2")
              ->CTRL_EDGE()
              ->NODE("data1", "Data")
              ->NODE("reducesum1", "ReduceSum")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("reducesum1", "ReduceSum")->NODE("cmo_1", "Cmo"));
    CHAIN(NODE("ClearFloatStatus", "NPUClearFloatStatus")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data3", "Data")->NODE("GetFloatStatus", "NPUGetFloatStatus")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data4", "Data")->NODE("GetFloatStatusv2", "NPUGetFloatStatusV2")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_ND, DT_FLOAT, {});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_ND, DT_INT32, {0});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto data3 = graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);

  auto data4 = graph->FindNode("data4");
  SetNoStorage(data4->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);

  auto reducesum1 = graph->FindNode("reducesum1");
  AddCompileResult(reducesum1, false);
  reducesum1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reducesum1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {});
  reducesum1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto clear_float_status = graph->FindNode("ClearFloatStatus");
  SetNoStorage(clear_float_status->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  clear_float_status->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  clear_float_status->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto clear_float_statusv2 = graph->FindNode("ClearFloatStatusv2");
  clear_float_statusv2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto cmo = graph->FindNode("cmo_1");
  cmo->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);
  AttrUtils::SetInt(cmo->GetOpDesc(), "max_size", 1024);
  AttrUtils::SetInt(cmo->GetOpDesc(), "type", 6);

  auto get_float_status = graph->FindNode("GetFloatStatus");
  SetNoStorage(get_float_status->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  get_float_status->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  get_float_status->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto get_float_statusv2 = graph->FindNode("GetFloatStatusv2");
  SetNoStorage(get_float_statusv2->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  get_float_statusv2->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  get_float_statusv2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reducesum1", "ClearFloatStatus", "GetFloatStatus", "GetFloatStatusv2"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1, 2, 3});
  return graph;
}

ComputeGraphPtr ShareGraph::AicoreWithCmoGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("reducesum1", "ReduceSum")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("cmo", "Cmo"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("reducesum1", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_ND, DT_FLOAT, {32, 64});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_ND, DT_INT32, {0});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto reducesum1 = graph->FindNode("reducesum1");
  AddCompileResult(reducesum1, false);
  reducesum1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reducesum1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {32, 64});
  reducesum1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto cmo_node = graph->FindNode("cmo");
  SetNoStorage(cmo_node->GetOpDesc(), FORMAT_ND, DT_INT32, {32, 64});
  AttrUtils::SetInt(cmo_node->GetOpDesc(), "offset", 32);
  cmo_node->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  cmo_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reducesum1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *                             NetOutput
 *                   /               |      \
 *                  /                |       \
 *                 /                 |        \
 *        reducesum1                 |         \
 *        /       \                  |          \
 *     data1    data2     GetFloatDebugStatus       GetFloatStatusv2
 *     /                         \                    \
 * ClearFloatDebugStatus          data3                data4
 */
ComputeGraphPtr ShareGraph::AicoreWithRtsDebugOverflowGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("ClearFloatDebugStatus1", "NPUClearFloatDebugStatus")
              ->CTRL_EDGE()
              ->NODE("data1", "Data")
              ->NODE("reducesum1", "ReduceSum")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("reducesum1", "ReduceSum"));
    CHAIN(NODE("ClearFloatDebugStatus2", "NPUClearFloatDebugStatus")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data3", "Data")->NODE("GetFloatDebugStatus1", "NPUGetFloatDebugStatus")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data4", "Data")->NODE("GetFloatDebugStatus2", "NPUGetFloatDebugStatus")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_ND, DT_FLOAT, {});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_ND, DT_INT32, {0});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto data3 = graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);

  auto data4 = graph->FindNode("data4");
  SetNoStorage(data4->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);

  auto reducesum1 = graph->FindNode("reducesum1");
  AddCompileResult(reducesum1, false);
  reducesum1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reducesum1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {});
  reducesum1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto clear_float_status = graph->FindNode("ClearFloatDebugStatus2");
  SetNoStorage(clear_float_status->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  clear_float_status->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  clear_float_status->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto clear_float_statusv2 = graph->FindNode("ClearFloatDebugStatus1");
  clear_float_statusv2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto get_float_status = graph->FindNode("GetFloatDebugStatus1");
  SetNoStorage(get_float_status->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  get_float_status->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  get_float_status->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto get_float_statusv2 = graph->FindNode("GetFloatDebugStatus2");
  SetNoStorage(get_float_statusv2->GetOpDesc(), FORMAT_ND, DT_INT32, {8});
  get_float_statusv2->GetOpDesc()->SetOpEngineName(ge::kEngineNameRts);
  get_float_statusv2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reducesum1", "ClearFloatDebugStatus2", "GetFloatDebugStatus1", "GetFloatDebugStatus2"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1, 2, 3});
  return graph;
}

/*
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::FrameworkOPGraph(const string &real_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "FrameworkOp")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "FrameworkOP"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto add1 = graph->FindNode("add1");
  if (!ge::AttrUtils::SetStr(add1->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, real_node_type)) {
    GELOGE(ge::FAILED, "Set attr[%s] for Op[%s] failed.", ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
           add1->GetName().c_str());
    return nullptr;
  }
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reshape1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *     add1
 *    /  \
 * data1 var
 */
ComputeGraphPtr ShareGraph::VariableOPGraph(const string &real_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "FrameworkOp")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data"));
    CHAIN(NODE("var", "Variable")->EDGE(0, 1)->NODE("add1", "FrameworkOP"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto var = graph->FindNode("var");
  SetNoStorage(var->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {1, 2, 3, 4});
  var->GetOpDesc()->SetOutputOffset({137438953472U});
  ge::TensorUtils::SetSize(*var->GetOpDesc()->MutableOutputDesc(0), 512);
  auto add1 = graph->FindNode("add1");
  if (!ge::AttrUtils::SetStr(add1->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, real_node_type)) {
    GELOGE(ge::FAILED, "Set attr[%s] for Op[%s] failed.", ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
           add1->GetName().c_str());
    return nullptr;
  }
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reshape1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            NetOutput
 *                |
 *            reshape1
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::AicoreStaticGraph(bool is_with_atomic) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->EDGE(0, 1)
              ->NODE("reshape1", "Reshape")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reshape1", "Reshape"));
  };
  auto graph = ToComputeGraph(g1);
  SetGraphOutShapeRange(graph);
  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(add1, is_with_atomic);

  auto reshape1 = graph->FindNode("reshape1");
  reshape1->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape1->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);
  reshape1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reshape1"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return graph;
}
/*
 *     data1(stream:0)
 *       |
 *     shape
 * (stream:0)(send:1)   data2(stream:0)(send:0)
 *             \      /
 *              add (stream:1)(send:[2],recive:[0,1])
 *               |
 *            netoutput(stream:0)(recive:[2])
 */
ge::ComputeGraphPtr ShareGraph::MultiStreamWithHostMemAccessCrossStream(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 3;
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("shape", "Shape")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);
  // SetGraphOutShapeRange(graph);
  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  data1->GetOpDesc()->SetStreamId(0);

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetStreamId(0);
  AttrUtils::SetListInt(data2->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {0});
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {4});

  auto shape = graph->FindNode("shape");
  shape->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  shape->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  shape->GetOpDesc()->SetStreamId(0);
  AttrUtils::SetListInt(shape->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {1});

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  add1->GetOpDesc()->SetStreamId(1);
  AddCompileResult(add1, false);
  AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0, 1});
  AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {2});
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {4});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetStreamId(0);
  AttrUtils::SetListInt(net_output->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {2});
  graph->SetGraphUnknownFlag(true);
  return graph;
}

ge::ComputeGraphPtr ShareGraph::ReshapeAbnormalGraph() {
  auto data_0 = OP_DATA(0).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT64, {1, 1, 4, 4});

  std::vector<int64_t> const_data_0{8, 2};
  GeTensorDesc const_tensor_desc_0(GeShape(vector<int64_t>{2}), FORMAT_NCHW, DT_INT64);
  GeTensorPtr const_tensor_0 = std::make_shared<GeTensor>(
      const_tensor_desc_0, reinterpret_cast<uint8_t *>(const_data_0.data()), sizeof(int64_t) * const_data_0.size());
  auto constant_0 = OP_CFG(CONSTANT).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT64, {2}).Weight(const_tensor_0);
  auto reshape_0 = OP_CFG(RESHAPE).InCnt(2).OutCnt(1);
  auto abs_0 = OP_CFG("Abs").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT64, {8, 4});
  auto reshape_1 = OP_CFG(RESHAPE).InCnt(1).OutCnt(1);

  DEF_GRAPH(g) {
    CHAIN(NODE("data_0", data_0)->EDGE(0, 0)->NODE("reshape_0", reshape_0));
    CHAIN(NODE("constant_0", constant_0)->EDGE(0, 1)->NODE("reshape_0", reshape_0));

    CHAIN(NODE("reshape_0", reshape_0)->EDGE(0, 0)->NODE("abs_0", abs_0));
    CHAIN(NODE("abs_0", abs_0)->EDGE(0, 0)->NODE("reshape_1", reshape_1));
    CHAIN(NODE("reshape_1")->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g);
  auto reshape_0_node = compute_graph->FindNode("reshape_0");
  auto reshape_0_opdesc = reshape_0_node->GetOpDesc();
  reshape_0_opdesc->UpdateInputDesc(0, GeTensorDesc(GeShape({1, 1, 4, 4}), FORMAT_NCHW, DT_INT64));
  reshape_0_opdesc->UpdateInputDesc(1, GeTensorDesc(GeShape({2}), FORMAT_NCHW, DT_INT64));
  reshape_0_opdesc->UpdateOutputDesc(0, GeTensorDesc(GeShape({8, 4}), FORMAT_NCHW, DT_INT64));

  auto reshape_1_node = compute_graph->FindNode("reshape_1");
  auto reshape_1_opdesc = reshape_1_node->GetOpDesc();
  reshape_1_opdesc->UpdateInputDesc(0, GeTensorDesc(GeShape({8, 4}), FORMAT_NCHW, DT_INT64));
  reshape_1_opdesc->UpdateOutputDesc(0, GeTensorDesc(GeShape({16}), FORMAT_NCHW, DT_INT64));
  return compute_graph;
}

/*
 *             data1
 *                |
 *            shape
 *            /   \      data2
 *        relu     \      /
 *         \        add
 *          \       /
 *          netoutput
 */
ge::ComputeGraphPtr ShareGraph::ShapeToMultiAiCoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(
        NODE("data1", "Data")->NODE("shape", "Shape")->EDGE(0, 0)->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("shape", "Shape")->EDGE(0, 0)->NODE("relu", "Relu")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  // SetGraphOutShapeRange(graph);
  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  data1->GetOpDesc()->SetStreamId(0);

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {4});

  auto shape = graph->FindNode("shape");
  shape->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  shape->GetOpDesc()->AppendIrAttrName("dtype");
  shape->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  AttrUtils::SetInt(shape->GetOpDesc(), "dtype", static_cast<int>(DT_INT32));
  SetNoStorage(shape->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {4});

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add1, true);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {4});

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AttrUtils::SetStr(relu->GetOpDesc(), "_kernel_bin_id", "te_relu_12345");
  AddCompileResult(relu, false);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {4});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1", "relu"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  graph->SetGraphUnknownFlag(true);
  return graph;
}


/*
 *
 *                                                                       c
 *                                          +---> netoutput <--------+---------- batch_lengths(data1)
 *                                         /       \                  \
 *                             y_reshape(,0)        \                  \
 *                           /           \0          \                  \
 *                          /         transdata_17   transdata_15(,1)   transdata_13(,2)
 *                         /                     \        |            /
 *                        /                       lstmp_0_dynamicrnn_v3
 * <-------------------------------------------------------------------------------------+
 *                       /                      / \             \                            \                         \
 * \
 *                      /              transdata_4
 * \             \                            \                         \                        \
 *                     /                   | \             \                            \                         \ \
 *                    /                x_reshape                    \             \                         transdata_8
 * transdata_10              \
 *                   /                  / \                   \             \                            \ \ \
 * y_reshape_const(4,0) inputs_float32(data0,4) x_reshape_const(3,5) c2894_3(2,1) c2894_4(1,2)
 * hidden_state_float32(data2,6) cell_state_float32(data3,7)   c2894_5(0,3)
 */

ComputeGraphPtr ShareGraph::LstmpGraph() {
  DEF_GRAPH(g1) {
    auto drnnv3_cfg = OP_CFG("DynamicRNNV3")
                          .InNames({"x", "w", "b", "init_h", "init_c", "project"})
                          .OutNames({"y", "output_h", "output_c", "i", "j", "f", "o", "tanhc"});
    CHAIN(NODE("inputs_float32", "Data")->NODE("x_reshape", "Reshape"));
    CHAIN(NODE("x_reshape_const", "Const")
              ->EDGE(0, 1)
              ->NODE("x_reshape", "Reshape")
              ->NODE("transdata_4", "TransData")
              ->NODE("drnnv3", drnnv3_cfg));
    CHAIN(NODE("c2894_3", "Const")->EDGE(0, 1)->NODE("drnnv3", drnnv3_cfg));
    CHAIN(NODE("c2894_4", "Const")->EDGE(0, 2)->NODE("drnnv3", drnnv3_cfg));
    CHAIN(
        NODE("hidden_state_float32", "Data")->NODE("transdata_8", "TransData")->EDGE(0, 3)->NODE("drnnv3", drnnv3_cfg));
    CHAIN(
        NODE("cell_state_float32", "Data")->NODE("transdata_10", "TransData")->EDGE(0, 4)->NODE("drnnv3", drnnv3_cfg));
    CHAIN(NODE("c2894_5", "Const")->EDGE(0, 5)->NODE("drnnv3", drnnv3_cfg));
    CHAIN(NODE("drnnv3", drnnv3_cfg)
              ->EDGE(0, 0)
              ->NODE("transdata_17", "TransData")
              ->NODE("y_reshape", "Reshape")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("y_reshape_const", "Const")->EDGE(0, 1)->NODE("y_reshape", "Reshape"));
    CHAIN(NODE("drnnv3", drnnv3_cfg)
              ->EDGE(1, 0)
              ->NODE("transdata_15", "TransData")
              ->EDGE(0, 1)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("drnnv3", drnnv3_cfg)
              ->EDGE(2, 0)
              ->NODE("transdata_13", "TransData")
              ->EDGE(0, 2)
              ->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto c2894_5 = graph->FindNode("c2894_5");
  c2894_5->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  c2894_5->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({512, 512}));
  c2894_5->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({32, 32, 16, 16}));
  c2894_5->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND_RNN_BIAS);
  c2894_5->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  GeTensor tensor(c2894_5->GetOpDesc()->GetOutputDesc(0));
  tensor.SetData(std::vector<uint8_t>(512 * 512 * 2, 0));
  AttrUtils::SetTensor(c2894_5->GetOpDesc(), "value", tensor);

  auto c2894_4 = graph->FindNode("c2894_4");
  c2894_4->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  c2894_4->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({32, 32, 16, 16}));
  c2894_4->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({512, 512}));
  c2894_4->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  c2894_4->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  GeTensor c2894_4_tensor(c2894_4->GetOpDesc()->GetOutputDesc(0));
  c2894_4_tensor.SetData(std::vector<uint8_t>(32 * 32 * 16 * 16 * 2));
  AttrUtils::SetTensor(c2894_4->GetOpDesc(), "value", c2894_4_tensor);

  auto c2894_3 = graph->FindNode("c2894_3");
  c2894_3->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  c2894_3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({48, 128, 16, 16}));
  c2894_3->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({768, 2048}));
  c2894_3->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_FRACTAL_ZN_RNN);
  c2894_3->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  GeTensor c2894_3_tensor(c2894_3->GetOpDesc()->GetOutputDesc(0));
  c2894_3_tensor.SetData(std::vector<uint8_t>(48 * 128 * 16 * 16 * 2));
  AttrUtils::SetTensor(c2894_3->GetOpDesc(), "value", c2894_3_tensor);

  auto x_reshape_const = graph->FindNode("x_reshape_const");
  x_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
  x_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({3}));
  x_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({3}));
  x_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  x_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  TensorUtils::SetSize(*x_reshape_const->GetOpDesc()->MutableOutputDesc(0), 24);
  GeTensor x_reshape_const_tensor(x_reshape_const->GetOpDesc()->GetOutputDesc(0));
  std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
  x_reshape_const_tensor.SetData(reinterpret_cast<uint8_t *>(x_reshape_const_data.data()),
                                 x_reshape_const_data.size() * sizeof(int64_t));
  AttrUtils::SetTensor(x_reshape_const->GetOpDesc(), "value", x_reshape_const_tensor);

  auto y_reshape_const = graph->FindNode("y_reshape_const");
  y_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
  y_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({3}));
  y_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({3}));
  y_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  y_reshape_const->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  TensorUtils::SetSize(*y_reshape_const->GetOpDesc()->MutableOutputDesc(0), 24);
  GeTensor y_reshape_const_tensor(y_reshape_const->GetOpDesc()->GetOutputDesc(0));
  std::vector<int64_t> y_reshape_const_data({-1, 1, 512});
  y_reshape_const_tensor.SetData(reinterpret_cast<uint8_t *>(x_reshape_const_data.data()),
                                 x_reshape_const_data.size() * sizeof(int64_t));
  AttrUtils::SetTensor(y_reshape_const->GetOpDesc(), "value", y_reshape_const_tensor);

  auto inputs_float32 = graph->FindNode("inputs_float32");
  AttrUtils::SetInt(inputs_float32->GetOpDesc(), "index", 0);
  inputs_float32->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  inputs_float32->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, 256}));
  inputs_float32->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, 256}));
  inputs_float32->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  inputs_float32->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  inputs_float32->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto hidden_state_float32 = graph->FindNode("hidden_state_float32");
  AttrUtils::SetInt(hidden_state_float32->GetOpDesc(), "index", 1);
  hidden_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  hidden_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  hidden_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, 512}));
  hidden_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  hidden_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  hidden_state_float32->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto cell_state_float32 = graph->FindNode("cell_state_float32");
  AttrUtils::SetInt(cell_state_float32->GetOpDesc(), "index", 2);
  cell_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  cell_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  cell_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, 512}));
  cell_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  cell_state_float32->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  cell_state_float32->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto x_reshape = graph->FindNode("x_reshape");
  x_reshape->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  x_reshape->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 256}));
  x_reshape->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, 256}));
  x_reshape->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  x_reshape->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  x_reshape->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"shape", 1}};
  x_reshape->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  x_reshape->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);

  auto transdata_4 = graph->FindNode("transdata_4");
  transdata_4->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  transdata_4->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 256}));
  transdata_4->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 16, -1, 16, 16}));
  transdata_4->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  transdata_4->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  transdata_4->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(transdata_4, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(transdata_4->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(transdata_4->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(transdata_4->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  transdata_4->GetOpDesc()->MutableAllInputName() = {{"src", 0}};
  AttrUtils::SetBool(transdata_4->GetOpDesc(), "support_dynamicshape", true);
  transdata_4->GetOpDesc()->SetWorkspaceBytes({2});

  auto transdata_8 = graph->FindNode("transdata_8");
  transdata_8->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  transdata_8->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  transdata_8->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 32, -1, 16, 16}));
  transdata_8->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  transdata_8->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  transdata_8->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(transdata_8, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(transdata_8->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(transdata_8->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(transdata_8->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  transdata_8->GetOpDesc()->MutableAllInputName() = {{"src", 0}};
  AttrUtils::SetBool(transdata_8->GetOpDesc(), "support_dynamicshape", true);
  transdata_8->GetOpDesc()->SetWorkspaceBytes({2});

  auto transdata_10 = graph->FindNode("transdata_10");
  transdata_10->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  transdata_10->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  transdata_10->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 32, -1, 16, 16}));
  transdata_10->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  transdata_10->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  transdata_10->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(transdata_10, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(transdata_10->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(transdata_10->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(transdata_10->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  transdata_10->GetOpDesc()->MutableAllInputName() = {{"src", 0}};
  AttrUtils::SetBool(transdata_10->GetOpDesc(), "support_dynamicshape", true);
  transdata_10->GetOpDesc()->SetWorkspaceBytes({2});

  auto transdata_13 = graph->FindNode("transdata_13");
  transdata_13->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  transdata_13->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  transdata_13->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, 512}));
  transdata_13->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  transdata_13->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  transdata_13->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(transdata_13, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(transdata_13->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(transdata_13->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(transdata_13->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  transdata_13->GetOpDesc()->MutableAllInputName() = {{"src", 0}};
  AttrUtils::SetBool(transdata_13->GetOpDesc(), "support_dynamicshape", true);
  transdata_13->GetOpDesc()->SetWorkspaceBytes({2});

  auto transdata_15 = graph->FindNode("transdata_15");
  transdata_15->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  transdata_15->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  transdata_15->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, 512}));
  transdata_15->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  transdata_15->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  transdata_15->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(transdata_15, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(transdata_15->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(transdata_15->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(transdata_15->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  transdata_15->GetOpDesc()->MutableAllInputName() = {{"src", 0}};
  AttrUtils::SetBool(transdata_15->GetOpDesc(), "support_dynamicshape", true);
  transdata_15->GetOpDesc()->SetWorkspaceBytes({2});

  auto transdata_17 = graph->FindNode("transdata_17");
  transdata_17->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  transdata_17->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, -1, 512}));
  transdata_17->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, 512}));
  transdata_17->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  transdata_17->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  transdata_17->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(transdata_17, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AttrUtils::SetListInt(transdata_17->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(transdata_17->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(transdata_17->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  transdata_17->GetOpDesc()->MutableAllInputName() = {{"src", 0}};
  AttrUtils::SetBool(transdata_17->GetOpDesc(), "support_dynamicshape", true);
  transdata_17->GetOpDesc()->SetWorkspaceBytes({2});

  auto drnnv3 = graph->FindNode("drnnv3");
  drnnv3->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  for (size_t i = 0; i < drnnv3->GetAllOutDataAnchorsSize(); ++i) {
    drnnv3->GetOpDesc()->MutableOutputDesc(i)->SetDataType(ge::DT_FLOAT16);
    drnnv3->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({1, -1, 512}));
    drnnv3->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({1, 32, -1, 16, 16}));
    drnnv3->GetOpDesc()->MutableOutputDesc(i)->SetFormat(FORMAT_FRACTAL_NZ);
    drnnv3->GetOpDesc()->MutableOutputDesc(i)->SetOriginFormat(FORMAT_NCHW);
  }
  AddCompileResult(drnnv3, true);
  AttrUtils::SetListInt(drnnv3->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});
  AttrUtils::SetStr(drnnv3->GetOpDesc(), "_atomic_compile_info_key", "_atomic_compile_info_key");
  AttrUtils::SetStr(drnnv3->GetOpDesc(), "_atomic_compile_info_json",
                    "{\"vars\": {\"ub_size\": 131072, \"core_num\": 8, \"workspace_num\": 1}}");
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info;
  std::map<int64_t, int64_t> index_offset;
  index_offset[0] = 0;
  atomic_workspace_info[drnnv3->GetOpDesc()->GetName()] = index_offset;
  drnnv3->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  AttrUtils::SetBool(drnnv3->GetOpDesc(), "support_dynamicshape", true);
  drnnv3->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto y_reshape = graph->FindNode("y_reshape");
  y_reshape->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
  y_reshape->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, 512}));
  y_reshape->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, 256}));
  y_reshape->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  y_reshape->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  y_reshape->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"shape", 1}};
  y_reshape->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  y_reshape->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"transdata_17", "transdata_15", "transdata_13"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1, 2});
  SetGraphOutShapeRange(graph);
  for (const auto &node : graph->GetDirectNode()) {
    for (const auto &src_anchor : node->GetAllOutDataAnchors()) {
      auto src_desc = node->GetOpDesc()->GetOutputDescPtr(src_anchor->GetIdx());
      for (const auto &dst_anchor : src_anchor->GetPeerInDataAnchors()) {
        auto dst_node = dst_anchor->GetOwnerNode();
        auto dst_desc = dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx());
        dst_desc->SetDataType(src_desc->GetDataType());
        dst_desc->SetShape(src_desc->GetShape());
        dst_desc->SetOriginShape(src_desc->GetOriginShape());
        dst_desc->SetFormat(src_desc->GetFormat());
        dst_desc->SetOriginFormat(src_desc->GetOriginFormat());
      }
    }
  }

  return graph;
}

/*
 *
 *  netoutput
 *      |
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::BuildStringNodeGraph() {
  std::vector<int64_t> shape = {2, 2};
  auto data1 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
      .Build("data1");
  data1->SetOutputOffset({1024});
  auto data2 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
      .Attr(ge::ATTR_NAME_INDEX, (int32_t)1)
      .Build("data2");
  data2->SetOutputOffset({2048});
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto node_data1 = graph->FindNode("data1");
  AttrUtils::SetInt(node_data1->GetOpDesc(), "index", 0);
  SetNoStorage(node_data1->GetOpDesc(), ge::FORMAT_ND, DT_STRING, {-1, -1});
  SetShapeRangeNoStorage(node_data1->GetOpDesc(), {1, 1}, {-1, -1});
  node_data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto node_data2 = graph->FindNode("data2");
  AttrUtils::SetInt(node_data2->GetOpDesc(), "index", 1);
  SetNoStorage(node_data2->GetOpDesc(), ge::FORMAT_ND, DT_STRING, {-1, -1});
  SetShapeRangeNoStorage(node_data2->GetOpDesc(), {1, 1}, {-1, -1});
  node_data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_STRING, {-1, -1});
  add1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 1}, {-1, -1});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add1, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_STRING, {2, 3});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1, 1}, {-1, -1});
  noutput->GetOpDesc()->SetInputOffset({10});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *    netoutput
 *        |
 * constplaceholder1
 */
ComputeGraphPtr ShareGraph::BuildSingleConstPlaceHolderGraph(void *addr, size_t len) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("constplaceholder1", "ConstPlaceHolder")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto constplaceholder1 = graph->FindNode("constplaceholder1");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("origin_shape");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("origin_format");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("storage_shape");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("storage_format");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("expand_dim_rules");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("dtype");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("addr");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("size");
  constplaceholder1->GetOpDesc()->AppendIrAttrName("placement");
  auto constplaceholder_op_desc1 = constplaceholder1->GetOpDesc();

  // set attr
  std::vector<int64_t> shape_ori = {5, 5};
  std::initializer_list<int64_t> shape_new = {5, 5};
  ge::AttrUtils::SetListInt(constplaceholder_op_desc1, "origin_shape", shape_ori);
  ge::AttrUtils::SetListInt(constplaceholder_op_desc1, "storage_shape", shape_ori);
  DataType data_type = DT_FLOAT;
  ge::AttrUtils::SetDataType(constplaceholder_op_desc1, "dtype", data_type); // float
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "size", len);
  int64_t placement = 1L;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "placement", placement); // device
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "addr", reinterpret_cast<int64_t>(addr));

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape_new);
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"constplaceholder1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *  netoutput
 *      |
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::BuildSingleNodeGraph(const std::string &node_type,
                                                 std::vector<std::initializer_list<int64_t>> shape,
                                                 std::vector<std::initializer_list<int64_t>> min_shape,
                                                 std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", node_type)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[1], max_shape[1]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(add1->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AttrUtils::SetBool(add1->GetOpDesc(), "SmallShapeHostcpu", true);
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  AddCompileResult(add1, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *  netoutput
 *      |  \
 *      |  unsqueeze
 *      |  /
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::BuildAddUnSqueezeGraph(const std::string &node_type,
                                                 std::vector<std::initializer_list<int64_t>> shape,
                                                 std::vector<std::initializer_list<int64_t>> min_shape,
                                                 std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("add1", node_type)->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", node_type));
                  CHAIN(NODE("add1", node_type)->EDGE(0, 0)->NODE("unsqueeze", UNSQUEEZE)->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[1], max_shape[1]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(add1->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AttrUtils::SetBool(add1->GetOpDesc(), "SmallShapeHostcpu", true);
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  AddCompileResult(add1, false);

  auto unqueeze = graph->FindNode("unsqueeze");
  unqueeze->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameGeLocal);
  SetNoStorage(unqueeze->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(unqueeze->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(unqueeze->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(unqueeze, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add1", "unsqueeze"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *  netoutput
 *      | |
 *     add1
 *    /   \
 * data1   data2
 */
ComputeGraphPtr ShareGraph::BuildAddAsTwoOutputGraph(const std::string &node_type,
                                                   std::vector<std::initializer_list<int64_t>> shape,
                                                   std::vector<std::initializer_list<int64_t>> min_shape,
                                                   std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("add1", node_type)->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", node_type));
                  CHAIN(NODE("add1", node_type)->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[1], max_shape[1]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(add1->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AttrUtils::SetBool(add1->GetOpDesc(), "SmallShapeHostcpu", true);
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  AddCompileResult(add1, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add1", "add1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *  netoutput
 *     | |
 *  unsqueeze
 *      |
 *    data1
 */
ComputeGraphPtr ShareGraph::BuildUnsqueezeAsTwoOutputGraph(std::vector<std::initializer_list<int64_t>> shape,
                                                           std::vector<std::initializer_list<int64_t>> min_shape,
                                                           std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("unsqueeze", UNSQUEEZE)->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("unsqueeze", UNSQUEEZE)->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto unqueeze = graph->FindNode("unsqueeze");
  unqueeze->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameGeLocal);
  SetNoStorage(unqueeze->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(unqueeze->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(unqueeze->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(unqueeze, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"unsqueeze", "unsqueeze"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *  netoutput
 *     | |
 *    data1
 */
ComputeGraphPtr ShareGraph::BuildInputAsTwoOutputGraph(std::vector<std::initializer_list<int64_t>> shape,
                                                           std::vector<std::initializer_list<int64_t>> min_shape,
                                                           std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"data1", "data1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *  netoutput
 *     | |
 *     Assign
 *      |   \
 *    data1  data2
 */
ComputeGraphPtr ShareGraph::BuildAssignAsTwoOutputGraph(std::vector<std::initializer_list<int64_t>> shape,
                                                       std::vector<std::initializer_list<int64_t>> min_shape,
                                                       std::vector<std::initializer_list<int64_t>> max_shape) {

  DEF_GRAPH(g1) {
                  auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, shape[2])
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ge::ATTR_NAME_REFERENCE, true)
                      .Build("assign");
                  CHAIN(NODE("data1", "Data")->NODE(assign)->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("data2", "Data")->NODE(assign));
                  CHAIN(NODE(assign)->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[1], max_shape[1]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto assign = graph->FindNode("assign");
  AddCompileResult(assign, false);
  SetNoStorage(assign->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(assign->GetOpDesc(), min_shape[0], max_shape[2]);
  assign->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  assign->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  assign->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"assign", "assign"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *  netoutput
 *      | |
 *   unsqueeze
 *      |
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::BuildAddToUnSqueezeGraph(const std::string &node_type,
                                                   std::vector<std::initializer_list<int64_t>> shape,
                                                   std::vector<std::initializer_list<int64_t>> min_shape,
                                                   std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("add1", node_type));
                  CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", node_type));
                  CHAIN(NODE("add1", node_type)->NODE("unsqueeze", UNSQUEEZE)->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("unsqueeze", UNSQUEEZE)->EDGE(0,1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[1], max_shape[1]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(add1->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AttrUtils::SetBool(add1->GetOpDesc(), "SmallShapeHostcpu", true);
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  AddCompileResult(add1, false);

  auto unqueeze = graph->FindNode("unsqueeze");
  unqueeze->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameGeLocal);
  SetNoStorage(unqueeze->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(unqueeze->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(unqueeze->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(unqueeze, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"unsqueeze", "unsqueeze"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *
 *  netoutput
 *      |
 *     hcomreduce
 *    /
 * data1
 */
ComputeGraphPtr ShareGraph::BuildSingleHcclNodeGraph(const std::string &node_type,
                                                     std::vector<std::initializer_list<int64_t>> shape,
                                                     std::vector<std::initializer_list<int64_t>> min_shape,
                                                     std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", "Data")->NODE("hcom_reduce", node_type)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto hcom_reduce = graph->FindNode("hcom_reduce");
  hcom_reduce->GetOpDesc()->MutableAllInputName() = {{"x1", 0}};
  hcom_reduce->GetOpDesc()->SetOpKernelLibName("ops_kernel_info_hccl");
  SetNoStorage(hcom_reduce->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(hcom_reduce->GetOpDesc(), min_shape[1], max_shape[1]);
  AttrUtils::SetInt(hcom_reduce->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  int32_t root_id = 0;
  AttrUtils::SetInt(hcom_reduce->GetOpDesc(), HCOM_ATTR_ROOT_RANK, root_id);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[2], max_shape[2]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"hcom_reduce"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *    netoutput
 *       |
 *   hcomreduce
 *      /
 *     add1
 *    /  \
 * data1  data2
 */
ComputeGraphPtr ShareGraph::BuildTwoHcclNodeGraph(const std::string &node_type,
                                                     std::vector<std::initializer_list<int64_t>> shape,
                                                     std::vector<std::initializer_list<int64_t>> min_shape,
                                                     std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", node_type)->NODE("hcom_reduce", node_type)
        ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[0], max_shape[0]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(add1->GetOpDesc(), min_shape[1], max_shape[1]);
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AddCompileResult(add1, false);

  auto hcom_reduce = graph->FindNode("hcom_reduce");
  hcom_reduce->GetOpDesc()->MutableAllInputName() = {{"x1", 0}};
  hcom_reduce->GetOpDesc()->SetOpKernelLibName("ops_kernel_info_hccl");
  SetNoStorage(hcom_reduce->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(hcom_reduce->GetOpDesc(), min_shape[1], max_shape[1]);
  AttrUtils::SetInt(hcom_reduce->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  int32_t root_id = 0;
  AttrUtils::SetInt(hcom_reduce->GetOpDesc(), HCOM_ATTR_ROOT_RANK, root_id);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[2], max_shape[2]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"hcom_reduce"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *  count   seed   mean   stddev
 *       \   \      /     /
 *           randomnormal
 *                |
 *             netoutput
 */

ComputeGraphPtr ShareGraph::BuildDsaRandomNormalGraph(const std::string &node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("count", "Data")->EDGE(0, 0)->NODE("randomnormal", node_type)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("seed", "Data")->EDGE(0, 1)->NODE("randomnormal", node_type));
    CHAIN(NODE("mean", "Data")->EDGE(0, 2)->NODE("randomnormal", node_type));
    CHAIN(NODE("stddev", "Data")->EDGE(0, 3)->NODE("randomnormal", node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("count");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("seed");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 3, 4}, {1, 100, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("mean");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 0);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data3->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data4 = graph->FindNode("stddev");
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 1);
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1, 3, 4});
  SetShapeRangeNoStorage(data4->GetOpDesc(), {1, 1, 3, 4}, {1, 100, 3, 4});
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto randomnormal = graph->FindNode("randomnormal");
  randomnormal->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}, {"x3", 2}, {"x4", 3}};
  randomnormal->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameDsa);
  SetNoStorage(randomnormal->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});
  SetShapeRangeNoStorage(randomnormal->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});
  AttrUtils::SetInt(randomnormal->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(randomnormal, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"randomnormal"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            NetOutput
 *                |
 *              nonzero
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::BuildAiCoreThirdClassNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->EDGE(0, 1)
              ->NODE("nonzero", "NonZero")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("nonzero", "NonZero"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add1, false);

  auto nonzero = graph->FindNode("nonzero");
  nonzero->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  nonzero->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(nonzero->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(nonzero->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  AttrUtils::SetInt(nonzero->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  AttrUtils::SetStr(nonzero->GetOpDesc(), "_kernel_bin_id", "te_nonzero_12345");
  AddCompileResult(
      nonzero, false,
      "{\"_input_type\": [1,2,3], \"_exist_output_after_reduce\": true, \"_exist_workspace_after_reduce\": true, "
      "\"_available_ub_size\": {}, \"_common_info\": [1,2,3], \"_norm_vars\": {}}");

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"nonzero"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            NetOutput
 *                |
 *              add2
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::BuildTwoAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AddCompileResult(add1, false);
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(add2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add2, false);
  add2->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *                      NetOutput
 *                        |
 *                  fakse_node3
 *                  /        \
 *              fake_node2   data4
 *             /       \
 *       fake_node1     \
 *          /   \       data3
 *         /   data2
 *        /
 *      data1
 */
ge::ComputeGraphPtr ShareGraph::BuildFakeGetTensorNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("fake_node1", "FakeShapeRangeNode")
        ->NODE("fake_node2", "FakeAicoreNode")->NODE("fake_node3", "FakeShapeRangeNode")
        ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("fake_node1"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("fake_node2"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 1)->NODE("fake_node3"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data3->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data4 = graph->FindNode("data4");
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data4->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto fake_node1 = graph->FindNode("fake_node1");
  fake_node1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);
  SetNoStorage(fake_node1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  AddCompileResult(fake_node1, false);

  auto fake_node2 = graph->FindNode("fake_node2");
  fake_node2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(fake_node2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(fake_node2, false);

  auto fake_node3 = graph->FindNode("fake_node3");
  fake_node3->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node3->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);
  SetNoStorage(fake_node3->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node3->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node3->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  AddCompileResult(fake_node3, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"fake_node3"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *           NetOutput
 *             |
 *       fake_node1
 *          /   \
 *         /   data2
 *        /
 *      data1
 */
ge::ComputeGraphPtr ShareGraph::BuildFakeDeterministicNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("fake_node1", "FakeAicoreNodeWithDeterministc")
        ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("fake_node1"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto fake_node1 = graph->FindNode("fake_node1");
  fake_node1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(fake_node1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(fake_node1, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"fake_node1"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  (void)AttrUtils::SetInt(graph, ge::DETERMINISTIC, 1);
  (void)AttrUtils::SetInt(graph, "ge.deterministicLevel", 2);
  return graph;
}

/*
 *                ----- NetOutput
 *               |         |
 *               |   fakse_node3
 *               |   /        \
 *              fake_node2   data4
 *             /       \
 *       fake_node1     \
 *          /   \       data3
 *         /   data2
 *        /
 *      data1
 */
ge::ComputeGraphPtr ShareGraph::BuildFakeGetTensorNodeZeroCopyGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("fake_node1", "FakeShapeRangeNode")
        ->NODE("fake_node2", "FakeAicoreNode")->NODE("fake_node3", "FakeShapeRangeNode")
        ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("fake_node1"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("fake_node2")->EDGE(0, 1)->NODE("NetOutput"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 1)->NODE("fake_node3"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data3->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data4 = graph->FindNode("data4");
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data4->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto fake_node1 = graph->FindNode("fake_node1");
  fake_node1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);
  SetNoStorage(fake_node1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  AddCompileResult(fake_node1, false);

  auto fake_node2 = graph->FindNode("fake_node2");
  fake_node2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(fake_node2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(fake_node2, false);

  auto fake_node3 = graph->FindNode("fake_node3");
  fake_node3->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  fake_node3->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);
  SetNoStorage(fake_node3->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(fake_node3->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(fake_node3->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  AddCompileResult(fake_node3, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"fake_node2", "fake_node3"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *            NetOutput
 *                |
 *              add2
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticTwoAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT,{1, 1, 1, 1});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *  netoutput
 *     \
 *    add
 *   /   \
 * exp   Relu
 *   \   /
 *    Abs1
 *      |
 *    Data
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsReluExpAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs1", "Abs")
              ->NODE("exp", "Exp")
              ->NODE("add", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("abs1", "Abs")->EDGE(0, 0)->NODE("relu", "Relu")->EDGE(0, 1)->NODE("add", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(abs1, false);

  auto exp = graph->FindNode("exp");
  exp->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  exp->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  exp->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(exp->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(exp, false);

  auto add1 = graph->FindNode("add");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(add1, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(relu, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *  netoutput
 *   /   \
 * Exp   Relu
 *   \   /
 *    Abs
 *      |
 *    Data
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsExpReluNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("abs", "Abs")->NODE("exp", "Exp")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("abs", "Abs")->EDGE(0, 0)->NODE("relu", "Relu")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs = graph->FindNode("abs");
  abs->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(abs, false);

  auto exp = graph->FindNode("exp");
  exp->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  exp->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  exp->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(exp->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(exp, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(relu, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"exp", "relu"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/**
 *  netoutput
 *      \
 *      add2
 *   /     \
 *  Abs      add1
 *   |    /    \
 * Data1 Data2 Data3
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsTwoAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("abs", "Abs")->EDGE(0, 0)->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data3->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs = graph->FindNode("abs");
  abs->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(abs, false);

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *  netoutput
 *   /     \
 *  exp      add
 *     \    /    \
 *      Abs     data2
 *      |
 *     data1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsAddExpNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("abs", "Abs")->NODE("exp", "Exp")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("add", "Add")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("abs", "Abs")->EDGE(0, 1)->NODE("add", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs = graph->FindNode("abs");
  abs->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs, false);

  auto add = graph->FindNode("add");
  add->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add, false);

  auto exp = graph->FindNode("exp");
  exp->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  exp->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  exp->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(exp->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(exp, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"exp", "add"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/**
 *   netoutput
 *    /        \
 *  ReduceSum   Relu
 *   /   \       /
 * const1   Abs1
 *           |
 *         Data1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsReduceReluNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("abs1", "Abs")->NODE("reduceSum", "ReduceSum")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceSum", "ReduceSum"));
    CHAIN(NODE("abs1", "Abs")->EDGE(0, 0)->NODE("relu", "Relu")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {3});

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(abs1, false);

  auto reduceSum = graph->FindNode("reduceSum");
  reduceSum->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(reduceSum, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(relu, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reduceSum", "relu"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/**
 *                Netoutput
 *                    |
 *                   Add3
 *               /         \
 *      reduceSum1           reduceSum2
 *     /         \            /      \
 *  const1      Add1       Add2      const2
 *            /   \      /   \
 *  Data2(2,2,2)    Abs1     Data3(2,2,2)
 *                   |
 *               Data1(2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticTwoReduceThreeAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs1", "Abs")
              ->NODE("add1", "Add")
              ->NODE("reduceSum1", "ReduceSum")
              ->NODE("add3", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("abs1", "Abs")
              ->EDGE(0, 0)
              ->NODE("add2", "Add")
              ->NODE("reduceSum2", "ReduceSum")
              ->EDGE(0, 1)
              ->NODE("add3", "Add"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceSum1", "ReduceSum"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("reduceSum2", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data3->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(abs1, false);

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add2, false);

  auto add3 = graph->FindNode("add3");
  add3->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add3->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add3, false);

  auto reduceSum1 = graph->FindNode("reduceSum1");
  reduceSum1->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  for (size_t i = 0; i < reduceSum1->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceSum1->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2, 2}));
    reduceSum1->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2, 2}));
  }
  AddCompileResult(reduceSum1, false);

  auto reduceSum2 = graph->FindNode("reduceSum2");
  reduceSum2->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  for (size_t i = 0; i < reduceSum2->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceSum2->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2, 2}));
    reduceSum2->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2, 2}));
  }
  AddCompileResult(reduceSum2, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2});
  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const2"), {2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add3"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *                Netoutput
 *                    |
 *                   Add2
 *                  |      \
 *                 |       reduceSum2
 *                |        /      \
 *              Add1      /       const2
 *            /   \      /
 *  Data2(2)    Data1(2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAddReduceNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->EDGE(0, 0)
              ->NODE("add2", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reduceSum2", "ReduceSum")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("reduceSum2", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(add2, false);

  auto reduceSum2 = graph->FindNode("reduceSum2");
  reduceSum2->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2});
  for (size_t i = 0; i < reduceSum2->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceSum2->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2}));
    reduceSum2->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2}));
  }
  AddCompileResult(reduceSum2, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const2"), {1});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *               Netoutput
 *              |      \
 *            Add1       reduceSum2
 *            |   \     /      \
 * Data2(2,2,2)      Abs1      const2(1）
 *                  /
 *               Data1(2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsAddReduceNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->EDGE(0, 0)
              ->NODE("abs1", "Abs")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->EDGE(0, 0)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("abs1", "Abs")->EDGE(0, 0)->NODE("reduceSum2", "ReduceSum")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("reduceSum2", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(abs1, false);

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add1, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const2"), {1});

  auto reduceSum2 = graph->FindNode("reduceSum2");
  reduceSum2->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2});
  for (size_t i = 0; i < reduceSum2->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceSum2->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2}));
    reduceSum2->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2}));
  }
  AddCompileResult(reduceSum2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1", "reduceSum2"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/**
 *       netoutput
 *       /        \
 *  ReduceSum   ReduceMAx
 *   /   \       /   \
 * const1   Abs    const2
 *           |
 *         Data1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsTwoReduceNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("abs1", "Abs")->NODE("reduceSum", "ReduceSum")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("abs1", "Abs")->EDGE(0, 0)->NODE("reduceMax", "ReduceMax")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceSum", "ReduceSum"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("reduceMax", "ReduceMax"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2});
  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const2"), {2});

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs1, false);

  auto reduceSum = graph->FindNode("reduceSum");
  reduceSum->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AddCompileResult(reduceSum, false);

  auto reduceMax = graph->FindNode("reduceMax");
  reduceMax->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceMax->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceMax->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(reduceMax->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AddCompileResult(reduceMax, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reduceSum", "reduceMax"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/**
 *       netoutput
 *         /
 *    ReduceSum
 *     /   \
 *  const2 ReduceMax
 *           |     \
 *         Data1   const1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticTwoReduceNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("reduceMax", "ReduceMax")
              ->NODE("reduceSum", "ReduceSum")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceMax", "ReduceMax"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("reduceSum", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {0});
  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const2"), {0});

  auto reduceMax = graph->FindNode("reduceMax");
  reduceMax->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceMax->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceMax->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(reduceMax->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(reduceMax, false);

  auto reduceSum = graph->FindNode("reduceSum");
  reduceSum->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(reduceSum, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reduceSum"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *        /
 *       Relu
 *       |
 *      abs1
 *       \
 *      ReduceMax
 *         |     \
 *       abs0   const1
 *       |
 *       data1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticReduceAbsReluNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs0", "Abs")
              ->EDGE(0, 0)
              ->NODE("reduceMax", "ReduceMax")
              ->NODE("abs1", "Abs")
              ->NODE("relu", "Relu")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceMax", "ReduceMax"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs0 = graph->FindNode("abs0");
  abs0->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs0->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs0->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs0->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs0, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2});

  auto reduceMax = graph->FindNode("reduceMax");
  reduceMax->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceMax->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceMax->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(reduceMax->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  for (size_t i = 0; i < reduceMax->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceMax->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2, 2}));
    reduceMax->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2, 2}));
  }
  AddCompileResult(reduceMax, false);

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(abs1, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(relu, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *        /
 *       Relu
 *       |
 *      add1
 *     /     \
 *  Data2    ReduceMax
 *           |     \
 *       Data1   const1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticReduceAddReluNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("reduceMax", "ReduceMax")
              ->NODE("add1", "Add")
              ->NODE("relu", "Relu")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceMax", "ReduceMax"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2});

  auto reduceMax = graph->FindNode("reduceMax");
  reduceMax->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceMax->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceMax->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(reduceMax->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  for (size_t i = 0; i < reduceMax->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceMax->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2, 2}));
    reduceMax->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2, 2}));
  }
  AddCompileResult(reduceMax, false);

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add1, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(relu, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *        /
 *      reduceSum
 *     /     \
 *  const2    ReduceMax
 *            |     \
 *          abs1   const1
 *         /
 *       Data1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticTwoReduceReluNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs1", "Abs")
              ->EDGE(0, 0)
              ->NODE("reduceMax", "ReduceMax")
              ->EDGE(0, 0)
              ->NODE("reduceSum", "ReduceSum")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceMax", "ReduceMax"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("reduceSum", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs1, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2});
  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const2"), {1});

  auto reduceMax = graph->FindNode("reduceMax");
  reduceMax->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceMax->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceMax->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(reduceMax->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  for (size_t i = 0; i < reduceMax->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceMax->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2, 2}));
    reduceMax->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2, 2}));
  }
  AddCompileResult(reduceMax, false);

  auto reduceSum = graph->FindNode("reduceSum");
  reduceSum->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  for (size_t i = 0; i < reduceSum->GetOpDesc()->GetOutputsSize(); ++i) {
    reduceSum->GetOpDesc()->MutableOutputDesc(i)->SetShape(GeShape({2}));
    reduceSum->GetOpDesc()->MutableOutputDesc(i)->SetOriginShape(GeShape({2}));
  }
  AddCompileResult(reduceSum, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reduceSum"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *        /
 *       exp
 *        |
 *       abs2
 *       |
 *      relu
 *       \
 *      abs1
 *         |
 *       Data1
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsReluAbsExpNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs1", "Abs")
              ->NODE("relu", "Relu")
              ->NODE("abs2", "Abs")
              ->NODE("exp", "Exp")
              ->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs1, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(relu, false);

  auto abs2 = graph->FindNode("abs2");
  abs2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs2, false);

  auto exp = graph->FindNode("exp");
  exp->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  exp->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  exp->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(exp->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(exp, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"exp"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *        |
 *       abs2
 *        |
 *       add1
 *       |  \
 *      relu  data2(2,2,2,2)
 *       \
 *      abs1
 *         |
 *       Data1(2,2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsReluAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs1", "Abs")
              ->NODE("relu", "Relu")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->NODE("abs2", "Abs")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs1, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(relu, false);

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(add1, false);

  auto abs2 = graph->FindNode("abs2");
  abs2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(abs2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"abs2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *        |
 *       reduceSum
 *       |  \
 *      relu  const1(2)
 *       \
 *      abs1
 *         |
 *       Data1(2,2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticAbsReluReduceSumNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("abs1", "Abs")
              ->NODE("relu", "Relu")
              ->EDGE(0, 0)
              ->NODE("reduceSum", "ReduceSum")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceSum", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(abs1, false);

  auto relu = graph->FindNode("relu");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(relu, false);

  auto reduceSum = graph->FindNode("reduceSum");
  reduceSum->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2});
  AddCompileResult(reduceSum, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reduceSum"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *       |
 *      relu2
 *       |
 *      abs1
 *        |
 *       add1
 *       |    \
 *      relu1  data2(2,2,2,2)
 *        |
 *     Data1(2,2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticReluAddAbsReluNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("relu1", "Relu")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->NODE("abs1", "Abs")
              ->NODE("relu2", "Relu")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto relu = graph->FindNode("relu1");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(relu, false);

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(add1, false);

  auto abs1 = graph->FindNode("abs1");
  abs1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  abs1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  abs1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(abs1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(abs1, false);

  auto relu2 = graph->FindNode("relu2");
  relu2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu2->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(relu2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/**
 *     netoutput
 *       |
 *      reduceSum
 *        |     \
 *       add1    const1
 *       |    \
 *      relu1  data2(2,2,2,2)
 *        |
 *     Data1(2,2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticReluAddReduceSumNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("relu1", "Relu")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->NODE("reduceSum", "ReduceSum")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reduceSum", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto relu = graph->FindNode("relu1");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(relu, false);

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(add1, false);

  auto reduceSum = graph->FindNode("reduceSum");
  reduceSum->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axes", 1}};
  reduceSum->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  reduceSum->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reduceSum->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(reduceSum, false);

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {3});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reduceSum"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *    netoutput
 *       |
 *       add2
 *       |     \
 *      add1    Data3(2,2,2,2)
 *      |    \
 *     relu1  Data2(2,2,2)
 *       |
 *    Data1(2,2)
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticReluAddAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("relu1", "Relu")
              ->EDGE(0, 0)
              ->NODE("add1", "Add")
              ->EDGE(0, 0)
              ->NODE("add2", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data1->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto relu = graph->FindNode("relu1");
  relu->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  relu->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(relu->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2});
  AddCompileResult(relu, false);

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data2->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2});
  AddCompileResult(add1, false);

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 1);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  data3->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 2, 2, 2});
  AddCompileResult(add2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            NetOutput
 *                |
 *              add2
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::BuildThreeAddNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add3", "Add")->EDGE(0, 0)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add2, false);

  auto add3 = graph->FindNode("add3");
  add3->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add3->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 1, 1});
  AttrUtils::SetInt(add3->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add3, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            NetOutput
 *                |
 *              add2
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::BuildTwoAddNodeKnownShapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  add2->GetOpDesc()->SetWorkspaceBytes({1024});
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  AddCompileResult(add2, false);

  auto netoutput = graph->FindNode("NetOutput");
  SetNoStorage(netoutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            NetOutput
 *                |
 *              add2
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::Aicpu4thGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpu);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpu);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(add2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add2, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *  netoutput
 *      |
 *   reshape1
 *    /  \
 * data1 const1
 */
ComputeGraphPtr ShareGraph::BuildReshapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("reshape1", "Reshape")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("reshape1", "Reshape"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2, -1});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reshape1"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto reshape1 = graph->FindNode("reshape1");
  SetNoStorage(reshape1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, -1});
  reshape1->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape1->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);
  auto &name_index = reshape1->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["x"] = 0;
  name_index["shape"] = 1;
  graph->SetGraphUnknownFlag(true);

  return graph;
}

/*
 *
 *  netoutput
 *      |
 *   reshape1
 *    /  \
 * data1 fileconstant1
 */
ComputeGraphPtr ShareGraph::BuildReshapeGraph2() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("reshape1", "Reshape")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("fileconstant1", "FileConstant")->EDGE(0, 1)->NODE("reshape1", "Reshape"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reshape1"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto reshape1 = graph->FindNode("reshape1");
  SetNoStorage(reshape1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, -1});
  reshape1->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape1->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);
  auto &name_index = reshape1->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["x"] = 0;
  name_index["shape"] = 1;

  auto file_constant = graph->FindNode("fileconstant1");

  SetNoStorage(file_constant->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {5, 5});
  file_constant->GetOpDesc()->AppendIrAttrName("file_path");
  file_constant->GetOpDesc()->AppendIrAttrName("file_id");
  file_constant->GetOpDesc()->AppendIrAttrName("shape");
  file_constant->GetOpDesc()->AppendIrAttrName("dtype");

  // set attr
  std::vector<int64_t> shape = {5, 5};
  std::vector<int64_t> original_shape = {1, 5, 5};
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "offset", 0);
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "length", 0);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "location", "");
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_path", "test_weight.bin");
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_id", "");
  ge::AttrUtils::SetDataType(file_constant->GetOpDesc(), "dtype", DT_INT32);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "shape", shape);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "original_shape", original_shape);

  std::string file_name = "test_weight.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out0(file_name, std::ios::binary);
  if (!out0.is_open()) {
    return nullptr;
  }
  out0.write(reinterpret_cast<char*>(data), sizeof(data));
  out0.close();

  graph->SetGraphUnknownFlag(true);

  return graph;
}

/*
 *
 *  netoutput
 *      |
 *   gathershapes
 *    /  \
 * data1 const1
 */
ComputeGraphPtr ShareGraph::BuildGatherShapesGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("gathershapes", "GatherShapes")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("gathershapes", "GatherShapes"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});

  SetConstValue<int64_t, ge::DT_INT64>(graph->FindNode("const1"), {2, -1});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"gathershapes"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto goal_node = graph->FindNode("gathershapes");
  auto input_desc = goal_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(0));
  input_desc->SetShape(ge::GeShape({-1, -1, -1, -1}));
  input_desc = goal_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(1));
  input_desc->SetShape(ge::GeShape({2, -1}));

  goal_node->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  goal_node->GetOpDesc()->AppendIrAttrName("axes");
  goal_node->GetOpDesc()->AppendIrAttrName("dtype");
  auto &name_index = goal_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["x"] = 0;

  ge::AttrUtils::SetListListInt(goal_node->GetOpDesc(), "axes", {{1, 0}, {0, 2}});
  ge::AttrUtils::SetInt(goal_node->GetOpDesc(), "dtype", ge::DT_INT64);
  return graph;
}

/*
 *
 *  netoutput
 *      |
 *   gathershapes
 *    /  \
 * data1 const1
 */
ge::ComputeGraphPtr ShareGraph::BuildMultiBatchShapesGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("case", "Case")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("data2", "Data")->EDGE(0, 2)->NODE("case"));
      CHAIN(NODE("data3", "Data")->EDGE(0, 3)->NODE("case"));
      CHAIN(NODE("data4", "Data")->EDGE(0, 4)->NODE("case"));
      CTRL_CHAIN(NODE("data1")->NODE("shape_data", "DATA"));
      CTRL_CHAIN(NODE("data2")->NODE("shape_data"));
      CTRL_CHAIN(NODE("data3")->NODE("shape_data"));
      CHAIN(NODE("shape_data")->NODE("mapIndex", "MapIndex"));
      CHAIN(NODE("const", "Const")->EDGE(0, 1)->NODE("mapIndex")->EDGE(0, 0)->NODE("case"));
      CHAIN(NODE("case")->EDGE(1, 1)->NODE("NetOutput"));
      CHAIN(NODE("case")->EDGE(2, 2)->NODE("NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  auto data1 = main_graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  std::vector<int32_t> unknown_dim_index_1 = {0, 3};
  (void)ge::AttrUtils::SetListInt(data1->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index_1);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  auto data2 = main_graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  std::vector<int32_t> unknown_dim_index_2 = {0, 3};
  (void)ge::AttrUtils::SetListInt(data2->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index_2);
  if (!AttrUtils::SetInt(data2->GetOpDesc(), "index", 1)) {
    return nullptr;
  }
  auto data3 = main_graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  std::vector<int32_t> unknown_dim_index_3 = {3};
  (void)ge::AttrUtils::SetListInt(data3->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index_3);
  if (!AttrUtils::SetInt(data3->GetOpDesc(), "index", 2)) {
    return nullptr;
  }
  auto data4 = main_graph->FindNode("data4");
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  if (!AttrUtils::SetInt(data4->GetOpDesc(), "index", 3)) {
    return nullptr;
  }
  auto shape_data = main_graph->FindNode("shape_data");
  SetNoStorage(shape_data->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {5});
  std::vector<int32_t> unknown_shape_data_index = {0, 1, 2};
  (void)ge::AttrUtils::SetListInt(shape_data->GetOpDesc(), "_dynamic_batch_unknown_data_index", unknown_shape_data_index);
  if (!AttrUtils::SetInt(shape_data->GetOpDesc(), "index", 4)) {
    return nullptr;
  }
  (void)AttrUtils::SetStr(shape_data->GetOpDesc(), "_ge_attr_lowering_func", "multi_batch_shape_data");
  (void)AttrUtils::SetBool(shape_data->GetOpDesc(), "_is_multi_batch_shape_data", true);
  auto const_node = main_graph->FindNode("const");
  GeTensor weight;
  //std::vector<uint8_t> data = {2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<int32_t> data = {8, 4, 4, 0, 0, 0, 0, 0,
                               100, 100, 10, 0, 0, 0, 0, 0,
                               8, 4, 4, 0, 0, 0, 0, 0,
                               100, 100, 10, 0, 0, 0, 0, 0,
                               100, 100, 10, 0, 0, 0, 0, 0};

  weight.SetData((uint8_t *)data.data(), data.size() * sizeof(int32_t));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({40}));
  weight_desc.SetOriginShape(GeShape({40}));
  weight.SetTensorDesc(weight_desc);
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", weight);

  auto map_index = main_graph->FindNode("mapIndex");
  map_index->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu);
  AddCompileResult(map_index, false);
  map_index->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"data_seq", 1}};

  auto case_node = main_graph->FindNode("case");
  case_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(2)->SetDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(2)->SetOriginDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(3)->SetDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(3)->SetOriginDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(4)->SetDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(4)->SetOriginDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->AppendIrInput("branch_index", kIrInputRequired);
  case_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  (void)AttrUtils::SetBool(case_node->GetOpDesc(), ATTR_INSERT_BY_MBATCH, true);

  auto &name_index = case_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["branch_index"] = 0;
  name_index["input0"] = 1;
  name_index["input1"] = 2;
  name_index["input2"] = 3;
  name_index["input3"] = 4;

  auto graph_0 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1_batch_0", "Data")->NODE("shape_batch0", "Shape")->NODE("NetOutput_batch0", "NetOutput"));
      CHAIN(NODE("data2_batch_0", "Data")->NODE("shape1_batch0", "Shape")->EDGE(0, 1)->NODE("NetOutput_batch0"));
      CHAIN(NODE("data3_batch_0", "Data")->NODE("shape2_batch0", "Shape")->EDGE(0, 2)->NODE("NetOutput_batch0"));
    };
    return ToComputeGraph(g);
  }();
  graph_0->SetName("branch0");
  auto data1_batch0 =  graph_0->FindNode("data1_batch_0");
  SetNoStorage(data1_batch0->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  ge::AttrUtils::SetInt(data1_batch0->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data1_batch0->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto data2_batch0 =  graph_0->FindNode("data2_batch_0");
  SetNoStorage(data2_batch0->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  ge::AttrUtils::SetInt(data2_batch0->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data2_batch0->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);

  auto data3_batch0 =  graph_0->FindNode("data3_batch_0");
  SetNoStorage(data3_batch0->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  ge::AttrUtils::SetInt(data3_batch0->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data3_batch0->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 3);

  auto graph_1 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1_batch_1", "Data")->NODE("shape_batch1", "Shape")->NODE("NetOutput_batch1", "NetOutput"));
      CHAIN(NODE("data2_batch_1", "Data")->NODE("shape1_batch1", "Shape")->EDGE(0, 1)->NODE("NetOutput_batch1"));
      CHAIN(NODE("data3_batch_1", "Data")->NODE("shape2_batch1", "Shape")->EDGE(0, 2)->NODE("NetOutput_batch1"));
    };
    return ToComputeGraph(g);
  }();
  graph_1->SetName("branch1");
  auto data1_batch1 =  graph_1->FindNode("data1_batch_1");
  SetNoStorage(data1_batch1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  ge::AttrUtils::SetInt(data1_batch1->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data1_batch1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto data2_batch1 =  graph_1->FindNode("data2_batch_1");
  SetNoStorage(data2_batch1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  ge::AttrUtils::SetInt(data2_batch1->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data2_batch1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);

  auto data3_batch1 =  graph_1->FindNode("data3_batch_1");
  SetNoStorage(data3_batch1->GetOpDesc(), ge::FORMAT_NCHW, DT_FLOAT, {8, 3, 1, 100});
  ge::AttrUtils::SetInt(data3_batch1->GetOpDesc(), "index", 2);
  ge::AttrUtils::SetInt(data3_batch1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 3);

  graph_0->SetParentGraph(main_graph);
  graph_0->SetParentNode(case_node);
  graph_1->SetParentGraph(main_graph);
  graph_1->SetParentNode(case_node);

  main_graph->AddSubgraph(graph_0);
  main_graph->AddSubgraph(graph_1);
  case_node->GetOpDesc()->AddSubgraphName("branch0");
  case_node->GetOpDesc()->AddSubgraphName("branch1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "branch0");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "branch1");
  main_graph->TopologicalSorting();

  auto net_output_0 = graph_0->FindNode("NetOutput_batch0");
  net_output_0->GetOpDesc()->SetSrcName({"shape_batch0", "shape1_batch0", "shape2_batch0"});
  net_output_0->GetOpDesc()->SetSrcIndex({0, 1, 2});

  auto net_output_1 = graph_1->FindNode("NetOutput_batch1");
  net_output_1->GetOpDesc()->SetSrcName({"shape_batch1", "shape1_batch1", "shape2_batch1"});
  net_output_1->GetOpDesc()->SetSrcIndex({0, 1, 2});

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"case", "case", "case"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1, 2});

  main_graph->SetGraphUnknownFlag(true);
  main_graph->SetGraphID(0);
  (void)ge::AttrUtils::SetBool(main_graph, "_enable_dynamic_batch", true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
    subgraph->SetGraphID(0);
  }
  return main_graph;
}

/*
 *                    NetOutput
 *                        |
 *                     reshape
 *                     /      \
 * const3(dim) --> concat1     \
 *                 /   \        |
 *            gather  const2    |
 *           /   \              |
 *      shape    const1         |
 *        |                     |
 *      data1 ------------------+
 */
ge::ComputeGraphPtr ShareGraph::BuildShapeToReshapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("const3", "Const")
              ->NODE("concat1", "Concat")
              ->EDGE(0, 1)
              ->NODE("reshape1", "Reshape")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->NODE("shape1", "Shape")
              ->NODE("gather1", "Gather")
              ->EDGE(0, 1)
              ->NODE("concat1", "Concat"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("gather1", "Gather"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 2)->NODE("concat1", "Concat"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reshape1", "Reshape"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  auto const1 = graph->FindNode("const1");
  SetConstValue<int64_t, ge::DT_INT64>(const1, {0});
  const1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto const2 = graph->FindNode("const2");
  SetConstValue<int64_t, ge::DT_INT64>(const2, {-1});
  const2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto const3 = graph->FindNode("const3");
  SetConstValue<int64_t, ge::DT_INT64>(const3, {0});
  const3->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto shape1 = graph->FindNode("shape1");
  shape1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto reshape1 = graph->FindNode("reshape1");
  reshape1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  reshape1->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape1->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);

  auto gather1 = graph->FindNode("gather1");
  gather1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(gather1, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");

  auto concat1 = graph->FindNode("concat1");
  concat1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(concat1, true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reshape1"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return graph;
}

/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  | NetOutput |
 *       NetOutput          |   |       |  |   |       |
 *           |              |  Shape    |  |  Shape    |
 *          if  <---------> |   |       |  |   |       |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("shape", "Shape")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("shape", "Shape")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}
/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  | NetOutput |
 *       NetOutput          |   |       |  |   |       |
 *           |              |  Shape    |  |  Shape    |
 *          if  <---------> |   |       |  |   |       |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Const) +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfGraphWithConstInput() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Const")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetTensor(main_graph->FindNode("input")->GetOpDesc(), "value", CreateScalarGeTensor(0));


  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("shape", "Shape")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("shape", "Shape")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}
  /*
 *                          +-----------+
 *                          |Then Graph |
 *                          |           |  +-----------+
 *                          | NetOutput |  |Else Graph |
 *       Netoutput          |   |       |  |           |
 *           |              | squeeze   |  | NetOutput |
 *         cast             |    |      |  |   |       |
 *           |              |  cast     |  |  Shape    |
 *          if  <---------> |   |       |  |   |       |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfGraphRankChangedOneBranch() {

  auto main_graph = [&]() {
    auto input = OP_CFG(DATA)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {2,3,1,3})
                  .InCnt(1)
                  .OutCnt(1)
                  .Attr(ATTR_NAME_INDEX, 1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("input");
    auto pred = OP_CFG(DATA)
                      .TensorDesc(FORMAT_ND, DT_INT32, {})
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("pred");
    auto relu = OP_CFG(RELU)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {2,3,1,3})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("relu_main");
    DEF_GRAPH(g) {
      CHAIN(NODE(pred)->NODE("if", "If")->NODE(relu)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(input)->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  auto relu_main = main_graph->FindNode("relu_main");
  AddCompileResult(relu_main, false);
  relu_main->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  relu_main->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  std::vector<std::pair<NodePtr, int32_t>> main_outputs_info = {std::make_pair(relu_main, 0)};
  main_graph->SetGraphOutNodesInfo(main_outputs_info);

  auto then_graph = [&]() {
    auto relu = OP_CFG(RELU)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {2,3,1,3})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("relu_then");
    auto relu1 = OP_CFG(RELU)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {2,3,1,3})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("relu1_then");
    auto squeeze = OP_CFG(SQUEEZE)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {2,3,1,3})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("squeeze_then");
    DEF_GRAPH(g) {
      CHAIN(NODE("data_then_graph", "Data")->NODE(relu)->NODE(relu1)->NODE(squeeze)->NODE("NetOutput_then_graph", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto else_graph = [&]() {
    auto relu = OP_CFG(RELU)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {2,3,1,3})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("relu_else");
    DEF_GRAPH(g) {
      CHAIN(NODE("data_else_graph", "Data")->NODE(relu)->NODE("NetOutput_else_graph", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  auto relu_else = else_graph->FindNode("relu_else");
  AddCompileResult(relu_else, false);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();
  return main_graph;
}
/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  |           |
 *       NetOutput          |    :      |  |           |
 *           |              |  Clean    |  | NetOutput |
 *          if  <---------> |   |       |  |    :      |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfOneBranchGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
      CTRL_CHAIN(NODE("if", "If")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  int index = 0;
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
    }
  }

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("clean", "Clean"));
      CTRL_CHAIN(NODE("clean", "Clean")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CTRL_CHAIN(NODE("data", "Data")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  main_graph->TopologicalSorting();

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
 *                                          +---------------------------+   +------------------------------------+
 *                                          | Cond Graph                |   | Body Graph                         |
 *                    NetOutput             |      NetOutput            |   |      NetOutput  <------------+     |
 *                      |                   |         |                 |   |      /       \               |     |
 *                      +-------------------+        Foo                |   |    Bar       Add             |     |
 *                      |0,1,2              |      /     \              |   |    |        /   \            |     |
 *        +------>    while  <---------+    | input0   max_value(Const) |   |  input0 input1  one(Const) input2  |
 *        |             |              |    +---------------------------+   +------------------------------------+
 * input(Data)  loop_counter(Const)  max_iterations(Const)
 */
ComputeGraphPtr ShareGraph::WhileGraph(bool instance_name_as_graph_name) {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->EDGE(0, 0)->NODE("while", "While")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("loop_counter", "Constant")
                ->EDGE(0, 1)
                ->NODE("while", "While")
                ->EDGE(1, 1)
                ->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("max_iterations", "Constant")
                ->EDGE(0, 2)
                ->NODE("while", "While")
                ->EDGE(2, 2)
                ->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while", "while", "while"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 1, 2});
  }
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetTensor(main_graph->FindNode("loop_counter")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  ge::AttrUtils::SetTensor(main_graph->FindNode("max_iterations")->GetOpDesc(), "value", CreateScalarGeTensor(-1));

  auto cond_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->EDGE(0, 0)->NODE("foo", "Foo")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("max_value", "Constant")->EDGE(0, 1)->NODE("foo", "Foo"));
    };
    return ToComputeGraph(g);
  }();
  {
    auto netoutput = cond_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
  }

  cond_graph->SetName("cond_instance");
  ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetTensor(cond_graph->FindNode("max_value")->GetOpDesc(), "value", CreateScalarGeTensor(10));

  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input0", "Data")->EDGE(0, 0)->NODE("bar", "Bar")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));

      CHAIN(NODE("input1", "Data")->EDGE(0, 0)->NODE("add", "Add")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("one", "Constant")->EDGE(0, 1)->NODE("add", "Add"));

      CHAIN(NODE("input2", "Data")->EDGE(0, 2)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body_instance");
  {
    auto netoutput = body_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"bar", "add", "input2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 0, 0});
  }

  ge::AttrUtils::SetInt(body_graph->FindNode("input0")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("input0")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("input1")->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(body_graph->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(body_graph->FindNode("input2")->GetOpDesc(), "index", 2);
  ge::AttrUtils::SetInt(body_graph->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
  ge::AttrUtils::SetTensor(body_graph->FindNode("one")->GetOpDesc(), "value", CreateScalarGeTensor(1));

  auto while_node = main_graph->FindFirstNodeMatchType("While");
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(while_node);

  main_graph->AddSubgraph(cond_graph);
  main_graph->AddSubgraph(body_graph);
  if (instance_name_as_graph_name) {
    while_node->GetOpDesc()->AddSubgraphName(cond_graph->GetName());
    while_node->GetOpDesc()->AddSubgraphName(body_graph->GetName());
  } else {
    while_node->GetOpDesc()->AddSubgraphName("cond");
    while_node->GetOpDesc()->AddSubgraphName("body");
  }
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());

  main_graph->SetGraphUnknownFlag(true);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}

/*
 *
 * +----------------------------------------------------------------------------------------------------------------+
 * | Partitioncall2                          +---------------------------+   +------------------------------------+ |
 * |                                         | Cond Graph                |   | Body Graph                         | |
 * |                   NetOutput             |      NetOutput            |   |      NetOutput  <------------+     | |
 * |                     |                   |         |                 |   |      /       \               |     | |
 * |                     +-------------------+        Foo                |   |    Bar       Add             |     | |
 * |                     |0,1,2              |      /     \              |   |    |        /   \            |     | |
 * |       +------>    while  <---------+    | input0   max_value(Const) |   |  input0 input1  one(Const) input2  | |
 * |       |             |              |    +---------------------------+   +------------------------------------+ |
 * |input(Data)  loop_counter(Data)   max_iterations(Data)                                                          |
 * +----------------------------------------------------------------------------------------------------------------+
 *             |0                   |1      |2
 *             |                    |       |
 *         input(Data)              |0      |1
 *                +---------------------------------------------------------+
 *                |Partitioncall1                                           |
 *                |                    NetOutput<-----------+               |
 *                |                    |                    |               |
 *                |    loop_counter(Const)  max_iterations(Const)           |
 *                +---------------------------------------------------------+
 */
ComputeGraphPtr ShareGraph::WhileGraphInPartitionCall(bool instance_name_as_graph_name) {
  auto p1_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("loop_counter", "Constant")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("max_iterations", "Constant")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  p1_graph->SetName("p1");
  {
    auto netoutput = p1_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"loop_counter", "max_iterations"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 1});
    AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  }

  auto p2_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->EDGE(0, 0)->NODE("while", "While")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(
          NODE("loop_counter", "Data")->EDGE(0, 1)->NODE("while", "While")->EDGE(1, 1)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("max_iterations", "Data")
                ->EDGE(0, 2)
                ->NODE("while", "While")
                ->EDGE(2, 2)
                ->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  p2_graph->SetName("p2");
  {
    auto netoutput = p2_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while", "while", "while"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 1, 2});
    AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(2), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
  }
  ge::AttrUtils::SetInt(p2_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(p2_graph->FindNode("loop_counter")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(p2_graph->FindNode("max_iterations")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);

  auto cond_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->EDGE(0, 0)->NODE("foo", "Foo")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("max_value", "Constant")->EDGE(0, 1)->NODE("foo", "Foo"));
    };
    return ToComputeGraph(g);
  }();
  {
    auto netoutput = cond_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
  }

  cond_graph->SetName("cond_instance");
  ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetTensor(cond_graph->FindNode("max_value")->GetOpDesc(), "value", CreateScalarGeTensor(10));

  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input0", "Data")->EDGE(0, 0)->NODE("bar", "Bar")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));

      CHAIN(NODE("input1", "Data")->EDGE(0, 0)->NODE("add", "Add")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("one", "Constant")->EDGE(0, 1)->NODE("add", "Add"));

      CHAIN(NODE("input2", "Data")->EDGE(0, 2)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body_instance");
  {
    auto netoutput = body_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"bar", "add", "input2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 0, 0});
  }

  ge::AttrUtils::SetInt(body_graph->FindNode("input0")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("input0")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("input1")->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(body_graph->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(body_graph->FindNode("input2")->GetOpDesc(), "index", 2);
  ge::AttrUtils::SetInt(body_graph->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
  ge::AttrUtils::SetTensor(body_graph->FindNode("one")->GetOpDesc(), "value", CreateScalarGeTensor(1));

  auto while_node = p2_graph->FindFirstNodeMatchType("While");
  cond_graph->SetParentGraph(p2_graph);
  cond_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(p2_graph);
  body_graph->SetParentNode(while_node);

  if (instance_name_as_graph_name) {
    while_node->GetOpDesc()->AddSubgraphName(cond_graph->GetName());
    while_node->GetOpDesc()->AddSubgraphName(body_graph->GetName());
  } else {
    while_node->GetOpDesc()->AddSubgraphName("cond");
    while_node->GetOpDesc()->AddSubgraphName("body");
  }
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());

  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")
                ->EDGE(0, 0)
                ->NODE("partitioncall2", ge::PARTITIONEDCALL)
                ->EDGE(0, 0)
                ->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("partitioncall1", ge::PARTITIONEDCALL)
                ->EDGE(0, 1)
                ->NODE("partitioncall2", ge::PARTITIONEDCALL)
                ->EDGE(1, 1)
                ->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("partitioncall1", ge::PARTITIONEDCALL)
                ->EDGE(1, 2)
                ->NODE("partitioncall2", ge::PARTITIONEDCALL)
                ->EDGE(2, 2)
                ->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"partitioncall2", "partitioncall2", "partitioncall2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 1, 2});
    auto p1 = main_graph->FindNode("partitioncall1");
    p1->GetOpDesc()->AddSubgraphName(p1_graph->GetName());
    p1->GetOpDesc()->SetSubgraphInstanceName(0, p1_graph->GetName());
    p1_graph->SetParentGraph(main_graph);
    p1_graph->SetParentNode(p1);

    auto p2 = main_graph->FindNode("partitioncall2");
    p2->GetOpDesc()->AddSubgraphName(p2_graph->GetName());
    p2->GetOpDesc()->SetSubgraphInstanceName(0, p2_graph->GetName());
    p2_graph->SetParentGraph(main_graph);
    p2_graph->SetParentNode(p2);

    ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 0);
  }

  main_graph->AddSubGraph(p1_graph);
  main_graph->AddSubGraph(p2_graph);
  main_graph->AddSubgraph(cond_graph);
  main_graph->AddSubgraph(body_graph);

  main_graph->SetGraphUnknownFlag(true);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}

/*
 *             +-------------+   +-------------+
 *             | Cond Graph  |   | Body Graph  |
 *             | NetOutput   |   | NetOutput   |
 *             |    |        |   |    |        |
 * NetOutput   | LessThan_5  |   |  Add_1      |
 *   |         |   |         |   |   |         |
 * while  -----+ input       |   + input       |
 *   |         +-------------+   +-------------+
 * input
 */
ComputeGraphPtr ShareGraph::WhileGraph2(bool instance_name_as_graph_name) {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("while", "While")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 0);
  }

  auto cond_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("foo", "LessThan_5")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  {
    cond_graph->SetName("cond_instance");
    ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto netoutput = cond_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    netoutput->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{}), FORMAT_ND, DT_BOOL));
  }

  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("bar", "Add_1")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body_instance");
  {
    ge::AttrUtils::SetInt(body_graph->FindNode("input")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(body_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto netoutput = body_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"bar"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
  }

  auto while_node = main_graph->FindFirstNodeMatchType("While");
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(while_node);

  main_graph->AddSubgraph(cond_graph);
  main_graph->AddSubgraph(body_graph);
  if (instance_name_as_graph_name) {
    while_node->GetOpDesc()->AddSubgraphName(cond_graph->GetName());
    while_node->GetOpDesc()->AddSubgraphName(body_graph->GetName());
  } else {
    while_node->GetOpDesc()->AddSubgraphName("cond");
    while_node->GetOpDesc()->AddSubgraphName("body");
  }
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }
  return main_graph;
}

/*
 *             	+-------------+  +------------+
 * NetOutput   	| Cond Graph  |  | Body Graph |
 *   |  |      	| NetOutput   |  | NetOutput  |
 *   while  ----|    |        |  |   \  /     |
 *   |  |     	| LessThan_5  |  |	  \/      |
 * in1 in2      |   |         |  |    /\      |
 *              | in1         |  |   /  \     |
 *              +-------------+  | in1 in2    |
 *                               +------------+
 */
ComputeGraphPtr ShareGraph::WhileGraphXBody(bool instance_name_as_graph_name) {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->EDGE(0, 0)->NODE("while", "While")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input2", "Data")->EDGE(0, 1)->NODE("while", "While")->EDGE(1, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while", "while"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 1});
    ge::AttrUtils::SetInt(main_graph->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(main_graph->FindNode("input2")->GetOpDesc(), "index", 1);
  }

  auto cond_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("foo", "LessThan_5")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input2", "Data")->CTRL_EDGE()->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  {
    cond_graph->SetName("cond_instance");
    ge::AttrUtils::SetInt(cond_graph->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(cond_graph->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(cond_graph->FindNode("input2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(cond_graph->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto netoutput = cond_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    netoutput->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{}), FORMAT_ND, DT_BOOL));
  }

  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input2", "Data")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input1", "Data")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body_instance");
  {
    ge::AttrUtils::SetInt(body_graph->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(body_graph->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(body_graph->FindNode("input2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(body_graph->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto netoutput = body_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"input1", "input2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 0});
  }

  auto while_node = main_graph->FindFirstNodeMatchType("While");
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(while_node);

  main_graph->AddSubgraph(cond_graph);
  main_graph->AddSubgraph(body_graph);
  if (instance_name_as_graph_name) {
    while_node->GetOpDesc()->AddSubgraphName(cond_graph->GetName());
    while_node->GetOpDesc()->AddSubgraphName(body_graph->GetName());
  } else {
    while_node->GetOpDesc()->AddSubgraphName("cond");
    while_node->GetOpDesc()->AddSubgraphName("body");
  }
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }
  return main_graph;
}

/*
 *
 * NetOutput
 *   |  |      	+-------------+  +------------+
 *   while2 ----| Cond Graph  |  | Body Graph |
 *   |  |       | NetOutput   |  | NetOutput  |
 *   while1 ----|    |        |  |   \  /     |
 *   |  |     	| LessThan_x  |  |	  \/      |
 * in1 in2      |   |         |  |    /\      |
 *              | in1         |  |   /  \     |
 *              +-------------+  | in1 in2    |
 *                               +------------+
 */
ComputeGraphPtr ShareGraph::WhileGraphCascade(bool instance_name_as_graph_name) {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->EDGE(0, 0)->NODE("while1", "While")->EDGE(0, 0)->NODE("while2", "While")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input2", "Data")->EDGE(0, 1)->NODE("while1", "While")->EDGE(1, 1)->NODE("while2", "While")->EDGE(1, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while2", "while2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 1});
    ge::AttrUtils::SetInt(main_graph->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(main_graph->FindNode("input2")->GetOpDesc(), "index", 1);
  }

  auto cond_graph1 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("foo1", "LessThan_5")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input2", "Data")->CTRL_EDGE()->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  {
    cond_graph1->SetName("cond_instance1");
    ge::AttrUtils::SetInt(cond_graph1->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(cond_graph1->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(cond_graph1->FindNode("input2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(cond_graph1->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto netoutput = cond_graph1->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo1"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    netoutput->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{}), FORMAT_ND, DT_BOOL));
  }

  auto body_graph1 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input2", "Data")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input1", "Data")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph1->SetName("body_instance1");
  {
    ge::AttrUtils::SetInt(body_graph1->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(body_graph1->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(body_graph1->FindNode("input2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(body_graph1->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto netoutput = body_graph1->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"input1", "input2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 0});
  }

  auto cond_graph2 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("foo2", "LargerThan_1")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input2", "Data")->CTRL_EDGE()->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  {
    cond_graph2->SetName("cond_instance2");
    ge::AttrUtils::SetInt(cond_graph2->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(cond_graph2->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(cond_graph2->FindNode("input2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(cond_graph2->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto netoutput = cond_graph2->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo2"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    netoutput->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{}), FORMAT_ND, DT_BOOL));
  }

  auto body_graph2 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input2", "Data")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input1", "Data")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph2->SetName("body_instance2");
  {
    ge::AttrUtils::SetInt(body_graph2->FindNode("input1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(body_graph2->FindNode("input1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(body_graph2->FindNode("input2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(body_graph2->FindNode("input2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto netoutput = body_graph2->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"input1", "input2"});
    netoutput->GetOpDesc()->SetSrcIndex({0, 0});
  }

  auto while_node1 = main_graph->FindNode("while1");
  cond_graph1->SetParentGraph(main_graph);
  cond_graph1->SetParentNode(while_node1);
  body_graph1->SetParentGraph(main_graph);
  body_graph1->SetParentNode(while_node1);
  auto while_node2 = main_graph->FindNode("while2");
  cond_graph2->SetParentGraph(main_graph);
  cond_graph2->SetParentNode(while_node2);
  body_graph2->SetParentGraph(main_graph);
  body_graph2->SetParentNode(while_node2);

  main_graph->AddSubgraph(cond_graph1);
  main_graph->AddSubgraph(body_graph1);
  main_graph->AddSubgraph(cond_graph2);
  main_graph->AddSubgraph(body_graph2);
  if (instance_name_as_graph_name) {
    while_node1->GetOpDesc()->AddSubgraphName(cond_graph1->GetName());
    while_node1->GetOpDesc()->AddSubgraphName(body_graph1->GetName());
    while_node2->GetOpDesc()->AddSubgraphName(cond_graph2->GetName());
    while_node2->GetOpDesc()->AddSubgraphName(body_graph2->GetName());
  } else {
    while_node1->GetOpDesc()->AddSubgraphName("cond1");
    while_node1->GetOpDesc()->AddSubgraphName("body1");
    while_node2->GetOpDesc()->AddSubgraphName("cond2");
    while_node2->GetOpDesc()->AddSubgraphName("body2");
  }
  while_node1->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph1->GetName());
  while_node1->GetOpDesc()->SetSubgraphInstanceName(1, body_graph1->GetName());
  while_node2->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph2->GetName());
  while_node2->GetOpDesc()->SetSubgraphInstanceName(1, body_graph2->GetName());

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }
  return main_graph;
}

/*
 *             +--------------------+   +--------------------+
 *             | Cond Graph         |   | Body Graph         |
 *             |  LabelSet_0        |   |  LabelSet_1        |
 *             |    |               |   |    |               |
 * input       | StreamActive       |   | StreamActive       |
 *   |         |    |               |   |    |               |
 * while-------+  input             |   +  input             |
 *   |         |    |               |   |    |               |
 * NetOutput   | LessThan_5         |   |  Add_1             |
 *             |    |               |   |    |               |
 *             | LabelSwitchByIndex |   | LabelGotoEx        |
 *             +--------------------+   |   |                |
 *                                      | LabelSet_2         |
 *                                      |   |                |
 *                                      | NetOutput          |
 *                                      +--------------------+
 */
ComputeGraphPtr ShareGraph::WhileGraph3(bool instance_name_as_graph_name) {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("While", "While")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 0);
  }

  auto cond_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("LabelSet_0", "LabelSet")->NODE("Stream_0", "StreamActive")->NODE("input", "Data")
            ->NODE("LessThan_5", "Conv2D")->NODE("SwitchByIndex", "LabelSwitchByIndex"));
    };
    return ToComputeGraph(g);
  }();
  {
    cond_graph->SetName("cond_instance");
    ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(cond_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetListInt(cond_graph->FindNode("SwitchByIndex")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_LIST, {1, 2});
    ge::AttrUtils::SetInt(cond_graph->FindNode("LabelSet_0")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 0);
  }

  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("LabelSet_1", "LabelSet")->NODE("Stream_1", "StreamActive")->NODE("input", "Data")
            ->NODE("add", "Add_1")->NODE("LabelGoto", "LabelGotoEx")->NODE("LabelSet_2", "LabelSet")
            ->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body_instance");
  {
    ge::AttrUtils::SetInt(body_graph->FindNode("input")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(body_graph->FindNode("input")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(body_graph->FindNode("LabelGoto")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 0);
    ge::AttrUtils::SetInt(body_graph->FindNode("LabelSet_1")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 1);
    ge::AttrUtils::SetInt(body_graph->FindNode("LabelSet_2")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 2);
  }

  auto while_node = main_graph->FindFirstNodeMatchType("While");
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(while_node);

  main_graph->AddSubgraph(cond_graph);
  main_graph->AddSubgraph(body_graph);

  if (instance_name_as_graph_name) {
    while_node->GetOpDesc()->AddSubgraphName(cond_graph->GetName());
    while_node->GetOpDesc()->AddSubgraphName(body_graph->GetName());
  } else {
    while_node->GetOpDesc()->AddSubgraphName("cond");
    while_node->GetOpDesc()->AddSubgraphName("body");
  }
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }
  return main_graph;
}

/*
 *                          +---------------+  +--------------+
 *                          |Then Graph     |  |Else Graph    |
 * pred(Data)  input(Data)  |               |  |              |
 *         \   /            | SwitchByIndex |  |  LabelSet_1  |
 *          if  <---------> |     |         |  |     |        |
 *           |              |  LabelSet_0   |  | StreamActive |
 *       NetOutput          |     |         |  |     |        |
 *                          | StreamActive  |  |   Data(1)    |
 *                          |     |         |  |     |        |
 *                          |   Data(1)     |  |   Shape      |
 *                          |     |         |  |     |        |
 *                          |   Shape       |  |  LabelSet_2  |
 *                          |     |         |  +--------------+
 *                          | LabelGotoEx   |
 *                          +---------------+
 */
ComputeGraphPtr ShareGraph::IfGraphWithSwitch() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("SwitchByIndex", "LabelSwitchByIndex")
            ->NODE("LabelSet_0", "LabelSet")->NODE("Stream_0", "StreamActive")
            ->NODE("shape", "Shape")->NODE("LabelGoto", "LabelGotoEx"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetListInt(then_graph->FindNode("SwitchByIndex")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_LIST, {0, 1});
  ge::AttrUtils::SetInt(then_graph->FindNode("LabelSet_0")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 0);
  ge::AttrUtils::SetInt(then_graph->FindNode("LabelGoto")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 2);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("LabelSet_1", "LabelSet")
            ->NODE("Stream_1", "StreamActive")->NODE("shape", "Shape")
            ->NODE("LabelSet_2", "LabelSet"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(else_graph->FindNode("LabelSet_1")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 1);
  ge::AttrUtils::SetInt(else_graph->FindNode("LabelSet_2")->GetOpDesc(), ge::ATTR_NAME_LABEL_SWITCH_INDEX, 2);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}

/*
 *
 *      Data  CONSTANT  Data
 *         \   /        /
 *       FillWindowCache
 *            |
 *          Conv
 *            |
 *        NetOutput
 *
 */
ComputeGraphPtr ShareGraph::GraphWithFifoWindowCache() {
  std::vector<int64_t> shape = {2, 2};
  auto data_0 = OP_CFG(DATA)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("data_0");
  data_0->SetOutputOffset({0});
  AttrUtils::SetListInt(data_0, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
  TensorUtils::SetSize(*data_0->MutableOutputDesc(0), 8);

  auto data_1 = OP_CFG(CONSTANT)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("data_1");
  data_1->SetOutputOffset({0});
  AttrUtils::SetListInt(data_1, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
  TensorUtils::SetSize(*data_1->MutableOutputDesc(0), 8);
  ge::AttrUtils::SetInt(*data_1->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);

  auto data_2 = OP_CFG(DATA)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("data_2");
  data_2->SetOutputOffset({8});
  AttrUtils::SetListInt(data_2, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
  TensorUtils::SetSize(*data_2->MutableOutputDesc(0), 8);

  auto fifo = OP_CFG("FillWindowCache")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(3)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_REFERENCE, true)
                    .Build("fillwindowcache");
  vector<bool> is_input_const_vec = {false, true, false};
  fifo->SetIsInputConst(is_input_const_vec);
  fifo->SetInputOffset({0,1024,8});
  fifo->SetOutputOffset({1024});
  AttrUtils::SetListInt(fifo, ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HBM, RT_MEMORY_L1, RT_MEMORY_L1});
  AttrUtils::SetListInt(fifo, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
  auto output_tensordesc = fifo->MutableOutputDesc(0);
  ge::AttrUtils::SetInt(*fifo->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);
  ge::AttrUtils::SetInt(*fifo->MutableInputDesc(1), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);
  TensorUtils::SetReuseInput(*output_tensordesc, true);
  TensorUtils::SetReuseInputIndex(*output_tensordesc, 1);

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(fifo)->EDGE(0, 0)->
    NODE("conv", CONV2D)->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data_1)->EDGE(0, 1)->NODE("fillwindowcache"));
    CHAIN(NODE(data_2)->EDGE(0, 2)->NODE("fillwindowcache"));
  };
  auto graph = ToComputeGraph(g1);
  auto netoutput = graph->FindNode("NetOutput");
  netoutput->GetOpDesc()->SetSrcName({"conv"});
  netoutput->GetOpDesc()->SetSrcIndex({0});

  return graph;
}
ComputeGraphPtr ShareGraph::MatmulOmBinaryGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("x1", "Data")->NODE("matmul", "MatMulV2")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("x2", "Data")->NODE("matmul", "MatMulV2"));
      CHAIN(NODE("bias", "Data")->NODE("matmul", "MatMulV2"));
      CHAIN(NODE("offset_w", "Data")->NODE("matmul", "MatMulV2"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"matmul"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    ge::AttrUtils::SetInt(main_graph->FindNode("x1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(main_graph->FindNode("x2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(main_graph->FindNode("bias")->GetOpDesc(), "index", 2);
    ge::AttrUtils::SetInt(main_graph->FindNode("offset_w")->GetOpDesc(), "index", 3);
  }

  auto matmul = main_graph->FindFirstNodeMatchType("MatMulV2");
  matmul->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({UNKNOWN_DIM}));
  auto &ir_name_to_index = matmul->GetOpDesc()->MutableAllInputName();
  ir_name_to_index.clear();
  ir_name_to_index["x1"] = 0;
  ir_name_to_index["x2"] = 1;
  ir_name_to_index["bias"] = 2;
  ir_name_to_index["offset_w"] = 3;

  return main_graph;
}

ComputeGraphPtr ShareGraph::MatmulV2Graph(bool with_bias, bool with_offset) {
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("x1", "Data")->NODE("matmul", "MatMulV2")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("x2", "Data")->NODE("matmul", "MatMulV2"));
      if (with_bias) {
        CHAIN(NODE("bias", "Data")->NODE("matmul", "MatMulV2"));
      }
      if (with_offset) {
        CHAIN(NODE("offset_w", "Data")->NODE("matmul", "MatMulV2"));
      }
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("origin");
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"matmul"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
    ge::AttrUtils::SetInt(main_graph->FindNode("x1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(main_graph->FindNode("x2")->GetOpDesc(), "index", 1);
    if (with_bias) {
      ge::AttrUtils::SetInt(main_graph->FindNode("bias")->GetOpDesc(), "index", 2);
    }
    if (with_offset) {
      ge::AttrUtils::SetInt(main_graph->FindNode("offset_w")->GetOpDesc(), "index", (with_bias ? 3 : 2));
    }
  }

  auto matmul = main_graph->FindFirstNodeMatchType("MatMulV2");
  auto &ir_name_to_index = matmul->GetOpDesc()->MutableAllInputName();
  ir_name_to_index.clear();
  ir_name_to_index["x1"] = 0;
  ir_name_to_index["x2"] = 1;
  if (with_bias) {
    ir_name_to_index["bias"] = 2;
  }
  if (with_offset) {
    ir_name_to_index["offset_w"] = (with_bias ? 3 : 2);
  }

  return main_graph;
}

/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  | NetOutput |
 *       NetOutput          |   |       |  |   |       |
 *           |              |  Shape    |  |  Rank     |
 *          if  <---------> |   |       |  |   |       |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfGraph2() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);

  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  auto if_node = main_graph->FindNode("if");
  if_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  if_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT64);

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("shape", "Shape")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto shape_node = then_graph->FindFirstNodeMatchType("Shape");
  shape_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
  shape_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT64);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("rank", "Rank")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto rank_node = else_graph->FindFirstNodeMatchType("Rank");
  rank_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
  rank_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT64);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  then_graph->SetGraphUnknownFlag(true);
  else_graph->SetGraphUnknownFlag(true);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}
/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *       NetOutput          |           |  |           |
 *           |              |           |  | NetOutput |
 *          Add <---+       |           |  |   |       |
 *           |      |       | NetOutput |  |  Add      |
 *          if  <---|-----> |   |       |  |  ||       |
 *        /    \    |       | Const     |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfGraph3() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("add0", "Add")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("add0", "Add"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);

  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  auto if_node = main_graph->FindNode("if");
  if_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  if_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  auto add_node = main_graph->FindFirstNodeMatchType("Add");
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->AppendIrInput("x1", kIrInputRequired);
  add_node->GetOpDesc()->AppendIrInput("x2", kIrInputRequired);
  add_node->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-2}));
  AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add_node, false);

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("c0", "Const")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");

  auto c0 = then_graph->FindNode("c0");
  c0->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  c0->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1}));
  c0->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1}));
  c0->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  c0->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  GeTensor tensor(c0->GetOpDesc()->GetOutputDesc(0));
  tensor.SetData(std::vector<uint8_t>(4, 0));
  AttrUtils::SetTensor(c0->GetOpDesc(), "value", tensor);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("data", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  add_node = else_graph->FindFirstNodeMatchType("Add");
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_ND);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-2}));
  add_node->GetOpDesc()->AppendIrInput("x1", kIrInputRequired);
  add_node->GetOpDesc()->AppendIrInput("x2", kIrInputRequired);
  add_node->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add_node, false);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  then_graph->SetGraphUnknownFlag(true);
  else_graph->SetGraphUnknownFlag(true);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetGraphOutShapeRange(then_graph);
  SetGraphOutShapeRange(else_graph);
  return main_graph;
}
/*
 *                          +----------  -+  +-----------+
 *                          |Then Graph   |  |Else Graph |
 *       NetOutput          |             |  |           |
 *           |              | NetOutput   |  | NetOutput |
 *          add2 ----data2  |    |        |  |   |       |
 *           |              |  add3 ---+  |  |  add4     |
 *          if  <-------->  |   |      |  |  |  ||       |
 *        /    \            |Const Data(1)|  | Data(1)   |
 * pred(Data)   add1        +-------------+  +-----------+
 *             /    \
 *          add0   data1
 *           ||
 *          data0
 */
ComputeGraphPtr ShareGraph::IfGraph4() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input0", "Data")->NODE("add0", "Add")->NODE("add1", "Add")->EDGE(0, 1)->NODE("if", "If"));
      CHAIN(NODE("input0", "Data")->EDGE(0, 1)->NODE("add0", "Add"));
      CHAIN(NODE("input1", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
      CHAIN(NODE("input2", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);

  ge::AttrUtils::SetInt(main_graph->FindNode("input0")->GetOpDesc(), "index", 1);
  main_graph->FindNode("input0")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input0")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  ge::AttrUtils::SetInt(main_graph->FindNode("input1")->GetOpDesc(), "index", 2);
  main_graph->FindNode("input1")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input1")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  ge::AttrUtils::SetInt(main_graph->FindNode("input2")->GetOpDesc(), "index", 3);
  main_graph->FindNode("input2")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input2")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  auto if_node = main_graph->FindNode("if");
  if_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  if_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input"] = 1;
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  std::vector<NodePtr> add_nodes;
  auto add_node = main_graph->FindNode("add0");
  add_nodes.emplace_back(add_node);
  add_node = main_graph->FindNode("add1");
  add_nodes.emplace_back(add_node);
  add_node = main_graph->FindNode("add2");
  add_nodes.emplace_back(add_node);

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("c0", "Const")->NODE("add3", "Add")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("data", "Data")->EDGE(0, 1)->NODE("add3", "Add"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  add_node = then_graph->FindNode("add3");
  add_nodes.emplace_back(add_node);

  auto c0 = then_graph->FindNode("c0");
  c0->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  c0->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1}));
  c0->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1}));
  c0->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  c0->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  GeTensor tensor(c0->GetOpDesc()->GetOutputDesc(0));
  tensor.SetData(std::vector<uint8_t>(4, 0));
  AttrUtils::SetTensor(c0->GetOpDesc(), "value", tensor);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("add4", "Add")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("data", "Data")->EDGE(0, 1)->NODE("add4", "Add"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  add_node = else_graph->FindNode("add4");
  add_nodes.emplace_back(add_node);

  for (auto &node : add_nodes) {
    node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
    node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
    node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
    node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
    node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT);
    node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
    node->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
    node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_ND);
    node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
    node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
    node->GetOpDesc()->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_ND);
    node->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_ND);
    node->GetOpDesc()->AppendIrInput("x1", kIrInputRequired);
    node->GetOpDesc()->AppendIrInput("x2", kIrInputRequired);
    node->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
    node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
    AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
    AttrUtils::SetStr(node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
    AddCompileResult(node, false);
  }

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  then_graph->SetGraphUnknownFlag(true);
  else_graph->SetGraphUnknownFlag(true);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
*                          +------------------+  +-----------------+
*                          |     Then Graph   |  |    Else Graph   |
*                          |                  |  |                 |
*                          |     NetOutput    |  |    NetOutput    |
*       NetOutput          |         |        |  |        |        |
*           |              |        Add       |  |       Add       |
*          if  <---------> |       /   \      |  |      /   \      |
*        /    \            |  Data(1) Data(2) |  | Data(1) Data(2) |
* pred(Data)  input(Data)  +------------------+  +-----------------+
*/
ComputeGraphPtr ShareGraph::IfGraph5() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto then_graph = AicoreGraph();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindNode("data1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(then_graph->FindNode("data2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = AicoreGraph();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindNode("data1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(else_graph->FindNode("data2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}

/*
 *                          +-----------+  +-----------+  +-----------+
 *                          | Graph 0   |  | Graph 1   |  | Graph 2   |
 *                          |           |  |           |  |           |
 *                          | NetOutput |  | NetOutput |  | NetOutput |
 *       NetOutput          |   |       |  |   |       |  |   |       |
 *           |              |  Shape    |  |  Rank     |  |  Size     |
 *         Case <---------> |   |       |  |   |       |  |   |       |
 *        /    \            | Data(1)   |  | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::CaseGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("index", "Data")->NODE("case", "Case")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("case", "Case"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("index")->GetOpDesc(), "index", 0);
  main_graph->FindNode("index")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  main_graph->FindNode("index")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);

  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  auto case_node = main_graph->FindNode("case");
  case_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->AppendIrInput("branch_index", kIrInputRequired);
  case_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);

  auto &name_index = case_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["branch_index"] = 0;
  name_index["input0"] = 1;

  auto graph_1 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("shape", "Shape")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph_1->SetName("branch1");
  ge::AttrUtils::SetInt(graph_1->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(graph_1->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto graph_2 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("rank", "Rank")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph_2->SetName("branch2");
  ge::AttrUtils::SetInt(graph_2->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(graph_2->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto graph_3 = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("size", "Size")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph_3->SetName("branch3");
  ge::AttrUtils::SetInt(graph_3->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(graph_3->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  graph_1->SetParentGraph(main_graph);
  graph_1->SetParentNode(case_node);
  graph_2->SetParentGraph(main_graph);
  graph_2->SetParentNode(case_node);
  graph_3->SetParentGraph(main_graph);
  graph_3->SetParentNode(case_node);

  main_graph->AddSubgraph(graph_1);
  main_graph->AddSubgraph(graph_2);
  main_graph->AddSubgraph(graph_3);
  case_node->GetOpDesc()->AddSubgraphName("branch1");
  case_node->GetOpDesc()->AddSubgraphName("branch2");
  case_node->GetOpDesc()->AddSubgraphName("branch3");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "branch1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "branch2");
  case_node->GetOpDesc()->SetSubgraphInstanceName(2, "branch3");
  main_graph->TopologicalSorting();

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"case"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}

/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  |           |
 *       NetOutput          |    :      |  |           |
 *           |              |  Clean    |  |           |
 *          if  <---------> |   |       |  |           |
 *        /    \            | Data(1)   |  |           |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfOneBranchGraph2() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
      CTRL_CHAIN(NODE("if", "If")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  int index = 0;
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
    }
  }

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("clean", "Clean"));
      CTRL_CHAIN(NODE("clean", "Clean")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = std::make_shared<ge::ComputeGraph>("else");
  else_graph->SetName("else");

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;
  main_graph->TopologicalSorting();
  return main_graph;
}
/*
 *                                 +--------------+
 *                                 | Else Graph   |
 *                 +------------+  |              |
 *                 | Then graph |  |   NetOutput  |
 *                 |            |  |      |       |
 *    NetOutput    | NetOutput  |  |   ReShape    |
 *     |     \     |    |       |  |     / \      |
 *     |     If -- |  Data      |  | Data  Const  |
 *     \    /  \   +------------+  +--------------+
 *      data   pred
 */
ge::ComputeGraphPtr ShareGraph::IfGraphShapeChangedOneBranch() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If"));
      CHAIN(NODE("input", "Data")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
      CTRL_CHAIN(NODE("if", "If")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  main_graph->FindNode("pred")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);

  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  main_graph->FindNode("input")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);

  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"input", "if"});
  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 0});

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("reshape", "Reshape")->NODE("NetOutput", "NetOutput"));
      CTRL_CHAIN(NODE("shape_const", "Const")->EDGE(0, 1)->NODE("reshape", "Reshape"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  GeTensor shape_const_value;
  shape_const_value.MutableTensorDesc().SetFormat(ge::FORMAT_ND);
  shape_const_value.MutableTensorDesc().SetOriginFormat(ge::FORMAT_ND);
  shape_const_value.MutableTensorDesc().SetShape(GeShape({2}));
  shape_const_value.MutableTensorDesc().SetOriginShape(GeShape({2}));
  shape_const_value.MutableTensorDesc().SetDataType(ge::DT_INT64);
  shape_const_value.MutableTensorDesc().SetOriginDataType(ge::DT_INT64);
  std::vector<int64_t> tensor_value{2, -1};
  shape_const_value.SetData(reinterpret_cast<uint8_t *>(tensor_value.data()), tensor_value.size() * sizeof(int64_t));
  ge::AttrUtils::SetTensor(else_graph->FindFirstNodeMatchType("Const")->GetOpDesc(), "value", shape_const_value);
  *else_graph->FindFirstNodeMatchType("Const")->GetOpDesc()->MutableOutputDesc(0) = shape_const_value.GetTensorDesc();

  auto reshape_node = else_graph->FindFirstNodeMatchType("Reshape");
  SetNoStorage(reshape_node->GetOpDesc(), ge::FORMAT_ND, DT_INT64, {-1, -1});
  reshape_node->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape_node->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);
  auto &reshape_names_indexes = reshape_node->GetOpDesc()->MutableAllInputName();
  reshape_names_indexes.clear();
  reshape_names_indexes["x"] = 0;
  reshape_names_indexes["shape"] = 1;

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }

  return main_graph;
}
/*
 *
 *                   NetOutput
 *                       |
 *                      If
 *                     /   \
 *        ConditionCalc    input(Data)
 *         /       \
 * cond1(Data)    conv2(Data)
 */
ComputeGraphPtr ShareGraph::BinaryKernelTypicalGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("cond1", "Data")
                ->NODE("condition_calc", "ConditionCalc")
                ->NODE("if", "If")
                ->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("cond2", "Data")->EDGE(0, 1)->NODE("condition_calc", "ConditionCalc"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  int index = 0;
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
      node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
    } else if (node->GetType() == "ConditionCalc") {
      ge::AttrUtils::SetStr(node->GetOpDesc(), "cond_func", "changeme");
      std::vector<int64_t> input_dependency;
      input_dependency.resize(node->GetAllInDataAnchorsSize(), 0);
      ge::AttrUtils::SetListInt(node->GetOpDesc(), "x_dependency", input_dependency);
      node->GetOpDesc()->MutableAllInputName() = {{"input0", 0}, {"input1", 1}};
    }
  }

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("foo", "Foo")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  auto then_data = then_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(then_data->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_data->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  then_data->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("bar", "Bar")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  auto else_data = else_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(else_data->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_data->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  else_data->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;

  main_graph->SetGraphUnknownFlag(true);
  for (auto &subgraph : main_graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }
  return main_graph;
}

void ModifyInputNameIndexesForIf(ComputeGraphPtr graph) {
  for (auto &node : graph->GetAllNodes()) {
    if (!(node->GetType() == "If")) {
      continue;
    }
    auto &names_indexes = node->GetOpDesc()->MutableAllInputName();
    names_indexes.clear();
    auto inputs_num = node->GetInDataNodesAndAnchors().size();
    names_indexes.clear();
    names_indexes["cond"] = 0;
    for (size_t i = 1U; i < inputs_num; ++i) {
      names_indexes["input" + std::to_string(i - 1)] = i;
    }
    if (node->GetOpDesc()->GetIrInputs().empty()) {
      node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
      node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
    }
  }
}

ComputeGraphPtr ShareGraph::MatmulOmBinaryGraphV2() {
  auto main_graph_builder = [](const std::string &prefix) {
    auto graph = []() {
      DEF_GRAPH(g) {
        CHAIN(NODE("x1", "Data")->EDGE(0, 0)->NODE("condition_calc", "FakeConditionCalc"));
        CHAIN(NODE("x2", "Data")->EDGE(0, 1)->NODE("condition_calc", "FakeConditionCalc"));
        CHAIN(NODE("bias", "Data")->EDGE(0, 2)->NODE("condition_calc", "FakeConditionCalc"));
        CHAIN(NODE("offset_w", "Data")->EDGE(0, 3)->NODE("condition_calc", "FakeConditionCalc"));

        CHAIN(NODE("condition_calc", "FakeConditionCalc")->EDGE(0, 0)->NODE("if", "If"));
        CHAIN(NODE("x1", "Data")->EDGE(0, 1)->NODE("if", "If"));
        CHAIN(NODE("x2", "Data")->EDGE(0, 2)->NODE("if", "If"));
        CHAIN(NODE("bias", "Data")->EDGE(0, 3)->NODE("if", "If"));
        CHAIN(NODE("offset_w", "Data")->EDGE(0, 4)->NODE("if", "If"));

        CHAIN(NODE("if", "If")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
      };
      return ToComputeGraph(g);
    }();
    graph->SetName(prefix);
    graph->FindNode("x1")->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({UNKNOWN_DIM}));
    ge::AttrUtils::SetInt(graph->FindNode("x1")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(graph->FindNode("x2")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(graph->FindNode("bias")->GetOpDesc(), "index", 2);
    ge::AttrUtils::SetInt(graph->FindNode("offset_w")->GetOpDesc(), "index", 3);

    auto cond_calc = graph->FindNode("condition_calc");
    ge::AttrUtils::SetStr(cond_calc->GetOpDesc(), "cond_func", cond_calc->GetName());
    ge::AttrUtils::SetListInt(cond_calc->GetOpDesc(), "x_dependency",
                              std::vector<int64_t>(cond_calc->GetAllInDataAnchorsSize(), 0));
    graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"if"});
    graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
    auto if_node = graph->FindNode("if");
    if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
    if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
    auto &name_index = if_node->GetOpDesc()->MutableAllInputName();
    name_index.clear();
    name_index["cond"] = 0;
    name_index["input0"] = 1;
    name_index["input1"] = 2;
    name_index["input2"] = 3;
    name_index["input3"] = 4;
    return graph;
  };

  auto sub_graph_builder = [](const std::string &name, const std::string &type, ge::NodePtr parent,
                              bool sub_sub = false) {
    std::string prefix = name + "/";
    auto graph = [&prefix, &type, &sub_sub]() {
      if (type == "If") {
        DEF_GRAPH(g) {
          CHAIN(NODE(prefix + "0", "Data")->EDGE(0, 0)->NODE(prefix + "tiling", "OpTiling"));
          CHAIN(NODE(prefix + "1", "Data")->EDGE(0, 1)->NODE(prefix + "tiling", "OpTiling"));
          CHAIN(NODE(prefix + "2", "Data")->EDGE(0, 2)->NODE(prefix + "tiling", "OpTiling"));
          CHAIN(NODE(prefix + "3", "Data")->EDGE(0, 3)->NODE(prefix + "tiling", "OpTiling"));

          CHAIN(NODE(prefix + "tiling", "OpTiling")->EDGE(0, 0)->NODE(prefix + "if", type));
          CHAIN(NODE(prefix + "0", "Data")->EDGE(0, 1)->NODE(prefix + "if", type)->NODE(prefix + "out", "NetOutput"));
          CHAIN(NODE(prefix + "1", "Data")->EDGE(0, 2)->NODE(prefix + "if", type));
          CHAIN(NODE(prefix + "2", "Data")->EDGE(0, 3)->NODE(prefix + "if", type));
          CHAIN(NODE(prefix + "3", "Data")->EDGE(0, 4)->NODE(prefix + "if", type));
          CHAIN(NODE(prefix + "tiling", "OpTiling")->EDGE(1, 5)->NODE(prefix + "if", type));
        };
        return ToComputeGraph(g);
      } else {
        DEF_GRAPH(g) {
          CHAIN(NODE(prefix + "0", "Data")->EDGE(0, 0)->NODE(prefix + "mul", type)->NODE(prefix + "out", "NetOutput"));
          CHAIN(NODE(prefix + "1", "Data")->EDGE(0, 1)->NODE(prefix + "mul", type));
          CHAIN(NODE(prefix + "2", "Data")->EDGE(0, 2)->NODE(prefix + "mul", type));
          CHAIN(NODE(prefix + "3", "Data")->EDGE(0, 3)->NODE(prefix + "mul", type));
          if (sub_sub) {
            CTRL_CHAIN(NODE(prefix + "4", "Data")->NODE(prefix + "mul", type));
          }
        };
        return ToComputeGraph(g);
      }
    }();
    graph->SetName(name);
    graph->FindNode(prefix + "0")->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({UNKNOWN_DIM}));
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "0")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "1")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "2")->GetOpDesc(), "index", 2);
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "3")->GetOpDesc(), "index", 3);
    if (sub_sub) {
      ge::AttrUtils::SetInt(graph->FindNode(prefix + "4")->GetOpDesc(), "index", 4);
    }

    ge::AttrUtils::SetInt(graph->FindNode(prefix + "0")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 3);
    ge::AttrUtils::SetInt(graph->FindNode(prefix + "3")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 4);
    if (sub_sub) {
      ge::AttrUtils::SetInt(graph->FindNode(prefix + "4")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 5);
    }

    graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->SetSrcName({prefix + "mul"});
    graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->SetSrcIndex({0});

    graph->SetParentNode(parent);
    graph->SetParentGraph(parent->GetOwnerComputeGraph());
    ge::GraphUtils::FindRootGraph(parent->GetOwnerComputeGraph())->AddSubgraph(graph);
    parent->GetOpDesc()->AddSubgraphName(graph->GetName());  // Om is already like this...
    parent->GetOpDesc()->SetSubgraphInstanceName(parent->GetOpDesc()->GetSubgraphInstanceNames().size() - 1U,
                                                 graph->GetName());
    return graph;
  };

  auto main_graph = main_graph_builder("main");
  auto if_node = main_graph->FindFirstNodeMatchType("If");
  auto then_graph = sub_graph_builder("then", "MatMulV2", if_node);
  auto else_graph = sub_graph_builder("else", "If", if_node);

  auto else_if_node = else_graph->FindFirstNodeMatchType("If");
  auto else_graph_then = sub_graph_builder("else_then", "MatMulV2", else_if_node, true);
  auto else_graph_else = sub_graph_builder("else_else", "MatMulV2", else_if_node, true);
  ModifyInputNameIndexesForIf(main_graph);
  return main_graph;
}

ComputeGraphPtr ShareGraph::SimpleFooGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "RefData")->NODE("foo", "Foo")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  int index = 0;
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == "RefData") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
      node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
    }
  }
  auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  return main_graph;
}

ComputeGraphPtr ShareGraph::SimpleVariableGraph(const std::string &var_name) {
  auto main_graph = [&var_name]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(var_name, "Variable")->NODE("foo", "Foo"));
      CHAIN(NODE("constant", "Constant")->EDGE(0, 1)->NODE("foo", "Foo")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();

  ge::AttrUtils::SetTensor(main_graph->FindNode("constant")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  main_graph->SetName("main");
  int index = 0;
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
      node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
    }
  }

  return main_graph;
}

ComputeGraphPtr ShareGraph::SimpleVariableAddGraph() {
  DEF_GRAPH(g) {
    CHAIN(NODE("var1", "Variable")->NODE("add", "Add"));
    CHAIN(NODE("var2", "Variable")->EDGE(0, 1)->NODE("add", "Add")->NODE("NetOutput", "NetOutput"));
  };

  auto root_graph = ToComputeGraph(g);

  auto var1 = root_graph->FindNode("var1");
  var1->GetOpDescBarePtr()->SetOutputOffset({137438953472});
  TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  auto var2 = root_graph->FindNode("var2");
  var2->GetOpDescBarePtr()->SetOutputOffset({137438953572});
  TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 64);

  auto add = root_graph->FindNode("add");
  AddCompileResult(add, false);

  auto netoutput = root_graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"add"});
  netoutput->GetOpDesc()->SetSrcIndex({0});

  return root_graph;
}

ComputeGraphPtr ShareGraph::SimpleFileConstantGraph(const std::string &var_name, const std::string &file_constant_name,
                                                    const std::string &location) {
  auto main_graph = [&var_name, &file_constant_name]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(var_name, "Variable")->NODE("foo", "Foo"));
      CHAIN(NODE(file_constant_name, "FileConstant")->EDGE(0, 1)->NODE("foo", "Foo")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();

  main_graph->SetName("main");
  int index = 0;
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
      node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
    }
  }
  auto file_constant = main_graph->FindFirstNodeMatchType("FileConstant");
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "offset", 0);
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "length", 0);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "location", location);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_path", location);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_id", "");
  ge::AttrUtils::SetDataType(file_constant->GetOpDesc(), "dtype", ge::DT_FLOAT);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "shape", {});
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "original_shape", {});

  return main_graph;
}

/*
      var     constant
         \     /
          assgin
            |
            -------->netoutput
*/
ComputeGraphPtr ShareGraph::SimpleVariableAssignGraph(const std::string &var_name) {
  auto main_graph = [&var_name]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(var_name, "Variable")->NODE("assign", "Assign"));
      CHAIN(NODE("constant", "Constant")
                ->EDGE(0, 1)
                ->NODE("assign", "Assign")
                ->CTRL_EDGE()
                ->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetTensor(main_graph->FindNode("constant")->GetOpDesc(), "value", CreateScalarGeTensor(0));

  return main_graph;
}

/*
 *   netoutput
 *       |
 *   expandims
 *     /   \(axis)
 * data1   data2
 *
 */
ComputeGraphPtr ShareGraph::BuildDataDependencyNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("expandims", "ExpandDims")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("expandims"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  AttrUtils::SetInt(graph->FindNode("data2")->GetOpDesc(), "index", 1);
  SetNoStorage(graph->FindNode("data2")->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {});

  graph->FindNode("expandims")->GetOpDesc()->SetOpInferDepends({"axis"});
  graph->FindNode("expandims")->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"axis", 1}};
  graph->FindNode("expandims")->GetOpDesc()->MutableAllOutputName() = {{"y", 0}};
  SetNoStorage(graph->FindNode("expandims")->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {-1, -1, -1, -1, -1});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"expandims"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *   NetOutput
 *     |
 *   ConditionCalc
 *     |
 *    add
 *    /  \
 * data0 data1
 */
ge::ComputeGraphPtr ShareGraph::BuildAddConditionCalcGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")
              ->NODE("add1", "Add")
              ->NODE("condition_calc", "ConditionCalc")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 1);

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AddCompileResult(add1, false);

  auto condition_calc = graph->FindNode("condition_calc");
  condition_calc->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameGeLocal);
  AttrUtils::SetListInt(condition_calc->GetOpDesc(), "x_dependency", {1});
  AttrUtils::SetStr(condition_calc->GetOpDesc(), "cond_func", "cond_func");

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"condition_calc"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*          NetOutput 0 1 2 3
 *                 |
 *         (real tiling node)
 *         /               \
 *     data1              data2
 */

ComputeGraphPtr ShareGraph::BuildOpTilingGraph(const std::string &node_type) {
  DEF_GRAPH(root_graph) {
    CHAIN(NODE("data1", "Data")->NODE("real_tiling_node", node_type)->NODE("NetOutput_0", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("real_tiling_node", node_type));
    CHAIN(NODE("real_tiling_node", node_type)->EDGE(1, 1)->NODE("NetOutput_0", "NetOutput"));
    CHAIN(NODE("real_tiling_node", node_type)->EDGE(2, 2)->NODE("NetOutput_0", "NetOutput"));
    CHAIN(NODE("real_tiling_node", node_type)->EDGE(3, 3)->NODE("NetOutput_0", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("OpTiling_Test", "OpTiling")->NODE("NetOutput0", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("OpTiling_Test", "OpTiling"));
    CHAIN(NODE("OpTiling_Test", "OpTiling")->EDGE(1, 1)->NODE("NetOutput0", "NetOutput"));
    CHAIN(NODE("OpTiling_Test", "OpTiling")->EDGE(2, 2)->NODE("NetOutput0", "NetOutput"));
    CHAIN(NODE("OpTiling_Test", "OpTiling")->EDGE(3, 3)->NODE("NetOutput0", "NetOutput"));
  };
  auto graph = ToComputeGraph(root_graph);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  AttrUtils::SetInt(graph->FindNode("data2")->GetOpDesc(), "index", 1);
  SetNoStorage(graph->FindNode("data2")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  graph->FindNode("real_tiling_node")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(graph->FindNode("real_tiling_node")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  AddCompileResult(graph->FindNode("real_tiling_node"), false, "");

  graph->FindNode("OpTiling_Test")->GetOpDesc()->SetOpKernelLibName("OpTiling");
  SetNoStorage(graph->FindNode("OpTiling_Test")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  AddCompileResult(graph->FindNode("OpTiling_Test"), false);
  AttrUtils::SetStr(graph->FindNode("OpTiling_Test")->GetOpDesc(), "tiling_node", "real_tiling_node");

  graph->FindNode("NetOutput0")
      ->GetOpDesc()
      ->SetSrcName({"OpTiling_Test", "OpTiling_Test", "OpTiling_Test", "OpTiling_Test"});
  graph->FindNode("NetOutput0")->GetOpDesc()->SetSrcIndex({0, 1, 2, 3});
  graph->FindNode("NetOutput_0")
      ->GetOpDesc()
      ->SetSrcName({"real_tiling_node", "real_tiling_node", "real_tiling_node", "real_tiling_node"});
  graph->FindNode("NetOutput_0")->GetOpDesc()->SetSrcIndex({0, 1, 2, 3});
  return graph;
}

/*
 *                                       ret_val
 *                                          |
 *  netoutput                              mul_a
 *      |                                /     \
 *     add_mul     ----------------   add_a     data_c
 *    /  \     \                    /      \
 * data1 data2 data3              data_a   data_b
 */
ComputeGraphPtr ShareGraph::BuildGraphWithUBFusionNode() {
  std::vector<int64_t> shape = {2, 2, 3, 2};  // NCHW
  // subgraph
  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("data_a");

  auto data_b = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                    .Build("data_b");
  auto data_c = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 2)
                    .Build("data_c");
  auto netoutput_sub = OP_CFG("_RetVal")
                           .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                           .InCnt(1)
                           .OutCnt(1)
                           .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Build("netoutput_sub");
  DEF_GRAPH(fuse_origin_graph) {
    CHAIN(NODE(data_a)->NODE("add_a", "Add")->NODE("mul_a", "Mul")->NODE(netoutput_sub));
    CHAIN(NODE(data_b)->EDGE(0, 1)->NODE("add_a"));
    CHAIN(NODE(data_c)->EDGE(0, 1)->NODE("mul_a"));
  };

  // root graph
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data2");
  auto data3 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .Build("data3");
  auto add_mul = OP_CFG("Add").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(1).Build("add_mul");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->NODE(add_mul)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add_mul));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(add_mul));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add_mul"});
  compute_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  auto add_mul_fused_node = compute_graph->FindNode("add_mul");
  add_mul_fused_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  add_mul_fused_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-2}));
  auto fused_compute_graph = ToComputeGraph(fuse_origin_graph);
  AttrUtils::SetGraph(add_mul_fused_node->GetOpDesc(), "_original_fusion_graph", fused_compute_graph);
  AddCompileResult(compute_graph->FindNode("add_mul"), false);
  add_mul_fused_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  // set engine
  fused_compute_graph->FindNode("add_a")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AddCompileResult(fused_compute_graph->FindNode("add_a"), false);
  fused_compute_graph->FindNode("mul_a")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AddCompileResult(fused_compute_graph->FindNode("mul_a"), false);
  SetGraphOutShapeRange(compute_graph);
  return compute_graph;
}

/*
                            g1
                                        (1,1)
                      ┌────────────────────────────┐
                      │                            ∨
┌────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ _known_subgraph │ ───────> │ netoutput │
└────────┘          └─────────────────┘          └───────────┘
                      │     |           (2,2)      ∧
                      └─────|──────────────────────┘
                            |
                            g2
------------------------------------------------------------------------------------------------
|                                       (0,0)                                                   |
|                      ┌─────────────────────────────────────────────────────────────────┐      |
|                      ∨                                                                 │      |
|┌────────┐  (0,2)   ┌───────────────┐  (1,1)   ┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌──────┐ |
|│ const1 │ ───────> │ netoutput_sub │ <─────── │ data1 │ ───────> │ conv2d │ ───────> │ relu │  |
|└────────┘          └───────────────┘          └───────┘          └────────┘          └──────┘  |
-------------------------------------------------------------------------------------------------
 */
ge::ComputeGraphPtr ShareGraph::BuildWithKnownSubgraph(bool no_netoutput, bool external_weight) {
  std::vector<int64_t> shape = {2, 2};           // NCHW
  std::vector<int64_t> unknown_shape = {2, -1};  // NCHW

  int64_t shape_size;
  TensorUtils::CalcTensorMemSize(GeShape(shape), FORMAT_NCHW, DT_FLOAT, shape_size);
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                   .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
                   .Build("data1");
  data1->SetOutputOffset({0});
  TensorUtils::SetSize(*data1->MutableOutputDesc(0), shape_size);

  auto const1 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("const1");
  ge::AttrUtils::SetTensor(const1, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  const1->SetOutputOffset({1024});
  TensorUtils::SetSize(*const1->MutableOutputDesc(0), shape_size);
  if (external_weight) {
    const1->SetType(FILECONSTANT);
    AttrUtils::SetStr(const1, "file_path", "file1.bin");
  }
  vector<int64_t> test_int64_list_attr = {1, 2, 3};
  auto conv2d = OP_CFG("Conv2d")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr("string_attr", "test")
                    .Attr("int32_attr", (int32_t)1)
                    .Attr("uint32_attr", (uint32_t)1)
                    .Attr("data_format", "NHWC")  // attr on operator
                    .Attr("dilations", test_int64_list_attr)
                    .Attr("groups", (int32_t)1)
                    .Attr("offset_x", (int32_t)1)
                    .Build("conv2d");
  conv2d->SetOpEngineName("AIcoreEngine");
  conv2d->SetOpKernelLibName("AIcoreEngine");
  conv2d->SetInputOffset({0});
  conv2d->SetOutputOffset({16});
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(conv2d->GetName(), std::move(kernel_bin));
  conv2d->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle);
  AttrUtils::SetStr(conv2d, conv2d->GetName() + "_kernelname", conv2d->GetName());
  AttrUtils::SetStr(conv2d, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(conv2d, ATTR_NAME_KERNEL_BIN_ID, "te_conv2d_123");

  auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("relu");
  relu->SetInputOffset({16});
  relu->SetOutputOffset({48});
  TensorUtils::SetSize(*relu->MutableOutputDesc(0), shape_size);

  relu->SetOpEngineName("AIcoreEngine");
  relu->SetOpKernelLibName("AIcoreEngine");
  TBEKernelPtr relu_kernel_handle = MakeShared<OpKernelBin>(relu->GetName(), std::move(kernel_bin));
  relu->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, relu_kernel_handle);
  AttrUtils::SetStr(relu, relu->GetName() + "_kernelname", relu->GetName());
  AttrUtils::SetStr(relu, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(relu, ATTR_NAME_KERNEL_BIN_ID, "te_relu_123");

  auto netoutput_sub = OP_CFG("NetOutput")
                           .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                           .InCnt(3)
                           .OutCnt(1)
                           .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"relu", "data1", "const1"});
  netoutput_sub->SetSrcIndex({0, 0, 0});
  netoutput_sub->SetInputOffset({48, 0, 1024});

  DEF_GRAPH(g2) {
    if (no_netoutput) {
      CHAIN(NODE(data1)->NODE(conv2d)->NODE(relu));
    } else {
      CHAIN(NODE(data1)->NODE(conv2d)->NODE(relu)->EDGE(0, 0)->NODE(netoutput_sub));
      CHAIN(NODE(data1)->EDGE(0, 1)->NODE(netoutput_sub));
      CHAIN(NODE(const1)->EDGE(0, 2)->NODE(netoutput_sub));
    }
  };

  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .InCnt(1)
                            .OutCnt(3)
                            .Build(ge::PARTITIONEDCALL);

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(3).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->EDGE(1, 1)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->EDGE(2, 2)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  AddCompileResult(parent_node, false);
  return compute_graph;
}

/*
                            g1
                                        (1,1)
                      ┌────────────────────────────┐
                      │                            ∨
┌────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ _known_subgraph │ ───────> │ netoutput │
└────────┘          └─────────────────┘          └───────────┘
                      │     |           (2,2)      ∧
                      └─────|──────────────────────┘
                            |
                            g2
------------------------------------------------------------------------------------------------
|                                       (0,0)                                                   |
|                      ┌─────────────────────────────────────────────────────────────────┐      |
|                      ∨                                                                 │      |
|┌────────┐  (0,2)   ┌───────────────┐  (1,1)   ┌───────┐  (0,0)   ┌────────┐          ┌──────┐ |
|│ const1 │ ───────> │ netoutput_sub │ <─────── │ data1 │ ───────> │ conv2d │          │const2│ |
|└────────┘          └───────────────┘          └───────┘          └────────┘          └──────┘ |
-------------------------------------------------------------------------------------------------
 */
ge::ComputeGraphPtr ShareGraph::BuildWithKnownSubgraphWithTwoConst(bool no_netoutput) {
  std::vector<int64_t> shape = {2, 2};           // NCHW
  std::vector<int64_t> unknown_shape = {2, -1};  // NCHW

  int64_t shape_size;
  TensorUtils::CalcTensorMemSize(GeShape(shape), FORMAT_NCHW, DT_FLOAT, shape_size);
  auto data1 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
      .Build("data1");
  data1->SetOutputOffset({0});
  TensorUtils::SetSize(*data1->MutableOutputDesc(0), shape_size);

  auto const1 = OP_CFG("Const")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("const1");
  ge::AttrUtils::SetTensor(const1, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  const1->SetOutputOffset({1024});
  TensorUtils::SetSize(*const1->MutableOutputDesc(0), shape_size);

  vector<int64_t> test_int64_list_attr = {1, 2, 3};
  auto conv2d = OP_CFG("Conv2d")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr("string_attr", "test")
      .Attr("int32_attr", (int32_t)1)
      .Attr("uint32_attr", (uint32_t)1)
      .Attr("data_format", "NHWC")  // attr on operator
      .Attr("dilations", test_int64_list_attr)
      .Attr("groups", (int32_t)1)
      .Attr("offset_x", (int32_t)1)
      .Build("conv2d");
  conv2d->SetOpEngineName("AIcoreEngine");
  conv2d->SetOpKernelLibName("AIcoreEngine");
  conv2d->SetInputOffset({0});
  conv2d->SetOutputOffset({16});

  auto const2 = OP_CFG("Const").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("const2");
  ge::AttrUtils::SetTensor(const2, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  const2->SetOutputOffset({48});
  TensorUtils::SetSize(*const2->MutableOutputDesc(0), shape_size);

  auto netoutput_sub = OP_CFG("NetOutput")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(3)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"const2", "data1", "const1"});
  netoutput_sub->SetSrcIndex({0, 0, 0});
  netoutput_sub->SetInputOffset({48, 0, 1024});
  ge::TensorUtils::SetDataOffset(*netoutput_sub->MutableInputDesc(0), 48);
  ge::TensorUtils::SetDataOffset(*netoutput_sub->MutableInputDesc(2), 1024);
  netoutput_sub->SetIsInputConst({true, false, true});

  DEF_GRAPH(g2) {
                  if (no_netoutput) {
                    CHAIN(NODE(data1)->NODE(conv2d)->NODE(const2));
                  } else {
                    CHAIN(NODE(data1)->NODE(conv2d));
                    CHAIN(NODE(const2)->EDGE(0, 0)->NODE(netoutput_sub));
                    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(netoutput_sub));
                    CHAIN(NODE(const1)->EDGE(0, 2)->NODE(netoutput_sub));
                  }
                };

  auto data_a = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(3)
      .Build(ge::PARTITIONEDCALL);

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(3).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
                  CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(netoutput));
                  CHAIN(NODE(known_subgraph)->EDGE(1, 1)->NODE(netoutput));
                  CHAIN(NODE(known_subgraph)->EDGE(2, 2)->NODE(netoutput));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  auto sub_netoutput = sub_compute_graph->FindNode("netoutput_sub");
  ge::TensorUtils::SetDataOffset(*sub_netoutput->GetOpDescBarePtr()->MutableInputDesc(0), 48);
  ge::TensorUtils::SetDataOffset(*sub_netoutput->GetOpDescBarePtr()->MutableInputDesc(2), 1024);
  sub_netoutput->GetOpDescBarePtr()->SetIsInputConst({true, false, true});

  AddCompileResult(parent_node, false);
  return compute_graph;
}
/*
                            g1
                                        (1,1)
                      ┌────────────────────────────┐
                      │                            ∨
┌────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ _known_subgraph │ ───────> │ netoutput │
└────────┘          └─────────────────┘          └───────────┘
                      │     |           (2,2)      ∧
                      └─────|──────────────────────┘
                            |
                            g2
            ---------------------------+
            |     data    const        |
            |        \  /              |
            |        assign            |
            |           |              |
            |         netoutput        |
            ---------------------------+
 */
ge::ComputeGraphPtr ShareGraph::BuildWithKnownSubgraphWithRefNode() {
  std::vector<int64_t> shape = {2, 2};           // NCHW
  std::vector<int64_t> unknown_shape = {2, -1};  // NCHW

  int64_t shape_size;
  TensorUtils::CalcTensorMemSize(GeShape(shape), FORMAT_NCHW, DT_FLOAT, shape_size);
  auto data1 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
      .Build("data1");
  data1->SetOutputOffset({0});
  TensorUtils::SetSize(*data1->MutableOutputDesc(0), shape_size);

  auto const1 = OP_CFG("Const")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("const1");
  ge::AttrUtils::SetTensor(const1, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  const1->SetOutputOffset({1024});
  TensorUtils::SetSize(*const1->MutableOutputDesc(0), shape_size);

  auto assign = OP_CFG(ASSIGN)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
                    .InCnt(2)
                    .OutCnt(1)
                    .InNames({"ref", "value"})
                    .OutNames({"ref"})
                    .Attr(ATTR_NAME_REFERENCE, true)
                    .Build("assign");
  assign->SetOpKernelLibName(ge::kEngineNameAiCore);
  assign->SetInputOffset({0});
  assign->SetOutputOffset({0});

  auto netoutput_sub = OP_CFG("NetOutput")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"assign"});
  netoutput_sub->SetSrcIndex({0});
  netoutput_sub->SetInputOffset({0});

  DEF_GRAPH(g2) {
    CHAIN(NODE(data1)->NODE(assign)->NODE(netoutput_sub));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(assign));
  };

  auto data_a = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_a");
  data_a->SetOutputOffset({2048});
  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build(ge::PARTITIONEDCALL);

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(3).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  AddCompileResult(parent_node, false);
  return compute_graph;
}

/*
                            g1
                                        (1,1)                         (1,1)
                      ┌────────────────────────────────┐   ┌─────────────────────────────┐
                      │                                ∨   │                             ∨
┌────────┐  (0,0)   ┌───────────────────┐  (0,0)   ┌───────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ _known_subgraph_1 │ ───────> │ _known_subgraph_2 │ ───────> │ netoutput │
└────────┘          └───────────────────┘          └───────────────────┘          └───────────┘
                      │     |              (2,2)       ∧   │     |           (2,2)       ∧
                      └─────|──────────────────────────┘   └─────|───────────────────────┘
                            |                                    |
                            g2                                   g3
            ---------------------------+                ---------------------------+
            |     data    const        |                |     data    const        |
            |        \  /              |                |        \  /              |
            |        assign            |                |        assign            |
            |           |              |                |           |              |
            |       netoutput          |                |       netoutput          |
            ---------------------------+                ---------------------------+
 */
ge::ComputeGraphPtr ShareGraph::BuildWithMulitKnownSubgraphs() {
  std::vector<int64_t> shape = {2, 2};           // NCHW
  std::vector<int64_t> unknown_shape = {2, -1};  // NCHW

  int64_t shape_size;
  TensorUtils::CalcTensorMemSize(GeShape(shape), FORMAT_NCHW, DT_FLOAT, shape_size);
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).Attr(ge::ATTR_NAME_INDEX, (int32_t)0).Build("data1");
  data1->SetOutputOffset({0});
  TensorUtils::SetSize(*data1->MutableOutputDesc(0), shape_size);

  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("const1");
  ge::AttrUtils::SetTensor(const1, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  const1->SetOutputOffset({1024});
  TensorUtils::SetSize(*const1->MutableOutputDesc(0), shape_size);

  auto assign = OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(2).OutCnt(1).InNames({"ref", "value"})
                    .OutNames({"ref"}).Attr(ATTR_NAME_REFERENCE, true).Build("assign");
  assign->SetOpKernelLibName(ge::kEngineNameAiCore);
  assign->SetInputOffset({0});
  assign->SetOutputOffset({0});

  auto netoutput_sub = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).Build("netoutput_sub");
  netoutput_sub->SetSrcName({"assign"});
  netoutput_sub->SetSrcIndex({0});
  netoutput_sub->SetInputOffset({0});

  DEF_GRAPH(g2) {
    CHAIN(NODE(data1)->NODE(assign)->NODE(netoutput_sub));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(assign));
  };

  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).Attr(ge::ATTR_NAME_INDEX, (int32_t)0).Build("data2");
  data2->SetOutputOffset({0});
  TensorUtils::SetSize(*data2->MutableOutputDesc(0), shape_size);

  auto const2 = OP_CFG("Const").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("const2");
  ge::AttrUtils::SetTensor(const2, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  const2->SetOutputOffset({1024});
  TensorUtils::SetSize(*const2->MutableOutputDesc(0), shape_size);

  auto assign2 = OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(2).OutCnt(1).InNames({"ref", "value"})
                    .OutNames({"ref"}).Attr(ATTR_NAME_REFERENCE, true).Build("assign2");
  assign2->SetOpKernelLibName(ge::kEngineNameAiCore);
  assign2->SetInputOffset({0});
  assign2->SetOutputOffset({0});

  auto netoutput_sub2 = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).Build("netoutput_sub2");
  netoutput_sub2->SetSrcName({"assign"});
  netoutput_sub2->SetSrcIndex({0});
  netoutput_sub2->SetInputOffset({0});

  DEF_GRAPH(g3) {
    CHAIN(NODE(data2)->NODE(assign2)->NODE(netoutput_sub2));
    CHAIN(NODE(const2)->EDGE(0, 1)->NODE(assign2));
  };

  auto data_a = OP_CFG("Data").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1)
      .Build(ge::PARTITIONEDCALL);
  auto known_subgraph2 = OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1)
      .Build("known_subgraph2");

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(3).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(known_subgraph2)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  AddCompileResult(parent_node, false);

  auto parent_node2 = compute_graph->FindNode("known_subgraph2");
  auto sub_graph2 = ToGeGraph(g3);
  auto sub_compute_graph2 = ge::GraphUtilsEx::GetComputeGraph(sub_graph2);
  sub_compute_graph2->SetGraphUnknownFlag(false);

  // set sub graph
  SetSubGraph(compute_graph, parent_node2, sub_compute_graph2);
  AddCompileResult(parent_node2, false);
  return compute_graph;
}

/*
                            g1

                                        (2,2)
                      ┌────────────────────────────┐
                      │                            ∨
┌────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ PartitionedCall │ ───────> │ netoutput │
└────────┘          └─────────────────┘          └───────────┘
                      │                 (1,1)      ∧
                      └────────────────────────────┘

                                 g2

                                            (0,0)
                     ┌─────────────────────────────────┐
                     │                                 ∨
┌───────┐  (0,0)   ┌──────────────────────┐  (1,1)   ┌───────────────┐
│ data1 │ ───────> │ sub_partitioned_call │ ───────> │ netoutput_sub │
└───────┘          └──────────────────────┘          └───────────────┘
                     │                      (2,2)      ∧
                     └─────────────────────────────────┘

                             g3

                                 (2,2)
                          ┌─────────────────┐
                          │                 ∨
┌────────────┐  (0,0)   ┌──────┐  (0,0)   ┌─────────────────┐
│ inner_data │ ───────> │ Cast │ ───────> │ inner_netoutput │
└────────────┘          └──────┘          └─────────────────┘
                          │      (1,1)      ∧
                          └─────────────────┘
 */
ge::ComputeGraphPtr ShareGraph::BuildWithNestingKnownSubgraph() {
  std::vector<int64_t> shape = {2, 2};           // NCHW
  std::vector<int64_t> unknown_shape = {2, -1};  // NCHW

  auto inner_data = OP_CFG("Data")
                        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                        .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
                        .Build("inner_data");
  inner_data->SetOutputOffset({0});
  auto cast = OP_CFG("Cast").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("Cast");
  cast->SetInputOffset({32});
  cast->SetOutputOffset({48});
  cast->SetOpEngineName("AIcoreEngine");
  cast->SetOpKernelLibName("AIcoreEngine");

  auto inner_netoutput = OP_CFG("NetOutput")
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                             .InCnt(3)
                             .OutCnt(1)
                             .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Build("inner_netoutput");
  inner_netoutput->SetSrcName({"cast", "cast", "cast"});
  inner_netoutput->SetSrcIndex({0, 0, 0});
  inner_netoutput->SetInputOffset({0, 0, 0});

  DEF_GRAPH(g3) {
    CHAIN(NODE(inner_data)->NODE(cast)->EDGE(0, 0)->NODE(inner_netoutput));
    CHAIN(NODE(cast)->EDGE(0, 1)->NODE(inner_netoutput));
    CHAIN(NODE(cast)->EDGE(0, 2)->NODE(inner_netoutput));
  };

  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                   .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
                   .Build("data1");
  data1->SetOutputOffset({0, 0});
  auto sub_partitioned_call = OP_CFG(ge::PARTITIONEDCALL)
                                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                                  .InCnt(1)
                                  .OutCnt(1)
                                  .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                                  .Build("sub_partitioned_call");

  auto netoutput_sub = OP_CFG("NetOutput")
                           .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                           .InCnt(3)
                           .OutCnt(1)
                           .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"sub_partitioned_call", "sub_partitioned_call", "sub_partitioned_call"});
  netoutput_sub->SetSrcIndex({0, 0, 0});
  netoutput_sub->SetInputOffset({0, 0, 0});

  DEF_GRAPH(g2) {
    CHAIN(NODE(data1)->NODE(sub_partitioned_call)->NODE(netoutput_sub));
    CHAIN(NODE(sub_partitioned_call)->NODE(netoutput_sub));
    CHAIN(NODE(sub_partitioned_call)->NODE(netoutput_sub));
  };

  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .InCnt(1)
                            .OutCnt(3)
                            .Build(ge::PARTITIONEDCALL);

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(1).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  auto inner_graph = ToGeGraph(g3);
  auto inner_compute_graph = ge::GraphUtilsEx::GetComputeGraph(inner_graph);
  inner_compute_graph->SetGraphUnknownFlag(false);
  auto sub_parent_node = sub_compute_graph->FindNode("sub_partitioned_call");
  auto inner_netoutput_node = inner_compute_graph->FindNode("inner_netoutput");
  for (size_t i = 0U; i < inner_netoutput_node->GetOpDesc()->GetInputsSize(); ++i) {
    auto input_tensor_desc = inner_netoutput_node->GetOpDesc()->MutableInputDesc(i);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(input_tensor_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, i));
  }
  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  SetSubGraph(sub_compute_graph, sub_parent_node, inner_compute_graph);
  compute_graph->AddSubGraph(inner_compute_graph);
  auto get_graph = compute_graph->GetSubgraph("g3");
  AddCompileResult(parent_node, false);
  return compute_graph;
}

/*
                             g1

                                        (1,1)
                      ┌────────────────────────────┐
                      │                            ∨
┌────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ PartitionedCall │ ───────> │ netoutput │
└────────┘          └─────────────────┘          └───────────┘
                      │                 (2,2)      ∧
                      └────────────────────────────┘

                           g2

┌────────┐  (0,0)   ┌───────────────┐  (0,2)   ┌─────────┐
│ const1 │ ───────> │ netoutput_sub │ <─────── │ const3 │
└────────┘          └───────────────┘          └─────────┘
                      ∧
                      │ (0,1)
                      │
                    ┌───────────────┐
                    │    const2     │
 */
ge::ComputeGraphPtr ShareGraph::BuildWithAllConstKnownSubgraph() {
  std::vector<int64_t> shape = {1};          // NCHW
  std::vector<int64_t> unknown_shape = {1};  // NCHW

  auto const1 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const1");
  auto const2 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const2");
  auto const3 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const3");

  auto netoutput_sub = OP_CFG("NetOutput")
                           .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                           .InCnt(3)
                           .OutCnt(1)
                           .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"const1", "const2", "const3"});
  netoutput_sub->SetSrcIndex({0, 0, 0});

  DEF_GRAPH(g2) {
    CHAIN(NODE(const1)->NODE(netoutput_sub));
    CHAIN(NODE(const2)->NODE(netoutput_sub));
    CHAIN(NODE(const3)->NODE(netoutput_sub));
  };

  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .InCnt(1)
                            .OutCnt(3)
                            .Build(ge::PARTITIONEDCALL);

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(1).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  ge::AttrUtils::SetBool(parent_node->GetOpDesc(), "without_kernel_store", 1);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  ge::AttrUtils::SetTensor(sub_compute_graph->FindNode("const1")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  ge::AttrUtils::SetTensor(sub_compute_graph->FindNode("const2")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  ge::AttrUtils::SetTensor(sub_compute_graph->FindNode("const3")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  return compute_graph;
}

/*
 *    netoutput
 *        |
 *       add                                      netoutput
 *    /   |     \                                    |
 * data const2 partitionedcall   ----------------  const1
 */
ge::ComputeGraphPtr ShareGraph::BuildWithAllConstKnownSubgraph2() {
  std::vector<int64_t> shape = {1};           // NCHW
  std::vector<int64_t> unknown_shape = {1};  // NCHW

  auto const1 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const1");

  auto netoutput_sub = OP_CFG("NetOutput")
                           .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                           .InCnt(1)
                           .OutCnt(1)
                           .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"const1"});
  netoutput_sub->SetSrcIndex({0});
  netoutput_sub->SetInputOffset({0});

  DEF_GRAPH(g2) {
    CHAIN(NODE(const1)->NODE(netoutput_sub));
  };

  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .InCnt(0)
                            .OutCnt(1)
                            .Build(ge::PARTITIONEDCALL);

  auto add = OP_CFG("Add")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(3)
                    .OutCnt(1)
                    .Build("add");
  add->SetOpEngineName("AIcoreEngine");
  add->SetOpKernelLibName("AIcoreEngine");

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("netoutput");
  netoutput->SetSrcName({"add"});
  netoutput->SetSrcIndex({0});

  auto const2 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const2");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(add));
    CHAIN(NODE(const2)->NODE(add));
    CHAIN(NODE(known_subgraph)->NODE(add)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);

  auto add_node = compute_graph->FindNode("add");
  AddCompileResult(add_node, false);

  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);
  ge::AttrUtils::SetBool(sub_compute_graph, "without_kernel_store", true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);

  ge::AttrUtils::SetTensor(sub_compute_graph->FindNode("const1")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  ge::AttrUtils::SetTensor(compute_graph->FindNode("const2")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  return compute_graph;
}

/*
                             g1

                                        (1,1)
                      ┌────────────────────────────┐
                      │                            ∨
┌────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ PartitionedCall │ ───────> │ netoutput │
└────────┘          └─────────────────┘          └───────────┘
                      │                 (2,2)      ∧
                      └────────────────────────────┘

                           g2

┌────────┐  (0,0)   ┌───────────────┐  (0,2)   ┌─────────┐
│ data_1 │ ───────> │ netoutput_sub │ <─────── │ const3 │
└────────┘          └───────────────┘          └─────────┘
                      ∧
                      │ (0,1)
                      │
                    ┌───────────────┐
                    │    const2     │
 */
ge::ComputeGraphPtr ShareGraph::BuildWithInnerDataSubgraph() {
  std::vector<int64_t> shape = {1};          // NCHW
  std::vector<int64_t> unknown_shape = {1};  // NCHW

  auto data_1 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("data_1");
  auto const2 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const2");
  auto const3 = OP_CFG("Const")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Build("const3");

  auto netoutput_sub = OP_CFG("NetOutput")
                           .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                           .InCnt(3)
                           .OutCnt(1)
                           .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                           .Build("netoutput_sub");
  netoutput_sub->SetSrcName({"data_1", "const2", "const3"});
  netoutput_sub->SetSrcIndex({0, 0, 0});

  DEF_GRAPH(g2) {
    CHAIN(NODE(data_1)->NODE(netoutput_sub));
    CHAIN(NODE(const2)->NODE(netoutput_sub));
    CHAIN(NODE(const3)->NODE(netoutput_sub));
  };

  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_a");

  auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .InCnt(1)
                            .OutCnt(3)
                            .Build(ge::PARTITIONEDCALL);

  auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(3).OutCnt(1).Build("netoutput");
  netoutput->SetSrcName({ge::PARTITIONEDCALL, ge::PARTITIONEDCALL, ge::PARTITIONEDCALL});
  netoutput->SetSrcIndex({0, 1, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->NODE(netoutput));
    CHAIN(NODE(known_subgraph)->NODE(netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
  ge::AttrUtils::SetBool(parent_node->GetOpDesc(), "without_kernel_store", 1);
  auto sub_graph = ToGeGraph(g2);
  auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
  sub_compute_graph->SetGraphUnknownFlag(false);

  ge::AttrUtils::SetTensor(sub_compute_graph->FindNode("const2")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  ge::AttrUtils::SetTensor(sub_compute_graph->FindNode("const3")->GetOpDesc(), "value", CreateScalarGeTensor(0));
  // set sub graph
  SetSubGraph(compute_graph, parent_node, sub_compute_graph);
  return compute_graph;
}

ComputeGraphPtr ShareGraph::SimpleStaticGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("static_data", "Data")->EDGE(0, 0)->NODE("static_foo", "StaticFoo"));
      CHAIN(NODE("static_foo", "StaticFoo")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  AttrUtils::SetInt(graph->FindNode("static_data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetBool(graph, "_stub_force_known_shape", true);
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"static_foo"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  graph->FindNode("NetOutput")->GetOpDesc()->SetInputOffset({0});
  SetOffsetForDataNetoutput(graph);
  return graph;
}

ComputeGraphPtr ShareGraph::SimpleStaticPartitionedCallGraph() {
  auto static_graph = SimpleStaticGraph();
  static_graph->SetName("static");
  AttrUtils::SetInt(static_graph->FindNode("static_data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE("pcall", "PartitionedCall"));
      CHAIN(NODE("pcall", "PartitionedCall")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(graph->FindNode("data")->GetOpDesc(), "index", 0);
  auto pcall = graph->FindNode("pcall");
  pcall->GetOpDesc()->AddSubgraphName("f");
  pcall->GetOpDesc()->SetSubgraphInstanceName(0, static_graph->GetName());
  static_graph->SetParentNode(pcall);
  static_graph->SetParentGraph(graph);
  graph->AddSubgraph(static_graph);
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"pcall"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  graph->FindNode("NetOutput")->GetOpDesc()->SetInputOffset({0});
  SetOffsetForDataNetoutput(graph);
  return graph;
}

SingleNodeGraphBuilder::SingleNodeGraphBuilder(const std::string &graph_name, const std::string &type)
    : name_(graph_name), type_(type) {}
SingleNodeGraphBuilder &SingleNodeGraphBuilder::NumInputs(size_t num_inputs) {
  num_inputs_ = num_inputs;
  return *this;
}
SingleNodeGraphBuilder &SingleNodeGraphBuilder::NumOutputs(size_t num_outputs) {
  num_outputs_ = num_outputs;
  return *this;
}

ComputeGraphPtr SingleNodeGraphBuilder::Build(const ge::NodePtr &parent) {
  auto prefix = name_.empty() ? "" : (name_ + "/");
  auto data_prefix = prefix + "data_";
  auto node_name = prefix + type_;
  auto net_output_name = prefix + "net_output";

  auto graph = [&]() {
    DEF_GRAPH(g) {
      for (size_t i = 0U; i < num_inputs_; i++) {
        CHAIN(NODE(data_prefix + std::to_string(i), "Data")->EDGE(0, i)->NODE(node_name, type_));
      }
      for (size_t i = 0U; i < num_outputs_; i++) {
        CHAIN(NODE(node_name, type_)->EDGE(i, i)->NODE(net_output_name, "NetOutput"));
      }
    };
    return ToComputeGraph(g);
  }();

  int64_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != ge::DATA) {
      continue;
    }
    AttrUtils::SetInt(node->GetOpDesc(), "index", index++);
  }

  std::vector<std::string> names(num_outputs_, node_name);
  std::vector<int64_t> indexes;
  for (size_t i = 0U; i < num_outputs_; i++) {
    indexes.push_back(i);
  }

  auto op = ge::OperatorFactory::CreateOperator("temp", type_.c_str());
  op.BreakConnect();
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc != nullptr) {
    auto target = graph->FindFirstNodeMatchType(type_);
    GE_ASSERT_NOTNULL(target);
    for (auto &ir_and_type : op_desc->GetIrInputs()) {
      target->GetOpDesc()->AppendIrInput(ir_and_type.first, ir_and_type.second);
    }
    for (auto &ir_and_type : op_desc->GetIrOutputs()) {
      target->GetOpDesc()->AppendIrOutput(ir_and_type.first, ir_and_type.second);
    }
  }

  auto net_output = graph->FindNode(net_output_name);
  net_output->GetOpDesc()->SetSrcName(names);
  net_output->GetOpDesc()->SetSrcIndex(indexes);
  net_output->GetOpDesc()->SetInputOffset({0});
  graph->SetName(name_);

  if (parent != nullptr) {
    size_t subgraph_index = parent->GetOpDesc()->GetSubgraphInstanceNames().size();
    parent->GetOpDesc()->AddSubgraphName(name_);
    parent->GetOpDesc()->SetSubgraphInstanceName(subgraph_index, name_);
    graph->SetParentNode(parent);
    graph->SetParentGraph(parent->GetOwnerComputeGraph());
    ge::GraphUtils::FindRootGraph(parent->GetOwnerComputeGraph())->AddSubgraph(graph);
  }

  return graph;
}

ComputeGraphPtr SingleNodeGraphBuilder::BuildSubGraph(const ge::NodePtr &parent, int64_t parent_start) {
  auto graph = Build(parent);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != ge::DATA) {
      continue;
    }
    AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, parent_start++);
  }
  return graph;
}

ComputeGraphPtr ShareGraph::IfWithKnownShapeSubGraph(const std::string &graph_name) {
  auto graph = SingleNodeGraphBuilder(graph_name, "If").NumInputs(2).Build();
  graph->SetGraphUnknownFlag(true);
  auto if_node = graph->FindFirstNodeMatchType("If");
  auto &name_index = if_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["cond"] = 0;
  name_index["input0"] = 1;
  auto then_graph = SingleNodeGraphBuilder("static_then", "StaticFoo").BuildSubGraph(if_node, 1);
  ge::AttrUtils::SetBool(then_graph, "_stub_force_known_shape", true);
  auto else_graph = SingleNodeGraphBuilder("static_else", "StaticFoo").BuildSubGraph(if_node, 1);
  ge::AttrUtils::SetBool(else_graph, "_stub_force_known_shape", true);
  SetOffsetForDataNetoutput(graph);
  return graph;
}

/**
 *
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  | NetOutput |
 *       NetOutput          |  \ | /    |  |   |       |
 *         \ | /            |StaticFoo  |  | StaticFoo |
 *          if  <---------> |   |       |  |   |       |
 *        /    \            | Data(0)   |  | Data(1)   |
 *      Data   Data         +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfWithKnownSubGraphAndMultiOutputs(const std::string &graph_name) {
  auto graph = SingleNodeGraphBuilder(graph_name, "If").NumInputs(2).NumOutputs(3).Build();
  graph->SetGraphUnknownFlag(false);

  auto if_node = graph->FindFirstNodeMatchType("If");
  auto &name_index = if_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["cond"] = 0;
  name_index["input0"] = 1;
  auto then_graph = SingleNodeGraphBuilder("static_then", "StaticFoo").NumOutputs(3).BuildSubGraph(if_node, 0);
  ge::AttrUtils::SetBool(then_graph, "_stub_force_known_shape", true);
  auto else_graph = SingleNodeGraphBuilder("static_else", "StaticFoo").BuildSubGraph(if_node, 1);
  ge::AttrUtils::SetBool(else_graph, "_stub_force_known_shape", true);

  return graph;
}

// No compile result
ComputeGraphPtr ShareGraph::AicoreNoCompileResultGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto add1 = graph->FindNode("add1");
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  return graph;
}

/*
 *   reshape
 *     /   \(shape)
 * data1   data2
 *
 */
ComputeGraphPtr ShareGraph::BuildDataDependencySingleOpNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("reshape", "Reshape")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("reshape", "Reshape"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  AttrUtils::SetInt(graph->FindNode("data2")->GetOpDesc(), "index", 1);
  SetNoStorage(graph->FindNode("data2")->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1});

  graph->FindNode("reshape")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(graph->FindNode("reshape")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1});
  AddCompileResult(graph->FindNode("reshape"), false);

  auto reshape1 = graph->FindNode("reshape");
  reshape1->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape1->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);
  auto &name_index = reshape1->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["x"] = 0;
  name_index["shape"] = 1;

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"reshape"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *     netoutput ---------
 *     |(h)   |(d)       |
 *     |    identity    send
 *     |      |(d)       |
 *     identity_n---------
 *     |(h)  |
 *  shape    |(unknown)
 *     |     |
 *  data1   data2
 */
ComputeGraphPtr ShareGraph::BuildIdentityNGraph() {
  DEF_GRAPH(g1) {
    CHAIN(
        NODE("data1", "Data")->NODE("shape", "Shape")->NODE("identity_n", "IdentityN")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")
              ->EDGE(0, 1)
              ->NODE("identity_n", "IdentityN")
              ->EDGE(1, 0)
              ->NODE("identity", "Identity")
              ->EDGE(0, 1)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("identity_n", "IdentityN")
              ->CTRL_EDGE()
              ->NODE("send", "Send")
              ->CTRL_EDGE()
              ->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(graph->FindNode("data2")->GetOpDesc(), "index", 1);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetNoStorage(graph->FindNode("data2")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  graph->FindNode("shape")->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  graph->FindNode("shape")->GetOpDesc()->AppendIrAttrName("dtype");
  AttrUtils::SetInt(graph->FindNode("shape")->GetOpDesc(), "dtype", 3);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"identity_n", "identity"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

/*
 *            netoutput
 *           ___/  \___
 *         /           \
 *      add1<--NoOp<-- add2
 *     /  \             /  \
 * data1 data2      data3 data4
 */
ComputeGraphPtr ShareGraph::BuildNoOpGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data3", "Data")->NODE("add2", "Add")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 1)->NODE("add2", "Add"));
    CHAIN(NODE("add2", "Add")->CTRL_EDGE()->NODE("noop", "NoOp")->CTRL_EDGE()->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data4 = graph->FindNode("data4");
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1, 3, 4});
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});
  AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add2->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add2, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add1", "add2"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}
/*
 *            NetOutput
 *                |
 *              nonzero
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ge::ComputeGraphPtr ShareGraph::ThirdAicpuOpGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->EDGE(0, 1)
              ->NODE("nonzero", "NonZero")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("nonzero", "NonZero"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {-1, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 3, 4}, {-1, 100, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpu);
  (void)ge::AttrUtils::SetStr(add1->GetOpDesc(), "kernelSo", "libcust_aicpu_kernel.so");
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 2, 3, 4}, {-1, 2, 3, 4});
  AddCompileResult(add1, false);

  auto nonzero = graph->FindNode("nonzero");
  nonzero->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  nonzero->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpu);
  SetNoStorage(nonzero->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(nonzero->GetOpDesc(), {1, 2, 3, 4}, {-1, 2, 3, 4});
  AttrUtils::SetInt(nonzero->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  AddCompileResult(nonzero, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"nonzero"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌───────────┐  (0,0)   ┌───────────┐
│ data1 │ ───────> │  add1  │ ───────> │ sequence │ ───────> │ sequence2 │ ───────> │ NetOutput │
└───────┘          └────────┘          └──────────┘          └───────────┘          └───────────┘
                     ∧
                     │ (0,1)
                     │
                   ┌────────┐
                   │ data2  │
                   └────────┘
 */
ge::ComputeGraphPtr ShareGraph::BuildHostCpuDataFlowGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->EDGE(0, 0)
              ->NODE("sequence", "SequenceStub")
              ->EDGE(0, 0)
              ->NODE("sequence2", "SequenceStub")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 3, 4}, {1, 100, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});
  AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add1, false);

  auto sequence = graph->FindNode("sequence");
  sequence->GetOpDesc()->MutableAllInputName() = {{"x1", 0}};
  sequence->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu);

  auto sequence2 = graph->FindNode("sequence2");
  sequence2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}};
  sequence2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"sequence2"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *
 *  reducesum1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::BuildZeroInputAicoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("reducesum1", "ReduceSum")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("reducesum1", "ReduceSum"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_ND, DT_FLOAT, {});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_ND, DT_INT32, {0});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto reducesum1 = graph->FindNode("reducesum1");
  AddCompileResult(reducesum1, false);
  reducesum1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(reducesum1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {});
  reducesum1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"reducesum1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *  netoutput
 *      |
 *    size
 *      |
 *    data
 */
ComputeGraphPtr ShareGraph::BuildSizeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", "Data")->NODE("Size", "Size")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {2, 3, 4});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"Size"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto goal_node = graph->FindNode("Size");
  auto input_desc = goal_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(0));
  input_desc->SetShape(ge::GeShape({2, 3, 4}));

  goal_node->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  goal_node->GetOpDesc()->AppendIrAttrName("dtype");
  goal_node->GetOpDesc()->MutableAllInputName() = {{"x1", 0}};

  ge::AttrUtils::SetInt(goal_node->GetOpDesc(), "dtype", ge::DT_INT64);
  return graph;
}
/*
 *         NetOutput
 *        /        \
 *    cast0        cast1
 *      |         /
 *      |    const0
 *      |  /c
 *      add0
 *      /  \
 *  data0  data1
 */
ge::ComputeGraphPtr ShareGraph::BuildCtrlToConstGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->NODE("add0", "Add")->NODE("cast0", "Cast")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->NODE("add0", "Add"));
    CTRL_CHAIN(NODE("add0", "Add")->NODE("const0", "Const"));
    CHAIN(NODE("const0", "Const")->NODE("cast1", "Cast")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  GE_ASSERT_NOTNULL(graph);

  AttrUtils::SetInt(graph->FindNode("data0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 1);
  AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "dst_type", ge::DT_INT64);
  AttrUtils::SetInt(graph->FindNode("cast1")->GetOpDesc(), "dst_type", ge::DT_INT64);
  SetConstValue<float, ge::DT_FLOAT>(graph->FindNode("const0"), {1.0, 2.0});
  ge::AttrUtils::SetTensor(graph->FindNode("const0")->GetOpDesc(), "value", CreateScalarGeTensor(0));

  auto net_output = graph->FindNode("NetOutput");
  GE_ASSERT_NOTNULL(net_output);
  net_output->GetOpDesc()->SetSrcName({"cast0", "cast1"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}
/*
 *  netoutput
 *      |
 *    rank
 *      |
 *    data1
 */
ComputeGraphPtr ShareGraph::BuildRankGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("Rank", "Rank")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {2, 3, 4});
  graph->FindNode("Rank")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"Rank"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto goal_node = graph->FindNode("Rank");
  auto input_desc = goal_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(0));
  input_desc->SetShape(ge::GeShape({2, 3, 4}));

  goal_node->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  auto &name_index = goal_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["x"] = 0;

  return graph;
}
/*  Netoutput
 *      |
 * test_no_infer
 *    /  \
 * data1 data2
 */
ComputeGraphPtr ShareGraph::BuildCompatibleInferShapeRangeGraph() {
  InferShapeFuncRegister("TestNoInferShapeRange",
                         INFER_VERIFY_FUNC(TestNoInferShapeRange, TestNoInferShapeRangeInfer));

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("test_no_infer", "TestNoInferShapeRange")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("test_no_infer", "TestNoInferShapeRange"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_INT64, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto test_no_infer = graph->FindNode("test_no_infer");
  test_no_infer->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  SetNoStorage(test_no_infer->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(test_no_infer->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  AttrUtils::SetInt(test_no_infer->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  AddCompileResult(test_no_infer, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"test_no_infer"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*

     ┌──────────────────────────────────────────────┐
     │                                              │
     │   ┌──────┐                                   │
     │   │data_i├───┐                               │
     │   └──────┘   │ ┌───┐     ┌───────────────┐   │
     │              │►│add├────►│sub_1_netoutput│   │
     │   ┌───────┐  │ └───┘     └───────────────┘   │
     │   │const_1├──┘                               │
     │   └───────┘                                  │
     │                                              │
     └─────────────────────────┬────────────────────┘
                               │
 ┌───────┐     ┌────┐     ┌────▼───┐
 │ data_1├────►│rank├────►│known_op├───┐
 └───────┘     └────┘     └────────┘   │  ┌──────────────┐
                                       ├─►│root_netoutput│
 ┌───────┐               ┌──────────┐  │  └──────────────┘
 │ data_2├──────────────►│unknown_op├──┘
 └───────┘               └────▲─────┘
                              │
   ┌──────────────────────────┴──────────────────────┐
   │                                                 │
   │  ┌──────┐                                       │
   │  │data_a├────┐                                  │
   │  └──────┘    │  ┌────┐     ┌───────────────┐    │
   │              ├─►│mul ├────►│sub_2_netoutput│    │
   │  ┌───────┐   │  └────┘     └───────────────┘    │
   │  │const_2├───┘                                  │
   │  └───────┘                                      │
   │                                                 │
   └─────────────────────────────────────────────────┘

*/
ComputeGraphPtr ShareGraph::BuildDynamicAndStaticGraph() {
  std::vector<int64_t> shape = {2, 2};  // NCHW
  // sub1
  auto data_i = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_i");
  data_i->SetOutputOffset({32});

  auto const_1 = OP_CFG("Const")
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                     .InCnt(1)
                     .OutCnt(1)
                     .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .Build("const_1");
  ge::AttrUtils::SetTensor(const_1, "value", CreateVecTorGeTensor(shape, DT_FLOAT));

  auto add = OP_CFG("Add").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add");
  add->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  add->SetOpEngineName(ge::kEngineNameAiCpu.c_str());
  add->SetInputOffset({0, 16});
  add->SetOutputOffset({32});

  auto sub_1_netoutput = OP_CFG(ge::NETOUTPUT)
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                             .InCnt(1)
                             .OutCnt(1)
                             .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Build("sub_1_netoutput");
  sub_1_netoutput->SetInputOffset({0, 16});

  // sub2
  auto data_a = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_a");

  auto const_2 = OP_CFG("Const")
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                     .InCnt(1)
                     .OutCnt(1)
                     .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .Build("const_2");
  ge::AttrUtils::SetTensor(const_2, "value", CreateVecTorGeTensor(shape, DT_FLOAT));

  auto mul = OP_CFG("Mul").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("mul");
  mul->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());

  auto sub_2_netoutput = OP_CFG(ge::NETOUTPUT)
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                             .InCnt(1)
                             .OutCnt(1)
                             .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Build("sub_2_netoutput");

  // root
  auto data_1 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_1");

  auto data_2 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 1)
                    .Build("data_2");

  auto rank = OP_CFG("Rank").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("rank");
  rank->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());

  auto known_op =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("known_op");

  auto unknown_op =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("unknown_op");

  auto root_netoutput =
      OP_CFG(ge::NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("root_netoutput");
  root_netoutput->SetSrcName({"known_op", "unknown_op"});
  root_netoutput->SetSrcIndex({0, 1});

  DEF_GRAPH(sub_1) {
    CHAIN(NODE(data_i)->NODE(add)->NODE(sub_1_netoutput));
    CHAIN(NODE(const_1)->NODE(add));
  };

  DEF_GRAPH(sub_2) {
    CHAIN(NODE(data_a)->NODE(mul)->NODE(sub_2_netoutput));
    CHAIN(NODE(const_2)->NODE(mul));
  };

  DEF_GRAPH(root) {
    CHAIN(NODE(data_1)->NODE(rank)->NODE(known_op)->NODE(root_netoutput));
    CHAIN(NODE(data_2)->NODE(unknown_op)->NODE(root_netoutput));
  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto sub_graph1 = ToGeGraph(sub_1);
  auto compute_graph1 = ge::GraphUtilsEx::GetComputeGraph(sub_graph1);
  compute_graph1->SetGraphUnknownFlag(false);
  auto net_output = compute_graph1->FindNode("sub_1_netoutput");
  net_output->GetOpDesc()->SetSrcName({"mul"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto add_node = compute_graph1->FindNode("add");
  auto op_desc = add_node->GetOpDesc();
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "_add_fake_id");
  // cust aicpu kernel
  const char kernel_bin[] = "test";
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(op_desc->GetName(), std::move(buffer));
  op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_bin_ptr);
  (void)ge::AttrUtils::SetStr(op_desc, "kernelSo", "libtest_cust.so");
  // tbe kernel
  const char dummy_kernel_bin[] = "test";
  vector<char> buffer2(dummy_kernel_bin, dummy_kernel_bin + strlen(dummy_kernel_bin));
  ge::OpKernelBinPtr dummy_kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(op_desc->GetName(), std::move(buffer2));
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, dummy_kernel_bin_ptr);
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName());

  auto sub_graph2 = ToGeGraph(sub_2);
  auto compute_graph2 = ge::GraphUtilsEx::GetComputeGraph(sub_graph2);
  compute_graph2->SetGraphUnknownFlag(true);
  auto node_mul = compute_graph2->FindNode("mul");
  (void)AttrUtils::SetBool(node_mul->GetOpDesc(), "OwnerGraphIsUnknown", true);

  auto known_node = root_graph->FindNode("known_op");
  auto unknown_node = root_graph->FindNode("unknown_op");

  SetSubGraph(root_graph, known_node, compute_graph1);
  SetSubGraph(root_graph, unknown_node, compute_graph2);

  AddCompileResult(known_node, false);
  AddCompileResult(unknown_node, false);
  return root_graph;
}
/*
 *       NetOutput
 *       |        \
 *       |       Add
 *       |0     1/ \
 *      MinimumGrad |
 *     /    |    \  |
 * data0  data1  data2
 */
ge::ComputeGraphPtr ShareGraph::BuildMinimumGradAndAddGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->NODE("mg", "MinimumGrad")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->NODE("mg", "MinimumGrad"));
    CHAIN(NODE("data2", "Data")->NODE("mg", "MinimumGrad"));

    CHAIN(NODE("mg", "MinimumGrad")->EDGE(1, 0)->NODE("add0", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add0", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  for (int32_t i = 0; i < 3; ++i) {
    auto node_name = "data" + std::to_string(i);
    auto data_node = graph->FindNode(node_name);
    AttrUtils::SetInt(data_node->GetOpDesc(), "index", i);
    SetNoStorage(data_node->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
    SetShapeRangeNoStorage(data_node->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
    data_node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  }

  auto add0 = graph->FindNode("add0");
  add0->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add0->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add0->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(add0->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  AttrUtils::SetInt(add0->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add0->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add0, false);
  add0->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto mg = graph->FindNode("mg");
  mg->GetOpDesc()->MutableAllInputName() = {{"grads", 0}, {"x1", 1}, {"x2", 2}};
  mg->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(mg->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(mg->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  AttrUtils::SetBool(mg->GetOpDesc(), "grad_x", true);
  AttrUtils::SetBool(mg->GetOpDesc(), "grad_y", true);
  AttrUtils::SetInt(mg->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(mg->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(mg, false);
  mg->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"mg", "add0"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 0});
  return graph;
}

/*
 *    data1 data2   const
 *       \    |     /
 *         concatv2
 *            |
 *         netoutput
 *
 */
ge::ComputeGraphPtr ShareGraph::ConcatV2ConstDependencyGraph() {
  // root
  std::vector<int64_t> in_shape = {2, 3, 4, 5};
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data2");

  auto concat_dim = OP_CFG("Const")
                        .TensorDesc(FORMAT_NCHW, DT_INT32, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .Build("concat_dim");
  auto concatv2 = OP_CFG("ConcatV2").TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape).InCnt(3).OutCnt(1).Build("concatv2");

  DEF_GRAPH(test) {
    CHAIN(NODE(data1)->NODE(concatv2)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(concatv2));
    CHAIN(NODE(concat_dim)->EDGE(0, 2)->NODE(concatv2));
  };

  auto graph = ToComputeGraph(test);
  auto concatv2_node = graph->FindNode("concatv2");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  concatv2_node->GetOpDesc()->MutableInputDesc(2)->SetShape(ge::GeShape());
  concatv2_node->GetOpDesc()->MutableInputDesc(2)->SetDataType(ge::DT_INT32);
  concatv2_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape({2, 6, 4, 5}));
  concatv2_node->GetOpDesc()->SetOpInferDepends({"concat_dim"});  // 用V1的方式设置值依赖
  concatv2_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputDynamic);
  concatv2_node->GetOpDesc()->AppendIrInput("concat_dim", ge::kIrInputRequired);
  concatv2_node->GetOpDesc()->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"concat_dim", 2}};
  SetConstValue<int32_t, ge::DT_INT32>(graph->FindNode("concat_dim"), {1});

  netoutput->GetOpDesc()->SetSrcName({"concatv2"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *    data1 data2   data3     datan
 *       \    |     /    ...  /
 *           IFA
 *            |
 *         netoutput
 *
 */
ge::ComputeGraphPtr ShareGraph::IFASingleGraph() {
  // root
  std::vector<int64_t> query_shape = {1, 1, 3, 2};
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, query_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data0");
  data0->SetOutputOffset({0});
  std::vector<int64_t> key_shape = {3, 2};
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data1");
  data1->SetOutputOffset({128});
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .Build("data2");
  data2->SetOutputOffset({256});
  auto data3 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 3)
                   .Build("data3");
  data3->SetOutputOffset({384});
  auto data4 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 4)
                   .Build("data4");
  data4->SetOutputOffset({512});
  auto data5 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 5)
                   .Build("data5");
  data5->SetOutputOffset({640});
  auto data6 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 6)
                   .Build("data6");
  data6->SetOutputOffset({768});
  std::vector<int64_t> mask_shape = {1};
  auto data7 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, mask_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 7)
                   .Build("data7");
  data7->SetOutputOffset({896});
  auto data8 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 8)
                   .Build("data8");
  data8->SetOutputOffset({928});
  auto data9 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 9)
                   .Build("data9");
  data9->SetOutputOffset({1056});
  auto ifa = OP_CFG("IncreFlashAttention").TensorDesc(FORMAT_NCHW, DT_FLOAT16, key_shape)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
      .InCnt(19).OutCnt(1).Build("IncreFlashAttention");

  DEF_GRAPH(test) {
    CHAIN(NODE(data0)->NODE(ifa)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(ifa));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(ifa));
    CHAIN(NODE(data3)->EDGE(0, 3)->NODE(ifa));
    CHAIN(NODE(data4)->EDGE(0, 4)->NODE(ifa));
    CHAIN(NODE(data5)->EDGE(0, 5)->NODE(ifa));
    CHAIN(NODE(data6)->EDGE(0, 6)->NODE(ifa));
    CHAIN(NODE(data7)->EDGE(0, 9)->NODE(ifa));
    CHAIN(NODE(data8)->EDGE(0, 15)->NODE(ifa));
    CHAIN(NODE(data9)->EDGE(0, 16)->NODE(ifa));
  };

  auto graph = ToComputeGraph(test);
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  auto ifa_op = ifa_node->GetOpDesc();
  AttrUtils::SetStr(ifa_op, ATTR_NAME_KERNEL_BIN_ID, "_ifa_fake_id");
  const char kernel_bin[] = "kernel_bin";
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(ifa_op->GetName(), std::move(buffer));
  ifa_op->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
  ifa_op->MutableInputDesc(0)->SetShape(ge::GeShape({1, 1, 3, 2}));
  for (size_t i = 0UL; i < 7; i++) {
    ge::TensorUtils::SetSize(*(ifa_op->MutableInputDesc(i)), 24);
  }
  ifa_op->UpdateInputDesc(7, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->UpdateInputDesc(8, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->MutableInputDesc(9)->SetShape(ge::GeShape({1}));
  ge::TensorUtils::SetSize(*(ifa_op->MutableInputDesc(9)), 4);
  ifa_op->UpdateInputDesc(10, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->UpdateInputDesc(11, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->UpdateInputDesc(12, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->UpdateInputDesc(13, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->UpdateInputDesc(14, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ge::TensorUtils::SetSize(*(ifa_op->MutableInputDesc(15)), 24);
  ge::TensorUtils::SetSize(*(ifa_op->MutableInputDesc(16)), 24);
  ifa_op->UpdateInputDesc(17, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->UpdateInputDesc(18, GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  ifa_op->MutableOutputDesc(0)->SetShape(ge::GeShape({1, 1, 3, 2}));
  ge::TensorUtils::SetSize(*(ifa_op->MutableOutputDesc(0)), 24);

  ifa_op->AppendIrInput("query", ge::kIrInputRequired);
  ifa_op->AppendIrInput("key", ge::kIrInputDynamic);
  ifa_op->AppendIrInput("value", ge::kIrInputDynamic);
  ifa_op->AppendIrInput("pse_shift", ge::kIrInputOptional);
  ifa_op->AppendIrInput("atten_mask", ge::kIrInputOptional);
  ifa_op->AppendIrInput("actual_seq_lengths", ge::kIrInputOptional);
  ifa_op->AppendIrInput("dequant_scale1", ge::kIrInputOptional);
  ifa_op->AppendIrInput("quant_scale1", ge::kIrInputOptional);
  ifa_op->AppendIrInput("dequant_scale2", ge::kIrInputOptional);
  ifa_op->AppendIrInput("quant_scale2", ge::kIrInputOptional);
  ifa_op->AppendIrInput("quant_offset2", ge::kIrInputOptional);
  ifa_op->AppendIrInput("antiquant_scale", ge::kIrInputOptional);
  ifa_op->AppendIrInput("antiquant_offset", ge::kIrInputOptional);
  ifa_op->AppendIrInput("block_table", ge::kIrInputOptional);
  ifa_op->AppendIrInput("kv_padding_size", ge::kIrInputOptional);
  ifa_op->AppendIrOutput("attention_out", ge::kIrOutputRequired);
  ifa_op->SetInputOffset({0, 128, 256, 384, 512, 640, 768, 0, 0, 896, 0, 0, 0, 0, 0, 928, 1056, 0, 0});
  ifa_op->SetOutputOffset({1184});
  ifa_op->MutableAllInputName() = {{"query", 0}, {"key0", 1}, {"key1", 2}, {"key2", 3}, {"value0", 4},
                                   {"value1", 5}, {"value2", 6}, {"pse_shift", 7}, {"atten_mask", 8},
                                   {"actual_seq_lengths", 9}, {"dequant_scale1", 10},
                                   {"quant_scale1", 11}, {"dequant_scale2", 12},
                                   {"quant_scale2", 13}, {"quant_offset2", 14},
                                   {"antiquant_scale", 15}, {"antiquant_offset", 16},
                                   {"block_table", 17}, {"kv_padding_size", 18}};
  ifa_op->MutableAllOutputName() = {{"attention_out", 0}};

  netoutput->GetOpDesc()->SetSrcName({"IncreFlashAttention"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  netoutput->GetOpDesc()->SetInputOffset({1184});
  return graph;
}

/*
 *    data1 data2
 *       \    |
 *       CTCBeamSearchDecoder
 *            |
 *         netoutput
 *
 */
ge::ComputeGraphPtr ShareGraph::CTCBeamSearchDecoderSingleGraph() {
  // root
  std::vector<int64_t> shape = {1, 1, 3, 2};
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data1");
  auto ctc = OP_CFG("CTCBeamSearchDecoder")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
      .InCnt(2).OutCnt(4)
      .Build("CTCBeamSearchDecoder");

  auto add_n = OP_CFG("AddN")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
      .InCnt(4).OutCnt(1).Build("AddN");

  DEF_GRAPH(test) {
    CHAIN(NODE(data0)->NODE(ctc)->NODE(add_n)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(ctc)->EDGE(1, 1)->NODE(add_n));
    CHAIN(NODE(ctc)->EDGE(2, 2)->NODE(add_n));
    CHAIN(NODE(ctc)->EDGE(3, 3)->NODE(add_n));
  };
  auto graph = ToComputeGraph(test);
  auto ctc_node = graph->FindNode("CTCBeamSearchDecoder");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  auto ctc_op = ctc_node->GetOpDesc();
  for (size_t i = 0UL; i < 4; i++) {
    ge::TensorUtils::SetSize(*(ctc_op->MutableOutputDesc(i)), 24);
  }
  for (size_t i = 0UL; i < 2; i++) {
    ge::TensorUtils::SetSize(*(ctc_op->MutableInputDesc(i)), 24);
  }
  ctc_op->AppendIrInput("inputs", ge::kIrInputRequired);
  ctc_op->AppendIrInput("sequence_length", ge::kIrInputRequired);
  ctc_op->AppendIrOutput("decoded_indices", ge::kIrOutputDynamic);
  ctc_op->AppendIrOutput("decoded_values", ge::kIrOutputDynamic);
  ctc_op->AppendIrOutput("decoded_shape", ge::kIrOutputDynamic);
  ctc_op->MutableAllInputName() = {{"inputs", 0}, {"sequence_length", 1}};
  ctc_op->MutableAllOutputName() = {{"decoded_indices0", 0}, {"decoded_indices1", 1},
                                    {"decoded_shape0", 2}, {"decoded_shape1", 3}};

  netoutput->GetOpDesc()->SetSrcName({"AddN"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *    data1 data2   const
 *       \    |     /
 *      GroupedMatMulAllReduce
 *            |
 *         netoutput
 *
 */
ge::ComputeGraphPtr ShareGraph::GroupedMatMulAllReduceSingleGraph() {
  // root
  std::vector<int64_t> shape = {1, 1, 3, 2};
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .Build("data2");
  auto data3 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 3)
                   .Build("data3");
  std::vector<int64_t> list_shape = {1};
  auto data4 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, list_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 4)
                   .Build("data7");

  auto matmul = OP_CFG("GroupedMatMulAllReduce")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape).InCnt(5).OutCnt(2)
      .Build("GroupedMatMulAllReduce");

  auto add_n = OP_CFG("AddN")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
      .InCnt(2).OutCnt(1).Build("AddN");

  DEF_GRAPH(test) {
    CHAIN(NODE(data0)->NODE(matmul)->NODE(add_n)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(matmul)->EDGE(1, 1)->NODE(add_n));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(matmul));
    CHAIN(NODE(data3)->EDGE(0, 3)->NODE(matmul));
    CHAIN(NODE(data4)->EDGE(0, 4)->NODE(matmul));
  };

  auto graph = ToComputeGraph(test);
  auto matmul_node = graph->FindNode("GroupedMatMulAllReduce");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  auto matmul_op = matmul_node->GetOpDesc();
  matmul_op->MutableInputDesc(4)->SetShape(ge::GeShape({1}));
  matmul_op->MutableOutputDesc(0)->SetShape(ge::GeShape({1, 1, 3, 2}));
  matmul_op->MutableOutputDesc(1)->SetShape(ge::GeShape({1, 1, 3, 2}));
  for (size_t i = 0UL; i < 4; i++) {
    ge::TensorUtils::SetSize(*(matmul_op->MutableInputDesc(i)), 24);
  }
  ge::TensorUtils::SetSize(*(matmul_op->MutableInputDesc(4)), 4);
  for (size_t i = 0UL; i < 2; i++) {
    ge::TensorUtils::SetSize(*(matmul_op->MutableOutputDesc(i)), 24);
  }
  matmul_op->AppendIrInput("x", ge::kIrInputDynamic);
  matmul_op->AppendIrInput("weight", ge::kIrInputDynamic);
  matmul_op->AppendIrInput("bias", ge::kIrInputDynamic);
  matmul_op->AppendIrInput("group_list", ge::kIrInputOptional);
  matmul_op->AppendIrOutput("y", ge::kIrOutputDynamic);
  matmul_op->MutableAllInputName() = {{"x0", 0}, {"x1", 1},  {"weight0", 2}, {"weight1", 3}, {"group_list", 4}};
  matmul_op->MutableAllOutputName() = {{"y0", 0}, {"y1", 1}};

  netoutput->GetOpDesc()->SetSrcName({"IncreFlashAttention"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *    data0  data1 data2   data3
 *       \    |     /      /
 *         Batch
 *            |
 *         netoutput
 *
 */
ge::ComputeGraphPtr ShareGraph::BatchSingleGraph() {
  // root
  std::vector<int64_t> shape = {1, 1, 3, 2};
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data0");
  data0->SetOutputOffset({0});
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data1");
  data1->SetOutputOffset({128});
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .Build("data2");
  data2->SetOutputOffset({256});
  auto data3 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 3)
                   .Build("data3");
  data3->SetOutputOffset({384});
  auto batch = OP_CFG("Batch")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
      .InCnt(4).OutCnt(4)
      .Build("Batch");

  auto add_n = OP_CFG("AddN")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
      .InCnt(4).OutCnt(1).Build("AddN");
  add_n->SetInputOffset({512, 640, 768, 896});
  add_n->SetOutputOffset({1024});
  DEF_GRAPH(test) {
    CHAIN(NODE(data0)->NODE(batch)->NODE(add_n)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(batch)->EDGE(1, 1)->NODE(add_n));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(batch)->EDGE(2, 2)->NODE(add_n));
    CHAIN(NODE(data3)->EDGE(0, 3)->NODE(batch)->EDGE(3, 3)->NODE(add_n));
  };

  auto graph = ToComputeGraph(test);
  auto batch_node = graph->FindNode("Batch");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  auto batch_op = batch_node->GetOpDesc();

  AttrUtils::SetStr(batch_op, ATTR_NAME_KERNEL_BIN_ID, "_batch_fake_id");
  const char kernel_bin[] = "kernel_bin";
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
  batch_op->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);

  for (size_t i = 0UL; i < 4; i++) {
    ge::TensorUtils::SetSize(*(batch_op->MutableOutputDesc(i)), 24);
  }
  for (size_t i = 0UL; i < 4; i++) {
    ge::TensorUtils::SetSize(*(batch_op->MutableInputDesc(i)), 24);
  }

  batch_op->AppendIrInput("x_tensors", ge::kIrInputDynamic);
  batch_op->AppendIrOutput("y_tensors", ge::kIrOutputDynamic);
  batch_op->AppendIrOutput("y_index", ge::kIrOutputRequired);
  batch_op->AppendIrOutput("y_id", ge::kIrOutputRequired);
  batch_op->MutableAllInputName() = {{"x_tensors0", 0}, {"x_tensors1", 1},
                                     {"x_tensors2", 2}, {"x_tensors3", 3}};
  batch_op->MutableAllOutputName() = {{"y_tensors0", 0}, {"y_tensors1", 1},
                                      {"y_index", 2}, {"y_id", 3}};
  batch_op->SetInputOffset({0, 128, 256, 384});
  batch_op->SetOutputOffset({512, 640, 768, 896});

  netoutput->GetOpDesc()->SetSrcName({"AddN"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  netoutput->GetOpDesc()->SetInputOffset({1024});
  return graph;
}

/*                   data3
                       |
                     cast
                      |
 *    data1 data2  gather_shapes
 *       \    |     /
 *         concatv2
 *            |
 *         netoutput
 *
 */
ge::ComputeGraphPtr ShareGraph::ConcatV2ValueDependencyGraph() {
  // root
  std::vector<int64_t> in_shape = {2, 3, 4, 5};
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("data2");
  auto data3 = OP_CFG("Data")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .Build("data3");
  auto cast = OP_CFG("Cast")
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                  .Attr("dst_type", DT_FLOAT)
                  .InCnt(1)
                  .OutCnt(1)
                  .Build("cast");
  auto gather_shapes = OP_CFG("GatherShapes")
                           .TensorDesc(FORMAT_NCHW, DT_INT32, in_shape)
                           .Attr("axes", {0})
                           .InCnt(1)
                           .OutCnt(1)
                           .Build("gather_shapes");
  auto concatv2 = OP_CFG("ConcatV2").TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape).InCnt(3).OutCnt(1).Build("concatv2");

  DEF_GRAPH(test) {
    CHAIN(NODE(data1)->NODE(concatv2)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(concatv2));
    CHAIN(NODE(data3)->NODE(cast)->NODE(gather_shapes)->EDGE(0, 2)->NODE(concatv2));
  };

  auto graph = ToComputeGraph(test);
  auto concatv2_node = graph->FindNode("concatv2");
  auto cast_node = graph->FindNode("cast");
  auto gather_shapes_node = graph->FindNode("gather_shapes");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  concatv2_node->GetOpDesc()->MutableInputDesc(2)->SetShape(ge::GeShape());
  concatv2_node->GetOpDesc()->MutableInputDesc(2)->SetDataType(ge::DT_INT32);
  concatv2_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape({2, -1, 4, 5}));
  // 输出range没有0，aicore lowering的时候可以不做空tensor校验
  std::vector<std::pair<int64_t, int64_t>> out_shape_range = {{2, 2}, {2, 100}, {4, 4}, {5, 5}};
  concatv2_node->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(out_shape_range);
  concatv2_node->GetOpDesc()->SetOpInferDepends({"concat_dim"});  // 用V1的方式设置值依赖
  concatv2_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputDynamic);
  concatv2_node->GetOpDesc()->AppendIrInput("concat_dim", ge::kIrInputRequired);
  concatv2_node->GetOpDesc()->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"concat_dim", 2}};
  concatv2_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AddCompileResult(concatv2_node, false);
  concatv2_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  // SetConstValue<int32_t, ge::DT_INT32>(graph->FindNode("concat_dim"), {1});

  // set cast shape and IR info
  SetNoStorage(cast_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT16, {-1, -1 - 1, -1});
  AddCompileResult(cast_node, false);
  cast_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
  cast_node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  cast_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  gather_shapes_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputDynamic);
  gather_shapes_node->GetOpDesc()->MutableAllInputName() = {{"x0", 0}};
  gather_shapes_node->GetOpDesc()->AppendIrAttrName("axes");
  gather_shapes_node->GetOpDesc()->AppendIrAttrName("dtype");
  ge::AttrUtils::SetListListInt(gather_shapes_node->GetOpDesc(), "axes", {{0, 0}});
  ge::AttrUtils::SetInt(gather_shapes_node->GetOpDesc(), "dtype", ge::DT_INT64);
  gather_shapes_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(ge::GeShape(in_shape));
  gather_shapes_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  netoutput->GetOpDesc()->SetSrcName({"concatv2"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  return graph;
}
/*
 *    data1 data2   const
 *       \    |     /
 *         concatv2
 *          /   \
 *         /   size
 *         \    /
 *        netoutput
 */
ge::ComputeGraphPtr ShareGraph::ConcatV2MultiOutNodesGraph() {
  // root
  std::vector<int64_t> in_shape = {2, 3, 4, 5};
  auto data_1 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_1");
  auto data_2 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 1)
                    .Build("data_2");

  auto concat_dim = OP_CFG("Const")
                        .TensorDesc(FORMAT_NCHW, DT_INT32, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .Build("concat_dim");
  auto concatv2 = OP_CFG("ConcatV2").TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape).InCnt(3).OutCnt(1).Build("concatv2");
  auto size = OP_CFG("Size").TensorDesc(FORMAT_NCHW, DT_FLOAT16, in_shape).InCnt(1).OutCnt(1).Build("size");

  DEF_GRAPH(test) {
    CHAIN(NODE(data_1)->NODE(concatv2)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(concatv2)->EDGE(0, 1)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE(concat_dim)->EDGE(0, 2)->NODE(concatv2));
  };

  auto graph = ToComputeGraph(test);
  auto concatv2_node = graph->FindNode("concatv2");
  auto netoutput = graph->FindNode("netoutput");

  // set concatv2 shape and IR info
  concatv2_node->GetOpDesc()->MutableInputDesc(2)->SetShape(ge::GeShape());
  concatv2_node->GetOpDesc()->MutableInputDesc(2)->SetDataType(ge::DT_INT32);
  concatv2_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape({2, 6, 4, 5}));
  concatv2_node->GetOpDesc()->SetOpInferDepends({"concat_dim"});  // 用V1的方式设置值依赖
  concatv2_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputDynamic);
  concatv2_node->GetOpDesc()->AppendIrInput("concat_dim", ge::kIrInputRequired);
  concatv2_node->GetOpDesc()->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"concat_dim", 2}};
  SetConstValue<int32_t, ge::DT_INT32>(graph->FindNode("concat_dim"), {1});

  netoutput->GetOpDesc()->SetSrcName({"concatv2", "size"});
  netoutput->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

//                                    ┌──────────────────────────────────┐
//                                    │                                  │
//                                    │ ┌──────┐   ┌─────────────────┐   │
//                         -----------│ │ Data ├───►  RequireDevice  │   │
//                         -          │ └──────┘   └─────────────────┘   │
//                         -          │                                  │
//                         -          └──────────────────────────────────┘
// ┌───────────┐      ┌──────────┐
// │ GenDevice ├─────►│   If     │
// └───────────┘      └──────────┘     ┌─────────────────────────────────┐
//                         ▲ -         │                                 │
// ┌───────────┐           │ -         │ ┌──────┐   ┌──────────────┐     │
// │   Data    ├───────────┘ -         │ │ Data ├───► RequireHost  │     │
// └───────────┘             ----------│ └──────┘   └──────────────┘     │
//                                     │                                 │
//                                     └─────────────────────────────────┘
ComputeGraphPtr ShareGraph::IfWithDifferentPlacementSubgraph() {
  auto then_graph = []() {
    auto graph = std::make_shared<ge::ComputeGraph>("then");
    auto data = gert::NodeBuilder("data", ge::DATA)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Output(ge::DT_FLOAT)
                    .Build(graph);
    auto foo =
        gert::NodeBuilder("foo", "IfOrCaseNodeConverterUT_ReqDevice").Input(data).Output(ge::DT_FLOAT).Build(graph);
    gert::NodeBuilder("output", ge::NETOUTPUT).Input(foo).Build(graph);
    return graph;
  }();
  auto else_graph = []() {
    auto graph = std::make_shared<ge::ComputeGraph>("else");
    auto data = gert::NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Output(ge::DT_FLOAT)
                    .Build(graph);
    auto foo =
        gert::NodeBuilder("bar", "IfOrCaseNodeConverterUT_ReqHost").Input(data).Output(ge::DT_FLOAT).Build(graph);
    gert::NodeBuilder("output1", ge::NETOUTPUT).Input(foo).Build(graph);
    return graph;
  }();

  auto graph = std::make_shared<ge::ComputeGraph>("main");
  auto data = gert::NodeBuilder("input", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output(ge::DT_INT32).Build(graph);
  auto gen_device =
      gert::NodeBuilder("gen_device", "IfOrCaseNodeConverterUT_GenDevice").Output(ge::DT_FLOAT).Build(graph);
  auto if_op = gert::NodeBuilder("if", ge::IF)
                   .Input(data)
                   .Input(gen_device)
                   .Attr("then_branch", then_graph)
                   .Attr("else_branch", else_graph)
                   .Output(ge::DT_FLOAT)
                   .Build(graph);
  gert::NodeBuilder("output2", ge::NETOUTPUT).Input(if_op).Build(graph);
  return graph;
}

/*
 *
 * netoutput
 *   |
 * FileConstant
 */
ComputeGraphPtr ShareGraph::BuildFileConstantGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("FileConstant", "FileConstant")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"FileConstant"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetInputOffset({0});

  auto FileConstant = graph->FindNode("FileConstant");
  SetNoStorage(FileConstant->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {5, 5});

  FileConstant->GetOpDesc()->AppendIrAttrName("file_path");
  FileConstant->GetOpDesc()->AppendIrAttrName("file_id");
  FileConstant->GetOpDesc()->AppendIrAttrName("shape");
  FileConstant->GetOpDesc()->AppendIrAttrName("dtype");
  FileConstant->GetOpDesc()->SetOutputOffset({0});

  // set attr
  std::vector<int64_t> shape = {5, 5};
  std::vector<int64_t> original_shape = {1, 5, 5};
  ge::AttrUtils::SetInt(FileConstant->GetOpDesc(), "offset", 0);
  ge::AttrUtils::SetInt(FileConstant->GetOpDesc(), "length", 0);
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "location", "");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_path", "test_weight.bin");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_id", "");
  ge::AttrUtils::SetDataType(FileConstant->GetOpDesc(), "dtype", DT_INT32);
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "original_shape", original_shape);
  ge::TensorUtils::SetSize(*FileConstant->GetOpDesc()->MutableOutputDesc(0), 100);
  return graph;
}

/*
 *                   netoutput
 *               /              \
 *    file_constant_0------>file_constant_1
 */
ComputeGraphPtr ShareGraph::Build2FileConstantWithCtrlEdgeGraph() {
   DEF_GRAPH(g1) {
    CHAIN(NODE("FileConstant0", "FileConstant")->NODE("NetOutput", "NetOutput"));
	CHAIN(NODE("FileConstant1", "FileConstant")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
	CHAIN(NODE("FileConstant0", "FileConstant")->CTRL_EDGE()->NODE("FileConstant1", "FileConstant"));
  };
  auto graph = ToComputeGraph(g1);
  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"FileConstant0", "FileConstant1"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});

  auto FileConstant0 = graph->FindNode("FileConstant0");
  auto FileConstant1 = graph->FindNode("FileConstant1");
  std::vector<NodePtr> nodes{FileConstant0, FileConstant1};

  for (auto &FileConstant : nodes) {
	  SetNoStorage(FileConstant->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {5, 5});
	  FileConstant->GetOpDesc()->AppendIrAttrName("file_path");
	  FileConstant->GetOpDesc()->AppendIrAttrName("file_id");
	  FileConstant->GetOpDesc()->AppendIrAttrName("shape");
	  FileConstant->GetOpDesc()->AppendIrAttrName("dtype");

	  // set attr
	  std::vector<int64_t> shape = {5, 5};
	  std::vector<int64_t> original_shape = {1, 5, 5};
	  ge::AttrUtils::SetInt(FileConstant->GetOpDesc(), "offset", 0);
	  ge::AttrUtils::SetInt(FileConstant->GetOpDesc(), "length", 0);
	  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "location", "");
	  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_path", "test_weight.bin");
	  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_id", "");
	  ge::AttrUtils::SetDataType(FileConstant->GetOpDesc(), "dtype", DT_INT32);
	  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);
	  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "original_shape", original_shape);
  }
  return graph;
}

/*
 * ┌──────────────┐   ┌──────────────────────┐
 * │  FakeGetNext │   │ FakeCollectAllInputs │
 * └──────┬───────┘   └───────────┬──────────┘
 *  ┌─────┴─────┐         ┌───────┴───────┐
 *  │  Stage_1  ├─────────►    Stage_2    │
 *  └───────────┘         └───────────────┘
 */
ComputeGraphPtr ShareGraph::Build2StageGraph() {
  auto graph = std::make_shared<ge::ComputeGraph>("pipeline");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output(ge::DT_INT32, {1}).Build(graph);

  auto stage1_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage1");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto generator = NodeBuilder("generator", "FakeGetNext").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).Input(generator).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage2_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage2");
    auto data = NodeBuilder("data2", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto add = NodeBuilder("collect", "FakeCollectAllInputs").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(add).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage1_graph = stage1_builder();
  stage1_graph->SetGraphUnknownFlag(true);
  auto stage1 = NodeBuilder("stage_node1", ge::PARTITIONEDCALL)
                    .Input(data)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage1_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 0)
                    .Build(graph);

  auto stage2_graph = stage2_builder();
  stage2_graph->SetGraphUnknownFlag(true);
  auto stage2 = NodeBuilder("stage_node2", ge::PARTITIONEDCALL)
                    .Input(stage1)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage2_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 1)
                    .Build(graph);

  auto output = NodeBuilder("output", ge::NETOUTPUT).Input(stage2).Build(graph);
  GE_DUMP(graph, "Graph_2stage_pipeline_systemtest");
  return graph;
}

/**
 *                      ┌──────────────────────┐
 *                      │ FakeCollectAllInputs │
 * ┌──────────────┐     └───────────┬──────────┘
 * │  FakeGetNext │                 │
 * └──────┬───────┘         ┌───────┴───────┐
 *        │           ┌─────►    Stage_2    │
 *  ┌─────┴─────┐     │     └───────────────┘
 *  │  Stage_1  ├─────┤     ┌───────────────┐
 *  └───────────┘     └─────►    Stage_3    │
 *                          └───────┬───────┘
 *                       ┌──────────┴───────────┐
 *                       │ FakeCollectAllInputs │
 *                       └──────────────────────┘
 */
ComputeGraphPtr ShareGraph::Build1to2StageGraph() {
  auto graph = std::make_shared<ge::ComputeGraph>("pipeline");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output(ge::DT_INT32, {1}).Build(graph);

  auto stage1_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage1");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto generator = NodeBuilder("generator", "FakeGetNext").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).Input(generator).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage2_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage2");
    auto data = NodeBuilder("data2", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto add = NodeBuilder("collect1", "FakeCollectAllInputs").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(add).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage3_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage3");
    auto data = NodeBuilder("data3", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto add = NodeBuilder("collect2", "FakeCollectAllInputs").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output3", ge::NETOUTPUT).Input(add).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage1_graph = stage1_builder();
  stage1_graph->SetGraphUnknownFlag(true);
  auto stage1 = NodeBuilder("stage_node1", ge::PARTITIONEDCALL)
                    .Input(data)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage1_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 0)
                    .Build(graph);

  auto stage2_graph = stage2_builder();
  stage2_graph->SetGraphUnknownFlag(true);
  auto stage2 = NodeBuilder("stage_node2", ge::PARTITIONEDCALL)
                    .Input(stage1)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage2_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 1)
                    .Build(graph);

  auto stage3_graph = stage3_builder();
  stage3_graph->SetGraphUnknownFlag(true);
  auto stage3 = NodeBuilder("stage_node3", ge::PARTITIONEDCALL)
                    .Input(stage1)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage3_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 2)
                    .Build(graph);

  auto output = NodeBuilder("output", ge::NETOUTPUT).Input(stage2).Input(stage3).Build(graph);
  GE_DUMP(graph, "Graph_1to2_stage_pipeline_systemtest");
  return graph;
}

/**
 * 用例描述：测试两个串联Stage流水100次执行，同时存在Stage0的输出  同时给到多个Stage1的输入，验证输出Tensor的值符合预期
 * & Stage每个Step的输出符合预期 ┌──────────────┐     ┌──────────────────────┐ │  FakeGetNext │     │
 * FakeCollectAllInputs │ └──────┬───────┘     └───────────┬──────────┘ ┌─────┴─────┐   ┌──────►┌───────┴───────┐ │
 * Stage_1  ├───├──────►    Stage_2     │ └───────────┘   └──────►└───────────────┘
 */
ComputeGraphPtr ShareGraph::Build2StageWith1ToNGraph() {
  auto graph = std::make_shared<ge::ComputeGraph>("pipeline");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output(ge::DT_INT32, {1}).Build(graph);

  auto stage1_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage1");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto generator = NodeBuilder("generator", "FakeGetNext").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).Input(generator).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage2_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage2");
    auto data = NodeBuilder("data2", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto data1 = NodeBuilder("data3", ge::DATA)
                     .Attr(ge::ATTR_NAME_INDEX, 1)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                     .Output(ge::DT_INT32, {1})
                     .Build(stage_graph);
    auto data2 = NodeBuilder("data4", ge::DATA)
                     .Attr(ge::ATTR_NAME_INDEX, 2)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 2)
                     .Output(ge::DT_INT32, {1})
                     .Build(stage_graph);
    auto add = NodeBuilder("collect", "FakeCollectAllInputs")
                   .Input(data)
                   .Input(data1)
                   .Input(data2)
                   .Output(ge::DT_INT32, {1})
                   .Output(ge::DT_INT32, {1})
                   .Output(ge::DT_INT32, {1})
                   .Build(stage_graph);
    auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(add).Input(add, 1).Input(add, 2).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(2), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
    return stage_graph;
  };

  auto stage1_graph = stage1_builder();
  stage1_graph->SetGraphUnknownFlag(true);
  auto stage1 = NodeBuilder("stage_node1", ge::PARTITIONEDCALL)
                    .Input(data)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage1_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 0)
                    .Build(graph);

  auto stage2_graph = stage2_builder();
  stage2_graph->SetGraphUnknownFlag(true);
  auto stage2 = NodeBuilder("stage_node2", ge::PARTITIONEDCALL)
                    .Input(stage1)
                    .Input(stage1)
                    .Input(stage1)
                    .Output(ge::DT_INT32, {1})
                    .Output(ge::DT_INT32, {1})
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage2_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 1)
                    .Build(graph);

  auto output = NodeBuilder("output", ge::NETOUTPUT).Input(stage2).Input(stage2, 1).Input(stage2, 2).Build(graph);
  GE_DUMP(graph, "Graph_2stage_pipeline_systemtest");
  return graph;
}

/**
 * ┌──────────────┐
 * │  FakeGetNext │       ┌──────────────────────┐
 * └──────┬───────┘       │ FakeCollectAllInputs │
 *  ┌─────┴─────┐         └───────────┬──────────┘
 *  │  Stage_1  ├───────┐     ┌───────┴───────┐
 *  └───────────┘       ├─────►    Stage_3    │
 *  ┌───────────┐       │     └───────────────┘
 *  │  Stage_2  ├───────┘
 *  └─────┬─────┘
 * ┌──────┴───────┐
 * │  FakeGetNext │
 * └──────────────┘
 */
ComputeGraphPtr ShareGraph::Build2to1StageGraph() {
  auto graph = std::make_shared<ge::ComputeGraph>("pipeline");
  auto data1 = NodeBuilder("input1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output(ge::DT_INT32, {1}).Build(graph);
  auto data2 = NodeBuilder("input2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output(ge::DT_INT32, {1}).Build(graph);

  auto stage1_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage1");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto generator = NodeBuilder("generator1", "FakeGetNext").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).Input(generator).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage2_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage2");
    auto data = NodeBuilder("data2", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output(ge::DT_INT32, {1})
                    .Build(stage_graph);
    auto generator = NodeBuilder("generator2", "FakeGetNext").Input(data).Output(ge::DT_INT32, {1}).Build(stage_graph);
    auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(generator).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    return stage_graph;
  };

  auto stage3_builder = []() {
    auto stage_graph = std::make_shared<ge::ComputeGraph>("stage3");
    auto data1 = NodeBuilder("data3", ge::DATA)
                     .Attr(ge::ATTR_NAME_INDEX, 0)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                     .Output(ge::DT_INT32, {1})
                     .Build(stage_graph);
    auto data2 = NodeBuilder("data4", ge::DATA)
                     .Attr(ge::ATTR_NAME_INDEX, 1)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                     .Output(ge::DT_INT32, {1})
                     .Build(stage_graph);
    auto add = NodeBuilder("collect", "FakeCollectAllInputs")
                   .Input(data1)
                   .Input(data2)
                   .Output(ge::DT_INT32, {1})
                   .Output(ge::DT_INT32, {1})
                   .Build(stage_graph);
    auto output = NodeBuilder("output3", ge::NETOUTPUT).Input(add, 0).Input(add, 1).Build(stage_graph);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    return stage_graph;
  };

  auto stage1_graph = stage1_builder();
  stage1_graph->SetGraphUnknownFlag(true);
  auto stage1 = NodeBuilder("stage_node1", ge::PARTITIONEDCALL)
                    .Input(data1)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage1_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 0)
                    .Build(graph);

  auto stage2_graph = stage2_builder();
  stage2_graph->SetGraphUnknownFlag(true);
  auto stage2 = NodeBuilder("stage_node2", ge::PARTITIONEDCALL)
                    .Input(data2)
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage2_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 1)
                    .Build(graph);

  auto stage3_graph = stage3_builder();
  stage3_graph->SetGraphUnknownFlag(true);
  auto stage3 = NodeBuilder("stage_node3", ge::PARTITIONEDCALL)
                    .Input(stage1)
                    .Input(stage2)
                    .Output(ge::DT_INT32, {1})
                    .Output(ge::DT_INT32, {1})
                    .Attr("f", stage3_graph)
                    .Attr(ge::ATTR_STAGE_LEVEL, 2)
                    .Build(graph);

  auto output = NodeBuilder("output", ge::NETOUTPUT).Input(stage3, 0).Input(stage3, 1).Build(graph);
  GE_DUMP(graph, "Graph_2to1_stage_pipeline_systemtest");
  return graph;
}

//                        ┌────────┐
//                        │ data5  │
//                        └────────┘
//                        │
//                        │ (0,4)
//                        ∨
//  ┌───────┐  (0,0)   ┌──────────────────┐  (0,0)   ┌───────────┐
//  │ data1 │ ───────> │                  │ ───────> │ NetOutput │
//  └───────┘          │      fake1       │          └───────────┘
//  ┌───────┐  (0,3)   │                  │  (0,5)   ┌───────────┐
//  │ data4 │ ───────> │                  │ <─────── │   data6   │
//  └───────┘          └──────────────────┘          └───────────┘
//                      ∧         ∧
//                      │ (0,1)   │ (0,2)
//                      │         │
//                      ┌────────┐┌────────┐
//                      │ data2  ││ data3  │
//                      └────────┘└────────┘
ComputeGraphPtr ShareGraph::BuildFakeNodeGraphWithMultipleInput() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("fake1", "Fake")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("fake1", "Fake"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 2)->NODE("fake1", "Fake"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 3)->NODE("fake1", "Fake"));
    CHAIN(NODE("data5", "Data")->EDGE(0, 4)->NODE("fake1", "Fake"));
    CHAIN(NODE("data6", "Data")->EDGE(0, 5)->NODE("fake1", "Fake"));
  };
  auto graph = ge::ToComputeGraph(g1);
  const std::set<std::string> data_name = {"data1", "data2", "data3", "data4", "data5", "data6"};
  int32_t index = 0;
  for (auto &data : data_name) {
    auto data_node = graph->FindNode(data);
    ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", index);
    SetNoStorage(data_node->GetOpDesc(), ge::FORMAT_ND, ge::DT_FLOAT, {-1});
    SetShapeRangeNoStorage(data_node->GetOpDesc(), {1}, {100});
    data_node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
    ++index;
  }

  auto fake1 = graph->FindNode("fake1");
  fake1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}, {"x3", 2}, {"x4", 3}, {"x5", 4}, {"x6", 5}};
  fake1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(fake1->GetOpDesc(), ge::FORMAT_ND, ge::DT_FLOAT, {-1, -1});
  SetShapeRangeNoStorage(fake1->GetOpDesc(), {1}, {100000});
  ge::AttrUtils::SetInt(fake1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  ge::AttrUtils::SetStr(fake1->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(fake1, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, ge::DT_FLOAT, {-1});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1}, {100000});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"add1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *   NetOutput
 *       |
 *     many-add
 *     /   \
 * data1   data2
 */
ge::ComputeGraphPtr ShareGraph::BuildLotsOfNodes(size_t node_num) {
  auto graph = std::make_shared<ge::ComputeGraph>("g");

  auto op_desc = std::make_shared<ge::OpDesc>("data1", "Data");
  AttrUtils::SetInt(op_desc, "index", 0);
  SetNoStorage(op_desc, ge::FORMAT_ND, DT_FLOAT, {-1, -1});
  SetShapeRangeNoStorage(op_desc, {1, 1024}, {1, 1024});
  op_desc->MutableAllInputName() = {{"x", 0}};
  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  auto data0 = graph->AddNode(op_desc);

  op_desc = std::make_shared<ge::OpDesc>("data2", "Data");
  AttrUtils::SetInt(op_desc, "index", 1);
  SetNoStorage(op_desc, ge::FORMAT_ND, DT_FLOAT, {-1, -1});
  SetShapeRangeNoStorage(op_desc, {1, 1024}, {1, 1024});
  op_desc->MutableAllInputName() = {{"x", 0}};
  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  auto data1 = graph->AddNode(op_desc);

  std::vector<ge::NodePtr> add_nodes;
  for (size_t i = 0U; i < node_num; ++i) {
    op_desc = std::make_shared<ge::OpDesc>("add_" + std::to_string(i), "Add");
    GE_ASSERT_GRAPH_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
    GE_ASSERT_GRAPH_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
    GE_ASSERT_GRAPH_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));

    op_desc->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
    op_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
    SetNoStorage(op_desc, ge::FORMAT_ND, DT_FLOAT, {-1, -1});
    SetShapeRangeNoStorage(op_desc, {1, 1024}, {1, 1024});
    AttrUtils::SetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
    AttrUtils::SetStr(op_desc, "_kernel_bin_id", "te_add_12345");
    auto add_node = graph->AddNode(op_desc);
    AddCompileResult(add_node, false);
    add_nodes.emplace_back(add_node);
  }

  std::vector<std::string> src_names;
  std::vector<int64_t> src_indexes;
  op_desc = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  for (size_t i = 0U; i < node_num; ++i) {
    op_desc->AddInputDesc(ge::GeTensorDesc());
    SetNoStorage(op_desc, ge::FORMAT_ND, DT_FLOAT, {-1, -1});
    SetShapeRangeNoStorage(op_desc, {1, 1024}, {1, 1024});
    src_names.emplace_back("add_" + std::to_string(i));
    src_indexes.emplace_back(0);
  }
  op_desc->SetSrcName(src_names);
  op_desc->SetSrcIndex(src_indexes);
  graph->AddNode(op_desc);
  return graph;
}

/*
 *     NetOutput
 *         |
 *  TensorListPopBack
 *         |        \
 *         |       data4
 *  TensorListPushBack
 *         |        \
 *         |       data3
 *  EmptyTensorList
 *     /      \
 *  data1     data2
 */
ComputeGraphPtr ShareGraph::TensorListGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("EmptyTensorList1", "EmptyTensorList")
              ->NODE("TensorListPushBack1", "TensorListPushBack")
              ->NODE("TensorListPopBack1", "TensorListPopBack")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("EmptyTensorList1", "EmptyTensorList"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("TensorListPushBack1", "TensorListPushBack"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 1)->NODE("TensorListPopBack1", "TensorListPopBack"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data4 = graph->FindNode("data4");
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1});
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto EmptyTensorList1 = graph->FindNode("EmptyTensorList1");
  EmptyTensorList1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);

  auto TensorListPushBack1 = graph->FindNode("TensorListPushBack1");
  TensorListPushBack1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);

  auto TensorListPopBack1 = graph->FindNode("TensorListPopBack1");
  TensorListPopBack1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"TensorListPopBack1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetNoStorage(net_output->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1});
  return graph;
}

/*
 *
 * netoutput
 *   |
 * aippData1
 */
ComputeGraphPtr ShareGraph::BuildAippDataGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("aippData1", "AippData")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto aippData1 = graph->FindNode("aippData1");

  ge::AttrUtils::SetInt(aippData1->GetOpDesc(), "index", 0);  // set no match data index

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"aippData1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetInputOffset({0});
  return graph;
}
/*
 *
 *  NetOutput
 *      |
 *  TestDTString(DT_STRING)
 *     /  \
 *  data1 data2
 */
ComputeGraphPtr ShareGraph::AicpuOpWithDTSTRINGGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("asString1", "AsString")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("asString1", "AsString"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {-2});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {-2});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto asString1 = graph->FindNode("asString1");
  asString1->GetOpDesc()->MutableAllInputName() = {{"x0", 0}, {"x1", 1}};
  asString1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpu);
  SetNoStorage(asString1->GetOpDesc(), ge::FORMAT_ND, DT_STRING, {-2});
  AttrUtils::SetInt(asString1->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(asString1, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"asString1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetNoStorage(net_output->GetOpDesc(), ge::FORMAT_ND, DT_STRING, {});
  return graph;
}

/*
 *     NetOutput
 *        |
 *       add
 *      /   \
 *   data1 data2
 */
ComputeGraphPtr ShareGraph::BuildBlockGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);
  SetGraphOutShapeRange(graph);
  AttrUtils::SetInt(graph->FindNode("data1")->GetOpDesc(), "index", 0);
  SetNoStorage(graph->FindNode("data1")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1});

  AttrUtils::SetInt(graph->FindNode("data2")->GetOpDesc(), "index", 1);
  SetNoStorage(graph->FindNode("data2")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, -1});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto add1 = graph->FindNode("add1")->GetOpDesc();
  add1->SetOpKernelLibName(ge::kEngineNameAiCpuTf);
  AttrUtils::SetInt(add1, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AttrUtils::SetBool(add1, ge::ATTR_NAME_IS_BLOCKING_OP, true);
  AttrUtils::SetBool(add1, ge::ATTR_NAME_SUPPORT_BLOCKDIM_FLAG, true);
  SetNoStorage(add1, ge::FORMAT_ND, DT_FLOAT, {1, -1});
  add1->AppendIrInput("x1", kIrInputRequired);
  add1->AppendIrInput("x2", kIrInputRequired);

  auto &name_index = add1->MutableAllInputName();
  name_index.clear();
  name_index["x1"] = 0;
  name_index["x2"] = 1;
  graph->SetGraphUnknownFlag(true);

  return graph;
}

/*
data      data
  \       /
    AscBc
      |
    output
*/
ge::ComputeGraphPtr ShareGraph::AutoFuseNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("fused_graph", "AscBackend")->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("fused_graph"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("fused_graph1", "AscBackend")->EDGE(0, 1)->NODE("NetOutput"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 1)->NODE("fused_graph1"));
  };
  auto graph = ToComputeGraph(g1);
  auto data0 = graph->FindNode("data0");
  SetNoStorage(data0->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  AttrUtils::SetInt(data0->GetOpDesc(), "index", 0);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 2);

  auto data3 = graph->FindNode("data3");
  SetNoStorage(data3->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 3);

  auto autofuse_node = graph->FindNode("fused_graph");
  AddCompileResult(autofuse_node, false);
  SetNoStorage(autofuse_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  autofuse_node->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  autofuse_node->GetOpDesc()->SetWorkspaceBytes({2});

  auto autofuse_node1 = graph->FindNode("fused_graph1");
  AddCompileResult(autofuse_node1, false);
  SetNoStorage(autofuse_node1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  autofuse_node1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  autofuse_node1->GetOpDesc()->SetWorkspaceBytes({2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"fused_graph", "fused_graph1"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  SetNoStorage(autofuse_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1});
  return graph;
}

/*
 * main_graph
 *                   data1
 *                     |
 *      Data0 -- Partitioncall -- Output
 *
 * sub_graph
 *
 *                  Data1
 *                   |
 *      Data0 -- ReduceProd -- Output
 */
ComputeGraphPtr ShareGraph::AutofusePartitioncallGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data0", DATA)->NODE("partitioncall", PARTITIONEDCALL)->NODE("output", NETOUTPUT));
      CHAIN(NODE("data1", DATA)->EDGE(0, 1)->NODE("partitioncall", PARTITIONEDCALL));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  auto data0 = main_graph->FindNode("data0");
  SetNoStorage(data0->GetOpDesc(), FORMAT_NCHW, DT_INT32, {3, 4});
  AttrUtils::SetInt(data0->GetOpDesc(), "index", 0);

  auto data1 = main_graph->FindNode("data1");
  SetNoStorage(data0->GetOpDesc(), FORMAT_NCHW, DT_INT32, {1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);

  auto sub_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data0", DATA)->EDGE(0, 0)->NODE("reduceprod", REDUCEPROD)->NODE("NetOutput", NETOUTPUT));
      CHAIN(NODE("data1", DATA)->EDGE(0, 1)->NODE("reduceprod", REDUCEPROD));
    };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub_graph");
  ge::AttrUtils::SetInt(sub_graph->FindNode("data0")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(sub_graph->FindNode("data0")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(sub_graph->FindNode("data1")->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(sub_graph->FindNode("data1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  SetConstValue<int32_t, ge::DT_INT32>(sub_graph->FindNode("data1"), {0});

  auto partitioncall_node = main_graph->FindNode("partitioncall");
  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(partitioncall_node);;

  main_graph->AddSubgraph(sub_graph);
  partitioncall_node->GetOpDesc()->AddSubgraphName("sub_graph");
  partitioncall_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");
  main_graph->TopologicalSorting();
  return main_graph;
}

/*
 *
 * cast
 *   |
 * data
 */
ComputeGraphPtr ShareGraph::SingleInputAicoreGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", "Data")->NODE("cast", "Cast")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto cast = graph->FindNode("cast");
  SetNoStorage(cast->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AddCompileResult(cast, false);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  SetGraphOutShapeRange(graph);
  return graph;
}

/*
 *                             +-----------+  +-----------+
 *                             |Then Graph |  |Else Graph |
 *       NetOutput             |           |  |           |
 *           |                 | NetOutput |  | NetOutput |
 *          if <----------->   |   |       |  |   |       |
 *           |                 | Cast      |  |  Cast     |
 *           |-----add         |   |       |  |   |       |
 *        /    \    |          | Data     |   | DATA   |
 * pred(Data)  input(RefData)  |           |  |           |
 *                             +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfCondByShapeGraph(bool by_rank) {
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      if (by_rank) {
        CHAIN(NODE("pred", "Data")->NODE("rank", "Rank")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      } else {
        CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      }
      CHAIN(NODE("input", "RefData")->NODE("add0", "Add"));
      CHAIN(NODE("input", "RefData")->EDGE(0, 1)->NODE("add0", "Add"));
      CHAIN(NODE("add0", "Add")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto add_node = main_graph->FindFirstNodeMatchType("Add");
  add_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(add_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  add_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-2}));
  AddCompileResult(add_node, false);

  auto if_node = main_graph->FindFirstNodeMatchType("If");

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("input", "RefData")->NODE("cast0", "Cast")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  auto data_node = then_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  SetNoStorage(data_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});

  auto cast_node = then_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetInt(cast_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  SetNoStorage(cast_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AddCompileResult(cast_node, false);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("input", "RefData")->NODE("cast1", "Cast")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  data_node = else_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  SetNoStorage(data_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});

  cast_node = else_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetInt(cast_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  SetNoStorage(cast_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AddCompileResult(cast_node, false);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  then_graph->SetGraphUnknownFlag(true);
  else_graph->SetGraphUnknownFlag(true);
  SetGraphOutShapeRange(main_graph);
  SetGraphOutShapeRange(then_graph);
  SetGraphOutShapeRange(else_graph);
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
 *                             +-----------+  +-----------+
 *                             |Then Graph |  |Else Graph |
 *       NetOutput             |           |  |           |
 *           |                 | NetOutput |  | NetOutput |
 *          if <----------->   |   |       |  |   |       |
 *           |                 | Cast      |  |  Cast     |
 *           |-----add         |   |       |  |   |       |
 *        /    \    |          | Data      |   | DATA     |
 * pred(Data)  input(Data)     |           |  |           |
 *                             +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::BuildNeedInsertCastGraphWithSubgraph() {
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE("input0", "Data")->NODE("add0", "Add"));
      CHAIN(NODE("input1", "Data")->EDGE(0, 1)->NODE("add0", "Add"));
      CHAIN(NODE("add0", "Add")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  const auto data_desc0 = main_graph->FindNode("pred")->GetOpDesc();
  ge::AttrUtils::SetInt(data_desc0, "index", 0);
  data_desc0->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_desc0->MutableInputDesc(0)->SetShape(GeShape());
  data_desc0->MutableInputDesc(0)->SetOriginShape(GeShape());
  data_desc0->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_desc0->MutableOutputDesc(0)->SetShape(GeShape());
  data_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape());

  const auto data_desc1 = main_graph->FindNode("input0")->GetOpDesc();
  ge::AttrUtils::SetInt(data_desc1, "index", 1);
  data_desc1->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_desc1->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  data_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  data_desc1->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_desc1->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  data_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));


  const auto data_desc2 = main_graph->FindNode("input1")->GetOpDesc();
  ge::AttrUtils::SetInt(data_desc2, "index", 2);
  data_desc2->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_desc2->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  data_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  data_desc2->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_desc2->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  data_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  auto add_node = main_graph->FindFirstNodeMatchType("Add");
  add_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  add_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  add_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  add_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_FLOAT16);
  add_node->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({4, 4}));
  add_node->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(GeShape({4, 4}));
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  add_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  auto if_node = main_graph->FindFirstNodeMatchType("If");

  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data_then", "Data")->NODE("relu", "Relu")->NODE("then_NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();

  then_graph->SetName("then");
  auto data_node = then_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  data_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  data_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  data_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  auto relu_node = then_graph->FindFirstNodeMatchType("Relu");
  relu_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  relu_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  relu_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  relu_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  relu_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  relu_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  auto then_output_node = then_graph->FindNode("then_NetOutput");
  then_output_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  then_output_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  then_output_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data_else", "Data")->NODE("relu1", "Relu")->NODE("else_NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();

  else_graph->SetName("else");
  data_node = else_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  data_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  data_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  data_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  relu_node = else_graph->FindFirstNodeMatchType("Relu");
  relu_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  relu_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  relu_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  relu_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  relu_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  relu_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  auto else_output_node = else_graph->FindNode("else_NetOutput");
  else_output_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  else_output_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  else_output_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  if_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape());
  if_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape());
  if_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_FLOAT16);
  if_node->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({4, 4}));
  if_node->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(GeShape({4, 4}));
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({4, 4}));
  if_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 4}));

  main_graph->TopologicalSorting();

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  net_output->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  net_output->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({4, 4}));
  return main_graph;
}

ge::ComputeGraphPtr ShareGraph::IfCondGraphWithRefdata() {
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
                   CHAIN(NODE("input", "RefData")->EDGE(0, 1)->NODE("if", "If"));
                 };
    return ge::ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  auto if_node = main_graph->FindFirstNodeMatchType("If");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cond"] = 0;
  names_indexes["input0"] = 1;

  auto then_graph = []() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "RefData")->NODE("cast0", "Cast")->NODE("NetOutput", "NetOutput"));
                 };
    return ge::ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  auto data_node = then_graph->FindFirstNodeMatchType("RefData");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto cast_node = then_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  ge::AttrUtils::SetInt(cast_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  ge::AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "CastStubBin");
  // assume cast output ref from refdata
  ge::AttrUtils::SetStr(cast_node->GetOpDesc()->MutableOutputDesc(0), ge::REF_VAR_SRC_VAR_NAME, "data");
  AddCompileResult(cast_node, false);

  auto else_graph = []() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "RefData")->NODE("cast1", "Cast")->NODE("NetOutput", "NetOutput"));
                 };
    return ge::ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  data_node = else_graph->FindFirstNodeMatchType("RefData");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  cast_node = else_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  ge::AttrUtils::SetInt(cast_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  ge::AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(cast_node, false);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(true);
  then_graph->SetGraphUnknownFlag(true);
  else_graph->SetGraphUnknownFlag(true);
  SetGraphOutShapeRange(main_graph);
  SetGraphOutShapeRange(then_graph);
  SetGraphOutShapeRange(else_graph);
  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
 *        NetOutput            +-------------+  +-----------+
 *            |                |Then Graph   |  |Else Graph |
 *          cast               |             |  |           |
 *           |                 | NetOutput   |  | NetOutput |
 *          if <----------->   |   |         |  |   |       |
 *           |                 | assign      |  |  Cast     |
 *           |                 |   |  \      |  |   |       |
 *        /    \               | Data const  |  | Data      |
 * pred(Data)  input(RefData)  +-------------+  +-----------+
 */
ComputeGraphPtr ShareGraph::IfGraphWithRefData() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto refdata1 = OP_CFG("RefData")
                      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("input");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("cast", "Cast")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const1");

  auto assign =
      OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InNames({"ref", "value"}).OutNames({"ref"}).Build("assign");

  auto then_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE(assign)->NODE("cast", "Cast")->NODE("NetOutput1", "NetOutput"));
      CHAIN(NODE(const1)->EDGE(0, 1)->NODE("assign", "Assign"));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  auto data_node = then_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto netoutput_node = then_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto assign_node = then_graph->FindFirstNodeMatchType("Assign");
  assign_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(assign_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(assign_node, false);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("cast1", "Cast")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  data_node = else_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto cast_node = else_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(cast_node, false);
  auto else_netoutput_node = else_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(else_netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  then_graph->SetGraphUnknownFlag(false);
  else_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"cast"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
 *        NetOutput            +-------------+  +-----------+
 *            |                |   branch0   |  |  branch1  |
 *          cast               |             |  |           |
 *           |                 | NetOutput   |  | NetOutput |
 *          Case <-----------> |    |        |  |   |       |
 *           |                 | Cast        |  |  Cast     |
 *           |                 |   |         |  |   |       |
 *        /    \               | Data        |  | Data      |
 * pred(Data)  input(RefData)  +-------------+  +-----------+
 */
ComputeGraphPtr ShareGraph::CaseGraphWithRefData() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto refdata1 = OP_CFG("RefData")
                      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("input");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("case", "Case")->NODE("cast", "Cast")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE("case"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto case_node = main_graph->FindFirstNodeMatchType("Case");
  (void)AttrUtils::SetBool(case_node->GetOpDesc(), ATTR_INSERT_BY_MBATCH, true);

  std::vector<int64_t> shape_branch0 = {4, 1, 16, 16};  // HWCN
  auto data_branch_0 = OP_CFG("Data")
                      .TensorDesc(FORMAT_ND, DT_FLOAT, shape_branch0)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("data_branch_0");

  auto branch0_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data_branch_0)->NODE("cast_branch_0", "Cast")->NODE("output_branch_0", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  branch0_graph->SetName("branch_0");
  auto cast_node = branch0_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(cast_node, false);
  auto branch_0_netoutput_node = branch0_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(branch_0_netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  std::vector<int64_t> shape_branch1 = {8, 3, 16, 16};  // HWCN
  auto data_branch_1 = OP_CFG("Data")
                      .TensorDesc(FORMAT_ND, DT_FLOAT, shape_branch0)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("data_branch_1");

  auto branch1_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data_branch_1)->NODE("cast_branch_1", "Cast")->NODE("NetOutput_branch_1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  branch1_graph->SetName("branch_1");

  cast_node = branch1_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(cast_node, false);
  auto branch_1_netoutput_node = branch1_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(branch_1_netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  branch0_graph->SetParentGraph(main_graph);
  branch0_graph->SetParentNode(case_node);
  branch1_graph->SetParentGraph(main_graph);
  branch1_graph->SetParentNode(case_node);

  main_graph->AddSubgraph(branch0_graph);
  main_graph->AddSubgraph(branch1_graph);
  case_node->GetOpDesc()->AddSubgraphName("branch_0");
  case_node->GetOpDesc()->AddSubgraphName("branch_1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "branch_0");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "branch_1");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  branch0_graph->SetGraphUnknownFlag(false);
  branch1_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"cast"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
 *                             +-------------+  +-----------+
 *                             |Then Graph   |  |Else Graph |
 *       NetOutput             |             |  |           |
 *           |                 | NetOutput   |  | NetOutput |
 *          if <----------->   |   |         |  |   |       |
 *           |                 | assign      |  |  Cast     |
 *           |                 |   |  \      |  |   |       |
 *        /    \               |   |  conv2d |  | Data      |
 * pred(Data)  input(RefData)  |   |  /  \   |  +-----------+
 *                             |  Data const |
 *                             +-------------+
 */
ComputeGraphPtr ShareGraph::IfGraphWithRefDataAssignInsideSubgraph() {
  std::vector<int64_t> shape = {8,3,16,16};  // HWCN
  auto refdata1 = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("input");
  auto data = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {})
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("pred");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(data)->NODE("if", "If")->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const1");

  auto assign =
      OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(2).OutCnt(1).InNames({"ref", "value"}).OutNames({"ref"}).Build("assign");
  auto conv2d = OP_CFG(CONV2D)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("conv2d");

  auto then_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE(conv2d)->EDGE(0, 1)->NODE("assign", "Assign")->NODE("NetOutput1", "NetOutput"));
      CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE(assign));
      CHAIN(NODE(const1)->EDGE(0, 1)->NODE(conv2d));
    };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  auto data_node = then_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto netoutput_node = then_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto assign_node = then_graph->FindFirstNodeMatchType("Assign");
  assign_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(assign_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(assign_node, false);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE("cast1", "Cast")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  data_node = else_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto cast_node = else_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(cast_node, false);
  auto else_netoutput_node = else_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(else_netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  then_graph->SetGraphUnknownFlag(false);
  else_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}
/*
 *      netoutput
 *         |
 *     StrideSlice
 *   /  \   \    \
 * x  begin end stride
 */
ComputeGraphPtr ShareGraph::BuildStrideSliceGraph(std::vector<std::initializer_list<int64_t>> shape,
                                                  std::vector<std::initializer_list<int64_t>> min_shape,
                                                  std::vector<std::initializer_list<int64_t>> max_shape) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("x", "Data")->NODE("stride_slice", "StridedSlice")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("begin", "Data")->EDGE(0, 1)->NODE("stride_slice", "StridedSlice"));
    CHAIN(NODE("end", "Data")->EDGE(0, 2)->NODE("stride_slice", "StridedSlice"));
    CHAIN(NODE("stride", "Data")->EDGE(0, 3)->NODE("stride_slice", "StridedSlice"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("x");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[0]);
  SetShapeRangeNoStorage(data1->GetOpDesc(), min_shape[0], max_shape[0]);
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("begin");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data2->GetOpDesc(), min_shape[1], max_shape[1]);
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("end");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data3->GetOpDesc(), min_shape[1], max_shape[1]);
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data4 = graph->FindNode("stride");
  AttrUtils::SetInt(data4->GetOpDesc(), "index", 3);
  SetNoStorage(data4->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[1]);
  SetShapeRangeNoStorage(data4->GetOpDesc(), min_shape[1], max_shape[1]);
  data4->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto stride_slice = graph->FindNode("stride_slice");
  stride_slice->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  stride_slice->GetOpDesc()->AppendIrInput("begin", kIrInputRequired);
  stride_slice->GetOpDesc()->AppendIrInput("end", kIrInputRequired);
  stride_slice->GetOpDesc()->AppendIrInput("stride", kIrInputRequired);
  stride_slice->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"begin", 1}, {"end", 2}, {"stride", 3}};
  stride_slice->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(stride_slice->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[2]);
  SetShapeRangeNoStorage(stride_slice->GetOpDesc(), min_shape[2], max_shape[2]);
  AttrUtils::SetInt(stride_slice->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AttrUtils::SetStr(stride_slice->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(stride_slice, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, shape[3]);
  SetShapeRangeNoStorage(noutput->GetOpDesc(), min_shape[3], max_shape[3]);

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"stride_slice"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *      netoutput
 *         |
 *     SliceWrite
 *   /     \     \
 *   x    begin value
 */
ComputeGraphPtr ShareGraph::BuildSliceWriteNormalGraph(const std::string &node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("x", "Data")->EDGE(0, 0)->NODE("slice_write", node_type)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("begin", "Data")->EDGE(0, 1)->NODE("slice_write", node_type));
    CHAIN(NODE("value", "Data")->EDGE(0, 2)->NODE("slice_write", node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("x");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("begin");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1, 1, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 3, 4}, {1, 100, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("value");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  SetShapeRangeNoStorage(data3->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto slice_write = graph->FindNode("slice_write");
  slice_write->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  slice_write->GetOpDesc()->AppendIrInput("begin", kIrInputRequired);
  slice_write->GetOpDesc()->AppendIrInput("value", kIrInputRequired);
  slice_write->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"begin", 1}, {"value", 2}};
  slice_write->GetOpDesc()->MutableAllOutputName() = {{"x", 0}};
  slice_write->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu);
  SetNoStorage(slice_write->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 3, 4});
  SetShapeRangeNoStorage(slice_write->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});
  AttrUtils::SetBool(slice_write->GetOpDesc(), "SmallShapeHostcpu", true);
  AddCompileResult(slice_write, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"slice_write"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *     netoutput
 *         |
 *      refnode
 *      /     \
 *     x1     x2
 */
ComputeGraphPtr ShareGraph::BuildRefnodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("x1", "Data")->EDGE(0, 0)->NODE("refnode", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("x2", "Data")->EDGE(0, 1)->NODE("refnode", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("x1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("x2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1, 1, 3, 4});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 3, 4}, {1, 100, 3, 4});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto refnode = graph->FindNode("refnode");
  refnode->GetOpDesc()->AppendIrInput("x1", kIrInputRequired);
  refnode->GetOpDesc()->AppendIrInput("x2", kIrInputRequired);
  refnode->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  refnode->GetOpDesc()->MutableAllOutputName() = {{"x1", 0}};
  refnode->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu);
  SetNoStorage(refnode->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 1, 3, 4});
  SetShapeRangeNoStorage(refnode->GetOpDesc(), {1, 1, 3, 4}, {100, 100, 3, 4});
  AttrUtils::SetBool(refnode->GetOpDesc(), "SmallShapeHostcpu", true);
  AddCompileResult(refnode, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  SetShapeRangeNoStorage(noutput->GetOpDesc(), {1, 2, 3, 4}, {100, 2, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"refnode"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

///      Data    Data
///        \      /
///         Switch   Constant
///          |   \    /
///          |    Add
///          |    /
///          Merge
///           |
///        NetOutput
///
Graph ShareGraph::BuildSwitchMergeGraph() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");

  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto switch_1 = OP_CFG(SWITCH).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("switch_1");

  auto add_1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_1");

  auto merge_1 = OP_CFG(MERGE).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("merge_1");

  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(switch_1)->EDGE(0, 0)->NODE(merge_1)->EDGE(0, 0)->NODE(netoutput));
    CHAIN(NODE(switch_1)->EDGE(1, 0)->NODE(add_1)->EDGE(0, 1)->NODE(merge_1));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(switch_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    ADD_OUTPUT(netoutput, 0);
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

///      Data    Data
///        \      /
///         Switch   Constant
///          |   \    /
///          |    Add
///          |    /
///          Merge
///           | |
///        NetOutput
///
Graph ShareGraph::BuildSwitchMergeGraphWithTwoOutputs() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");

  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto switch_1 = OP_CFG(SWITCH).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("switch_1");

  auto add_1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_1");

  auto merge_1 = OP_CFG(MERGE).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("merge_1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(switch_1)->EDGE(0, 0)->NODE(merge_1)->EDGE(0, 0)->NODE("output_1", NETOUTPUT));
    CHAIN(NODE(merge_1)->EDGE(0, 1)->NODE("output_1", NETOUTPUT));
    CHAIN(NODE(switch_1)->EDGE(1, 0)->NODE(add_1)->EDGE(0, 1)->NODE(merge_1));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(switch_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

/*
 *     data_1 data_2
 *        \  /
 *        scatternd
 *           | |
 *        netoutput
 */
Graph ShareGraph::BuildAtomicNodeConnectNetoutput() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");
  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));

  auto scatter_nd = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape)
          .InCnt(2).OutCnt(2).Build("add_1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(scatter_nd)->NODE("output_1", NETOUTPUT));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(scatter_nd)->NODE("output_1", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto node = root_graph->FindNode("add_1");
  std::vector<int64_t> atomic_output_index{0, 1};
  (void) ge::AttrUtils::SetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  return graph;
}

/*
 *     data_1 data_2
 *        \  /
 *        scatternd
 *           |
 *         ref_node
 *           |
 *        netoutput
 */
Graph ShareGraph::BuildAtomicNodeConnectNetoutputThroughRefNode() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");
  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));

  auto scatter_nd = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_INT32, shape)
          .InCnt(2).OutCnt(2).Build("add_1");
  auto ref_node = OP_CFG(RELU).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).InNames({"ref"})
                     .OutNames({"ref"}).Attr(ATTR_NAME_REFERENCE, true).Build("ref_node");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(scatter_nd)->NODE(ref_node)->NODE("output_1", NETOUTPUT));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(scatter_nd)->NODE("output_1", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto node = root_graph->FindNode("add_1");
  std::vector<int64_t> atomic_output_index{0, 1};
  (void) ge::AttrUtils::SetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  return graph;
}

///
///    NetOutput
///         |
///       Merge
///      /   \
///     /    NEG
///    /      \
///   NEG    shape
///   F|     T|
///    Switch1
///   /       \
///  Data     Data
ge::Graph ShareGraph::BuildSwitchMergeGraphWithNeg() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");
  auto switch_1 = OP_CFG(SWITCH).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("switch_1");
  auto shape_node = OP_CFG(SHAPE).TensorDesc(FORMAT_NCHW, DT_INT32, {}).InCnt(1).OutCnt(1);
  auto merge_1 = OP_CFG(MERGE).TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(2).OutCnt(2).Build("merge_1");

  DEF_GRAPH(g0) {
    CHAIN(NODE(data_1)
        ->EDGE(0, 0)
        ->NODE(switch_1)
        ->EDGE(0, 0)
        ->NODE("shape", shape_node)
        ->NODE("neg0", NEG)
        ->NODE(merge_1)
        ->NODE(NODE_NAME_NET_OUTPUT,
        NETOUTPUT));
    CHAIN(NODE(data_2)
         ->EDGE(0, 1)
         ->NODE(switch_1)
         ->EDGE(1, 0)
         ->NODE("neg1", NEG)
         ->NODE(merge_1));
  };

  auto graph = ge::ToGeGraph(g0);
  return graph;
}

///      Data    Data
///        \      /
///         Switch     Constant
///          |   \    /   |
///          |    Add    |
///          |    |     /
///          |    |   /
///          |    Add
///          |    |
///          Merge
///           |
///        NetOutput
///
Graph ShareGraph::BuildSwitchMergeGraphWithMultiAddNodes() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");

  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto switch_1 = OP_CFG(SWITCH).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("switch_1");

  auto add_1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_1");

  auto add_2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_2");

  auto merge_1 = OP_CFG(MERGE).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(2).Build("merge_1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(switch_1)->EDGE(0, 0)->NODE(merge_1)->EDGE(0, 0)->NODE("output_1", NETOUTPUT));
    CHAIN(NODE(switch_1)->EDGE(1, 0)->NODE(add_1)->EDGE(0, 0)->NODE(add_2)->EDGE(0, 1)->NODE(merge_1));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(switch_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

ComputeGraphPtr ShareGraph::BuildDsaRandomNormalKnownGraph() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_2");
  auto data_3 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_3");
  auto data_4 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_4");
  auto random_normal =
      OP_CFG(DSARANDOMNORMAL).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(4).OutCnt(1).Build("random_normal");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(random_normal)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(random_normal));
    CHAIN(NODE(data_3)->EDGE(0, 2)->NODE(random_normal));
    CHAIN(NODE(data_4)->EDGE(0, 3)->NODE(random_normal));
  };
  auto graph = ToComputeGraph(g1);
  auto random_normal_node = graph->FindNode("random_normal");
  random_normal_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameDsa);
  return graph;
}

///      Data    Data Data    Data  Data  Data Data    Data
///        \      /    \      /       \    /    \      /
///         add1         add2         add3      add4
///            \         |             |        /
///             \        |            /       /
///               \      |          /       /
///                 \    |         /      /
///                    random_normal
///                       |
///                     NetOutput
///
ComputeGraphPtr ShareGraph::BuildAddAndDsaRandomNormalKnownGraph() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_2");
  auto data_3 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_3");
  auto data_4 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_4");
  auto data_5 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_5");
  auto data_6 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_6");
  auto data_7 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_7");
  auto data_8 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_8");
  auto add_1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_1");
  auto add_2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_2");
  auto add_3 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_3");
  auto add_4 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_4");
  auto random_normal =
      OP_CFG(DSARANDOMNORMAL).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(4).OutCnt(1).Build("random_normal");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(add_1)->EDGE(0, 0)->NODE(random_normal)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(data_3)->EDGE(0, 0)->NODE(add_2)->EDGE(0, 1)->NODE(random_normal));
    CHAIN(NODE(data_4)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(data_5)->EDGE(0, 0)->NODE(add_3)->EDGE(0, 2)->NODE(random_normal));
    CHAIN(NODE(data_6)->EDGE(0, 1)->NODE(add_3));
    CHAIN(NODE(data_7)->EDGE(0, 0)->NODE(add_4)->EDGE(0, 3)->NODE(random_normal));
    CHAIN(NODE(data_8)->EDGE(0, 1)->NODE(add_4));
  };
  auto graph = ToComputeGraph(g1);
  auto random_normal_node = graph->FindNode("random_normal");
  random_normal_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameDsa);
  return graph;
}

///      Data    Data   var
///        \      /      |
///         add1       split
///            \        / \
///             \      add2
///               \      |
///                 \    |
///                 NetOutput
///
ComputeGraphPtr ShareGraph::BuildVarConnectToSplit() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_2");
  vector<int32_t> data_value(1 * 2 * 3 * 8, 0);
  std::vector<int64_t> constant_shape{1, 2, 3, 8};
  GeTensorDesc data_tensor_desc(GeShape(constant_shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto var = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, constant_shape).InCnt(1).OutCnt(1).Build("var");

  auto add_1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_1");
  auto add_2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add_2");
  auto split = OP_CFG(SPLIT).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(2).Build("split");

  DEF_GRAPH(g1) {
                  CHAIN(NODE(data_1)->NODE(add_1)->Ctrl()->NODE("node_output", NETOUTPUT));
                  CHAIN(NODE(data_2)->NODE(add_1));
                  CHAIN(NODE(var)->NODE(split)->NODE(add_2)->Ctrl()->NODE("node_output", NETOUTPUT));
                  CHAIN(NODE(split)->NODE(add_2));
                };
  auto graph = ToComputeGraph(g1);
  auto split_node = graph->FindNode("split");
  (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(split_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

///     data  data
///       \   /
///        hcom
///         |
///      netoutput
ge::Graph ShareGraph::BuildHcomGraph() {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW,  DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");
  auto hcom_1 = OP_CFG(HCOMALLREDUCE)
                    .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
                    .InCnt(2)
                    .OutCnt(2)
                    .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                    .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                    .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                    .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
                    .Build("hcom_1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(hcom_1)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(hcom_1)->EDGE(1, 1)->NODE("output_1", "NetOutput"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0, 0});
  op_desc->SetWorkspaceBytes({32, 32});
  return graph;
}

///     data
///       \
///        hcom
///         | \
///         a  b
///         \  \
///      netoutput
ge::Graph ShareGraph::BuildHcomGraphWithTwoOutputs(const std::string hcom_node_type) {
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto hcom_1 = OP_CFG(hcom_node_type)
                    .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
                    .InCnt(1)
                    .OutCnt(2)
                    .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
                    .Build("hcom_1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->NODE(hcom_1)->NODE("a", RELU)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(hcom_1)->NODE("b", RELU)->NODE("output_1", "NetOutput"));
  };

  auto graph = ToGeGraph(g1);
  return graph;
}

///   refdata  refdata
///       \   /
///        hcom
///         |
///      netoutput
ge::Graph ShareGraph::BuildHcomGraphWithRefData() {
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(REFDATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(REFDATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_2");
  auto hcom_1 = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(2)
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
      .Build("hcom_1");

  DEF_GRAPH(g1) {
                  CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(hcom_1)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
                  CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(hcom_1)->EDGE(1, 1)->NODE("output_1", "NetOutput"));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
  op_desc->SetWorkspace({0, 0});
  op_desc->SetWorkspaceBytes({32, 32});
  return graph;
}
/*
 *      NetOutput
 *         |
 *      DstNode
 *      |    \
 *     id1  const1
 *      |
 *    data1
 */
ComputeGraphPtr ShareGraph::FixedAddrNodeGraph1() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("id1", "Identity")->NODE("DsaNode", "DsaNode")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("DsaNode", "DsaNode"));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"id3", "id4"});
  net_output->GetOpDesc()->SetSrcIndex({0, 0});
  graph->TopologicalSorting();
  return graph;
}
/*
 *   NetOutput
 *   /     \
 * id3     id4
 *   \     /
 *   DstNode
 *   /     \
 * id1     id2
 *   \     /
 *    data1
 */
ComputeGraphPtr ShareGraph::FixedAddrNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("id1", "Identity")
              ->NODE("DsaNode", "DsaNode")
              ->NODE("id3", "Identity")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->EDGE(0, 0)
              ->NODE("id2", "Identity")
              ->EDGE(0, 1)
              ->NODE("DsaNode", "DsaNode")
              ->EDGE(1, 0)
              ->NODE("id4", "Identity")
              ->EDGE(0, 1)
              ->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"id3", "id4"});
  net_output->GetOpDesc()->SetSrcIndex({0, 0});
  graph->TopologicalSorting();
  return graph;
}
/*
 *   NetOutput
 *      |
 *   DsaNode
 *      |
 *   PhonyConcat
 *   /     \
 * id1     id2
 *   \     /
 *    data1
 */
ge::ComputeGraphPtr ShareGraph::FixedAddrConnectToPhonyConcat() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("id1", "Identity")
              ->NODE("pc1", "PhonyConcat")
              ->NODE("DsaNode", "DsaNode")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("id2", "Identity")->EDGE(0, 1)->NODE("pc1", "PhonyConcat"));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"DsaNode"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  return graph;
}

/*
            (0,0)
  ┌────────────────────┐
  │                    ∨
┌─────────┐  (0,1)   ┌───────┐  (0,0)   ┌─────┐  (0,0)   ┌───────────┐     ┌────────┐
│ const_1 │ ───────> │ add_1 │ ───────> │ pc1 │ ───────> │ NetOutput │ <·· │ data_1 │
└─────────┘          └───────┘          └─────┘          └───────────┘     └────────┘
            (0,1)                         ∧
  ┌────────────────────┐                  │
  │                    ∨                  │
┌─────────┐  (0,0)   ┌───────┐  (0,1)     │
│ const_2 │ ───────> │ add_2 │ ───────────┘
└─────────┘          └───────┘
 */
ge::Graph ShareGraph::NetoutputNotSupportZeroCopy() {
  std::vector<int64_t> cons_shape{2, 2, 2, 2};
  vector<int32_t> data_value(2 * 2 * 2 * 2, 0);
  GeTensorDesc data_tensor_desc(GeShape(cons_shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  GeTensorDesc data_tensor_desc_x(GeShape(cons_shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor_x = make_shared<GeTensor>(data_tensor_desc_x, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_2 = OP_CFG(CONSTANTOP).Weight(tensor_x).Build("const_2");

  std::vector<int64_t> shape1{2, 2, 2, 2};
  auto data_1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape1)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_1");

  std::vector<int64_t> pc_shape{4, 2, 2, 2};
  auto pc_1 = OP_CFG("PhonyConcat")
      .TensorDesc(FORMAT_NCHW, DT_UINT32, pc_shape)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true)
      .Attr(ATTR_NAME_OUTPUT_REUSE_INPUT, true)
      .Attr(ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0)
      .Build("pc1");
  pc_1->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({4, 2, 2, 2})));
  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape1)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape1)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  DEF_GRAPH(g1) {
    CHAIN(NODE(const_1)->EDGE(0, 0)->NODE(add_1)->EDGE(0, 0)->NODE(pc_1)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_2)->EDGE(0, 0)->NODE(add_2)->EDGE(0, 1)->NODE(pc_1));
    CHAIN(NODE(const_2)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(data_1)->CTRL_EDGE()->NODE("NetOutput", "NetOutput"));
  };

  auto graph = ToGeGraph(g1);
  return graph;
}

/*
 *   NetOutput
 *     /   \
 *  id2    id3
 *      \/
 *   DsaNode
 *      |
 *     id1
 *      |
 *    data1
 */
ge::ComputeGraphPtr ShareGraph::FixedAddrConnectToMultiPeers() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("id1", "Identity")
              ->NODE("DsaNode", "DsaNode")
              ->NODE("id2", "Identity")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("DsaNode", "DsaNode")->EDGE(0, 0)->NODE("id3", "Identity")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"id2", "id3"});
  net_output->GetOpDesc()->SetSrcIndex({0, 0});
  graph->TopologicalSorting();
  return graph;
}
/*
 *            NetOutput
 *               |
 *             id1(Identity)
 *               |
 *             add2
 *           /     \
 *         add1  dsa1(Dsa)
 *        /   \
 *     data1 data2
 */
ge::ComputeGraphPtr ShareGraph::BuildAiCoreRtsDsaNodeKnownShapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->NODE("add2", "Add")
              ->NODE("id1", "Identity")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("dsa1", "Dsa")->EDGE(0, 1)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto dsa1 = graph->FindNode("dsa1");
  SetNoStorage(dsa1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  dsa1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  add2->GetOpDesc()->SetWorkspaceBytes({1024});
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  AddCompileResult(add2, false);

  auto id1 = graph->FindNode("id1");
  id1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  id1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);
  SetNoStorage(id1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});

  auto netoutput = graph->FindNode("NetOutput");
  SetNoStorage(netoutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"id1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}
/*
 *            NetOutput
 *               |
 *             add2
 *           /     \
 *         add1    id1(Identity)
 *        /   \        |
 *     data1 data2  dsa1(Dsa)
 */
ge::ComputeGraphPtr ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("dsa1", "Dsa")->NODE("id1", "Identity")->EDGE(0, 1)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto dsa1 = graph->FindNode("dsa1");
  SetNoStorage(dsa1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  dsa1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  AddCompileResult(add1, false);

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  AddCompileResult(add2, false);

  auto id1 = graph->FindNode("id1");
  id1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  id1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);
  SetNoStorage(id1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});

  auto netoutput = graph->FindNode("NetOutput");
  SetNoStorage(netoutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"id1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}
/*
 *              NetOutput
 *                 |
 *   data3 --->   add2  <--------+
 *                 |c            |
 *             ls1(LabelSwitch)  |
 *                 |             |
 *             id1(Identity)     |
 *                 |             |
 *               add1------------+
 *              /   \
 *           data1 data2
 */
ge::ComputeGraphPtr ShareGraph::BuildGraphHasLabelSwitch() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")
              ->NODE("add1", "Add")
              ->NODE("id1", "Identity")
              ->NODE("ls1", "LabelSwitchByIndex")
              ->CTRL_EDGE()
              ->NODE("add2", "Add")
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data3", "Data")->NODE("add2", "Add"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add")->EDGE(0, 1)->NODE("add2", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data3 = graph->FindNode("data3");
  AttrUtils::SetInt(data3->GetOpDesc(), "index", 2);
  SetNoStorage(data3->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  data3->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {});
  AddCompileResult(add1, false);

  auto id1 = graph->FindNode("id1");
  id1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  id1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);
  SetNoStorage(id1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {});

  auto ls1 = graph->FindNode("ls1");
  ls1->GetOpDesc()->SetWorkspaceBytes({100});
  ls1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  ls1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameRts);
  SetNoStorage(ls1->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {});

  auto add2 = graph->FindNode("add2");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});
  add2->GetOpDesc()->MutableInputDesc(1)->SetShape({});
  add2->GetOpDesc()->MutableInputDesc(1)->SetOriginShape({});
  AddCompileResult(add2, false);

  auto netoutput = graph->FindNode("NetOutput");
  SetNoStorage(netoutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {8, 8, 8, 8});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add2"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*                           body                       cond
 input(refdata)  data            +--------------------+      +---------------------+
       \         /                | data0(ref)   data1  |    |   data0(ref) data1  |
           while                  |      \      /      |     |       |    /c       |
              |                   |        assign      |     |      cast           |
           netoutput              |         |(0)       |     |       |(0)          |
                                  |      netoutput     |     |      netoutput      |
                                  +--------------------+     +---------------------+
 */
ge::ComputeGraphPtr ShareGraph::BuildGraphRefdataWhile() {
  std::vector<int64_t> shape = {8,3,16,16};  // HWCN
  auto refdata1 = OP_CFG("RefData")
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("input");
  auto cast = OP_CFG("Cast")
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("cast");
  auto netoutput = OP_CFG("NetOutput")
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("NetOutput");
  auto while_op = OP_CFG("While")
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(2)
      .OutCnt(2)
      .Build("while");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE(refdata1)->NODE(while_op)->EDGE(0, 0)->NODE(cast)->NODE(netoutput));
                   CHAIN(NODE("pred", "Data")->EDGE(0, 1)->NODE(while_op));
                 };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 1);

  auto while_node = main_graph->FindFirstNodeMatchType(WHILE);

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const1");

  auto assign =
      OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InNames({"ref", "value"}).OutNames({"ref"}).Build("assign");

  auto body_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE(assign)->NODE("NetOutput1", "NetOutput"));
                   CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE(assign));
                   CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("NetOutput1", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");
  auto data_node = body_graph->FindNode("data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto data1_node = body_graph->FindNode("data1");
  ge::AttrUtils::SetInt(data1_node->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data1_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto netoutput_node = body_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto cond_graph = []() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE("cast1", "Cast")->NODE("NetOutput2", "NetOutput"));
                   CHAIN(NODE("data1", "Data")->CTRL_EDGE()->NODE("cast1", "Cast"));
                 };
    return ToComputeGraph(g);
  }();
  cond_graph->SetName("cond");
  data_node = cond_graph->FindNode("data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  data1_node = cond_graph->FindNode("data1");
  ge::AttrUtils::SetInt(data1_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data1_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto cond_netoutput_node = cond_graph->FindFirstNodeMatchType("NetOutput2");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(while_node);
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(while_node);

  main_graph->AddSubgraph(body_graph);
  main_graph->AddSubgraph(cond_graph);
  while_node->GetOpDesc()->AddSubgraphName("cond");
  while_node->GetOpDesc()->AddSubgraphName("body");
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, "cond");
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, "body");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  cond_graph->SetGraphUnknownFlag(false);
  body_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"while"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

///-------------------------------------------------
///           Data0   Data1
///            \      /
///             \    /
///               Add1
///                |    +---Constant_1
///                |   /    /     |
///                Add2    /      |
///                |      /       |
///                |\    /        |
///                | Add3         |
///                |  \           |
///                |   \          |
///                |----Add4     /
///                      \      /
///                       \    /
///                       mul_1
///                        |
///                        NetOutput // set netoutput reuse
///------------------------------------------------
/// 构造图的输入内存比图中的其他op输出内存大的场景，对add进行info shape打桩
///  内存分配结果 -----offset ------ size ----------
///  data0            16896         512
///  data1            17408         512
///  add1             1024          512    // reuse netouput
///  add2             515           512    // reuse netouput
///  add3             1024          512    // reuse netouput
///  add4             0             512
///  mul_1            512         16384
///
ge::Graph ShareGraph::BuildIoReuseMemGraph() {
  std::vector<int64_t> shape{2, 2, 2, 2};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  auto data_1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("data_1");

  // add1-5
  auto add_1 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_3");

  auto add_4 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_4");

  auto mul_1 = OP_CFG(MUL)
      .InCnt(2)
      .OutCnt(1)
      .Build("mul_1");

  std::vector<int64_t> cons_shape{2, 2, 2, 2};
  vector<int32_t> data_value(2 * 2 * 2 * 2, 0);
  GeTensorDesc data_tensor_desc(GeShape(cons_shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3)
          ->EDGE(0, 0)
          ->NODE(add_4)
          ->EDGE(0, 0)
          ->NODE(mul_1)
          );

    CHAIN(NODE(data_1)->EDGE(0, 1)->NODE(add_1));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(mul_1));
    CHAIN(NODE(add_2)->EDGE(0, 1)->NODE(add_4));
    ADD_OUTPUT(mul_1, 0);
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

/*
 *
 *    netoutput
 *      |
 *     data1
 */
ComputeGraphPtr ShareGraph::BuildInputDirectlyConnectedToOutputGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), ge::FORMAT_ND, DT_INT32, {1, 2, 3, 4});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"data1"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});
  return graph;
}
/*
 *  data1  data2
 *    \   /
 *     add1(streamid=0)(send_id_list:[0])
 *      |
 *     relu (streamid=1)(send_id_list:[1],recive_id_list:[0])
 *      |
 *    netoutput(streamid=0)(recive_id_list:[1]))
 */
ComputeGraphPtr ShareGraph::MultiStreamTwoNodeGraph(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 2;

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("relu", "Relu")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data1->GetOpDescBarePtr()->SetStreamId(0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data2->GetOpDescBarePtr()->SetStreamId(0);

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  add1->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  add1->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {0});
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto relu = graph->FindNode("relu");
  AddCompileResult(relu, false);
  SetNoStorage(relu->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  relu->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  relu->GetOpDescBarePtr()->SetStreamId(1);
  AttrUtils::SetListInt(relu->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  AttrUtils::SetListInt(relu->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {1});
  relu->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  net_output->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(net_output->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {1});

  SetGraphOutShapeRange(graph);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *  data1  data2
 *    \   /
 *     add1
 *      |
 *    netoutput
 */
Graph ShareGraph::OnlyDataGraph(std::initializer_list<int64_t> data0_shape, std::initializer_list<int64_t> data1_shape) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("NetOutput"));
  };
  auto ge_graph = ToGeGraph(g1);
  auto graph = GraphUtilsEx::GetComputeGraph(ge_graph);

  auto data1 = graph->FindNode("data0");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, data0_shape);
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data1");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, data1_shape);
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"data0", "data1"});
  net_output->GetOpDesc()->SetSrcIndex({0, 0});
  return ge_graph;
}

/*

     ┌──────────────────────────────────────────────┐
     │                                              │
     │   ┌──────┐                                   │
     │   │data_i├───┐                               │
     │   └──────┘   │ ┌───┐     ┌───────────────┐   │
     │              │►│add├────►│sub_1_netoutput│   │
     |              | |s1 |     |               |   |
     │   ┌───────┐  │ └───┘     └───────────────┘   │
     │   │var_1  ├──┘                               │
     │   └───────┘                                  │
     │                                              │
     └─────────────────────────┬────────────────────┘
                               │
 ┌───────┐     ┌────┐     ┌────▼───┐
 │ data_1├────►│relu├────►│known_op├───┐
 |s0     |     | s1 |     |  s0    |   |
 └───────┘     └────┘     └────────┘   │  ┌──────────────┐
                                       ├─►│root_netoutput|
                                       │  |   s0         |
 ┌───────┐               ┌──────────┐  │  └──────────────┘
 │ data_2├──────────────►│transdata ├──┘
 |  s0   |               |    s0    |
 └───────┘               └────-─────┘
*/
ComputeGraphPtr ShareGraph::GraphDynamicAndStaticGraphWithVariables(int64_t &stream_num, int64_t &event_num) {
  std::vector<int64_t> shape = {2, 2};  // NCHW
  // sub1
  auto data_i = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_i");
  data_i->SetOutputOffset({32});

  auto var1 = OP_CFG("Variable")
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                  .InCnt(1)
                  .OutCnt(1)
                  .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                  .Attr(ATTR_NAME_INDEX, 0)
                  .Build("var1");
  var1->SetOutputOffset({137438953472});
  TensorUtils::SetSize(*var1->MutableOutputDesc(0), 64);
  ge::AttrUtils::SetStr(var1, ge::ATTR_VARIABLE_PLACEMENT, "host");

  auto add = OP_CFG("Add").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add");
  add->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  add->SetOpEngineName(ge::kEngineNameAiCpu.c_str());
  add->SetInputOffset({0, 16});
  add->SetOutputOffset({32});

  auto sub_1_netoutput = OP_CFG(ge::NETOUTPUT)
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                             .InCnt(1)
                             .OutCnt(1)
                             .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Build("sub_1_netoutput");
  sub_1_netoutput->SetInputOffset({0, 16});

  // root
  auto data_1 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_1");
  data_1->SetStreamId(0);
  auto var_2 = OP_CFG("Variable")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .Build("var2");
  var_2->SetStreamId(0);
  var_2->SetOutputOffset({137438953572});
  TensorUtils::SetSize(*var_2->MutableOutputDesc(0), 64);

  auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("relu");
  relu->SetOpKernelLibName(ge::kEngineNameAiCore);
  relu->SetOpEngineName(kEngineNameAiCore);
  relu->SetStreamId(0);

  auto known_op =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("known_op");
  known_op->SetStreamId(0);

  auto transdata = OP_CFG("TransData").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("transdata");
  transdata->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
  transdata->SetStreamId(0);
  auto root_netoutput =
      OP_CFG(ge::NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("root_netoutput");
  root_netoutput->SetSrcName({"known_op", "transdata"});
  root_netoutput->SetSrcIndex({0, 1});
  root_netoutput->SetStreamId(0);

  DEF_GRAPH(sub_1) {
    CHAIN(NODE(data_i)->NODE(add)->NODE(sub_1_netoutput));
    CHAIN(NODE(var1)->NODE(add));
  };

  DEF_GRAPH(root) {
    CHAIN(NODE(data_1)->NODE(relu)->NODE(known_op)->NODE(root_netoutput));
    CHAIN(NODE(var_2)->NODE(transdata)->EDGE(0, 1)->NODE(root_netoutput));
  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto sub_graph1 = ToGeGraph(sub_1);
  auto compute_graph1 = ge::GraphUtilsEx::GetComputeGraph(sub_graph1);
  compute_graph1->SetGraphUnknownFlag(false);
  auto net_output = compute_graph1->FindNode("sub_1_netoutput");
  net_output->GetOpDesc()->SetSrcName({"add"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto relu_node = root_graph->FindNode("relu");
  AddCompileResult(relu_node, false);
  SetNoStorage(relu_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  SetShapeRangeNoStorage(relu_node->GetOpDesc(), {1, 1}, {-1, -1});
  auto transdata_node = root_graph->FindNode("transdata");
  SetShapeRangeNoStorage(transdata_node->GetOpDesc(), {1, 1}, {-1, -1});
  AddCompileResult(transdata_node, false);
  SetNoStorage(transdata_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});

  auto known_node = root_graph->FindNode("known_op");
  SetSubGraph(root_graph, known_node, compute_graph1);

  AddCompileResult(known_node, false);
  return root_graph;
}

/*
 *      data1
 *       |
 *      cast(stream:0)(send[0])
 *      /    \
 * transdata  relu (stream:1)(send:[1],recive:[0])
 * (stream:0)  /
 *      \     /
 *    netoutput(stream:0)(recive_id_list:[1]))
 */
ComputeGraphPtr ShareGraph::MultiStreamGraphConsumersInAndCrossStream(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 2;

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("cast", "Cast")->NODE("transdata", "TransData")->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("cast", "Cast")->NODE("relu", "Relu")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data1->GetOpDescBarePtr()->SetStreamId(0);

  auto cast = graph->FindNode("cast");
  AddCompileResult(cast, false);
  SetNoStorage(cast->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  cast->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  cast->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  cast->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(cast->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {0});
  cast->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto transdata = graph->FindNode("transdata");
  AddCompileResult(transdata, false);
  SetNoStorage(transdata->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  transdata->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  transdata->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  transdata->GetOpDescBarePtr()->SetStreamId(0);
  transdata->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto relu = graph->FindNode("relu");
  AddCompileResult(relu, true);
  SetNoStorage(relu->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  relu->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  relu->GetOpDescBarePtr()->SetStreamId(1);
  AttrUtils::SetListInt(relu->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  AttrUtils::SetListInt(relu->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {1});
  relu->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"transdata", "relu"});
  net_output->GetOpDesc()->SetSrcIndex({0, 0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  net_output->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(net_output->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {1});
  SetGraphOutShapeRange(graph);
  return graph;
}

/*
 * refdata1   data1
 *(stream:0)  (stream:0)
 *      \     /
 *      assign(stream:0)(send[0])
 *      /    \
 * transdata  relu (stream:1)(send:[1],recive:[0])
 * (stream:0)  /
 *      \     /
 *    netoutput(stream:0)(recive_id_list:[1]))
 */
ComputeGraphPtr ShareGraph::MultiStreamGraphAccessRefMemCrossStream(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 2;
  auto assign_op = OP_CFG(ASSIGN)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1 - 1, -1})
                       .InCnt(2)
                       .OutCnt(1)
                       .InNames({"ref", "value"})
                       .OutNames({"ref"})
                       .Attr(ge::ATTR_NAME_REFERENCE, true)
                       .Build("assign");

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata1", "RefData")->NODE(assign_op)->EDGE(0, 0)->NODE("transdata", "TransData")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->EDGE(0, 1)
              ->NODE(assign_op)
              ->EDGE(0, 0)
              ->NODE("relu", "Relu")
              ->EDGE(0, 1)
              ->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto refdata1 = graph->FindNode("refdata1");
  SetNoStorage(refdata1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(refdata1->GetOpDesc(), "index", 0);
  refdata1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  refdata1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  refdata1->GetOpDescBarePtr()->SetStreamId(0);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data1->GetOpDescBarePtr()->SetStreamId(0);

  auto assign = graph->FindNode("assign");
  AddCompileResult(assign, false);
  SetNoStorage(assign->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  assign->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  assign->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  assign->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(assign->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {0});
  assign->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto transdata = graph->FindNode("transdata");
  AddCompileResult(transdata, false);
  SetNoStorage(transdata->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  transdata->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  transdata->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  transdata->GetOpDescBarePtr()->SetStreamId(0);
  transdata->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto relu = graph->FindNode("relu");
  AddCompileResult(relu, false);
  SetNoStorage(relu->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  relu->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  relu->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  relu->GetOpDescBarePtr()->SetStreamId(1);
  AttrUtils::SetListInt(relu->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  AttrUtils::SetListInt(relu->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {1});
  relu->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"transdata", "relu"});
  net_output->GetOpDesc()->SetSrcIndex({0, 0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  net_output->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(net_output->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {1});
  SetGraphOutShapeRange(graph);
  return graph;
}
/*
 * refdata1    data1
 *(stream:0)  (stream:0)
 *(send:[0])  (send:[1])
 *      \     /
 *      assign(stream:1)(send:[2], recive:[0,1])
 *        |
 *    netoutput(stream:0)(recive_id_list:[2]))
 */
ge::ComputeGraphPtr ShareGraph::MultiStreamGraphRefMemCrossStream(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 3;
  auto assign_op = OP_CFG(ASSIGN)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1 - 1, -1})
                       .InNames({"ref", "value"})
                       .OutNames({"ref"})
                       .Attr(ge::ATTR_NAME_REFERENCE, true)
                       .Attr(ge::REF_VAR_SRC_VAR_NAME, "refdata1")
                       .Build("assign");

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata1", "RefData")->NODE(assign_op)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE(assign_op));
  };
  auto graph = ToComputeGraph(g1);
  auto refdata1 = graph->FindNode("refdata1");
  SetNoStorage(refdata1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(refdata1->GetOpDesc(), "index", 0);
  refdata1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  refdata1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  refdata1->GetOpDescBarePtr()->SetStreamId(0);
  refdata1->GetOpDescBarePtr()->SetOutputOffset({0});
  AttrUtils::SetListInt(refdata1->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {0});

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data1->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(data1->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {1});
  data1->GetOpDescBarePtr()->SetOutputOffset({456});

  auto assign = graph->FindNode("assign");
  AddCompileResult(assign, false);
  SetNoStorage(assign->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  assign->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  assign->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  assign->GetOpDescBarePtr()->SetStreamId(1);
  AttrUtils::SetListInt(assign->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {2});
  AttrUtils::SetListInt(assign->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0, 1});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"assign"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  net_output->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(net_output->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {2});
  net_output->GetOpDescBarePtr()->SetInputOffset({0});
  SetGraphOutShapeRange(graph);
  return graph;
}
/*

     ┌──────────────────────────────────────────────┐
     │                                              │
     │   ┌──────┐                                   │
     │   │data_i├───┐                               │
     │   └──────┘   │ ┌───┐     ┌───────────────┐   │
     │              │►│add├────►│sub_1_netoutput│   │
     |              | |s1 |     |               |   |
     │   ┌───────┐  │ └───┘     └───────────────┘   │
     │   │const_1├──┘                               │
     │   └───────┘                                  │
     │                                              │
     └─────────────────────────┬────────────────────┘
                               │
 ┌───────┐     ┌────┐     ┌────▼───┐
 │ data_1├────►│relu├────►│known_op├───┐
 |s0     |     | s1 |     |  s0    |   |
 └───────┘     └────┘     └────────┘   │  ┌──────────────┐
                                       ├─►│root_netoutput|
                                       │  |   s0         |
 ┌───────┐               ┌──────────┐  │  └──────────────┘
 │ data_2├──────────────►│transdata ├──┘
 |  s0   |               |    s0    |
 └───────┘               └────-─────┘
*/
ComputeGraphPtr ShareGraph::MultiStreamGraphDynamicAndStaticGraph(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2 + 3;                   // dynamic stream 2, static stream 3
  event_num = 2 + 1;                    // dynamic event 3, static event 1
  std::vector<int64_t> shape = {2, 2};  // NCHW
  // sub1
  auto data_i = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_i");
  data_i->SetOutputOffset({32});

  auto const_1 = OP_CFG("Const")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("const_1");
  ge::AttrUtils::SetTensor(const_1, "value", CreateVecTorGeTensor(shape, DT_FLOAT));

  auto add = OP_CFG("Add").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add");
  add->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  add->SetOpEngineName(ge::kEngineNameAiCpu.c_str());
  add->SetInputOffset({0, 16});
  add->SetOutputOffset({32});
  add->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto sub_1_netoutput = OP_CFG(ge::NETOUTPUT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Build("sub_1_netoutput");
  sub_1_netoutput->SetInputOffset({0, 16});

  // root
  auto data_1 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_1");
  data_1->SetStreamId(0);
  AttrUtils::SetListInt(data_1, ge::ATTR_NAME_SEND_EVENT_IDS, {0});

  auto const_2 = OP_CFG("Const")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("const_2");
  const_2->SetStreamId(0);
  ge::AttrUtils::SetTensor(const_2, "value", CreateVecTorGeTensor(shape, DT_FLOAT));

  auto relu = OP_CFG("Relu")
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                  .InCnt(1)
                  .OutCnt(1)
                  .Build("relu");
  relu->SetOpKernelLibName(ge::kEngineNameAiCore);
  relu->SetOpEngineName(kEngineNameAiCore);
  relu->SetStreamId(1);
  AttrUtils::SetListInt(relu, ge::ATTR_NAME_SEND_EVENT_IDS, {1});
  AttrUtils::SetListInt(relu, ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  relu->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto known_op = OP_CFG(ge::PARTITIONEDCALL)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("known_op");
  known_op->SetStreamId(0);
  AttrUtils::SetListInt(known_op, ge::ATTR_NAME_RECV_EVENT_IDS, {1});

  auto transdata = OP_CFG("TransData").TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("transdata");
  transdata->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
  transdata->SetStreamId(0);
  transdata->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  auto root_netoutput = OP_CFG(ge::NETOUTPUT)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .InCnt(2)
                            .OutCnt(1)
                            .Build("root_netoutput");
  root_netoutput->SetSrcName({"known_op", "transdata"});
  root_netoutput->SetSrcIndex({0, 1});
  root_netoutput->SetStreamId(0);

  DEF_GRAPH(sub_1) {
                     CHAIN(NODE(data_i)->NODE(add)->NODE(sub_1_netoutput));
                     CHAIN(NODE(const_1)->NODE(add));
                   };

  DEF_GRAPH(root) {
                    CHAIN(NODE(data_1)->NODE(relu)->NODE(known_op)->NODE(root_netoutput));
                    CHAIN(NODE(const_2)->NODE(transdata)->EDGE(0,1)->NODE(root_netoutput));
                  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto sub_graph1 = ToGeGraph(sub_1);
  auto compute_graph1 = ge::GraphUtilsEx::GetComputeGraph(sub_graph1);
  compute_graph1->SetGraphUnknownFlag(false);
  auto net_output = compute_graph1->FindNode("sub_1_netoutput");
  net_output->GetOpDesc()->SetSrcName({"add"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  const char kernel_bin[] = "kernel_bin";
  auto add_node = compute_graph1->FindNode("add");
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
  add_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_bin_ptr);

  auto relu_node = root_graph->FindNode("relu");
  AddCompileResult(relu_node, false);
  SetNoStorage(relu_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  SetShapeRangeNoStorage(relu_node->GetOpDesc(), {1, 1}, {-1, -1});
  auto transdata_node = root_graph->FindNode("transdata");
  transdata_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  SetShapeRangeNoStorage(transdata_node->GetOpDesc(), {1, 1}, {-1, -1});
  AddCompileResult(transdata_node, false);
  SetNoStorage(transdata_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});

  auto known_node = root_graph->FindNode("known_op");
  SetSubGraph(root_graph, known_node, compute_graph1);

  AddCompileResult(known_node, false);
  return root_graph;
}
/*
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *       NetOutput          |           |  |           |
 *           |              |           |  | NetOutput |
 *          Add <---+       |           |  |   |       |
 *           |      |       | NetOutput |  |  Add      |
 *          if  <---|-----> |   |       |  |  ||       |
 *        /    \    |       | Const     |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr ShareGraph::MultiStreamGraphWithIfGraph(int64_t &stream_num, int64_t &event_num) {
  auto origin_graph = IfGraph3();

}

/*

 ┌───────---┐               ┌──────────┐   ┌──────────────┐
 │ fileconst├──────────────►│   relu1  ├-->│root_netoutput|
 |  s0      |               |    s1    |   |   s0         |
 └───────---┘               └────-─────┘   └──────────────┘
*/
ComputeGraphPtr ShareGraph::MultiStreamGraphFileConstantGraph(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 2;
  std::vector<int64_t> shape = {2, 2};  // NCHW

  auto const_2 = OP_CFG("FileConstant")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("const_2");
  const_2->SetStreamId(0);
  ge::AttrUtils::SetTensor(const_2, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  AttrUtils::SetListInt(const_2, ge::ATTR_NAME_SEND_EVENT_IDS, {0});

  SetNoStorage(const_2, ge::FORMAT_ND, DT_INT32, {5, 5});
  const_2->AppendIrAttrName("file_path");
  const_2->AppendIrAttrName("file_id");
  const_2->AppendIrAttrName("shape");
  const_2->AppendIrAttrName("dtype");

  // set attr
  std::vector<int64_t> file_const_shape = {5, 5};
  std::vector<int64_t> original_shape = {1, 5, 5};
  ge::AttrUtils::SetInt(const_2, "offset", 0);
  ge::AttrUtils::SetInt(const_2, "length", 0);
  ge::AttrUtils::SetStr(const_2, "location", "");
  ge::AttrUtils::SetStr(const_2, "file_path", "test_weight.bin");
  ge::AttrUtils::SetStr(const_2, "file_id", "");
  ge::AttrUtils::SetDataType(const_2, "dtype", DT_INT32);
  ge::AttrUtils::SetListInt(const_2, "shape", file_const_shape);
  ge::AttrUtils::SetListInt(const_2, "original_shape", original_shape);

  std::string file_name = "test_weight.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out0(file_name, std::ios::binary);
  if (!out0.is_open()) {
    return nullptr;
  }
  out0.write(reinterpret_cast<char*>(data), sizeof(data));
  out0.close();

  auto relu = OP_CFG("Relu")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("relu");
  relu->SetOpKernelLibName(ge::kEngineNameAiCore);
  relu->SetOpEngineName(kEngineNameAiCore);
  relu->SetStreamId(1);
  AttrUtils::SetListInt(relu, ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  AttrUtils::SetListInt(relu, ge::ATTR_NAME_SEND_EVENT_IDS, {1});
  relu->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  auto root_netoutput = OP_CFG(ge::NETOUTPUT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("root_netoutput");
  root_netoutput->SetSrcName({"relu"});
  root_netoutput->SetSrcIndex({0});
  root_netoutput->SetStreamId(0);
  AttrUtils::SetListInt(root_netoutput, ge::ATTR_NAME_RECV_EVENT_IDS, {1});

  DEF_GRAPH(root) {
    CHAIN(NODE(const_2)->NODE(relu)->NODE(root_netoutput));
  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto relu_node = root_graph->FindNode("relu");
  AddCompileResult(relu_node, false);
  relu_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  SetNoStorage(relu_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  return root_graph;
}
/*

 ┌───────---┐               ┌──────────┐   ┌──────────────┐
 │ fileconst├──────────────►│   relu1  ├-->│root_netoutput|
 |  s1      |               |    s1    |   |   s0         |
 └───────---┘               └────-─────┘   └──────────────┘
     Data(s0)  fileConstant(s1)
             \/
             reshape(s0)
             |
             netoutput
*/
ComputeGraphPtr ShareGraph::MultiStreamGraphFileConstantToHostGraph(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 1;
  std::vector<int64_t> shape = {2, 2};  // NCHW

  auto const_2 = OP_CFG("FileConstant")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("const_2");
  const_2->SetStreamId(1);
  ge::AttrUtils::SetTensor(const_2, "value", CreateVecTorGeTensor(shape, DT_FLOAT));
  AttrUtils::SetListInt(const_2, ge::ATTR_NAME_SEND_EVENT_IDS, {0});

  SetNoStorage(const_2, ge::FORMAT_ND, DT_INT32, {5, 5});
  const_2->AppendIrAttrName("file_path");
  const_2->AppendIrAttrName("file_id");
  const_2->AppendIrAttrName("shape");
  const_2->AppendIrAttrName("dtype");

  // set attr
  std::vector<int64_t> file_const_shape = {5, 5};
  std::vector<int64_t> original_shape = {1, 5, 5};
  ge::AttrUtils::SetInt(const_2, "offset", 0);
  ge::AttrUtils::SetInt(const_2, "length", 0);
  ge::AttrUtils::SetStr(const_2, "location", "");
  ge::AttrUtils::SetStr(const_2, "file_path", "test_weight.bin");
  ge::AttrUtils::SetStr(const_2, "file_id", "");
  ge::AttrUtils::SetDataType(const_2, "dtype", DT_INT32);
  ge::AttrUtils::SetListInt(const_2, "shape", file_const_shape);
  ge::AttrUtils::SetListInt(const_2, "original_shape", original_shape);

  std::string file_name = "test_weight.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out0(file_name, std::ios::binary);
  if (!out0.is_open()) {
    return nullptr;
  }
  out0.write(reinterpret_cast<char*>(data), sizeof(data));
  out0.close();

  auto data_1 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_1");
  data_1->SetStreamId(0);

  auto reshape = OP_CFG("Reshape")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("reshape");
  reshape->AppendIrInput("x", kIrInputRequired);
  reshape->AppendIrInput("shape", kIrInputRequired);
  reshape->SetOpKernelLibName(ge::kEngineNameGeLocal);
  reshape->SetOpEngineName(kEngineNameGeLocal);
  reshape->SetStreamId(0);
  reshape->SetOpInferDepends({"shape"});  // 用V1的方式设置值依赖
  AttrUtils::SetListInt(reshape, ge::ATTR_NAME_RECV_EVENT_IDS, {0});

  auto root_netoutput = OP_CFG(ge::NETOUTPUT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("root_netoutput");
  root_netoutput->SetSrcName({"reshape"});
  root_netoutput->SetSrcIndex({0});
  root_netoutput->SetStreamId(0);

  DEF_GRAPH(root) {
    CHAIN(NODE(data_1)->NODE(reshape)->NODE(root_netoutput));
    CHAIN(NODE(const_2)->EDGE(0, 1)->NODE(reshape));
  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);
  return root_graph;
}
/*
 *  data1(s0)  data2(streamid=1)(send_id_list:[0])
 *         \   /
 *         add1(streamid=0)(recive_id_list:[0])
 *          |
 *       netoutput(streamid=0)
 */
ComputeGraphPtr ShareGraph::MultiStreamGraphWithFirstEventSyncGraph(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 1;

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
                };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data1->GetOpDescBarePtr()->SetStreamId(0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data2->GetOpDescBarePtr()->SetStreamId(1);
  AttrUtils::SetListInt(data2->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {0});

  auto add1 = graph->FindNode("add1");
  AddCompileResult(add1, false);
  SetNoStorage(add1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  add1->GetOpDesc()->SetOpEngineName(kEngineNameAiCore);
  add1->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(add1->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  add1->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"add1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  net_output->GetOpDescBarePtr()->SetStreamId(0);

  SetGraphOutShapeRange(graph);
  graph->SetGraphUnknownFlag(true);
  return graph;
}
/*
 *        data1(s0)                            data2(s0)(send_id_list:[1])
 *               \                                  |
 *             relu(streamid=0)(send_id_list:[0])   |
 *             /          \                         |
 *   netoutput(streamid=0)  add(s1)(recive_id_list:[0,1])
 */
ComputeGraphPtr ShareGraph::MultiStreamGraphWithLastEventSyncGraph(int64_t &stream_num, int64_t &event_num) {
  stream_num = 2;
  event_num = 2;

  auto relu = OP_CFG("Relu")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1})
      .InCnt(1)
      .OutCnt(1)
      .Build("relu");
  relu->SetOpKernelLibName(ge::kEngineNameAiCore);
  relu->SetOpEngineName(kEngineNameAiCore);
  relu->SetStreamId(0);
  AttrUtils::SetListInt(relu, ge::ATTR_NAME_SEND_EVENT_IDS, {0});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE(relu)->NODE("add1", "Add"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE(relu)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  data1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data1->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data1->GetOpDescBarePtr()->SetStreamId(0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  data2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  data2->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  data2->GetOpDescBarePtr()->SetStreamId(0);
  AttrUtils::SetListInt(data2->GetOpDesc(), ge::ATTR_NAME_SEND_EVENT_IDS, {1});

  auto add = graph->FindNode("add1");
  AddCompileResult(add, false);
  SetNoStorage(add->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {-1, -1 - 1, -1});
  add->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCpuTf);
  add->GetOpDesc()->SetOpEngineName(kEngineNameAiCpuTf);
  add->GetOpDescBarePtr()->SetStreamId(1);
  AttrUtils::SetListInt(add->GetOpDesc(), ge::ATTR_NAME_RECV_EVENT_IDS, {0,1});
  add->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto relu_node = graph->FindFirstNodeMatchType("Relu");
  AddCompileResult(relu_node, false);
  relu_node->GetOpDesc()->SetWorkspaceBytes({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"relu"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  net_output->GetOpDescBarePtr()->SetStreamId(0);

  SetGraphOutShapeRange(graph);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *       NetOutput
 *       |        \
 *       |       Add
 *       |0     1/ \
 *      MinimumGrad |
 *     /    |    \  |
 * data0  data1  data2
 */
ge::ComputeGraphPtr ShareGraph::BuildStaticMinimumGradAndAddGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->NODE("mg", "MinimumGrad")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")->NODE("mg", "MinimumGrad"));
    CHAIN(NODE("data2", "Data")->NODE("mg", "MinimumGrad"));

    CHAIN(NODE("mg", "MinimumGrad")->EDGE(1, 0)->NODE("add0", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add0", "Add"));
  };
  auto graph = ToComputeGraph(g1);

  for (int32_t i = 0; i < 3; ++i) {
    auto node_name = "data" + std::to_string(i);
    auto data_node = graph->FindNode(node_name);
    AttrUtils::SetInt(data_node->GetOpDesc(), "index", i);
    SetNoStorage(data_node->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
    data_node->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  }

  auto add0 = graph->FindNode("add0");
  add0->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add0->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(add0->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  AttrUtils::SetStr(add0->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(add0, false);

  auto mg = graph->FindNode("mg");
  mg->GetOpDesc()->MutableAllInputName() = {{"grads", 0}, {"x1", 1}, {"x2", 2}};
  mg->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  SetNoStorage(mg->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});
  AttrUtils::SetBool(mg->GetOpDesc(), "grad_x", true);
  AttrUtils::SetBool(mg->GetOpDesc(), "grad_y", true);
  AttrUtils::SetStr(mg->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  AddCompileResult(mg, false);

  auto noutput = graph->FindNode("NetOutput");
  SetNoStorage(noutput->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4});

  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"mg", "add0"});
  graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 0});
  return graph;
}

Graph ShareGraph::BuildCVParallelGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)
              ->NODE("aic1", RELU)
              ->NODE("reshape", RESHAPE)
              ->NODE("aiv1", RELU)
              ->NODE("aic2", RELU)
              ->NODE("output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->NODE("aiv2", RELU)->NODE("aiv3", RELU)->NODE("output"));
  };
  auto graph = ToGeGraph(g1);
  auto root_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_ASSERT_TRUE(root_graph != nullptr);
  auto aiv1 = root_graph->FindNode("aiv1");
  GE_ASSERT_TRUE(aiv1 != nullptr);
  auto aiv2 = root_graph->FindNode("aiv2");
  GE_ASSERT_TRUE(aiv2 != nullptr);
  auto aiv3 = root_graph->FindNode("aiv3");
  GE_ASSERT_TRUE(aiv3 != nullptr);

  AttrUtils::SetStr(aiv1->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAiv);
  AttrUtils::SetStr(aiv2->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAiv);
  AttrUtils::SetStr(aiv3->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  AttrUtils::SetBool(aiv3->GetOpDesc(), "_mix_is_aiv", true);

  auto aic1 = root_graph->FindNode("aic1");
  GE_ASSERT_TRUE(aiv1 != nullptr);
  auto aic2 = root_graph->FindNode("aic2");
  GE_ASSERT_TRUE(aiv2 != nullptr);
  AttrUtils::SetStr(aic1->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAicore);
  AttrUtils::SetStr(aic2->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAicore);
  return graph;
}

Graph ShareGraph::BuildCVSerialGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)
              ->NODE("aic1", RELU)
              ->NODE("aiv1", RELU)
              ->NODE("aic2", RELU)
              ->NODE("aiv2", RELU)
              ->NODE("output", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto root_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_ASSERT_TRUE(root_graph != nullptr);
  auto aiv1 = root_graph->FindNode("aiv1");
  GE_ASSERT_TRUE(aiv1 != nullptr);
  auto aiv2 = root_graph->FindNode("aiv2");
  GE_ASSERT_TRUE(aiv2 != nullptr);

  AttrUtils::SetStr(aiv1->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAiv);
  AttrUtils::SetStr(aiv2->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAiv);

  auto aic1 = root_graph->FindNode("aic1");
  GE_ASSERT_TRUE(aiv1 != nullptr);
  auto aic2 = root_graph->FindNode("aic2");
  GE_ASSERT_TRUE(aiv2 != nullptr);
  AttrUtils::SetStr(aic1->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAicore);
  AttrUtils::SetStr(aic2->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kTaskTypeAicore);
  return graph;
}

/*
 * 子图构造
 * data -> sqrt -> output
 */
ComputeGraphPtr ShareGraph::BuildSubGraph(const std::string& name, int64_t parent_node_index) {
  auto subgraph_name = name + "_subgraph";
  auto graph = std::make_shared<ge::ComputeGraph>(subgraph_name);
  auto data = NodeBuilder(subgraph_name + "_data", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 0)
                      .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)
                      .Output()
                      .Build(graph);
  auto sqrt = NodeBuilder(subgraph_name + "_sqrt1", SQRT)
              .Input(data)
              .Output()
              .Build(graph);
  auto output = NodeBuilder(subgraph_name + "_output", NETOUTPUT).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                        .Input(sqrt).Build(graph);
  AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  output->GetOpDesc()->SetSrcName({subgraph_name + "_sqrt1"});
  output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 * 含partitioncall的子图构造
 *
 * data -> partitioncall -> output
 *              |
 *     data -> sqrt -> output
 */
ComputeGraphPtr ShareGraph::BuildNestPartitioncallSubGraph(const ComputeGraphPtr &main_graph, const std::string &name) {
  auto subgrpah_name = name + "_with_partitioncall_subgraph";
  auto graph = std::make_shared<ge::ComputeGraph>(subgrpah_name);
  auto data = NodeBuilder(subgrpah_name + "_data", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 0)
                      .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                      .Output()
                      .Build(graph);

  auto sub_grpah = BuildSubGraph(subgrpah_name + "_particall");
  auto partitioncall = NodeBuilder(name + "_partitioncall", ge::PARTITIONEDCALL)
                      .Input(data)
                      .Attr(ge::ATTR_STAGE_LEVEL, 1)
                      .Output()
                      .Attr("f", sub_grpah)
                      .Build(graph);
  auto output = NodeBuilder(subgrpah_name + "_output", NETOUTPUT).Input(partitioncall)
                        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).Build(graph);
  AttrUtils::SetInt(output->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  output->GetOpDesc()->SetSrcName({name + "_sqrt1"});
  output->GetOpDesc()->SetSrcIndex({0});
  (void)main_graph->AddSubGraph(sub_grpah);
  return graph;
}

// if的条件输入是data
ComputeGraphPtr ShareGraph::BuildNestIfGraph() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data_0", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data_1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);

  auto cast = NodeBuilder("cast1", CAST).Input(data).Output().Build(main_graph);
  auto then_graph = BuildSubGraph("then",1);
  auto else_graph = BuildSubGraph("else", 1);
  auto if_node = NodeBuilder("if1", IF)
                  .Input(cast)
                  .Input(data1)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(main_graph);

  auto sqrt = NodeBuilder("abs1", "Abs").Input(if_node).Output().Build(main_graph);
  auto output = NodeBuilder("output", NETOUTPUT).Input(sqrt).Build(main_graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  for (auto &node : main_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    for (auto &td : op_desc->GetAllInputsDesc()) {
      td.SetOriginFormat(td.GetFormat());
    }
  }
  return main_graph;
}

// case的条件输入是data
ComputeGraphPtr ShareGraph::BuildNestCaseGraph() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data_0", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data_1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);

  auto batch1 = BuildSubGraph("batch1",1);
  auto batch2 = BuildSubGraph("batch2", 1);
  auto if_node = NodeBuilder("case1", CASE)
                  .Input(data)
                  .Input(data1)
                  .Output()
                  .Attr("batch1", batch1)
                  .Attr("batch2", batch2)
                  .Build(main_graph);

  auto sqrt = NodeBuilder("abs1", "Abs").Input(if_node).Output().Build(main_graph);
  auto output = NodeBuilder("output", NETOUTPUT).Input(sqrt).Build(main_graph);
  (void)main_graph->AddSubGraph(batch1);
  (void)main_graph->AddSubGraph(batch2);
  for (auto &node : main_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    for (auto &td : op_desc->GetAllInputsDesc()) {
      td.SetOriginFormat(td.GetFormat());
    }
  }
  return main_graph;
}

// if的条件输入是其它算子的输出
ComputeGraphPtr ShareGraph::BuildNestIfGraph1() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data_0", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data_1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);
  auto data2 =  NodeBuilder("data_2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 2).Output().Build(main_graph);

  auto sub = NodeBuilder("sub", SUB).Input(data).Input(data1).Output().Build(main_graph);

  auto then_graph = BuildSubGraph("then",1);
  auto else_graph = BuildSubGraph("else", 1);
  auto if_node = NodeBuilder("if1", IF)
                  .Input(sub)
                  .Input(data2)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(main_graph);

  auto output = NodeBuilder("output", NETOUTPUT).Input(if_node).Build(main_graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  for (auto &node : main_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    for (auto &td : op_desc->GetAllInputsDesc()) {
      td.SetOriginFormat(td.GetFormat());
    }
  }
  return  main_graph;
}


// 子图含有if节点, if的条件输入是data
ComputeGraphPtr ShareGraph::BuildNestIfSubGraph(const ge::ComputeGraphPtr &main_graph, const std::string &name) {
  auto subgrpah_name = name + "_with_if_subgraph";
  auto graph = std::make_shared<ge::ComputeGraph>(subgrpah_name);
  auto data = NodeBuilder(subgrpah_name + "_data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0).Output().Build(graph);
  auto data1 = NodeBuilder(subgrpah_name + "_data1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1).Output().Build(graph);

  auto then_graph = BuildSubGraph(subgrpah_name + "_then",1);
  auto else_graph = BuildSubGraph(subgrpah_name + "_else", 1);
  auto if_node = NodeBuilder(subgrpah_name + "_if2", IF)
                  .Input(data)
                  .Input(data1)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(graph);

  auto output = NodeBuilder(subgrpah_name + "_output", NETOUTPUT).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                        .Input(if_node).Build(graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  return  graph;
}

// if子图嵌套if算子，且条件输入都是data
ComputeGraphPtr ShareGraph::BuildNestIfGraph2 () {
  auto main_graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data_0", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data_1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);
  auto then_graph = BuildNestIfSubGraph(main_graph, "then");
  auto else_graph = BuildNestIfSubGraph(main_graph, "else");
  auto if_node = NodeBuilder("if1", IF)
                  .Input(data)
                  .Input(data1)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(main_graph);
  auto output = NodeBuilder("output", NETOUTPUT).Input(if_node).Build(main_graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  for (auto &node : main_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    for (auto &td : op_desc->GetAllInputsDesc()) {
      td.SetOriginFormat(td.GetFormat());
    }
  }
  return  main_graph;
}


// 子图中包含if节点, 且if的条件输入是其它算子的输出
ComputeGraphPtr ShareGraph::BuildNestIfSubGraph1(const ge::ComputeGraphPtr &main_graph, const std::string &name) {
  auto graph_name = name + "_with_if_subgraph1";
  auto graph = std::make_shared<ge::ComputeGraph>(graph_name);
  auto data = NodeBuilder(graph_name + "_data_0", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0).Output().Build(graph);
  auto data1 = NodeBuilder(graph_name + "_data_1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1).Output().Build(graph);
  auto data2 = NodeBuilder(graph_name + "_data_2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 2)
                     .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 2).Output().Build(graph);

  auto sub = NodeBuilder(graph_name + "_sub", SUB).Input(data).Input(data1).Output().Build(graph);

  auto then_graph = BuildSubGraph(graph_name + "_then",1);
  auto else_graph = BuildSubGraph(graph_name + "_else", 1);
  auto if_node = NodeBuilder(graph_name + "_if1", IF)
                  .Input(sub)
                  .Input(data2)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(graph);

  auto output = NodeBuilder(graph_name + "_output", NETOUTPUT).Input(if_node)
                        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).Build(graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  return  graph;
}

// if子图嵌套if算子，根图的if条件输入是data，子图的if条件输入不是data
ComputeGraphPtr ShareGraph::BuildNestIfGraph3() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data_0", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data_1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);
  auto data2 = NodeBuilder("data_2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);

  auto then_graph = BuildNestIfSubGraph1(main_graph, "then");
  auto else_graph = BuildNestIfSubGraph1(main_graph, "else");
  auto if_node = NodeBuilder("if1", IF)
                  .Input(data)
                  .Input(data1)
                  .Input(data2)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(main_graph);
  auto output = NodeBuilder("output", NETOUTPUT).Input(if_node).Build(main_graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  for (auto &node : main_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    for (auto &td : op_desc->GetAllInputsDesc()) {
      td.SetOriginFormat(td.GetFormat());
    }
  }
  return  main_graph;
}

/*
 * partitioncall子图嵌套
 *
 */
ComputeGraphPtr ShareGraph::BuildNestedPartitionedCallTwice() {
  auto graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 0)
                      .Output()
                      .Build(graph);
  auto sub_graph = BuildNestPartitioncallSubGraph(graph, "subgraph");
  auto partitioncall = NodeBuilder("partitioncall", ge::PARTITIONEDCALL)
                      .Input(data)
                      .Output()
                      .Attr("f", sub_graph)
                      .Build(graph);
  auto output = NodeBuilder("output", NETOUTPUT).Input(partitioncall).Build(graph);
  (void)graph->AddSubGraph(sub_graph);
  return graph;
}


/*
 * if子图嵌套
 *
 *
 * data -> if -> output
 *          |
 * data1 ___|
 *
 */
ComputeGraphPtr ShareGraph::BuildIfWithNestedPartitionedCall() {
  auto main_graph = std::make_shared<ComputeGraph>("root");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);

  auto then_graph = BuildNestPartitioncallSubGraph(main_graph, "then");
  auto else_graph = BuildNestPartitioncallSubGraph(main_graph, "else");
  auto if_node = NodeBuilder("if1", IF)
                  .Input(data)
                  .Input(data1)
                  .Output()
                  .Attr("then_graph", then_graph)
                  .Attr("else_graph", else_graph)
                  .Build(main_graph);

  auto output = NodeBuilder("ouput", NETOUTPUT).Input(if_node).Build(main_graph);
  (void)main_graph->AddSubGraph(then_graph);
  (void)main_graph->AddSubGraph(else_graph);
  return main_graph;
}

// case子图嵌套partitioncall
ComputeGraphPtr ShareGraph::BuildCaseWithNestedPartitionedCall() {
  auto main_graph = std::make_shared<ComputeGraph>("root");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data1 = NodeBuilder("data1", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);

  auto batch1 = BuildNestPartitioncallSubGraph(main_graph, "batch1");
  auto batch2 = BuildNestPartitioncallSubGraph(main_graph, "batch2");
  auto case_node = NodeBuilder("case", "Case")
    .Input(data).Input(data1)
    .Output()
    .Attr("batch1", batch1)
    .Attr("batch2",  batch2)
    .Build(main_graph);
  auto output = NodeBuilder("ouput", NETOUTPUT).Input(case_node).Build(main_graph);
  (void)main_graph->AddSubGraph(batch1);
  (void)main_graph->AddSubGraph(batch2);
  return main_graph;
}
}  // namespace gert
