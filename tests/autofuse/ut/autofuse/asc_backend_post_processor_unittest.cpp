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
#include "graph/utils/graph_utils_ex.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "ascir_ops.h"
#include "utils/autofuse_utils.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "post_process/scheduler_adapter/adaption_fallback_scalar.h"
#include "post_process/asc_backend_post_processor.h"
#include "utils/autofuse_attrs.h"
#include "ascgen_log.h"
#include "attribute_group/attr_group_shape_env.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"
#include "ascir_registry.h"
#include "platform_context.h"

namespace ge {
using namespace autofuse;
class AscBackendPostProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
//    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
//      setenv("DUMP_GE_GRAPH", "1", 1);
//      setenv("DUMP_GRAPH_LEVEL", "1", 1);
//      setenv("DUMP_GRAPH_PATH", "./", 1);
  }
  void TearDown() override {
//    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
//     unsetenv("DUMP_GE_GRAPH");
//     unsetenv("DUMP_GRAPH_LEVEL");
//     unsetenv("DUMP_GRAPH_PATH");
  }
  std::string work_path;
};

namespace {
  class GraphBuilder {
   public:
    GraphBuilder(const std::string &name) {
      graph_ = std::make_shared<ComputeGraph>(name);
    }
  
    GraphBuilder(const std::string &name, const std::string &node_type) {
      graph_ = std::make_shared<ComputeGraph>(name);
      node_type_ = node_type;
    }
  
    NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                    Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                    std::vector<int64_t> shape = {1, 1, 1, 1}) {
      auto tensor_desc = std::make_shared<GeTensorDesc>();
      tensor_desc->SetShape(GeShape(std::move(shape)));
      tensor_desc->SetFormat(format);
      tensor_desc->SetDataType(data_type);
      tensor_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
  
      auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscBackend");
      for (int i = 0; i < in_cnt; ++i) {
        op_desc->AddInputDesc(tensor_desc->Clone());
      }
      for (int i = 0; i < out_cnt; ++i) {
        op_desc->AddOutputDesc(tensor_desc->Clone());
      }
      op_desc->AddInferFunc([](Operator &op) { return GRAPH_SUCCESS; });
      return graph_->AddNode(op_desc);
    }
  
    void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
      GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
    }
  
    NodePtr AddNodeByIr(const std::string &op_name, const std::string &op_type) {
      auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
      if (op.IsEmpty()) {
        return nullptr;
      }
      OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
      return graph_->AddNode(op_desc);
    }
  
    void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
      GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
    }
  
    ComputeGraphPtr GetGraph() {
      graph_->TopologicalSorting();
      return graph_;
    }
  
    static void AddSubgraph(const ComputeGraphPtr &graph, const string &call_name, const ComputeGraphPtr &subgraph) {
      const auto &call_node = graph->FindNode(call_name);
      if (call_node == nullptr) {
        return;
      }
      call_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);
  
      size_t index = call_node->GetOpDesc()->GetSubgraphInstanceNames().size();
      call_node->GetOpDesc()->AddSubgraphName(subgraph->GetName());
      call_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph->GetName());
  
      subgraph->SetParentNode(call_node);
      subgraph->SetParentGraph(graph);
      GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph);
    }
  
   private:
    ComputeGraphPtr graph_;
    std::string node_type_;
  };

  std::shared_ptr<AscGraph> CreatAddAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output("x_output");
    x_output.x = x_out.y;
    x_output.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output.attr.sched.loop_axis = c.id;
    x_output.y.dtype = DT_FLOAT16;
    *x_output.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output.y.repeats = {A, B, C, D, E};
    *x_output.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
/*
 *      data
 *    /     \
 * load1   load2
 *   |       |
 * split1  split2
 *   |       |
 * store1  store2
*/
  std::shared_ptr<AscGraph> CreatSplitDoubleOutputAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Split split1("x1split");
    split1.InstanceOutputy(1);
    split1.ir_attr.SetIndex(0);
    split1.x = x1Local.y;
    split1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    split1.y[0].dtype = DT_FLOAT16;
    AscOutputAttrDataType x1Local_output_data_type(&split1, 0);
    x1Local_output_data_type = ge::DT_FLOAT16;
    *split1.y[0].axis = {a.id, b.id, c.id, d.id, e.id};
    *split1.y[0].repeats = {A, B, C, D, E};
    *split1.y[0].strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x1_out("x1_out");
    x1_out.x = split1.y[0];
    x1_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1_out.attr.sched.loop_axis = c.id;
    x1_out.y.dtype = DT_FLOAT16;
    *x1_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1_out.y.repeats = {A, B, C, D, E};
    *x1_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Split split2("x2split");
    split2.InstanceOutputy(1);
    split2.ir_attr.SetIndex(1);
    split2.x = x1Local.y;
    split2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    split2.y[0].dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&split2, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *split2.y[0].axis = {a.id, b.id, c.id, d.id, e.id};
    *split2.y[0].repeats = {A, B, C, D, E};
    *split2.y[0].strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x2_out("x2_out");
    x2_out.x = split2.y[0];
    x2_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_out.attr.sched.loop_axis = c.id;
    x2_out.y.dtype = DT_FLOAT16;
    *x2_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2_out.y.repeats = {A, B, C, D, E};
    *x2_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x1_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT16;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x2_out.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT16;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x1_out_node = graph.FindNode("x_output1");
    auto x2_out_node = graph.FindNode("x_output2");
    auto compute_graph = x2_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x1_out_node, 0},{x2_out_node, 1}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithDataTensorOK(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_1", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT;
    *x2.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x3("x3_1", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {A, B, C, D, E};
    *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, D, E};
    *x1Local.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT;
    *x2Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Local.y.repeats = {A, C, B, D, E};
    *x2Local.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("x3Local_2");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT;
    *x3Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x3Local.y.repeats = {A, C, B, D, E};
    *x3Local.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul1("mul1");
    mul1.x1 = x1Local.y;
    mul1.x2 = x2Local.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_FLOAT16;
    *mul1.y.axis = {};
    *mul1.y.repeats = {};
    *mul1.y.strides = {};

    ge::ascir_op::Mul mul2("mul2");
    mul2.x1 = x1Local.y;
    mul2.x2 = x3Local.y;
    mul2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul2_output_data_type(&mul2, 0);
    mul2_output_data_type = ge::DT_FLOAT16;
    *mul2.y.axis = {};
    *mul2.y.repeats = {};
    *mul2.y.strides = {};

    ge::ascir_op::Mul mul3("mul3");
    mul3.x1 = x2Local.y;
    mul3.x2 = x3Local.y;
    mul3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul3.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul3_output_data_type(&mul3, 0);
    mul3_output_data_type = ge::DT_FLOAT16;
    *mul3.y.axis = {};
    *mul3.y.repeats = {};
    *mul3.y.strides = {};
  
    ge::ascir_op::Store x_store1("x_store1");
    x_store1.x = mul1.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_FLOAT;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store2("x_store2");
    x_store2.x = mul2.y;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_FLOAT;
    *x_store2.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x_store2.y.repeats = {A, C, B, D, E};
    *x_store2.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store3("x_store3");
    x_store3.x = mul3.y;
    x_store3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store3.attr.sched.loop_axis = c.id;
    x_store3.y.dtype = DT_FLOAT;
    *x_store3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store3.y.repeats = {A, B, C, D, E};
    *x_store3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_store2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_store3.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto x_out_node3 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}, {x_out_node3, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithLoadAndStoreHasTranspose(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {b.id, c.id, a.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {C * A * D * E, A * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {b.id, c.id, a.id, d.id, e.id};
    *x1Local.y.repeats = {B, C, A, D, E};
    *x1Local.y.strides = {C * A * D * E, A * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("x1_abs");
    abs1.x = x1Local.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT;
    *abs1.y.axis = {b.id, c.id, a.id, d.id, e.id};
    *abs1.y.repeats = {B, C, A, D, E};
    *abs1.y.strides = {C * A * D * E, A * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("x2_1", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul1("mul1");
    mul1.x1 = abs1.y;
    mul1.x2 = x2Local.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_FLOAT16;
    *mul1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul1.y.repeats = {A, B, C, D, E};
    *mul1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store1("x_store1");
    x_store1.x = mul1.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_FLOAT;
    *x_store1.y.axis = {b.id, c.id, a.id, d.id, e.id};
    *x_store1.y.repeats = {B, C, A, D, E};
    *x_store1.y.strides = {C * A * D * E, A * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    // ge::ascir_op::Store x_store2("x_store2");
    // x_store2.x = mul1.y;
    // x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    // x_store2.attr.sched.loop_axis = c.id;
    // x_store2.y.dtype = DT_FLOAT;
    // *x_store2.y.axis = {b.id, c.id, a.id, d.id, e.id};
    // *x_store2.y.repeats = {B, C, A, D, E};
    // *x_store2.y.strides = {C * A * D * E, A * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    // auto x_out_node2 = graph.FindNode("x_store2");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}};
    // std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}};
    compute_graph->SetOutputSize(1U);
    // compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithNoAxisTranspose(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT8;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};
  
    ge::ascir_op::Data x2("x2_1", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_INT8;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};
    ge::ascir_op::Data x3("x3_1", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_INT8;
    *x3.y.axis = {};
    *x3.y.repeats = {};
    *x3.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT8;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_INT8;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("x3Local_2");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_INT8;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul1("mul1");
    mul1.x1 = x1Local.y;
    mul1.x2 = x2Local.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_INT8;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_INT8;
    *mul1.y.axis = {};
    *mul1.y.repeats = {};
    *mul1.y.strides = {};

    ge::ascir_op::Mul mul2("mul2");
    mul2.x1 = x1Local.y;
    mul2.x2 = x3Local.y;
    mul2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul2.y.dtype = DT_INT8;
    AscOutputAttrDataType mul2_output_data_type(&mul2, 0);
    mul2_output_data_type = ge::DT_INT8;
    *mul2.y.axis = {};
    *mul2.y.repeats = {};
    *mul2.y.strides = {};

    ge::ascir_op::Mul mul3("mul3");
    mul3.x1 = x2Local.y;
    mul3.x2 = x3Local.y;
    mul3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul3.y.dtype = DT_INT8;
    AscOutputAttrDataType mul3_output_data_type(&mul3, 0);
    mul3_output_data_type = ge::DT_INT8;
    *mul3.y.axis = {};
    *mul3.y.repeats = {};
    *mul3.y.strides = {};
  
    ge::ascir_op::Store x_store1("x_store1");
    x_store1.x = mul1.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_INT8;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store2("x_store2");
    x_store2.x = mul2.y;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_INT8;
    *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store2.y.repeats = {A, B, C, D, E};
    *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store3("x_store3");
    x_store3.x = mul3.y;
    x_store3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store3.attr.sched.loop_axis = c.id;
    x_store3.y.dtype = DT_INT8;
    *x_store3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store3.y.repeats = {A, B, C, D, E};
    *x_store3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_store2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_store3.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto x_out_node3 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}, {x_out_node3, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithBroadcastAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, ONE, C, D, ONE};
    *x1.y.strides = {C * D, ZERO, D, ONE, ZERO};
  
    ge::ascir_op::Data x2("x2_1", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT;
    *x2.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2.y.repeats = {ONE, B, C, D, E};
    *x2.y.strides = {ZERO, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x3("x3_1", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {ONE, ONE, C, D, E};
    *x3.y.strides = {ZERO, ZERO, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT;
    *x2Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("x3Local_2");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT;
    *x3Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul1("mul1");
    mul1.x1 = x1Local.y;
    mul1.x2 = x2Local.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_FLOAT16;
    *mul1.y.axis = {};
    *mul1.y.repeats = {};
    *mul1.y.strides = {};

    ge::ascir_op::Mul mul2("mul2");
    mul2.x1 = x1Local.y;
    mul2.x2 = x3Local.y;
    mul2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul2_output_data_type(&mul2, 0);
    mul2_output_data_type = ge::DT_FLOAT16;
    *mul2.y.axis = {};
    *mul2.y.repeats = {};
    *mul2.y.strides = {};

    ge::ascir_op::Mul mul3("mul3");
    mul3.x1 = x2Local.y;
    mul3.x2 = x3Local.y;
    mul3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul3.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul3_output_data_type(&mul3, 0);
    mul3_output_data_type = ge::DT_FLOAT16;
    *mul3.y.axis = {};
    *mul3.y.repeats = {};
    *mul3.y.strides = {};
  
    ge::ascir_op::Store x_store1("x_store1");
    x_store1.x = mul1.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_FLOAT;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store2("x_store2");
    x_store2.x = mul2.y;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_FLOAT;
    *x_store2.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x_store2.y.repeats = {A, B, C, D, E};
    *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store3("x_store3");
    x_store3.x = mul3.y;
    x_store3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store3.attr.sched.loop_axis = c.id;
    x_store3.y.dtype = DT_FLOAT;
    *x_store3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store3.y.repeats = {A, B, C, D, E};
    *x_store3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_store2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_store3.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto x_out_node3 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}, {x_out_node3, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWith2InputTranspose(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression C = graph.CreateSizeVar("C");
    const Expression B = graph.CreateSizeVar("B");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto c = graph.CreateAxis("C", C);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, ONE, E};
    *x1Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x2("x1_2_mul", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};

    ge::ascir_op::Load x2Local("x2Local_2_mul");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Local.y.repeats = {A, C, B, ONE, E};
    *x2Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {A, C, B, D, E};
    *x1Broadcast.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Broadcast x2Broadcast("x2Broadcast_2_mul");
    x2Broadcast.x = x2Local.y;
    x2Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x2Broadcast.y.dtype = DT_FLOAT16;
    *x2Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Broadcast.y.repeats = {A, C, B, D, E};
    *x2Broadcast.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x2Transpose("x2Transpose_2_mul");
    x2Transpose.x = x2Broadcast.y;
    x2Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Transpose.y.dtype = DT_FLOAT16;
    *x2Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Transpose.y.repeats = {A, B, C, D, E};
    *x2Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x2Transpose.y;
    abs2.attr.sched.axis = {};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = x2Transpose.y;
    abs3.attr.sched.axis = {};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {};
    *abs3.y.repeats = {};
    *abs3.y.strides = {};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = abs1.y;
    add1.attr.sched.axis = {};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {};
    *add1.y.repeats = {};
    *add1.y.strides = {};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add1.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, C, B, D, E};
    *x_out.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out_out("x_out1_mul_output");
    x_out_out.x = x_out.y;
    x_out_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out_out.attr.sched.loop_axis = c.id;
    x_out_out.y.dtype = DT_FLOAT16;
    *x_out_out.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x_out_out.y.repeats = {A, C, B, D, E};
    *x_out_out.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = abs3.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1_out("x_out2_mul_output");
    x_out1_out.x = x_out1.y;
    x_out1_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1_out.attr.sched.loop_axis = c.id;
    x_out1_out.y.dtype = DT_FLOAT16;
    *x_out1_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1_out.y.repeats = {A, B, C, D, E};
    *x_out1_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_out1_mul_output");
    auto x_out_node1 = graph.FindNode("x_out2_mul_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithTransposeDtypeSupported(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression C = graph.CreateSizeVar("C");
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto c = graph.CreateAxis("C", C);
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT64;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_INT64;
    *x1Local.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {C, A, B, ONE, E};
    *x1Local.y.strides = {A * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_INT64;
    *x1Broadcast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {C, A, B, D, E};
    *x1Broadcast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast x1Cast("x1Cast1");
    x1Cast.x = x1Broadcast.y;
    x1Cast.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Cast.y.dtype = DT_FLOAT;
    *x1Cast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Cast.y.repeats = {C, A, B, D, E};
    *x1Cast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Cast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {};
    abs1.y.dtype = DT_FLOAT;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {};
    abs2.y.dtype = DT_FLOAT;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = abs2.y;
    abs3.attr.sched.axis = {};
    abs3.y.dtype = DT_FLOAT;
    *abs3.y.axis = {};
    *abs3.y.repeats = {};
    *abs3.y.strides = {};

    ge::ascir_op::Cast Cast2("Cast2");
    Cast2.x = abs3.y;
    Cast2.attr.sched.axis = {};
    Cast2.y.dtype = DT_INT64;
    *Cast2.y.axis = {};
    *Cast2.y.repeats = {};
    *Cast2.y.strides = {};

    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = Cast2.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_INT64;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1_out("x_out2_mul_output");
    x_out1_out.x = x_out1.y;
    x_out1_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1_out.attr.sched.loop_axis = c.id;
    x_out1_out.y.dtype = DT_INT64;
    *x_out1_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1_out.y.repeats = {A, B, C, D, E};
    *x_out1_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_out2_mul_output");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWith2TransposeInLoadStore(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, d.id, e.id, c.id};
    *x1Local.y.repeats = {A, B, ONE, E, C};
    *x1Local.y.strides = {B * E * C, E * C, ZERO, C, ONE};

    ge::ascir_op::Data x2("x1_2_mul", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};

    ge::ascir_op::Load x2Local("x2Local_2_mul");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, ONE, E};
    *x2Local.y.strides = {B * C * E, C * E, E, ZERO, ONE};
  
    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Local.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs1.y.repeats = {A, B, C, D, E};
    *abs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x2Local.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs2.y.repeats = {A, B, C, D, E};
    *abs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = abs1.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add1.y.repeats = {A, B, C, D, E};
    *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add1.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, d.id, e.id, c.id};
    *x_out.y.repeats = {A, B, D, E, C};
    *x_out.y.strides = {B * D * E * C, D * E * C, E * C, C, ONE};
  
    ge::ascir_op::Output x_out_out("x_out1_mul_output");
    x_out_out.x = x_out.y;
    x_out_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out_out.attr.sched.loop_axis = c.id;
    x_out_out.y.dtype = DT_FLOAT16;
    *x_out_out.y.axis = {};
    *x_out_out.y.repeats = {};
    *x_out_out.y.strides = {};

    auto x_out_node = graph.FindNode("x_out1_mul_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWith2InputTransposeMulReference(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x0("x0_1_mul", graph);
    x0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x0.attr.sched.loop_axis = c.id;
    x0.y.dtype = DT_FLOAT16;
    *x0.y.axis = {};
    *x0.y.repeats = {};
    *x0.y.strides = {};

    ge::ascir_op::Load x0Local("x0Local_2_mul");
    x0Local.x = x0.y;
    x0Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x0Local.y.dtype = DT_FLOAT16;
    *x0Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x0Local.y.repeats = {A, C, B, ONE, E};
    *x0Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, ONE, E};
    *x1Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Add add0("add_0_mul");
    add0.x1 = x0Local.y;
    add0.x2 = x1Local.y;
    add0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add0.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add0_mul_output_data_type(&add0, 0);
    add0_mul_output_data_type = ge::DT_FLOAT16;
    *add0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add0.y.repeats = {A, B, C, D, E};
    *add0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("x1_2_mul", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};

    ge::ascir_op::Load x2Local("x2Local_2_mul");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = x2Local.y;
    abs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs3.y.repeats = {A, B, C, D, E};
    *abs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs3.y;
    add1.x2 = add0.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add1.y.repeats = {A, B, C, D, E};
    *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add1.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out_out("x_out1_mul_output");
    x_out_out.x = x_out.y;
    x_out_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out_out.attr.sched.loop_axis = c.id;
    x_out_out.y.dtype = DT_FLOAT16;
    *x_out_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out_out.y.repeats = {A, B, C, D, E};
    *x_out_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = add0.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1_out("x_out2_mul_output");
    x_out1_out.x = x_out1.y;
    x_out1_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1_out.attr.sched.loop_axis = c.id;
    x_out1_out.y.dtype = DT_FLOAT16;
    *x_out1_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1_out.y.repeats = {A, B, C, D, E};
    *x_out1_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_out1_mul_output");
    auto x_out_node1 = graph.FindNode("x_out2_mul_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWith2InputTranspose1ScalarNoTranspose(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression C = graph.CreateSizeVar("C");
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto c = graph.CreateAxis("C", C);
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {C, A, B, ONE, E};
    *x1Local.y.strides = {A * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x2("x1_2_mul", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};

    ge::ascir_op::Load x2Local("x2Local_2_mul");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x2Local.y.repeats = {C, A, B, ONE, E};
    *x2Local.y.strides = {A * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Scalar scalar("scalar", graph);
    scalar.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    scalar.attr.sched.loop_axis = c.id;
    scalar.y.dtype = DT_FLOAT16;
    *scalar.y.axis = {};
    *scalar.y.repeats = {};
    *scalar.y.strides = {};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {C, A, B, D, E};
    *x1Broadcast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Broadcast x2Broadcast("x2Broadcast_2_mul");
    x2Broadcast.x = x2Local.y;
    x2Broadcast.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x2Broadcast.y.dtype = DT_FLOAT16;
    *x2Broadcast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x2Broadcast.y.repeats = {C, A, B, D, E};
    *x2Broadcast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x2Transpose("x2Transpose_2_mul");
    x2Transpose.x = x2Broadcast.y;
    x2Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Transpose.y.dtype = DT_FLOAT16;
    *x2Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Transpose.y.repeats = {A, B, C, D, E};
    *x2Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x2Transpose.y;
    abs2.attr.sched.axis = {};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = x2Transpose.y;
    abs3.attr.sched.axis = {};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {};
    *abs3.y.repeats = {};
    *abs3.y.strides = {};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = abs1.y;
    add1.attr.sched.axis = {};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {};
    *add1.y.repeats = {};
    *add1.y.strides = {};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = scalar.y;
    add2.attr.sched.axis = {};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add2_mul_output_data_type(&add2, 0);
    add2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {};
    *add2.y.repeats = {};
    *add2.y.strides = {};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {C, A, B, D, E};
    *x_out.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out_out("x_out1_mul_output");
    x_out_out.x = x_out.y;
    x_out_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out_out.attr.sched.loop_axis = c.id;
    x_out_out.y.dtype = DT_FLOAT16;
    *x_out_out.y.axis = {};
    *x_out_out.y.repeats = {};
    *x_out_out.y.strides = {};

    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = abs3.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1_out("x_out2_mul_output");
    x_out1_out.x = x_out1.y;
    x_out1_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1_out.attr.sched.loop_axis = c.id;
    x_out1_out.y.dtype = DT_FLOAT16;
    *x_out1_out.y.axis = {};
    *x_out1_out.y.repeats = {};
    *x_out1_out.y.strides = {};

    auto x_out_node = graph.FindNode("x_out1_mul_output");
    auto x_out_node1 = graph.FindNode("x_out2_mul_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWith2InputTranspose1ScalarHasTranspose(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {C, A, B, ONE, E};
    *x1Local.y.strides = {A * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x2("x1_2_mul", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};

    ge::ascir_op::Load x2Local("x2Local_2_mul");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x2Local.y.repeats ={C, A, B, ONE, E};
    *x2Local.y.strides = {A * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x3("x1_3_mul", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT16;
    *x3.y.axis = {};
    *x3.y.repeats = {};
    *x3.y.strides = {};

    ge::ascir_op::Load x3Local("x3Local_2_mul");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT16;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, ONE, E};
    *x3Local.y.strides = {B * C * E, C * E, E, ZERO, ONE};

    ge::ascir_op::Scalar scalar("scalar", graph);
    scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    scalar.attr.sched.loop_axis = c.id;
    scalar.y.dtype = DT_FLOAT16;
    *scalar.y.axis = {};
    *scalar.y.repeats = {};
    *scalar.y.strides = {};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {C, A, B, D, E};
    *x1Broadcast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Broadcast x2Broadcast("x2Broadcast_2_mul");
    x2Broadcast.x = x2Local.y;
    x2Broadcast.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x2Broadcast.y.dtype = DT_FLOAT16;
    *x2Broadcast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x2Broadcast.y.repeats = {C, A, B, D, E};
    *x2Broadcast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x2Transpose("x2Transpose_2_mul");
    x2Transpose.x = x2Broadcast.y;
    x2Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Transpose.y.dtype = DT_FLOAT16;
    *x2Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Transpose.y.repeats = {A, B, C, D, E};
    *x2Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x2Transpose.y;
    abs2.attr.sched.axis = {};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = x2Transpose.y;
    abs3.attr.sched.axis = {};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {};
    *abs3.y.repeats = {};
    *abs3.y.strides = {};

    ge::ascir_op::Add add0("add_0_mul");
    add0.x1 = abs2.y;
    add0.x2 = abs1.y;
    add0.attr.sched.axis = {};
    add0.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add0_mul_output_data_type(&add0, 0);
    add0_mul_output_data_type = ge::DT_FLOAT16;
    *add0.y.axis = {};
    *add0.y.repeats = {};
    *add0.y.strides = {};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = add0.y;
    add1.x2 = x3Local.y;
    add1.attr.sched.axis = {};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {};
    *add1.y.repeats = {};
    *add1.y.strides = {};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = scalar.y;
    add2.attr.sched.axis = {};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add2_mul_output_data_type(&add2, 0);
    add2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {};
    *add2.y.repeats = {};
    *add2.y.strides = {};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out_out("x_out1_mul_output");
    x_out_out.x = x_out.y;
    x_out_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out_out.attr.sched.loop_axis = c.id;
    x_out_out.y.dtype = DT_FLOAT16;
    *x_out_out.y.axis = {};
    *x_out_out.y.repeats = {};
    *x_out_out.y.strides = {};

    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = abs3.y;
    x_out1.attr.sched.axis = {c.id, a.id, b.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x_out1.y.repeats = {C, A, B, D, E};
    *x_out1.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1_out("x_out2_mul_output");
    x_out1_out.x = x_out1.y;
    x_out1_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1_out.attr.sched.loop_axis = c.id;
    x_out1_out.y.dtype = DT_FLOAT16;
    *x_out1_out.y.axis = {};
    *x_out1_out.y.repeats = {};
    *x_out1_out.y.strides = {};

    auto x_out_node = graph.FindNode("x_out1_mul_output");
    auto x_out_node1 = graph.FindNode("x_out2_mul_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWith2InputTranspose2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, ONE, E};
    *x1Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x2("x1_2_mul", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};

    ge::ascir_op::Load x2Local("x2Local_2_mul");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Local.y.repeats = {A, C, B, ONE, E};
    *x2Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Data x3("x1_3_mul", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT16;
    *x3.y.axis = {};
    *x3.y.repeats = {};
    *x3.y.strides = {};

    ge::ascir_op::Load x3Local("x3Local_2_mul");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT16;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, ONE, E};
    *x3Local.y.strides = {B * C * E, C * E, E, ZERO, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {A, C, B, D, E};
    *x1Broadcast.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Broadcast x2Broadcast("x2Broadcast_2_mul");
    x2Broadcast.x = x2Local.y;
    x2Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x2Broadcast.y.dtype = DT_FLOAT16;
    *x2Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Broadcast.y.repeats = {A, C, B, D, E};
    *x2Broadcast.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x2Transpose("x2Transpose_2_mul");
    x2Transpose.x = x2Broadcast.y;
    x2Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Transpose.y.dtype = DT_FLOAT16;
    *x2Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Transpose.y.repeats = {A, B, C, D, E};
    *x2Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x2Transpose.y;
    abs2.attr.sched.axis = {};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = abs1.y;
    add1.attr.sched.axis = {};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {};
    *add1.y.repeats = {};
    *add1.y.strides = {};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = x3Local.y;
    add2.attr.sched.axis = {};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_2_mul_output_data_type(&add2, 0);
    add_2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {};
    *add2.y.repeats = {};
    *add2.y.strides = {};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = add2.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithBroadcastNodeAscGraph1(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, ONE, E};
    *x1Local.y.strides = {C * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {A, C, B, D, E};
    *x1Broadcast.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs1.y.repeats = {A, B, C, D, E};
    *abs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x1Transpose.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs2.y.repeats = {A, B, C, D, E};
    *abs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("x1Local_2_Cast0_mul");
    Local2Cast0.x = x1Transpose.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast1("x1Local_2_Cast1_mul");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT16;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = Local2Cast1.y;
    abs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs3.y.repeats = {A, B, C, D, E};
    *abs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = Local2Cast1.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add1.y.repeats = {A, B, C, D, E};
    *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = abs1.y;
    add2.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_2_mul_output_data_type(&add2, 0);
    add_2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add2.y.repeats = {A, B, C, D, E};
    *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = Local2Cast1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out2("x_out3_mul");
    x_out2.x = abs3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_out2.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto x_out_node2 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
  std::shared_ptr<AscGraph> TransposeAscGraphWithBroadcastNodeAscGraph2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {C, A, B, ONE, E};
    *x1Local.y.strides = {A * B * E, B * E, E, ZERO, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {c.id, a.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {C, A, B, D, E};
    *x1Broadcast.y.strides = {A * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs1.y.repeats = {A, B, C, D, E};
    *abs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x1Transpose.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs2.y.repeats = {A, B, C, D, E};
    *abs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast Local2Cast0("x1Local_2_Cast0_mul");
    Local2Cast0.x = x1Transpose.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast Local2Cast1("x1Local_2_Cast1_mul");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT16;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = Local2Cast1.y;
    abs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs3.y.repeats = {A, B, C, D, E};
    *abs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = Local2Cast1.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add1.y.repeats = {A, B, C, D, E};
    *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = abs1.y;
    add2.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_2_mul_output_data_type(&add2, 0);
    add_2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add2.y.repeats = {A, B, C, D, E};
    *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = Local2Cast1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out2("x_out3_mul");
    x_out2.x = abs3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_out2.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto x_out_node2 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
  std::shared_ptr<AscGraph> TransposeAscGraphWithBroadcastNodeAscGraph0(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x2("x1_t_b", graph);

    ge::ascir_op::Load x2Local("x2_t_b");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {b.id, c.id, a.id, e.id, d.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {b.id, c.id, a.id, e.id, d.id};
    *x2Local.y.repeats = {B, C, A, ONE, D};
    *x2Local.y.strides = {C * A * D, A * D, D, ZERO, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x2Local.y;
    x1Broadcast.attr.sched.axis = {b.id, c.id, a.id, e.id, d.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {b.id, c.id, a.id, e.id, d.id};
    *x1Broadcast.y.repeats = {B, C, A, E, D};
    *x1Broadcast.y.strides = {C * A * E * D, A * E * D, E * D, D, ONE};

    ge::ascir_op::Transpose x1Transpose("x1Transpose_2_mul");
    x1Transpose.x = x1Broadcast.y;
    x1Transpose.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Transpose.y.dtype = DT_FLOAT16;
    *x1Transpose.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Transpose.y.repeats = {A, B, C, D, E};
    *x1Transpose.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Transpose.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs1.y.repeats = {A, B, C, D, E};
    *abs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x1Transpose.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs2.y.repeats = {A, B, C, D, E};
    *abs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("x1Local_2_Cast0_mul");
    Local2Cast0.x = x1Transpose.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast1("x1Local_2_Cast1_mul");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT16;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = Local2Cast1.y;
    abs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs3.y.repeats = {A, B, C, D, E};
    *abs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = Local2Cast1.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add1.y.repeats = {A, B, C, D, E};
    *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = abs1.y;
    add2.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_2_mul_output_data_type(&add2, 0);
    add_2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add2.y.repeats = {A, B, C, D, E};
    *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = Local2Cast1.y;
    x_out1.attr.sched.axis = {b.id, c.id, a.id, e.id, d.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {b.id, c.id, a.id, e.id, d.id};
    *x_out1.y.repeats = {B, C, A, E, D};
    *x_out1.y.strides = {C * A * E * D, A * E * D, E * D, D, ONE};
    ge::ascir_op::Store x_out2("x_out3_mul");
    x_out2.x = abs3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_out2.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto x_out_node2 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }


  // torch流程到了后处理的图是带有broadcast和transpose节点的
  std::shared_ptr<AscGraph> TransposeAscGraphWithBroadcastNodeAscGraph01(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
    std::string extern_name = "_t_b";

    ge::ascir_op::Data x1(("x1_1" + extern_name).c_str(), graph);

    ge::ascir_op::Data x2(("x2_1" + extern_name).c_str(), graph);

    ge::ascir_op::Data x3(("x3_1" + extern_name).c_str(), graph);

    ge::ascir_op::Load x1Local(("x1Local_2" + extern_name).c_str());
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local(("x2Local_2" + extern_name).c_str());
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
    x2Local.y.dtype = DT_FLOAT;
    *x2Local.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x2Local.y.repeats = {A, C, B, ONE, D};
    *x2Local.y.strides = {C * B * D, B * D, D, ZERO, ONE};

    ge::ascir_op::Load x3Local(("x3Local_2" + extern_name).c_str());
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Broadcast x2Broadcast(("x2Broadcast" + extern_name).c_str());
    x2Broadcast.x = x2Local.y;
    x2Broadcast.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
    x2Broadcast.y.dtype = DT_FLOAT;
    *x2Broadcast.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x2Broadcast.y.repeats = {A, C, B, E, D};
    *x2Broadcast.y.strides = {C * B * D * E, B * D * E, D * E, D, ONE};

    ge::ascir_op::Transpose x2Transpose1(("x2Transpose1" + extern_name).c_str());
    x2Transpose1.x = x2Broadcast.y;
    x2Transpose1.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x2Transpose1.y.dtype = DT_FLOAT;
    *x2Transpose1.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Transpose1.y.repeats = {A, C, B, D, E};
    *x2Transpose1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Transpose x2Transpose2(("x2Transpose2" + extern_name).c_str());
    x2Transpose2.x = x2Transpose1.y;
    x2Transpose2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Transpose2.y.dtype = DT_FLOAT;
    *x2Transpose2.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Transpose2.y.repeats = {A, B, C, D, E};
    *x2Transpose2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul1(("mul1" + extern_name).c_str());
    mul1.x1 = x1Local.y;
    mul1.x2 = x2Transpose2.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_FLOAT16;
    *mul1.y.axis = {};
    *mul1.y.repeats = {};
    *mul1.y.strides = {};

    ge::ascir_op::Mul mul2(("mul2" + extern_name).c_str());
    mul2.x1 = x1Local.y;
    mul2.x2 = x3Local.y;
    mul2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul2_output_data_type(&mul2, 0);
    mul2_output_data_type = ge::DT_FLOAT16;
    *mul2.y.axis = {};
    *mul2.y.repeats = {};
    *mul2.y.strides = {};

    ge::ascir_op::Mul mul3(("mul3" + extern_name).c_str());
    mul3.x1 = x2Transpose2.y;
    mul3.x2 = x3Local.y;
    mul3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul3.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul3_output_data_type(&mul3, 0);
    mul3_output_data_type = ge::DT_FLOAT16;
    *mul3.y.axis = {};
    *mul3.y.repeats = {};
    *mul3.y.strides = {};
  
    ge::ascir_op::Transpose x_Transpose1(("x_Transpose1" + extern_name).c_str());
    x_Transpose1.x = mul1.y;
    x_Transpose1.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x_Transpose1.attr.sched.loop_axis = c.id;
    x_Transpose1.y.dtype = DT_FLOAT;
    *x_Transpose1.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x_Transpose1.y.repeats = {A, C, B, D, E};
    *x_Transpose1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Transpose x_Transpose3(("x_Transpose3" + extern_name).c_str());
    x_Transpose3.x = mul3.y;
    x_Transpose3.attr.sched.axis = {a.id, b.id, c.id, e.id, d.id};
    x_Transpose3.attr.sched.loop_axis = c.id;
    x_Transpose3.y.dtype = DT_FLOAT;
    *x_Transpose3.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x_Transpose3.y.repeats = {A, B, C, E, D};
    *x_Transpose3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store1(("x_store1" + extern_name).c_str());
    x_store1.x = x_Transpose1.y;
    x_store1.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_FLOAT;
    *x_store1.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x_store1.y.repeats = {A, C, B, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store2(("x_store2" + extern_name).c_str());
    x_store2.x = mul2.y;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_FLOAT;
    *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store2.y.repeats = {A, B, C, D, E};
    *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store3(("x_store3" + extern_name).c_str());
    x_store3.x = x_Transpose3.y;
    x_store3.attr.sched.axis = {a.id, b.id, c.id, e.id, d.id};
    x_store3.attr.sched.loop_axis = c.id;
    x_store3.y.dtype = DT_FLOAT;
    *x_store3.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x_store3.y.repeats = {A, B, C, E, D};
    *x_store3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_store2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_store3.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto x_out_node3 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}, {x_out_node3, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithDiffAxisSize(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT8;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_1", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_INT8;
    *x2.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x3("x3_1", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_INT8;
    *x3.y.axis = {a.id, b.id, c.id, d.id};
    *x3.y.repeats = {A, B, C, D};
    *x3.y.strides = {B * C * D, C * D, D, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT8;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, D, E};
    *x1Local.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_INT8;
    *x2Local.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x2Local.y.repeats = {A, C, B, E, D};
    *x2Local.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

    ge::ascir_op::Load x3Local("x3Local_2");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id};
    x3Local.y.dtype = DT_INT8;
    *x3Local.y.axis = {a.id, c.id, b.id, d.id};
    *x3Local.y.repeats = {A, C, B, D};
    *x3Local.y.strides = {C * B * D, B * D, D, ONE};

    ge::ascir_op::Mul mul1("mul1");
    mul1.x1 = x1Local.y;
    mul1.x2 = x2Local.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_INT8;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_INT8;
    *mul1.y.axis = {};
    *mul1.y.repeats = {};
    *mul1.y.strides = {};

    ge::ascir_op::Mul mul2("mul2");
    mul2.x1 = x1Local.y;
    mul2.x2 = x3Local.y;
    mul2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul2.y.dtype = DT_INT8;
    AscOutputAttrDataType mul2_output_data_type(&mul2, 0);
    mul2_output_data_type = ge::DT_INT8;
    *mul2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul2.y.repeats = {A, B, C, D, E};
    *mul2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul3("mul3");
    mul3.x1 = x2Local.y;
    mul3.x2 = x3Local.y;
    mul3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul3.y.dtype = DT_INT8;
    AscOutputAttrDataType mul3_output_data_type(&mul3, 0);
    mul3_output_data_type = ge::DT_INT8;
    *mul3.y.axis = {};
    *mul3.y.repeats = {};
    *mul3.y.strides = {};
  
    ge::ascir_op::Store x_store1("x_store1");
    x_store1.x = mul1.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_INT8;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store2("x_store2");
    x_store2.x = mul2.y;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_INT8;
    *x_store2.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x_store2.y.repeats = {A, C, B, E, D};
    *x_store2.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};
  
    ge::ascir_op::Store x_store3("x_store3");
    x_store3.x = mul3.y;
    x_store3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store3.attr.sched.loop_axis = c.id;
    x_store3.y.dtype = DT_INT8;
    *x_store3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store3.y.repeats = {A, B, C, D, ONE};
    *x_store3.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_store2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_store3.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto x_out_node3 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}, {x_out_node3, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> TransposeAscGraphWithNeedDelTransposeNode(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {};
    *x1.y.repeats = {};
    *x1.y.strides = {};
  
    ge::ascir_op::Data x2("x2_1", graph);
    x2.attr.sched.axis = {};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {};
    *x2.y.repeats = {};
    *x2.y.strides = {};
  
    ge::ascir_op::Data x3("x3_1", graph);
    x3.attr.sched.axis = {};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT16;
    *x3.y.axis = {};
    *x3.y.repeats = {};
    *x3.y.strides = {};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Local.y.repeats = {A, C, B, D, E};
    *x1Local.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x2Local.y.repeats = {A, C, B, E, D};
    *x2Local.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

    ge::ascir_op::Load x3Local("x3Local_2");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id};
    x3Local.y.dtype = DT_FLOAT16;
    *x3Local.y.axis = {a.id, c.id, b.id, d.id};
    *x3Local.y.repeats = {A, C, B, D};
    *x3Local.y.strides = {C * B * D, B * D, D, ONE};

    ge::ascir_op::Abs x1Abs("x1Transpose");
    x1Abs.x = x1Local.y;
    x1Abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Abs.y.dtype = DT_FLOAT16;
    *x1Abs.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Abs.y.repeats = {A, C, B, D, E};
    *x1Abs.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

    ge::ascir_op::Abs x2Abs("x2Transpose");
    x2Abs.x = x2Local.y;
    x2Abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Abs.y.dtype = DT_FLOAT16;
    *x2Abs.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x2Abs.y.repeats = {A, C, B, E, D};
    *x2Abs.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

    ge::ascir_op::Abs x3Abs("x3Transpose");
    x3Abs.x = x3Local.y;
    x3Abs.attr.sched.axis = {a.id, b.id, c.id, d.id};
    x3Abs.y.dtype = DT_FLOAT16;
    *x3Abs.y.axis = {a.id, c.id, b.id, d.id};
    *x3Abs.y.repeats = {A, C, B, D};
    *x3Abs.y.strides = {C * B * D, B * D, D, ONE};

    ge::ascir_op::Mul mul1("mul1");
    mul1.x1 = x1Abs.y;
    mul1.x2 = x2Abs.y;
    mul1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul1_output_data_type(&mul1, 0);
    mul1_output_data_type = ge::DT_FLOAT16;
    *mul1.y.axis = {};
    *mul1.y.repeats = {};
    *mul1.y.strides = {};

    ge::ascir_op::Mul mul2("mul2");
    mul2.x1 = x1Abs.y;
    mul2.x2 = x3Abs.y;
    mul2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul2_output_data_type(&mul2, 0);
    mul2_output_data_type = ge::DT_FLOAT16;
    *mul2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul2.y.repeats = {A, B, C, D, E};
    *mul2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul3("mul3");
    mul3.x1 = x2Abs.y;
    mul3.x2 = x3Abs.y;
    mul3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    mul3.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType mul3_output_data_type(&mul3, 0);
    mul3_output_data_type = ge::DT_FLOAT16;
    *mul3.y.axis = {};
    *mul3.y.repeats = {};
    *mul3.y.strides = {};
  
    ge::ascir_op::Store x_store1("x_store1");
    x_store1.x = mul1.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    x_store1.y.dtype = DT_FLOAT16;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_store2("x_store2");
    x_store2.x = mul2.y;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_FLOAT16;
    *x_store2.y.axis = {a.id, c.id, b.id, e.id, d.id};
    *x_store2.y.repeats = {A, C, B, E, D};
    *x_store2.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};
  
    ge::ascir_op::Store x_store3("x_store3");
    x_store3.x = mul3.y;
    x_store3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store3.attr.sched.loop_axis = c.id;
    x_store3.y.dtype = DT_FLOAT16;
    *x_store3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store3.y.repeats = {A, B, C, D, ONE};
    *x_store3.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_store1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_store2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_store3.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto x_out_node3 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}, {x_out_node3, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithGraphInvalidAxis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", ONE);
    auto c = graph.CreateAxis("C", C);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, c.id};
    *x1.y.repeats = {ONE, ONE};
    *x1.y.strides = {ZERO, ZERO};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id};
    *x1Local.y.repeats = {A, C};
    *x1Local.y.strides = {C, ONE};

    ge::ascir_op::Abs x1LocalAbs("x1LocalAbs");
    x1LocalAbs.x = x1Local.y;
    x1LocalAbs.attr.sched.axis = {};
    x1LocalAbs.y.dtype = DT_FLOAT16;
    *x1LocalAbs.y.axis = {};
    *x1LocalAbs.y.repeats = {};
    *x1LocalAbs.y.strides = {};

    ge::ascir_op::Store x_store("x_store");
    x_store.x = x1LocalAbs.y;
    x_store.attr.sched.axis = {a.id, c.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, c.id};
    *x_store.y.repeats = {A, C};
    *x_store.y.strides = {C, ONE};

    ge::ascir_op::Output x_out("x_out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};
    auto x_out_node = graph.FindNode("x_out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithGraphAllAxisInvalid(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", ONE);
    auto b = graph.CreateAxis("B", ONE);
    auto c = graph.CreateAxis("C", ONE);
    auto d = graph.CreateAxis("D", ONE);
    auto e = graph.CreateAxis("E", ONE);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, ONE, ONE};
    *x1.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {ONE, ONE, ONE, ONE, ONE};
    *x1Local.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Abs x1LocalAbs("x1LocalAbs");
    x1LocalAbs.x = x1Local.y;
    x1LocalAbs.attr.sched.axis = {};
    x1LocalAbs.y.dtype = DT_FLOAT16;
    *x1LocalAbs.y.axis = {};
    *x1LocalAbs.y.repeats = {};
    *x1LocalAbs.y.strides = {};

    ge::ascir_op::Store x_store("x_store");
    x_store.x = x1LocalAbs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {ONE, ONE, ONE, ONE, ONE};
    *x_store.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Output x_out("x_out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};
    auto x_out_node = graph.FindNode("x_out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarAndAllAxisInvalid(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", ONE);
    auto b = graph.CreateAxis("B", ONE);
    auto c = graph.CreateAxis("C", ONE);
    auto d = graph.CreateAxis("D", ONE);
    auto e = graph.CreateAxis("E", ONE);

    ge::ascir_op::Scalar x1Local("x1Local_2", graph);
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {};
    *x1Local.y.repeats = {};
    *x1Local.y.strides = {};
  
    ge::ascir_op::Abs x1LocalAbs("x1LocalAbs");
    x1LocalAbs.x = x1Local.y;
    x1LocalAbs.attr.sched.axis = {};
    x1LocalAbs.y.dtype = DT_FLOAT16;
    *x1LocalAbs.y.axis = {};
    *x1LocalAbs.y.repeats = {};
    *x1LocalAbs.y.strides = {};

    ge::ascir_op::Store x_store("x_store");
    x_store.x = x1LocalAbs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {ONE, ONE, ONE, ONE, ONE};
    *x_store.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Output x_out("x_out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};
    auto x_out_node = graph.FindNode("x_out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  // 内轴无效轴（尾轴无效轴）,data轴不作为无效轴判断依据但是判断为无效轴后，data对应的轴也需要删除
  std::shared_ptr<AscGraph> CreatAscGraphWithGraphInvalidAxis2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", ONE);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, ONE};
    *x1.y.strides = {B, ONE, C};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id};
    *x1Local.y.repeats = {A, B, ONE};
    *x1Local.y.strides = {B, ONE, ZERO};

    ge::ascir_op::Abs x1LocalAbs("x1LocalAbs");
    x1LocalAbs.x = x1Local.y;
    x1LocalAbs.attr.sched.axis = {};
    x1LocalAbs.y.dtype = DT_FLOAT16;
    *x1LocalAbs.y.axis = {};
    *x1LocalAbs.y.repeats = {};
    *x1LocalAbs.y.strides = {};

    ge::ascir_op::Store x_store("x_store");
    x_store.x = x1LocalAbs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, c.id};
    *x_store.y.repeats = {A, B, ONE};
    *x_store.y.strides = {B, ONE, ZERO};

    ge::ascir_op::Output x_out("x_out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};
    auto x_out_node = graph.FindNode("x_out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithGraphInvalidAxisNodeValidAxis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", ONE);
    auto c = graph.CreateAxis("C", C);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {ONE, ONE, ONE};
    *x1.y.strides = {ZERO, ONE, ZERO};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id};
    *x1Local.y.repeats = {A, ONE, C};
    *x1Local.y.strides = {C, C, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, c.id};
    *x_out.y.repeats = {A, C};
    *x_out.y.strides = {C, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, c.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT16;
    *x_output1.y.axis = {a.id, c.id};
    *x_output1.y.repeats = {A, C};
    *x_output1.y.strides = {C, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithAxisIdDiffFromIndex(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", ONE);
    auto c = graph.CreateAxis("C", C);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, c.id};
    *x1.y.repeats = {ONE, ONE};
    *x1.y.strides = {ZERO, ZERO};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, c.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, c.id};
    *x1Local.y.repeats = {A, C};
    *x1Local.y.strides = {C, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, c.id};
    *x_out.y.repeats = {A, C};
    *x_out.y.strides = {C, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, c.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT16;
    *x_output1.y.axis = {a.id, c.id};
    *x_output1.y.repeats = {A, C};
    *x_output1.y.strides = {C, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

std::shared_ptr<AscGraph> CreatCalcRstdAscGraphWithStoreMulReference(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_calc_rstd", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("x1Local_calc_rstd");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_calc_rstd", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("x2Local_calc_rstd");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x3("x3_calc_rstd", graph);
  x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x3.attr.sched.loop_axis = c.id;
  *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3.y.repeats = {A, B, C, D, E};
  *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_calc_rstd");
  x3Local.x = x3.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::FlashSoftmax calcRstd("calcRstd_calc_rstd");
  calcRstd.x1 = x1Local.y;
  calcRstd.x2 = x2Local.y;
  calcRstd.x3 = x3Local.y;
  calcRstd.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y1.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y1.repeats = {A, B, C, D, E};
  *calcRstd.y1.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  *calcRstd.y2.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y2.repeats = {A, B, C, D, E};
  *calcRstd.y2.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  *calcRstd.y3.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y3.repeats = {A, B, C, D, E};
  *calcRstd.y3.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = calcRstd.y1;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = calcRstd.y2;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = calcRstd.y2;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out1.y.repeats = {A, B, C, D, E};
  *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out2.y.repeats = {A, B, C, D, E};
  *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}
std::shared_ptr<AscGraph> TwoSub2InputallMulReferenceAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, ONE, ONE, ONE};
  *x1.y.strides = {B, ONE, ZERO, ZERO, ZERO};

  ge::ascir_op::Load x1Local("x1Local_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("x2Local_add");
  x2Local.x = x1.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_add");
  x3Local.x = x2.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x4Local("x4Local_add");
  x4Local.x = x2.y;
  x4Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.repeats = {A, B, C, D, E};
  *x4Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add1("add_1");
  add1.x1 = x1Local.y;
  add1.x2 = x3Local.y;
  add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add1.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add1_output_data_type(&add1, 0);
  add1_output_data_type = ge::DT_FLOAT;
  *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add1.y.repeats = {A, B, C, D, E};
  *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add2("add_2");
  add2.x1 = x2Local.y;
  add2.x2 = x4Local.y;
  add2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add2.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add2_output_data_type(&add2, 0);
  add2_output_data_type = ge::DT_FLOAT;
  *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add2.y.repeats = {A, B, C, D, E};
  *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = add1.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = add2.y;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = add2.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {};
  *x_out.y.repeats = {};
  *x_out.y.strides = {};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {};
  *x_out1.y.repeats = {};
  *x_out1.y.strides = {};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {};
  *x_out2.y.repeats = {};
  *x_out2.y.strides = {};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}
std::shared_ptr<AscGraph> ThreeSub2InputallMulReferenceAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, ONE, ONE, ONE};
  *x1.y.strides = {B, ONE, ZERO, ZERO, ZERO};

  ge::ascir_op::Load x1Local("x1Local_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("x2Local_add");
  x2Local.x = x1.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x5Local("x5Local_add");
  x5Local.x = x1.y;
  x5Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x5Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x5Local.y.repeats = {A, B, C, D, E};
  *x5Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_add");
  x3Local.x = x2.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x4Local("x4Local_add");
  x4Local.x = x2.y;
  x4Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.repeats = {A, B, C, D, E};
  *x4Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x6Local("x6Local_add");
  x6Local.x = x2.y;
  x6Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.repeats = {A, B, C, D, E};
  *x6Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add1("add_1");
  add1.x1 = x1Local.y;
  add1.x2 = x3Local.y;
  add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add1.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add1_output_data_type(&add1, 0);
  add1_output_data_type = ge::DT_FLOAT;
  *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add1.y.repeats = {A, B, C, D, E};
  *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add2("add_2");
  add2.x1 = x2Local.y;
  add2.x2 = x4Local.y;
  add2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add2.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add2_output_data_type(&add2, 0);
  add2_output_data_type = ge::DT_FLOAT;
  *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add2.y.repeats = {A, B, C, D, E};
  *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add3("add_3");
  add3.x1 = x5Local.y;
  add3.x2 = x6Local.y;
  add3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add3.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add3_output_data_type(&add3, 0);
  add3_output_data_type = ge::DT_FLOAT;
  *add3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add3.y.repeats = {A, B, C, D, E};
  *add3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = add1.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = add2.y;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = add3.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {};
  *x_out.y.repeats = {};
  *x_out.y.strides = {};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {};
  *x_out1.y.repeats = {};
  *x_out1.y.strides = {};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {};
  *x_out2.y.repeats = {};
  *x_out2.y.strides = {};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CastWithSameDtypeAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, ONE, ONE, ONE};
  *x1.y.strides = {B, ONE, ZERO, ZERO, ZERO};

  ge::ascir_op::Load x1Local("x1Local_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("x2Local_add");
  x2Local.x = x1.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_add");
  x3Local.x = x2.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x4Local("x4Local_add");
  x4Local.x = x2.y;
  x4Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.repeats = {A, B, C, D, E};
  *x4Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add1("add_1");
  add1.x1 = x1Local.y;
  add1.x2 = x3Local.y;
  add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add1.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add1_output_data_type(&add1, 0);
  add1_output_data_type = ge::DT_FLOAT;
  *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add1.y.repeats = {A, B, C, D, E};
  *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Cast Cast1ToStore("Cast1ToStore");
  Cast1ToStore.x = add1.y;
  Cast1ToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  Cast1ToStore.y.dtype = DT_INT8;
  *Cast1ToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *Cast1ToStore.y.repeats = {A, B, C, D, E};
  *Cast1ToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Cast Cast3ToStore("Cast3ToStore");
  Cast3ToStore.x = Cast1ToStore.y;
  Cast3ToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  Cast3ToStore.y.dtype = DT_INT16;
  *Cast3ToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *Cast3ToStore.y.repeats = {A, B, C, D, E};
  *Cast3ToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add2("add_2");
  add2.x1 = x2Local.y;
  add2.x2 = x4Local.y;
  add2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add2.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add2_output_data_type(&add2, 0);
  add2_output_data_type = ge::DT_FLOAT;
  *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add2.y.repeats = {A, B, C, D, E};
  *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Cast Cast2ToStore("Cast2ToStore");
  Cast2ToStore.x = add2.y;
  Cast2ToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  Cast2ToStore.y.dtype = DT_INT8;
  *Cast2ToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *Cast2ToStore.y.repeats = {A, B, C, D, E};
  *Cast2ToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Cast Cast4ToStore("Cast4ToStore");
  Cast4ToStore.x = Cast2ToStore.y;
  Cast4ToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  Cast4ToStore.y.dtype = DT_INT32;
  *Cast4ToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *Cast4ToStore.y.repeats = {A, B, C, D, E};
  *Cast4ToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = Cast3ToStore.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  x_store.y.dtype = DT_INT16;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = Cast4ToStore.y;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  x_store1.y.dtype = DT_INT32;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = Cast4ToStore.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  x_store2.y.dtype = DT_INT32;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {};
  *x_out.y.repeats = {};
  *x_out.y.strides = {};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {};
  *x_out1.y.repeats = {};
  *x_out1.y.strides = {};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {};
  *x_out2.y.repeats = {};
  *x_out2.y.strides = {};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> ThreeScalarAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Scalar x1Local("x1_scalar", graph);
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {};
  *x1Local.y.repeats = {};
  *x1Local.y.strides = {};
  x1Local.ir_attr.SetValue("1");
  ge::ascir_op::Scalar x2Local("x2_scalar", graph);
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {};
  *x2Local.y.repeats = {};
  *x2Local.y.strides = {};
  x2Local.ir_attr.SetValue("1");
  ge::ascir_op::Scalar x5Local("x3_scalar", graph);
  x5Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x5Local.y.axis = {};
  *x5Local.y.repeats = {};
  *x5Local.y.strides = {};
  x5Local.ir_attr.SetValue("1");

  ge::ascir_op::Data x2("x2_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_add");
  x3Local.x = x2.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x4Local("x4Local_add");
  x4Local.x = x2.y;
  x4Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.repeats = {A, B, C, D, E};
  *x4Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x6Local("x6Local_add");
  x6Local.x = x2.y;
  x6Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.repeats = {A, B, C, D, E};
  *x6Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add1("add_1");
  add1.x1 = x1Local.y;
  add1.x2 = x3Local.y;
  add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add1.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add1_output_data_type(&add1, 0);
  add1_output_data_type = ge::DT_FLOAT;
  *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add1.y.repeats = {A, B, C, D, E};
  *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add2("add_2");
  add2.x1 = x2Local.y;
  add2.x2 = x4Local.y;
  add2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add2.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add2_output_data_type(&add2, 0);
  add2_output_data_type = ge::DT_FLOAT;
  *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add2.y.repeats = {A, B, C, D, E};
  *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add3("add_3");
  add3.x1 = x5Local.y;
  add3.x2 = x6Local.y;
  add3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add3.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add3_output_data_type(&add3, 0);
  add3_output_data_type = ge::DT_FLOAT;
  *add3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add3.y.repeats = {A, B, C, D, E};
  *add3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = add1.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = add2.y;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = add3.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {};
  *x_out.y.repeats = {};
  *x_out.y.strides = {};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {};
  *x_out1.y.repeats = {};
  *x_out1.y.strides = {};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {};
  *x_out2.y.repeats = {};
  *x_out2.y.strides = {};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}
std::shared_ptr<AscGraph> ThreeScalarTwoSameValueAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Scalar x1Local("x1_scalar", graph);
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {};
  *x1Local.y.repeats = {};
  *x1Local.y.strides = {};
  x1Local.ir_attr.SetValue("1");
  ge::ascir_op::Scalar x2Local("x2_scalar", graph);
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {};
  *x2Local.y.repeats = {};
  *x2Local.y.strides = {};
  x2Local.ir_attr.SetValue("1");
  ge::ascir_op::Scalar x5Local("x3_scalar", graph);
  x5Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x5Local.y.axis = {};
  *x5Local.y.repeats = {};
  *x5Local.y.strides = {};
  x5Local.ir_attr.SetValue("2");

  ge::ascir_op::Data x2("x2_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_add");
  x3Local.x = x2.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x4Local("x4Local_add");
  x4Local.x = x2.y;
  x4Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.repeats = {A, B, C, D, E};
  *x4Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x6Local("x6Local_add");
  x6Local.x = x2.y;
  x6Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.repeats = {A, B, C, D, E};
  *x6Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add1("add_1");
  add1.x1 = x1Local.y;
  add1.x2 = x3Local.y;
  add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add1.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add1_output_data_type(&add1, 0);
  add1_output_data_type = ge::DT_FLOAT;
  *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add1.y.repeats = {A, B, C, D, E};
  *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add2("add_2");
  add2.x1 = x2Local.y;
  add2.x2 = x4Local.y;
  add2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add2.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add2_output_data_type(&add2, 0);
  add2_output_data_type = ge::DT_FLOAT;
  *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add2.y.repeats = {A, B, C, D, E};
  *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add3("add_3");
  add3.x1 = x5Local.y;
  add3.x2 = x6Local.y;
  add3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add3.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add3_output_data_type(&add3, 0);
  add3_output_data_type = ge::DT_FLOAT;
  *add3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add3.y.repeats = {A, B, C, D, E};
  *add3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = add1.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = add2.y;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = add3.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {};
  *x_out.y.repeats = {};
  *x_out.y.strides = {};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {};
  *x_out1.y.repeats = {};
  *x_out1.y.strides = {};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {};
  *x_out2.y.repeats = {};
  *x_out2.y.strides = {};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}
std::shared_ptr<AscGraph> ThreeScalarTwoSameValueDtypeAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Scalar x1Local("x1_scalar", graph);
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {};
  *x1Local.y.repeats = {};
  *x1Local.y.strides = {};
  x1Local.y.dtype = DT_FLOAT16;
  x1Local.ir_attr.SetValue("1");
  ge::ascir_op::Scalar x2Local("x2_scalar", graph);
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {};
  *x2Local.y.repeats = {};
  *x2Local.y.strides = {};
  x2Local.y.dtype = DT_FLOAT16;
  x2Local.ir_attr.SetValue("1");
  ge::ascir_op::Scalar x5Local("x3_scalar", graph);
  x5Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x5Local.y.axis = {};
  *x5Local.y.repeats = {};
  *x5Local.y.strides = {};
  x5Local.y.dtype = DT_INT32;
  x5Local.ir_attr.SetValue("1");

  ge::ascir_op::Data x2("x2_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_add");
  x3Local.x = x2.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x4Local("x4Local_add");
  x4Local.x = x2.y;
  x4Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x4Local.y.repeats = {A, B, C, D, E};
  *x4Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x6Local("x6Local_add");
  x6Local.x = x2.y;
  x6Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x6Local.y.repeats = {A, B, C, D, E};
  *x6Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add1("add_1");
  add1.x1 = x1Local.y;
  add1.x2 = x3Local.y;
  add1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add1.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add1_output_data_type(&add1, 0);
  add1_output_data_type = ge::DT_FLOAT;
  *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add1.y.repeats = {A, B, C, D, E};
  *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add2("add_2");
  add2.x1 = x2Local.y;
  add2.x2 = x4Local.y;
  add2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add2.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add2_output_data_type(&add2, 0);
  add2_output_data_type = ge::DT_FLOAT;
  *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add2.y.repeats = {A, B, C, D, E};
  *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
  ge::ascir_op::Add add3("add_3");
  add3.x1 = x5Local.y;
  add3.x2 = x6Local.y;
  add3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  add3.y.dtype = DT_FLOAT16;
  AscOutputAttrDataType add3_output_data_type(&add3, 0);
  add3_output_data_type = ge::DT_FLOAT;
  *add3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add3.y.repeats = {A, B, C, D, E};
  *add3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = add1.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = add2.y;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store2("x_store2_calc_rstd");
  x_store2.x = add3.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {};
  *x_out.y.repeats = {};
  *x_out.y.strides = {};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {};
  *x_out1.y.repeats = {};
  *x_out1.y.strides = {};
  ge::ascir_op::Output x_out2("x_out2_calc_rstd");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {};
  *x_out2.y.repeats = {};
  *x_out2.y.strides = {};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}
  std::shared_ptr<AscGraph> CreatAddAscGraphWithTranspose(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {b.id, a.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {b.id, a.id, c.id, d.id, e.id};
    *x2.y.repeats = {B, A, C, D, E};
    *x2.y.strides = {A * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToStore(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToBroadcstToStore(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, ONE, E};
    *x1.y.strides = {B * C * E, C * E, E, ZERO, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToBroadcstAndTransposeToStore(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, ONE, E};
    *x1.y.strides = {B * C * E, C * E, E, ZERO, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, ONE, E};
    *x1Local.y.strides = {B * C * E, C * E, E, ZERO, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x_out.y.repeats = {A, B, C, E, D};
    *x_out.y.strides = {B * C * E * D, C * E * D, E * D, D, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToBroadcstAndTransposeToStore2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x1.y.repeats = {A, B, C, E, ONE};
    *x1.y.strides =  {B * C * E, C * E, E, ONE, ZERO};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x1Local.y.repeats = {A, B, C, E, ONE};
    *x1Local.y.strides = {B * C * E, C * E, E, ONE, ZERO};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToBroadcstAndTransposeToStore3(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x1.y.repeats = {ONE, B, C, E, D};
    *x1.y.strides =  {ZERO, C * E * D, E * D, D, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, e.id, d.id};
    *x1Local.y.repeats = {ONE, B, C, E, D};
    *x1Local.y.strides = {ZERO, C * E * D, E * D, D, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1Local.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToCastToStore(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local2("x1Local2_2");
    x1Local2.x = x1.y;
    x1Local2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local2.y.dtype = DT_INT16;
    *x1Local2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local2.y.repeats = {A, B, C, D, E};
    *x1Local2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast LoadToCastToStore("LoadToCast1");
    LoadToCastToStore.x = x1Local.y;
    LoadToCastToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore.y.dtype = DT_FLOAT;
    *LoadToCastToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast LoadToCastToStore2("CastToCastToStore2");
    LoadToCastToStore2.x = LoadToCastToStore.y;
    LoadToCastToStore2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore2.y.dtype = DT_INT16;
    *LoadToCastToStore2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore2.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast LoadToCastToStore3("LoadToCastToStore3");
    LoadToCastToStore3.x = x1Local2.y;
    LoadToCastToStore3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore3.y.dtype = DT_FLOAT;
    *LoadToCastToStore3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore3.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = LoadToCastToStore2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_INT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = LoadToCastToStore3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithImprovePrecisionBlackList0(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs LoadToCastToStore("LoadToCastToStore");
    LoadToCastToStore.x = x1Local.y;
    LoadToCastToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore.y.dtype = DT_FLOAT16;
    *LoadToCastToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs LoadToCastToStore2("LoadToCastToStore2");
    LoadToCastToStore2.x = LoadToCastToStore.y;
    LoadToCastToStore2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore2.y.dtype = DT_FLOAT16;
    *LoadToCastToStore2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore2.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs LoadToCastToStore3("LoadToCastToStore3");
    LoadToCastToStore3.x = x1Local.y;
    LoadToCastToStore3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore3.y.dtype = DT_FLOAT16;
    *LoadToCastToStore3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore3.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = LoadToCastToStore2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = LoadToCastToStore3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithImprovePrecisionBlackList1(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs LoadToCastToStore("Abs");
    LoadToCastToStore.x = x1Local.y;
    LoadToCastToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore.y.dtype = DT_FLOAT16;
    *LoadToCastToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Sqrt LoadToCastToStore2("Sqrt");
    LoadToCastToStore2.x = LoadToCastToStore.y;
    LoadToCastToStore2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore2.y.dtype = DT_FLOAT16;
    *LoadToCastToStore2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore2.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Rsqrt LoadToCastToStore3("Rsqrt");
    LoadToCastToStore3.x = x1Local.y;
    LoadToCastToStore3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore3.y.dtype = DT_FLOAT16;
    *LoadToCastToStore3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore3.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = LoadToCastToStore2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = LoadToCastToStore3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithImprovePrecisionWhiteList(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Sum LoadToCastToStore("Sum");
    LoadToCastToStore.x = x1Local.y;
    LoadToCastToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore.y.dtype = DT_BF16;
    *LoadToCastToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Sqrt LoadToCastToStore2("Sqrt");
    LoadToCastToStore2.x = LoadToCastToStore.y;
    LoadToCastToStore2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore2.y.dtype = DT_FLOAT16;
    *LoadToCastToStore2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore2.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Rsqrt LoadToCastToStore3("Rsqrt");
    LoadToCastToStore3.x = x1Local.y;
    LoadToCastToStore3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore3.y.dtype = DT_FLOAT16;
    *LoadToCastToStore3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore3.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = LoadToCastToStore2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = LoadToCastToStore3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithImproveBF16(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_BF16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_BF16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Sum LoadToCastToStore("Sum");
    LoadToCastToStore.x = x1Local.y;
    LoadToCastToStore.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore.y.dtype = DT_BF16;
    *LoadToCastToStore.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Sqrt LoadToCastToStore2("Sqrt");
    LoadToCastToStore2.x = LoadToCastToStore.y;
    LoadToCastToStore2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore2.y.dtype = DT_BF16;
    *LoadToCastToStore2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore2.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Rsqrt LoadToCastToStore3("Rsqrt");
    LoadToCastToStore3.x = x1Local.y;
    LoadToCastToStore3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore3.y.dtype = DT_BF16;
    *LoadToCastToStore3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore3.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = LoadToCastToStore2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_BF16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = LoadToCastToStore3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_BF16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToCastToStore2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local2("x1Local2_2");
    x1Local2.x = x1.y;
    x1Local2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local2.y.dtype = DT_FLOAT;
    *x1Local2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local2.y.repeats = {A, B, C, D, E};
    *x1Local2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast LoadToCastToStore3("LoadToCastToStore3");
    LoadToCastToStore3.x = x1Local2.y;
    LoadToCastToStore3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    LoadToCastToStore3.y.dtype = DT_FLOAT16;
    *LoadToCastToStore3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *LoadToCastToStore3.y.repeats = {A, B, C, D, E};
    *LoadToCastToStore3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = LoadToCastToStore3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadToCastToStore3(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local2("x1Local2_2");
    x1Local2.x = x1.y;
    x1Local2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local2.y.dtype = DT_FLOAT16;
    *x1Local2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local2.y.repeats = {A, B, C, D, E};
    *x1Local2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out1("x_out1_5");
    x_out1.x = x1Local2.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, ONE};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = x1Local2.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs1.y.repeats = {A, B, C, D, E};
    *abs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out2("x_out2_5");
    x_out2.x = abs1.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, ONE}; // slice切分场景会在store改变repeats
    *x_out2.y.strides = {B * C * D, C * D, D, ONE, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out2.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node1 = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node1->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node1, 0}, {x_out_node2, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAddAscGraphWithEmptyRepeats(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, ONE, E};
    *x1Local.y.strides = {B * C * E, C * E, E, E, ONE};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, ONE, ONE, ONE};
    *x2.y.strides = {B, ONE, ZERO, ZERO, ZERO};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, ONE, E};
    *x2Local.y.strides = {B * C * E, C * E, E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
  
  static std::shared_ptr<ge::AscGraph> CreatConcatAscGraphForNoImprovePrecision(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("load1");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("load2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x3("data3", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT16;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {A, B, C, D, E};
    *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("load3");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT16;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Concat concat(graph.GetName().c_str());
    concat.x = {x1.y, x2.y, x3.y};
    concat.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    concat.y.dtype = DT_FLOAT16;
    *concat.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *concat.y.repeats = {A, B, C, D, E};
    *concat.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = concat.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT32;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT16;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = gather.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision32_8(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_FLOAT;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast1("cast1_32-16");
    cast1.x = gather.y;
    cast1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast1.y.dtype = DT_FLOAT16;
    *cast1.y.axis = {a.id, b.id, d.id, e.id};
    *cast1.y.repeats = {A, B, D, E};
    *cast1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast2("cast2_16-8");
    cast2.x = cast1.y;
    cast2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast2.y.dtype = DT_INT8;
    *cast2.y.axis = {a.id, b.id, d.id, e.id};
    *cast2.y.repeats = {A, B, D, E};
    *cast2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = cast2.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_INT8;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_INT8;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision32_8_MulReference(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_FLOAT;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast1("cast1_32-16");
    cast1.x = gather.y;
    cast1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast1.y.dtype = DT_FLOAT16;
    *cast1.y.axis = {a.id, b.id, d.id, e.id};
    *cast1.y.repeats = {A, B, D, E};
    *cast1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store0("store0");
    x_store0.x = cast1.y;
    x_store0.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store0.attr.sched.loop_axis = c.id;
    x_store0.y.dtype = DT_FLOAT16;
    *x_store0.y.axis = {a.id, b.id, d.id, e.id};
    *x_store0.y.repeats = {A, B, D, E};
    *x_store0.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs("abs");
    abs.x = cast1.y;
    abs.attr.sched.axis = {a.id, b.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT16;
    *abs.y.axis = {a.id, b.id, d.id, e.id};
    *abs.y.repeats = {A, B, D, E};
    *abs.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store2("store2");
    x_store2.x = abs.y;
    x_store2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_FLOAT16;
    *x_store2.y.axis = {a.id, b.id, d.id, e.id};
    *x_store2.y.repeats = {A, B, D, E};
    *x_store2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast2("cast2_16-8");
    cast2.x = cast1.y;
    cast2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast2.y.dtype = DT_INT8;
    *cast2.y.axis = {a.id, b.id, d.id, e.id};
    *cast2.y.repeats = {A, B, D, E};
    *cast2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store1");
    x_store.x = cast2.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_INT8;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out0("out0");
    x_out0.x = x_store0.y;
    x_out0.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out0.attr.sched.loop_axis = c.id;
    x_out0.y.dtype = DT_FLOAT16;
    *x_out0.y.axis = {a.id, b.id, d.id, e.id};
    *x_out0.y.repeats = {A, B, D, E};
    *x_out0.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out1");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_INT8;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out2("out2");
    x_out2.x = x_store2.y;
    x_out2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, D, E};
    *x_out2.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node0 = graph.FindNode("out0");
    auto x_out_node1 = graph.FindNode("out1");
    auto x_out_node2 = graph.FindNode("out2");
    auto compute_graph = x_out_node0->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node0, 0}, {x_out_node1, 0}, {x_out_node2, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision8_32_MulReference(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT8;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT8;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_INT8;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast1("cast1_8-16");
    cast1.x = gather.y;
    cast1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast1.y.dtype = DT_FLOAT16;
    *cast1.y.axis = {a.id, b.id, d.id, e.id};
    *cast1.y.repeats = {A, B, D, E};
    *cast1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store0("store0");
    x_store0.x = cast1.y;
    x_store0.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store0.attr.sched.loop_axis = c.id;
    x_store0.y.dtype = DT_FLOAT16;
    *x_store0.y.axis = {a.id, b.id, d.id, e.id};
    *x_store0.y.repeats = {A, B, D, E};
    *x_store0.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs("abs");
    abs.x = cast1.y;
    abs.attr.sched.axis = {a.id, b.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT16;
    *abs.y.axis = {a.id, b.id, d.id, e.id};
    *abs.y.repeats = {A, B, D, E};
    *abs.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store2("store2");
    x_store2.x = abs.y;
    x_store2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    x_store2.y.dtype = DT_FLOAT16;
    *x_store2.y.axis = {a.id, b.id, d.id, e.id};
    *x_store2.y.repeats = {A, B, D, E};
    *x_store2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast2("cast2_16-32");
    cast2.x = cast1.y;
    cast2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast2.y.dtype = DT_FLOAT;
    *cast2.y.axis = {a.id, b.id, d.id, e.id};
    *cast2.y.repeats = {A, B, D, E};
    *cast2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store1");
    x_store.x = cast2.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out0("out0");
    x_out0.x = x_store0.y;
    x_out0.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out0.attr.sched.loop_axis = c.id;
    x_out0.y.dtype = DT_FLOAT16;
    *x_out0.y.axis = {a.id, b.id, d.id, e.id};
    *x_out0.y.repeats = {A, B, D, E};
    *x_out0.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out1");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out2("out2");
    x_out2.x = x_store2.y;
    x_out2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, D, E};
    *x_out2.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node0 = graph.FindNode("out0");
    auto x_out_node1 = graph.FindNode("out1");
    auto x_out_node2 = graph.FindNode("out2");
    auto compute_graph = x_out_node0->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node0, 0}, {x_out_node1, 0}, {x_out_node2, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision8_16_8(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT8;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT8;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_INT8;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast1("cast1_8-16");
    cast1.x = gather.y;
    cast1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast1.y.dtype = DT_FLOAT16;
    *cast1.y.axis = {a.id, b.id, d.id, e.id};
    *cast1.y.repeats = {A, B, D, E};
    *cast1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast2("cast2_16-8");
    cast2.x = cast1.y;
    cast2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast2.y.dtype = DT_INT8;
    *cast2.y.axis = {a.id, b.id, d.id, e.id};
    *cast2.y.repeats = {A, B, D, E};
    *cast2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = cast2.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_INT8;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_INT8;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision16_8_16(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT16;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast1("cast1_8-16");
    cast1.x = gather.y;
    cast1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast1.y.dtype = DT_INT8;
    *cast1.y.axis = {a.id, b.id, d.id, e.id};
    *cast1.y.repeats = {A, B, D, E};
    *cast1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast2("cast2_16-8");
    cast2.x = cast1.y;
    cast2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast2.y.dtype = DT_FLOAT16;
    *cast2.y.axis = {a.id, b.id, d.id, e.id};
    *cast2.y.repeats = {A, B, D, E};
    *cast2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = cast2.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForImprovePrecision8_32(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT8;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT8;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_INT8;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast1("cast1_32-16");
    cast1.x = gather.y;
    cast1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast1.y.dtype = DT_FLOAT16;
    *cast1.y.axis = {a.id, b.id, d.id, e.id};
    *cast1.y.repeats = {A, B, D, E};
    *cast1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast2("cast2_16-8");
    cast2.x = cast1.y;
    cast2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast2.y.dtype = DT_FLOAT;
    *cast2.y.axis = {a.id, b.id, d.id, e.id};
    *cast2.y.repeats = {A, B, D, E};
    *cast2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = cast2.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForCse(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, d.id};
    x1.attr.sched.loop_axis = d.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, d.id};
    *x1.y.repeats = {A, B, D};
    *x1.y.strides = {B * D, D, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, d.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT32;
    *x2.y.axis = {a.id, b.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather1("gather1_cse");
    gather1.x1 = x1.y;
    gather1.x2 = x2.y;
    gather1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather1.y.dtype = DT_FLOAT16;
    gather1.ir_attr.SetAxis(2);
    *gather1.y.axis = {a.id, b.id, d.id, e.id};
    *gather1.y.repeats = {A, B, D, E};
    *gather1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Gather gather2("gather2_cse");
    gather2.x1 = x1.y;
    gather2.x2 = x2.y;
    gather2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather2.y.dtype = DT_FLOAT16;
    gather2.ir_attr.SetAxis(2);
    *gather2.y.axis = {a.id, b.id, d.id, e.id};
    *gather2.y.repeats = {A, B, D, E};
    *gather2.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_gather");
    add.x1 = gather1.y;
    add.x2 = gather2.y;
    add.attr.sched.axis = {a.id, b.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Store x_store("store");
    x_store.x = add.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = d.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = d.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForData2AllInvalidAxis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression D = graph.CreateSizeVar(1);
    const Expression E = graph.CreateSizeVar(1);

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x1.attr.sched.loop_axis = d.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, d.id};
    *x1.y.repeats = {A, B, D};
    *x1.y.strides = {B * D, D, ZERO};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT32;
    *x2.y.axis = {a.id, b.id};
    *x2.y.repeats = {ONE, ONE};
    *x2.y.strides = {ZERO, ZERO};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT16;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, ONE, ONE};
    *gather.y.strides = {B, ONE, ZERO, ZERO};

    ge::ascir_op::Store x_store("store");
    x_store.x = gather.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = d.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, ONE, ONE};
    *x_store.y.strides = {B, ONE, ZERO, ZERO};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = d.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForDataAllInvalidAxis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar(1);
    const Expression B = graph.CreateSizeVar(1);
    const Expression D = graph.CreateSizeVar(1);
    const Expression E = graph.CreateSizeVar(1);

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x1.attr.sched.loop_axis = d.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, d.id};
    *x1.y.repeats = {A, B, D};
    *x1.y.strides = {ZERO, ZERO, ZERO};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT32;
    *x2.y.axis = {a.id, b.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {ZERO, ZERO};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT16;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Store x_store("store");
    x_store.x = gather.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = d.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = d.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForData1Data2InvalidAxis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar(1);
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar(1);

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x1.attr.sched.loop_axis = d.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, d.id};
    *x1.y.repeats = {A, B, D};
    *x1.y.strides = {D, ZERO, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT32;
    *x2.y.axis = {a.id, b.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {ONE, ZERO};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT16;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {D, ZERO, ONE, ZERO};

    ge::ascir_op::Store x_store("store");
    x_store.x = gather.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = d.id;
    x_store.y.dtype = DT_FLOAT16;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {D, ZERO, ONE, ZERO};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = d.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {};
    *x_out.y.repeats = {};
    *x_out.y.strides = {};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphForInsertCastImprovePrecision(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {A, B, C};
    *x1.y.strides = {B * C, C, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {d.id, e.id};
    x2.attr.sched.loop_axis = d.id;
    x2.y.dtype = DT_INT32;
    *x2.y.axis = {d.id, e.id};
    *x2.y.repeats = {D, E};
    *x2.y.strides = {E, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = x1.y;
    gather.x2 = x2.y;
    gather.attr.sched.axis = {a.id, b.id, d.id, e.id};
    gather.y.dtype = DT_FLOAT16;
    gather.ir_attr.SetAxis(2);
    *gather.y.axis = {a.id, b.id, d.id, e.id};
    *gather.y.repeats = {A, B, D, E};
    *gather.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Cast cast("cast");
    cast.x = gather.y;
    cast.attr.sched.axis = {a.id, b.id, d.id, e.id};
    cast.y.dtype = DT_FLOAT;
    *cast.y.axis = {a.id, b.id, d.id, e.id};
    *cast.y.repeats = {A, B, D, E};
    *cast.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = cast.y;
    x_store.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_FLOAT;
    *x_store.y.axis = {a.id, b.id, d.id, e.id};
    *x_store.y.repeats = {A, B, D, E};
    *x_store.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, d.id, e.id};
    *x_out.y.repeats = {A, B, D, E};
    *x_out.y.strides = {B * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAddAscGraphWithEmptyRepeatsAfterLoad(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {B, C, D, E};
    *x1Local.y.strides = {C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, ONE, ONE, ONE};
    *x2.y.strides = {B, ONE, ZERO, ZERO, ZERO};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatBrcBackwardAscGraphWithMulInputs(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {b.id, c.id, d.id, e.id};
    *x1.y.repeats = {B, ONE, D, E};
    *x1.y.strides = {B * E, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {ONE, B, ONE, D, E};
    *x2.y.strides = {ZERO, B * E, ZERO, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, ONE, D, E};
    *x2.y.strides = {B * D * E, D * E, ZERO, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, ONE};
    *x1.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, ONE, ONE};
    *x2.y.strides = {B * C, C, ONE, ZERO, ZERO};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs3(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, ONE, ONE, D, E};
    *x2.y.strides = {D * E, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs4(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, ONE, ONE, D, E};
    *x2.y.strides = {D * E, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = add.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out1("x_out_6");
    x_out1.x = abs1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node2, 0}, {x_out_node, 1}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs5(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, ONE, C, D, E};
    *x2.y.strides = {C * D * E, ZERO, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs6(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, ONE, D, E};
    *x2.y.strides = {B * D * E, D * E, ZERO, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Data x3("x2_4", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    x3.y.dtype = DT_FLOAT16;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {A, B, C, D, E};
    *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("x2Local_5");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT16;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add2("add_5");
    add2.x1 = x3Local.y;
    add2.x2 = add.y;
    add2.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x3Local_output_data_type(&add2, 0);
    x3Local_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {};
    *add2.y.repeats = {};
    *add2.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add2.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = abs1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs7(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar(1);

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ZERO};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {ONE, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ZERO};

    ge::ascir_op::Abs abs_empty_tensor("abs_empty_tensor");
    abs_empty_tensor.x = x1Local.y;
    abs_empty_tensor.attr.sched.axis = {};
    abs_empty_tensor.y.dtype = DT_FLOAT16;
    *abs_empty_tensor.y.axis = {};
    *abs_empty_tensor.y.repeats = {};
    *abs_empty_tensor.y.strides = {};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {ONE, B, ONE, D, E};
    *x2.y.strides = {B * D * E, D * E, ZERO, E, ZERO};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {ONE, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ZERO};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs_empty_tensor.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Broadcast bro1("broadcast_1");
    bro1.x = abs1.y;
    bro1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    bro1.y.dtype = DT_FLOAT16;
    *bro1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *bro1.y.repeats = {A, B, C, D, E};
    *bro1.y.strides = {B * C * D * E, C * D * E, D * E, E, ZERO};

    ge::ascir_op::Abs abs2("abs_5");
    abs2.x = bro1.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis =  {a.id, b.id, c.id, d.id, e.id};
    *abs2.y.repeats = {A, B, C, D, ONE};
    *abs2.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE}; // 刻意内轴设置为1，构造无效轴不删除场景叠加broadcast后移内轴刷新正确

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs8(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, C, D, E};
    *x1.y.strides = {ZERO, ZERO, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {ONE, ONE, C, D, E};
    *x1Local.y.strides = {ZERO, ZERO, D * E, E, ONE};

    ge::ascir_op::Broadcast x1Broadcast("x1Broadcast_2_mul");
    x1Broadcast.x = x1Local.y;
    x1Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x1Broadcast.y.dtype = DT_FLOAT16;
    *x1Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x1Broadcast.y.repeats = {ONE, B, C, D, E};
    *x1Broadcast.y.strides = {ZERO, C * D * E, D * E, E, ONE};

    ge::ascir_op::Broadcast x2Broadcast("x2Broadcast_2_mul");
    x2Broadcast.x = x1Broadcast.y;
    x2Broadcast.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
    x2Broadcast.y.dtype = DT_FLOAT16;
    *x2Broadcast.y.axis = {a.id, c.id, b.id, d.id, e.id};
    *x2Broadcast.y.repeats = {A, B, C, D, E};
    *x2Broadcast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs2("abs_3");
    abs2.x = x2Broadcast.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Add add("add_4");
    add.x1 = x2Broadcast.y;
    add.x2 = abs2.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Store x_out1("x_out_6");
    x_out1.x = abs1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};


    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs9(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, C, D, E};
    *x1.y.strides = {ZERO, ZERO, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x1Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs1("abs_4");
    abs1.x = add.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Store x_out1("x_out_6");
    x_out1.x = abs1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out1.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatPartBrcBackwardAscGraphWithMulInputs10(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, C, D, E};
    *x1.y.strides = {ZERO, ZERO, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs_3");
    abs1.x = x1Local.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Relu abs2("relu_4");
    abs2.x = x1Local.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out1("x_out_6");
    x_out1.x = abs1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node2 = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node2, 0}, {x_out_node, 1}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithLoadAndCastMulReference(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1_mul", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2_mul");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs1("abs1_x1Local_2_mul");
    abs1.x = x1Local.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs1.y.repeats = {A, B, C, D, E};
    *abs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs2("abs2_x1Local_2_mul");
    abs2.x = x1Local.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs2.y.repeats = {A, B, C, D, E};
    *abs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("x1Local_2_Cast0_mul");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast1("x1Local_2_Cast1_mul");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT16;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs3("abs3_x1Local_2_mul");
    abs3.x = Local2Cast1.y;
    abs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs3.y.repeats = {A, B, C, D, E};
    *abs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add1("add_1_mul");
    add1.x1 = abs2.y;
    add1.x2 = Local2Cast1.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add1.y.repeats = {A, B, C, D, E};
    *add1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add2("add_2_mul");
    add2.x1 = add1.y;
    add2.x2 = abs1.y;
    add2.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_2_mul_output_data_type(&add2, 0);
    add_2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add2.y.repeats = {A, B, C, D, E};
    *add2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = Local2Cast1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out2("x_out3_mul");
    x_out2.x = abs3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_out2.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto x_out_node2 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
  std::shared_ptr<AscGraph> CreatAscGraphWithLoadAndCastMulReferenceAndInsertBroadcast(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("Data", graph);
    x1.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("Load");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {B, C, ONE, E};
    *x1Local.y.strides = {C * E, E, E, ONE};

    ge::ascir_op::Abs abs1("abs1");
    abs1.x = x1Local.y;
    abs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Abs abs2("abs2");
    abs2.x = x1Local.y;
    abs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2.y.dtype = DT_FLOAT16;
    *abs2.y.axis = {};
    *abs2.y.repeats = {};
    *abs2.y.strides = {};
  
    ge::ascir_op::Cast Local2Cast0("Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast1("Cast1");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT16;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs3("abs3");
    abs3.x = Local2Cast1.y;
    abs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs3.y.dtype = DT_FLOAT16;
    *abs3.y.axis = {};
    *abs3.y.repeats = {};
    *abs3.y.strides = {};

    ge::ascir_op::Add add1("add1");
    add1.x1 = abs2.y;
    add1.x2 = Local2Cast1.y;
    add1.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add1.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_1_mul_output_data_type(&add1, 0);
    add_1_mul_output_data_type = ge::DT_FLOAT16;
    *add1.y.axis = {a.id, c.id, e.id};
    *add1.y.repeats = {A, C, E};
    *add1.y.strides = {C * E, E, ONE};

    ge::ascir_op::Add add2("add2");
    add2.x1 = add1.y;
    add2.x2 = abs1.y;
    add2.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add2.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType add_2_mul_output_data_type(&add2, 0);
    add_2_mul_output_data_type = ge::DT_FLOAT16;
    *add2.y.axis = {a.id, c.id, e.id};
    *add2.y.repeats = {A, C, E};
    *add2.y.strides = {C * E, E, ONE};

    ge::ascir_op::Store x_out("x_out1_mul");
    x_out.x = add2.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out1("x_out2_mul");
    x_out1.x = Local2Cast1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_out2("x_out3_mul");
    x_out2.x = abs3.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    x_out2.y.dtype = DT_FLOAT16;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output3("x_output3");
    x_output3.x = x_out2.y;
    x_output3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output3.attr.sched.loop_axis = c.id;
    x_output3.y.dtype = DT_FLOAT;
    *x_output3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output3.y.repeats = {A, B, C, D, E};
    *x_output3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_output1");
    auto x_out_node1 = graph.FindNode("x_output2");
    auto x_out_node2 = graph.FindNode("x_output3");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAddAscGraphWithNoEmptyRepeats(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, ONE, ONE, ONE};
    *x2.y.strides = {B, ONE, ZERO, ZERO, ZERO};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Abs abs("abs_4");
    abs.x = add.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT16;
    *abs.y.axis = {};
    *abs.y.repeats = {};
    *abs.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }


  std::shared_ptr<AscGraph> CreatAddAscGraphWithDiffRepeats(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, ONE, ONE, ONE};
    *x2.y.strides = {B, ONE, ZERO, ZERO, ZERO};
  
    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithDiffRepeatsMutilReference(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x1Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarToStore(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Scalar x2_scalar("x2_scalar", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x2_scalar.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithDiffRepeatsWithScalar(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Scalar x2_scalar("x2_scalar", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x2_scalar.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarToAdd0(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Scalar x2_scalar("scalar_to_add", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, e.id, d.id}; // 为了测试补轴sched.axis是否从graph获取
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x2_scalar.y;
    add.x2 = x1Local.y;
    add.attr.sched.axis = {};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarToAdd1Int8(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Scalar x2_scalar("scalar_to_add", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_INT8;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x2_scalar.y;
    add.x2 = x1Local.y;
    add.attr.sched.axis = {};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Store x_out0("x_out_4");
    x_out0.x = x2_scalar.y;
    x_out0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out0.attr.sched.loop_axis = c.id;
    x_out0.y.dtype = DT_FLOAT16;
    *x_out0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out0.y.repeats = {A, B, C, D, E};
    *x_out0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out0.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node0 = graph.FindNode("x_output1");
    auto x_out_node = graph.FindNode("x_output2");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node0, 0}, {x_out_node, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarAbsToAdd(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Scalar x2_scalar("scalar_to_add", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};

    ge::ascir_op::Abs abs1("abs1");
    abs1.x = x2_scalar.y;
    abs1.attr.sched.axis = {};
    abs1.y.dtype = DT_FLOAT16;
    *abs1.y.axis = {};
    *abs1.y.repeats = {};
    *abs1.y.strides = {};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs1.y;
    add.x2 = x1Local.y;
    add.attr.sched.axis = {};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarMulRefToAdd(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Scalar x2_scalar("scalar_to_add", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};

    ge::ascir_op::Add add("add_4");
    add.x1 = x2_scalar.y;
    add.x2 = x2_scalar.y;
    add.attr.sched.axis = {};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  // 测试IsDtypeNotSupportOp返回true的场景：Cast节点且Broadcast不支持该数据类型
  std::shared_ptr<AscGraph> CreatAscGraphCastDtypeNotSupport(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);

    // 创建一个不支持的数据类型，假设DT_INT16是Broadcast不支持的类型
    ge::ascir_op::Data x1("x1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT16;
    *x1.y.axis = {a.id, b.id, c.id};
    *x1.y.repeats = {ONE, ONE, C};
    *x1.y.strides = {ZERO, ZERO, ONE};

    // Broadcast节点，使用与输入相同的不支持的数据类型
    ge::ascir_op::Load x1Local("x1Local");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id};
    x1Local.y.dtype = DT_INT16;
    *x1Local.y.axis = {a.id, b.id, c.id};
    *x1Local.y.repeats = {A, B, C};
    *x1Local.y.strides = {B * C, C, ONE};

    // Abs节点
    ge::ascir_op::Abs abs("abs");
    abs.x = x1Local.y;
    abs.attr.sched.axis = {a.id, b.id, c.id};
    abs.y.dtype = DT_INT16;
    *abs.y.axis = {a.id, b.id, c.id};
    *abs.y.repeats = {A, B, C};
    *abs.y.strides = {B * C, C, ONE};

    // Cast节点，转换为支持的数据类型
    ge::ascir_op::Cast cast0("cast_dtype_support");
    cast0.x = abs.y;
    cast0.attr.sched.axis = {a.id, b.id, c.id};
    cast0.y.dtype = DT_FLOAT;  // 使用Broadcast支持的数据类型
    *cast0.y.axis = {a.id, b.id, c.id};
    *cast0.y.repeats = {A, B, C};
    *cast0.y.strides = {B * C, C, ONE};

    // Cast节点，转换为不支持的数据类型
    ge::ascir_op::Cast cast("cast_dtype_not_support");
    cast.x = cast0.y;
    cast.attr.sched.axis = {a.id, b.id, c.id};
    cast.y.dtype = DT_BF16;  // 使用Broadcast不支持的数据类型
    *cast.y.axis = {a.id, b.id, c.id};
    *cast.y.repeats = {A, B, C};
    *cast.y.strides = {B * C, C, ONE};

    // Store和Output节点
    ge::ascir_op::Store x_store("x_store_dtype_not_support");
    x_store.x = cast.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id};
    x_store.attr.sched.loop_axis = c.id;
    x_store.y.dtype = DT_BF16;
    *x_store.y.axis = {a.id, b.id, c.id};
    *x_store.y.repeats = {A, B, C};
    *x_store.y.strides = {B * C, C, ONE};

    ge::ascir_op::Output x_out("x_out_dtype_not_support");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_BF16;
    *x_out.y.axis = {a.id, b.id, c.id};
    *x_out.y.repeats = {A, B, C};
    *x_out.y.strides = {B * C, C, ONE};

    // 设置输出节点
    auto x_out_node = graph.FindNode("x_out_dtype_not_support");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarToAdd1(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Scalar x2_scalar("x2_scalar", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};
  
    ge::ascir_op::Scalar x2_scalar2("x2_scalar2", graph);
    x2_scalar2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar2.attr.sched.loop_axis = c.id;
    x2_scalar2.y.dtype = DT_FLOAT16;
    *x2_scalar2.y.axis = {};
    *x2_scalar2.y.repeats = {};
    *x2_scalar2.y.strides = {};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x2_scalar.y;
    add.x2 = x2_scalar2.y;
    add.attr.sched.axis = {};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {};
    *add.y.repeats = {};
    *add.y.strides = {};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithScalarToAbs(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Scalar x2_scalar("x2_scalar", graph);
    x2_scalar.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2_scalar.attr.sched.loop_axis = c.id;
    x2_scalar.y.dtype = DT_FLOAT16;
    *x2_scalar.y.axis = {};
    *x2_scalar.y.repeats = {};
    *x2_scalar.y.strides = {};
  
    ge::ascir_op::Abs x1LocalAbs1("x1LocalAbs1");
    x1LocalAbs1.x = x2_scalar.y;
    x1LocalAbs1.attr.sched.axis = {};
    x1LocalAbs1.y.dtype = DT_FLOAT16;
    *x1LocalAbs1.y.axis = {};
    *x1LocalAbs1.y.repeats = {};
    *x1LocalAbs1.y.strides = {};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x1LocalAbs1.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithDiffRepeatsMutilReference2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs x1LocalAbs1("x1LocalAbs1");
    x1LocalAbs1.x = x1Local.y;
    x1LocalAbs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1LocalAbs1.y.dtype = DT_FLOAT16;
    *x1LocalAbs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1LocalAbs1.y.repeats = {A, B, C, D, E};
    *x1LocalAbs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs x1LocalAbs2("x1LocalAbs2");
    x1LocalAbs2.x = x1Local.y;
    x1LocalAbs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1LocalAbs2.y.dtype = DT_FLOAT16;
    *x1LocalAbs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1LocalAbs2.y.repeats = {A, B, C, D, E};
    *x1LocalAbs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = x1LocalAbs1.y;
    add.x2 = x1LocalAbs2.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  // 当前ascir_op不支持Cast，当前的Cast节点使用Abs代替，人为再后面改为Cast
  std::shared_ptr<AscGraph> CreatAddWtihCastInFrontStoreAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_BF16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_BF16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_FLOAT;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = Local2Cast0.y;
    add.x2 = Local4Cast0.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_FLOAT16;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = add_Cast.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT16;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs_Cast("abs_Cast");
    abs_Cast.x = abs.y;
    abs_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_Cast.y.dtype = DT_FLOAT16;
    *abs_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_Cast.y.repeats = {A, B, C, D, E};
    *abs_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs_Cast.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAddWtihInt8ToFloat16(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT8;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT8;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_INT8;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local4Cast1("Local4Cast1");
    Local4Cast1.x = Local4Cast0.y;
    Local4Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast1.y.dtype = DT_FLOAT16;
    *Local4Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast1.y.repeats = {A, B, C, D, E};
    *Local4Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = Local2Cast0.y;
    add.x2 = Local4Cast1.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_UINT8;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add_Cast.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_UINT8;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAddWtihNoCastInfrontStoreAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_BF16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_BF16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_FLOAT;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = Local2Cast0.y;
    add.x2 = Local4Cast0.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_FLOAT;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = add_Cast.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> DtypeNoFloatAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_INT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_INT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_INT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_INT16;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = Local2Cast0.y;
    add.x2 = Local4Cast0.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_INT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_INT16;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = add_Cast.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_INT16;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_INT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> BroadcastBackwardMulInputsAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_INT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_INT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_INT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs cast2abs("abs_3");
    cast2abs.x = Local2Cast0.y;
    cast2abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    cast2abs.y.dtype = DT_INT16;
    *cast2abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *cast2abs.y.repeats = {A, B, C, D, E};
    *cast2abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs2Cast0("Abs2Cast0");
    abs2Cast0.x = cast2abs.y;
    abs2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs2Cast0.y.dtype = DT_FLOAT;
    *abs2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs2Cast0.y.repeats = {A, B, C, D, E};
    *abs2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_INT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_INT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_FLOAT;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = abs2Cast0.y;
    add.x2 = Local4Cast0.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_FLOAT;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs("abs_4");
    abs.x = add_Cast.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  // dim 2做了reduce
  std::shared_ptr<AscGraph> BroadcastBackwardReduceAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_reduce", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_reduce");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs abs("abs_4");
    abs.x = x1Local.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT;
    *abs.y.axis = {};
    *abs.y.repeats = {};
    *abs.y.strides = {};

    ge::ascir_op::Max reduce("reduce_reduce");
    reduce.x = abs.y;
    reduce.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *reduce.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *reduce.y.repeats = {A, ONE, C, D, E};
    *reduce.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

    ge::ascir_op::Store x_store("x_store_reduce");
    x_store.x = reduce.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, ONE, C, D, E};
    *x_store.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

    ge::ascir_op::Output x_out("x_out_reduce");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, ONE, C, D, E};
    *x_out.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_out_reduce");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> EmptyTensorAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_calc_rstd", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, ZERO};
    *x1.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Load x1Local("x1Local_calc_rstd");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {ONE, ONE, ONE, D, ZERO};
    *x1Local.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Abs x1Abs("x1_abs");
    x1Abs.x = x1Local.y;
    x1Abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};

    ge::ascir_op::Store x_store("x_store_calc_rstd");
    x_store.x = x1Abs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {ONE, ONE, ONE, D, ZERO};
    *x_store.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    ge::ascir_op::Output x_out("x_out_calc_rstd");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {ONE, ONE, ONE, D, ZERO};
    *x_out.y.strides = {ZERO, ZERO, ZERO, ZERO, ZERO};

    auto x_out_node = graph.FindNode("x_out_calc_rstd");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> BroadcastBackwardMulOutputAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_calc_rstd", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_calc_rstd");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("x2_calc_rstd", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_calc_rstd");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x3("x3_calc_rstd", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {A, B, C, D, E};
    *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("x3Local_calc_rstd");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::FlashSoftmax calcRstd("calcRstd_calc_rstd");
    calcRstd.x1 = x1Local.y;
    calcRstd.x2 = x2Local.y;
    calcRstd.x3 = x3Local.y;
    calcRstd.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *calcRstd.y1.axis = {a.id, b.id, c.id, d.id, e.id};
    *calcRstd.y1.repeats = {A, B, C, D, E};
    *calcRstd.y1.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    *calcRstd.y2.axis = {a.id, b.id, c.id, d.id, e.id};
    *calcRstd.y2.repeats = {A, B, C, D, E};
    *calcRstd.y2.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    *calcRstd.y3.axis = {a.id, b.id, c.id, d.id, e.id};
    *calcRstd.y3.repeats = {A, B, C, D, E};
    *calcRstd.y3.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("x_store_calc_rstd");
    x_store.x = calcRstd.y1;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_store1("x_store1_calc_rstd");
    x_store1.x = calcRstd.y2;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_store2("x_store2_calc_rstd");
    x_store2.x = calcRstd.y2;
    x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store2.attr.sched.loop_axis = c.id;
    *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store2.y.repeats = {A, B, C, D, E};
    *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("x_out_calc_rstd");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Output x_out1("x_out1_calc_rstd");
    x_out1.x = x_store1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Output x_out2("x_out2_calc_rstd");
    x_out2.x = x_store2.y;
    x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out2.attr.sched.loop_axis = c.id;
    *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out2.y.repeats = {A, B, C, D, E};
    *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("x_out_calc_rstd");
    auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
    auto x_out_node2 = graph.FindNode("x_out2_calc_rstd");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(3U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> MidCastAndLoadCastAndStoreCastAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_FLOAT16;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = Local2Cast0.y;
    add.x2 = Local4Cast0.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_INT16;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = add_Cast.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_INT16;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_INT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> MidCastAndLoadCastAndStoreCastAscGraph2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_INT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Data x2("x2_3", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    x2.y.dtype = DT_FLOAT16;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_4");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local4Cast0("Local4Cast0");
    Local4Cast0.x = x2Local.y;
    Local4Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local4Cast0.y.dtype = DT_INT16;
    *Local4Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local4Cast0.y.repeats = {A, B, C, D, E};
    *Local4Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Add add("add_4");
    add.x1 = Local2Cast0.y;
    add.x2 = Local4Cast0.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_INT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast add_Cast("add_Cast");
    add_Cast.x = add.y;
    add_Cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add_Cast.y.dtype = DT_FLOAT16;
    *add_Cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add_Cast.y.repeats = {A, B, C, D, E};
    *add_Cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = add_Cast.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT16;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast("abs_4_cast");
    abs_4_cast.x = abs.y;
    abs_4_cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast.y.dtype = DT_FLOAT;
    *abs_4_cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast.y.repeats = {A, B, C, D, E};
    *abs_4_cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs_4_cast.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithStoreMutilReference(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs x1LocalAbs1("x1LocalAbs1");
    x1LocalAbs1.x = x1Local.y;
    x1LocalAbs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1LocalAbs1.y.dtype = DT_FLOAT16;
    *x1LocalAbs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1LocalAbs1.y.repeats = {A, B, C, D, E};
    *x1LocalAbs1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs x1LocalAbs2("x1LocalAbs2");
    x1LocalAbs2.x = x1Local.y;
    x1LocalAbs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1LocalAbs2.y.dtype = DT_FLOAT16;
    *x1LocalAbs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1LocalAbs2.y.repeats = {A, B, C, D, E};
    *x1LocalAbs2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = x1LocalAbs1.y;
    add.x2 = x1LocalAbs2.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs x1LocalAbs3("x1LocalAbs3");
    x1LocalAbs3.x = add.y;
    x1LocalAbs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1LocalAbs3.y.dtype = DT_FLOAT16;
    *x1LocalAbs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1LocalAbs3.y.repeats = {A, B, C, D, E};
    *x1LocalAbs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = add.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out1("x_out_6");
    x_out1.x = x1LocalAbs3.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    auto x_out_node1 = graph.FindNode("x_output2");
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAscGraphWithCompleteAttrAndBroadcast(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT16;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {ONE, ONE, ONE, D, E};
    *x1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT16;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_2");
    x2Local.x = x1.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2Local.y.dtype = DT_FLOAT16;
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {ONE, ONE, ONE, D, E};
    *x2Local.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Load x3Local("x3Local_2");
    x3Local.x = x1.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3Local.y.dtype = DT_FLOAT16;
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {ONE, B, ONE, D, E};
    *x3Local.y.strides = {ZERO, D * E, ZERO, E, ONE};

    ge::ascir_op::Abs x2LocalAbs1("x2LocalAbs1");
    x2LocalAbs1.x = x2Local.y;
    x2LocalAbs1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2LocalAbs1.y.dtype = DT_FLOAT16;
    *x2LocalAbs1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2LocalAbs1.y.repeats = {ONE, ONE, ONE, D, E};
    *x2LocalAbs1.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Abs x3LocalAbs2("x3LocalAbs2");
    x3LocalAbs2.x = x3Local.y;
    x3LocalAbs2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3LocalAbs2.y.dtype = DT_FLOAT16;
    *x3LocalAbs2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3LocalAbs2.y.repeats = {ONE, B, ONE, D, E};
    *x3LocalAbs2.y.strides = {ZERO, D * E, ZERO, E, ONE};

    ge::ascir_op::Add add("add_4");
    add.x1 = x1Local.y;
    add.x2 = x3LocalAbs2.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    add.y.dtype = DT_FLOAT16;
    AscOutputAttrDataType x2Local_output_data_type(&add, 0);
    x2Local_output_data_type = ge::DT_FLOAT16;
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs x1LocalAbs3("x1LocalAbs3");
    x1LocalAbs3.x = add.y;
    x1LocalAbs3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1LocalAbs3.y.dtype = DT_FLOAT16;
    *x1LocalAbs3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1LocalAbs3.y.repeats = {A, B, C, D, E};
    *x1LocalAbs3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = x2LocalAbs1.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT16;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {ONE, ONE, ONE, D, E};
    *x_out.y.strides = {ZERO, ZERO, ZERO, E, ONE};

    ge::ascir_op::Store x_out1("x_out_6");
    x_out1.x = x1LocalAbs3.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    x_out1.y.dtype = DT_FLOAT16;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output2("x_output2");
    x_output2.x = x_out1.y;
    x_output2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output2.attr.sched.loop_axis = c.id;
    x_output2.y.dtype = DT_FLOAT;
    *x_output2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output2.y.repeats = {A, B, C, D, E};
    *x_output2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    auto x_out_node1 = graph.FindNode("x_output2");
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 0}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> ContinuousCastAfterLoadAndBeforeStoreAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast1("Local2Cast1");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast2("Local2Cast2");
    Local2Cast2.x = Local2Cast1.y;
    Local2Cast2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast2.y.dtype = DT_FLOAT16;
    *Local2Cast2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast2.y.repeats = {A, B, C, D, E};
    *Local2Cast2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast3("Local2Cast3");
    Local2Cast3.x = Local2Cast2.y;
    Local2Cast3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast3.y.dtype = DT_FLOAT;
    *Local2Cast3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast3.y.repeats = {A, B, C, D, E};
    *Local2Cast3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = Local2Cast3.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_FLOAT;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast("abs_4_cast");
    abs_4_cast.x = abs.y;
    abs_4_cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast.y.dtype = DT_FLOAT16;
    *abs_4_cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast.y.repeats = {A, B, C, D, E};
    *abs_4_cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast1("abs_4_cast1");
    abs_4_cast1.x = abs_4_cast.y;
    abs_4_cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast1.y.dtype = DT_FLOAT;
    *abs_4_cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast1.y.repeats = {A, B, C, D, E};
    *abs_4_cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast2("abs_4_cast2");
    abs_4_cast2.x = abs_4_cast1.y;
    abs_4_cast2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast2.y.dtype = DT_FLOAT16;
    *abs_4_cast2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast2.y.repeats = {A, B, C, D, E};
    *abs_4_cast2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast3("abs_4_cast3");
    abs_4_cast3.x = abs_4_cast2.y;
    abs_4_cast3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast3.y.dtype = DT_FLOAT;
    *abs_4_cast3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast3.y.repeats = {A, B, C, D, E};
    *abs_4_cast3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs_4_cast3.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> ContinuousCastAfterLoadAndBeforeStoreAscGraph2(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
  
    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
  
    ge::ascir_op::Data x1("x1_1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    x1.y.dtype = DT_FLOAT;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("x1Local_2");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1Local.y.dtype = DT_FLOAT;
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast Local2Cast0("Local2Cast0");
    Local2Cast0.x = x1Local.y;
    Local2Cast0.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast0.y.dtype = DT_FLOAT16;
    *Local2Cast0.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast0.y.repeats = {A, B, C, D, E};
    *Local2Cast0.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast1("Local2Cast1");
    Local2Cast1.x = Local2Cast0.y;
    Local2Cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast1.y.dtype = DT_FLOAT;
    *Local2Cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast1.y.repeats = {A, B, C, D, E};
    *Local2Cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast2("Local2Cast2");
    Local2Cast2.x = Local2Cast1.y;
    Local2Cast2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast2.y.dtype = DT_FLOAT16;
    *Local2Cast2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast2.y.repeats = {A, B, C, D, E};
    *Local2Cast2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Cast Local2Cast3("Local2Cast3");
    Local2Cast3.x = Local2Cast2.y;
    Local2Cast3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    Local2Cast3.y.dtype = DT_INT16;
    *Local2Cast3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *Local2Cast3.y.repeats = {A, B, C, D, E};
    *Local2Cast3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Abs abs("abs_4");
    abs.x = Local2Cast3.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs.y.dtype = DT_INT16;
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast("abs_4_cast");
    abs_4_cast.x = abs.y;
    abs_4_cast.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast.y.dtype = DT_FLOAT16;
    *abs_4_cast.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast.y.repeats = {A, B, C, D, E};
    *abs_4_cast.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast1("abs_4_cast1");
    abs_4_cast1.x = abs_4_cast.y;
    abs_4_cast1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast1.y.dtype = DT_FLOAT;
    *abs_4_cast1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast1.y.repeats = {A, B, C, D, E};
    *abs_4_cast1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast2("abs_4_cast2");
    abs_4_cast2.x = abs_4_cast1.y;
    abs_4_cast2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast2.y.dtype = DT_FLOAT16;
    *abs_4_cast2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast2.y.repeats = {A, B, C, D, E};
    *abs_4_cast2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Cast abs_4_cast3("abs_4_cast3");
    abs_4_cast3.x = abs_4_cast2.y;
    abs_4_cast3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    abs_4_cast3.y.dtype = DT_FLOAT;
    *abs_4_cast3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs_4_cast3.y.repeats = {A, B, C, D, E};
    *abs_4_cast3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  
    ge::ascir_op::Store x_out("x_out_5");
    x_out.x = abs_4_cast3.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    x_out.y.dtype = DT_FLOAT;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_output1("x_output1");
    x_output1.x = x_out.y;
    x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_output1.attr.sched.loop_axis = c.id;
    x_output1.y.dtype = DT_FLOAT;
    *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_output1.y.repeats = {A, B, C, D, E};
    *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_output1");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  /**
   *      netoutput1
   *         |
   *       shapeNo1
   *        |
   *      addnYes1
   *     /    \.
   *   /       \.
   * const1   const2
   */
  ComputeGraphPtr BuildGraph1(const std::string node_type = "") {
    auto builder = GraphBuilder("test", node_type);
    auto const1 = builder.AddNode("const1", "CONSTANT", 0, 1);
    auto const2 = builder.AddNode("const2", "CONSTANT", 0, 1);
    auto addn1 = builder.AddNode("addn1", "AddNYes", 2, 1);
    auto shape1 = builder.AddNode("shape1", "ShapeNo", 1, 1);
    auto netoutput1 = builder.AddNode("netoutput", "NETOUTPUT", 1, 0);
  
    builder.AddDataEdge(const1, 0, addn1, 0);
    builder.AddDataEdge(const2, 0, addn1, 1);
    builder.AddDataEdge(addn1, 0, shape1, 0);
    builder.AddDataEdge(shape1, 0, netoutput1, 0);
  
    return builder.GetGraph();
  }

  ComputeGraphPtr BuildGraphWithSubGraph(const std::string node_type = "") {
    auto root_builder = GraphBuilder("root", node_type);
    const auto &data0 = root_builder.AddNode("data0", "Data", 1, 1);
    const auto &case0 = root_builder.AddNode("case0", "Case", 1, 1);
    const auto &relu0 = root_builder.AddNode("relu0", "Relu", 1, 1);
    const auto &relu1 = root_builder.AddNode("relu1", "Relu", 1, 1);
    const auto &netoutput = root_builder.AddNode("netoutput", "NetOutput", 1, 1);
    const auto &root_graph = root_builder.GetGraph();
    root_builder.AddDataEdge(data0, 0, case0, 0);
    root_builder.AddDataEdge(case0, 0, relu0, 0);
    root_builder.AddDataEdge(relu0, 0, relu1, 0);
    root_builder.AddDataEdge(relu1, 0, netoutput, 0);

    auto sub_builder1 = GraphBuilder("sub1", node_type);
    const auto &data1 = sub_builder1.AddNode("data1", "Data", 0, 1);
    const auto &sub_graph1 = sub_builder1.GetGraph();
    root_graph->AddSubGraph(sub_graph1);
    sub_graph1->SetParentNode(case0);
    sub_graph1->SetParentGraph(root_graph);
    case0->GetOpDesc()->AddSubgraphName("branch1");
    case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

    auto sub_builder2 = GraphBuilder("sub2", node_type);
    const auto &data2 = sub_builder2.AddNode("data2", "Data", 0, 1);
    const auto &sub_graph2 = sub_builder2.GetGraph();
    root_graph->AddSubGraph(sub_graph2);
    sub_graph2->SetParentNode(case0);
    sub_graph2->SetParentGraph(root_graph);
    case0->GetOpDesc()->AddSubgraphName("branch2");
    case0->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
    root_graph->TopologicalSorting();
    return root_graph;
  }
  
}

TEST_F(AscBackendPostProcessorTest, AddOutput_NodeAscBcAndDirectAscBc_FAILED) {
  ComputeGraphPtr graph = BuildGraph1("AscBackend");
  ASSERT_NE(graph, nullptr);

  int loop_cnt = 0;
  static std::atomic<int64_t> i{0};
  ge::AscGraph graph_fuse1("add");
  auto asc_graph_fuse1 = CreatAddAscGraph(graph_fuse1);
  ASSERT_NE(asc_graph_fuse1, nullptr);
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    if (loop_cnt == 0) {
      std::string type = "FusedAscBackend";
      op_desc->SetType(type);
      auto attr = GetOrCreateAutoFuseAttrs(op_desc);
      ASSERT_NE(attr, nullptr);
      attr->SetFuseComputeGraph(AscGraphUtils::GetComputeGraph(*asc_graph_fuse1));
    }
    loop_cnt++;
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  ASSERT_NE(post_processor.Do(graph), SUCCESS);
}

TEST_F(AscBackendPostProcessorTest, AdapterAndPass_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
}

TEST_F(AscBackendPostProcessorTest, Adaption_EmptyRepeatsAfterLoad_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithEmptyRepeatsAfterLoad(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output") {
      continue;
    }
    printf("Adaption_EmptyRepeatsAfterLoad_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "abs_empty_tensor") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      // Broadcast后移后 Abs的repeats向前更新为Broadcast前Load节点
      std::vector<ge::Expression> expect_repeats = {ONE, ONE, ONE, D, E};
      std::vector<ge::Expression> expect_strides = {ZERO, ZERO, ZERO, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }
    if (node->GetName() == "add_4" || node->GetName() == "abs_4" || node->GetName() == "abs_5") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {ONE, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {ZERO, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("Adaption_EmptyRepeats_OK To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
}

TEST_F(AscBackendPostProcessorTest, Adaption_EmptyRepeats_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithEmptyRepeats(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("Adaption_EmptyRepeats_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("Adaption_EmptyRepeats_OK To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    if (node->GetType() == "Output") {
      continue;
    }
    printf("Adaption_EmptyRepeats_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "add_4" || node->GetName() == "abs_4" || node->GetName() == "abs_5") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("Adaption_EmptyRepeats_OK To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
}

TEST_F(AscBackendPostProcessorTest, Split_Pass_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatSplitDoubleOutputAscGraph(add_graph1),loop::FuseType::kSplit);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

}

// 测试两个不同dtype的cast不做cse消除，两个相同的dtype的cast会做cse消除
TEST_F(AscBackendPostProcessorTest, Adaption_CastWithSameDtype_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CastWithSameDtypeAscGraph");
  attr1->SetAscGraph(CastWithSameDtypeAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  size_t add_cnt = 0U;
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("2Sub2InputallMulReference node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    } else if (node->GetType() == "Add") {
      add_cnt++;
      continue;
    } else if (node->GetType() == "Cast") {
      cast_cnt++;
      continue;
    }
  }
  EXPECT_EQ(output_cnt, 3);
  EXPECT_EQ(add_cnt, 1);
  EXPECT_EQ(cast_cnt, 4);
}

// 测试两个sub的两个输入都是可以做cse消除，同时sub也能cse消除
TEST_F(AscBackendPostProcessorTest, Adaption_2Sub2InputCanCse_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TwoSub2InputallMulReferenceAscGraph");
  attr1->SetAscGraph(TwoSub2InputallMulReferenceAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  size_t add_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("2Sub2InputallMulReference node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    }
    if (node->GetType() == "Add") {
      add_cnt++;
      continue;
    }
  }
  EXPECT_EQ(output_cnt, 3);
  EXPECT_EQ(add_cnt, 1);
}

// 测试三个输出多引用也是可以消除的
TEST_F(AscBackendPostProcessorTest, Adaption_3Sub2InputCanCse_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("ThreeSub2InputallMulReferenceAscGraph");
  attr1->SetAscGraph(ThreeSub2InputallMulReferenceAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  size_t add_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("2Sub2InputallMulReference node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    }
    if (node->GetType() == "Add") {
      add_cnt++;
      continue;
    }
  }
  EXPECT_EQ(output_cnt, 3);
  EXPECT_EQ(add_cnt, 1);
}

// 测试三个scalar输出多引用也是可以消除的
TEST_F(AscBackendPostProcessorTest, Adaption_3ScalarCanCse_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("ThreeScalarAscGraph");
  attr1->SetAscGraph(ThreeScalarAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  size_t add_cnt = 0U;
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("2Sub2InputallMulReference node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    } else if (node->GetType() == "Add") {
      add_cnt++;
      continue;
    } else if (node->GetType() == "Scalar") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {ONE, ONE, ONE, ONE, ONE};
      std::vector<ge::Expression> expect_strides = {ZERO, ZERO, ZERO, ZERO, ZERO};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }
  }
  EXPECT_EQ(output_cnt, 3);
  EXPECT_EQ(add_cnt, 1);
}

// 测试三个scalar输出多引用,两个value一样的可以消除，一个不一样的不能消除
TEST_F(AscBackendPostProcessorTest, Adaption_3ScalarHas2ScalarCanCse_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("ThreeScalarTwoSameValueAscGraph");
  attr1->SetAscGraph(ThreeScalarTwoSameValueAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  size_t add_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("2Sub2InputallMulReference node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    }
    if (node->GetType() == "Add") {
      add_cnt++;
      continue;
    }
  }
  EXPECT_EQ(output_cnt, 3);
  EXPECT_EQ(add_cnt, 2);
}

// 测试三个scalar输出多引用,两个value一样而且dtype一样的可以消除，一个dtype不一样的不能消除
TEST_F(AscBackendPostProcessorTest, Adaption_3ScalarHas2ScalarCanCse2_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("ThreeScalarTwoSameValueDtypeAscGraph");
  attr1->SetAscGraph(ThreeScalarTwoSameValueDtypeAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  size_t add_cnt = 0U;
  size_t scalar_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("2Sub2InputallMulReference node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    } else if (node->GetType() == "Add") {
      add_cnt++;
      continue;
    } else if (node->GetType() == "Scalar") {
      scalar_cnt++;
      continue;
    }
  }
  EXPECT_EQ(output_cnt, 3);
  EXPECT_EQ(add_cnt, 2);
  EXPECT_EQ(scalar_cnt, 2);
}

TEST_F(AscBackendPostProcessorTest, Adaption_StoreMulReference_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatCalcRstdAscGraphWithStoreMulReference(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("Adaption_EmptyRepeats_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("Adaption_EmptyRepeats_OK To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果 output不能被cse删除，因为outut和下一个图的输出是连在一起的，不能通过多引用表达
  size_t output_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output") {
      output_cnt++;
      continue;
    }
  }
  EXPECT_EQ(output_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CreatAscGraphWithLoadAndCastMulReferenceAndInsertBroadcast) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add_2350");
  attr1->SetAscGraph(CreatAscGraphWithLoadAndCastMulReferenceAndInsertBroadcast(add_graph1));

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    printf("Adaption_CreatAscGraphWithLoadAndCastMulReferenceAndInsertBroadcast %s %s %s\n.", node->GetName().c_str(),
           node->GetType().c_str(), TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") { // cse会消除一个asb2和一个cast
      if (cast_cnt == 0) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      } else if (cast_cnt == 1) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      } else if (cast_cnt == 2) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      } else if (cast_cnt == 3) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      }
      cast_cnt++;
    }
  }
  EXPECT_EQ(cast_cnt, 4);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CreatAscGraphWithLoadAndCastMulReference) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add_2409");
  attr1->SetAscGraph(CreatAscGraphWithLoadAndCastMulReference(add_graph1));

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    printf("Adaption_CreatAscGraphWithLoadAndCastMulReference %s %s %s\n.", node->GetName().c_str(),
           node->GetType().c_str(), TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") { // cse会消除一个asb2
      if (cast_cnt == 0) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      } else if (cast_cnt == 1) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      } else if (cast_cnt == 2) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      } else if (cast_cnt == 3) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      } else if (cast_cnt == 4) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      } else if (cast_cnt == 5) {
        EXPECT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      }
      cast_cnt++;
    }
  }
  EXPECT_EQ(cast_cnt, 3);
}

// // 测试复杂的带有broadcast和transpose的graph作为输入测试多个transpose反推场景,后端不支持一个ascgraph多个transpose，暂时不跑此用例
// TEST_F(AscBackendPostProcessorTest, Adaption_TransposeWithBroadcastNodeAscGraph0_OK) {
//   ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
//   EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

//   auto addn1 = compute_graph->FindNode("addn1");
//   ASSERT_NE(addn1, nullptr);
//   auto op_desc1 = addn1->GetOpDescBarePtr();
//   ASSERT_NE(op_desc1, nullptr);
//   auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
//   ASSERT_NE(attr1, nullptr);
//   std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
//                                                         {"const2", DT_FLOAT16},
//                                                         {"shape1", DT_FLOAT16},
//                                                         {"netoutput", DT_FLOAT16}};
//   for (auto name : names) {
//     auto node = compute_graph->FindNode(name.first);
//     ASSERT_NE(node, nullptr);
//     auto op_desc = node->GetOpDescBarePtr();
//     ASSERT_NE(op_desc, nullptr);
//     op_desc->SetType("NotAscBc");
//     auto attr = GetOrCreateAutoFuseAttrs(op_desc);
//     ASSERT_NE(attr, nullptr);
//   }

//   ge::AscGraph add_graph1("Transpose_10");
//   attr1->SetAscGraph(TransposeAscGraphWithBroadcastNodeAscGraph0(add_graph1));

//   // 走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
//   AscBackendPostProcessor post_processor;
//   EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
//   auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
//   ASSERT_NE(shape_env_attr, nullptr);
//   EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
//   // 校验结果
//   auto ONE = Symbol(1);
//   auto ZERO = Symbol(0);
//   const Expression A = add_graph1.CreateSizeVar("A");
//   const Expression B = add_graph1.CreateSizeVar("B");
//   const Expression C = add_graph1.CreateSizeVar("C");
//   const Expression D = add_graph1.CreateSizeVar("D");
//   const Expression E = add_graph1.CreateSizeVar("E");
//   int64_t t_cnt = 0;
//   int64_t b_cnt = 0;
//   for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
//     if (node->GetType() == "Output" || node->GetType() == "Data") {
//       continue;
//     }
//     const auto &op_desc = node->GetOpDesc();
//     auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
//     ASSERT_NE(node_attr, nullptr);
//     printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
//     if (node->GetName() == "x1_t_b" || node->GetName() == "x3_t_b") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<int64_t> expect_sched_axis = {1, 2, 0, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//       std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//     } else if (node->GetName() == "x2_t_b") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {1, 2, 0, 4, 3};
//       std::vector<int64_t> expect_sched_axis = {1, 2, 0, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {B, C, A, ONE, D};
//       std::vector<ge::Expression> expect_strides = {C * A * D, A * D, D, ZERO, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//     } else if (node->GetName() == "x_out2_mul") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {1, 2, 0, 4, 3};
//       std::vector<int64_t> expect_sched_axis = {1, 2, 0, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {B, C, A, E, D};
//       std::vector<ge::Expression> expect_strides = {C * A * E * D, A * E * D, E * D, D, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//     } else if (node->GetType() == "Transpose") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_sched_axis = {1, 2, 0, 4, 3};
//       if (t_cnt == 0 || t_cnt == 3) {
//         std::vector<int64_t> expect_axis = {2, 1, 0, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {C, B, A, E, D};
//         std::vector<ge::Expression> expect_strides = {B * A * E * D, A * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//       } else if (t_cnt == 1 || t_cnt == 4) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, E, D};
//         std::vector<ge::Expression> expect_strides = {B * C * E * D, C * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//       } else if (t_cnt == 2 || t_cnt == 5) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//       } else if (t_cnt == 6) {
//         std::vector<int64_t> expect_axis = {2, 1, 0, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {C, B, A, D, E};
//         std::vector<ge::Expression> expect_strides = {B * A * D * E, A * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//       } else if (t_cnt == 7) {
//         std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {B, C, A, D, E};
//         std::vector<ge::Expression> expect_strides = {C * A * D * E, A * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//       } else if (t_cnt == 8) {
//         std::vector<int64_t> expect_axis = {1, 2, 0, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {B, C, A, E, D};
//         std::vector<ge::Expression> expect_strides = {C * A * E * D, A * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(node_attr->sched.axis, expect_sched_axis);
//       }
//       t_cnt++;
//     } else if (node->GetType() == "Broadcast") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       if (b_cnt == 0) {
//         std::vector<int64_t> expect_axis = {1, 2, 0, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {B, C, A, E, D};
//         std::vector<ge::Expression> expect_strides = {C * A * E * D, A * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//       }
//       b_cnt++;
//     }
//   }
//   ASSERT_EQ(b_cnt, 1);
//   ASSERT_EQ(t_cnt, 6);
//   // 取消走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
// }

// 测试的2输入带transpose和两输入elewise垂直融合后移变成1个transpose
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWith2InputTranspose2_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWith2InputTranspose2");
  attr1->SetAscGraph(TransposeAscGraphWith2InputTranspose2(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 2, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, C, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {C * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Store") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
  } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      } else if (b_cnt == 1 || b_cnt == 2) { // 做了broadcast后移动
        std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
        std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 2);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试load和store各有两对轴变化场景通过transpose数量优化为一个load上有两对轴变换
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWith2TransposeInLoadStore_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWith2TransposeInLoadStore");
  attr1->SetAscGraph(TransposeAscGraphWith2TransposeInLoadStore(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 1, 3, 4, 2};
      std::vector<ge::Expression> expect_repeats = {A, B, ONE, E, C};
      std::vector<ge::Expression> expect_strides = {B * E * C, E * C, ZERO, C, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out1_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 3, 4, 2};
      std::vector<ge::Expression> expect_repeats = {A, B, D, E, C};
      std::vector<ge::Expression> expect_strides = {B * D * E * C, D * E * C, E * C, C, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 3, 4, 2};
        std::vector<ge::Expression> expect_repeats = {A, B, ONE, E, C};
        std::vector<ge::Expression> expect_strides = {B * E * C, E * C, ZERO, C, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0 || b_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 3, 4, 2};
        std::vector<ge::Expression> expect_repeats = {A, B, D, E, C};
        std::vector<ge::Expression> expect_strides = {B * D * E * C, D * E * C, E * C, C, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 1);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试transpose上移到transpose dtype支持的节点后
// 1、load上transpose数量和store一样时transpose数量优化会把transpose移到store
// 2、load上transpose数量比store多时，会从最下面开始往上面找位置插入
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWithTransposeDtypeSupported_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWithTransposeDtypeSupported");
  attr1->SetAscGraph(TransposeAscGraphWithTransposeDtypeSupported(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {A * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out1_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, D, E};
      std::vector<ge::Expression> expect_strides = {A * B * D * E, B * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, ONE, E};
        std::vector<ge::Expression> expect_strides = {B * C * E, C * E, E, ZERO, ONE};
//        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E}; // 后续支持多transpose后使用
//        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_INT64);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0 || b_cnt == 1) {
        std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 1);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试基本的2输入带transpose后移变成1个transpose
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWith2InputTranspose_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWith2InputTranspose");
  attr1->SetAscGraph(TransposeAscGraphWith2InputTranspose(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, C, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {C * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out1_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
      std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0 || b_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
        std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 2);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试transpose后引用
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWith2InputTransposeMulReference_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWith2InputTransposeMulReference");
  attr1->SetAscGraph(TransposeAscGraphWith2InputTransposeMulReference(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 2, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, C, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {C * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out1_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {C * B * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0 || b_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
        std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 1);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试基本的2输入带transpose后移变成1个transpose + scalar处于transpose分支
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWith2InputTranspose1_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWith2InputTranspose1ScalarHasTranspose");
  attr1->SetAscGraph(TransposeAscGraphWith2InputTranspose1ScalarHasTranspose(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {2, 0, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {A * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out1_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {2, 0, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, D, E};
      std::vector<ge::Expression> expect_strides = {A * B * D * E, B * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 6 || b_cnt == 7) {
        std::vector<int64_t> expect_axis = {2, 0, 1, 3, 4};
        ASSERT_EQ(attr->axis, expect_axis);
      } else {
//        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4}; // 后续支持多transpose后使用
        std::vector<int64_t> expect_axis = {2, 0, 1, 3, 4};
        ASSERT_EQ(attr->axis, expect_axis);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 6);
  ASSERT_EQ(t_cnt, 0);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试基本的2输入带transpose后移变成1个transpose + scalar处于非transpose分支
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWith2InputTranspose3_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("TransposeAscGraphWith2InputTranspose1ScalarNoTranspose");
  attr1->SetAscGraph(TransposeAscGraphWith2InputTranspose1ScalarNoTranspose(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {A * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_out1_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, D, E};
      std::vector<ge::Expression> expect_strides = {A * B * D * E, B * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "scalar") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      ASSERT_EQ(attr->axis, expect_axis);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0 || b_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        ASSERT_EQ(attr->axis, expect_axis);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 6);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试基本的带有broadcast和transpose的graph作为输入测试transpose反推场景
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeWithBroadcastNodeAscGraph1_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("Transpose_10");
  attr1->SetAscGraph(TransposeAscGraphWithBroadcastNodeAscGraph1(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {0, 2, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, C, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {C * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Store") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, ONE, E};
        std::vector<ge::Expression> expect_strides = {B * C * E, C * E, E, ZERO, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {C * B * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 2);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 测试带有broadcast和级联transpose的graph作为输入测试transpose反推为1个transpose
TEST_F(AscBackendPostProcessorTest, Adaption_TransposeWithBroadcastNodeAscGraph2_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("2_Transpose");
  attr1->SetAscGraph(TransposeAscGraphWithBroadcastNodeAscGraph2(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  int64_t b_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2_mul") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis =  {2, 0, 1, 3, 4};
      std::vector<ge::Expression> expect_repeats = {C, A, B, ONE, E};
      std::vector<ge::Expression> expect_strides = {A * B * E, B * E, E, ZERO, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Store") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, ONE, E};
        std::vector<ge::Expression> expect_strides = {B * C * E, C * E, E, ZERO, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    } else if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (b_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {C * B * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      b_cnt++;
    }
  }
  ASSERT_EQ(b_cnt, 2);
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

TEST_F(AscBackendPostProcessorTest, Adaption_TransposeWithNoAxisTranspose_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("Transpose");
  attr1->SetAscGraph(TransposeAscGraphWithNoAxisTranspose(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Transpose") {
      t_cnt++;
    }
  }
  EXPECT_EQ(t_cnt, 0);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 后端不支持一个ascgraph多个transpose，只融合了一个transpose但是load和store都有transpose的情况下，先做数量优化为1个transpose
TEST_F(AscBackendPostProcessorTest, TransposeAscGraphWithLoadAndStoreHasTranspose_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("Transpose");
  attr1->SetAscGraph(TransposeAscGraphWithLoadAndStoreHasTranspose(add_graph1));

  // 走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
  AscBackendPostProcessor post_processor;
  // EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  int64_t t_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output") {
      continue;
    }
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "x1Local_2" || node->GetName() == "x3Local_2" || node->GetName() == "x_store1") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
      std::vector<ge::Expression> expect_repeats = {B, C, A, D, E};
      std::vector<ge::Expression> expect_strides = {C * A * D * E, A * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x2Local_2" || node->GetName() == "x_store3") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "x_store2") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
      std::vector<ge::Expression> expect_repeats = {B, C, A, D, E};
      std::vector<ge::Expression> expect_strides = {C * A * D * E, A * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetType() == "Transpose") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (t_cnt == 0 || t_cnt == 1 || t_cnt == 2) {
        std::vector<int64_t> expect_axis = {1, 2, 0, 3, 4};
        std::vector<ge::Expression> expect_repeats = {B, C, A, D, E};
        std::vector<ge::Expression> expect_strides = {C * A * D * E, A * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      }
      t_cnt++;
    }
  }
  ASSERT_EQ(t_cnt, 1);
  // 取消走torch流程
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
}

// 后端不支持一个ascgraph多个transpose，暂时不跑此用例
// TEST_F(AscBackendPostProcessorTest, Torch_TransposeAscGraph_OK) {
//   ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
//   EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

//   auto addn1 = compute_graph->FindNode("addn1");
//   ASSERT_NE(addn1, nullptr);
//   auto op_desc1 = addn1->GetOpDescBarePtr();
//   ASSERT_NE(op_desc1, nullptr);
//   auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
//   ASSERT_NE(attr1, nullptr);
//   std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
//                                                         {"const2", DT_FLOAT16},
//                                                         {"shape1", DT_FLOAT16},
//                                                         {"netoutput", DT_FLOAT16}};
//   for (auto name : names) {
//     auto node = compute_graph->FindNode(name.first);
//     ASSERT_NE(node, nullptr);
//     auto op_desc = node->GetOpDescBarePtr();
//     ASSERT_NE(op_desc, nullptr);
//     op_desc->SetType("NotAscBc");
//     auto attr = GetOrCreateAutoFuseAttrs(op_desc);
//     ASSERT_NE(attr, nullptr);
//   }

//   ge::AscGraph add_graph1("Transpose");
//   attr1->SetAscGraph(TransposeAscGraphWithDataTensorOK(add_graph1));

//   // 走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
//   AscBackendPostProcessor post_processor;
//   EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
//   auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
//   ASSERT_NE(shape_env_attr, nullptr);
//   EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

//   // 校验结果
//   auto ONE = Symbol(1);
//   auto ZERO = Symbol(0);
//   const Expression A = add_graph1.CreateSizeVar("A");
//   const Expression B = add_graph1.CreateSizeVar("B");
//   const Expression C = add_graph1.CreateSizeVar("C");
//   const Expression D = add_graph1.CreateSizeVar("D");
//   const Expression E = add_graph1.CreateSizeVar("E");
//   int64_t t_cnt = 0;
//   for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
//     if (node->GetType() == "Output") {
//       continue;
//     }
//     printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
//     if (node->GetName() == "x1Local_2" || node->GetName() == "x3Local_2" || node->GetName() == "x2Local_2") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//       std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x_store1" || node->GetName() == "x_store3") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//       std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x_store2") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//       std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetType() == "Transpose") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       if (t_cnt == 0 || t_cnt == 1 || t_cnt == 2) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//       } else if (t_cnt == 3) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//         std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//       }
//       t_cnt++;
//     }
//   }
//   ASSERT_EQ(t_cnt, 2);
//   // 取消走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
// }

// // TODO非Torch流程用例需要再调试
// TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWithBroadcastAscGraph_OK) {
//   ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
//   EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

//   auto addn1 = compute_graph->FindNode("addn1");
//   ASSERT_NE(addn1, nullptr);
//   auto op_desc1 = addn1->GetOpDescBarePtr();
//   ASSERT_NE(op_desc1, nullptr);
//   auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
//   ASSERT_NE(attr1, nullptr);
//   std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
//                                                         {"const2", DT_FLOAT16},
//                                                         {"shape1", DT_FLOAT16},
//                                                         {"netoutput", DT_FLOAT16}};
//   for (auto name : names) {
//     auto node = compute_graph->FindNode(name.first);
//     ASSERT_NE(node, nullptr);
//     auto op_desc = node->GetOpDescBarePtr();
//     ASSERT_NE(op_desc, nullptr);
//     op_desc->SetType("NotAscBc");
//     auto attr = GetOrCreateAutoFuseAttrs(op_desc);
//     ASSERT_NE(attr, nullptr);
//   }

//   ge::AscGraph add_graph1("TransposeAscGraphWithBroadcast");
//   attr1->SetAscGraph(TransposeAscGraphWithBroadcastAscGraph(add_graph1));

//   AscBackendPostProcessor post_processor;
//   EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
//   auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
//   ASSERT_NE(shape_env_attr, nullptr);
//   EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

//   // 校验结果
//   auto ONE = Symbol(1);
//   auto ZERO = Symbol(0);
//   const Expression A = add_graph1.CreateSizeVar("A");
//   const Expression B = add_graph1.CreateSizeVar("B");
//   const Expression C = add_graph1.CreateSizeVar("C");
//   const Expression D = add_graph1.CreateSizeVar("D");
//   const Expression E = add_graph1.CreateSizeVar("E");
//   size_t bro_cnt = 0U;
//   for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
//     if (node->GetType() == "Output") {
//       continue;
//     }
//     printf("TransposeAscGraphWithBroadcast node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
//     if (node->GetName() == "x1Local_2" || node->GetName() == "x3Local_2" || node->GetName() == "x2Local_2") {
//       // torch不需要改load和store
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//       std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x_store2") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//       std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetType() == "Broadcast") {
//       GeTensorDescPtr output_tensor_desc;
//       ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//       auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//       ASSERT_NE(attr, nullptr);
//       if (bro_cnt == 0) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, ONE, C, D, E};
//         std::vector<ge::Expression> expect_strides = {C * D * E, ZERO, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//       } else if (bro_cnt == 1) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//       }
//       bro_cnt++;
//     }
//   }
//   EXPECT_EQ(bro_cnt, 5U);
// }

// 后端不支持一个ascgraph多个transpose，暂时不跑此用例
// TEST_F(AscBackendPostProcessorTest, Adaption_TransposeAscGraphWithNeedDelTransposeNode_OK) {
//   ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
//   EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

//   auto addn1 = compute_graph->FindNode("addn1");
//   ASSERT_NE(addn1, nullptr);
//   auto op_desc1 = addn1->GetOpDescBarePtr();
//   ASSERT_NE(op_desc1, nullptr);
//   auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
//   ASSERT_NE(attr1, nullptr);
//   std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
//                                                         {"const2", DT_FLOAT16},
//                                                         {"shape1", DT_FLOAT16},
//                                                         {"netoutput", DT_FLOAT16}};
//   for (auto name : names) {
//     auto node = compute_graph->FindNode(name.first);
//     ASSERT_NE(node, nullptr);
//     auto op_desc = node->GetOpDescBarePtr();
//     ASSERT_NE(op_desc, nullptr);
//     op_desc->SetType("NotAscBc");
//     auto attr = GetOrCreateAutoFuseAttrs(op_desc);
//     ASSERT_NE(attr, nullptr);
//   }

//   ge::AscGraph add_graph1("add");
//   attr1->SetAscGraph(TransposeAscGraphWithNeedDelTransposeNode(add_graph1));

//   for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
//     if (node->GetType() == "Abs") {
//       const auto &op_desc = node->GetOpDesc();
//       ASSERT_NE(op_desc, nullptr);
//       op_desc->SetType("Transpose");
//     }
//   }
//   // 走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
//   AscBackendPostProcessor post_processor;
//   EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
//   auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
//   ASSERT_NE(shape_env_attr, nullptr);
//   EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

//   // 校验结果
//   auto ONE = Symbol(1);
//   auto ZERO = Symbol(0);
//   const Expression A = add_graph1.CreateSizeVar("A");
//   const Expression B = add_graph1.CreateSizeVar("B");
//   const Expression C = add_graph1.CreateSizeVar("C");
//   const Expression D = add_graph1.CreateSizeVar("D");
//   const Expression E = add_graph1.CreateSizeVar("E");
//   int64_t t_cnt = 0;
//   int64_t current_topo_id = 0;

//   for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
//     if (node->GetType() == "Output") {
//       continue;
//     }
//     const auto node_opdesc = node->GetOpDesc();
//     ASSERT_EQ(node_opdesc->GetId(), current_topo_id++);
//     GeTensorDescPtr output_tensor_desc;
//     ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//     auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//     ASSERT_NE(attr, nullptr);
//     printf("Adaption_TransposeAscGraph_OK node %s %s axis:%s repeats:%s strides:%s, topo id:%ld\n.",
//         node->GetName().c_str(), node->GetType().c_str(),
//         AutofuseUtils::VectorToStr(attr->axis).c_str(),
//         AutofuseUtils::VectorToStr(attr->repeats).c_str(),
//         AutofuseUtils::VectorToStr(attr->strides).c_str(),
//         node_opdesc->GetId());
//     if (node->GetName() == "x_store1") {
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//       std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x_store2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//       std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x_store3") {
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, ONE};
//       std::vector<ge::Expression> expect_strides = {B * C * D, C * D, D, ONE, ZERO};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x1Local_2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//       std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x2Local_2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//       std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "x3Local_2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, ONE};
//       std::vector<ge::Expression> expect_strides = {C * B * D, B * D, D, ONE, ZERO};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetName() == "mul2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//       std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//     } else if (node->GetType() == "Transpose") {
//       if (t_cnt == 0) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//         std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_FLOAT);
//       } else if (t_cnt == 1) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_FLOAT);
//       } else if (t_cnt == 2) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//         std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_FLOAT);
//       } else if (t_cnt == 3) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_FLOAT);
//       } else if (t_cnt == 4) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//         std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_FLOAT);
//       } else if (t_cnt == 5) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//         std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_FLOAT);
//       }
//       t_cnt++;
//     }
//   }
//   ASSERT_EQ(t_cnt, 4);
//   // 在用例退出前取消走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
// }

// 后端不支持一个ascgraph多个transpose，暂时不跑此用例
// TEST_F(AscBackendPostProcessorTest, Adaption_TorchTransposeWithDiffAxisSize_OK) {
//   ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
//   EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

//   auto addn1 = compute_graph->FindNode("addn1");
//   ASSERT_NE(addn1, nullptr);
//   auto op_desc1 = addn1->GetOpDescBarePtr();
//   ASSERT_NE(op_desc1, nullptr);
//   auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
//   ASSERT_NE(attr1, nullptr);
//   std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
//                                                         {"const2", DT_FLOAT16},
//                                                         {"shape1", DT_FLOAT16},
//                                                         {"netoutput", DT_FLOAT16}};
//   for (auto name : names) {
//     auto node = compute_graph->FindNode(name.first);
//     ASSERT_NE(node, nullptr);
//     auto op_desc = node->GetOpDescBarePtr();
//     ASSERT_NE(op_desc, nullptr);
//     op_desc->SetType("NotAscBc");
//     auto attr = GetOrCreateAutoFuseAttrs(op_desc);
//     ASSERT_NE(attr, nullptr);
//   }

//   ge::AscGraph add_graph1("add");
//   attr1->SetAscGraph(TransposeAscGraphWithDiffAxisSize(add_graph1));

//   // 走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kTorch;
//   AscBackendPostProcessor post_processor;
//   EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
//   auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
//   ASSERT_NE(shape_env_attr, nullptr);
//   EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

//   // 校验结果
//   auto ONE = Symbol(1);
//   auto ZERO = Symbol(0);
//   const Expression A = add_graph1.CreateSizeVar("A");
//   const Expression B = add_graph1.CreateSizeVar("B");
//   const Expression C = add_graph1.CreateSizeVar("C");
//   const Expression D = add_graph1.CreateSizeVar("D");
//   const Expression E = add_graph1.CreateSizeVar("E");
//   int64_t t_cnt = 0;
//   int64_t current_topo_id = 0;

//   for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
//     if (node->GetType() == "Output") {
//       continue;
//     }
//     const auto node_opdesc = node->GetOpDesc();
//     ASSERT_EQ(node_opdesc->GetId(), current_topo_id++);
//     GeTensorDescPtr output_tensor_desc;
//     ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
//     auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
//     ASSERT_NE(attr, nullptr);
//     printf("Adaption_TransposeAscGraph_OK node %s %s axis:%s repeats:%s strides:%s, topo id:%ld\n.",
//         node->GetName().c_str(), node->GetType().c_str(),
//         AutofuseUtils::VectorToStr(attr->axis).c_str(),
//         AutofuseUtils::VectorToStr(attr->repeats).c_str(),
//         AutofuseUtils::VectorToStr(attr->strides).c_str(),
//         node_opdesc->GetId());
//     if (node->GetName() == "x_store1") {
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//       std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetName() == "x_store2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//       std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetName() == "x_store3") {
//       std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, B, C, D, ONE};
//       std::vector<ge::Expression> expect_strides = {B * C * D, C * D, D, ONE, ZERO};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetName() == "x1Local_2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//       std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetName() == "x2Local_2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//       std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetName() == "x3Local_2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, ONE};
//       std::vector<ge::Expression> expect_strides = {C * B * D, B * D, D, ONE, ZERO};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetName() == "mul2") {
//       std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//       std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//       std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//       ASSERT_EQ(attr->axis, expect_axis);
//       ASSERT_EQ(attr->repeats, expect_repeats);
//       ASSERT_EQ(attr->strides, expect_strides);
//       ASSERT_EQ(attr->dtype, DT_INT8);
//     } else if (node->GetType() == "Transpose") {
//       if (t_cnt == 0) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//         std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_INT8);
//       } else if (t_cnt == 1) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_INT8);
//       } else if (t_cnt == 2) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//         std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_INT8);
//       } else if (t_cnt == 3) {
//         std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//         std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_INT8);
//       } else if (t_cnt == 4) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 3, 4};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, D, E};
//         std::vector<ge::Expression> expect_strides = {C * B * D * E, B * D * E, D * E, E, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_INT8);
//       } else if (t_cnt == 5) {
//         std::vector<int64_t> expect_axis = {0, 2, 1, 4, 3};
//         std::vector<ge::Expression> expect_repeats = {A, C, B, E, D};
//         std::vector<ge::Expression> expect_strides = {C * B * E * D, B * E * D, E * D, D, ONE};
//         ASSERT_EQ(attr->axis, expect_axis);
//         ASSERT_EQ(attr->repeats, expect_repeats);
//         ASSERT_EQ(attr->strides, expect_strides);
//         ASSERT_EQ(attr->dtype, DT_INT8);
//       }
//       t_cnt++;
//     }
//   }
//   ASSERT_EQ(t_cnt, 4);

//   // 在用例退出前取消走torch流程
//   AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
// }

TEST_F(AscBackendPostProcessorTest, Adaption_NoEmptyRepeats_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithNoEmptyRepeats(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("Adaption_EmptyRepeats_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("Adaption_EmptyRepeats_OK To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    if (node->GetType() == "Output") {
      continue;
    }
    printf("Adaption_EmptyRepeats_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetName() == "add_4") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    } else if (node->GetName() == "abs_4") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("Adaption_EmptyRepeats_OK To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
}

TEST_F(AscBackendPostProcessorTest, Adaption_DiffRepeats_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithDiffRepeats(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
}

TEST_F(AscBackendPostProcessorTest, Adaption_DiffRepeatsMutilReference_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithDiffRepeatsMutilReference(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Cast") {
      ASSERT_EQ(attr->dtype, cast_cnt == 0 ? DT_FLOAT : DT_FLOAT16);
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
    if (node->GetType() != "Data" && node->GetType() != "Load" && node->GetType() != "Store"){
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }}
  ASSERT_EQ(broadcast_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, Adaption_ScalarToStore_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarToStore(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Scalar") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      continue;
    }
    if (node->GetType() == "Cast") {
      cast_cnt++;
      continue;
    }
  }
  ASSERT_EQ(broadcast_cnt, 5);
  ASSERT_EQ(cast_cnt, 0);
}

TEST_F(AscBackendPostProcessorTest, Adaption_DiffRepeatsWithScalar_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithDiffRepeatsWithScalar(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT); // dtype类型变换是因为现在scalar也插了broadcast后不做broadcast后移，
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Scalar") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      continue;
    }
    if (node->GetType() == "Cast") {
      if (cast_cnt == 0) {
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      }
      if (cast_cnt == 1) {
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      }
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
    if (node->GetType() != "Data" && node->GetType() != "Load" && node->GetType() != "Store"){
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }}
  ASSERT_EQ(broadcast_cnt, 8);
  ASSERT_EQ(cast_cnt, 2);
}

TEST_F(AscBackendPostProcessorTest, Adaption_ScalarToAdd0_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarToAdd0(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output" || node->GetType() == "Data" ) {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Scalar") {
      const auto &op_desc = node->GetOpDesc();
      auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      EXPECT_EQ(node_attr->sched.axis, expect_axis);
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
    }
    if (node->GetType() == "Add") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      // 循环外提把broadcast移到了最下面
      std::vector<ge::Expression> expect_repeats = { A, B, C, D, E };
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      continue;
    }
    if (node->GetType() == "Cast") {
      if (cast_cnt == 0) {
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      }
      if (cast_cnt == 1) {
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      }
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(broadcast_cnt, 8);
  ASSERT_EQ(cast_cnt, 2);
}

TEST_F(AscBackendPostProcessorTest, Adaption_ScalarToAdd0WithoutCheckoutType_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarToAdd1Int8(add_graph1));
  EXPECT_EQ(asc_adapt::FallbackScalarToBroadcastWithoutCheckType(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output" || node->GetType() == "Data" ) {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      if (broadcast_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        // 循环外提把broadcast移到了最下面
        std::vector<ge::Expression> expect_repeats = {ONE, ONE, ONE, ONE, E};
        std::vector<ge::Expression> expect_strides = {ZERO, ZERO, ZERO, ZERO, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      } else if (broadcast_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        // 循环外提把broadcast移到了最下面
        std::vector<ge::Expression> expect_repeats = {ONE, ONE, ONE, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, ZERO, ZERO, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      } else if (broadcast_cnt == 2) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        // 循环外提把broadcast移到了最下面
        std::vector<ge::Expression> expect_repeats = {ONE, ONE, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO,ZERO, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      } else if (broadcast_cnt == 3) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        // 循环外提把broadcast移到了最下面
        std::vector<ge::Expression> expect_repeats = {ONE, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      } else if (broadcast_cnt == 4) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        // 循环外提把broadcast移到了最下面
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
      } 
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_INT8);
      broadcast_cnt++;
    }
    if (node->GetType() == "Scalar") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_INT8);
    }
    if (node->GetType() == "Add") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      continue;
    }
    if (node->GetType() == "Cast") {
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(broadcast_cnt, 5);
  ASSERT_EQ(cast_cnt, 0);
}

TEST_F(AscBackendPostProcessorTest, Adaption_ScalarToAdd1_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarToAdd1(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output" || node->GetType() == "Data" ) {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT16);
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Scalar") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
    }
    if (node->GetType() == "Add") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      // 循环外提把broadcast移到了最下面
      std::vector<ge::Expression> expect_repeats = {ONE, ONE, ONE, ONE, ONE};
      std::vector<ge::Expression> expect_strides = {ZERO, ZERO, ZERO, ZERO, ZERO};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      continue;
    }
    if (node->GetType() == "Cast") {
      if (cast_cnt == 0) {
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      }
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(broadcast_cnt, 5);
  ASSERT_EQ(cast_cnt, 1);
}

TEST_F(AscBackendPostProcessorTest, Adaption_ScalarToAbs_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarToAbs(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output" || node->GetType() == "Data" ) {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Scalar") {
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
    }
    if (node->GetType() == "Abs") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      // 循环外提把broadcast移到了最下面
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      continue;
    }
    if (node->GetType() == "Cast") {
      if (cast_cnt == 0) {
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      }
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(broadcast_cnt, 5);
  ASSERT_EQ(cast_cnt, 1);
}

TEST_F(AscBackendPostProcessorTest, Adaption_DiffRepeatsMutilReference2_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("load_mul_reference2");
  attr1->SetAscGraph(CreatAscGraphWithDiffRepeatsMutilReference2(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to fallback load, current node(%s), type:%s, repeats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  // 序列化反序列化验证AscNode创造的正确性
  std::string output;
  EXPECT_EQ(AscGraphUtils::SerializeToReadable(*(attr1->GetAscGraph()), output), SUCCESS);
  printf("graph_aftre_serialize :%s\n", output.c_str());
  ge::AscGraph graph_aftre_serialize("graph_aftre_serialize");
  EXPECT_EQ(AscGraphUtils::DeserializeFromReadable(output, graph_aftre_serialize), SUCCESS);

  size_t broadcast_cnt = 0;
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Output") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Cast") {
      ASSERT_EQ(attr->dtype, cast_cnt == 0 ? DT_FLOAT : DT_FLOAT16);
      cast_cnt++;
      continue;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
    if (node->GetType() != "Data" && node->GetType() != "Load" && node->GetType() != "Store"){
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }}
  ASSERT_EQ(broadcast_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, ConcatDontImproveprecision) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatConcatAscGraphForNoImprovePrecision(add_graph1), loop::FuseType::kConcat);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("ConcatDontImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    EXPECT_NE(node->GetType(), "Cast");
  }
}

TEST_F(AscBackendPostProcessorTest, GatherhData1Data2InvalidAxis) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("CreatGatherAscGraphForData1Data2InvalidAxis");
  attr1->SetAscGraph(CreatGatherAscGraphForData1Data2InvalidAxis(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  int gather_cnt = 0;
  int data_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    std::vector<int64_t> sched_axis = {0, 1};
    if (node->GetType() == "Gather") {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {0, 1};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 2);
        EXPECT_EQ(tensor_attr->strides.size(), 2);
        int64_t gather_axis_index = std::numeric_limits<int64_t>::max();
        asc_adapt::GetGatherAxis(node, gather_axis_index);
        EXPECT_EQ(gather_axis_index, 1);
        gather_cnt++;
    } else if (node->GetType() == "Data") {
      if (data_cnt == 0) {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {0, 1};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 2);
        EXPECT_EQ(tensor_attr->strides.size(), 2);
      } else if (data_cnt == 1) {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {1};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 1);
        EXPECT_EQ(tensor_attr->strides.size(), 1);
      }
      data_cnt++;
    }
    cnt++;
  }
  EXPECT_NE(cnt, 2);
  EXPECT_EQ(gather_cnt, 1);
}

TEST_F(AscBackendPostProcessorTest, GatherhDataAllInvalidAxis) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("CreatGatherAscGraphForDataAllInvalidAxis");
  attr1->SetAscGraph(CreatGatherAscGraphForDataAllInvalidAxis(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  int gather_cnt = 0;
  int data_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    std::vector<int64_t> sched_axis = {0};
    if (node->GetType() == "Gather") {
      gather_cnt++;
      int64_t gather_axis_index = std::numeric_limits<int64_t>::max();
      asc_adapt::GetGatherAxis(node, gather_axis_index);
      EXPECT_EQ(gather_axis_index, 0);
    } else if (node->GetType() == "Data") {
      if (data_cnt == 0) {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {0};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 1);
        EXPECT_EQ(tensor_attr->strides.size(), 1);
      } else if (data_cnt == 1) {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {0};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 1);
        EXPECT_EQ(tensor_attr->strides.size(), 1);
      }
      data_cnt++;
    }
    cnt++;
  }
  EXPECT_NE(cnt, 2);
  EXPECT_EQ(gather_cnt, 1);
}

TEST_F(AscBackendPostProcessorTest, GatherhData2AllInvalidAxis) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("CreatGatherAscGraphForData2AllInvalidAxis");
  attr1->SetAscGraph(CreatGatherAscGraphForData2AllInvalidAxis(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  int gather_cnt = 0;
  int data_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    std::vector<int64_t> sched_axis = {0, 1, 2};
    if (node->GetType() == "Gather") {
      gather_cnt++;
      int64_t gather_axis_index = std::numeric_limits<int64_t>::max();
      asc_adapt::GetGatherAxis(node, gather_axis_index);
      EXPECT_EQ(gather_axis_index, 2);
    } else if (node->GetType() == "Data") {
      if (data_cnt == 0) {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {0, 1, 2};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 3);
        EXPECT_EQ(tensor_attr->strides.size(), 3);
      } else if (data_cnt == 1) {
        AscTensorAttr *tensor_attr = nullptr;
        asc_adapt::GetOutputTensorAttr(node, tensor_attr);
        const auto &op_desc = node->GetOpDesc();
        auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(node_attr->sched.axis, sched_axis);
        std::vector<int64_t> axis = {2};
        EXPECT_EQ(tensor_attr->axis, axis);
        EXPECT_EQ(tensor_attr->repeats.size(), 1);
        EXPECT_EQ(tensor_attr->strides.size(), 1);
      }
      data_cnt++;
    }
    cnt++;
  }
  EXPECT_NE(cnt, 2);
  EXPECT_EQ(gather_cnt, 1);
}

TEST_F(AscBackendPostProcessorTest, GatherhCse) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("CreatGatherAscGraphForCse");
  attr1->SetAscGraph(CreatGatherAscGraphForCse(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  int gather_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
  printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
  if (node->GetType() == "Gather") {
    gather_cnt++;
  }
  cnt++;
  }
  EXPECT_NE(cnt, 2);
  EXPECT_EQ(gather_cnt, 1);
}

TEST_F(AscBackendPostProcessorTest, GatherImproveprecision) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
  printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
  cnt++;
  }
  EXPECT_NE(cnt, 2);
}

// 测试跨精度能正常升精度处理
TEST_F(AscBackendPostProcessorTest, GatherImproveprecision32_8) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision32_8(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Cast") {
      if (cnt == 0) {
        const auto &op_desc = node->GetOpDesc();
        auto output_tensor_desc = op_desc->MutableOutputDesc(0);
        ASSERT_NE(output_tensor_desc, nullptr);
        auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
        ASSERT_NE(tensor_attr, nullptr);
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      } else if (cnt == 1) {
        const auto &op_desc = node->GetOpDesc();
        auto output_tensor_desc = op_desc->MutableOutputDesc(0);
        ASSERT_NE(output_tensor_desc, nullptr);
        auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
        ASSERT_NE(tensor_attr, nullptr);
        EXPECT_EQ(tensor_attr->dtype, DT_INT8);
      }
      cnt++;
    }
  }
  EXPECT_EQ(cnt, 2);
}

// 测试跨精度多引用能正常升精度处理
TEST_F(AscBackendPostProcessorTest, GatherImproveprecision32_8_MulReference) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision32_8_MulReference(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    printf("GatherImproveprecision node %s %s %s\n.", node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") {
      if (cnt == 0) {
        ASSERT_NE(tensor_attr, nullptr);
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      } else if (cnt == 1) {
        const auto &op_desc = node->GetOpDesc();
        auto output_tensor_desc = op_desc->MutableOutputDesc(0);
        ASSERT_NE(output_tensor_desc, nullptr);
        auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
        ASSERT_NE(tensor_attr, nullptr);
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      } else if (cnt == 2) {
        const auto &op_desc = node->GetOpDesc();
        auto output_tensor_desc = op_desc->MutableOutputDesc(0);
        ASSERT_NE(output_tensor_desc, nullptr);
        auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
        ASSERT_NE(tensor_attr, nullptr);
        EXPECT_EQ(tensor_attr->dtype, DT_INT8);
      }
      cnt++;
    }
  }
  EXPECT_EQ(cnt, 3);
}

// 测试跨精度多引用能正常升精度处理
TEST_F(AscBackendPostProcessorTest, GatherImproveprecision8_32_MulReference) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision8_32_MulReference(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    ASSERT_NE(tensor_attr, nullptr);
    printf("GatherImproveprecision node %s %s %s\n.", node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") {
      if (cnt == 0) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      } else if (cnt == 1) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT);
      } else if (cnt == 2) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      }
      cnt++;
    }
  }
  EXPECT_EQ(cnt, 3);
}

// 测试跨精度能正常升精度处理
TEST_F(AscBackendPostProcessorTest, GatherImproveprecision8_32) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision8_32(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    ASSERT_NE(tensor_attr, nullptr);
    printf("GatherImproveprecision node %s %s %s\n.", node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") {
      if (cnt == 0) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      } else if (cnt == 1) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT);
      }
      cnt++;
    }
  }
  EXPECT_EQ(cnt, 2);
}

// 测试跨精度后接cast能正常升精度处理
TEST_F(AscBackendPostProcessorTest, GatherImproveprecision8_16_8) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision8_16_8(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    ASSERT_NE(tensor_attr, nullptr);
    printf("GatherImproveprecision node %s %s %s\n.", node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") {
      if (cnt == 0) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      } else if (cnt == 1) {
        EXPECT_EQ(tensor_attr->dtype, DT_INT8);
      }
      cnt++;
    }
  }
  EXPECT_EQ(cnt, 2);
}

// 测试跨精度后接cast能正常升精度处理
TEST_F(AscBackendPostProcessorTest, GatherImproveprecision16_8_16) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                         {"const2", DT_FLOAT16},
                                                         {"shape1", DT_FLOAT16},
                                                         {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForImprovePrecision16_8_16(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    ASSERT_NE(tensor_attr, nullptr);
    printf("GatherImproveprecision node %s %s %s\n.", node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    if (node->GetType() == "Cast") {
      if (cnt == 0) {
        EXPECT_EQ(tensor_attr->dtype, DT_INT8);
      } else if (cnt == 1) {
        EXPECT_EQ(tensor_attr->dtype, DT_FLOAT16);
      }
      cnt++;
    }
  }
  EXPECT_EQ(cnt, 2);
}

TEST_F(AscBackendPostProcessorTest, GatherInsertCastImproveprecision) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
  {"const2", DT_FLOAT16},
  {"shape1", DT_FLOAT16},
  {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
  auto node = compute_graph->FindNode(name.first);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDescBarePtr();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("NotAscBc");
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph gather_graph1("gather");
  attr1->SetAscGraph(CreatGatherAscGraphForInsertCastImprovePrecision(gather_graph1), loop::FuseType::kGather);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  int cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
  printf("GatherImproveprecision node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
  cnt++;
  }
  EXPECT_NE(cnt, 2);
}

TEST_F(AscBackendPostProcessorTest, CompleteAttrWithGraphInvalidAxisNodeValidAxis) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithGraphInvalidAxisNodeValidAxis(add_graph1));
  const auto graph_attr = AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetAttrsGroup<AscGraphAttr>();
  auto size = graph_attr->axis.size();

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  EXPECT_EQ(graph_attr->axis.size(), size);
  printf("FallBackWithGraphInvalidAxis node size %lu\n.", size);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("FallBackWithGraphInvalidAxis node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    EXPECT_EQ(node_attr->sched.axis.size(), size);
    ASSERT_NE(tensor_attr, nullptr);
    if (node->GetType() != "Output" && node->GetType() != "Scalar") {
      printf("node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
      EXPECT_EQ(tensor_attr->axis[0], 0);
      EXPECT_EQ(tensor_attr->axis[1], 1);
      EXPECT_EQ(tensor_attr->axis[2], 2);
      EXPECT_EQ(tensor_attr->repeats.size(), size);
      EXPECT_EQ(tensor_attr->strides.size(), size);
    }
  }
}

// 测试scalar节点所有轴都是无效轴，会保留一根轴
TEST_F(AscBackendPostProcessorTest, CompleteAttrWithScalarAndAllAxisInvalid) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarAndAllAxisInvalid(add_graph1));
  const auto graph_attr = AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetAttrsGroup<AscGraphAttr>();
  auto size = graph_attr->axis.size() - 4U;

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  EXPECT_EQ(graph_attr->axis.size(), size);
  printf("CreatAscGraphWithScalarAndAllAxisInvalid node size %lu\n.", size);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("FallBackWithGraphInvalidAxis node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    EXPECT_EQ(node_attr->sched.axis.size(), size);
    EXPECT_EQ(node_attr->sched.axis[0], 0);
    ASSERT_NE(tensor_attr, nullptr);
    if (node->GetType() != "Output") {
      printf("node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
      EXPECT_EQ(tensor_attr->axis.size(), size);
      EXPECT_EQ(tensor_attr->axis[0], 0);
      EXPECT_EQ(tensor_attr->repeats.size(), size);
      EXPECT_EQ(tensor_attr->strides.size(), size);
    }
  }
}

// 测试所有轴都是无效轴，会保留一根轴
TEST_F(AscBackendPostProcessorTest, CompleteAttrWithGraphAllAxisInvalid) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithGraphAllAxisInvalid(add_graph1));
  const auto graph_attr = AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetAttrsGroup<AscGraphAttr>();
  auto size = graph_attr->axis.size() - 4U;

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  EXPECT_EQ(graph_attr->axis.size(), size);
  printf("CreatAscGraphWithGraphAllAxisInvalid node size %lu\n.", size);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("FallBackWithGraphInvalidAxis node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    EXPECT_EQ(node_attr->sched.axis.size(), size);
    EXPECT_EQ(node_attr->sched.axis[0], 0);
    ASSERT_NE(tensor_attr, nullptr);
    if (node->GetType() != "Output" && node->GetType() != "Scalar") {
      printf("node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
      EXPECT_EQ(tensor_attr->axis.size(), size);
      EXPECT_EQ(tensor_attr->axis[0], 0);
      EXPECT_EQ(tensor_attr->repeats.size(), size);
      EXPECT_EQ(tensor_attr->strides.size(), size);
    }
  }
}

TEST_F(AscBackendPostProcessorTest, CompleteAttrWithGraphInvalidAxis) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithGraphInvalidAxis(add_graph1));
  const auto graph_attr = AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetAttrsGroup<AscGraphAttr>();
  auto size = graph_attr->axis.size();

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  EXPECT_EQ(graph_attr->axis.size(), size - 1U);
  printf("FallBackWithGraphInvalidAxis node size %lu\n.", size);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("FallBackWithGraphInvalidAxis node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    EXPECT_EQ(node_attr->sched.axis.size(), size - 1U);
    EXPECT_EQ(node_attr->sched.axis[0], 0);
    EXPECT_EQ(node_attr->sched.axis[1], 1);
    ASSERT_NE(tensor_attr, nullptr);
    if (node->GetType() != "Output" && node->GetType() != "Scalar") {
      printf("node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
      EXPECT_EQ(tensor_attr->axis.size(), size - 1U);
      EXPECT_EQ(tensor_attr->axis[0], 0);
      EXPECT_EQ(tensor_attr->axis[1], 1);
      EXPECT_EQ(tensor_attr->repeats.size(), size - 1U);
      EXPECT_EQ(tensor_attr->strides.size(), size - 1U);
    }
  }
}

// 测试内轴为无效轴
TEST_F(AscBackendPostProcessorTest, CompleteAttrWithGraphInvalidAxis2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithGraphInvalidAxis2(add_graph1));
  const auto graph_attr = AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetAttrsGroup<AscGraphAttr>();
  auto size = graph_attr->axis.size();

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  EXPECT_EQ(graph_attr->axis.size(), size - 1U);
  printf("FallBackWithGraphInvalidAxis node size %lu\n.", size);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("FallBackWithGraphInvalidAxis node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    ASSERT_NE(output_tensor_desc, nullptr);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    EXPECT_EQ(node_attr->sched.axis.size(), size - 1U);
    EXPECT_EQ(node_attr->sched.axis[0], 0);
    EXPECT_EQ(node_attr->sched.axis[1], 1);
    ASSERT_NE(tensor_attr, nullptr);
    if (node->GetType() != "Output" && node->GetType() != "Scalar") {
      printf("node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
      EXPECT_EQ(tensor_attr->axis.size(), size - 1U);
      EXPECT_EQ(tensor_attr->axis[0], 0);
      EXPECT_EQ(tensor_attr->axis[1], 1);
      EXPECT_EQ(tensor_attr->repeats.size(), size - 1U);
      EXPECT_EQ(tensor_attr->strides.size(), size - 1U);
    }
  }
}

TEST_F(AscBackendPostProcessorTest, FallBackWithAxisIdDiffFromIndex) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithAxisIdDiffFromIndex(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 2);
}

/**
 *      netoutput1
 *         |
 *       shape
 *        |
 *      add
 *     /    \.
 *   /       \.
 * const1   const2
 *******************
 *      netoutput1
 *         |
 *      AscBc_1
 *     /    \.
 *   /       \.
 * const1   const2
 */
// asc ir op 暂不支持cast节点插入提升精度，暂时不测试
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_NoCast) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
}

/**
 *      netoutput1
 *         |
 *       shape
 *        |
 *      add
 *     /    \.
 *   /       \.
 * const1   const2
 *******************
 *      netoutput1
 *         |
 *      AscBc_1
 *     /    \.
 *   /       \.
 * const1   const2
 */
// asc ir op 暂不支持cast节点插入提升精度，暂时不测试
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_HasCast) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddWtihCastInFrontStoreAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // attr->dtype = DT_FLOAT16;
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(PrecisionImprover::ImprovePrecisionToFp32(compute_graph), SUCCESS);
}

/**
 *      netoutput1
 *         |
 *       shape
 *        |
 *      add
 *     /    \.
 *   /       \.
 * const1   const2
 *******************
 *      netoutput1
 *         |
 *      AscBc_1
 *     /    \.
 *   /       \.
 * const1   const2
 */
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_HasCastAndStoreFp32) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddWtihNoCastInfrontStoreAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetName() == "x_out_5") {
      attr->dtype = DT_FLOAT;
    } else if (node->GetName() == "x2_3") {
      attr->dtype = DT_BF16;
    } else if (node->GetName() == "x2Local_4") {
      attr->dtype = DT_BF16;
    } else if (node->GetName() == "Local4Cast0") {
      attr->dtype = DT_FLOAT;
    } else {
      attr->dtype = DT_FLOAT16;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(PrecisionImprover::ImprovePrecisionToFp32(compute_graph), SUCCESS);
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
    if (node->GetType() != "Data" && node->GetType() != "Load" && node->GetType() != "Store" && node->GetType() != "Output"){
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }
    ASSERT_NE(node->GetName(), "add_Cast");
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(cast_cnt, 2);
}

TEST_F(AscBackendPostProcessorTest, IncreasePrecision_Int8ToFloat16) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddWtihInt8ToFloat16(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("Int8ToFloat16 node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    auto dtype = output_tensor_desc->GetDataType();
    if (node->GetType() == "Cast") {
      if (cast_cnt == 0) {
        EXPECT_EQ(dtype, DT_FLOAT16);
      } else if (cast_cnt == 1) {
        EXPECT_EQ(dtype, DT_INT8);
      } else if (cast_cnt == 2) {
        EXPECT_EQ(dtype, DT_FLOAT16);
      } else if (cast_cnt == 3) {
        EXPECT_EQ(dtype, DT_FLOAT);
      } else if (cast_cnt == 4) {
        EXPECT_EQ(dtype, DT_FLOAT);
      } else if (cast_cnt == 5) {
        EXPECT_EQ(dtype, DT_FLOAT16);
      } else if (cast_cnt == 6) {
        EXPECT_EQ(dtype, DT_UINT8);
      }
      cast_cnt++;
    } else if (node->GetType() != "Data" && node->GetType() != "Load" && node->GetType() != "Store" && node->GetType() != "Output") {
      EXPECT_EQ(dtype, DT_FLOAT);
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(cast_cnt, 7);
  ASSERT_EQ(asc_adapt::CheckCastDtype(DT_FLOAT16, DT_FLOAT) , true);
  ASSERT_EQ(asc_adapt::CheckCastDtype(DT_FLOAT, DT_BF16) , true);
  ASSERT_EQ(asc_adapt::CheckCastDtype(DT_INT4, DT_FLOAT) , false);
  ASSERT_EQ(asc_adapt::CheckCastDtype(DT_FLOAT, DT_INT8) , false);
  ASSERT_EQ(asc_adapt::CheckCastDtype(DT_INT8, DT_FLOAT16) , true);
  ASSERT_EQ(asc_adapt::CheckCastDtype(DT_FLOAT16, DT_INT4) , true);
}

/**
 *      netoutput1
 *         |
 *       shape
 *        |
 *      add
 *     /    \.
 *   /       \.
 * const1   const2
 *******************
 *      netoutput1
 *         |
 *      AscBc_1
 *     /    \.
 *   /       \.
 * const1   const2
 */
// asc ir op 暂不支持cast节点插入提升精度，暂时不测试
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_HasCastAndStoreFp16) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddWtihNoCastInfrontStoreAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetName() == "x_out_5") {
      attr->dtype = DT_FLOAT16;
    } else if (node->GetName() == "x2_3") {
      attr->dtype = DT_BF16;
    } else if (node->GetName() == "x2Local_4") {
      attr->dtype = DT_BF16;
    } else if (node->GetName() == "Local4Cast0") {
      attr->dtype = DT_FLOAT;
    } else {
      attr->dtype = DT_FLOAT16;
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(PrecisionImprover::ImprovePrecisionToFp32(compute_graph), SUCCESS);
  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    auto dtype = output_tensor_desc->GetDataType();
    if (node->GetType() == "Cast") {
      if (cast_cnt < 2) {
        ASSERT_EQ(dtype, DT_FLOAT);
      } else { // store前面的cast dtype和store的dtype应该保持一致
        ASSERT_EQ(dtype, DT_FLOAT16);
      }
      cast_cnt++;
    } else if (node->GetType() != "Data" && node->GetType() != "Load" && node->GetType() != "Store" && node->GetType() != "Output") {
      ASSERT_EQ(dtype, DT_FLOAT);
    }
    ASSERT_NE(node->GetName(), "add_Cast");
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(cast_cnt, 3);
}

/**
 *      netoutput1
 *         |
 *       shape
 *        |
 *      add
 *     /    \.
 *   /       \.
 * const1   const2
 *******************
 *      netoutput1
 *         |
 *      AscBc_1
 *     /    \.
 *   /       \.
 * const1   const2
 */
// 为float32的load后面的转fp16的cast会被删除
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_MidCastNotDelLoadCastDel) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(MidCastAndLoadCastAndStoreCastAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(PrecisionImprover::ImprovePrecisionToFp32(compute_graph), SUCCESS);

  size_t cast_cnt = 0;
  bool has_cast_mid = false;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
    if (node->GetName() == "x1_1") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "x2_3") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "x1Local_2") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "x2Local_4") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "add_Cast") {
      has_cast_mid = true;
      ASSERT_EQ(attr->dtype, DT_INT16);
    } else if (node->GetName() == "abs_4") {
      ASSERT_EQ(attr->dtype, DT_INT16);
    } else if (node->GetName() == "x_out_5") {
      ASSERT_EQ(attr->dtype, DT_INT16);
    } else {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(has_cast_mid, true);
  ASSERT_EQ(cast_cnt, 3 - 2);
}

// dtype为fp32的store同时前面cast为fp32，cast前面为fp16的节点，这个cast会被删除
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_MidCastDelLoadCastNotDelStoreCastDel) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(MidCastAndLoadCastAndStoreCastAscGraph2(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0;
  bool has_cast_mid = false;
  bool has_cast_before_store = false;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("AscAdapterTest_Ok node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
    if (node->GetName() == "x1_1") {
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    } else if (node->GetName() == "x2_3") {
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    } else if (node->GetName() == "x1Local_2") {
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    } else if (node->GetName() == "Local2Cast0") {
      ASSERT_EQ(attr->dtype, DT_INT16);
    } else if (node->GetName() == "x2Local_4") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "Local4Cast0") {
      ASSERT_EQ(attr->dtype, DT_INT16);
    } else if (node->GetName() == "add_4") {
      ASSERT_EQ(attr->dtype, DT_INT16);
    } else if (node->GetName() == "add_Cast") {
      has_cast_mid = true;
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "abs_4") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "abs_4_cast") {
      has_cast_before_store = true;
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else if (node->GetName() == "x_out_5") {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    } else {
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  ASSERT_EQ(has_cast_mid, true);
  ASSERT_EQ(has_cast_before_store, false);
  ASSERT_EQ(cast_cnt, 3);
}

// 多个连续cast在fp16和fp32之间互转测试,load后以及store之前的cast节点删除
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_ContinuousCastAfterLoadAndBeforeStoreAscGraph) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(ContinuousCastAfterLoadAndBeforeStoreAscGraph(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    EXPECT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    EXPECT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  EXPECT_EQ(cast_cnt, 0);
}

// 多个连续cast在fp16和fp32之间互转测试,load后以及store之前的cast节点删除, 其他dtype类型不删除
TEST_F(AscBackendPostProcessorTest, IncreasePrecision_ContinuousCastAfterLoadAndBeforeStoreAscGraph2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(ContinuousCastAfterLoadAndBeforeStoreAscGraph2(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("AscAdapterTest_Ok To find node to change precision, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    EXPECT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    EXPECT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      if (node->GetName() == "Local2Cast3") {
        EXPECT_EQ(attr->dtype, DT_INT16);
      } else if (node->GetName() == "abs_4_cast") {
        EXPECT_EQ(attr->dtype, DT_FLOAT);
      }
      cast_cnt++;
    }
  }
  EXPECT_EQ(cast_cnt, 2);
}

// 测试abs未配置黑名单后升精度
TEST_F(AscBackendPostProcessorTest, Not_ImprovePrecision_BlackList) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackList0");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionBlackList0(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  // cse会消除一个cast
  EXPECT_EQ(cast_cnt, 3U);
}

// 测试abs,配置黑名单后不升精度
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BlackList1) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackList0");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionBlackList0(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  // cse会消除一个cast
  EXPECT_EQ(cast_cnt, 0U);
}

// 配置Abs,Sqrt为黑名单,Rqrt不配置到黑名单，预期升精度
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BlackList2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackList1");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionBlackList1(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sqrt");
  EXPECT_EQ(AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.find("Abs") !=
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.end(), true);
  EXPECT_EQ(AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.find("Sqrt") !=
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.end(), true);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }

  EXPECT_EQ(cast_cnt, 3U);
}

// 配置Abs,Sqrt,Rqrt为黑名单，预期不升精度
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BlackList3) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackList1");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionBlackList1(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Rsqrt");
  EXPECT_EQ(AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.find("Abs") !=
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.end(), true);
  EXPECT_EQ(AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.find("Sqrt") !=
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.end(), true);
  EXPECT_EQ(AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.find("Rsqrt") !=
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.end(), true);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }

  EXPECT_EQ(cast_cnt, 0U);
}

// 配置Abs,Sqrt,Rqrt.为黑名单，预期不升精度
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BlackList4) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackList1");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionBlackList1(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Rsqrt");
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }

  EXPECT_EQ(cast_cnt, 0U);
}

// 配置Abs,Sqrt,Rqrt;为黑名单，预期不升精度
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BlackList5) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackList1");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionBlackList1(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Rsqrt");
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }

  EXPECT_EQ(cast_cnt, 0U);
}

// 配置Sum;为黑名单，Sum的类型后端不支持报错
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BlackList6) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionWhiteList");
  attr1->SetAscGraph(CreatAscGraphWithImprovePrecisionWhiteList(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Rsqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sum");
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_NE(post_processor.Do(compute_graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();
}

// 解决bug：dtype为bf16时，升精度后预期降精度回bf16，而不是fp16
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_BF16) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImproveBF16");
  attr1->SetAscGraph(CreatAscGraphWithImproveBF16(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    if (node->GetType() == "Cast") {
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (cast_cnt == 0) {
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else {
        ASSERT_EQ(attr->dtype, DT_BF16);
      }
      cast_cnt++;
    }
  }
  ASSERT_EQ(cast_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, ImprovePrecision_WithLoadToCastToStore) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithLoadToCastToStore");
  attr1->SetAscGraph(CreatAscGraphWithLoadToCastToStore(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  // cse不会消除dtype不同的节点，因此cast还是有3个
  EXPECT_EQ(cast_cnt, 2U);
}

TEST_F(AscBackendPostProcessorTest, ImprovePrecision_WithLoadToCastToStore2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithLoadToCastToStore2");
  attr1->SetAscGraph(CreatAscGraphWithLoadToCastToStore2(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
      cast_cnt++;
    }
  }
  EXPECT_EQ(cast_cnt, 1U);
}

TEST_F(AscBackendPostProcessorTest, ImprovePrecision_WithLoadToCastToStore3) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithLoadToCastToStore3");
  attr1->SetAscGraph(CreatAscGraphWithLoadToCastToStore3(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验结果
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (cast_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else if (cast_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      }
      cast_cnt++;
    }
  }
  EXPECT_EQ(cast_cnt, 2U);
}

// load直连store不应该升精度处理
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_WithLoadToStore) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithLoadToStore(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  ASSERT_EQ(cast_cnt, 0U);
}

// load直连broadcast直连store不应该升精度处理
TEST_F(AscBackendPostProcessorTest, ImprovePrecision_WithLoadToBroadcastToStore) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithLoadToBroadcstToStore(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  ASSERT_EQ(cast_cnt, 0U);
}

// load直连broadcast直连transpose直连store轴处理正确，先插transpose再插broadcast，transpose数量优化
TEST_F(AscBackendPostProcessorTest, Fallback_CreatAscGraphWithLoadToBroadcstAndTransposeToStore) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithLoadToBroadcstAndTransposeToStore");
  attr1->SetAscGraph(CreatAscGraphWithLoadToBroadcstAndTransposeToStore(add_graph1));
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("CreatAscGraphWithLoadToBroadcstAndTransposeToStore , current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
    if (node->GetType() == "Broadcast") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 4, 3};
      std::vector<ge::Expression> expect_repeats = {A, B, C, E, D};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, D, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    }else if (node->GetType() == "Transpose") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 4, 3};
      std::vector<ge::Expression> expect_repeats = {A, B, C, E, ONE};
      std::vector<ge::Expression> expect_strides =  {B * C * E, C * E, E, ONE, ZERO};
//      std::vector<ge::Expression> expect_repeats = {A, B, C, E, D}; // 后续支持多transpose后使用
//      std::vector<ge::Expression> expect_strides =  {B * C * E * D, C * E * D, E * D, D, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    }
  }
  ASSERT_EQ(cast_cnt, 0U);
}

// load直连broadcast直连transpose直连store轴处理正确，先插transpose再插broadcast
TEST_F(AscBackendPostProcessorTest, Fallback_CreatAscGraphWithLoadToBroadcstAndTransposeToStore2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithLoadToBroadcstAndTransposeToStore2");
  attr1->SetAscGraph(CreatAscGraphWithLoadToBroadcstAndTransposeToStore2(add_graph1));
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("CreatAscGraphWithLoadToBroadcstAndTransposeToStore2, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
    if (node->GetType() == "Broadcast") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * E * D, C * E * D, E * D, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    }else if (node->GetType() == "Transpose") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, ONE, E};
      std::vector<ge::Expression> expect_strides =  {B * C * E, C * E, E, ZERO, ONE};
//      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
//      std::vector<ge::Expression> expect_strides =  {B * C * D * E, C * D * E, D * E, E, ONE}; // 后续支持多transpose后使用
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    }
  }
  ASSERT_EQ(cast_cnt, 0U);
}

// load直连broadcast直连transpose直连store轴处理正确，先插transpose再插broadcast,broadcast后移
TEST_F(AscBackendPostProcessorTest, Fallback_CreatAscGraphWithLoadToBroadcstAndTransposeToStore3) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithLoadToBroadcstAndTransposeToStore3");
  attr1->SetAscGraph(CreatAscGraphWithLoadToBroadcstAndTransposeToStore3(add_graph1));
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    printf("CreatAscGraphWithLoadToBroadcstAndTransposeToStore3, current node(%s), type:%s, speats %s, dtype:%s in graph %s.\n",
      node->GetName().c_str(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(attr->repeats).c_str(),
      TypeUtils::DataTypeToSerialString(attr->dtype).c_str(), AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetName().c_str());
    if (node->GetType() == "Broadcast") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
      std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    }else if (node->GetType() == "Transpose") {
      std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
      std::vector<ge::Expression> expect_repeats = {ONE, B, C, D, E};
      std::vector<ge::Expression> expect_strides =  {ZERO, C * D * E, D * E, E, ONE};
      ASSERT_EQ(attr->axis, expect_axis);
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
      ASSERT_EQ(attr->dtype, DT_FLOAT16);
    }
  }
  ASSERT_EQ(cast_cnt, 0U);
}

// 非正式功能用例，待transpose方案确定后删除此用例，补充正式方案用例
TEST_F(AscBackendPostProcessorTest, Fallback_Transpose) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithTranspose(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);
}

// 反推插入的broadcast节点dtype和load保持一致
TEST_F(AscBackendPostProcessorTest, FallBackBroadcastDtype_DtypeNoFloatAscGraph) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(DtypeNoFloatAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
       ASSERT_EQ(attr->dtype, DT_INT16);
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CyclicExternalLiftPassMutilReference_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithDiffRepeatsMutilReference(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (broadcast_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {ONE, ONE, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, ZERO, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      } else if (broadcast_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {ONE, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      } else if (broadcast_cnt == 2) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT16);
      }
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Store") {
      NodePtr peer_node;
      ASSERT_EQ(asc_adapt::GetPeerOutNode(node, peer_node, 0), SUCCESS);
      ASSERT_EQ(peer_node->GetType(), "Broadcast");
      continue;
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CyclicExternalLiftPassWithScalar_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithDiffRepeatsWithScalar(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Store") {
      NodePtr peer_node;
      ASSERT_EQ(asc_adapt::GetPeerOutNode(node, peer_node, 0), SUCCESS);
      ASSERT_EQ(peer_node->GetType(), "Cast"); // scalar后面插了broadcast，就没走broadcast后移了
      continue;
    }
    if (node->GetType() == "Scalar") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      ASSERT_EQ(output_tensor_desc->GetDataType(), DT_FLOAT);
      continue;
    }
  }
  ASSERT_EQ(broadcast_cnt, 8);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CyclicExternalLiftPassMutilReference2_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("load_mul_reference2");
  attr1->SetAscGraph(CreatAscGraphWithDiffRepeatsMutilReference2(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
      continue;
    }
    if (node->GetType() == "Store") {
      NodePtr peer_node;
      ASSERT_EQ(asc_adapt::GetPeerOutNode(node, peer_node, 0), SUCCESS);
      ASSERT_EQ(peer_node->GetType(), "Broadcast");
      continue;
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CompleteAttrAndBroadcast_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithCompleteAttrAndBroadcast");
  attr1->SetAscGraph(CreatAscGraphWithCompleteAttrAndBroadcast(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = add_graph1.CreateSizeVar("A");
  const Expression B = add_graph1.CreateSizeVar("B");
  const Expression C = add_graph1.CreateSizeVar("C");
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    printf("Adaption_Torch_TransposeAscGraph_OK node %s %s\n.", node->GetName().c_str(), node->GetType().c_str());
    if (node->GetType() == "Broadcast") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      if (broadcast_cnt == 0) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {ONE, ONE, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, ZERO, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else if (broadcast_cnt == 1) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {ONE, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else if (broadcast_cnt == 2) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else if (broadcast_cnt == 3) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {ONE, B, ONE, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, D * E, ZERO, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else if (broadcast_cnt == 4) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {ONE, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {ZERO, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      } else if (broadcast_cnt == 5) {
        std::vector<int64_t> expect_axis = {0, 1, 2, 3, 4};
        std::vector<ge::Expression> expect_repeats = {A, B, C, D, E};
        std::vector<ge::Expression> expect_strides = {B * C * D * E, C * D * E, D * E, E, ONE};
        ASSERT_EQ(attr->axis, expect_axis);
        ASSERT_EQ(attr->repeats, expect_repeats);
        ASSERT_EQ(attr->strides, expect_strides);
        ASSERT_EQ(attr->dtype, DT_FLOAT);
      }
      broadcast_cnt++;
    }
  }
  ASSERT_EQ(broadcast_cnt, 6);
}

TEST_F(AscBackendPostProcessorTest, Adaption_CyclicExternalLiftPassMutilReference3_OK) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("load_mul_reference2");
  attr1->SetAscGraph(CreatAscGraphWithStoreMutilReference(add_graph1));
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    // ascir_op暂不支持Cast，认为设置type为cast
    if (node->GetName().find("Cast") != std::string::npos) {
      const auto &op_desc = node->GetOpDesc();
      ASSERT_NE(op_desc, nullptr);
      op_desc->SetType("Cast");
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
  }
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
      ASSERT_EQ(node->GetOutDataNodes().size(), 1);
      continue;
    }
    if (node->GetType() == "Store") {
      NodePtr peer_node;
      ASSERT_EQ(asc_adapt::GetPeerOutNode(node, peer_node, 0), SUCCESS);
      ASSERT_EQ(peer_node->GetType(), "Broadcast");
      continue;
    }
  }
  ASSERT_EQ(broadcast_cnt, 6);
}

TEST_F(AscBackendPostProcessorTest, Serialize_GetHashedExtraParamBuilder_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }
  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithEmptyRepeatsAfterLoad(add_graph1));

  auto attr3 = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(attr3, nullptr);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 校验属性设置成功，方法运行无报错
  auto *serialize_function = op_desc1->GetExtAttr<std::function<std::string()>>("_extra_param_builder");
  auto hash_str = op_desc1->GetExtAttr<std::string>("_hashed_extra_param_builder");
  ASSERT_NE(serialize_function, nullptr);
  auto serialize_str = (*serialize_function)();
  ASSERT_NE(serialize_str, "");
  ASSERT_NE(*hash_str, "");

  // 验证hash_str不包含symbol_to_value和value_to_symbol字段
  ASSERT_EQ(hash_str->find("symbol_to_value"), std::string::npos);
  ASSERT_EQ(hash_str->find("value_to_symbol"), std::string::npos);
}

TEST_F(AscBackendPostProcessorTest, Serialize_SameGraph_HashedExtraParamBuilder_Equal) {
  ComputeGraphPtr compute_graph1 = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph1->GetAllNodesSize(), 5);
  auto addn1 = compute_graph1->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph1->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("AscBackend");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
    ge::AscGraph add_graph(name.first.c_str());
    attr->SetAscGraph(BroadcastBackwardMulInputsAscGraph(add_graph));
  }
  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphWithEmptyRepeatsAfterLoad(add_graph1));
  auto attr3 = compute_graph1->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(attr3, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph1), SUCCESS);
  EXPECT_EQ(post_processor.Do(compute_graph1), SUCCESS);

  // 校验属性设置成功，方法运行无报错
  auto *serialize_function = op_desc1->GetExtAttr<std::function<std::string()>>("_extra_param_builder");
  auto hash_str = op_desc1->GetExtAttr<std::string>("_hashed_extra_param_builder");
  ASSERT_NE(serialize_function, nullptr);
  auto serialize_str = (*serialize_function)();
  ASSERT_NE(serialize_str, "");
  ASSERT_NE(*hash_str, "");

  // 结构完全相同的计算图2
  ComputeGraphPtr compute_graph2 = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph2->GetAllNodesSize(), 5);
  auto addn2 = compute_graph2->FindNode("addn1");
  ASSERT_NE(addn2, nullptr);
  auto op_desc2 = addn2->GetOpDescBarePtr();
  ASSERT_NE(op_desc2, nullptr);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr2, nullptr);
  for (auto name : names) {
    auto node = compute_graph2->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }
  ge::AscGraph add_graph2("add");
  attr2->SetAscGraph(CreatAddAscGraphWithEmptyRepeatsAfterLoad(add_graph2));
  auto attr4 = compute_graph2->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(attr4, nullptr);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph2), SUCCESS);
  EXPECT_EQ(post_processor.Do(compute_graph2), SUCCESS);

  auto *serialize_function2 = op_desc2->GetExtAttr<std::function<std::string()>>("_extra_param_builder");
  auto *hash_str2 = op_desc2->GetExtAttr<std::string>("_hashed_extra_param_builder");
  ASSERT_NE(serialize_function2, nullptr);
  auto serialize_str2 = (*serialize_function2)();
  ASSERT_NE(serialize_str2, "");
  ASSERT_NE(*hash_str2, "");

  // 校验相同图求得hash值完全相同
  ASSERT_EQ(*hash_str2, *hash_str);
}

// Broadcast后移连边正确
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_MulInputAscGraph_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(BroadcastBackwardMulInputsAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "abs_3") {
      // 校验Broadcast节点后移到了多输入节点前（Add）
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetName(), "Local2Cast0");
    }

    if (node->GetName() == "Abs2Cast0") {
      // 校验Broadcast节点后移到了计算节点后
      std::vector<NodePtr> after_bro_node;
      asc_adapt::GetPeerInNodes(node, after_bro_node, 0);
      ASSERT_EQ(after_bro_node[0]->GetType(), "Broadcast");
    }

    if (node->GetName() == "add_4") {
      // 校验Broadcast节点后移到了多输入节点前（Add）
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移后Broadcast节点dataype更新
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_UpdateBroadcastDataType_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");

  // 构造存在Cast节点场景，后移Broadcast后，更新Broadcast的属性
  attr1->SetAscGraph(BroadcastBackwardMulInputsAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
      // 校验后移到Cast节点后更新Broadcast节点data_type
      ASSERT_EQ(attr->dtype, DT_FLOAT);
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移后提前的计算节点repeats、strides刷新成功
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_UpdateComputeNodeOutputTensor_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(BroadcastBackwardMulInputsAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  std::vector<ge::Expression> expect_repeats = {ONE, ONE, ONE, D, E};
  std::vector<ge::Expression> expect_strides = {ZERO, ZERO, ZERO, E, ONE};
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);

    if (node->GetName() == "abs_3") {
      // 校验更新后移计算节点的OutputTensor
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }

    if (node->GetName() == "Abs2Cast0") {
      // 校验更新后移计算节点的OutputTensor
      ASSERT_EQ(attr->repeats, expect_repeats);
      ASSERT_EQ(attr->strides, expect_strides);
    }

    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移判断存在view类算子不再向后移
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_ReduceAscGraph_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("reduce");
  attr1->SetAscGraph(BroadcastBackwardReduceAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "reduce_reduce") {
      // 校验Broadcast节点没有后移到reduce后
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移判断多输出算子不再向后移
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_MulOutputAscGraph_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("mul_output");
  attr1->SetAscGraph(BroadcastBackwardMulOutputAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "calcRstd_calc_rstd") {
      // 校验Broadcast节点没有后移到FlashSoftmax后
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// 空Tensor后处理做了补属性和序列化
TEST_F(AscBackendPostProcessorTest, AscBackendNoKernel_HasAttrs_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  op_desc1->SetType(kAscBackendNoKernelType);
  ge::AscGraph add_graph1("empty_tensor_no_kernel");
  attr1->SetAscGraph(EmptyTensorAscGraph(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    // 获取数据类型属性,校验补了repeats、strides
    ASSERT_EQ(attr->repeats.size(), 5);
    ASSERT_EQ(attr->strides.size(), 5);

    // 校验补充了index
    if (node->GetType() == kDataType) {
      const auto attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      int64_t res_index;
      attr->ir_attr->GetAttrValue("index", res_index);
      ASSERT_EQ(res_index, 0);
    }
  }

  // 校验走了序列化
  auto *serialize_function = op_desc1->GetExtAttr<std::function<std::string()>>("_extra_param_builder");
  ASSERT_NE(serialize_function, nullptr);
  auto serialize_str = (*serialize_function)();
  ASSERT_NE(serialize_str, "");
}

// Broadcast后移判断多输入算子合分支Brc节点都相同继续后移
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_MulInputAscGraph_Ok2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatBrcBackwardAscGraphWithMulInputs(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 2);
}

// Broadcast后移判断多输入算子合分支Brc节点存在相同子集则子集继续后移
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移判断多输入算子合分支Brc节点存在相同子集则子集继续后移（Brc尾轴）
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs2(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 2);
}

// Broadcast后移判断多输入算子合分支Brc节点存在相同子集则子集继续后移 多个公共轴
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok3) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs3(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Cast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 5);
}

// Broadcast后移判断多输入算子合分支Brc节点存在相同子集但后续节点多引用
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok4) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs4(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Cast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 5);
}

// Broadcast后移判断多输入算子合分支Brc节点存在相同子集(Brc链中间的后移)
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok5) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs5(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移判断多输入算子合分支Brc节点存在相同子集 (后续节点多输入，且后移所在anchor_idx非默认0)
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok6) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs6(add_graph1));

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Cast");
    }
    if (node->GetName() == "add_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Cast");
      asc_adapt::GetPeerOutNode(node, pre_add_node, 1);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移判断图中中间节点的Brc（非load后）也支持后移
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok7) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs7(add_graph1));

  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression D = add_graph1.CreateSizeVar("D");
  const Expression E = add_graph1.CreateSizeVar("E");
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
      ASSERT_EQ(pre_add_node->GetName(), "broadcast_1");
    } else if (node->GetName() == "abs_5") {
      GeTensorDescPtr output_tensor_desc;
      ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
      // 获取数据类型属性
      auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      ASSERT_NE(attr, nullptr);
      std::vector<ge::Expression> expect_strides = {ZERO, D, ZERO, ONE, ZERO};
      ASSERT_EQ(attr->strides, expect_strides);
    }
  }
  ASSERT_EQ(broadcast_cnt, 3);
}

// Broadcast后移判断单输出多引用场景合分支都支持后移且回归到同一个非Store节点支持后移
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok8) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs8(add_graph1));
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_6") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      //     多引用不后移时 ASSERT_EQ(pre_add_node->GetType(), "Abs");
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 2);
}

// Broadcast后移判断单输出多引用场景合分支能回归到同一个非Store节点支持后移（包含反推）
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok9) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs9(add_graph1));
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);

  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_6") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      //      多引用不后移时 ASSERT_EQ(pre_add_node->GetType(), "Abs");
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 2);
}

// Broadcast后移判断单输出多引用场景合分支都不能回归到同一个非Store节点不支持后移
TEST_F(AscBackendPostProcessorTest, PartBroadcastBackward_AscGraph_Ok10) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatPartBrcBackwardAscGraphWithMulInputs10(add_graph1));
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  AutofuseUtils::DumpGraphToOnnx(*ge::AscGraphUtils::GetComputeGraph(add_graph1), "yzw", "before.onnx");
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);
  AutofuseUtils::DumpGraphToOnnx(*ge::AscGraphUtils::GetComputeGraph(add_graph1), "yzw", "after.onnx");
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "x_out_6") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Abs");
    }
  }
  ASSERT_EQ(broadcast_cnt, 2);
}

// Broadcast后移判断Scalar节点后的Brc在其后计算节点不支持Scalar时不进行后移
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_ScalarNotOK) {
  // 不打桩场景，Add节点不支持Scalar输入（Add实际支持但由于未注册所以查询结果为不支持）
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarAbsToAdd(add_graph1));
  EXPECT_EQ(asc_adapt::FallbackScalarToBroadcastWithoutCheckType(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);

  // 校验结果
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    // 获取数据类型属性
    auto attr = output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    ASSERT_NE(attr, nullptr);
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "add_4") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Abs");
      asc_adapt::GetPeerOutNode(node, pre_add_node, 1);
      ASSERT_EQ(pre_add_node->GetType(), "Load");
    }
  }
  ASSERT_EQ(broadcast_cnt, 5);
}

// Broadcast后移判断Scalar节点后的Brc在其后计算节点支持Scalar时也支持后移
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_ScalarOK) {
  // 添加自定义AscIrCodegen实现，使IsScalarInputSupported返回true
  class TestAscIrCodegenStub : public ge::ascir::AscIrCodegen {
   public:
    bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
      return true;
    }
  };

  // 获取当前平台信息
  std::string platform_name;
  ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_name);

  // 保存原始注册表状态
  auto &registry = ge::ascir::AscirRegistry::GetInstance();
  auto original_registry = registry.GetAll();

  // 创建AscIrImpl对象，设置codegen创建函数
  ge::ascir::AscIrImpl ir_impl;
  ir_impl.codegen = []() { return std::unique_ptr<ge::ascir::AscIrCodegen>(new TestAscIrCodegenStub()); };

  // 创建AscIrDef对象并添加实现
  ge::ascir::AscIrDef ir_def;
  ir_def.Init("Abs", __FILE__, __LINE__);
  ir_def.AddSocImpl({platform_name}, ir_impl);

  // 注册到AscirRegistry
  registry.RegisterAscIr("Abs", ir_def);

  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarAbsToAdd(add_graph1));
  EXPECT_EQ(asc_adapt::FallbackScalarToBroadcastWithoutCheckType(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);

  // 校验结果
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "add_4") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
      asc_adapt::GetPeerOutNode(node, pre_add_node, 1);
      ASSERT_EQ(pre_add_node->GetType(), "Load");
    }
  }
  ASSERT_EQ(broadcast_cnt, 5);

  // 恢复原始注册表状态
  registry.ClearAll();
  for (const auto &item : original_registry) {
    registry.RegisterAscIr(item.first, item.second);
  }
}

// Broadcast后移判断Scalar节点后的Brc多引用场景在其后计算节点支持Scalar时也支持后移
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_ScalarMulRefsOK) {
  // 添加自定义AscIrCodegen实现，使IsScalarInputSupported返回true
  class TestAscIrCodegenStub : public ge::ascir::AscIrCodegen {
   public:
    bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
      return true;
    }
  };

  // 获取当前平台信息
  std::string platform_name;
  ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_name);

  // 保存原始注册表状态
  auto &registry = ge::ascir::AscirRegistry::GetInstance();
  auto original_registry = registry.GetAll();

  // 创建AscIrImpl对象，设置codegen创建函数
  ge::ascir::AscIrImpl ir_impl;
  ir_impl.codegen = []() { return std::unique_ptr<ge::ascir::AscIrCodegen>(new TestAscIrCodegenStub()); };

  // 创建AscIrDef对象并添加实现
  ge::ascir::AscIrDef ir_def;
  ir_def.Init("Add", __FILE__, __LINE__);
  ir_def.AddSocImpl({platform_name}, ir_impl);

  // 注册到AscirRegistry
  registry.RegisterAscIr("Add", ir_def);

  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphWithScalarMulRefToAdd(add_graph1));
  EXPECT_EQ(asc_adapt::FallbackScalarToBroadcastWithoutCheckType(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);

  // 校验结果
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "add_4") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Scalar");
      asc_adapt::GetPeerOutNode(node, pre_add_node, 1);
      ASSERT_EQ(pre_add_node->GetType(), "Scalar");
    }
    if (node->GetName() == "x_out_5") {
      NodePtr pre_add_node;
      asc_adapt::GetPeerOutNode(node, pre_add_node, 0);
      ASSERT_EQ(pre_add_node->GetType(), "Broadcast");
    }
  }
  ASSERT_EQ(broadcast_cnt, 5);

  // 恢复原始注册表状态
  registry.ClearAll();
  for (const auto &item : original_registry) {
    registry.RegisterAscIr(item.first, item.second);
  }
}

// Broadcast后移测试IsDtypeNotSupportOp返回true的场景
TEST_F(AscBackendPostProcessorTest, BroadcastBackward_DtypeNotSupportOp_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAscGraphCastDtypeNotSupport(add_graph1));
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  BroadcastBackwardPass broadcast_backward_pass;
  EXPECT_EQ(broadcast_backward_pass.Run(compute_graph), SUCCESS);

  // 校验结果：Broadcast后移不会移动到第二个Cast后
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetName() == "cast_dtype_not_support") {
      // 校验Cast节点的前驱不是Broadcast节点，说明Broadcast后移没有发生
      NodePtr pre_node;
      asc_adapt::GetPeerOutNode(node, pre_node, 0);
      ASSERT_EQ(pre_node->GetType(), "Broadcast");
    }
  }
  // 确保Broadcast节点数量符合预期
  ASSERT_EQ(broadcast_cnt, 2);
}

// CyclicExternalLiftPass测试IsDtypeNotSupport返回true的场景
TEST_F(AscBackendPostProcessorTest, Adaption_CyclicExternalLiftPass_DtypeNotSupport_Ok) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {
      {"const1", DT_FLOAT16}, {"const2", DT_FLOAT16}, {"shape1", DT_FLOAT16}, {"netoutput", DT_FLOAT16}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("cyclic_external_lift_dtype_not_support");
  attr1->SetAscGraph(CreatAscGraphCastDtypeNotSupport(add_graph1));

  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  EXPECT_EQ(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(compute_graph), SUCCESS);
  CyclicExternalLiftPass cycle_external_lift_pass;
  EXPECT_EQ(cycle_external_lift_pass.Run(compute_graph), SUCCESS);

  // 验证Store节点的直接前驱不是Broadcast节点，说明循环外提没有发生
  size_t broadcast_cnt = 0;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Output" || node->GetType() == "Data") {
      continue;
    }
    if (node->GetType() == "Broadcast") {
      broadcast_cnt++;
    }
    if (node->GetType() == "Store") {
      // 校验Cast节点的前驱不是Broadcast节点，说明Broadcast后移没有发生
      NodePtr pre_node;
      asc_adapt::GetPeerOutNode(node, pre_node, 0);
      ASSERT_EQ(pre_node->GetType(), "Cast");
    }
  }
  // 确保Broadcast节点数量符合预期
  ASSERT_EQ(broadcast_cnt, 2);
}
}  // namespace ge
