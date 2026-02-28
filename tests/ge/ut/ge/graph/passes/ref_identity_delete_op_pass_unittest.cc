/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the RefIdentityDeleteOpPass

#include <string>
#include <gtest/gtest.h>
#include "graph/passes/variable_optimize/ref_identity_delete_op_pass.h"
#include "graph/passes/pass_manager.h"
#include "graph/utils/node_utils.h"

using namespace domi;
using namespace ge;
class UTestRefIdentityDeleteOpPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string& name, const std::string& type) {
    op_desc_ = std::make_shared<OpDesc>(name, type);
  }

  NodeBuilder& AddInputDesc(const std::string &name, const std::vector<int64_t> &shape,
                            ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(name, ge::GeTensorDesc(GeShape(shape), format, data_type));
    return *this;
  }

  NodeBuilder& AddOutputDesc(const std::string &name, const std::vector<int64_t> &shape,
                             ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(name, ge::GeTensorDesc(GeShape(shape), format, data_type));
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr& graph) {
    return graph->AddNode(op_desc_);
  }

 private:
  ge::OpDescPtr op_desc_;
};

/**
 *   variable
 *      |
 *      |
 *  refidentity-->variable_ref   ==>     variable--------------
 *      |          ^   |                     |                |
 *      |          |   |                     |                |
 * applymomentum---|   |                 aplymomentum--->variable_ref
 *      |              |                     |
 *      |              |                     |
 *     add<-------------                    add
 */
TEST_F(UTestRefIdentityDeleteOpPass, ref_identity_delete_without_transnode_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
                                  .AddInputDesc("input", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc("output", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr ref_identity_node = NodeBuilder("RefIdentity", REFIDENTITY)
                                      .AddInputDesc("input", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc("output", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::NodePtr apply_monetum_node = NodeBuilder("Applymomentum", APPLYMOMENTUM)
                                       .AddInputDesc("var", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                       .AddOutputDesc("no_var", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                       .Build(graph);

  ge::NodePtr add_node = NodeBuilder("Add", ADD)
                             .AddInputDesc("x", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                             .AddOutputDesc("y", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                             .Build(graph);

  ge::NodePtr variable_ref = NodeBuilder("VariableRef", VARIABLE)
                                 .AddInputDesc("x", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                 .AddOutputDesc("y", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                 .Build(graph);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), ref_identity_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(ref_identity_node->GetOutDataAnchor(0), apply_monetum_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(ref_identity_node->GetOutDataAnchor(0), variable_ref->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(apply_monetum_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(variable_ref->GetOutControlAnchor(), add_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(apply_monetum_node->GetOutControlAnchor(), variable_ref->GetInControlAnchor());
  auto desc = apply_monetum_node->GetOpDesc()->GetInputDesc(0);
  desc.SetRefPortByIndex({0});
  apply_monetum_node->GetOpDesc()->UpdateInputDesc(0, desc);

  ge::ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  ge::NodePtr parent_node = NodeBuilder("parent_node", "PartitionedCall")
                                .AddInputDesc("input", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc("output", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(root_graph);
  ASSERT_EQ(ge::NodeUtils::AddSubgraph(*parent_node, "test", graph), ge::GRAPH_SUCCESS);

  PassManager pass_manager;
  pass_manager.AddPass("RefIdentityDeleteOpPass", new (std::nothrow) ge::RefIdentityDeleteOpPass);
  ge::Status status = pass_manager.Run(root_graph);
  EXPECT_EQ(status, ge::SUCCESS);

  EXPECT_EQ(variable_node->GetOutDataNodes().size(), 2);
  EXPECT_EQ(variable_node->GetOutControlNodes().size(), 0);
  const auto name0 = variable_node->GetOutDataNodes().at(0)->GetName();
  const auto name1 = variable_node->GetOutDataNodes().at(1)->GetName();
  const bool check0 = (name0 == "VariableRef" || name1 == "VariableRef");
  const bool check1 = (name0 == "Applymomentum" || name1 == "Applymomentum");
  EXPECT_EQ(check0, true);
  EXPECT_EQ(check1, true);

  EXPECT_EQ(apply_monetum_node->GetOutDataNodes().size(), 1);
  EXPECT_EQ(apply_monetum_node->GetOutControlNodes().size(), 1);
  EXPECT_EQ(apply_monetum_node->GetOutDataNodes().at(0)->GetName(), "Add");
  EXPECT_EQ(apply_monetum_node->GetOutControlNodes().at(0)->GetName(), "VariableRef");

  EXPECT_EQ(variable_ref->GetOutControlNodes().size(), 1);
}

TEST_F(UTestRefIdentityDeleteOpPass, ref_identity_delete_without_ref_identity) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
          .AddInputDesc("input", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .AddOutputDesc("output", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .Build(graph);

  ge::NodePtr apply_monetum_node = NodeBuilder("Applymomentum", APPLYMOMENTUM)
          .AddInputDesc("var", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .AddOutputDesc("no_var", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .Build(graph);

  ge::NodePtr add_node = NodeBuilder("Add", ADD)
          .AddInputDesc("x", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .AddOutputDesc("y", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .Build(graph);
          
  ge::NodePtr variable_ref = NodeBuilder("VariableRef", VARIABLE)
          .AddInputDesc("x", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .AddOutputDesc("y", {2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
          .Build(graph);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_monetum_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(apply_monetum_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(variable_ref->GetOutControlAnchor(), add_node->GetInControlAnchor());
  ge::GraphUtils::AddEdge(apply_monetum_node->GetOutControlAnchor(), variable_ref->GetInControlAnchor());
  auto desc = apply_monetum_node->GetOpDesc()->GetInputDesc(0);
  desc.SetRefPortByIndex({0});
  apply_monetum_node->GetOpDesc()->UpdateInputDesc(0, desc);

  PassManager pass_manager;
  pass_manager.AddPass("RefIdentityDeleteOpPass", new (std::nothrow) ge::RefIdentityDeleteOpPass);
  ge::Status status = pass_manager.Run(graph);
  EXPECT_EQ(status, ge::SUCCESS);
}
