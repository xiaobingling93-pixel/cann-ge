/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_82B0A0F961034EDDAF89F4D4CC1D3074_H
#define INC_82B0A0F961034EDDAF89F4D4CC1D3074_H

#include <common/types.h>

#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/shape.h"
#include "common/checker.h"
#include "graph/ge_tensor.h"

namespace gert {
struct ShareGraph {
  static ge::ComputeGraphPtr BuildCustomOpGraph();
  static ge::ComputeGraphPtr BuildOnlyCustomOpKnowShapeGraph();
  static ge::ComputeGraphPtr BuildCustomOpWithAddKnowShapeGraph();
  static ge::ComputeGraphPtr AicoreGraph();
  static ge::ComputeGraphPtr AicoreGraphTwoAdd();
  static ge::ComputeGraphPtr AtcNanoGraph();
  static ge::ComputeGraphPtr AicoreStaticGraph(bool is_with_atomic = false);
  static ge::ComputeGraphPtr LstmpGraph();
  static ge::ComputeGraphPtr IfGraph();
  static ge::ComputeGraphPtr IfGraph2();
  static ge::ComputeGraphPtr IfGraph3();
  static ge::ComputeGraphPtr IfGraph4();
  static ge::ComputeGraphPtr IfGraph5();
  static ge::ComputeGraphPtr IfGraphWithSwitch();
  static ge::ComputeGraphPtr IfGraphRankChangedOneBranch();
  static ge::ComputeGraphPtr IfGraphWithConstInput();
  static ge::ComputeGraphPtr GraphWithFifoWindowCache();
  static ge::ComputeGraphPtr CaseGraphWithRefData();
  static ge::ComputeGraphPtr IfWithKnownShapeSubGraph(const std::string &graph_name = "");
  static ge::ComputeGraphPtr IfWithKnownSubGraphAndMultiOutputs(const std::string &graph_name = "");
  static ge::ComputeGraphPtr CaseGraph();
  static ge::ComputeGraphPtr WhileGraph(bool instance_name_as_graph_name = false);
  static ge::ComputeGraphPtr WhileGraph2(bool instance_name_as_graph_name = false);
  static ge::ComputeGraphPtr WhileGraph3(bool instance_name_as_graph_name = false);
  static ge::ComputeGraphPtr WhileGraphXBody(bool instance_name_as_graph_name = false);
  static ge::ComputeGraphPtr WhileGraphCascade(bool instance_name_as_graph_name = false);
  static ge::ComputeGraphPtr WhileGraphInPartitionCall(bool instance_name_as_graph_name);
  static ge::ComputeGraphPtr BuildGraphRefdataWhile();
  static ge::ComputeGraphPtr BuildFakeDeterministicNodeGraph();
  static ge::ComputeGraphPtr IfOneBranchGraph();
  static ge::ComputeGraphPtr IfOneBranchGraph2();
  static ge::ComputeGraphPtr IfGraphShapeChangedOneBranch();
  static ge::ComputeGraphPtr IfGraphWithRefData();
  static ge::ComputeGraphPtr IfGraphWithRefDataAssignInsideSubgraph();
  static ge::ComputeGraphPtr IfWithDifferentPlacementSubgraph();
  static ge::ComputeGraphPtr BinaryKernelTypicalGraph();
  static ge::ComputeGraphPtr MatmulOmBinaryGraph();
  static ge::ComputeGraphPtr MatmulOmBinaryGraphV2();
  static ge::ComputeGraphPtr BuildFakeGetTensorNodeGraph();
  static ge::ComputeGraphPtr BuildFakeGetTensorNodeZeroCopyGraph();
  static ge::ComputeGraphPtr MatmulV2Graph(bool with_bias, bool with_offset);
  static ge::ComputeGraphPtr SimpleFooGraph();
  static ge::ComputeGraphPtr SimpleVariableGraph(const std::string &var_name = "variable");
  static ge::ComputeGraphPtr AutoFuseNodeGraph();
  static ge::ComputeGraphPtr AutofusePartitioncallGraph();
  static ge::ComputeGraphPtr SimpleVariableAddGraph();
  static ge::ComputeGraphPtr BuildNeedInsertCastGraphWithSubgraph();
  static ge::ComputeGraphPtr ReshapeAbnormalGraph();
  static ge::ComputeGraphPtr SimpleFileConstantGraph(const std::string &var_name = "variable",
                                                     const std::string &file_constant_name = "file_constant",
                                                     const std::string &location = "test_file_constant.bin");
  static ge::ComputeGraphPtr SimpleVariableAssignGraph(const std::string &var_name = "variable");
  static ge::ComputeGraphPtr SimpleStaticGraph();
  static ge::ComputeGraphPtr SimpleStaticPartitionedCallGraph();
  static ge::ComputeGraphPtr BuildAtomicAicoreGraph();
  static ge::ComputeGraphPtr BuildMemSetAicoreGraph();
  static ge::ComputeGraphPtr BuildStringNodeGraph();
  static ge::ComputeGraphPtr IFASingleGraph();
  static ge::ComputeGraphPtr BatchSingleGraph();
  static ge::ComputeGraphPtr GroupedMatMulAllReduceSingleGraph();
  static ge::ComputeGraphPtr CTCBeamSearchDecoderSingleGraph();
  static ge::ComputeGraphPtr BuildSingleNodeGraph(
      const std::string &node_type = "Add",
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildAddUnSqueezeGraph(
      const std::string &node_type = "Add",
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildAddAsTwoOutputGraph(
      const std::string &node_type = "Add",
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildUnsqueezeAsTwoOutputGraph(
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildInputAsTwoOutputGraph(
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildAssignAsTwoOutputGraph(
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildAddToUnSqueezeGraph(
      const std::string &node_type = "Add",
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildSingleConstPlaceHolderGraph(void *addr, size_t len);
  static ge::ComputeGraphPtr BuildSingleHcclNodeGraph(
      const std::string &node_type = "HcomReduce",
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildTwoHcclNodeGraph(
      const std::string &node_type = "HcomReduce",
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildAiCoreThirdClassNodeGraph();
  static ge::ComputeGraphPtr BuildTwoAddNodeGraph();
  static ge::ComputeGraphPtr BuildStaticTwoAddNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsReluExpAddNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsExpReluNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsTwoAddNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsAddExpNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsReduceReluNodeGraph();
  static ge::ComputeGraphPtr BuildStaticTwoReduceThreeAddNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAddReduceNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsAddReduceNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsTwoReduceNodeGraph();
  static ge::ComputeGraphPtr BuildStaticTwoReduceNodeGraph();
  static ge::ComputeGraphPtr BuildStaticReduceAbsReluNodeGraph();
  static ge::ComputeGraphPtr BuildStaticReduceAddReluNodeGraph();
  static ge::ComputeGraphPtr BuildStaticTwoReduceReluNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsReluAbsExpNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsReluAddNodeGraph();
  static ge::ComputeGraphPtr BuildStaticAbsReluReduceSumNodeGraph();
  static ge::ComputeGraphPtr BuildStaticReluAddAbsReluNodeGraph();
  static ge::ComputeGraphPtr BuildStaticReluAddReduceSumNodeGraph();
  static ge::ComputeGraphPtr BuildStaticReluAddAddNodeGraph();
  static ge::ComputeGraphPtr BuildThreeAddNodeGraph();
  static ge::ComputeGraphPtr BuildTwoAddNodeKnownShapeGraph();
  static ge::ComputeGraphPtr BuildDsaRandomNormalGraph(const std::string &node_type = "DSARandomNormal");
  static ge::ComputeGraphPtr BuildReshapeGraph();
  static ge::ComputeGraphPtr BuildReshapeGraph2();
  static ge::ComputeGraphPtr BuildGatherShapesGraph();
  static ge::ComputeGraphPtr BuildDataDependencyNodeGraph();
  static ge::ComputeGraphPtr BuildShapeToReshapeGraph();
  static ge::ComputeGraphPtr BuildAddConditionCalcGraph();
  static ge::ComputeGraphPtr BuildOpTilingGraph(const std::string &node_type);
  static ge::ComputeGraphPtr BuildGraphWithUBFusionNode();
  static ge::ComputeGraphPtr BuildWithKnownSubgraph(bool no_netoutput = false, bool external_weight = false);
  static ge::ComputeGraphPtr BuildWithKnownSubgraphWithTwoConst(bool no_netoutput = false);
  static ge::ComputeGraphPtr BuildWithKnownSubgraphWithRefNode();
  static ge::ComputeGraphPtr BuildWithMulitKnownSubgraphs();
  static ge::ComputeGraphPtr BuildWithNestingKnownSubgraph();
  static ge::ComputeGraphPtr BuildWithAllConstKnownSubgraph();
  static ge::ComputeGraphPtr BuildWithAllConstKnownSubgraph2();
  static ge::ComputeGraphPtr BuildWithInnerDataSubgraph();
  static ge::ComputeGraphPtr AicoreNoCompileResultGraph();
  static ge::ComputeGraphPtr BuildIdentityNGraph();
  static ge::ComputeGraphPtr ThirdAicpuOpGraph();
  static ge::ComputeGraphPtr FrameworkOPGraph(const string &real_node_type);
  static ge::ComputeGraphPtr VariableOPGraph(const string &real_node_type);
  static ge::ComputeGraphPtr BuildDataDependencySingleOpNodeGraph();
  static ge::ComputeGraphPtr Aicpu4thGraph();
  static ge::ComputeGraphPtr BuildHostCpuDataFlowGraph();
  static ge::ComputeGraphPtr BuildZeroInputAicoreGraph();
  static ge::ComputeGraphPtr BuildNoOpGraph();
  static ge::ComputeGraphPtr BuildSizeGraph();
  static ge::ComputeGraphPtr BuildCtrlToConstGraph();
  static ge::ComputeGraphPtr BuildRankGraph();
  static ge::ComputeGraphPtr BuildCompatibleInferShapeRangeGraph();
  static ge::ComputeGraphPtr BuildDynamicAndStaticGraph();
  static ge::ComputeGraphPtr AicoreWithRtsOverflowGraph();
  static ge::ComputeGraphPtr AicoreWithCmoGraph();
  static ge::ComputeGraphPtr AddWith4InputsAicoreGraph();
  static ge::ComputeGraphPtr BuildMinimumGradAndAddGraph();
  static ge::ComputeGraphPtr ConcatV2ConstDependencyGraph();
  static ge::ComputeGraphPtr ConcatV2ValueDependencyGraph();
  static ge::ComputeGraphPtr ConcatV2MultiOutNodesGraph();
  static ge::ComputeGraphPtr BuildFileConstantGraph();
  static ge::ComputeGraphPtr Build2FileConstantWithCtrlEdgeGraph();
  static ge::ComputeGraphPtr Build2StageGraph();
  static ge::ComputeGraphPtr Build1to2StageGraph();
  static ge::ComputeGraphPtr BuildAippDataGraph();
  static ge::ComputeGraphPtr Build2to1StageGraph();
  static ge::ComputeGraphPtr Build2StageWith1ToNGraph();
  static ge::ComputeGraphPtr BuildFakeNodeGraphWithMultipleInput();
  static ge::ComputeGraphPtr BuildLotsOfNodes(size_t node_num);
  static ge::ComputeGraphPtr AicpuOpWithDTSTRINGGraph();
  static ge::ComputeGraphPtr TensorListGraph();
  static ge::ComputeGraphPtr BuildBlockGraph();
  static ge::ComputeGraphPtr SingleInputAicoreGraph();
  static ge::ComputeGraphPtr IfCondByShapeGraph(bool by_rank = false);
  static ge::ComputeGraphPtr IfCondGraphWithRefdata();
  static ge::ComputeGraphPtr BuildStrideSliceGraph(
      std::vector<std::initializer_list<int64_t>> shape = {{-1}, {-1}, {-1}, {-1}},
      std::vector<std::initializer_list<int64_t>> min_shape = {{1}, {1}, {1}, {1}},
      std::vector<std::initializer_list<int64_t>> max_shape = {{-1}, {-1}, {-1}, {-1}});
  static ge::ComputeGraphPtr BuildSliceWriteNormalGraph(const std::string &node_type = "SliceWrite");
  static ge::ComputeGraphPtr BuildRefnodeGraph();
  static ge::Graph BuildSwitchMergeGraph();
  static ge::Graph BuildSwitchMergeGraphWithTwoOutputs();
  static ge::Graph BuildAtomicNodeConnectNetoutput();
  static ge::Graph BuildAtomicNodeConnectNetoutputThroughRefNode();
  static ge::Graph BuildSwitchMergeGraphWithNeg();
  static ge::Graph BuildSwitchMergeGraphWithMultiAddNodes();
  static ge::ComputeGraphPtr BuildDsaRandomNormalKnownGraph();
  static ge::ComputeGraphPtr BuildAddAndDsaRandomNormalKnownGraph();
  static ge::ComputeGraphPtr BuildVarConnectToSplit();
  static ge::ComputeGraphPtr FixedAddrNodeGraph();
  static ge::ComputeGraphPtr FixedAddrNodeGraph1();
  static ge::ComputeGraphPtr FixedAddrConnectToPhonyConcat();
  static ge::Graph NetoutputNotSupportZeroCopy();
  static ge::ComputeGraphPtr FixedAddrConnectToMultiPeers();
  static ge::ComputeGraphPtr BuildAiCoreRtsDsaNodeKnownShapeGraph();
  static ge::ComputeGraphPtr BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  static ge::ComputeGraphPtr BuildGraphHasLabelSwitch();
  static ge::Graph BuildHcomGraph();
  static ge::Graph BuildHcomGraphWithTwoOutputs(const std::string hcom_node_type = ge::HCOMALLGATHER);
  static ge::Graph BuildIoReuseMemGraph();
  static ge::Graph BuildHcomGraphWithRefData();
  static ge::Graph MultiBatchGraph();
  static ge::Graph BuildCVParallelGraph();
  static ge::Graph BuildCVSerialGraph();
  static ge::Graph OnlyDataGraph(std::initializer_list<int64_t> data0_shape, std::initializer_list<int64_t> data1_shape);
  static ge::ComputeGraphPtr ShapeToMultiAiCoreGraph();
  static ge::ComputeGraphPtr BuildMultiBatchShapesGraph();
  static ge::ComputeGraphPtr AicoreWithRtsDebugOverflowGraph();
  static ge::ComputeGraphPtr BuildInputDirectlyConnectedToOutputGraph();
  static ge::ComputeGraphPtr MultiStreamTwoNodeGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr GraphDynamicAndStaticGraphWithVariables(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphConsumersInAndCrossStream(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamWithHostMemAccessCrossStream(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphAccessRefMemCrossStream(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphRefMemCrossStream(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphDynamicAndStaticGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphFileConstantGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphFileConstantToHostGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphWithIfGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphWithFirstEventSyncGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr MultiStreamGraphWithLastEventSyncGraph(int64_t &stream_num, int64_t &event_num);
  static ge::ComputeGraphPtr BuildStaticMinimumGradAndAddGraph();
  static ge::ComputeGraphPtr BuildSubGraph(const std::string& name = "subgraph", int64_t parent_node_index = 0);
  static ge::ComputeGraphPtr BuildNestPartitioncallSubGraph(const ge::ComputeGraphPtr &main_graph, const std::string &name);
  static ge::ComputeGraphPtr BuildNestIfGraph();
  static ge::ComputeGraphPtr BuildNestCaseGraph();
  static ge::ComputeGraphPtr BuildNestIfGraph1();
  static ge::ComputeGraphPtr BuildNestIfSubGraph(const ge::ComputeGraphPtr &main_graph, const std::string &name);
  static ge::ComputeGraphPtr BuildNestIfGraph2();
  static ge::ComputeGraphPtr BuildNestIfSubGraph1(const ge::ComputeGraphPtr &main_graph, const std::string &name);
  static ge::ComputeGraphPtr BuildNestIfGraph3();
  static ge::ComputeGraphPtr BuildNestedPartitionedCallTwice();
  static ge::ComputeGraphPtr BuildIfWithNestedPartitionedCall();
  static ge::ComputeGraphPtr BuildCaseWithNestedPartitionedCall();
  // TODO value depend, aicpu, ffts, refdata
  };

class SingleNodeGraphBuilder {
 public:
  SingleNodeGraphBuilder(const std::string &graph_name, const std::string &type);
  SingleNodeGraphBuilder &NumInputs(size_t num_inputs);
  SingleNodeGraphBuilder &NumOutputs(size_t num_outputs);

  ge::ComputeGraphPtr Build(const ge::NodePtr& parent = nullptr);
  ge::ComputeGraphPtr BuildSubGraph(const ge::NodePtr &parent, int64_t parent_start);

 private:
  std::string name_;
  std::string type_;
  size_t num_inputs_ = 1U;
  size_t num_outputs_ = 1U;
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) {
    desc_ = std::make_shared<ge::OpDesc>(name, type);
  }
  NodeBuilder &Input(const ge::NodePtr &src, size_t index = 0U, const std::string &name = "") {
    inputs_[src][desc_->GetAllInputsSize()] = index;
    if (!name.empty()) {
      desc_->AddInputDesc(name, src->GetOpDesc()->GetOutputDesc(index));
    } else {
      desc_->AddInputDesc(src->GetOpDesc()->GetOutputDesc(index));
    }
    return *this;
  }
  NodeBuilder &ControlInput(const ge::NodePtr &src) {
    control_inputs_.insert(src);
    return *this;
  }
  NodeBuilder &Output(ge::DataType type = ge::DT_FLOAT, const std::vector<int64_t> &dims = {},
                      ge::Format format = ge::FORMAT_NCHW, const std::string &name = "") {
    ge::GeTensorDesc desc(ge::GeShape{dims}, format, type);
    if (!name.empty()) {
      desc_->AddOutputDesc(name, desc);
    } else {
      desc_->AddOutputDesc(desc);
    }
    return *this;
  }

  NodeBuilder &Attr(const std::string &name, const ge::ComputeGraphPtr &graph) {
    subgraphs_.emplace_back(std::make_pair(name, graph));
    return *this;
  }

  NodeBuilder &Attr(const std::string &name, int64_t v) {
    ge::AttrUtils::SetInt(desc_, name, v);
    return *this;
  }

  NodeBuilder &Attr(const std::string &name, const std::string &v) {
    ge::AttrUtils::SetStr(desc_, name, v);
    return *this;
  }

  NodeBuilder &Attr(const std::string &name, const ge::GeTensor &tensor) {
    ge::AttrUtils::SetTensor(desc_, name, tensor);
    return *this;
  }

  NodeBuilder &AttrBool(const std::string &name, const bool &v) {
    ge::AttrUtils::SetBool(desc_, name, v);
    return *this;
  }

  ge::NodePtr Build(ge::ComputeGraphPtr &parent);

 private:
  std::shared_ptr<ge::OpDesc> desc_;
  std::vector<std::pair<std::string, ge::ComputeGraphPtr>> subgraphs_;
  std::map<ge::NodePtr, std::map<size_t, size_t>> inputs_;
  std::set<ge::NodePtr> control_inputs_;
};

void AddCompileResult(const ge::NodePtr &node, bool atomic,
                      const char *compile_info_json = "{\"vars\": {\"tune_shape_list\":[[-1,-1,0]]}}");
void SetGraphOutShapeRange(const ge::ComputeGraphPtr graph);
}  // namespace gert

#endif
