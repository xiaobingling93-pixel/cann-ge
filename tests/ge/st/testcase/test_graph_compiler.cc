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
#include "ge/ge_api.h"
#include "ge/ge_api_v2.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/passes/feature/set_ffts_plus_attr_pass.h"
#include "base/err_mgr.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/manager/graph_manager.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "graph/build/memory/block_type_list.h"
#include "hybrid/node_executor/node_executor.h"
#include "macro_utils/dt_public_unscope.h"
#include "dflow/base/model/endpoint.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "framework/memory/memory_assigner.h"
#include "generator/ge_generator.h"
#include "graph/attr_value.h"
#include "api/gelib/gelib.h"
#include "graph/utils/tensor_utils.h"
#include "graph/operator_factory_impl.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/manager/graph_manager.h"
#include "graph/ge_context.h"
#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.h"
#include "register/ops_kernel_builder_registry.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/tensor_utils.h"
#include "framework/common/ge_types.h"
#include "framework/memory/memory_api.h"
#include "ge/ge_api_types.h"
#include "graph/passes/memory_conflict/set_input_output_offset_pass.h"
#include "common/sgt_slice_type.h"
#include "ge_running_env/fake_op.h"
#include "depends/runtime/src/runtime_stub.h"
#include "omg/ge_init.h"
#include "stub/gert_runtime_stub.h"
#include "graph/manager/mem_manager.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "faker/global_data_faker.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/mock_ops_kernel_builder.h"
#include "common/share_graph.h"
#include "tests/ge/st/stubs/utils/synchronizer.h"
#include "graph/passes/memory_conflict/hccl_memcpy_pass.h"
#include "array.h"
#include "framework/ge_runtime_stub/include/common/compliant_share_graph.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/passes/feature/auto_fuse_pass.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "common/env_path.h"
#include "graph/build/memory/checker/atomic_clean_checker.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "common/platform_context.h"
#include "common/mem_conflict_share_graph.h"
#include "graph/optimize/autofuse/autofuse_optimize.h"
#include "graph/manager/graph_var_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace ge;

// Temporary code that Wait for the FE formal solution to implment.
namespace{
static const char_t *const kGeLocalEngineName = "DNN_VM_GE_LOCAL";
static const char_t *const kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";
static const char_t *const kAIcoreEngine = "AIcoreEngine";
static constexpr const char_t *TBE_OP_ATOMIC_DTYPES = "tbe_op_atomic_dtypes";
static constexpr const char_t *TBE_OP_ATOMIC_INT64_VALUES = "tbe_op_atomic_int64_values";
static constexpr const char_t *TBE_OP_ATOMIC_FLOAT_VALUES = "tbe_op_atomic_float_values";
graphStatus StubInferFunction(Operator &op) { return GRAPH_SUCCESS; }
enum class kGraphOptimizerOption {
  kNormal,
  kMockDifferentMemSetAttrs,
  kMockSameMemSetAttrs
};
kGraphOptimizerOption graph_optimizer_option{kGraphOptimizerOption::kNormal};
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    return dlsym(handle, func_name);
  }
};
class FakeAicoreMemSetOptimizer : public FakeGraphOptimizer {
 public:
  Status OptimizeFusedGraph(ComputeGraph &graph) override {
    if (graph_optimizer_option == kGraphOptimizerOption::kNormal) {
      return SUCCESS;
    }
    std::vector<uint32_t> value;
    value.push_back(0);
    int32_t val = 0;
    for (auto &node : graph.GetAllNodes()) {
      if (node->GetType() == ADD) {
        EXPECT_EQ(AttrUtils::SetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
        EXPECT_EQ(AttrUtils::SetListInt(node->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {static_cast<int32_t>(DT_INT64)}), true);
        if (graph_optimizer_option == kGraphOptimizerOption::kMockDifferentMemSetAttrs) {
          EXPECT_EQ(AttrUtils::SetListInt(node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {val}), true);
        } else {
          EXPECT_EQ(AttrUtils::SetListInt(node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0}), true);
        }
        val++;
      }
    }
    return SUCCESS;
  }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = kAIcoreEngine;
    return 0;
  }
};

bool test_origin_hcom_unordered = false;
bool test_subgraph_hcom_unordered = false;

class FakeHcclOptimizer : public FakeGraphOptimizer {
 public:
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    if (test_subgraph_hcom_unordered) {
      auto reduce_2 = graph.FindNode("allreduce2");
      auto gather_1 = graph.FindNode("allgather1");
      if ((reduce_2 != nullptr) && (gather_1 != nullptr)) {
        GraphUtils::AddEdge(reduce_2->GetOutControlAnchor(), gather_1->GetInControlAnchor());
      }
    }
    return SUCCESS;
  }

  Status OptimizeOriginalGraph(ComputeGraph &graph) override {
    if (test_origin_hcom_unordered) {
      auto reduce_2 = graph.FindNode("allreduce2");
      auto gather_1 = graph.FindNode("allgather1");
      if ((reduce_2 != nullptr) && (gather_1 != nullptr)) {
        GraphUtils::AddEdge(reduce_2->GetOutControlAnchor(), gather_1->GetInControlAnchor());
      }
    }
    return SUCCESS;
  }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = "DNN_HCCL";
    return 0;
  }
};

struct FakeCompilerOpsKernelInfoStore : public FakeOpsKernelInfoStore {
  FakeCompilerOpsKernelInfoStore() : FakeOpsKernelInfoStore() {}
  bool CheckSupported(const NodePtr &node, std::string &reason, CheckSupportFlag &flag) const override {
    if (node->GetName() == "not_support_check_support") {
      return false;
    }
    return true;
  }
};
class MockGraphOptimizer {
 public:
  explicit MockGraphOptimizer(kGraphOptimizerOption option) {
    auto infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      return GRAPH_SUCCESS;
    };
    auto cast_infer_fun = [](Operator &op) -> graphStatus {
      return GRAPH_SUCCESS;
    };
    auto ge_env = GeRunningEnvFaker();
    auto fake_aicore_memset_optimizer = MakeShared<FakeAicoreMemSetOptimizer>();
    auto fake_hccl_optimizer = MakeShared<FakeHcclOptimizer>();
    ge_env.Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine(kAIcoreEngine)
                     .KernelInfoStore(kAIcoreEngine)
                     .GraphOptimizer(kAIcoreEngine, fake_aicore_memset_optimizer))
        .Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU"))
        .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
        .Install(FakeEngine(kEngineNameAiCpuTf).KernelInfoStore(kEngineNameAiCpuTf))
        .Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").
                 GraphOptimizer("DNN_HCCL", fake_hccl_optimizer))
        .Install(FakeOp(RELU).InfoStoreAndBuilder(kAIcoreEngine).InferShape(infer_fun))
        .Install(FakeOp(CONV2D).InfoStoreAndBuilder(kAIcoreEngine).InferShape(infer_fun))
        .Install(FakeOp(CAST).InfoStoreAndBuilder(kAIcoreEngine).InferShape(cast_infer_fun))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(IDENTITY).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(VARIABLEV2).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONTROLTRIGGER).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(ASSIGN).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeOp(ADD).InfoStoreAndBuilder(kAIcoreEngine).InferShape(infer_fun))
        .Install(FakeOp(MEMSET).InfoStoreAndBuilder(kAIcoreEngine))
        .Install(FakeOp(HCOMALLGATHER).InfoStoreAndBuilder("ops_kernel_info_hccl").InferShape(infer_fun))
        .Install(FakeOp(HCOMALLREDUCE).InfoStoreAndBuilder("ops_kernel_info_hccl").InferShape(infer_fun))
        .Install(FakeOp("SuperKernel").InfoStoreAndBuilder(kAIcoreEngine))
        .Install(FakeOp(MATMUL).InfoStoreAndBuilder(kAIcoreEngine))
        .Install(FakeOp(DEQUANTIZE).InfoStoreAndBuilder(kAIcoreEngine))
        .Install(FakeOp(BATCHMATMUL).InfoStoreAndBuilder(kAIcoreEngine))
        .Install(FakeOp(SEND).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(RECV).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"));
    graph_optimizer_option = option;
  }

  virtual ~MockGraphOptimizer() {
    auto ge_env = GeRunningEnvFaker();
    ge_env.InstallDefault();
    graph_optimizer_option = kGraphOptimizerOption::kNormal;
  }
};

static void BuildFftsDynamicGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &dsp_graph, ComputeGraphPtr &ffts_graph) {
  const auto SetUnknownOpKernel = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    static uint32_t index = 0U;
    const static std::set<std::string> kGeLocalTypes{ DATA, CONSTANT, VARIABLE, NETOUTPUT, AIPP_DATA_TYPE };

    GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
    TensorUtils::SetSize(tensor, 64);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      (void)AttrUtils::SetBool(op_desc, "OwnerGraphIsUnknown", true);
      std::string op_kernel_name =  (kGeLocalTypes.count(op_desc->GetType()) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE";
      op_desc->SetOpKernelLibName(op_kernel_name);

      vector<int64_t> output_offset;
      for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
        op_desc->UpdateOutputDesc(i, tensor);
        output_offset.emplace_back(index * 64);
        ++index;
      }
      op_desc->SetOutputOffset(output_offset);
      op_desc->SetWorkspace({});
      op_desc->SetWorkspaceBytes({});
    }

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      vector<int64_t> input_offset;
      for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
        op_desc->UpdateInputDesc(i, tensor);
        if (node->GetType() == NETOUTPUT && node->GetName() != NODE_NAME_NET_OUTPUT) {
          AttrUtils::SetInt(op_desc->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
        }

        const auto in_anchor = node->GetInDataAnchor(i);
        if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
          input_offset.emplace_back(-1);
          continue;
        }

        const auto out_anchor = in_anchor->GetPeerOutAnchor();
        const auto peer_node = out_anchor->GetOwnerNode();
        const vector<int64_t> output_offset = peer_node->GetOpDesc()->GetOutputOffset();
        if (static_cast<size_t>(out_anchor->GetIdx()) >= output_offset.size()) {
          input_offset.emplace_back(-1);
          continue;
        }

        input_offset.emplace_back(output_offset.at(out_anchor->GetIdx()));
      }
      op_desc->SetInputOffset(input_offset);
    }
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
  };

  root_graph = ToComputeGraph(g1);
  SetUnknownOpKernel(root_graph->GetDirectNode());
  const auto root_call_0 = root_graph->FindNode("PartitionedCall_0");
  EXPECT_NE(root_call_0, nullptr);

  auto dsp_graph_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto dsp_graph_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(g2) {
    CHAIN(NODE("dsp_graph/_arg_0", dsp_graph_data_0)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_0", IDENTITY)->EDGE(0, 0)
              ->NODE("dsp_graph/PartitionedCall_0", PARTITIONEDCALL)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_2", IDENTITY)->EDGE(0, 0)
              ->NODE("dsp_graph/Node_Output", NETOUTPUT)
    );
    CHAIN(NODE("dsp_graph/_arg_1", dsp_graph_data_1)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_1", IDENTITY)->EDGE(0, 1)
              ->NODE("dsp_graph/PartitionedCall_0")
    );
  };

  dsp_graph = ToComputeGraph(g2);
  SetUnknownOpKernel(dsp_graph->GetDirectNode());
  dsp_graph->SetGraphUnknownFlag(true);
  dsp_graph->SetParentNode(root_call_0);
  dsp_graph->SetParentGraph(root_graph);
  root_call_0->GetOpDesc()->AddSubgraphName("f");
  root_call_0->GetOpDesc()->SetSubgraphInstanceName(0, dsp_graph->GetName());
  root_graph->AddSubgraph(dsp_graph);
  const auto dsp_graph_call_0 = dsp_graph->FindNode("dsp_graph/PartitionedCall_0");
  EXPECT_NE(dsp_graph_call_0, nullptr);
  AttrUtils::SetBool(dsp_graph_call_0->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);

  auto sgt_graph_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sgt_graph_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto sgt_graph_conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                                        .Attr(ATTR_NAME_IMPLY_TYPE, 1)           // domi::ImplyType::TVM
                                        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto sgt_graph_relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                                      .Attr(ATTR_NAME_IMPLY_TYPE, 1)
                                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  DEF_GRAPH(g3) {
    CHAIN(NODE("sgt_graph/_arg_0", sgt_graph_data_0)->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", sgt_graph_conv_0)->EDGE(0, 0)
              ->NODE("sgt_graph/Relu", sgt_graph_relu_0)->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT)
    );
    CHAIN(NODE("sgt_graph/_arg_1", sgt_graph_data_1)->EDGE(0, 1)
              ->NODE("sgt_graph/Conv2D", sgt_graph_conv_0)
    );
  };

  ffts_graph = ToComputeGraph(g3);
  SetUnknownOpKernel(ffts_graph->GetDirectNode());
  ffts_graph->SetGraphUnknownFlag(true);
  ffts_graph->SetParentNode(dsp_graph_call_0);
  ffts_graph->SetParentGraph(dsp_graph);
  dsp_graph_call_0->GetOpDesc()->AddSubgraphName("f");
  dsp_graph_call_0->GetOpDesc()->SetSubgraphInstanceName(0, ffts_graph->GetName());
  root_graph->AddSubgraph(ffts_graph);
}


void SetNoPaddingContinousInputs(ComputeGraphPtr &graph, const std::string &name) {
  auto node = graph->FindNode(name);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
}

void SetNoPaddingContinousOutputs(ComputeGraphPtr &graph, const std::string &name) {
  auto node = graph->FindNode(name);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
}

static void BuildFftsGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &dsp_graph, ComputeGraphPtr &ffts_graph) {
  const auto SetOpSize = [](const ComputeGraph::Vistor<NodePtr> &all_nodes, int64_t size) {
    const static std::set<std::string> kGeLocalTypes{ DATA, CONSTANT, VARIABLE, NETOUTPUT, AIPP_DATA_TYPE };
    GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
    TensorUtils::SetSize(tensor, size);
    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      std::string op_kernel_name =  (kGeLocalTypes.count(op_desc->GetType()) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE";
      op_desc->SetOpKernelLibName(op_kernel_name);
      for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
        op_desc->UpdateInputDesc(i, tensor);
        if (node->GetType() == NETOUTPUT && node->GetName() != NODE_NAME_NET_OUTPUT) {
          AttrUtils::SetInt(op_desc->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
        }
      }
      for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
        op_desc->UpdateOutputDesc(i, tensor);
      }
    }
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
  };

  root_graph = ToComputeGraph(g1);
  SetOpSize(root_graph->GetDirectNode(), 1024);
  const auto root_call_0 = root_graph->FindNode("PartitionedCall_0");
  EXPECT_NE(root_call_0, nullptr);

  auto dsp_graph_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto dsp_graph_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(g2) {
    CHAIN(NODE("dsp_graph/_arg_0", dsp_graph_data_0)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_0", IDENTITY)->EDGE(0, 0)
              ->NODE("dsp_graph/PartitionedCall_0", PARTITIONEDCALL)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_2", IDENTITY)->EDGE(0, 0)
              ->NODE("dsp_graph/Node_Output", NETOUTPUT)
    );
    CHAIN(NODE("dsp_graph/_arg_1", dsp_graph_data_1)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_1", IDENTITY)->EDGE(0, 1)
              ->NODE("dsp_graph/PartitionedCall_0")
    );
  };

  dsp_graph = ToComputeGraph(g2);
  SetOpSize(dsp_graph->GetDirectNode(), 2048);
  dsp_graph->SetParentNode(root_call_0);
  dsp_graph->SetParentGraph(root_graph);
  root_call_0->GetOpDesc()->AddSubgraphName("f");
  root_call_0->GetOpDesc()->SetSubgraphInstanceName(0, dsp_graph->GetName());
  root_graph->AddSubgraph(dsp_graph);
  const auto dsp_graph_call_0 = dsp_graph->FindNode("dsp_graph/PartitionedCall_0");
  EXPECT_NE(dsp_graph_call_0, nullptr);
  AttrUtils::SetBool(dsp_graph_call_0->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);

  auto sgt_graph_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sgt_graph_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto sgt_graph_conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
      .Attr(ATTR_NAME_IMPLY_TYPE, 1)           // domi::ImplyType::TVM
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto sgt_graph_relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
      .Attr(ATTR_NAME_IMPLY_TYPE, 1)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto sgt_graph_bn_0 = OP_CFG(RELU).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
      .Attr(ATTR_NAME_IMPLY_TYPE, 1)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto sgt_graph_matmul_0 = OP_CFG(RELU).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
      .Attr(ATTR_NAME_IMPLY_TYPE, 1)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  DEF_GRAPH(g3) {
                  CHAIN(NODE("sgt_graph/_arg_0", sgt_graph_data_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/Conv2D", sgt_graph_conv_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/Bn", sgt_graph_conv_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/Relu", sgt_graph_bn_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/Mul", sgt_graph_matmul_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/Add1", sgt_graph_matmul_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/PonyConcat", sgt_graph_matmul_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/PonyReduce", sgt_graph_matmul_0)->EDGE(0, 0)
                            ->NODE("sgt_graph/Node_Output", NETOUTPUT)
                  );
                  CHAIN(NODE("sgt_graph/_arg_1", sgt_graph_data_1)->EDGE(0, 1)
                            ->NODE("sgt_graph/Conv2D", sgt_graph_conv_0)
                  );
                  CHAIN(NODE("sgt_graph/Mul", sgt_graph_matmul_0)->EDGE(1, 0)
                            ->NODE("sgt_graph/Add2", sgt_graph_matmul_0)->EDGE(0, 1)
                            ->NODE("sgt_graph/PonyConcat", sgt_graph_matmul_0)
                  );
                  CHAIN(NODE("sgt_graph/PonyReduce", sgt_graph_matmul_0)->EDGE(1, 0)
                            ->NODE("sgt_graph/Node_Output", NETOUTPUT)
                  );
  };

  ffts_graph = ToComputeGraph(g3);
  SetNoPaddingContinousInputs(ffts_graph, "sgt_graph/PonyConcat");
  SetNoPaddingContinousOutputs(ffts_graph, "sgt_graph/PonyReduce");
  SetOpSize(ffts_graph->GetDirectNode(), 4096);

  // workspace size
  for (auto &cur_node : ffts_graph->GetAllNodes()) {
    if (cur_node != nullptr && cur_node->GetOpDesc() != nullptr) {
      cur_node->GetOpDesc()->SetWorkspaceBytes({2, 10, 12});
    }
  }
  ffts::ThreadSliceMapPtr slice_info_ptr1 = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr1->thread_mode = 1U;
  for (const auto &cur_node : ffts_graph->GetDirectNode()) {
    (void)ge::AttrUtils::SetInt(cur_node->GetOpDesc(), ATTR_NAME_THREAD_SCOPE_ID, 1);
    slice_info_ptr1->parallel_window_size = 4U;
    slice_info_ptr1->slice_instance_num = 8U;
    (void)cur_node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr1);
  }

  ffts_graph->SetParentNode(dsp_graph_call_0);
  ffts_graph->SetParentGraph(dsp_graph);
  dsp_graph_call_0->GetOpDesc()->AddSubgraphName("f");
  dsp_graph_call_0->GetOpDesc()->SetSubgraphInstanceName(0, ffts_graph->GetName());
  root_graph->AddSubgraph(ffts_graph);
}

/**
 *           ------>Const
 *          |            |
 * var->transdata1->transdata2>cast->conv2d
 */
void BuildGraphHasTrans(ComputeGraphPtr &graph) {
  GeTensorDesc nc1hwc0_tensor1(GeShape({1, 1, 224, 224, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  GeTensorDesc nhwc_tensor(GeShape({1, 224, 224, 16}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  GeTensorDesc nc1hwc0_tensor2(GeShape({1, 1, 224, 224, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto var_desc = std::make_shared<OpDesc>("Variable_1", VARIABLE);
  var_desc->SetOpEngineName(kGeLocalEngineName);
  var_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  var_desc->AddOutputDesc(nc1hwc0_tensor1);
  auto var_node = graph->AddNode(var_desc);

  auto transdata_desc1 = std::make_shared<OpDesc>("Transdata_1", "TransData");
  transdata_desc1->SetOpEngineName(kGeLocalEngineName);
  transdata_desc1->SetOpKernelLibName(kGeLocalOpKernelLibName);
  transdata_desc1->AddInputDesc(nc1hwc0_tensor1);
  transdata_desc1->AddOutputDesc(nhwc_tensor);
  auto transdata_node1 = graph->AddNode(transdata_desc1);

  auto transdata_desc2 = std::make_shared<OpDesc>("Transdata_2", "TransData");
  transdata_desc2->SetOpEngineName(kGeLocalEngineName);
  transdata_desc2->SetOpKernelLibName(kGeLocalOpKernelLibName);
  transdata_desc2->AddInputDesc(nhwc_tensor);
  transdata_desc2->AddOutputDesc(nc1hwc0_tensor1);
  auto transdata_node2 = graph->AddNode(transdata_desc2);

  auto cast_desc = std::make_shared<OpDesc>("Cast1", CAST);
  cast_desc->SetOpEngineName(kGeLocalEngineName);
  cast_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  cast_desc->AddInputDesc(nc1hwc0_tensor1);
  cast_desc->AddOutputDesc(nc1hwc0_tensor2);
  auto cast_node = graph->AddNode(cast_desc);

  auto conv2d_desc = std::make_shared<OpDesc>("conv2d", "Conv2D");
  conv2d_desc->SetOpEngineName(kGeLocalEngineName);
  conv2d_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  conv2d_desc->AddInputDesc(nc1hwc0_tensor2);
  conv2d_desc->AddOutputDesc(nc1hwc0_tensor2);
  auto conv2d_node = graph->AddNode(conv2d_desc);

  auto y_desc = std::make_shared<OpDesc>("y", DATA);
  y_desc->SetOpEngineName(kGeLocalEngineName);
  y_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  y_desc->AddInputDesc(scalar_tensor);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  (void)GraphUtils::AddEdge(var_node->GetOutDataAnchor(0), transdata_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(transdata_node1->GetOutDataAnchor(0), transdata_node2->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(transdata_node2->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutControlAnchor(), transdata_node2->GetInControlAnchor());
  (void)GraphUtils::AddEdge(transdata_node1->GetOutControlAnchor(), y_node->GetInControlAnchor());
}

void make_graph_can_recompute(ComputeGraphPtr &graph) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto reshape_desc = std::make_shared<OpDesc>("Reshape_ReshapeRecoveryPass_1", RESHAPE);
  reshape_desc->SetOpEngineName(kGeLocalEngineName);
  reshape_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  GeTensorDesc tensor_desc(GeShape({-1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  reshape_desc->AddInputDesc(tensor_desc);
  reshape_desc->AddInputDesc(tensor_desc);
  reshape_desc->AddOutputDesc(scalar_tensor);
  auto reshape_node = graph->AddNode(reshape_desc);

  auto x_desc = std::make_shared<OpDesc>("x", DATA);
  x_desc->SetOpEngineName(kGeLocalEngineName);
  x_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  x_desc->AddInputDesc(scalar_tensor);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", DATA);
  (void)ge::AttrUtils::SetBool(y_desc, "_backward", true);
  y_desc->SetOpEngineName(kGeLocalEngineName);
  y_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  y_desc->AddInputDesc(scalar_tensor);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", DATA);
  z_desc->SetOpEngineName(kGeLocalEngineName);
  z_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  z_desc->AddInputDesc(scalar_tensor);
  z_desc->AddOutputDesc(tensor_desc);
  auto z_node = graph->AddNode(z_desc);

  auto w_desc = std::make_shared<OpDesc>("w", DATA);
  w_desc->SetOpEngineName(kGeLocalEngineName);
  w_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  w_desc->AddInputDesc(scalar_tensor);
  w_desc->AddOutputDesc(scalar_tensor);
  auto w_node = graph->AddNode(w_desc);

  auto add_desc = std::make_shared<OpDesc>("Add", ADD);
  add_desc->SetOpEngineName(kGeLocalEngineName);
  add_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  (void)ge::AttrUtils::SetBool(add_desc, "_recompute", true);
  add_desc->AddInputDesc(scalar_tensor);
  add_desc->AddInputDesc(scalar_tensor);
  add_desc->AddOutputDesc(scalar_tensor);
  auto add_node = graph->AddNode(add_desc);

  auto add_desc1 = std::make_shared<OpDesc>("gradients/Add1", ADD);
  add_desc1->SetOpEngineName(kGeLocalEngineName);
  add_desc1->SetOpKernelLibName(kGeLocalOpKernelLibName);
  (void)ge::AttrUtils::SetBool(add_desc1, "_backward", true);
  add_desc1->AddInputDesc(scalar_tensor);
  add_desc1->AddInputDesc(scalar_tensor);
  add_desc1->AddOutputDesc(scalar_tensor);
  auto add_node1 = graph->AddNode(add_desc1);

  auto mul_desc = std::make_shared<OpDesc>("matmul", MATMUL);
  mul_desc->SetOpEngineName(kGeLocalEngineName);
  mul_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  mul_desc->AddInputDesc(scalar_tensor);
  mul_desc->AddInputDesc(scalar_tensor);
  mul_desc->AddOutputDesc(scalar_tensor);
  auto mul_node = graph->AddNode(mul_desc);

  auto mul_desc1 = std::make_shared<OpDesc>("gradients/matmul", MATMUL);
  mul_desc1->SetOpEngineName(kGeLocalEngineName);
  mul_desc1->SetOpKernelLibName(kGeLocalOpKernelLibName);
  (void)ge::AttrUtils::SetBool(mul_desc1, "_backward", true);
  mul_desc1->AddInputDesc(scalar_tensor);
  mul_desc1->AddInputDesc(scalar_tensor);
  mul_desc1->AddOutputDesc(scalar_tensor);
  auto mul_node1 = graph->AddNode(mul_desc1);

  auto sqrt_desc = std::make_shared<OpDesc>("gradients/sqrt", SQRT);
  sqrt_desc->SetOpEngineName(kGeLocalEngineName);
  sqrt_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  (void)ge::AttrUtils::SetBool(sqrt_desc, "_backward", true);
  sqrt_desc->AddInputDesc(scalar_tensor);
  sqrt_desc->AddOutputDesc(scalar_tensor);
  auto sqrt_node = graph->AddNode(sqrt_desc);

  auto output_desc = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  output_desc->SetOpEngineName(kGeLocalEngineName);
  output_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  output_desc->AddInputDesc(scalar_tensor);
  output_desc->AddOutputDesc(scalar_tensor);
  auto output_node = graph->AddNode(output_desc);

  auto partitioncall_desc = std::make_shared<OpDesc>("partitioncall", PARTITIONEDCALL);
  partitioncall_desc->SetOpEngineName(kGeLocalEngineName);
  partitioncall_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  partitioncall_desc->AddInputDesc(scalar_tensor);
  partitioncall_desc->AddOutputDesc(scalar_tensor);
  auto partitioncall_node = graph->AddNode(partitioncall_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), add_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(w_node->GetOutDataAnchor(0), add_node1->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node1->GetOutDataAnchor(0), mul_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), mul_node1->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node1->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(sqrt_node->GetOutControlAnchor(), mul_node1->GetInControlAnchor());
  (void)GraphUtils::AddEdge(mul_node1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), partitioncall_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(partitioncall_node->GetOutControlAnchor(), output_node->GetInControlAnchor());

  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>("test_subgraph");
  auto x1_desc = std::make_shared<OpDesc>("x1", DATA);
  x1_desc->SetOpEngineName(kGeLocalEngineName);
  x1_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  x1_desc->AddInputDesc(scalar_tensor);
  x1_desc->AddOutputDesc(scalar_tensor);
  AttrUtils::SetInt(x1_desc, ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto x1_node = subgraph->AddNode(x1_desc);

  auto output1_desc = std::make_shared<OpDesc>("NetOutput1", NETOUTPUT);
  output1_desc->SetOpEngineName(kGeLocalEngineName);
  output1_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
  GeTensorDesc scalar_tensor1(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  AttrUtils::SetInt(scalar_tensor1, ATTR_NAME_PARENT_NODE_INDEX, 0);
  output1_desc->AddInputDesc(scalar_tensor1);
  output1_desc->AddOutputDesc(scalar_tensor);
  auto output1_node = subgraph->AddNode(output1_desc);
  (void)GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), output1_node->GetInDataAnchor(0));

  subgraph->SetParentNode(partitioncall_node);
  partitioncall_node->GetOpDesc()->AddSubgraphName("test_subgraph");
  partitioncall_node->GetOpDesc()->SetSubgraphInstanceName(0, "test_subgraph");
  subgraph->SetParentGraph(graph);
  graph->AddSubgraph(subgraph);
}

/**
 *              Add
 *               |
 *          Phonyconcat
 *             /     \
 *           relu   Netoutput
 */
void MakeGraphDataInParent3(ComputeGraphPtr &graph) {
  auto desc_ptr = MakeShared<ge::GeTensorDesc>();
  auto desc = *desc_ptr;
  OpDescPtr op_desc_data = MakeShared<OpDesc>("Add", ADD);
  op_desc_data->AddOutputDesc(desc);

  OpDescPtr op_desc_input = MakeShared<OpDesc>("Phonyconcat", CONCAT);
  op_desc_input->AddOutputDesc(desc);
  op_desc_input->AddInputDesc(desc);

  OpDescPtr op_desc_out = MakeShared<OpDesc>("Netoutput", NETOUTPUT);
  op_desc_out->AddInputDesc(desc);

  OpDescPtr op_desc_out_relu = MakeShared<OpDesc>("Relu", RELU);
  op_desc_out_relu->AddInputDesc(desc);

  vector<int> connect_output = {0};
  AttrUtils::SetListInt(op_desc_input, ATTR_NAME_NODE_CONNECT_OUTPUT, connect_output);
  bool attr_no_task = true;
  ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_NOTASK, attr_no_task);
  bool is_input_continuous = true;
  ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
  op_desc_input->SetOutputOffset({0});
  op_desc_data->SetOutputOffset({0});

  std::vector<int64_t> offsets_for_fusion = {0};
  AttrUtils::SetListInt(op_desc_data, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);

  NodePtr data_node = graph->AddNode(op_desc_data);
  NodePtr phony_node = graph->AddNode(op_desc_input);
  NodePtr out_node = graph->AddNode(op_desc_out);
  NodePtr out_relu_node = graph->AddNode(op_desc_out_relu);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), phony_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(phony_node->GetOutDataAnchor(0), out_relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(phony_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
}

void make_graph_can_recompute1(ComputeGraphPtr &graph, bool create_cycle = false) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", DATA);
  x_desc->AddInputDesc(scalar_tensor);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto pow_a_desc = std::make_shared<OpDesc>("pow_a", POW);
  (void)ge::AttrUtils::SetBool(pow_a_desc, "_recompute", true);
  pow_a_desc->AddInputDesc(scalar_tensor);
  pow_a_desc->AddOutputDesc(scalar_tensor);
  auto pow_a_node = graph->AddNode(pow_a_desc);

  auto pow_b_desc = std::make_shared<OpDesc>("pow_b", POW);
  pow_b_desc->AddInputDesc(scalar_tensor);
  pow_b_desc->AddOutputDesc(scalar_tensor);
  auto pow_b_node = graph->AddNode(pow_b_desc);

  auto pow_desc0 = std::make_shared<OpDesc>("pow", POW);
  (void)ge::AttrUtils::SetBool(pow_desc0, "_recompute", true);
  pow_desc0->AddInputDesc(scalar_tensor);
  pow_desc0->AddOutputDesc(scalar_tensor);
  auto pow_node0 = graph->AddNode(pow_desc0);

  auto pow_desc1 = std::make_shared<OpDesc>("pow_1", POW);
  (void)ge::AttrUtils::SetBool(pow_desc1, "_recompute", true);
  pow_desc1->AddInputDesc(scalar_tensor);
  pow_desc1->AddOutputDesc(scalar_tensor);
  auto pow_node1 = graph->AddNode(pow_desc1);

  auto sqrt_desc0 = std::make_shared<OpDesc>("sqrt", SQRT);
  sqrt_desc0->AddInputDesc(scalar_tensor);
  sqrt_desc0->AddOutputDesc(scalar_tensor);
  auto sqrt_node0 = graph->AddNode(sqrt_desc0);

  auto sqrt_desc1 = std::make_shared<OpDesc>("gradients/sqrt", SQRT);
  (void)ge::AttrUtils::SetBool(sqrt_desc1, "_backward", true);
  sqrt_desc1->AddInputDesc(scalar_tensor);
  sqrt_desc1->AddOutputDesc(scalar_tensor);
  auto sqrt_node1 = graph->AddNode(sqrt_desc1);

  auto pow_desc2 = std::make_shared<OpDesc>("gradients/pow_1", POW);
  (void)ge::AttrUtils::SetBool(pow_desc2, "_backward", true);
  pow_desc2->AddInputDesc(scalar_tensor);
  pow_desc2->AddOutputDesc(scalar_tensor);
  auto pow_node2 = graph->AddNode(pow_desc2);

  auto addn_desc = std::make_shared<OpDesc>("gradients/AddN", ADDN);
  (void)ge::AttrUtils::SetBool(addn_desc, "_backward", true);
  addn_desc->AddInputDesc(scalar_tensor);
  addn_desc->AddInputDesc(scalar_tensor);
  addn_desc->AddInputDesc(scalar_tensor);
  addn_desc->AddOutputDesc(scalar_tensor);
  auto addn_node = graph->AddNode(addn_desc);

  auto output_desc = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  output_desc->AddInputDesc(scalar_tensor);
  output_desc->AddOutputDesc(scalar_tensor);
  auto output_node = graph->AddNode(output_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), pow_a_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_a_node->GetOutDataAnchor(0), pow_b_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_b_node->GetOutDataAnchor(0), pow_node0->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node0->GetOutDataAnchor(0), pow_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node0->GetOutDataAnchor(0), addn_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node1->GetOutDataAnchor(0), sqrt_node0->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node1->GetOutDataAnchor(0), pow_node2->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node2->GetOutDataAnchor(0), addn_node->GetInDataAnchor(1));
  if (create_cycle) {
    (void)GraphUtils::AddEdge(pow_node2->GetOutControlAnchor(), sqrt_node1->GetInControlAnchor());
  }
  (void)GraphUtils::AddEdge(sqrt_node0->GetOutDataAnchor(0), sqrt_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(sqrt_node1->GetOutDataAnchor(0), addn_node->GetInDataAnchor(2));
  (void)GraphUtils::AddEdge(addn_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
}

/**
 *       Data
 *         |
 *       Relu    Const
 *         \      /
 *          Switch
 *           |   \
 *           |    Relu
 *           |    /
 *           Merge
 *          /     \
 *        Relu   Relu
 *          |     |
 *         NetOutput
 */
Graph BuildSwitchMergeGraph() {
  GeTensorDesc tensor_4_desc(ge::GeShape({2,3,4,5}), FORMAT_NCHW, DT_INT32);

  auto data1 = std::make_shared<OpDesc>("data1", DATA);
  data1->AddInputDesc(tensor_4_desc);
  data1->AddOutputDesc(tensor_4_desc);

  auto relu1 = std::make_shared<OpDesc>("relu1", RELU);
  relu1->AddInputDesc(tensor_4_desc);
  relu1->AddOutputDesc(tensor_4_desc);

  int32_t data_value_vec1[1] = {1};
  GeTensorDesc data_tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->NODE(relu1)->EDGE(0, 0)->NODE("switch", SWITCH)->EDGE(0, 0)->NODE("merge", MERGE)
          ->EDGE(0, 0)->NODE("relu3", RELU)->NODE("output", NETOUTPUT));
    CHAIN(NODE("switch")->EDGE(1, 0)->NODE("relu2", RELU)->EDGE(0, 1)->NODE("merge")->EDGE(1, 0)
          ->NODE("relu4", RELU)->NODE("output"));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("switch"));
  };
  return ToGeGraph(g1);
}

Graph BuildGraphWithHcclOrderNode() {
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu1");
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu2");
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu3");
  auto allreduce1 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).Attr("group", "group_a").Build("allreduce1");

  auto allreduce2 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).Attr("group", "group_a").Build("allreduce2");

  auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(0).OutCnt(1).Build("var1");
  auto var2 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(0).OutCnt(1).Build("var2");
  auto identity = OP_CFG(READVARIABLEOP).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(1).OutCnt(1).Build("read_var");

  auto data1 = OP_CFG(DATA).Build("data1");
  auto broadcast1 = OP_CFG(HCOMBROADCAST).InCnt(1).OutCnt(1).Attr("group", "group_a").Build("broadcast1");

  auto data2 = OP_CFG(DATA).Build("data2");
  auto broadcast2 = OP_CFG(HCOMBROADCAST).InCnt(1).OutCnt(1).Attr("group", "group_a").Build("broadcast2");

  auto broadcast3 = OP_CFG(HCOMBROADCAST).InCnt(2).OutCnt(2).
      Attr(ATTR_NAME_MODIFY_INPUT, true).
      Attr("group", "group_a").
      Attr(ATTR_NAME_CONTINUOUS_INPUT, true).Build("broadcast3");
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add1");
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add2");

  auto allgather1 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape).
      Attr(ATTR_NAME_MODIFY_INPUT, true).Attr("group", "group_a").Build("allgather1");
  auto allgather2 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape).
      Attr(ATTR_NAME_MODIFY_INPUT, true).Attr("group", "group_a").Build("allgather2");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(allreduce1)->EDGE(0, 0)->NODE(allgather1));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(allreduce2)->EDGE(0, 0)->NODE(allgather2));
    CHAIN(NODE(allgather1)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(allgather2)->EDGE(0, 1)->NODE(add1));
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

Graph BuildGraphWithHcclNode() {

  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT)
      .Weight(data_tensor)
      .InCnt(0)
      .OutCnt(1)
      .Build("const1");

  // input mutable
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu1");
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu2");
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu3");
  auto allreduce1 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).Attr("group", "group_a").Attr("_input_mutable", true).Build("allreduce1");

  auto allreduce2 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).Attr("group", "group_a").Attr("_input_mutable", true).Build("allreduce2");

  auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(0).OutCnt(1).Build("var1");
  auto var2 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(0).OutCnt(1).Build("var2");
  auto identity = OP_CFG(READVARIABLEOP).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(1).OutCnt(1).Build("read_var");

  auto data1 = OP_CFG(DATA).Build("data1");
  auto assign1 = OP_CFG(ASSIGN).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(2).OutCnt(1).Build("data1_Assign");
  auto broadcast1 = OP_CFG(HCOMBROADCAST).InCnt(1).OutCnt(1).Attr("group", "group_a").Attr("_input_mutable", true).Build("broadcast1");

  auto data2 = OP_CFG(DATA).Build("data2");
  auto broadcast2 = OP_CFG(HCOMBROADCAST).InCnt(1).OutCnt(1).Attr("group", "group_a").Attr("_input_mutable", true).Build("broadcast2");

  auto split = OP_CFG("SplitD").InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2,3,4,5})
      .Attr("split_dim", 0)
      .Attr("num_split", 2).Build("split");

  // input continuous
  auto data3 = OP_CFG(DATA).Build("data3");
  auto var3 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2,3,4,5}).InCnt(0).OutCnt(1).Build("var3");
  vector<int64_t> mem_type = {0x11, 0x11};
  auto broadcast3 = OP_CFG(HCOMBROADCAST).InCnt(2).OutCnt(2).
                    Attr(ATTR_NAME_MODIFY_INPUT, true).
                    Attr("group", "group_a").
                    Attr(ATTR_NAME_CONTINUOUS_INPUT, true).
                    Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type).Build("broadcast3");

  // allgather with buffer pool
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor1 = GenerateTensor(shape);
  auto const3 = OP_CFG(CONSTANT).Weight(data_tensor1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
                    .InCnt(0).OutCnt(1).Build("const3");
  auto w1 = OP_CFG(CONSTANT).Weight(data_tensor1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
      .InCnt(0).OutCnt(1).Build("w1");
  auto w2 = OP_CFG(CONSTANT).Weight(data_tensor1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
      .InCnt(0).OutCnt(1).Build("w2");
  auto w3 = OP_CFG(CONSTANT).Weight(data_tensor1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
      .InCnt(0).OutCnt(1).Build("w3");
  auto w4 = OP_CFG(CONSTANT).Weight(data_tensor1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
      .InCnt(0).OutCnt(1).Build("w4");

  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add1");
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add2");
  auto add3 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add3");
  auto add4 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add4");

  auto allgather1 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape).
    Attr(ATTR_NAME_MODIFY_INPUT, true).Attr("group", "group_a").
    Attr(ATTR_NAME_BUFFER_POOL_ID, 0).
    Attr(ATTR_NAME_BUFFER_POOL_SIZE, 5600).Build("allgather1");
  auto allgather2 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape).
      Attr(ATTR_NAME_MODIFY_INPUT, true).Attr("group", "group_a").
      Attr(ATTR_NAME_BUFFER_POOL_ID, 1).
      Attr(ATTR_NAME_BUFFER_POOL_SIZE, 2560).Build("allgather2");
  auto allgather3 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape).
      Attr(ATTR_NAME_MODIFY_INPUT, true).Attr("group", "group_a").
      Attr(ATTR_NAME_BUFFER_POOL_ID, 0).
      Attr(ATTR_NAME_BUFFER_POOL_SIZE, 5600).Build("allgather3");
  auto allgather4 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape).
      Attr(ATTR_NAME_MODIFY_INPUT, true).Attr("group", "group_a").
      Attr(ATTR_NAME_BUFFER_POOL_ID, 0).
      Attr(ATTR_NAME_BUFFER_POOL_SIZE, 5600).Build("allgather4");

  auto partitioned_call = OP_CFG(PARTITIONEDCALL).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape);
  auto net_output = OP_CFG(NETOUTPUT).InCnt(1).Build("sub_output");

  DEF_GRAPH(sub1) {
    auto sub_data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    CHAIN(NODE(w4)->EDGE(0, 0)->NODE(allgather4)->EDGE(0, 1)->NODE(add4)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE("sub_data", sub_data)->EDGE(0, 0)->NODE(add4));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)->NODE(allreduce1));
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE(split));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(split));
    CHAIN(NODE(split)->EDGE(0, 0)->NODE(relu1));
    CHAIN(NODE(split)->EDGE(0, 0)->NODE(allreduce1));

    CHAIN(NODE(var2)->EDGE(0, 0)->NODE(assign1)->NODE(broadcast1));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(var2)->EDGE(0, 0)->NODE(broadcast2));
    CHAIN(NODE(var2)->EDGE(0, 0)->NODE(identity));
    CHAIN(NODE(identity)->NODE(relu3)->NODE("partitioned_call", partitioned_call, sub1)->NODE("relu5", RELU));
    CHAIN(NODE(identity)->CTRL_EDGE()->NODE(assign1));

    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(broadcast3));
    CHAIN(NODE(var3)->EDGE(0, 1)->NODE(broadcast3));
    CHAIN(NODE(var3)->NODE(relu2));

    CHAIN(NODE(w1)->EDGE(0, 0)->NODE(allgather1)->EDGE(0, 1)->NODE(add1));
    CHAIN(NODE(w2)->EDGE(0, 0)->NODE(allgather2)->EDGE(0, 1)->NODE(add2));
    CHAIN(NODE(w3)->EDGE(0, 0)->NODE(allgather3)->EDGE(0, 1)->NODE(add3));

    CHAIN(NODE(const3)->EDGE(0, 0)->NODE(add1)->EDGE(0, 0)->NODE(add2)->EDGE(0, 0)->NODE(add3));

  };

  auto sub_graph = ToComputeGraph(sub1);
  const auto netoutput = sub_graph->FindNode("sub_output");
  const auto output_desc = netoutput->GetOpDesc();
  AttrUtils::SetInt(output_desc->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto root_graph = ToComputeGraph(g1);
  const auto allgather_node1 = root_graph->FindNode("allgather1");
  const auto allgather_node2 = root_graph->FindNode("allgather2");
  const auto allgather_node3 = root_graph->FindNode("allgather3");
  EXPECT_TRUE(allgather_node1 != nullptr);
  TensorUtils::SetSize(*allgather_node1->GetOpDesc()->MutableOutputDesc(0), 1536);
  EXPECT_TRUE(allgather_node2 != nullptr);
  TensorUtils::SetSize(*allgather_node2->GetOpDesc()->MutableOutputDesc(0), 1536);
  EXPECT_TRUE(allgather_node3 != nullptr);
  TensorUtils::SetSize(*allgather_node3->GetOpDesc()->MutableOutputDesc(0), 1536);
  const auto allgather_node4 = sub_graph->FindNode("allgather4");
  EXPECT_TRUE(allgather_node4 != nullptr);
  TensorUtils::SetSize(*allgather_node4->GetOpDesc()->MutableOutputDesc(0), 1536);

  auto graph = ToGeGraph(g1);
  return graph;
}

Graph BuildRWGraph() {
  return BuildGraphWithHcclNode();
}

Graph BuildRWGraph2() {
  GeTensorDesc tensor_4_desc(ge::GeShape({2, 3, 4, 5}), FORMAT_NCHW, DT_INT32);
  auto var1 = std::make_shared<OpDesc>("var1", VARIABLE);
  var1->AddInputDesc(tensor_4_desc);
  var1->AddOutputDesc(tensor_4_desc);
  int32_t data_value_vec1[1] = {1};
  GeTensorDesc data_tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)
              ->EDGE(0, 0)
              ->NODE("read_var", READVARIABLEOP)
              ->EDGE(0, 0)
              ->NODE("mul", MUL)
              ->EDGE(0, 0)
              ->NODE("output", NETOUTPUT));
    CHAIN(NODE(var1)->EDGE(0, 1)->NODE("mul"));
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE("assgin_var", ASSIGNVARIABLEOP)->CTRL_EDGE()->NODE("mul"));  // write then read
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("assgin_var"));
    CHAIN(NODE("read_var")->CTRL_EDGE()->NODE("assgin_var"));  // read then write
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

Graph BuildWRGraph1() {
  GeTensorDesc tensor_4_desc(ge::GeShape({2,3,4,5}), FORMAT_NCHW, DT_INT32);
  auto var1 = std::make_shared<OpDesc>("var1", VARIABLE);
  var1->AddInputDesc(tensor_4_desc);
  var1->AddOutputDesc(tensor_4_desc);
  auto var2 = std::make_shared<OpDesc>("var2", VARIABLE);
  var2->AddInputDesc(tensor_4_desc);
  var2->AddOutputDesc(tensor_4_desc);
  int32_t data_value_vec1[1] = {1};
  GeTensorDesc data_tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);


  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)
              ->EDGE(0, 0)
              ->NODE("read_var", READVARIABLEOP)
              ->EDGE(0, 0)
              ->NODE("mul", MUL)
              ->EDGE(0, 0)
              ->NODE("output", NETOUTPUT));
    CHAIN(NODE(var2)->EDGE(0, 1)->NODE("mul"));
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE("assgin_var", ASSIGNVARIABLEOP)->CTRL_EDGE()->NODE("read_var"));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("assgin_var"));
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

Graph BuildWRGraph2() {
  GeTensorDesc tensor_4_desc(ge::GeShape({2,3,4,5}), FORMAT_NCHW, DT_INT32);
  auto var1 = std::make_shared<OpDesc>("var1", VARIABLE);
  var1->AddInputDesc(tensor_4_desc);
  var1->AddOutputDesc(tensor_4_desc);
  auto var2 = std::make_shared<OpDesc>("var2", VARIABLE);
  var2->AddInputDesc(tensor_4_desc);
  var2->AddOutputDesc(tensor_4_desc);
  int32_t data_value_vec1[1] = {1};
  GeTensorDesc data_tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)
              ->EDGE(0, 0)
              ->NODE("read_var", READVARIABLEOP)
              ->EDGE(0, 0)
              ->NODE("mul", MUL)
              ->EDGE(0, 0)
              ->NODE("output", NETOUTPUT));
    CHAIN(NODE(var2)->EDGE(0, 1)->NODE("mul"));
    CHAIN(NODE(var1)
              ->EDGE(0, 0)
              ->NODE("assgin_var", ASSIGNVARIABLEOP)
              ->CTRL_EDGE()
              ->NODE(var2)
              ->CTRL_EDGE()
              ->NODE("read_var"));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("assgin_var"));
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

class RuntimeMock910A : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
    (void)strcpy_s(version, maxLen, "Ascend910A");
    return RT_ERROR_NONE;
  }
};

class RuntimeMock910B1 : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
    (void)strcpy_s(version, maxLen, "Ascend910B1");
    return RT_ERROR_NONE;
  }
  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "2201");
    return RT_ERROR_NONE;
  }
};

void SetTensorSize(const GeShape &shape,
                   const Format format,
                   const DataType data_type,
                   GeTensorDesc &tensor_desc) {
  int64_t tensor_size = 0;
  TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  TensorUtils::SetSize(tensor_desc, tensor_size);
}

void UpdateGraphTensorSize(ComputeGraphPtr &graph) {
  for (auto &node : graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      SetTensorSize(input->GetShape(), input->GetFormat(), input->GetDataType(), *input);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      SetTensorSize(output->GetShape(), output->GetFormat(), output->GetDataType(), *output);
    }
  }
}
}

static void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    ge::OpKernelBinPtr tbeKernelPtr = std::make_shared<ge::OpKernelBin>("test_tvm", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("AIcoreEngine", aicore_func);
}

static void MockAIcoreEngineEnGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  MockForGenerateTask("AIcoreEngine", aicore_func);
}

class GraphCompilerTest : public testing::Test {
  void SetUp() {
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    const std::vector<rtMemType_t> mem_type{RT_MEMORY_HBM, RT_MEMORY_P2P_DDR};
    (void) ge::MemManager::Instance().Initialize(mem_type);
    ge::PlatformContext::GetInstance().SetPlatform("2201");
    MockGenerateTask();
  }
  void TearDown() {
    char runtime2_env[MMPA_MAX_PATH] = {'1'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    graph_optimizer_option = kGraphOptimizerOption::kNormal;
    hybrid::NpuMemoryAllocator::Finalize();
    ge::MemManager::Instance().Finalize();
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("AIcoreEngine");
    unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
  }
public:
  void SetFakerBuilder() {
    ge::OpsKernelBuilderRegistry::GetInstance().Unregister("DNN_VM_GE_LOCAL_OP_STORE");
    REGISTER_OPS_KERNEL_BUILDER("DNN_VM_GE_LOCAL_OP_STORE", FakeOpsKernelBuilder);
  }
  void SetGeLocalBuilder() {
    ge::OpsKernelBuilderRegistry::GetInstance().Unregister("DNN_VM_GE_LOCAL_OP_STORE");
    REGISTER_OPS_KERNEL_BUILDER("DNN_VM_GE_LOCAL_OP_STORE", ge::ge_local::GeLocalOpsKernelBuilder);
  }
};

/**
 *      data1  data2
 *         \   /
 *          Add
 */
Graph CreateGraphForTestStorageFormat(int64_t dim) {
  const auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 16, 224, 224})
      .Build("data1");
  const auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 16, 224, 224})
      .Build("data2");
  const auto add = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 16, 16, 224, 224})
      .Build("add1");
  const auto net_output = OP_CFG(NETOUTPUT)
      .InCnt(1)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("NetOutput");

  (void)AttrUtils::SetInt(data1->MutableInputDesc(0), ATTR_NAME_STORAGE_FORMAT, FORMAT_NC1HWC0);
  (void)AttrUtils::SetInt(data2->MutableInputDesc(0), ATTR_NAME_STORAGE_FORMAT, FORMAT_NC1HWC0);
  if (dim != (int64_t)0x9FFFFFFFFFFFFFFFUL) {
    (void)AttrUtils::SetListInt(data1->MutableInputDesc(0), ATTR_NAME_STORAGE_SHAPE, {16, dim, 224, 224, 16});
  }
  (void)AttrUtils::SetListInt(data2->MutableInputDesc(0), ATTR_NAME_STORAGE_SHAPE, {16, 1, 224, 224, 16});

  (void)AttrUtils::SetInt(net_output->MutableInputDesc(0), ATTR_NAME_STORAGE_FORMAT, FORMAT_NC1HWC0);
  (void)AttrUtils::SetListInt(net_output->MutableInputDesc(0), ATTR_NAME_STORAGE_SHAPE, {16, 1, 224, 224, 16});

  std::vector<std::pair<int64_t, int64_t>> origin_range({{1,16}, {16, 16}, {224, 224}, {224, 224}});
  data1->MutableInputDesc(0)->SetOriginShapeRange(origin_range);

  data1->SetOpEngineName("DNN_VM_GE_LOCAL");
  data1->SetOpKernelLibName(kEngineNameGeLocal);
  data2->SetOpEngineName("DNN_VM_GE_LOCAL");
  data2->SetOpKernelLibName(kEngineNameGeLocal);
  add->SetOpEngineName("DNN_VM_GE_LOCAL");
  add->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->SetOpEngineName("DNN_VM_GE_LOCAL");
  net_output->SetOpKernelLibName(kEngineNameGeLocal);

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));
  };

  return ToGeGraph(g1);
}

/**
 *      data   data
 *        \    /|
 *         add  |
 *           \  |
 *            add
 */
TEST_F(GraphCompilerTest, test_build_no_tiling_fail) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 224, 224})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list);
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 224, 224})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_MAX_SHAPE, "1, 10, 224, 224");
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 224, 224});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 224, 224});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  node->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange({{1, 1}, {1, -1}, {224, 224}, {224, 224}});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  (void)session.BuildGraph(1, inputs);

  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
    EXPECT_EQ(graph->GetDirectNodesSize(), 5);
    for (auto node : graph->GetAllNodes()) {
      if (node->GetName() == "add_1" || node->GetName() == "add_2") {
        bool is_no_tiling = false;
        EXPECT_EQ(
          AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_OP_NO_TILING, is_no_tiling),
          true);
        EXPECT_TRUE(is_no_tiling == false);
      }
    }
  };
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

/**
 *      data  data
 *        \   / |
 *         add  |
 *           \  |
 *            add
 */
TEST_F(GraphCompilerTest, test_build_no_tiling) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 224, 224})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_MAX_SHAPE, "1, 10, 224, 224");
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 224, 224})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_MAX_SHAPE, "20, 10, 224, 224");
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto print = OP_CFG("Print");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_2", add2)->EDGE(0, 0)->NODE("Print", print));
  };
  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
    EXPECT_EQ(graph->GetDirectNodesSize(), 5);
    for (auto node : graph->GetAllNodes()) {
      if (node->GetName() == "add_1" || node->GetName() == "add_2") {
        bool is_no_tiling = false;
        EXPECT_EQ(
          AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_OP_NO_TILING, is_no_tiling),
          true);
        // need expect true, but this case can not construct unknown shape of add_1 and add_2
      }
    }
  };
}

/**
 *      data  data     data    data
 *        \   /          /      /
 *         customop1    /      /
 *             |       /     /
 *            customop2    /
 *             |         /
 *             |       /
 *             add
 */
TEST_F(GraphCompilerTest, test_build_with_null_output) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("customop");
  op_impl_func->NullableOutput(1);
  op_impl_func->NullableOutput(0);

  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto customop1 = OP_CFG("customop").TensorDesc(FORMAT_NCHW, DT_BOOL, {1, 2, 3, 4})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_MAX_SHAPE, "1, 10, 224, 224")
                         .InCnt(2).OutCnt(2).InNames({"x", "y"})
                         .OutNames({"u", "v"});
  auto customop2 = OP_CFG("customop").TensorDesc(FORMAT_NCHW, DT_BOOL, {1, 2, 3, 4})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_MAX_SHAPE, "20, 10, 224, 224")
                         .InCnt(2).OutCnt(2).InNames({"x", "y"})
                         .OutNames({"u", "v"});

  auto add3 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 2, 3, 4})
                         .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                         .Attr(ATTR_NAME_OP_MAX_SHAPE, "20, 10, 224, 224");

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  auto data4 = OP_CFG(DATA);
  auto print = OP_CFG("Print");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("c_1", customop1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("c_1", customop1));
    CHAIN(NODE("c_1", customop1)->EDGE(0, 0)->NODE("c_2", customop2));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("c_2", customop2));
    CHAIN(NODE("c_2", customop2)->EDGE(1, 0)->NODE("add_3", add3));
    CHAIN(NODE("data_4", data4)->EDGE(0, 1)->NODE("add_3", add3));
    CHAIN(NODE("add_3", add3)->EDGE(0, 0)->NODE("Print", print));
  };
  auto graph = ToGeGraph(g1);

  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto c_1_node = root_graph->FindNode("c_1");
  c_1_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
  c_1_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  c_1_node->GetOpDesc()->AppendIrInput("x", IrInputType::kIrInputRequired);
  c_1_node->GetOpDesc()->AppendIrInput("y", IrInputType::kIrInputRequired);
  c_1_node->GetOpDesc()->AppendIrOutput("u", IrOutputType::kIrOutputRequired);
  c_1_node->GetOpDesc()->AppendIrOutput("v", IrOutputType::kIrOutputRequired);
  c_1_node->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"y", 1}};
  c_1_node->GetOpDesc()->MutableAllOutputName() = {{"u", 0}, {"v", 1}};

  auto c_2_node = root_graph->FindNode("c_2");
  c_2_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
  c_2_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  c_2_node->GetOpDesc()->AppendIrInput("x", IrInputType::kIrInputRequired);
  c_2_node->GetOpDesc()->AppendIrInput("y", IrInputType::kIrInputRequired);
  c_2_node->GetOpDesc()->AppendIrOutput("u", IrOutputType::kIrOutputRequired);
  c_2_node->GetOpDesc()->AppendIrOutput("v", IrOutputType::kIrOutputRequired);
  c_2_node->GetOpDesc()->MutableAllInputName() = {{"x", 0}, {"y", 1}};
  c_2_node->GetOpDesc()->MutableAllOutputName() = {{"u", 0}, {"v", 1}};

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllNodesSize(), 8);
    EXPECT_EQ(graph->GetDirectNodesSize(), 8);
    for (auto node : graph->GetAllNodes()) {
      bool is_null_output = true;
      if (node->GetName() == "c_1") {
        bool is_null_output = false;
        auto output_tensor_desc_0 = node->GetOpDesc()->GetOutputDesc(0);
        EXPECT_EQ(
          ge::AttrUtils::GetBool(output_tensor_desc_0, ATTR_NAME_IS_NULL_OUTPUT, is_null_output), false);
        auto output_tensor_desc_1 = node->GetOpDesc()->GetOutputDesc(1);
        EXPECT_EQ(
          ge::AttrUtils::GetBool(output_tensor_desc_1, ATTR_NAME_IS_NULL_OUTPUT, is_null_output), true);
        EXPECT_EQ(is_null_output, true);
      }
      if (node->GetName() == "c_2") {
        bool is_null_output = false;
        auto output_tensor_desc_0= node->GetOpDesc()->GetOutputDesc(0);
        EXPECT_EQ(
          ge::AttrUtils::GetBool(output_tensor_desc_0, ATTR_NAME_IS_NULL_OUTPUT, is_null_output), true);
        EXPECT_EQ(is_null_output, true);
        auto output_tensor_desc_1 = node->GetOpDesc()->GetOutputDesc(1);
        EXPECT_EQ(
          ge::AttrUtils::GetBool(output_tensor_desc_1, ATTR_NAME_IS_NULL_OUTPUT, is_null_output), false);
      }
    }
  };
}

TEST_F(GraphCompilerTest, test_build_with_engine_partition_with_attr_group_copyed) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 224, 224})
                  .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                  .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                  .Attr(ATTR_NAME_OP_MAX_SHAPE, "1, 10, 224, 224");
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 224, 224})
                  .Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list)
                  .Attr(ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engine_list)
                  .Attr(ATTR_NAME_OP_MAX_SHAPE, "20, 10, 224, 224");
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto print = OP_CFG("Print");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_2", add2)->EDGE(0, 0)->NODE("Print", print));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  // todoo: graph属性组的序列化在metadef中未实现，当前先不校验图，待metadef合入后，校验图中是否有属性组。
}

TEST_F(GraphCompilerTest, test_build_super_kernel) {
  MockAIcoreEngineEnGenerateTask();
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr("_super_kernel_scope", "scope1");

  auto relu = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_2", add2)->EDGE(0, 0)->NODE("relu", relu));
  };
  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_super_kernel_strict_check) {
  MockAIcoreEngineEnGenerateTask();
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  auto relu = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_2", add2)->EDGE(0, 0)->NODE("relu", relu));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto add1_node = compute_graph->FindNode("add_1");
  EXPECT_NE(add1_node, nullptr);
  AttrUtils::SetStr(add1_node->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  auto relu_node = compute_graph->FindNode("relu");
  EXPECT_NE(relu_node, nullptr);
  AttrUtils::SetStr(relu_node->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.CompileGraph(1);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetStr(add1_node->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(relu_node->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_select_sk_non_hccl_stream) {
  MockAIcoreEngineEnGenerateTask();
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kNormal);
  DEF_GRAPH(g1) {
    const auto hcom1 = OP_CFG(HCOMALLGATHER).Attr("_super_kernel_scope", "scope1");
    const auto relu = OP_CFG(RELU).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("hcom1", hcom1)->EDGE(0, 0)->NODE("relu", relu)->EDGE(0, 0)->
          NODE("net_output", NETOUTPUT));

  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  auto hcom_1 = compute_graph->FindNode("hcom1");
  AttrUtils::SetStr(hcom_1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "2");
  AttrUtils::SetBool(hcom_1->GetOpDesc(), "_hccl", true);
  auto relu = compute_graph->FindNode("relu");
  AttrUtils::SetStr(relu->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "1");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_sk_append_ws) {
  MockAIcoreEngineEnGenerateTask();
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
    .Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto print = OP_CFG("Print");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_2", add2)->EDGE(0, 0)->NODE("Print", print));
  };
  auto graph = ToGeGraph(g1);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);
  auto add_1 = compute_graph->FindNode("add_1");
  std::vector<int64_t> append_ws_vec{100,200};
  AttrUtils::SetListInt(add_1->GetOpDesc(), "_append_ws", append_ws_vec);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_append_ws) {
  MockAIcoreEngineEnGenerateTask();
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto print = OP_CFG("Print");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_2", add2)->EDGE(0, 0)->NODE("Print", print));
  };
  auto graph = ToGeGraph(g1);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);
  auto add_1 = compute_graph->FindNode("add_1");
  std::vector<int64_t> append_ws_vec{100,200};
  AttrUtils::SetListInt(add_1->GetOpDesc(), "_append_ws", append_ws_vec);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_super_kernel_cmo) {
  MockAIcoreEngineEnGenerateTask();
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kNormal);

  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_2 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_2 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_2 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_3 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto dequant_3 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto batch_matmul_3 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");

    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("matmul_1", matmul_1)->EDGE(0, 0)->
        NODE("dequant_1", dequant_1)->EDGE(0, 0)->
        NODE("batch_matmul_1", batch_matmul_1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("matmul_2", matmul_2)->EDGE(0, 0)->
        NODE("dequant_2", dequant_2)->EDGE(0, 0)->
        NODE("batch_matmul_2", batch_matmul_2)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data3", DATA)->EDGE(0, 0)->NODE("matmul_3", matmul_3)->EDGE(0, 0)->
        NODE("dequant_3", dequant_3)->EDGE(0, 0)->
        NODE("batch_matmul_3", batch_matmul_3)->EDGE(0, 2)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("batch_matmul_1")->CTRL_EDGE()->NODE("matmul_2"));

    CHAIN(NODE("batch_matmul_2")->CTRL_EDGE()->NODE("matmul_3"));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");

  AttrUtils::SetStr(matmul_1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "1");
  AttrUtils::SetStr(dequant_1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "1");
  AttrUtils::SetStr(batch_matmul_1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "1");

  auto matmul_2 = compute_graph->FindNode("matmul_2");
  auto dequant_2 = compute_graph->FindNode("dequant_2");
  auto batch_matmul_2 = compute_graph->FindNode("batch_matmul_2");

  AttrUtils::SetStr(matmul_2->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "2");
  AttrUtils::SetStr(dequant_2->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "2");
  AttrUtils::SetStr(batch_matmul_2->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "2");

  auto matmul_3 = compute_graph->FindNode("matmul_3");
  auto dequant_3 = compute_graph->FindNode("dequant_3");
  auto batch_matmul_3 = compute_graph->FindNode("batch_matmul_3");

  AttrUtils::SetStr(matmul_3->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "3");
  AttrUtils::SetStr(dequant_3->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "3");
  AttrUtils::SetStr(batch_matmul_3->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "3");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;

  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_partiton_stable_topo) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  vector<float> value{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 2, 2}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(value.data()), sizeof(float) * value.size());
  auto const3 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).Weight(const_tensor1);
  vector<float> value2{0, 6, 1, 1};
  GeTensorPtr const_tensor2 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(value2.data()), sizeof(float) * value2.size());
  auto const2 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).Weight(const_tensor2);
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast4 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu7 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu8 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu9 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu10 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu11 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu12 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});

  auto add1 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto add2 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto add3 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto variable1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto ref_data = OP_CFG(REFDATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2}).Attr("ref_var_src_var_name", "variable1");
  auto add4 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1)->EDGE(0, 0)->
        NODE("relu1", relu1)->EDGE(0, 0)->NODE("relu2", relu2)->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("data1", data0)->EDGE(0, 0)->NODE("cast2", cast2)->EDGE(0, 0)->
        NODE("relu4", relu4)->EDGE(0, 0)->NODE("relu5", relu5)->EDGE(0, 0)->NODE("relu6", relu6)->
        EDGE(0, 0)->NODE("add1", add1)->EDGE(0, 0)->NODE("add3", add3));
    CHAIN(NODE("const2", const2)->EDGE(0, 0)->NODE("cast3", cast3)->EDGE(0, 0)->
        NODE("relu7", relu7)->EDGE(0, 0)->NODE("relu8", relu8)->EDGE(0, 0)->NODE("relu9", relu9));
    CHAIN(NODE("const3", const3)->EDGE(0, 0)->NODE("cast4", cast4)->EDGE(0, 0)->
        NODE("relu10", relu10)->EDGE(0, 0)->NODE("relu11", relu11)->EDGE(0, 0)->NODE("relu12", relu12)->
        EDGE(0, 0)->NODE("add2", add2)->EDGE(0, 1)->NODE("add3", add3)->NODE("add4", add4));
    CHAIN(NODE("relu3")->EDGE(0, 1)->NODE("add1"));
    CHAIN(NODE("relu9")->EDGE(0, 1)->NODE("add2"));
    CHAIN(NODE("variable1", variable1)->EDGE(0, 0)->NODE("ref_data", ref_data)->EDGE(0, 1)->NODE("add4"));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_tmp = compute_graph->FindNode("add3");
  auto op_desc = node_tmp->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, ATTR_INPUT_MEMORY_TYPE, RT_MEMORY_HBM);

  compute_graph->TopologicalSorting();
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      node->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
      node->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
    }
  }
  map<AscendString, AscendString> options = {{OPTION_TOPOSORTING_MODE, "3"}};
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    std::map<std::string, std::set<std::string>> subgraph_name_to_node_name_map = {
        {"g1_sub_1_input", {"const3", "const2"}},
        {"g1_sub_0_unknow", {"relu4", "relu5", "relu6", "relu1", "relu2", "relu3", "add1"}},
        {"g1_sub_2_know", {"relu10", "relu11", "relu12", "relu7", "relu8", "relu9", "add2"}},
        {"g1_sub_3_unknow", {"add3", "variable1", "add4"}}};
    ASSERT_EQ(graph->GetAllSubgraphs().size(), 4);
    for (auto subgraph : graph->GetAllSubgraphs()) {
      std::cout << "subgraph name: " << subgraph->GetName() << std::endl;
      int32_t subgraph_node_num = 0;
      for (auto sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          std::cout << "sub_node name: " << sub_node->GetName() << std::endl;
          auto it = subgraph_name_to_node_name_map[subgraph->GetName()].find(sub_node->GetName());
          subgraph_node_num++;
          EXPECT_NE(it, subgraph_name_to_node_name_map[subgraph->GetName()].end());
        }
      }
      EXPECT_EQ(subgraph_node_num, subgraph_name_to_node_name_map[subgraph->GetName()].size());
    }
  };
}

TEST_F(GraphCompilerTest, test_graph_with_transdata) {
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("transdata_graph");
  BuildGraphHasTrans(graph);
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  auto ret = graph_manager.OptimizeStage1(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS); // topo success, no cycle
  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
}

TEST_F(GraphCompilerTest, SetAllowMultiGraphParallelCompileTrue_Check_CloseVarialbePass) {
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("transdata_graph");
  BuildGraphHasTrans(graph);
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  OptionSetter setter{{OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE, "1"}};
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();
  auto ret = graph_manager.OptimizeStage1(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto log_check = stub.GetSlogStub().FindInfoLogRegex(
    "get option ge.AllowMultiGraphParallelCompile = \"1\", turn off VariableOpPass");
  EXPECT_NE(log_check, -1);
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS); // topo success, no cycle
  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
}

// merge为动态，会被识别为no tiling节点，走静态图
TEST_F(GraphCompilerTest, test_build_no_tiling_01) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list);
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list);
  auto less = OP_CFG(LESS).TensorDesc(FORMAT_NCHW, DT_BOOL, {}).Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto switch1 = OP_CFG(SWITCH);
  auto switch2 = OP_CFG(SWITCH);
  auto switch3 = OP_CFG(SWITCH);
  auto identity1 = OP_CFG(IDENTITY).TensorDesc(FORMAT_NCHW, DT_BOOL, {}).InCnt(1).OutCnt(1);
  auto identity2 = OP_CFG(IDENTITY).TensorDesc(FORMAT_NCHW, DT_BOOL, {}).InCnt(1).OutCnt(1);
  GeTensor weight;
  std::vector<uint8_t> data = {1, 2, 3, 4};
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape(std::vector<int64_t>({})));
  weight.SetTensorDesc(weight_desc);
  auto constant1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto constant2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);

  auto merge = OP_CFG(MERGE).TensorDesc(FORMAT_NCHW, DT_INT32, {-2});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("less", less));
    CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("less", less));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("switch1", switch1));
    CHAIN(NODE("less", less)->EDGE(0, 1)->NODE("switch1", switch1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("switch2", switch2));
    CHAIN(NODE("less", less)->EDGE(0, 1)->NODE("switch2", switch2));
    CHAIN(NODE("less", less)->EDGE(0, 0)->NODE("switch3", switch3));
    CHAIN(NODE("less", less)->EDGE(0, 1)->NODE("switch3", switch3));
    CHAIN(NODE("switch3", switch3)->EDGE(0, 0)->NODE("identity1", identity1));
    CHAIN(NODE("switch3", switch3)->EDGE(1, 0)->NODE("identity2", identity2));
    CHAIN(NODE("identity1", identity1)->CTRL_EDGE()->NODE("constant1", constant1));
    CHAIN(NODE("identity2", identity2)->CTRL_EDGE()->NODE("constant2", constant2));
    CHAIN(NODE("switch1", switch1)->EDGE(0, 0)->NODE("add1", add1));
    CHAIN(NODE("constant1", constant1)->EDGE(0, 1)->NODE("add1", add1));
    CHAIN(NODE("switch2", switch2)->EDGE(0, 0)->NODE("add2", add2));
    CHAIN(NODE("constant2", constant2)->EDGE(0, 1)->NODE("add2", add2));
    CHAIN(NODE("add1", add1)->EDGE(0, 0)->NODE("merge", merge));
    CHAIN(NODE("add2", add2)->EDGE(0, 1)->NODE("merge", merge));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto merge_node = compute_graph->FindNode("merge");
  auto merge_node_out_td = merge_node->GetOpDesc()->MutableOutputDesc(0);
  std::vector<std::vector<int64_t>> shape_range{{1, 100}};
  const std::string TENSOR_UTILS_SHAPE_RANGE = "shape_range";
  (void)AttrUtils::SetListListInt(merge_node_out_td, TENSOR_UTILS_SHAPE_RANGE, shape_range);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  auto ret = session.CompileGraph(1);
  ASSERT_EQ(ret, SUCCESS);
  ret = session.LoadGraph(1, {}, nullptr);
  ASSERT_EQ(ret, SUCCESS);
  auto summary = session.GetCompiledGraphSummary(1);
  ASSERT_NE(summary, nullptr);
  EXPECT_EQ(summary->IsStatic(), true);
  EXPECT_EQ(ret, SUCCESS);
}

// merge为静态图走静态
TEST_F(GraphCompilerTest, test_build_no_tiling_02) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list);
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list);
  auto less = OP_CFG(LESS).TensorDesc(FORMAT_NCHW, DT_BOOL, {}).Attr(ATTR_NAME_OP_TILING_INLINE_ENGINE, engine_list);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto switch1 = OP_CFG(SWITCH);
  auto switch2 = OP_CFG(SWITCH);
  auto switch3 = OP_CFG(SWITCH);
  auto identity1 = OP_CFG(IDENTITY).TensorDesc(FORMAT_NCHW, DT_BOOL, {}).InCnt(1).OutCnt(1);
  auto identity2 = OP_CFG(IDENTITY).TensorDesc(FORMAT_NCHW, DT_BOOL, {}).InCnt(1).OutCnt(1);
  GeTensor weight;
  std::vector<uint8_t> data = {1, 2, 3, 4};
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape(std::vector<int64_t>({})));
  weight.SetTensorDesc(weight_desc);
  auto constant1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto constant2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);

  auto merge = OP_CFG(MERGE).TensorDesc(FORMAT_NCHW, DT_INT32, {1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("less", less));
    CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("less", less));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("switch1", switch1));
    CHAIN(NODE("less", less)->EDGE(0, 1)->NODE("switch1", switch1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("switch2", switch2));
    CHAIN(NODE("less", less)->EDGE(0, 1)->NODE("switch2", switch2));
    CHAIN(NODE("less", less)->EDGE(0, 0)->NODE("switch3", switch3));
    CHAIN(NODE("less", less)->EDGE(0, 1)->NODE("switch3", switch3));
    CHAIN(NODE("switch3", switch3)->EDGE(0, 0)->NODE("identity1", identity1));
    CHAIN(NODE("switch3", switch3)->EDGE(1, 0)->NODE("identity2", identity2));
    CHAIN(NODE("identity1", identity1)->CTRL_EDGE()->NODE("constant1", constant1));
    CHAIN(NODE("identity2", identity2)->CTRL_EDGE()->NODE("constant2", constant2));
    CHAIN(NODE("switch1", switch1)->EDGE(0, 0)->NODE("add1", add1));
    CHAIN(NODE("constant1", constant1)->EDGE(0, 1)->NODE("add1", add1));
    CHAIN(NODE("switch2", switch2)->EDGE(0, 0)->NODE("add2", add2));
    CHAIN(NODE("constant2", constant2)->EDGE(0, 1)->NODE("add2", add2));
    CHAIN(NODE("add1", add1)->EDGE(0, 0)->NODE("merge", merge));
    CHAIN(NODE("add2", add2)->EDGE(0, 1)->NODE("merge", merge));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto merge_node = compute_graph->FindNode("merge");
  auto merge_node_out_td = merge_node->GetOpDesc()->MutableOutputDesc(0);
  std::vector<std::vector<int64_t>> shape_range{{1, 100}};
  const std::string TENSOR_UTILS_SHAPE_RANGE = "shape_range";
  (void)AttrUtils::SetListListInt(merge_node_out_td, TENSOR_UTILS_SHAPE_RANGE, shape_range);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  auto ret = session.CompileGraph(1);
  ASSERT_EQ(ret, SUCCESS);
  ret = session.LoadGraph(1, {}, nullptr);
  ASSERT_EQ(ret, SUCCESS);
  auto summary = session.GetCompiledGraphSummary(1);
  ASSERT_NE(summary, nullptr);
  EXPECT_EQ(summary->IsStatic(), true);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_ffts_plus) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr dsp_graph;
  ComputeGraphPtr ffts_graph;
  BuildFftsDynamicGraph(root_graph, dsp_graph, ffts_graph);
  SetFftsPlusAttrPass set_ffts_plus_attr_pass;
  Status ret = set_ffts_plus_attr_pass.Run(ffts_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_ffts_inner_no_reuse_plus) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr dsp_graph;
  ComputeGraphPtr ffts_graph;
  BuildFftsGraph(root_graph, dsp_graph, ffts_graph);
  root_graph->TopologicalSorting();
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  EXPECT_EQ(graph_mem_assigner.AssignMemory(), SUCCESS);
  TaskGenerator task_generate;
  const auto dsp_graph_call_0 = dsp_graph->FindNode("dsp_graph/PartitionedCall_0");
  EXPECT_NE(dsp_graph_call_0, nullptr);
  std::vector<domi::TaskDef> task_def_list_per_node;
  EXPECT_NE(task_generate.GenerateTaskForFftsNode(dsp_graph_call_0.get(), "test", task_def_list_per_node,
                                                  GetThreadLocalContext(), error_message::GetErrMgrContext(), 0),
            SUCCESS);
}

static void BuildContainUnSupportZerocopyNodeGraph(ComputeGraphPtr &root_graph, const std::string &name) {
  const auto SetOpSize = [](const ComputeGraph::Vistor<NodePtr> &all_nodes, int64_t size) {
    const static std::set<std::string> kGeLocalTypes{ DATA, CONSTANT, VARIABLE, NETOUTPUT, AIPP_DATA_TYPE };
    GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
    TensorUtils::SetSize(tensor, size);
    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      std::string op_kernel_name =  (kGeLocalTypes.count(op_desc->GetType()) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE";
      op_desc->SetOpKernelLibName(op_kernel_name);
      for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
        op_desc->UpdateInputDesc(i, tensor);
        if (node->GetType() == NETOUTPUT && node->GetName() != NODE_NAME_NET_OUTPUT) {
          AttrUtils::SetInt(op_desc->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
        }
      }
      for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
        op_desc->UpdateOutputDesc(i, tensor);
      }
    }
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("add", ADD)->NODE("Node_Output", NETOUTPUT));
  };

  root_graph = ToComputeGraph(g1);
  SetOpSize(root_graph->GetDirectNode(), 1024);
  const auto op_desc = root_graph->FindNode("add")->GetOpDesc();
  op_desc->SetOpKernelLibName(name);
}

TEST_F(GraphCompilerTest, test_memassigner_unsupport_zerocopyblock_hccl) {
  ComputeGraphPtr root_graph;
  BuildContainUnSupportZerocopyNodeGraph(root_graph, ge::kEngineNameHccl.c_str());
  root_graph->TopologicalSorting();
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  EXPECT_EQ(graph_mem_assigner.AssignMemory(), SUCCESS);
}

TEST_F(GraphCompilerTest, test_memassigner_unsupport_zerocopyblock_dsa) {
  ComputeGraphPtr root_graph;
  BuildContainUnSupportZerocopyNodeGraph(root_graph, ge::kEngineNameDsa.c_str());
  root_graph->TopologicalSorting();
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  EXPECT_EQ(graph_mem_assigner.AssignMemory(), SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_assigner_memory) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("");
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  graph_mem_assigner.AssignMemory();

  OpDescPtr op_desc_one = std::make_shared<OpDesc>("node_one", "type");
  NodePtr node_one = compute_graph->AddNode(op_desc_one);
  ge::AttrUtils::SetBool(node_one->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  map<uint64_t, size_t> mem_type_to_offset  = {};
  auto ret = graph_mem_assigner.ReAssignMemory(mem_type_to_offset);
  EXPECT_EQ(ret, SUCCESS);

  OpDescPtr op_desc_two = std::make_shared<OpDesc>("node_two", "type");
  NodePtr node_two = compute_graph->AddNode(op_desc_two);
  std::vector<int64_t> mem_type_list;
  mem_type_list.emplace_back(66);
  ge::AttrUtils::SetListInt(node_two->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type_list);
  ge::AttrUtils::SetBool(node_two->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  ret = graph_mem_assigner.ReAssignContinuousMemory();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_multiple_output_continous_success) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({32,32});

  compute_graph->TopologicalSorting();
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  graph_mem_assigner.AssignMemory();

  map<uint64_t, size_t> mem_type_to_offset  = {};
  auto ret = graph_mem_assigner.ReAssignMemory(mem_type_to_offset);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_partition_mark_support_addr_refresh) {
  constexpr char_t kIsSupportAddrRefresh[] = "_is_support_addr_refresh";
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {10,10});

    auto neg = OP_CFG(NEG)
        .InCnt(1)
        .OutCnt(1)
        .Attr(kIsSupportAddrRefresh, false)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10,10})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto neg1 = OP_CFG(NEG)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10,10})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .InputAttr(0, ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("data1", data1)->NODE("neg", neg)->NODE("neg1", neg1)->NODE("net_output", net_output));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto neg_node = compute_graph->FindNode("neg");
  ASSERT_NE(neg_node, nullptr);

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.CompileGraph(0);
  EXPECT_EQ(ret, SUCCESS);

  bool is_force_unknown = false;
  EXPECT_EQ(AttrUtils::GetBool(neg_node->GetOpDescBarePtr(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, is_force_unknown), true);
  EXPECT_EQ(is_force_unknown, true);
}

TEST_F(GraphCompilerTest, test_multiple_output_to_single_in_continous_success) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                   .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                   .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                   .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  std::vector<int64_t> memtype_list_h2 = {RT_MEMORY_HBM};
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                   .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list_h2)
                   .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list_h2)
                   .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("hcom_1", hcom2)->EDGE(1, 1)->NODE("net_output", NETOUTPUT));
  };

  GeShape shape({10, 32});
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto data1_node = compute_graph->FindNode("data_1");
  data1_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(shape);
  data1_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetOriginShape(shape);

  auto data2_node = compute_graph->FindNode("data_2");
  data2_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(shape);
  data2_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetOriginShape(shape);

  auto hcom_node1 = compute_graph->FindNode("hcom_1");
  auto op_desc = hcom_node1->GetOpDesc();
  op_desc->SetWorkspace({0, 0});
  op_desc->SetWorkspaceBytes({32, 32});
  op_desc->MutableInputDesc(0)->SetShape(shape);
  op_desc->MutableInputDesc(1)->SetShape(shape);
  op_desc->MutableOutputDesc(0)->SetShape(shape);
  op_desc->MutableOutputDesc(1)->SetShape(shape);
  op_desc->MutableInputDesc(0)->SetOriginShape(shape);
  op_desc->MutableInputDesc(1)->SetOriginShape(shape);
  op_desc->MutableOutputDesc(0)->SetOriginShape(shape);
  op_desc->MutableOutputDesc(1)->SetOriginShape(shape);
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 512);
  TensorUtils::SetSize(*op_desc->MutableInputDesc(1), 512);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 512);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(1), 512);

  auto hcom_node2 = compute_graph->FindNode("hcom_2");
  hcom_node2->GetOpDescBarePtr()->MutableInputDesc(0)->SetShape(shape);
  hcom_node2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(shape);
  TensorUtils::SetSize(*hcom_node2->GetOpDescBarePtr()->MutableInputDesc(0), 512);
  TensorUtils::SetSize(*hcom_node2->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
  hcom_node2->GetOpDescBarePtr()->MutableInputDesc(0)->SetOriginShape(shape);
  hcom_node2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetOriginShape(shape);
  compute_graph->TopologicalSorting();

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto hcom1 = graph->FindNode("hcom_1");
    ASSERT_NE(hcom1, nullptr);
    auto hcom2 = graph->FindNode("hcom_2");
    ASSERT_NE(hcom2, nullptr);
    EXPECT_EQ(hcom1->GetOpDescBarePtr()->GetOutputOffset()[0U], hcom2->GetOpDescBarePtr()->GetInputOffset()[0U]);
    std::size_t identity_nums{0UL};
    for (const auto &node : graph->GetAllNodesPtr()) {
      if (node->GetType() == IDENTITY) {
        ++identity_nums;
      }
    }
    EXPECT_EQ(identity_nums, 4UL);
  };
}

TEST_F(GraphCompilerTest, test_dynamic_batch_continous_success) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_BATCH_LABEL, "Batch_0")
      .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_BATCH_LABEL, "Batch_1")
      .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
      CHAIN(NODE("data_2", data2)->EDGE(1, 1)->NODE("hcom_2", hcom2));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 0)->NODE("data_4", data4));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  UpdateGraphTensorSize(compute_graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({32,32});

  compute_graph->TopologicalSorting();
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  graph_mem_assigner.AssignMemory();

  map<uint64_t, size_t> mem_type_to_offset  = {};
  auto ret = graph_mem_assigner.ReAssignMemory(mem_type_to_offset);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_multiple_output_continous_idx_err) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({32,32});

  compute_graph->TopologicalSorting();
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  graph_mem_assigner.AssignMemory();

  auto ret = graph_mem_assigner.ReAssignContinuousMemory();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_graph_memory_assign_fail_case) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("");
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  MemoryOffset mem_offset(2, 65UL * 1024UL * 1024UL *1024UL);
  graph_mem_assigner.memory_offset_.insert({2, mem_offset});
  VarManager::Instance(0)->use_max_mem_size_ = 0;

  map<uint64_t, size_t> mem_type_to_offset = {};
  Status ret = graph_mem_assigner.ReAssignMemory(mem_type_to_offset);
  EXPECT_EQ(ret, ACL_ERROR_GE_MEMORY_ALLOCATION);
}

TEST_F(GraphCompilerTest, test_build_graph_accord_stage_success) {
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;
  ComputeGraphPtr compute_graph = nullptr;

  Status ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_SYS, false, model_buffer,
                                            GraphStage::GRAPH_STAGE_FUZZ, compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  int64_t graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
  bool graph_has_been_added = false;
  // test attr has been cleared
  EXPECT_EQ(AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage), false);
  EXPECT_EQ(AttrUtils::GetBool(compute_graph, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, graph_has_been_added), false);

}

TEST_F(GraphCompilerTest, InitializeUserSetJitCompileFalseOn910A) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910A>());
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  std::map<std::string, std::string> options_map = {{JIT_COMPILE, "0"}};
  GetThreadLocalContext().SetGlobalOption(options_map);
  GeGenerator generator;
  generator.Initialize({});
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "0");
}

TEST_F(GraphCompilerTest, InitializeUserSetJitCompileTrueOn910B1) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910B1>());
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  std::map<std::string, std::string> options_map = {{JIT_COMPILE, "1"}};
  GetThreadLocalContext().SetGlobalOption(options_map);
  GeGenerator generator;
  generator.Initialize(options_map);
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "1");
}

TEST_F(GraphCompilerTest, InitializeCheckJitCompileDefaultValueOn910B1) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910B1>());
  GELib::Initialize({});
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "2");
}

TEST_F(GraphCompilerTest, InitializeCheckJitCompileDefaultValueOn910A) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910A>());
  std::map<std::string, std::string> options_map;
  options_map["ge.exec.rankId"] = "1";
  options_map["ge.exec.rankTableFile"] = "./";
  GELib::Initialize(options_map);
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "2");
}

TEST_F(GraphCompilerTest, initialize_valid_SyncTimeout) {
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT, "100"});
  options.insert({ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT, "200"});
  GELib::Initialize(options);
  EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), 100);
  EXPECT_EQ(ge::GetContext().EventSyncTimeout(), 200);
}

TEST_F(GraphCompilerTest, initialize_aicore_num) {
  const char *const kVectorcoreNum = "ge.vectorcoreNum";
  std::map<std::string, std::string> options;
  options.insert({ge::AICORE_NUM, "2|3"});
  GELib::Initialize(options);
  std::string aicore_num;
  ge::GetContext().GetOption(ge::AICORE_NUM, aicore_num);
  EXPECT_STREQ(aicore_num.c_str(), "2");

  std::string vector_core_num;
  ge::GetContext().GetOption(kVectorcoreNum, vector_core_num);
  EXPECT_STREQ(vector_core_num.c_str(), "3");
}

TEST_F(GraphCompilerTest, initialize_invalid_SyncTimeout) {
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT, "123456789987654321"});
  options.insert({ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT, "123456789987654321"});
  GELib::Initialize(options);
  EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), -1);
  EXPECT_EQ(ge::GetContext().EventSyncTimeout(), -1);
}

void FakeMultiDimsEngine(GeRunningEnvFaker &ge_env) {
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.InstallDefault();
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
}

TEST_F(GraphCompilerTest, test_subgraph_multi_dims) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  auto sub_data_1 = OP_CFG(DATA).Attr("index", 0)
                                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  auto sub_data_2 = OP_CFG(DATA).Attr("index", 1)
                                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto slice = OP_CFG(SLICE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  int32_t data_value_vec1[2] = {0, 0};
  int32_t data_value_vec2[2] = {2, 2};
  GeTensorDesc data_tensor_desc(GeShape({2}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, 2 * sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);
  GeTensorPtr data_tensor2 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec2, 2 * sizeof(int32_t));
  auto const2 = OP_CFG(CONSTANT).Weight(data_tensor2);
  auto sub_add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  auto sub_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  DEF_GRAPH(sub1) {
    CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
    CHAIN(NODE("sub_data_2", sub_data_2)->EDGE(0, 0)->NODE("slice", slice));
    CHAIN(NODE("const_1", const1)->EDGE(0, 1)->NODE("slice"));
    CHAIN(NODE("const_2", const2)->EDGE(0, 2)->NODE("slice"));
    CHAIN(NODE("slice")->EDGE(0, 1)->NODE("sub_add")->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
    CHAIN(NODE("sub_data_2", sub_data_2)->EDGE(0, 1)->NODE("sub_add"));
  };
  auto partitioned_call = OP_CFG(PARTITIONEDCALL).InCnt(2).OutCnt(1)
                                                 .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1})
                                                 .Attr(ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto data_3 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  auto add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1})
                        .Attr(ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0)
                        .Attr(ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0: 2, -1; 1: 2,-1")
                        .Attr(ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "2, 2; 4, 4; 8, 8");
  auto cast = OP_CFG(CAST).TensorDesc(FORMAT_NCHW, DT_INT32, {2, -1})
                          .Attr(ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  auto net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2,2});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data_1)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("data_2", data_2)->EDGE(0, 1)->NODE("add"));
    CHAIN(NODE("data_3", data_3)->EDGE(0, 0)->NODE("partitioned_call", partitioned_call, sub1));
    CHAIN(NODE("add")->EDGE(0, 1)->NODE("partitioned_call")->EDGE(0, 0)->NODE("cast", cast));
    CHAIN(NODE("cast")->EDGE(0, 0)->NODE("net_output", net_output));
  };
  sub1.Layout();
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add");
  const auto add_infer_func = [](Operator &op) {
    op.GetOutputDesc(0).SetShape(op.GetInputDesc(0).GetShape());
    return GRAPH_SUCCESS;
  };
  node->GetOpDesc()->AddInferFunc(add_infer_func);

  node = compute_graph->FindNode("cast");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, ATTR_INPUT_MEMORY_TYPE, RT_MEMORY_HBM);

  map<AscendString, AscendString> options = {{"ge.runFlag", "0"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  int data1[2][2] = {{12, 23}, {34, 45}};
  InputTensorInfo input_1 = {3, {2, 2}, data1, 16};
  int data2[2][2] = {{1, 2}, {2, 3}};
  InputTensorInfo input_2 = {3, {2, 2}, data2, 16};
  int data3[2][2] = {{3, 3}, {4, 4}};
  InputTensorInfo input_3 = {3, {2, 2}, data3, 16};
  inputs.push_back(input_1);
  inputs.push_back(input_2);
  inputs.push_back(input_3);
  auto ret = session.BuildGraph(1, inputs);

  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 5);
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 5);
  };

  inputs.clear();
  inputs.push_back(input_1);
  inputs.push_back(input_2);
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, 0);
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(GraphCompilerTest, VariableInDynamicSubGraph_Build_Success) {
  const auto old_level = ge::SlogStub::GetInstance()->GetLevel();
  ge::SlogStub::GetInstance()->SetLevel(DLOG_INFO);
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  // sub_graph1 is unknow sub graph
  auto transdata = OP_CFG(TRANSDATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  auto sub1_data = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto sub1_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  DEF_GRAPH(sub1) {
    CHAIN(NODE("sub1_variable", VARIABLE)->EDGE(0, 0)->NODE("sub1_transdata", transdata)->
          NODE("sub1_add", ADD)->NODE("sub1_netoutput", sub1_net_output));
    CHAIN(NODE("sub1_data", sub1_data)->EDGE(0, 1)->NODE("sub1_add", ADD));
  };

  // sub_graph2 is known sub graph
  auto sub2_data1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  auto sub2_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
  DEF_GRAPH(sub2) {
    CHAIN(NODE("sub2_data1", sub2_data1)->EDGE(0, 0)->NODE("sub2_add", ADD)->NODE("sub2_netoutput", sub2_net_output));
    CHAIN(NODE("sub2_variable", VARIABLE)->EDGE(0, 1)->NODE("sub2_add", ADD));
  };

  // root graph
  auto partitioned_call1 = OP_CFG(PARTITIONEDCALL).InCnt(1).OutCnt(1)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto partitioned_call2 = OP_CFG(PARTITIONEDCALL).InCnt(1).OutCnt(1)
          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, -1});
  auto net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2,2});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data_1)->EDGE(0, 0)->NODE("partitioned_call1", partitioned_call1, sub1)->
          NODE("partitioned_call2", partitioned_call2, sub2)->NODE("net_output", net_output));
  };
  sub1.Layout();
  sub2.Layout();

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  for (const auto &subgraph : compute_graph->GetAllSubgraphs()) {
    for (const auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() == NETOUTPUT) {
        AttrUtils::SetInt(node->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
      }
    }
  }

  map<AscendString, AscendString> options = {{"ge.runFlag", "0"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  int data1[2][2] = {{12, 23}, {34, 45}};
  InputTensorInfo input_1 = {3, {2, 2}, data1, 16};
  inputs.push_back(input_1);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  ge_env.Reset();
  ge_env.InstallDefault();
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
  ge::SlogStub::GetInstance()->SetLevel(old_level);
}

Status ConstructSession(std::shared_ptr<Session> &session) {
  std::vector<InputTensorInfo> inputs;
  map<AscendString, AscendString> options{
        {OPTION_GRAPH_RUN_MODE, "1"}};
  auto graph1 = BuildSwitchMergeGraph();
  uint32_t session_id = 1;
  session = std::make_shared<Session>(options);
  if (session == nullptr) {
    return FAILED;
  }
  session->AddGraph(session_id, graph1, options);
  auto ret = session->BuildGraph(session_id, inputs);
  if (ret != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

TEST_F(GraphCompilerTest, ReleasePhysicalMemoryWhenSessionFinalize_Success_MultiSession) {
  class MockRuntime : public ge::RuntimeStub {
  public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) override {
      ++alloc_count;
      *handle = (rtDrvMemHandle) new uint8_t[8];
      return 0;
    }
    rtError_t rtFreePhysical(rtDrvMemHandle handle) override {
      ++free_count;
      delete[] (uint8_t *)handle;
      return 0;
    }
    uint32_t alloc_count = 0U;
    uint32_t free_count = 0U;
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  std::shared_ptr<Session> session1 = nullptr;
  std::shared_ptr<Session> session2 = nullptr;
  ASSERT_EQ(ConstructSession(session1), SUCCESS);
  ASSERT_EQ(ConstructSession(session2), SUCCESS);

  const auto occupy_num = mock_runtime->alloc_count - mock_runtime->free_count;
  ASSERT_NE(occupy_num, 0);
  ASSERT_NE(mock_runtime->alloc_count, 0);
  ASSERT_EQ(mock_runtime->free_count, 0);
  for (size_t i = 0; i < 5; i++) {
    session2.reset();
    ASSERT_EQ(ConstructSession(session2), SUCCESS);
    session1.reset();
    ASSERT_EQ(ConstructSession(session1), SUCCESS);
  }
  // 校验物理内存没有增长
  EXPECT_EQ(occupy_num, mock_runtime->alloc_count - mock_runtime->free_count);
  session1.reset();
  session2.reset();

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(GraphCompilerTest, test_dynamic_stack_handle_and_engine) {
  int32_t max_size = 100;
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor =
      std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&max_size), sizeof(int32_t));
  const auto const_op = OP_CFG(CONSTANT).OutCnt(1).Weight(const_tensor);
  const auto stack = OP_CFG(STACK).InCnt(1).OutCnt(1).Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  const auto stack_push = OP_CFG(STACKPUSH).InCnt(2).OutCnt(1).Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  const auto stack_pop = OP_CFG(STACKPOP).InCnt(1).OutCnt(1).Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  const auto const_op_1 = OP_CFG(CONSTANT).OutCnt(1).Weight(const_tensor);

  DEF_GRAPH(g1) {
    CHAIN(NODE("const", const_op)->EDGE(0, 0)->NODE("stack", stack));

    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stack_push", stack_push));
    CHAIN(NODE("const", const_op)->EDGE(0, 1)->NODE("stack_push", stack_push));

    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stack_pop", stack_pop));
    CHAIN(NODE("stack_pop", stack_pop)->EDGE(0, 0)->NODE("add", ADD));
    CHAIN(NODE("const_1", const_op_1)->EDGE(0, 1)->NODE("add", ADD));
  };

  const auto graph = ToGeGraph(g1);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  static const std::unordered_set<std::string> kDataFlowOps = {STACK, STACKPUSH, STACKPOP, STACKCLOSE};
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (kDataFlowOps.count(node->GetType()) != 0UL) {
      node->GetOpDesc()->SetOpEngineName("DNN_VM_AICPU");
      node->GetOpDesc()->SetOpKernelLibName("DNN_VM_AICPU_ASCEND");
    }
  }

  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), AscendString("12800"));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  // 1. check handle
  const int64_t kStackHandle = 0;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (kDataFlowOps.count(node->GetType()) != 0UL) {
      int64_t handle = -1;
      const bool ret = AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, handle);
      EXPECT_TRUE(ret);
      EXPECT_EQ(handle, kStackHandle);
    }
  }

  // 2. check engine
  static const char_t *const kGeLocalEngineName = "DNN_VM_GE_LOCAL";
  static const char_t *const kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (kDataFlowOps.count(node->GetType()) != 0UL) {
      EXPECT_EQ(node->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
      EXPECT_EQ(node->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
    }
  }
}

TEST_F(GraphCompilerTest, test_control_triggle_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  const auto x_op = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_BOOL, {});
  const auto y_op = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_BOOL, {});
  const auto z_op = OP_DATA(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {});

  const auto switch_op = OP_CFG(SWITCH).InCnt(2).OutCnt(2);
  const auto switch_op2 = OP_CFG(SWITCH).InCnt(2).OutCnt(2);
  const auto identity_op = OP_CFG(IDENTITY).InCnt(1).OutCnt(1);
  const auto identity_op2 = OP_CFG(IDENTITY).InCnt(1).OutCnt(1);
  const auto control_trigger_op = OP_CFG(CONTROLTRIGGER);

  DEF_GRAPH(g1) {
    CHAIN(NODE("x_op", x_op)->EDGE(0, 1)->NODE("switch_op", switch_op));
    CHAIN(NODE("z_op", z_op)->EDGE(0, 0)->NODE("switch_op", switch_op));
    CHAIN(NODE("y_op", y_op)->EDGE(0, 1)->NODE("switch_op2", switch_op2));
    CHAIN(NODE("z_op", z_op)->EDGE(0, 0)->NODE("switch_op2", switch_op2));

    CHAIN(NODE("switch_op", switch_op)->EDGE(0, 0)->NODE("identity_op", identity_op));
    CHAIN(NODE("switch_op2", switch_op2)->EDGE(0, 0)->NODE("identity_op2", identity_op2));
    CTRL_CHAIN(NODE("identity_op", identity_op)->NODE("control_trigger_op", control_trigger_op));
    CTRL_CHAIN(NODE("identity_op2", identity_op2)->NODE("control_trigger_op", control_trigger_op));
  };
  const auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_data_flow_process) {
 auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto add_3 = OP_CFG(ADD);
  auto add_4 = OP_CFG(ADD).Attr("_force_unknown_shape", true);
  auto add_5 = OP_CFG(ADD).Attr("_force_unknown_shape", true);
  auto add_6 = OP_CFG(ADD).Attr("_force_unknown_shape", true);
  auto stack = OP_CFG("Stack");
  auto stackpush = OP_CFG("StackPush");
  auto stackpop = OP_CFG("StackPop");
  auto stack1 = OP_CFG("Stack");
  auto stackpush1 = OP_CFG("StackPush").Attr("_force_unknown_shape", true);
  auto stackpop1 = OP_CFG("StackPop").Attr("_force_unknown_shape", true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto op_ptr = OP_CFG(DATA)
    .InCnt(1)
    .OutCnt(1)
    .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
    .Attr("compile_info_key", "ddd")
    .Attr("compile_info_json", "cccc")
    .Attr("_force_unknown_shape", true)
    .Build("data3");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)
          ->NODE("add_2", add_2)->EDGE(0, 0)->NODE("add_3", add_3)
          ->NODE("add_4", add_4)->EDGE(0, 0)->NODE("add_5", add_5)
          ->NODE("add_6", add_6));

    CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE(op_ptr)->EDGE(0, 1)->NODE("add_4", add_4));
    CHAIN(NODE(op_ptr)->EDGE(0, 1)->NODE("add_5", add_5));

    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackpush", stackpush));
    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackpop", stackpop));
    CHAIN(NODE("add_1", add_1)->EDGE(0, 1)->NODE("stackpush", stackpush));
    CHAIN(NODE("stackpop", stackpop)->EDGE(0, 1)->NODE("add_3", add_3));
    CHAIN(NODE("stack1", stack)->EDGE(0, 0)->NODE("stackpush1", stackpush1));
    CHAIN(NODE("stack1", stack)->EDGE(0, 0)->NODE("stackpop1", stackpop1));
    CHAIN(NODE("add_4", add_4)->EDGE(0, 1)->NODE("stackpush1", stackpush1));
    CHAIN(NODE("stackpop1", stackpop1)->EDGE(0, 1)->NODE("add_6", add_6));

  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  for (auto &node : compute_graph->GetAllNodes()) {
    if (node->GetName() == "stack" || node->GetName() == "stackpush" || node->GetName() == "stackpop") {
      (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, 1);
    }
    if (node->GetName() == "stack1" || node->GetName() == "stackpush1" || node->GetName() == "stackpop1") {
      (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, 2);
    }
  }
  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph_2, options);
  std::vector<InputTensorInfo> inputs;
  EXPECT_NE(session.BuildGraph(1, inputs), ge::SUCCESS);

}

TEST_F(GraphCompilerTest, test_switch_dead_branch_merge_pass) {
  Graph graph = BuildSwitchMergeGraph();

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 7);
    auto switch1 = graph->FindNode("switch");
    EXPECT_EQ(switch1, nullptr);
    auto merge1 = graph->FindNode("merge");
    EXPECT_EQ(merge1, nullptr);
  };
}

TEST_F(GraphCompilerTest, test_origin_hccl_order) {
  Graph graph = BuildGraphWithHcclOrderNode();
  const auto& compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();
  // new session & add graph
  map<AscendString, AscendString> options;
  options[ge::OPTION_TOPOSORTING_MODE] = "3";
  Session session(options);
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kNormal);
  DUMP_GRAPH_WHEN("PreRunAfterBuild", "PreRunBegin")
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunBegin) {
    auto allgather1 = graph->FindNode("allgather1");
    EXPECT_NE(allgather1, nullptr);
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    EXPECT_TRUE(allreduce2->GetOpDesc()->GetId() > allgather1->GetOpDesc()->GetId());
  };
  CHECK_GRAPH(PreRunAfterBuild) {
    auto allgather1 = graph->FindNode("allgather1");
    EXPECT_NE(allgather1, nullptr);
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    EXPECT_TRUE(allreduce2->GetOpDesc()->GetId() > allgather1->GetOpDesc()->GetId());
  };
}

/**
* 用例描述：验证变量初始化功能是否正常，当变量具有合法的初始值（init_value）时，是否能正确地将初始值拷贝到设备内存中，并且拷贝后的值与原始值一致。
*
* test_var_match
* \
* 变量初始化（具有 init_value 属性）
* \
* 变量内存分配与初始化
*
* 测试步骤：
* 1. 初始化内存管理器和 VarManager
* 2. 创建变量描述（shape、data_type、format）
* 3. 创建与变量描述一致的 init_value 张量
* 4. 将 init_value 设置为变量的属性
* 5. 创建 OpDesc 并分配变量内存
* 6. 获取变量的逻辑地址和设备地址
* 7. 触发 InitVarIfHasInitValue，将 init_value 拷贝到设备内存
*
* 预期结果：
* 1. 变量内存分配成功
* 2. GetVarMemoryAddr 返回有效的设备地址
* 3. 变量描述与 init_value 描述一致（shape、format、data_type）
* 4. InitVarIfHasInitValue 成功执行，rtMemcpy 被调用
*/
TEST_F(GraphCompilerTest, test_var_init_with_init_value) {
     // Initialize VarManager
    const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
    EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

    VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
    EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

    // Create variable tensor with placement device
    std::vector<int64_t> var_shape{1, 1, 1, 1, 10};
    GeShape shape(var_shape);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetDataType(DT_FLOAT);
    tensor_desc.SetFormat(FORMAT_NCHW);
    TensorUtils::SetSize(tensor_desc, 10 * sizeof(float));

    // Create init_value tensor with matching format and type
    std::vector<float> init_data(10, 1.0f);
    auto init_tensor = std::make_shared<GeTensor>();
    GeTensorDesc init_desc(GeShape(var_shape), FORMAT_NCHW, DT_FLOAT);
    init_tensor->SetData(reinterpret_cast<uint8_t*>(init_data.data()), init_data.size() * sizeof(float));
    init_tensor->MutableTensorDesc() = init_desc;

    // Set init_value attribute
    EXPECT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ATTR_NAME_INIT_VALUE, init_tensor));

    // Create OpDesc
    OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_match", VARIABLE);
    op_desc->AddOutputDesc(tensor_desc);

    // Assign variable memory
    std::string var_name = "test_var_match";
    Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
    EXPECT_EQ(status, SUCCESS);

    VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);

    // Get the variable logical address
    uint8_t* logic_addr = nullptr;
    status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_NE(logic_addr, nullptr);

    // Get GetVarMemoryAddr to trigger InitVarIfHasInitValue
    uint8_t* dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);
    EXPECT_NE(dev_addr, nullptr);

    // Cast dev_addr to float pointer to access the data
    float *device_data = reinterpret_cast<float *>(dev_addr);

    // Compare the data in device memory with the init_data
    for (size_t i = 0; i < init_data.size(); ++i) {
      EXPECT_FLOAT_EQ(device_data[i], init_data[i]);
    }
}

TEST_F(GraphCompilerTest, test_origin_hccl_unorder) {
  Graph graph = BuildGraphWithHcclOrderNode();
  const auto& compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();
  // new session & add graph
  map<AscendString, AscendString> options;
  options[ge::OPTION_TOPOSORTING_MODE] = "3";
  Session session(options);
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kNormal);
  DUMP_GRAPH_WHEN("PreRunAfterOptimizeSubgraph", "PreRunBegin")
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  test_origin_hcom_unordered = true;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  test_origin_hcom_unordered = false;
  CHECK_GRAPH(PreRunBegin) {
    auto allgather1 = graph->FindNode("allgather1");
    EXPECT_NE(allgather1, nullptr);
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    EXPECT_TRUE(allreduce2->GetOpDesc()->GetId() > allgather1->GetOpDesc()->GetId());
  };
  CHECK_GRAPH(PreRunAfterOptimizeSubgraph) {
    auto allgather1 = graph->FindNode("allgather1");
    EXPECT_NE(allgather1, nullptr);
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    EXPECT_TRUE(allreduce2->GetOpDesc()->GetId() < allgather1->GetOpDesc()->GetId());
  };
}

TEST_F(GraphCompilerTest, test_subgraph_hccl_unorder) {
  Graph graph = BuildGraphWithHcclOrderNode();
  const auto& compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();
  // new session & add graph
  map<AscendString, AscendString> options;
  options[ge::OPTION_TOPOSORTING_MODE] = "3";
  Session session(options);
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kNormal);
  DUMP_GRAPH_WHEN("PreRunAfterOptimizeSubgraph", "PreRunAfterBuild")
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  test_subgraph_hcom_unordered = true;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  test_subgraph_hcom_unordered = false;
  CHECK_GRAPH(PreRunAfterOptimizeSubgraph) {
    auto allgather1 = graph->FindNode("allgather1");
    EXPECT_NE(allgather1, nullptr);
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    EXPECT_TRUE(allreduce2->GetOpDesc()->GetId() > allgather1->GetOpDesc()->GetId());
  };
  CHECK_GRAPH(PreRunAfterBuild) {
    auto allgather1 = graph->FindNode("allgather1");
    EXPECT_NE(allgather1, nullptr);
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    EXPECT_TRUE(allreduce2->GetOpDesc()->GetId() < allgather1->GetOpDesc()->GetId());
  };
}

TEST_F(GraphCompilerTest, test_compile_var_read_first_then_write_var) {
  Graph graph = BuildRWGraph();
  const auto& compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_NE(compute_graph->FindNode("read_var"), nullptr);
  DUMP_GRAPH_WHEN("PrepareAfterProcessAippStage2", "PrepareAfterPrepareOptimize")
  // new session & add graph
  map<AscendString, AscendString> options{{ge::OPTION_TOPOSORTING_MODE, "3"}};
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // `graph` is a para of lambda
  std::size_t origin_size;
  CHECK_GRAPH(PrepareAfterProcessAippStage2) {
    EXPECT_NE(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("relu3");
    origin_size = read_node->GetOutControlNodesSize();
  };
  // `graph` is a para of lambda
  CHECK_GRAPH(PrepareAfterPrepareOptimize) {
    // read_var is removed
    EXPECT_EQ(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("relu3");
    // read_var's ctrl output moved to read_node
    EXPECT_NE(read_node, nullptr);
    auto new_out_ctrl_nodes = read_node->GetOutControlNodes();
    EXPECT_EQ(new_out_ctrl_nodes.size(), origin_size + 1U);
    auto write_node = graph->FindNode("data1_Assign");
    bool write_node_is_ctrled_by_read_node =
        std::any_of(new_out_ctrl_nodes.begin(), new_out_ctrl_nodes.end(),
                    [&write_node](const NodePtr &node) { return (node == write_node); });
    EXPECT_TRUE(write_node_is_ctrled_by_read_node);
  };
  // TODO check other node
}

// for var cache
/**
 *  .. means ctrl
 *    ┌─────────────────────────────────────────────────────────────────┐
      │                                                                 │
      │                                                                 │
      │    ┌·········································┐                  │ (0,0)
      │    :                                         ∨                  ∨
      │    :                   ┌────────┐  (0,1)   ┌────────────┐     ┌────────┐  (0,0)   ┌────────┐
      │    :                   │ const1 │ ───────> │ assgin_var │ ··> │  mul   │ ───────> │ output │
      │    :                   └────────┘          └────────────┘     └────────┘          └────────┘
      │    :                                         ∧                  ∧
      │    :                                         │                  │ (0,1)
      │    :                                         │                  │
      │  ┌──────────┐  (0,0)   ┌────────┐  (0,0)     │                  │
      └─ │ read_var │ <─────── │  var1  │ ───────────┘                  │
         └──────────┘          └────────┘                               │
                                 │                                      │
                                 └──────────────────────────────────────┘

 */
TEST_F(GraphCompilerTest, test_compile_var_read_first_then_write_var_then_read_var) {
  Graph graph = BuildRWGraph2();
  const auto &compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_NE(compute_graph->FindNode("read_var"), nullptr);
  DUMP_GRAPH_WHEN("PrepareAfterProcessAippStage2", "PrepareAfterPrepareOptimize")
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // `graph` is a para of lambda
  std::size_t origin_size;
  CHECK_GRAPH(PrepareAfterProcessAippStage2) {
    EXPECT_NE(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("mul");
    origin_size = read_node->GetInControlNodesSize();
  };
  // `graph` is a para of lambda
  CHECK_GRAPH(PrepareAfterPrepareOptimize) {
    // read_var is for cache, can not remove
    auto read_variable_op = graph->FindNode("read_var");
    EXPECT_NE(read_variable_op, nullptr);
    auto read_node = graph->FindNode("mul");
    EXPECT_NE(read_node, nullptr);
    auto new_in_ctrl_nodes = read_node->GetInControlNodes();
    EXPECT_EQ(new_in_ctrl_nodes.size(), origin_size);
    auto write_node = graph->FindNode("assgin_var");
    EXPECT_NE(write_node, nullptr);
    bool write_node_ctrl_read_node = std::any_of(new_in_ctrl_nodes.begin(), new_in_ctrl_nodes.end(),
                                                 [&write_node](const NodePtr &node) { return (node == write_node); });
    EXPECT_TRUE(write_node_ctrl_read_node);
    auto write_node_in_ctrl_nodes = write_node->GetInControlNodes();
    bool write_node_is_ctrled_by_identity_node =
        std::any_of(write_node_in_ctrl_nodes.begin(), write_node_in_ctrl_nodes.end(),
                    [&read_variable_op](const NodePtr &node) { return (node == read_variable_op); });
    EXPECT_TRUE(write_node_is_ctrled_by_identity_node);
  };
  // TODO check other node
}

/**
 *                                            g1

┌────────┐  (0,1)   ┌────────────┐ (ctrl)   ┌──────────┐  (0,0)   ┌────────┐  (0,0)   ┌────────┐
│ const1 │ ───────> │ assgin_var │ ·······> │ read_var │ ───────> │  mul   │ ───────> │ output │
└────────┘          └────────────┘          └──────────┘          └────────┘          └────────┘
                     ∧                       ∧                     ∧
                      │ (0,0)                 │                     │ (0,1)
                      │                       │                     │
                    ┌────────────┐  (0,0)     │                   ┌────────┐
                    │    var1    │ ───────────┘                   │  var2  │
                    └────────────┘                                └────────┘

 */
TEST_F(GraphCompilerTest, test_compile_write_var_first_then_read_var1) {
  Graph graph = BuildWRGraph1();
  const auto &compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_NE(compute_graph->FindNode("read_var"), nullptr);
  DUMP_GRAPH_WHEN("PrepareAfterProcessAippStage2", "PrepareAfterPrepareOptimize")
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // `graph` is a para of lambda
  std::size_t origin_size;
  CHECK_GRAPH(PrepareAfterProcessAippStage2) {
    EXPECT_NE(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("mul");
    origin_size = read_node->GetInControlNodesSize();
  };
  // `graph` is a para of lambda
  CHECK_GRAPH(PrepareAfterPrepareOptimize) {
    // read_var is removed
    EXPECT_EQ(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("mul");
    // read_var's ctrl in moved to read_node
    EXPECT_NE(read_node, nullptr);
    auto new_in_ctrl_nodes = read_node->GetInControlNodes();
    EXPECT_EQ(new_in_ctrl_nodes.size(), origin_size + 1U);
    auto write_node = graph->FindNode("assgin_var");
    bool read_node_is_direct_ctrled_by_write_node =
        std::any_of(new_in_ctrl_nodes.begin(), new_in_ctrl_nodes.end(),
                    [&write_node](const NodePtr &node) { return (node == write_node); });
    EXPECT_TRUE(read_node_is_direct_ctrled_by_write_node);
  };
  // TODO check other node
}

/**
 *
 * `....` means ctrl edge
                                                   g1

                                                     (0,1)
                                              ┌──────────────────────────────────┐
                                              │                                  ∨
┌────────┐  (0,1)   ┌────────────┐          ┌──────┐     ┌──────────┐  (0,0)   ┌─────┐  (0,0)   ┌────────┐
│ const1 │ ───────> │ assgin_var │ ·······> │ var2 │ ··> │ read_var │ ───────> │ mul │ ───────> │ output │
└────────┘          └────────────┘          └──────┘     └──────────┘          └─────┘          └────────┘
                      ∧                                    ∧
                      │ (0,0)                              │
                      │                                    │
                    ┌────────────┐  (0,0)                  │
                    │    var1    │ ────────────────────────┘
                    └────────────┘

 */
TEST_F(GraphCompilerTest, test_compile_write_var_first_then_read_var2) {
  Graph graph = BuildWRGraph2();
  const auto &compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_NE(compute_graph->FindNode("read_var"), nullptr);
  DUMP_GRAPH_WHEN("PrepareAfterProcessAippStage2", "PrepareAfterPrepareOptimize")
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // `graph` is a para of lambda
  std::size_t origin_size;
  CHECK_GRAPH(PrepareAfterProcessAippStage2) {
    EXPECT_NE(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("mul");
    origin_size = read_node->GetInControlNodesSize();
  };
  // `graph` is a para of lambda
  CHECK_GRAPH(PrepareAfterPrepareOptimize) {
    // read_var is removed
    EXPECT_EQ(graph->FindNode("read_var"), nullptr);
    auto read_node = graph->FindNode("mul");
    // read_var's ctrl in moved to read_node
    EXPECT_NE(read_node, nullptr);
    auto new_in_ctrl_nodes = read_node->GetInControlNodes();
    EXPECT_EQ(new_in_ctrl_nodes.size(), origin_size + 1U);
    auto write_node = graph->FindNode("assgin_var");
    bool read_node_is_direct_ctrled_by_write_node =
        std::any_of(new_in_ctrl_nodes.begin(), new_in_ctrl_nodes.end(),
                    [&write_node](const NodePtr &node) { return (node == write_node); });
    EXPECT_FALSE(read_node_is_direct_ctrled_by_write_node);
    auto write_ctrl_node = graph->FindNode("var2");
    bool read_node_is_indirect_ctrled_by_write_node =
        std::any_of(new_in_ctrl_nodes.begin(), new_in_ctrl_nodes.end(),
                    [&write_ctrl_node](const NodePtr &node) { return (node == write_ctrl_node); });
    EXPECT_TRUE(read_node_is_indirect_ctrled_by_write_node);
  };
  // TODO check other node
}

TEST_F(GraphCompilerTest, test_build_graph_with_maxsize_success) {
  SetGeLocalBuilder();
  DEF_GRAPH(g1) {
    auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_ND, DT_STRING, {200, 200, 3})
                           .Attr("_op_max_shape", "200,200,3");
    auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_STRING, {0});
    auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_STRING, {0});
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  EXPECT_EQ(ge::AttrUtils::SetListInt(node->GetOpDesc(), "_op_max_size", {1000, 1000}), true);

  auto node_data = compute_graph->FindNode("data_1");
  (void)AttrUtils::SetBool(node_data->GetOpDesc(), "OwnerGraphIsUnknown", true);
  (void) AttrUtils::SetStr(node_data->GetOpDesc(), ATTR_NAME_ENGINE_NAME_FOR_LX, "DNN_VM_GE_LOCAL");
  (void) AttrUtils::SetStr(node_data->GetOpDesc(), ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, "DNN_VM_GE_LOCAL_OP_STORE");

  node_data = compute_graph->FindNode("data_2");
  (void)AttrUtils::SetBool(node_data->GetOpDesc(), "OwnerGraphIsUnknown", true);
  (void) AttrUtils::SetStr(node_data->GetOpDesc(), ATTR_NAME_ENGINE_NAME_FOR_LX, "DNN_VM_GE_LOCAL");
  (void) AttrUtils::SetStr(node_data->GetOpDesc(), ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, "DNN_VM_GE_LOCAL_OP_STORE");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  Status ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  SetFakerBuilder();
}

/**
 *      data1
 *        \
 *       hcom
 *         \
 *       netout
 */
TEST_F(GraphCompilerTest, test_build_memory_buffer_pool_fail) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32})
                    .Attr(ATTR_NAME_BUFFER_POOL_ID, 1)
                    .Attr(ATTR_NAME_BUFFER_POOL_SIZE, 10240);
  auto hcom = OP_CFG(HCOMALLREDUCE)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto netout = OP_CFG(NETOUTPUT)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("data_1");
  TensorUtils::SetSize(*node->GetOpDesc()->MutableOutputDesc(0), 5120);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data1
 *        |
 *     prefetch
 *        |
 *      split
 *       | |
 *      concat
 *        |
 *     netoutput
 */
TEST_F(GraphCompilerTest, test_build_memory_buffer_pool_success_ref_io) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(VARIABLE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 8});
  auto prefetch = OP_CFG(HCOMALLGATHER)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 8})
      .Attr(ATTR_NAME_BUFFER_POOL_ID, 1)
      .Attr(ATTR_NAME_BUFFER_POOL_SIZE, 10240);
  auto split = OP_CFG("SplitD")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 8});
  auto concat = OP_CFG(CONCAT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 8});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 8});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("prefetch", prefetch)->NODE("split", split)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("split", split)->EDGE(1, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("prefetch");
  TensorUtils::SetSize(*node->GetOpDesc()->MutableOutputDesc(0), 256);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /
 *         hcom
 *        /   \
 *     print  print
 */
TEST_F(GraphCompilerTest, test_build_memory_continuous) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom = OP_CFG(HCOMALLREDUCE)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_1", hcom)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({32,32});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_memory_invalid_workspace) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto hcom = OP_CFG(HCOMALLREDUCE);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
      CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_1", hcom)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({-2,32});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, FAILED);
}

TEST_F(GraphCompilerTest, test_build_memory_valid_neg1_workspace) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto hcom = OP_CFG(HCOMALLREDUCE);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_1", hcom)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({-1,32});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/*
add算子只有1个输出，该输出分别给到hcom1和hcom2
     data1    data2
       |  \   /   |
       |    add   |
    sqrt1  / \   sqrt2
       |  /   \   |
     hcom1     hcom2
        \       /
          netoutput
*/
TEST_F(GraphCompilerTest, test_build_memory_continuous_with_conflict) {
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .InCnt(2)
                  .OutCnt(2)
                  .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
                  .Build("hcom1");
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .InCnt(2)
                  .OutCnt(2)
                  .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
                  .Build("hcom2");
  auto relu1 = OP_CFG(RELU)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .InCnt(1)
                  .OutCnt(1)
                  .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
                  .Build("relu1");
  auto relu2 = OP_CFG(RELU)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .InCnt(1)
                  .OutCnt(1)
                  .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
                  .Build("relu2");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", DATA)->EDGE(0, 0)->NODE("add", ADD)->EDGE(0, 0)->NODE(hcom1)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_2", DATA)->EDGE(0, 1)->NODE("add", ADD)->EDGE(0, 0)->NODE(hcom2)->EDGE(0, 1)->NODE("netoutput"));
    CHAIN(NODE("data_1")->EDGE(0, 0)->NODE(relu1)->EDGE(0, 1)->NODE(hcom1));
    CHAIN(NODE("data_2")->EDGE(0, 0)->NODE(relu2)->EDGE(0, 1)->NODE(hcom2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize2")

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterOptimize2) {
    auto add_node = graph->FindFirstNodeMatchType(ADD);
    EXPECT_EQ(add_node->GetOutDataNodesSize(), 2);
    // check one of output of add is identity
    size_t identity_count = 0U, hcom_count = 0U;
    for (const auto &node : add_node->GetOutDataNodes()) {
      if (node->GetType() == IDENTITY) {
        identity_count++;
      }
      if (node->GetType() == HCOMALLREDUCE) {
        hcom_count++;
      }
    }
    EXPECT_EQ(identity_count, 1);
    EXPECT_EQ(hcom_count, 1);
  };
}

/*
 *     data
 *      |
 *      a
 *      |
 *  partitioned_call    +----------------------+
 *      |               | inner_data           |
 *      |               |     |                |
 *      |               |  reshape1            |
 *      |               |     |                |
 *      |               | continue_node        |
 *      |               |     |                |
 *      |               |  reshape2            |
 *      |               |     |                |
 *      b               | netoutput2           |
 *      |               +----------------------+
 *    netoutput1
 */
TEST_F(GraphCompilerTest, ContinuousNodeConnectSubGraphEdgeThroughRefNode_InsertIdentitySuccess) {
  const auto inner_data = OP_CFG(DATA).ParentNodeIndex(0);
  const auto reshape = OP_CFG(RELU).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("inner_data", inner_data)->NODE("reshape1", reshape)
              ->NODE("continue_node", RELU)->NODE("reshape2", reshape)->NODE("netoutput2", NETOUTPUT));
  };
  sub_1.Layout();
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("a", RELU)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
              ->NODE("b", RELU)->NODE("netoutput1", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(false);
  auto partitioned_call1_graph = compute_graph->GetAllSubgraphs().at(0);
  auto netoutput2 = partitioned_call1_graph->FindNode("netoutput2");
  AttrUtils::SetInt(netoutput2->GetOpDescBarePtr()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto continue_node = partitioned_call1_graph->FindNode("continue_node");
  AttrUtils::SetBool(continue_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  AttrUtils::SetBool(continue_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  std::vector<int64_t> known_shape = {1, 1, 224, 224};
  for (auto &node : compute_graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      out_tensor->SetShape(GeShape(known_shape));
      out_tensor->SetDataType(DT_FLOAT);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      in_tensor->SetShape(GeShape(known_shape));
      in_tensor->SetDataType(DT_FLOAT);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  compute_graph->TopologicalSorting();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto sub_graph = graph->GetAllSubgraphs().front();
    auto data = sub_graph->FindNode("inner_data");
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->GetOutNodes().at(0)->GetType(), IDENTITY);
    auto netoutput_node = sub_graph->FindNode("netoutput2");
    EXPECT_EQ(netoutput_node->GetInDataNodes().at(0)->GetType(), IDENTITY);
  };
}

/*
 *     data
 *      |
 *      a
 *      |
 *  partitioned_call    +----------------------+
 *      |               | inner_data   memset  |
 *      |               |     |       /(ctl)   |
 *      |               | atomic_node          |
 *      |               |     |                |
 *      |               |  reshape             |
 *      |               |     |                |
 *      b               | netoutput2           |
 *      |               +----------------------+
 *    netoutput1
 */
TEST_F(GraphCompilerTest, PartitionedCallWithAtomicNode_CheckOffsetSuccess) {
  const auto inner_data = OP_CFG(DATA).ParentNodeIndex(0);
  const auto reshape = OP_CFG(RELU).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("inner_data", inner_data)->NODE("atomic_node", RELU)->NODE("reshape", reshape)->NODE("netoutput2", NETOUTPUT));
  };
  sub_1.Layout();
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("a", RELU)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("b", RELU)->NODE("netoutput1", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(false);
  auto partitioned_call1_graph = compute_graph->GetAllSubgraphs().at(0);
  auto netoutput2 = partitioned_call1_graph->FindNode("netoutput2");
  AttrUtils::SetInt(netoutput2->GetOpDescBarePtr()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto atomic_node = partitioned_call1_graph->FindNode("atomic_node");
  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0});
  std::vector<int64_t> known_shape = {1, 1, 224, 224};
  for (auto &node : compute_graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      out_tensor->SetShape(GeShape(known_shape));
      out_tensor->SetDataType(DT_FLOAT);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      in_tensor->SetShape(GeShape(known_shape));
      in_tensor->SetDataType(DT_FLOAT);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  compute_graph->TopologicalSorting();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  auto partitioned_call = compute_graph->FindNode("partitioned_call");
  ASSERT_NE(partitioned_call, nullptr);
  EXPECT_EQ(partitioned_call->GetOpDescBarePtr()->GetOutputOffset().at(0), atomic_node->GetOpDescBarePtr()->GetOutputOffset().at(0));
}
/**
 *      data  data
 *        \   /
 *       broadcast
 *        /   \
 *     print   print
 */
TEST_F(GraphCompilerTest, test_build_memory_continuous_broadcast) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom = OP_CFG(HCOMBROADCAST)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto print1 = OP_CFG("Print");
  auto print2 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("print_1", print1));
    CHAIN(NODE("hcom_1", hcom)->EDGE(1, 0)->NODE("print_2", print2));
  };

  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a  scatter_nd   /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 *     异常分支覆盖
 */
TEST_F(GraphCompilerTest, AtomicCleanOutputCheckFailed) {
  DEF_GRAPH(graph) {
    auto atomic_memset = OP_CFG(MEMSET)
                             .InCnt(0)
                             .OutCnt(0);

    auto scatter_nd = OP_CFG("AtomicNode")
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_INT32, {16, 448});

    auto reshape = OP_CFG(RESHAPE)
                       .InCnt(2)
                       .OutCnt(1)
                       .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(0)
                          .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

    CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("scatter_nd", scatter_nd)->NODE("reshape", reshape)->NODE("netoutput", net_output));
    CHAIN(NODE("a", RELU)->NODE("reshape", reshape));
    CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("reshape", reshape));
  };
  auto root_graph = ToComputeGraph(graph);

  auto atomic_node = root_graph->FindNode("scatter_nd");
  std::vector<int32_t> data_list = {ge::DataType::DT_INT16};
  std::vector<int32_t> int_list = {0x1};
  std::vector<float32_t> float_list = {};
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), "tbe_op_atomic_dtypes", data_list);
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), "tbe_op_atomic_int64_values", int_list);
  (void) AttrUtils::SetListFloat(atomic_node->GetOpDesc(), "tbe_op_atomic_float_values", float_list);

  EXPECT_EQ(AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0}), true);
  for (auto &node : root_graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
  root_graph->TopologicalSorting();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  // 构造memset中的地址错误
  auto mem_set_node = root_graph->FindNode("mem_set");
  ASSERT_NE(mem_set_node, nullptr);
  const auto memset_origin_workspace = mem_set_node->GetOpDescBarePtr()->GetWorkspace();
  mem_set_node->GetOpDescBarePtr()->SetWorkspace({10000000});

  AtomicCleanChecker checker1(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret = checker1.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(memset_origin_workspace);
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *         hcom
 *        /
 *     netout
 */
TEST_F(GraphCompilerTest, AtomicClean_Success_CleanOutConnectCleanInput) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int32_t> input_indexes = {-1};
  auto hcom = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto add1 = OP_CFG(ADD).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_INPUT_INDEX, input_indexes);

  node = compute_graph->FindNode("add_1");
  op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *         hcom
 *        /
 *     netout
 */
TEST_F(GraphCompilerTest, AtomicClean_Success_CleanOutConnectCleanInput_SeperatelyCleanPolicy) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int32_t> input_indexes = {-1};
  auto hcom = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto add1 = OP_CFG(ADD).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_INPUT_INDEX, input_indexes);

  node = compute_graph->FindNode("add_1");
  op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<std::string, std::string> options{{ge::ATOMIC_CLEAN_POLICY, "1"}};
  options[ge::MEMORY_OPTIMIZATION_POLICY] = "MemoryPriority";
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, test_build_memory_atomic_netoutput) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> atomic_output_index = {0};
  auto netout = OP_CFG(NETOUTPUT);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto add = OP_CFG(ADD).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add));
    CHAIN(NODE("add_1", data2)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<std::string, std::string> options{{ge::ATOMIC_CLEAN_POLICY, "0"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *              variable
 *              /
 *      data  trans1
 *        \   /
 *         assign
 *            \
 *            trans2
 *             /
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, test_build_memory_atomic_transdata) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> atomic_output_index = {0};

  DEF_GRAPH(g1) {
    CHAIN(NODE("variable", VARIABLE)->NODE("trans1", TRANSDATA)->NODE("assign", ASSIGN)
              ->NODE("trans2", TRANSDATA)->NODE("out_1", NETOUTPUT));
    CHAIN(NODE("data1", DATA)->EDGE(0, 1)->NODE("assign", ASSIGN));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto trans2 = compute_graph->FindNode("trans2");
  auto op_desc = trans2->GetOpDesc();
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  ge::AttrUtils::SetStr(trans2->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, "variable");

  map<std::string, std::string> options{{ge::ATOMIC_CLEAN_POLICY, "0"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_tiling_depend) {
 auto ge_env = GeRunningEnvFaker();
  ge_env.Reset()
    .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine").GraphOptimizer("AIcoreEngine"))
    .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER"))
    .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp("MoeFFN").InfoStoreAndBuilder("AIcoreEngine"));
  auto netout = OP_CFG(NETOUTPUT);
  auto const_op = OP_CFG(CONSTANT);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto moe_op = OP_CFG("MoeFFN");

  DEF_GRAPH(g1) {
    CHAIN(NODE("const", const_op)->EDGE(0, 0)->NODE("moeFNN", moe_op));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("moeFNN"));
    CHAIN(NODE("data2", data2)->EDGE(0, 2)->NODE("moeFNN"));
    CHAIN(NODE("moeFNN")->EDGE(0, 0)->NODE("out", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto constant_node = compute_graph->FindNode("const");
  auto constant_desc = constant_node->GetOpDesc();
  std::vector<int64_t> known_shape = {2, 5};
  ge::GeTensorDesc const_desc0(GeShape(known_shape), ge::FORMAT_ND, DT_INT32);
  uint8_t c_data[40] = {0};
  c_data[0] = 8;
  ge::ConstGeTensorPtr const_tensor =
          std::make_shared<GeTensor>(const_desc0, c_data, 40);
  ge::AttrUtils::SetTensor(constant_desc, ge::ATTR_NAME_WEIGHTS, const_tensor);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  auto ret = session.CompileGraph(1);

  EXPECT_EQ(ret, SUCCESS);
  ge_env.InstallDefault();
}

TEST_F(GraphCompilerTest, test_tiling_depend_with_placement_aicpu) {
 auto ge_env = GeRunningEnvFaker();
  ge_env.Reset()
    .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine").GraphOptimizer("AIcoreEngine"))
    .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER"))
    .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
    .Install(FakeOp("IFN").InfoStoreAndBuilder("AIcoreEngine"));
  auto netout = OP_CFG(NETOUTPUT);
  auto const_op = OP_CFG(CONSTANT);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto ifn_op = OP_CFG("IFN");

  DEF_GRAPH(g1) {
    CHAIN(NODE("const", const_op)->EDGE(0, 0)->NODE("ifn", ifn_op));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("ifn"));
    CHAIN(NODE("data2", data2)->EDGE(0, 2)->NODE("ifn"));
    CHAIN(NODE("ifn")->EDGE(0, 0)->NODE("out", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto constant_node = compute_graph->FindNode("const");
  auto constant_desc = constant_node->GetOpDesc();
  std::vector<int64_t> known_shape = {2, 5};
  ge::GeTensorDesc const_desc0(GeShape(known_shape), ge::FORMAT_ND, DT_INT32);
  uint8_t c_data[40] = {0};
  c_data[0] = 8;
  ge::ConstGeTensorPtr const_tensor =
          std::make_shared<GeTensor>(const_desc0, c_data, 40);
  ge::AttrUtils::SetTensor(constant_desc, ge::ATTR_NAME_WEIGHTS, const_tensor);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_DEV_CAP_SUPPORT);
  map<AscendString, AscendString> options;
  options.insert({"ge.tiling_schedule_optimize", "1"});
  Session session(options);
  session.AddGraph(1, graph, options);
  auto ret = session.CompileGraph(1);

  EXPECT_EQ(ret, SUCCESS);
  ge_env.InstallDefault();
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicClean_Success_CleanOutAndWorkspace_FusionNode) {
  std::vector<int64_t> atomic_output_index = {0};
  auto netout = OP_CFG(NETOUTPUT);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto add = OP_CFG(ADD).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add));
    CHAIN(NODE("add_1", data2)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  (void)ge::AttrUtils::SetBool(op_desc, ATOMIC_ATTR_IS_FUSION_NODE, true);

  map<string, map<int64_t, int64_t>> workspace_info;
  workspace_info["add_1"][0] = 100;
  (void)op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /
 *       assignadd
 *          |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, test_build_memory_ref) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ASSIGNADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, true);
  op_desc->MutableAllOutputName().erase("__output0");
  op_desc->MutableAllOutputName()["__input0"] = 0;

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data_1  data_2
 *        \   /
 *       add_1
 *          |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, test_build_memory_atomic_connect_netoutput) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 =
      OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32}).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {0};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a  scatter_nd   /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 */
TEST_F(GraphCompilerTest, AtomicCleanOutput_MemsetOffsetOverlap) {
  DEF_GRAPH(graph) {
    auto atomic_memset = OP_CFG(MEMSET)
                             .InCnt(0)
                             .OutCnt(0);

    auto scatter_nd = OP_CFG("AtomicNode")
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_INT32, {16, 448});

    auto reshape = OP_CFG(RESHAPE)
                       .InCnt(2)
                       .OutCnt(1)
                       .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(0)
                          .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

    CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("scatter_nd", scatter_nd)->NODE("reshape", reshape)->NODE("netoutput", net_output));
    CHAIN(NODE("a", RELU)->NODE("reshape", reshape));
    CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("reshape", reshape));
  };
  auto root_graph = ToComputeGraph(graph);

  auto atomic_node = root_graph->FindNode("scatter_nd");
  std::vector<int32_t> data_list = {ge::DataType::DT_INT16};
  std::vector<int32_t> int_list = {0x1};
  std::vector<float32_t> float_list = {};
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), "tbe_op_atomic_dtypes", data_list);
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), "tbe_op_atomic_int64_values", int_list);
  (void) AttrUtils::SetListFloat(atomic_node->GetOpDesc(), "tbe_op_atomic_float_values", float_list);

  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0}), true);
  for (auto &node : root_graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
  root_graph->TopologicalSorting();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  // 构造memset的workspace中的offset，size有包含关系的场景
  auto mem_set_node = root_graph->FindNode("mem_set");
  ASSERT_NE(mem_set_node, nullptr);
  const auto memset_origin_workspace = mem_set_node->GetOpDescBarePtr()->GetWorkspace();
  std::vector<int64_t> fake_workspace;
  fake_workspace.emplace_back(memset_origin_workspace.back() - 2);
  fake_workspace.emplace_back(memset_origin_workspace.back() - 1);
  fake_workspace.emplace_back(memset_origin_workspace.back() + 1);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(fake_workspace);

  const auto memset_origin_workspace_sizes = mem_set_node->GetOpDescBarePtr()->GetWorkspaceBytes();
  std::vector<int64_t> fake_sizes;
  fake_sizes.emplace_back(memset_origin_workspace_sizes.back() + 10);
  fake_sizes.emplace_back(memset_origin_workspace_sizes.back() - 10);
  fake_sizes.emplace_back(memset_origin_workspace_sizes.back() - 10);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(fake_sizes);

  std::vector<int32_t> data_type_list{ge::DataType::DT_INT16, ge::DataType::DT_INT16, ge::DataType::DT_INT16};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list);
  std::vector<int32_t> int_list3 = {0x1, 2, 3};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list3);

  AtomicCleanChecker checker1(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret = checker1.Check(root_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 构造memset中的workspace offset有重叠的场景
  auto fake_workspace2 = memset_origin_workspace;
  fake_workspace2.emplace_back(memset_origin_workspace.back());
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(fake_workspace2);

  auto fake_sizes2 = memset_origin_workspace_sizes;
  fake_sizes2.emplace_back(memset_origin_workspace_sizes.back() / 2);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(fake_sizes2);

  std::vector<int32_t> data_type_list2{ge::DataType::DT_INT16, ge::DataType::DT_INT16};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list2);
  std::vector<int32_t> int_list4 = {0x1, 2};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list4);

  AtomicCleanChecker checker2(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret2 = checker2.Check(root_graph);
  EXPECT_EQ(ret2, SUCCESS);

  // workspace中offset较多，超过日志长度，分多行打印
  std::vector<int64_t> long_workspace;
  std::vector<int64_t> long_workspace_bytes;
  for (int64_t i = 0; i < 1000; ++i) {
    long_workspace.emplace_back(i);
    long_workspace_bytes.emplace_back(1);
  }
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(long_workspace);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(long_workspace_bytes);
  AtomicCleanChecker checker6(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker6.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
}
/**
 *      a      b
 *        \   /
 *       assign
 *          |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicClean_Failed_RefNodeNeedClean) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto assign =
      OP_CFG(ASSIGNADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32})
          .Attr(ATTR_NAME_REFERENCE, true).InNames({"x", "y"}).OutNames({"x"});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("a", RELU)->EDGE(0, 0)->NODE("assign", assign));
    CHAIN(NODE("data_2", data2)->NODE("b", RELU)->EDGE(0, 1)->NODE("assign", assign));
    CHAIN(NODE("assign", assign)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("assign");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {0};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  TensorUtils::SetReuseInput(*op_desc->MutableOutputDesc(0), true);
  TensorUtils::SetReuseInputIndex(*op_desc->MutableOutputDesc(0), 0);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_NE(ret, SUCCESS);
}

/**
 *      data_1  data_2
 *        \   /
 *       add_1
 *       | | |
 *       a b c
 *       | | |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicCleanCheck_Success_MultiOutputOnlyCleanTheSecond) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 =
      OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32}).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto relu = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->NODE("a", relu));
    CHAIN(NODE("add_1", add1)->NODE("b", relu));
    CHAIN(NODE("add_1", add1)->NODE("c", relu));
    CHAIN(NODE("a", relu)->NODE("out_1", netout));
    CHAIN(NODE("b", relu)->NODE("out_1", netout));
    CHAIN(NODE("c", relu)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data_1  data_2
 *        \   /
 *       add_1
 *       | | |
 *       a b c
 *       | | |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicCleanCheck_Success_MultiWorkspaceOnlyCleanTheSecond) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 =
      OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32}).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto relu = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->NODE("a", relu));
    CHAIN(NODE("add_1", add1)->NODE("b", relu));
    CHAIN(NODE("add_1", add1)->NODE("c", relu));
    CHAIN(NODE("a", relu)->NODE("out_1", netout));
    CHAIN(NODE("b", relu)->NODE("out_1", netout));
    CHAIN(NODE("c", relu)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  node->GetOpDesc()->SetWorkspaceBytes({1024, 2048, 4096});
  std::map<std::string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  atomic_workspace_index_size["add_1"][1] = 2048;
  node->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data_1  data_2
 *        \   /
 *       add_1
 *       | | |
 *       a b c
 *       | | |
 *        hcom
 *       | | |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicCleanCheck_Success_MultiNodeNeedClean) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 =
      OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32}).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto hcom = OP_CFG(HCOMALLGATHER)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32})
                  .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto relu = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->NODE("a", relu));
    CHAIN(NODE("add_1", add1)->NODE("b", relu));
    CHAIN(NODE("add_1", add1)->NODE("c", relu));
    CHAIN(NODE("a", relu)->NODE("hcom", hcom));
    CHAIN(NODE("b", relu)->NODE("hcom", hcom));
    CHAIN(NODE("c", relu)->NODE("hcom", hcom));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  node->GetOpDesc()->SetWorkspaceBytes({1024, 2048, 4096});
  std::map<std::string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  atomic_workspace_index_size["add_1"][1] = 2048;
  node->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);

  auto hcom_node = compute_graph->FindNode("hcom");
  ASSERT_NE(hcom_node, nullptr);
  std::vector<int64_t> atomic_in_index = {-1};
  (void)ge::AttrUtils::SetListInt(hcom_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_in_index);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data_1  data_2
 *        \   /
 *       add_1
 *       | | |
 *       a b c
 *       | | |
 *        hcom
 *       | | |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicCleanCheck_Success_HcomNeedCleanInputAndSetP2pMemoryType) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 =
      OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32}).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto hcom = OP_CFG(HCOMALLGATHER)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32})
                  .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto relu = OP_CFG(RELU).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->NODE("a", relu));
    CHAIN(NODE("add_1", add1)->NODE("b", relu));
    CHAIN(NODE("add_1", add1)->NODE("c", relu));
    CHAIN(NODE("a", relu)->NODE("hcom", hcom));
    CHAIN(NODE("b", relu)->NODE("hcom", hcom));
    CHAIN(NODE("c", relu)->NODE("hcom", hcom));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  node->GetOpDesc()->SetWorkspaceBytes({1024, 2048, 4096});
  std::map<std::string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  atomic_workspace_index_size["add_1"][1] = 2048;
  node->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  std::vector<int64_t> workspace_type_list{RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR};
  ge::AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);

  auto hcom_node = compute_graph->FindNode("hcom");
  ASSERT_NE(hcom_node, nullptr);
  std::vector<int64_t> atomic_in_index = {-1};
  (void)ge::AttrUtils::SetListInt(hcom_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_in_index);
  std::vector<int64_t> in_memory_types = {RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR};
  (void)ge::AttrUtils::SetListInt(hcom_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_types);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data_1  data_2
 *        \   /
 *       add_1
 *       | | |
 *        hcom
 *       | | |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, AtomicCleanCheck_Success_CleanOutAndInputConnected) {
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  OpsKernelManager &ops_kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  OpInfo oi;
  oi.engine = "DNN_VM_HCCL";
  oi.isAtomic = true;
  ops_kernel_manager.ops_kernel_info_[HCOMALLGATHER].emplace_back(oi);

  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 =
      OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32}).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto hcom = OP_CFG(HCOMALLGATHER)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32})
                  .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->NODE("hcom", hcom));
    CHAIN(NODE("add_1", add1)->NODE("hcom", hcom));
    CHAIN(NODE("add_1", add1)->NODE("hcom", hcom));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
    CHAIN(NODE("hcom", hcom)->NODE("out_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  auto hcom_node = compute_graph->FindNode("hcom");
  ASSERT_NE(hcom_node, nullptr);
  std::vector<int64_t> atomic_in_index = {-1};
  (void)ge::AttrUtils::SetListInt(hcom_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_in_index);
  std::vector<int64_t> in_memory_types = {RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR};
  (void)ge::AttrUtils::SetListInt(hcom_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_types);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /    /
 *       assignadd
 *          |
 *      NETOUTPUT
 */
TEST_F(GraphCompilerTest, TestBuildMemoryRef_WithUnfedAnchorOnNode_SUCCESS) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ASSIGNADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
                  CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
                  CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("out_1", netout));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, true);
  op_desc->MutableAllOutputName().erase("__output0");
  op_desc->MutableAllOutputName()["__input0"] = 0;

  auto invalid_td = std::make_shared<GeTensorDesc>();
  invalid_td->SetDataType(DT_UNDEFINED);
  invalid_td->SetFormat(FORMAT_RESERVED);
  op_desc->AddInputDesc(invalid_td->Clone());

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, ReAssignAtomicMemoryWithOutMergeNodesAttrs) {
  DEF_GRAPH(ge_graph) {
    auto atomic_memset = OP_CFG(MEMSET)
                             .InCnt(1)
                             .OutCnt(1)
                             .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto node1 = OP_CFG("AtomicNode")
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node2 = OP_CFG("AtomicNode")
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node3 = OP_CFG("AtomicNode")
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("atomic_node0", node1)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node1", node2)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node2", node3)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_memset", atomic_memset)->NODE("Node_Output", net_output));
  };
  auto graph = ToComputeGraph(ge_graph);
  UpdateGraphTensorSize(graph);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto atomic_node0 = graph->FindNode("atomic_node0");
  auto atomic_node1 = graph->FindNode("atomic_node1");
  auto atomic_node2 = graph->FindNode("atomic_node2");
  auto atomic_memset = graph->FindNode("atomic_memset");
  atomic_node0->GetOpDesc()->SetOutputOffset({0});
  atomic_node1->GetOpDesc()->SetOutputOffset({16});
  atomic_node2->GetOpDesc()->SetOutputOffset({32});

  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node0->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node1->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node0->GetOpDesc(), "is_atomic_node", true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node1->GetOpDesc(), "is_atomic_node", true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node2->GetOpDesc(), "is_atomic_node", true), true);
  std::vector<uint32_t> value;
  value.push_back(0);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), "atomic_output_index", value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), "atomic_output_index", value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), "atomic_output_index", value), true);
  // init graph_mem_assigner
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, 0x10);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  // mock fe set data_type/val_int/val_float attr
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_INT32}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_FLOAT16}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_INT64}), true);

  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0}), true);
  EXPECT_EQ(AttrUtils::SetListFloat(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, {0.1}), true);
  EXPECT_EQ(AttrUtils::SetListFloat(atomic_node1->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, {0.1}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0}), true);
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  EXPECT_EQ(graph_mem_assigner.ReAssignAtomicMemory(), SUCCESS);
}

/*
 *     data
 *      |
 *      a
 *      |
 *  partitioned_call    +----------------------+
 *      |               | inner_data           |
 *      |               |     |                |
 *      |               |     c                |
 *      |               |     |                |
 *      |               |     d                |
 *      |               |     |                |
 *      |               |    hcom              |
 *      |               |     |                |
 *      b               | netoutput2           |
 *      |               +----------------------+
 *    netoutput1
 */
TEST_F(GraphCompilerTest, SubGraphDataNotReuseWithAtomicCleanNode) {
  const auto inner_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("inner_data", inner_data)->NODE("c", RELU)
              ->NODE("d", RELU)->NODE("hcom", HCOMALLREDUCE)->NODE("netoutput2", NETOUTPUT));
  };
  sub_1.Layout();
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("a", RELU)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
              ->NODE("b", RELU)->NODE("netoutput1", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(false);
  auto partitioned_call1_graph = compute_graph->GetAllSubgraphs().at(0);
  auto netoutput2 = partitioned_call1_graph->FindNode("netoutput2");
  AttrUtils::SetInt(netoutput2->GetOpDescBarePtr()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto hcom = partitioned_call1_graph->FindNode("hcom");
  AttrUtils::SetBool(hcom->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  AttrUtils::SetBool(hcom->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  std::vector<int64_t> atomic_input_index{-1};
  AttrUtils::SetListInt(hcom->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
  AttrUtils::SetBool(hcom->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  BlockTypeList block_type_list;
  EXPECT_STREQ(block_type_list.ToString().c_str(), "");
  std::vector<int64_t> known_shape = {1, 1, 224, 224};
  for (auto &node : compute_graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      out_tensor->SetShape(GeShape(known_shape));
      out_tensor->SetDataType(DT_FLOAT);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      in_tensor->SetShape(GeShape(known_shape));
      in_tensor->SetDataType(DT_FLOAT);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  compute_graph->TopologicalSorting();
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto sub_graph = graph->GetAllSubgraphs().front();
    auto data = sub_graph->FindNode("inner_data");
    ASSERT_NE(data, nullptr);
    auto d = sub_graph->FindNode("d");
    ASSERT_NE(d, nullptr);
    EXPECT_NE(data->GetOpDescBarePtr()->GetOutputOffset().at(0), d->GetOpDescBarePtr()->GetOutputOffset().at(0));
  };
}


/**
 *      data_1 data_2  data_3  data_4
 *       |\     /        \     /
 *       | add_1         add_2
 *        \    \        /
 *         \    relu_1
 *          \   /
 *          add_3
 *           |
 *        NETOUTPUT_1
 *  这个场景存在问题，add_1需要集中清零，且是nopadding连续输入的首节点。
 *  首次分配内存时没有给add_1分配内存，而分配atomic内存时，只按照add_1自己的大小分配的内存，导致add2的输出内存错误，与add3的输出重叠了。
 *
 *  增加内存复用检查模块时校验报错了。但是本场景问题需要对IsOutNodeSetContinuousInput函数进行重构，修复风险很高，待需求合入后单独解决
 *
 *  atomic_addr_clean_pass.cc中对于atomic out连接no task的，设置为单独清零了，所以真实场景中不存在该问题
 */
TEST_F(GraphCompilerTest, CentralizedAtomicClean_And_NoPaddingContinousInput) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int32_t> input_indexes = {-1};
  auto relu = OP_CFG(RELU)
                  .Attr(ATTR_NAME_OUTPUT_REUSE_INPUT, true)
                  .Attr(ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0)
                  .Attr(ATTR_NAME_NOTASK, true)
                  .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  auto data4 = OP_CFG(DATA);
  auto add1 = OP_CFG(ADD);
  auto add2 = OP_CFG(ADD);
  auto add3 = OP_CFG(ADD);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_4", data4)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("relu_1", relu));
    CHAIN(NODE("add_2", add2)->EDGE(0, 1)->NODE("relu_1", relu));
    CHAIN(NODE("relu_1", relu)->EDGE(0, 0)->NODE("add_3", add3));
    CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("add_3", add3));
    CHAIN(NODE("add_3", add3)->EDGE(0, 0)->NODE("NETOUTPUT_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto relu_1 = compute_graph->FindNode("relu_1");
  ASSERT_NE(relu_1, nullptr);
  auto tensor_desc = relu_1->GetOpDescBarePtr()->GetOutputDesc(0);
  GeShape out_shape({1, 1, 224, 448});
  tensor_desc.SetShape(out_shape);
  relu_1->GetOpDescBarePtr()->UpdateOutputDesc(0, tensor_desc);

  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  (void)ret;
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data_1 data_2  data_3  data_4
 *       |\     /        \     /
 *       | add_1         add_2
 *        \    \        /
 *         \    relu_1
 *          \   /
 *          add_3
 *           |
 *        NETOUTPUT_1
 */
TEST_F(GraphCompilerTest, AtomicClean_Failed_CleanBothInputAndOutput) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int32_t> input_indexes = {-1};
  auto relu = OP_CFG(RELU)
                  .Attr(ATTR_NAME_OUTPUT_REUSE_INPUT, true)
                  .Attr(ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0)
                  .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  auto data4 = OP_CFG(DATA);
  auto add1 = OP_CFG(ADD);
  auto add2 = OP_CFG(ADD);
  auto add3 = OP_CFG(ADD);
  auto netout = OP_CFG(NETOUTPUT);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_4", data4)->EDGE(0, 1)->NODE("add_2", add2));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("relu_1", relu));
    CHAIN(NODE("add_2", add2)->EDGE(0, 1)->NODE("relu_1", relu));
    CHAIN(NODE("relu_1", relu)->EDGE(0, 0)->NODE("add_3", add3));
    CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("add_3", add3));
    CHAIN(NODE("add_3", add3)->EDGE(0, 0)->NODE("NETOUTPUT_1", netout));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto relu_1 = compute_graph->FindNode("relu_1");
  ASSERT_NE(relu_1, nullptr);
  auto tensor_desc = relu_1->GetOpDescBarePtr()->GetOutputDesc(0);
  GeShape out_shape({1, 1, 224, 448});
  tensor_desc.SetShape(out_shape);
  relu_1->GetOpDescBarePtr()->UpdateOutputDesc(0, tensor_desc);

  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
  std::vector<int64_t> atomic_output_index = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_NE(ret, SUCCESS);
}

/**
 *       data  data
 *         \  /
 *         add
 *          |
 *        SLICE
 *        |   \
 *     CONV2D CONV2D
 */
TEST_F(GraphCompilerTest, test_build_memory_output_continuous_nopading_fail) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto slice = OP_CFG(SLICE)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                   .Attr(ATTR_NAME_OUTPUT_REUSE_INPUT, true)
                   .Attr(ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0)
                   .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto conv1 = OP_CFG(CONV2D);
  auto conv2 = OP_CFG(CONV2D);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", slice)->EDGE(0, 0)->NODE("slice_1", slice));
    CHAIN(NODE("slice_1", slice)->EDGE(0, 0)->NODE("conv_1", conv1));
    CHAIN(NODE("slice_1", slice)->EDGE(1, 0)->NODE("conv_2", conv2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("slice_1");
  auto op_desc = node->GetOpDesc();
  op_desc->MutableOutputDesc(0)->MutableShape().SetDimNum(1);
  op_desc->MutableOutputDesc(0)->MutableShape().SetDim(0, 1);
  op_desc->MutableOutputDesc(1)->MutableShape().SetDimNum(1);
  op_desc->MutableOutputDesc(1)->MutableShape().SetDim(0, 1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *       data  data
 *         \  /
 *         add
 *          |
 *        SLICE
 *        |   \
 *     CONV2D CONV2D
 */
TEST_F(GraphCompilerTest, TestMemoryCheckWithMemorySizeCalcTypeAttrAndMemoryTypeL1) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto slice = OP_CFG(SLICE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
      .Attr(ATTR_NAME_OUTPUT_REUSE_INPUT, true)
      .Attr(ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0)
      .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto conv1 = OP_CFG(CONV2D);
  auto conv2 = OP_CFG(CONV2D);

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
                  CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
                  CHAIN(NODE("add_1", slice)->EDGE(0, 0)->NODE("slice_1", slice));
                  CHAIN(NODE("slice_1", slice)->EDGE(0, 0)->NODE("conv_1", conv1));
                  CHAIN(NODE("slice_1", slice)->EDGE(1, 0)->NODE("conv_2", conv2));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("slice_1");
  auto op_desc = node->GetOpDesc();
  op_desc->MutableOutputDesc(0)->MutableShape().SetDimNum(1);
  op_desc->MutableOutputDesc(0)->MutableShape().SetDim(0, 1);
  op_desc->MutableOutputDesc(1)->MutableShape().SetDimNum(1);
  op_desc->MutableOutputDesc(1)->MutableShape().SetDim(0, 1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  auto add_1_node = compute_graph->FindNode("add_1");
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_MEMORY_SIZE_CALC_TYPE, 1);
  ge::AttrUtils::SetInt(add_1_node->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_L1);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/*
 *  a  hcom1
 *  |  | | \
 *  hcom2 hcom3
 */
TEST_F(GraphCompilerTest, ContinuousInOut_Success) {
  DEF_GRAPH(g1) {
                    CHAIN(NODE("a", RELU)->NODE("hcom2", RELU));
                    CHAIN(NODE("hcom1", RELU)->NODE("hcom2", RELU));
                    CHAIN(NODE("hcom1", RELU)->NODE("hcom3", RELU));
                    CHAIN(NODE("hcom1", RELU)->NODE("hcom3", RELU));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  MemConflictShareGraph::SetContinuousOutput(compute_graph, "hcom1");
  MemConflictShareGraph::SetContinuousInput(compute_graph, "hcom2");
  MemConflictShareGraph::SetContinuousInput(compute_graph, "hcom3");
  MemConflictShareGraph::SetShapeForAllNodes(compute_graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(compute_graph);

  MemConflictShareGraph::TopologicalSortingMock(compute_graph, {"hcom1", "hcom3", "hcom2"});
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *        RELU
 *         |
 *        RELU
 */
TEST_F(GraphCompilerTest, test_build_memory_continuous_nopading_cascade) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto relu1 = OP_CFG(RELU)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_OUTPUT_REUSE_INPUT, true)
                  .Attr(ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0)
                  .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto relu2 = OP_CFG(RELU)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("relu_1", relu1));
    CHAIN(NODE("relu_1", relu1)->EDGE(0, 0)->NODE("relu_2", relu2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data  data
 *        \   /
 *      NETOUTPUT
 *
 */
TEST_F(GraphCompilerTest, test_build_memory_ddr) {
  putenv("OP_NO_REUSE_MEM=data_1,data_2");
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int32_t> input_indexes = {-1};
  std::vector<int64_t> atomic_output_index = {};
  auto hcom = OP_CFG(NETOUTPUT);
  auto data1 = OP_CFG(DATA)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("data_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> output_memory_types = {RT_MEMORY_P2P_DDR};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, SetOutputOffsetForConcat_succ){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  MakeGraphDataInParent3(graph);
  graph->TopologicalSorting();
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto net_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(net_node, nullptr);
  auto opdesc = net_node->GetOpDesc();
  EXPECT_NE(opdesc, nullptr);
  EXPECT_TRUE(opdesc->HasAttr(ATTR_ZERO_COPY_BASIC_OFFSET));
}

/**
 *      var
 *        \
 *       broadcast
 *       /
 *      print
 */
TEST_F(GraphCompilerTest, test_build_memory_var_broadcast) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto hcom = OP_CFG(HCOMBROADCAST).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto print1 = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  DEF_GRAPH(g1) {
    CHAIN(NODE("var_1", var1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("print_1", print1));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  TensorUtils::SetSize(*node->GetOpDesc()->MutableOutputDesc(0), 200736);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  auto jit_res = ge::GetThreadLocalContext().GetAllOptions()[JIT_COMPILE];
  EXPECT_EQ(jit_res, "2");

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      var
 *        \
 *        cast    const
 *        /   \   /test_graph_compilertest_graph_compiler
 *      print assign
 */
TEST_F(GraphCompilerTest, test_build_memory_var_assign) {
  // prepare running env
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto assign = OP_CFG(ASSIGN).InCnt(2).OutCnt(1);
  auto var1 = OP_CFG(VARIABLE);
  auto cast = OP_CFG(CAST);
  auto print = OP_CFG("Print");
  auto data = OP_CFG(DATA);


  DEF_GRAPH(g1) {
    CHAIN(NODE("var_1", var1)->EDGE(0, 0)->NODE("cast_1", cast));
    CHAIN(NODE("cast_1", cast)->EDGE(0, 0)->NODE("print_1", print));
    CHAIN(NODE("cast_1", cast)->EDGE(0, 0)->NODE("assign_1", assign));
    CHAIN(NODE("data_1", data)->EDGE(0, 1)->NODE("assign_1", assign));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("assign_1");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, ATTR_INPUT_MEMORY_TYPE, RT_MEMORY_HBM);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  uint64_t available_mem = 0U;
  EXPECT_EQ(GetGraphAvailableMemory(compute_graph, available_mem), SUCCESS);
  // reshape shape const no need apply 32 mem when reshape to scalar.
  // so available mem add 32
  EXPECT_EQ(available_mem, 109051872UL);
}

/**
 *      var
 *        \
 *        broadcast
 *        /
 *      print
 */
TEST_F(GraphCompilerTest, test_build_memory_var_ref) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto hcom = OP_CFG(HCOMBROADCAST).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto print1 = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  DEF_GRAPH(g1) {
      CHAIN(NODE("var_1", var1)->EDGE(0, 0)->NODE("hcom_1", hcom));
      CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("print_1", print1));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  auto output_tensordesc = op_desc->MutableOutputDesc(0);
  TensorUtils::SetSize(*output_tensordesc, 200736);
  ge::AttrUtils::SetStr(output_tensordesc, ASSIGN_VAR_NAME, "var_1");
  op_desc->UpdateOutputDesc(0, *output_tensordesc);
  op_desc->AddInferFunc([](Operator &op) { return ge::GRAPH_SUCCESS; });

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // not assign memory, use variable's memory
  vector<int64_t> out_list = op_desc->GetOutputOffset();
  EXPECT_EQ(out_list.size(), 1U);
  uint8_t *dev_ptr = nullptr;
  auto var_1_node = compute_graph->FindNode("var_1");
  VarManager::Instance(compute_graph->GetSessionID())->GetVarAddr("var_1", var_1_node->GetOpDesc()->GetOutputDesc(0), dev_ptr);
  EXPECT_EQ(out_list[0], (int64_t)dev_ptr);
}


TEST_F(GraphCompilerTest, test_build_with_lx_fuison_node) {
  auto assign = OP_CFG(ASSIGN).InCnt(2).OutCnt(1);
  auto var1 = OP_CFG(VARIABLE);
  auto cast = OP_CFG(CAST);
  auto print = OP_CFG("Print");
  auto data = OP_CFG(DATA);


  DEF_GRAPH(g1) {
    CHAIN(NODE("var_1", var1)->EDGE(0, 0)->NODE("cast_1", cast));
    CHAIN(NODE("cast_1", cast)->EDGE(0, 0)->NODE("print_1", print));
    CHAIN(NODE("cast_1", cast)->EDGE(0, 0)->NODE("assign_1", assign));
    CHAIN(NODE("data_1", data)->EDGE(0, 1)->NODE("assign_1", assign));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("assign_1");
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_L2_FUSION_GROUP_ID, 0);

  map<AscendString, AscendString> options;
  options.emplace(BUFFER_OPTIMIZE.c_str(), "l2_optimize");
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_with_profiling_task_with_env) {
  auto hr1 = OP_CFG(HCOMALLREDUCE ).InCnt(2).OutCnt(2);
  auto var1 = OP_CFG(VARIABLE);
  auto cast = OP_CFG(CAST);
  auto print = OP_CFG("Print");
  auto data = OP_CFG(DATA);


  DEF_GRAPH(g1) {
    CHAIN(NODE("var_1", var1)->EDGE(0, 0)->NODE("cast_1", cast));
    CHAIN(NODE("cast_1", cast)->EDGE(0, 0)->NODE("print_1", print));
    CHAIN(NODE("cast_1", cast)->EDGE(0, 0)->NODE("hr1", hr1));
    CHAIN(NODE("data_1", data)->EDGE(0, 1)->NODE("hr1", hr1));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  ProfilingProperties::Instance().SetTrainingTrace(true);
  map<AscendString, AscendString> options;
  options.emplace(OPTION_EXEC_PROFILING_OPTIONS, R"({"fp_point":"hr1","bp_point":"hr1"})");
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ProfilingProperties::Instance().SetTrainingTrace(false);
}

/**
 *      data1  data2
 *         \   /
 *         merge
 */
TEST_F(GraphCompilerTest, test_update_input_output) {
  gert::GertRuntimeStub runtime_stub;
  const auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {16, 16, 224, 224})
      .Build("data1");
  const auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, {16, 16, 224, 224})
      .Build("data2");
  const auto merge = OP_CFG(MERGE)
      .InCnt(1).OutCnt(2)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("merge1");
  const auto net_output = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("NetOutput");

  (void)AttrUtils::SetStr(data1, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  (void)AttrUtils::SetStr(data2, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  (void)AttrUtils::SetStr(data1, ATTR_ATC_USER_DEFINE_FORMAT, "NC1HWC0");
  (void)AttrUtils::SetStr(data2, ATTR_ATC_USER_DEFINE_FORMAT, "NC1HWC0");

  (void)AttrUtils::SetListStr(net_output, ATTR_ATC_USER_DEFINE_DATATYPE, {"0:DT_FLOAT16"});
  (void)AttrUtils::SetListStr(net_output, ATTR_ATC_USER_DEFINE_FORMAT, {"0:NC1HWC0"});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(merge)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(merge));
  };

  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), AscendString("12800"));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      const1  data2
 *         \   /
 *         merge
 */
TEST_F(GraphCompilerTest, test_check_ref_input_node_failed) {
  const auto const1 = OP_CFG(CONSTANT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {16, 16, 224, 224})
      .Build("const1");
  const auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, {16, 16, 224, 224})
      .Build("data2");
  const auto merge = OP_CFG(MERGE)
      .InCnt(1).OutCnt(2)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("merge1");
  const auto net_output = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("net_output1");

  (void)AttrUtils::SetStr(const1, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  (void)AttrUtils::SetStr(data2, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  (void)AttrUtils::SetStr(const1, ATTR_ATC_USER_DEFINE_FORMAT, "NC1HWC0");
  (void)AttrUtils::SetStr(data2, ATTR_ATC_USER_DEFINE_FORMAT, "NC1HWC0");

  (void)AttrUtils::SetListStr(net_output, ATTR_ATC_USER_DEFINE_DATATYPE, {"0:DT_FLOAT16"});
  (void)AttrUtils::SetListStr(net_output, ATTR_ATC_USER_DEFINE_FORMAT, {"0:NC1HWC0"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE(const1)->EDGE(0, 0)->NODE(merge)->EDGE(0, 0)->NODE(net_output));
                  CHAIN(NODE(data2)->EDGE(0, 1)->NODE(merge));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("merge1");
  auto op_desc = node->GetOpDesc();
  op_desc->MutableAllOutputName().erase("__output0");
  op_desc->MutableAllOutputName()["__input0"] = 0;

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, FAILED);
}

TEST_F(GraphCompilerTest, test_exclude_engines) {
  Graph graph = BuildGraphWithHcclNode();

  map<std::string, std::string> empty_options;
  Session session(empty_options);
  std::map<std::string, std::string> options =
      {{EXCLUDE_ENGINES, " AiCore |AiVector|   Dsa| FakeEngine|FftsPlus"},
       {ge::CORE_TYPE, "FakeEngine"}};
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_NE(ret, SUCCESS);
}

/**
 *      data1  data2
 *         \   /
 *          Add
 */
// TEST_F(GraphCompilerTest, test_storage_format) {
//   auto graph = CreateGraphForTestStorageFormat(1L);
//   map<AscendString, AscendString> options;
//   Session session(options);
//   session.AddGraph(1, graph, options);
//
//   auto &engine_mapping = ge::hybrid::NodeExecutorManager::GetInstance().engine_mapping_;
//   engine_mapping.emplace("AiCoreLib", ge::hybrid::NodeExecutorManager::ExecutorType::AICORE);
//   std::vector<InputTensorInfo> inputs;
//   auto ret = session.BuildGraph(1, inputs);
//
//   EXPECT_EQ(ret, SUCCESS);
// }

TEST_F(GraphCompilerTest, test_storage_format_err) {
  auto graph = CreateGraphForTestStorageFormat(0x9FFFFFFFFFFFFFFFUL);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_NE(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, select_engine_by_attr_when_opdesc_empty) {
  std::string engine_name = "engine_name";
  std::string kernel_name = "kernel_name";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, engine_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kernel_name);
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), SUCCESS);

  ASSERT_EQ(op_desc->GetOpEngineName(), engine_name);
  ASSERT_EQ(op_info.opKernelLib, kernel_name);
}

TEST_F(GraphCompilerTest, select_engine_by_when_opdesc_and_attr_empty) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), FAILED);

  ASSERT_TRUE(op_desc->GetOpEngineName().empty());
}

TEST_F(GraphCompilerTest, select_engine_by_when_opdesc_type_is_valid) {
  auto fake_ops = MakeShared<FakeCompilerOpsKernelInfoStore>();
  OpsKernelManager &ops_kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  auto store = ops_kernel_manager.ops_kernel_store_["AiCoreLib"];
  ops_kernel_manager.ops_kernel_store_["AiCoreLib"] = fake_ops;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("not_support_check_support", "Abs");
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), FAILED);

  ASSERT_TRUE(op_desc->GetOpEngineName().empty());
  ops_kernel_manager.ops_kernel_store_["AiCoreLib"] = store;
}

TEST_F(GraphCompilerTest, select_engine_when_opdesc_not_empty) {
  std::string engine_name = "engine_name";
  std::string kernel_name = "kernel_name";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  op_desc->SetOpEngineName(engine_name);
  op_desc->SetOpKernelLibName(kernel_name);
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), SUCCESS);
  DNNEngineManager::UpdateOpDescWithOpInfo(node_ptr->GetOpDesc(), op_info);

  std::string attr_engine_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, attr_engine_name);
  ASSERT_EQ(attr_engine_name, engine_name);
  std::string attr_kernel_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, attr_kernel_name);
  ASSERT_EQ(attr_kernel_name, kernel_name);
}

TEST_F(GraphCompilerTest, select_engine_when_opdesc_confilct_with_attr) {
  std::string op_engine_name = "op_engine_name";
  std::string op_kernel_name = "op_kernel_name";
  std::string attr_engine_name = "attr_engine_name";
  std::string attr_kernel_name = "attr_kernel_name";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  op_desc->SetOpEngineName(op_engine_name);
  op_desc->SetOpKernelLibName(op_kernel_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, attr_engine_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, attr_kernel_name);
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), SUCCESS);

  std::string fetched_attr_engine_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, fetched_attr_engine_name);
  ASSERT_EQ(fetched_attr_engine_name, attr_engine_name);
  std::string fetched_attr_kernel_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, fetched_attr_kernel_name);
  ASSERT_EQ(fetched_attr_kernel_name, attr_kernel_name);

  ASSERT_EQ(op_desc->GetOpEngineName(), op_engine_name);
  ASSERT_EQ(op_desc->GetOpKernelLibName(), op_kernel_name);
}

TEST_F(GraphCompilerTest, hccl_sequence_adjust_succ) {
  DEF_GRAPH(sub_1) {
    auto data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
    CHAIN(NODE("data1", data_1)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(sub_2) {
    auto data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
    CHAIN(NODE("data1", data_1)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(root_graph) {
    auto if1 = OP_CFG(IF).InCnt(4).OutCnt(1).Attr(ATTR_NAME_HCCL_FUSED_GROUP, "hccl");
    CTRL_CHAIN(NODE("allreduce1", HCOMALLREDUCE)->NODE("allreduce2", HCOMALLREDUCE));
    CHAIN(NODE("data1", DATA)->NODE("If", if1, sub_1, sub_2)->NODE("output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->NODE("allreduce1")->NODE("If"));
    CHAIN(NODE("data3", DATA)->NODE("allreduce2")->NODE("If"));
    CHAIN(NODE("data4", DATA)->NODE("allreduce3", HCOMALLREDUCE)->NODE("If"));
  };

  auto graph = ToGeGraph(root_graph);
  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION), AscendString("1"));
  GeSession session(options);
  session.AddGraph(1, graph, options);
  auto ret = session.CompileGraph(1, {});
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto allreduce2 = graph->FindNode("allreduce2");
    EXPECT_NE(allreduce2, nullptr);
    auto pre_node = allreduce2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
    EXPECT_NE(pre_node, nullptr);
    if (pre_node->GetType() == IDENTITY) {
      const auto pre_node_in_ctrl_nodes = pre_node->GetInControlNodes();
      EXPECT_EQ(pre_node_in_ctrl_nodes.size(), 3U);
    } else {
      const auto &in_ctrl_nodes = allreduce2->GetInControlNodes();
      EXPECT_EQ(in_ctrl_nodes.size(), 2);
    }
  };
}

TEST_F(GraphCompilerTest, hccl_sequence_adjust_succ_without_optimize) {
  DEF_GRAPH(sub_1) {
    auto data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
    CHAIN(NODE("data1", data_1)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(sub_2) {
    auto data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 2});
    CHAIN(NODE("data1", data_1)->NODE("output", NETOUTPUT));
  };
  DEF_GRAPH(root_graph) {
    auto if1 = OP_CFG(IF).InCnt(2).OutCnt(1).Attr(ATTR_NAME_HCCL_FUSED_GROUP, "hccl");
    CHAIN(NODE("data1", DATA)->NODE("If", if1, sub_1, sub_2)->NODE("output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->NODE("allreduce1", HCOMALLREDUCE)->NODE("If"));
  };

  auto graph = ToGeGraph(root_graph);
  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION), AscendString("1"));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto allreduce1 = graph->FindNode("allreduce1");
    EXPECT_NE(allreduce1, nullptr);
    auto pre_node = allreduce1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
    EXPECT_NE(pre_node, nullptr);
    if (pre_node->GetType() == IDENTITY) {
      const auto pre_node_in_ctrl_nodes = pre_node->GetInControlNodes();
      EXPECT_EQ(pre_node_in_ctrl_nodes.size(), 0U);
    } else {
      const auto &in_ctrl_nodes = allreduce1->GetInControlNodes();
      EXPECT_EQ(in_ctrl_nodes.size(), 0);
    }
  };
}

/*
 *                 data1
 *                /     \
 *      allreduce1.......allreduce2
 *          |                 |
 *        relu1             relu2         data2
 *          |                    \       /
 *      allreduce3                switch1
 *          |                    /       \
 *        relu3             relu4         relu5
 *                               \       /
 *                                merge1
 */
TEST_F(GraphCompilerTest, test_link_allreduce_to_output) {
  DEF_GRAPH(root_graph) {
    auto data2 = OP_CFG(DATA).Attr("index", 1).TensorDesc(FORMAT_ND, DT_BOOL).Build("data2");
    auto allreduce1 = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_HCCL_FUSED_FLAG, true).Build("allreduce1");
    auto allreduce2 = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_HCCL_FUSED_FLAG, true).Build("allreduce2");
    auto allreduce3 = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_HCCL_FUSED_FLAG, true).Build("allreduce3");
    CHAIN(NODE("data1", DATA)->NODE(allreduce1)->NODE("relu1", RELU)->NODE(allreduce2)->NODE("relu3", RELU));
    CHAIN(NODE("data1")->NODE(allreduce2)->NODE("relu2", RELU)->EDGE(0, 0)->NODE("switch1", SWITCH)->EDGE(0, 0)->
          NODE("relu4", RELU)->EDGE(0, 0)->NODE("merge1", MERGE));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE("switch1")->EDGE(1, 0)->NODE("relu5", RELU)->EDGE(0, 1)->NODE("merge1"));
    CTRL_CHAIN(NODE(allreduce1)->NODE(allreduce2));
  };

  auto graph = ToGeGraph(root_graph);
  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION), AscendString("1"));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto output = graph->FindFirstNodeMatchType(NETOUTPUT);
    EXPECT_NE(output, nullptr);
    bool has_ar = false;
    for (const auto &in_node : output->GetInControlNodes()) {
      if (in_node->GetName() == "allreduce2") {
        has_ar = true;
        break;
      }
    }
    EXPECT_TRUE(has_ar == true);
  };
}

/*
 *                 data1
 *                /     \
 *      allreduce1.......allreduce2
 *          |                 |
 *        relu1             relu2
 *          |
 *      allreduce3
 *          |
 *        relu3
 */
TEST_F(GraphCompilerTest, test_link_allreduce_to_output_no_switch) {
  DEF_GRAPH(root_graph) {
    auto allreduce1 = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_HCCL_FUSED_FLAG, true).Build("allreduce1");
    auto allreduce2 = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_HCCL_FUSED_FLAG, true).Build("allreduce2");
    auto allreduce3 = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_HCCL_FUSED_FLAG, true).Build("allreduce3");
    CHAIN(NODE("data1", DATA)->NODE(allreduce1)->NODE("relu1", RELU)->NODE(allreduce2)->NODE("relu3", RELU));
    CHAIN(NODE("data1")->NODE(allreduce2)->NODE("relu2", RELU));
    CTRL_CHAIN(NODE(allreduce1)->NODE(allreduce2));
  };

  auto graph = ToGeGraph(root_graph);
  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION), AscendString("1"));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto output = graph->FindFirstNodeMatchType(NETOUTPUT);
    EXPECT_NE(output, nullptr);
    bool has_ar = false;
    for (const auto &in_node : output->GetInControlNodes()) {
      if (in_node->GetType() == HCOMALLREDUCE) {
        has_ar = true;
        break;
      }
    }
    EXPECT_EQ(has_ar, false);
  };
}

TEST_F(GraphCompilerTest, test_keep_reshape_when_output_require_input_continuous) {
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault().Install(FakeOp(GATHERSHAPES).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction));
  DEF_GRAPH(root_graph) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("reshape", RESHAPE));
    CHAIN(NODE("reshape", RESHAPE)->EDGE(0, 0)->NODE("gathershape", GATHERSHAPES));
    CHAIN(NODE("data3", DATA)->EDGE(0, 1)->NODE("gathershape", GATHERSHAPES));
  };

  DUMP_GRAPH_WHEN("PreRunAfterOptimize1")
  auto graph = ToGeGraph(root_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterOptimize1) {
    auto reshape = graph->FindFirstNodeMatchType(RESHAPE);
    EXPECT_NE(reshape, nullptr);
  };
}

TEST_F(GraphCompilerTest, test_recompute) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  DUMP_GRAPH_WHEN("AfterRecompute")
  make_graph_can_recompute(graph);
  map<AscendString, AscendString> options{
      {RECOMPUTE, "manual"}, {OPTION_GRAPH_RUN_MODE, "1"},
      {AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), AscendString("12800")}};
  Session session(options);
  session.AddGraph(1, ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterRecompute) {
    for (const ge::NodePtr &node : graph->GetDirectNode()) {
      if (node->GetName() == "gradients/matmul") {
        EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Add_recompute_copy");
      }
      if (node->GetName() == "Add_recompute_copy") {
        EXPECT_EQ(node->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
        EXPECT_EQ(node->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(),
                  "gradients/sqrt");
      }
    }
  };
}

TEST_F(GraphCompilerTest, test_small_topo_bp_node_create_cycle_graph) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("recompute_graph");
  DUMP_GRAPH_WHEN("AfterRecompute")
  make_graph_can_recompute1(graph, true);
  map<AscendString, AscendString> options{
      {RECOMPUTE, "manual"}, {OPTION_GRAPH_RUN_MODE, "1"},
      {AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), AscendString("12800")}};
  Session session(options);
  session.AddGraph(1, ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterRecompute) {
    auto bp_pow_1 = graph->FindNode("gradients/pow_1");
    EXPECT_EQ(bp_pow_1 != nullptr, true);
    EXPECT_EQ(bp_pow_1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "pow_1");
    auto pow_recompute_copy = graph->FindNode("pow_recompute_copy");
    EXPECT_EQ(pow_recompute_copy != nullptr, true);
    EXPECT_EQ(pow_recompute_copy->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
    EXPECT_EQ(pow_recompute_copy->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(),
              "gradients/sqrt");
  };
}

TEST_F(GraphCompilerTest, test_auto_recompute) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("recompute_graph");
  make_graph_can_recompute1(graph);
  map<std::string, std::string> options{};
  Session session(options);
  map<std::string, std::string> graph_options{{RECOMPUTE, "auto"}};
  session.AddGraph(1, ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph), graph_options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, FAILED); // 去掉了MDAT的依赖 直接返回不支持
  MmpaStub::GetInstance().Reset();
}

TEST_F(GraphCompilerTest, test_aoe_recompute_dump) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("recompute_graph");
  make_graph_can_recompute1(graph);
  map<std::string, std::string> options{{BUILD_MODE, BUILD_MODE_TUNING}};
  Session session(options);
  map<std::string, std::string> graph_options{{RECOMPUTE, "auto"}, {TUNING_PATH, "./test_aoe.txt"}};
  session.AddGraph(1, ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph), graph_options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_convert_const_to_file_const) {
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  ge::GeTensorPtr tensor1 = std::make_shared<GeTensor>();
  tensor1->MutableTensorDesc().SetShape(GeShape(shape));
  tensor1->SetData(value);
  tensor1->MutableTensorDesc().SetDataType(DT_UINT8);
  ge::GeTensorPtr tensor2 = std::make_shared<GeTensor>();
  tensor2->MutableTensorDesc().SetShape(GeShape(shape));
  tensor2->SetData(value);
  tensor2->MutableTensorDesc().SetDataType(DT_UINT8);

  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).Weight(tensor1).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("const1");
    auto const2 = OP_CFG(CONSTANT).Weight(tensor2).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("const2");
    auto add = OP_CFG(ADD).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("add");
    CHAIN(NODE(const1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->NODE("output", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  map<std::string, std::string> options{{EXTERNAL_WEIGHT, "1"}, {"ge.variableMemoryMaxSize", "536028564"},
    {ge::VARIABLE_MEMORY_MAX_SIZE, "12800"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_convert_const_to_file_const_weight) {
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  ge::GeTensorPtr tensor1 = std::make_shared<GeTensor>();
  tensor1->MutableTensorDesc().SetShape(GeShape(shape));
  tensor1->SetData(value);
  tensor1->MutableTensorDesc().SetDataType(DT_UINT8);
  ge::GeTensorPtr tensor2 = std::make_shared<GeTensor>();
  tensor2->MutableTensorDesc().SetShape(GeShape(shape));
  tensor2->SetData(value);
  tensor2->MutableTensorDesc().SetDataType(DT_UINT8);

  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).Weight(tensor1).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("const1");
    auto const2 = OP_CFG(CONSTANT).Weight(tensor2).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("const2");
    auto add = OP_CFG(ADD).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("add");
    CHAIN(NODE(const1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->NODE("output", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  map<std::string, std::string> options{{EXTERNAL_WEIGHT, "1"}, {"ge.variableMemoryMaxSize", "536028564"},
    {ge::VARIABLE_MEMORY_MAX_SIZE, "12800"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(1);
  EXPECT_NE(summary, nullptr);
  std::vector<ExternalWeightDescPtr> externalWeightPaths;
  summary->GetExternalWeightPaths(externalWeightPaths);
  EXPECT_EQ(externalWeightPaths.size(), 1);
  ExternalWeightDescPtr externalWeightPath = externalWeightPaths[0];
  EXPECT_EQ(externalWeightPath->GetSize(), 3);
  EXPECT_EQ(externalWeightPath->GetOffset(), 0);
}

TEST_F(GraphCompilerTest, test_convert_const_to_constant) {
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  ge::GeTensorPtr tensor1 = std::make_shared<GeTensor>();
  tensor1->MutableTensorDesc().SetShape(GeShape(shape));
  tensor1->SetData(value);
  tensor1->MutableTensorDesc().SetDataType(DT_UINT8);

  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).Weight(tensor1).TensorDesc(FORMAT_ND, DT_UINT8, {3}).Build("const1");
    CHAIN(NODE(const1)->NODE("output", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  map<std::string, std::string> options{{OPTION_CONST_LIFECYCLE, "session"}, {"ge.variableMemoryMaxSize", "536028564"},
                                        {ge::VARIABLE_MEMORY_MAX_SIZE, "12800"}};
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto constant_op = graph->FindFirstNodeMatchType(CONSTANTOP);
    EXPECT_NE(constant_op, nullptr);
  };
}

TEST_F(GraphCompilerTest, test_build_with_before_qos) {
  putenv("OP_NO_REUSE_MEM=data_1,data_2");
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int32_t> input_indexes = {-1};
  std::vector<int64_t> atomic_output_index = {};
  auto hcom = OP_CFG(NETOUTPUT);
  auto data1 = OP_CFG(DATA)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("data_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> output_memory_types = {RT_MEMORY_P2P_DDR};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types);

  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::BUILD_STEP), AscendString(ge::BUILD_STEP_BEFORE_BUILD));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_build_with_after_qos) {
  putenv("OP_NO_REUSE_MEM=data_1,data_2");
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int32_t> input_indexes = {-1};
  std::vector<int64_t> atomic_output_index = {};
  auto hcom = OP_CFG(NETOUTPUT);
  auto data1 = OP_CFG(DATA)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("data_1");
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> output_memory_types = {RT_MEMORY_P2P_DDR};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types);

  map<AscendString, AscendString> options;
  options.emplace(AscendString(ge::BUILD_STEP), AscendString(ge::BUILD_STEP_AFTER_BUILD));
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/**
 *      data0  data1
 *        \   / |   \
 *         add  add add
 *           \  |  /
 *            cast
 */
TEST_F(GraphCompilerTest, test_build_with_atomic_node_merged) {
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kMockSameMemSetAttrs);
  DEF_GRAPH(ge_graph) {
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto node1 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node2 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node3 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto cast = OP_CFG(CAST)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    CHAIN(NODE("data0", data0)->NODE("add1", node1));
    CHAIN(NODE("data1", data1)->NODE("add1", node1));
    CHAIN(NODE("data0", data0)->NODE("add2", node2));
    CHAIN(NODE("data1", data1)->NODE("add2", node2));
    CHAIN(NODE("data0", data0)->NODE("add3", node3));
    CHAIN(NODE("data1", data1)->NODE("add3", node3));
    CHAIN(NODE("add1", node1)->NODE("cast", cast));
    CHAIN(NODE("add2", node2)->NODE("cast", cast));
    CHAIN(NODE("add3", node3)->NODE("cast", cast));
    CHAIN(NODE("cast", cast)->NODE(NODE_NAME_NET_OUTPUT, net_output));
  };
  auto graph = ToGeGraph(ge_graph);
  auto graph1 = ToComputeGraph(ge_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllNodesSize(), 8); // add memset node
    EXPECT_EQ(graph->GetDirectNodesSize(), 8);
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == MEMSET) {
        std::vector<int32_t> data_type_list;
        std::vector<int32_t> int_list;
        std::vector<float32_t> float_list;
        std::vector<int64_t> mem_sizes;
        std::vector<int64_t> mem_start_vector;
        auto op = node->GetOpDesc();
        EXPECT_EQ(AttrUtils::GetListInt(op, ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list), true);
        EXPECT_EQ(AttrUtils::GetListInt(op, ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list), true);
        EXPECT_EQ(AttrUtils::GetListFloat(op, ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT, float_list), false);
        EXPECT_EQ(ge::AttrUtils::GetListInt(op, ATTR_NAME_ATOMIC_MEMSET_SIZES, mem_sizes), true);
        EXPECT_EQ(ge::AttrUtils::GetListInt(op, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector), true);
        EXPECT_EQ(data_type_list.size(), 1);
        EXPECT_EQ(int_list.size(), 1);
        EXPECT_EQ(float_list.size(), 0);
        EXPECT_EQ(mem_sizes.size(), 1);
        EXPECT_EQ(mem_start_vector.size(), 1);
      }
    }
  };
}

/**
 * 用例描述：连续输入内存需要atomic清零，同时需要p2p内存
 *    data0
 *      \
 *   hcomallreduce (input need atomic clean)
 *         \
 *         relu
 * 测试步骤：
 * 1. 构造包含hcom allreduce的图，该算子需要连续输入，并且输入需要清零
 * 2. 创建session，进行图编译
 *
 * 预期结果：
 * 1. 编译成功
 * 2. MemSet算子有workspace_type属性
 * 3. 属性内容为p2p类型
 */
TEST_F(GraphCompilerTest, MemsetConnectToMemoryTypeP2P_CheckMemsetWorkspaceMemType_Correctly) {
  std::vector<int64_t> memtype_list = {RT_MEMORY_P2P_DDR};
  std::vector<int32_t> input_indexes{-1};
  DEF_GRAPH(ge_graph) {
                        auto data0 = OP_CFG(DATA)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_INT32, {16});

                        auto add = OP_CFG(ADD)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_INT32, {16});

                        auto hcomallreduce = OP_CFG(HCOMALLREDUCE)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_INT32, {16})
                            .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                            .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                            .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                            .Attr(ATOMIC_ATTR_INPUT_INDEX, input_indexes);

                        auto relu = OP_CFG(RELU)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_INT32, {16});

                        auto net_output = OP_CFG(NETOUTPUT)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_INT32, {16});

                        CHAIN(NODE("data0", data0)->NODE("add", add)->NODE("hcomallreduce", hcomallreduce)
                                  ->NODE("relu", relu)->NODE(NODE_NAME_NET_OUTPUT, net_output));
                      };

  auto graph = ToGeGraph(ge_graph);
  auto graph1 = GraphUtilsEx::GetComputeGraph(graph);
  const auto hcomallreduce = graph1->FindNode("hcomallreduce");
  ASSERT_TRUE(ge::AttrUtils::SetListInt(hcomallreduce->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, input_indexes));
  ASSERT_TRUE(ge::AttrUtils::SetListInt(hcomallreduce->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list));
  ASSERT_TRUE(ge::AttrUtils::SetListInt(hcomallreduce->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list));

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == MEMSET) {
        std::vector<int64_t> workspace_mem_type_list;
        auto op = node->GetOpDesc();
        auto ret = ge::AttrUtils::GetListInt(op, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_mem_type_list);
        ASSERT_TRUE(ret);
        ASSERT_EQ(workspace_mem_type_list.size(), 1);
        EXPECT_EQ(workspace_mem_type_list[0], RT_MEMORY_P2P_DDR);
      }
    }
  };
}
/**
 * 用例描述：变量直连split, split需要Nopadding连续输出，且输出引用输入
 *
 * 预置条件：
 * 1.构造计算图1，变量直连split
 *
 *  Data    Data   var
 *    \      /      |
 *     add1       split
 *        \        / \
 *         \      add2
 *           \      |
 *             \    |
 *             NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1
 * 2.编译后执行计算图1
 * 预期结果
 * 1. 校验split的输出内存连续
 * 2. var-split insert identity
 * 3. 执行成功，无报错
 */
TEST_F(GraphCompilerTest, VarConnectNoPaddingContinuousOutputNode_RunSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  gert::GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto compute_graph = gert::ShareGraph::BuildVarConnectToSplit();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  EXPECT_EQ(SUCCESS, session.AddGraph(graph_id, graph));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 0);
  TensorDesc desc_2(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_2);

  runtime_stub.Clear();
  Synchronizer sync;
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  sync.WaitFor(5);
  runtime_stub.Clear();
  CHECK_GRAPH(PreRunAfterBuild) {
    auto split_node = graph->FindNode("split");
    ASSERT_NE(split_node, nullptr);
    auto split_output_offsets = split_node->GetOpDesc()->GetOutputOffset();
    int64_t split_output0_size = 0;
    TensorUtils::GetTensorSizeInBytes(split_node->GetOpDesc()->GetOutputDesc(0), split_output0_size);
    EXPECT_EQ(split_output_offsets[1], split_output_offsets[0] + split_output0_size);
    EXPECT_EQ(split_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/**
 * 用例描述：使用1G大页申请变量
 *
 * 预置条件：
 * 1.构造包含变量的图
 *
 * 测试步骤
 * 1.配置变量使用1G大页的option
 * 2.编译后执行计算图1
 * 预期结果
 * 1. 变量内存使用1G大页
 */
TEST_F(GraphCompilerTest, VariableUse1GHugePage_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  gert::GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(AscendString(ge::OPTION_VARIABLE_USE_1G_HUGE_PAGE), "2");
  Session session(options);

  auto compute_graph = gert::ShareGraph::BuildVarConnectToSplit();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  EXPECT_EQ(SUCCESS, session.AddGraph(graph_id, graph));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 0);
  TensorDesc desc_2(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_2);

  runtime_stub.Clear();
  Synchronizer sync;
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  sync.WaitFor(5);
  runtime_stub.Clear();
  CHECK_GRAPH(PreRunAfterBuild) {
    auto split_node = graph->FindNode("split");
    ASSERT_NE(split_node, nullptr);
    auto split_output_offsets = split_node->GetOpDesc()->GetOutputOffset();
    int64_t split_output0_size = 0;
    TensorUtils::GetTensorSizeInBytes(split_node->GetOpDesc()->GetOutputDesc(0), split_output0_size);
    EXPECT_EQ(split_output_offsets[1], split_output_offsets[0] + split_output0_size);
    EXPECT_EQ(split_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}


/**
 *      data0  data1
 *        \   / |   \
 *         add  add add
 *           \  |  /
 *            cast
 */
TEST_F(GraphCompilerTest, test_build_with_atomic_node_no_merge) {
  auto graph_optimizer = MockGraphOptimizer(kGraphOptimizerOption::kMockDifferentMemSetAttrs);
  DEF_GRAPH(ge_graph) {
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto node1 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node2 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node3 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto cast = OP_CFG(CAST)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT64, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    CHAIN(NODE("data0", data0)->NODE("add1", node1));
    CHAIN(NODE("data1", data1)->NODE("add1", node1));
    CHAIN(NODE("data0", data0)->NODE("add2", node2));
    CHAIN(NODE("data1", data1)->NODE("add2", node2));
    CHAIN(NODE("data0", data0)->NODE("add3", node3));
    CHAIN(NODE("data1", data1)->NODE("add3", node3));
    CHAIN(NODE("add1", node1)->NODE("cast", cast));
    CHAIN(NODE("add2", node2)->NODE("cast", cast));
    CHAIN(NODE("add3", node3)->NODE("cast", cast));
    CHAIN(NODE("cast", cast)->NODE(NODE_NAME_NET_OUTPUT, net_output));
  };
  auto graph = ToGeGraph(ge_graph);
  auto graph1 = ToComputeGraph(ge_graph);
  auto cast_node = graph1->FindNode("cast");
  auto output_desc = cast_node->GetOpDesc()->GetOutputDesc(0);
  output_desc.SetDataType(DT_INT64);
  output_desc.SetOriginDataType(DT_INT64);
  cast_node->GetOpDesc()->UpdateOutputDesc(0, output_desc);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllNodesSize(), 8); // add memset node
    EXPECT_EQ(graph->GetDirectNodesSize(), 8);
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == MEMSET) {
        std::vector<int32_t> data_type_list;
        std::vector<int32_t> int_list;
        std::vector<float32_t> float_list;
        std::vector<int64_t> mem_sizes;
        std::vector<int64_t> mem_start_vector;
        auto op = node->GetOpDesc();
        EXPECT_EQ(AttrUtils::GetListInt(op, ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list), true);
        EXPECT_EQ(AttrUtils::GetListInt(op, ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list), true);
        EXPECT_EQ(AttrUtils::GetListFloat(op, ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT, float_list), false);
        EXPECT_EQ(ge::AttrUtils::GetListInt(op, ATTR_NAME_ATOMIC_MEMSET_SIZES, mem_sizes), true);
        EXPECT_EQ(ge::AttrUtils::GetListInt(op, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector), true);
        EXPECT_EQ(data_type_list.size(), 3);
        EXPECT_EQ(int_list.size(), 3);
        EXPECT_EQ(float_list.size(), 0);
        EXPECT_EQ(mem_sizes.size(), 3);
        EXPECT_EQ(mem_start_vector.size(), 3);
      }
    }
  };
}

TEST_F(GraphCompilerTest, SaveSoftSyncOpWeight_success) {
  std::vector<int64_t> shape = {16};
  DEF_GRAPH(add_graph) {
    auto add = OP_CFG(ADDN).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF").TensorDesc(FORMAT_ND, DT_INT32, shape)
        .InCnt(3).Build("add");
    auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data1");
    data1->SetOutputOffset({0});

    int32_t data_value_vec1[2] = {2, 4};
    GeTensorDesc data_tensor_desc(GeShape({2}), FORMAT_ND, DT_INT32);
    TensorUtils::SetDataOffset(data_tensor_desc, 16);
    TensorUtils::SetWeightSize(data_tensor_desc, 16);
    TensorUtils::SetSize(data_tensor_desc, 16);
    GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, 2 * sizeof(int32_t));
    auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add)->NODE("noop", NOOP));
    CHAIN(NODE("const_op", const1)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE("variable", VARIABLE)->EDGE(0, 2)->NODE(add));
  };

  auto graph = ToComputeGraph(add_graph);
  EXPECT_NE(graph, nullptr);

  auto add = graph->FindNode("add");
  EXPECT_NE(add, nullptr);
  auto op_desc = add->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  std::map<std::string, uint32_t> input_name_idx;
  input_name_idx["x"] = 0;
  input_name_idx["axes"] = 1;
  input_name_idx["vvv"] = 2;
  op_desc->UpdateInputName(input_name_idx);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  std::vector<std::string> depends = {"x", "axes", "vvv", "666"};
  AttrUtils::SetListStr(op_desc, "_op_infer_depends", depends);

  auto ge_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  map<AscendString, AscendString> options;
  options.emplace("ge.hardwareInfo", "");
  Session session(options);
  session.AddGraph(1, ge_graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphCompilerTest, test_oom) {
  const map<string, string> options{};
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("recompute_graph");
  make_graph_can_recompute1(graph);
  Session session(options);
  map<std::string, std::string> graph_options{{RECOMPUTE, "auto"}, {"ge.hardwareInfo", "memory_size:1024"}};
  session.AddGraph(1, ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph), graph_options);
  std::vector<InputTensorInfo> inputs;
  VarManager::Instance(session.GetSessionId())->SetMemoryMallocSize(options, 1024UL);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(GraphCompilerTest, GraphPrepare_UpdateConstPlaceHolderByStorageFormat_ok) {
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  std::map<std::string, std::string> init_options;
  instance_ptr->Initialize(init_options);
  auto compute_graph = gert::ShareGraph::BuildSingleConstPlaceHolderGraph(nullptr, 100L);
  auto set_op_desc_nd_to_nz = [](OpDescPtr op_desc) {
    AttrUtils::SetInt(op_desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(FORMAT_FRACTAL_NZ));
    vector<int64_t> storage_shape = {1, 1, 1, 16, 16};
    AttrUtils::SetListInt(op_desc, ATTR_NAME_STORAGE_SHAPE, storage_shape);
  };

  const auto node = compute_graph->FindNode("constplaceholder1");
  ASSERT_NE(node, nullptr);
  const auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  set_op_desc_nd_to_nz(op_desc);

  DUMP_GRAPH_WHEN("PreRunAfterPrepareRunningFormatRefiner");
  auto ge_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, ge_graph, options);
  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);
  CHECK_GRAPH(PreRunAfterPrepareRunningFormatRefiner) {
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == CONSTPLACEHOLDER) {
        auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
        EXPECT_EQ(out_desc->GetFormat(), FORMAT_FRACTAL_NZ);
        vector<int64_t> storage_shape = {1, 1, 1, 16, 16};
        EXPECT_EQ(out_desc->GetShape().GetDims(), storage_shape);
      }
    }
  };
}

// 异常场景覆盖
TEST_F(GraphCompilerTest, NoPaddingContinuousInputsHasMultiOutCheckReusOthers) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("A", "A")
                            ->NODE("B", "B")
                            ->NODE("C", "C")
                            ->NODE("PhonyConcat", "PhonyConcat")
                            ->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("D", "D")->NODE("PhonyConcat", "PhonyConcat"));
                  CHAIN(NODE("C", "C")->NODE("E", "E")->NODE("NetOutput", "NetOutput"));
                };

  auto graph1 = ToGeGraph(g1);
  auto graph = ge::GraphUtilsEx::GetComputeGraph(graph1);
  graph->TopologicalSortingGraph();
  int64_t stream_id = 0;
  for (const auto &node : graph->GetAllNodes()) {
    node->GetOpDescBarePtr()->SetStreamId(stream_id++);
  }

  // no pading continuous inputs
  auto node = graph->FindNode("PhonyConcat");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, true);

  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  auto ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  DependencyAnalyzer nmda = DependencyAnalyzer(graph, anchor_to_symbol, symbol_to_anchors);
  nmda.Init();

  std::unordered_map<std::string, NodePtr> name_to_node;
  for (const auto &a : graph->GetAllNodes()) {
    name_to_node[a->GetName()] = a;
  }

  ASSERT_TRUE(nmda.CanAReuseB(name_to_node["C"].get(), 1U, name_to_node["A"].get(), 0U));
  ASSERT_FALSE(nmda.CanAReuseB(name_to_node["C"].get(), 0U, name_to_node["A"].get(), 0U));
  nmda.WhyACannotReuseB(name_to_node["C"].get(), 0U, name_to_node["A"].get(), 0U);
  nmda.Debug();
}

// 异常场景覆盖
TEST_F(GraphCompilerTest, CheckReusalbe_Success_WithOneSubgraph) {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)
                               ->NODE("B", "B")
                               ->EDGE(0, 0)
                               ->NODE("C", "C")
                               ->NODE("E", "E")
                               ->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("B", "B")->EDGE(0, 0)->NODE("D", "D")->NODE("E", "E"));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)
                            ->NODE("A", "A")
                            ->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("F", "F")
                            ->NODE("netoutput", NETOUTPUT));
                };

  sub_1.Layout();
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  auto ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  DependencyAnalyzer nmda = DependencyAnalyzer(graph, anchor_to_symbol, symbol_to_anchors);
  nmda.Init();

  std::unordered_map<std::string, NodePtr> name_to_node;
  for (const auto &a : graph->GetAllNodes()) {
    name_to_node[a->GetName()] = a;
  }

  nmda.WhyACannotReuseB(name_to_node["E"].get(), 0U, name_to_node["C"].get(), 0U);
  nmda.Debug();
}

// 异常场景覆盖
TEST_F(GraphCompilerTest, NestingSubgraph) {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub2) {
                    CHAIN(NODE("sub2_data", sub_data)->NODE("C", "C")->NODE("sub2_netoutput", NETOUTPUT));
                  };
  DEF_GRAPH(sub3) {
                    CHAIN(NODE("sub3_data", sub_data)
                              ->NODE("D", "D")
                              ->NODE("sub3_netoutput", NETOUTPUT));
                  };
  DEF_GRAPH(sub1) {
                    CHAIN(NODE("sub1_data", sub_data)
                              ->NODE("if", IF, sub2, sub3)
                              ->NODE("sub1_netoutput", NETOUTPUT));
                  };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("p1", PARTITIONEDCALL, sub1)->NODE("netoutput", NETOUTPUT));
                };

  auto root_graph = ToComputeGraph(g1);
  auto sub1_graph = ToComputeGraph(sub1);
  auto sub2_graph = ToComputeGraph(sub2);
  auto sub3_graph = ToComputeGraph(sub3);

  auto p1 = root_graph->FindNode("p1");
  auto if_parent = sub1_graph->FindNode("if");
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(if_parent, nullptr);

  sub1_graph->SetParentGraph(root_graph);
  sub2_graph->SetParentGraph(sub1_graph);
  sub3_graph->SetParentGraph(sub1_graph);
  root_graph->SetParentGraph(nullptr);

  sub1_graph->SetParentNode(p1);
  sub2_graph->SetParentNode(if_parent);
  sub3_graph->SetParentNode(if_parent);

  root_graph->AddSubgraph("sub1_graph", sub1_graph);
  root_graph->AddSubgraph("sub2_graph", sub2_graph);
  root_graph->AddSubgraph("sub3_graph", sub3_graph);

  p1->GetOpDesc()->AddSubgraphName("sub1_graph");
  if_parent->GetOpDesc()->AddSubgraphName("sub2_graph");
  if_parent->GetOpDesc()->AddSubgraphName("sub3_graph");

  p1->GetOpDesc()->SetSubgraphInstanceName(0, "sub1_graph");
  if_parent->GetOpDesc()->SetSubgraphInstanceName(0, "sub2_graph");
  if_parent->GetOpDesc()->SetSubgraphInstanceName(1, "sub3_graph");

  root_graph->TopologicalSorting();

  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  auto ret = GraphUtils::GetRefMapping(root_graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  DependencyAnalyzer nmda = DependencyAnalyzer(root_graph, anchor_to_symbol, symbol_to_anchors);
  nmda.Init();
  nmda.Debug();
}
// 异常场景覆盖
// stream0 a-b-c
// stream2 d-e
TEST_F(GraphCompilerTest, CCannotReuseD_DiffStream_CheckError) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("b", RELU)->NODE("c", RELU));
    CHAIN(NODE("d", RELU)->NODE("e", RELU));
    CHAIN(NODE("b", RELU)->Ctrl()->NODE("d", RELU));
  };
  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(graph);
  MemConflictShareGraph::SetStreamForNodes(graph, 0, {"a", "b", "c"});
  MemConflictShareGraph::SetStreamForNodes(graph, 1, {"d", "e"});

  HybridMemAssigner hybrid_mem_assigner(graph);
  ASSERT_EQ(hybrid_mem_assigner.Assign(), SUCCESS);
  ASSERT_NE(hybrid_mem_assigner.GetReuseChecker(), nullptr);

  const auto b = graph->FindNode("b");
  const auto d = graph->FindNode("d");
  ASSERT_NE(b, nullptr);
  ASSERT_NE(d, nullptr);
  d->GetOpDescBarePtr()->SetOutputOffset(b->GetOpDescBarePtr()->GetOutputOffset());
  EXPECT_NE(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);
}

TEST_F(GraphCompilerTest, CheckEngineTypeSupport_invalid_case) {
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  auto compute_graph = std::make_shared<ComputeGraph>("my_graph");
  auto node = compute_graph->AddNode(op_desc);
  EXPECT_EQ(GeGenerator::CheckEngineTypeSupport(node, ENGINE_SYS), SUCCESS);
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICUBE), SUCCESS);
  OpsKernelManager &ops_kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  std::vector<OpInfo> vec;
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_VECTOR), SUCCESS);
  OpInfo op_info;
  vec.emplace_back(op_info);
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_VECTOR), SUCCESS);
  OpInfo oi;
  oi.engine = "AIcoreEngine";
  oi.opKernelLib = "opKernelLib";
  vec.push_back(oi);
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  auto p = std::make_shared<FakeOpsKernelInfoStore>();
  ops_kernel_manager.ops_kernel_store_["opKernelLib"] = p;
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  if (instance_ptr != nullptr) {
    instance_ptr->Finalize();
  }
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
}

TEST_F(GraphCompilerTest, Hccl_memcpy_pass_insert_assign_test)
{
  GELib::Initialize({});
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);

  GeTensorDesc scalarTensorDesc(GeShape({7, 7, 3, 1}));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto hcomBroadcast_op = std::make_shared<OpDesc>("HcomBroadcast", HCOMBROADCAST);
  (void)ge::AttrUtils::SetBool(hcomBroadcast_op, "_input_mutable", true);
  hcomBroadcast_op->AddInputDesc(scalarTensorDesc);
  hcomBroadcast_op->AddInputDesc(scalarTensorDesc);
  hcomBroadcast_op->AddOutputDesc(scalarTensorDesc);
  auto hcomBroadcast_node = graph->AddNode(hcomBroadcast_op);

  HcclMemcpyPass hcclMemcpyPass;
  Status ret = hcclMemcpyPass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(scalarTensorDesc);
  variable_op->AddInputDesc(scalarTensorDesc);
  auto variable_node = graph->AddNode(variable_op);

  int32_t data_value_vec1[2] = {2, 4};
  GeTensorDesc data_tensor_desc(GeShape({2}), FORMAT_ND, DT_INT32);
  TensorUtils::SetDataOffset(data_tensor_desc, 16);
  TensorUtils::SetWeightSize(data_tensor_desc, 16);
  TensorUtils::SetSize(data_tensor_desc, 16);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, 2 * sizeof(int32_t));

  auto const_op = std::make_shared<OpDesc>("Const", CONSTANT);
  const_op->AddOutputDesc(data_tensor_desc);
  const_op->AddInputDesc(data_tensor_desc);
  auto const_node = graph->AddNode(const_op);
  AttrUtils::SetTensor(const_op, ATTR_NAME_WEIGHTS, data_tensor1);
  (void) GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), hcomBroadcast_node->GetInDataAnchor(0));
  (void) GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), hcomBroadcast_node->GetInDataAnchor(1));

  DUMP_GRAPH_WHEN("PrepareAfterPrepareOptimize");
  auto ge_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, ge_graph, options);
  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);
  CHECK_GRAPH(PrepareAfterPrepareOptimize) {
    auto assgin = graph->FindFirstNodeMatchType("Assign");
    EXPECT_NE(assgin, nullptr);
  };

  if (instance_ptr != nullptr) {
    instance_ptr->Finalize();
  }
}

TEST_F(GraphCompilerTest, test_graph_with_autofuse) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910B1>());
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  std::map<std::string, std::string> options;
  GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  auto graph = cg::BuildAbsAddReluReluGraph({4, 5, 6});
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);

  auto asc_node = graph->FindFirstNodeMatchType("AscBackend");
  EXPECT_NE(asc_node, nullptr);
  auto op_desc = asc_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  auto out_desc = op_desc->GetOutputDescPtr(0);
  EXPECT_NE(out_desc, nullptr);
  EXPECT_NE(out_desc->GetAttrsGroup<SymbolicDescAttr>(), nullptr);
  EXPECT_EQ(out_desc->GetAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.GetOriginSymbolShape().GetDims(),
            std::vector<Expression>({Symbol(4), Symbol(5), Symbol(6)}));

  auto data_node = graph->FindFirstNodeMatchType(DATA);
  EXPECT_NE(data_node, nullptr);
  op_desc = data_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  out_desc = op_desc->GetOutputDescPtr(0);
  EXPECT_NE(out_desc, nullptr);
  EXPECT_NE(out_desc->GetAttrsGroup<SymbolicDescAttr>(), nullptr);
  EXPECT_EQ(out_desc->GetAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.GetOriginSymbolShape().GetDims(),
            std::vector<Expression>({Symbol(4), Symbol(5), Symbol(6)}));

  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
  unsetenv("LD_LIBRARY_PATH");
  ge::GetThreadLocalContext().SetGlobalOption(global_options);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::GetThreadLocalContext().SetSessionOption(session_options);
}

TEST_F(GraphCompilerTest, test_graph_with_autofuse_op_precisious) {
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  std::map<std::string, std::string> options;
  GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  DUMP_GRAPH_WHEN("AutoFuser_BeforeAutoFuse");
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  auto graph = cg::BuildNeedInsertCastGraph({4, 5, 6});
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);

  CHECK_GRAPH(AutoFuser_BeforeAutoFuse) {
    EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(
              {{"Data", 1},
              {"Cast", 1},
              {"StridedSliceD", 4},
              {"NetOutput", 1}}),
              "success");
  };

  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
  unsetenv("LD_LIBRARY_PATH");
  ge::GetThreadLocalContext().SetGlobalOption(global_options);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::GetThreadLocalContext().SetSessionOption(session_options);
}

TEST_F(GraphCompilerTest, test_graph_with_autofuse_op_precisious_with_subgraph) {
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  std::map<std::string, std::string> options;
  GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  DUMP_GRAPH_WHEN("AutoFuser_BeforeAutoFuse");
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  auto graph = gert::ShareGraph::BuildNeedInsertCastGraphWithSubgraph();
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td0;
  inputs.emplace_back(td0);
  GeTensorDesc td1;
  td1.SetShape((GeShape({4, 4})));
  td1.SetOriginShape((GeShape({4, 4})));
  inputs.emplace_back(td1);
  GeTensorDesc td2;
  td2.SetShape((GeShape({4, 4})));
  td2.SetOriginShape((GeShape({4, 4})));
  inputs.emplace_back(td2);
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);

  EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(
    {{"Data", 3},
    {"Add", 1},
    {"If", 1},
    {"NetOutput", 1}}),
    "success");

  for (const auto &subgraph: graph->GetAllSubgraphs()) {
    EXPECT_EQ(gert::SummaryChecker(subgraph).StrictDirectNodeTypes(
        {{"Data", 1},
        {"Relu", 1},
        {"NetOutput", 1}}),
        "success");
  }
  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
  unsetenv("LD_LIBRARY_PATH");
  ge::GetThreadLocalContext().SetGlobalOption(global_options);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::GetThreadLocalContext().SetSessionOption(session_options);
}

TEST_F(GraphCompilerTest, test_graph_with_autofuse_reshape_handle) {
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  std::map<std::string, std::string> options;
  GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  DUMP_GRAPH_WHEN("AutoFuser_BeforeAutoFuse");
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  auto graph = gert::ShareGraph::ReshapeAbnormalGraph();
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td0;
  td0.SetShape((GeShape({1, 1, 4, 4})));
  td0.SetOriginShape((GeShape({1, 1, 4, 4})));
  inputs.emplace_back(td0);
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);
  CHECK_GRAPH(AutoFuser_BeforeAutoFuse) {
    EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(
      {{"Reshape", 2},
       {"Abs", 1},
       {"Cast", 1},
       {"Const", 2},
       {"Data", 1},
       {"NetOutput", 1}}),
       "success");
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() == "Reshape") {
        EXPECT_EQ(node->GetInNodes().size(), 2);
      }
    }
  };
  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
  unsetenv("LD_LIBRARY_PATH");
  ge::GetThreadLocalContext().SetGlobalOption(global_options);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::GetThreadLocalContext().SetSessionOption(session_options);
}

TEST_F(GraphCompilerTest, test_graph_with_autofuse_withcontrolgraph) {
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  std::map<std::string, std::string> options;
  GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  auto graph = gert::ShareGraph::IfGraph();
  EXPECT_NE(graph, nullptr);
  auto input_node = graph->FindNode("input");
  ASSERT_NE(input_node, nullptr);
  auto input_op_desc = input_node->GetOpDesc();
  ASSERT_NE(input_op_desc,  nullptr);
  input_op_desc->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1}));
  input_op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  input_op_desc->MutableInputDesc(0)->SetDataType(DT_INT64);
  input_op_desc->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1}));
  input_op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  input_op_desc->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto pred_node = graph->FindNode("pred");
  ASSERT_NE(pred_node, nullptr);
  auto pred_op_desc = pred_node->GetOpDesc();
  ASSERT_NE(pred_op_desc,  nullptr);
  pred_op_desc->MutableInputDesc(0)->SetShape(GeShape());
  pred_op_desc->MutableInputDesc(0)->SetOriginShape(GeShape());
  pred_op_desc->MutableInputDesc(0)->SetDataType(DT_BOOL);
  pred_op_desc->MutableOutputDesc(0)->SetShape(GeShape());
  pred_op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape());
  pred_op_desc->MutableOutputDesc(0)->SetDataType(DT_BOOL);

  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td0;
  td0.SetShape((GeShape()));
  td0.SetOriginShape((GeShape()));
  GeTensorDesc td1;
  td1.SetShape((GeShape({4, 5, 6})));
  td1.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td0);
  inputs.emplace_back(td1);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);

  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
  unsetenv("LD_LIBRARY_PATH");
  ge::GetThreadLocalContext().SetGlobalOption(global_options);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::GetThreadLocalContext().SetSessionOption(session_options);
}

TEST_F(GraphCompilerTest, test_graph_with_autofuse_single_op) {
  auto global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  std::map<std::string, std::string> options;
  GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto session_0_var_manager = VarManager::Instance(0);
  session_0_var_manager->Init(0,0,0,0);
  auto graph = cg::BuildAbsAddReluReluGraph({-1, -1, -1});
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);

  auto data_node = graph->FindFirstNodeMatchType(DATA);
  EXPECT_NE(data_node, nullptr);
  auto op_desc = data_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  auto out_desc = op_desc->GetOutputDescPtr(0);
  EXPECT_NE(out_desc, nullptr);
  EXPECT_EQ(out_desc->GetAttrsGroup<SymbolicDescAttr>(), nullptr);

  session_0_var_manager->Destory();
  VarManagerPool::Instance().RemoveVarManager(0);
  unsetenv("AUTOFUSE_FLAGS");
  ge::GetThreadLocalContext().SetGlobalOption(global_options);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::GetThreadLocalContext().SetSessionOption(session_options);
}

TEST_F(GraphCompilerTest, test_UpdateInputWithHintShape_hint_shape_not_empty_all_inputs_unknown_shape) {
  GraphManager graph_manager;
  std::vector<GeShape> hint_shape;
  hint_shape.emplace_back(GeShape({2, 1, 3}));
  hint_shape.emplace_back(GeShape({2, 1, 3}));
  std::vector<GeTensor> inputs;
  GeTensorDesc tensor;
  tensor.SetShape(GeShape({-1, 1, 3}));
  tensor.SetOriginShape(GeShape({-1, 1, 3}));
  inputs.emplace_back(tensor);
  inputs.emplace_back(tensor);
  EXPECT_EQ(graph_manager.UpdateInputWithHintShape(hint_shape, inputs), SUCCESS);
}

TEST_F(GraphCompilerTest, test_UpdateInputWithHintShape_hint_shape_not_empty_one_input_unknown_shape) {
  GraphManager graph_manager;
  std::vector<GeShape> hint_shape;
  hint_shape.emplace_back(GeShape({2, 1, 3}));
  hint_shape.emplace_back(GeShape({2, 1, 3}));
  std::vector<GeTensor> inputs;
  GeTensorDesc tensor;
  tensor.SetShape(GeShape({-1, 1, 3}));
  tensor.SetOriginShape(GeShape({-1, 1, 3}));
  inputs.emplace_back(tensor);
  GeTensorDesc tensor1;
  tensor1.SetShape(GeShape({2, 1, 3}));
  tensor1.SetOriginShape(GeShape({2, 1, 3}));
  inputs.emplace_back(tensor1);
  EXPECT_EQ(graph_manager.UpdateInputWithHintShape(hint_shape, inputs), SUCCESS);
}

TEST_F(GraphCompilerTest, test_UpdateInputWithHintShape_hint_shape_error) {
  GraphManager graph_manager;
  std::vector<GeShape> hint_shape;
  hint_shape.emplace_back(GeShape({2, 1, 3}));
  std::vector<GeTensor> inputs;
  GeTensorDesc tensor;
  tensor.SetShape(GeShape({-1, 1, 3}));
  tensor.SetOriginShape(GeShape({-1, 1, 3}));
  inputs.emplace_back(tensor);
  inputs.emplace_back(tensor);
  EXPECT_NE(graph_manager.UpdateInputWithHintShape(hint_shape, inputs), SUCCESS);
}