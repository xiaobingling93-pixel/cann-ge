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

#include <memory>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>
#include <future>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <condition_variable>

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/graph_manager.h"
#include "api/gelib/gelib.h"
#include "engines/manager/engine/engine_manager.h"

#include "graph/graph.h"
#include "common/context/local_context.h"
#include "common/op/transop_util.h"
#include "graph/ge_context.h"
#include "graph/partition/dynamic_shape_partition.h"
#include "graph/passes/variable_optimize/variable_op_pass.h"
#include "graph/utils/tensor_adapter.h"
#include "api/aclgrph/option_utils.h"
#include "formats/utils/formats_trans_utils.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/ops_stub.h"
#include "ge_attr_value.h"
#include "graph/manager/graph_context.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "graph/execute/model_executor.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge/ge_api.h"
#include "common/share_graph.h"
#include "ge/ge_api_error_codes.h"
#include "common/model/ge_model.h"
#include "framework/executor/ge_executor.h"
#include "common/model/external_allocator_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "host_kernels/kernel_factory.h"
#include "host_kernels/kernel.h"
#include "common/env_path.h"
#include "stub/gert_runtime_stub.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/compliant_share_graph.h"
#include "common/mem_conflict_share_graph.h"
#include "faker/space_registry_faker.h"
#include "graph/optimize/autofuse/autofuse_optimize.h"
#include "compiler/graph/build/graph_compile_summary_impl.h"

using namespace std;
using namespace testing;
using namespace domi;

namespace {
const char *kCast = "Cast";
const uint32_t kNotAdded = 0;
const uint32_t kStartAdd = 1;
const uint32_t kDoneAdded = 2;
}

namespace ge {
namespace {
Graph BuildHCCLGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("hcom1", HCOMALLREDUCE)->NODE("relu1", RELU)->NODE("output", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("hcom1")->NODE("relu2", RELU)->NODE("output"));
                };
  return ToGeGraph(g1);
}
void CreateSummaryCompiledModelWithStreamInfo(GraphNodePtr &graph_node, GeModelPtr &ge_model) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(BuildHCCLGraph());;
  auto hcom1 = compute_graph->FindNode("hcom1");
  hcom1->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetInt(hcom1->GetOpDesc(), "used_stream_num", 3);
  AttrUtils::SetStr(hcom1->GetOpDesc(), public_attr::USER_STREAM_LABEL, "aaa");
  AttrUtils::SetInt(hcom1->GetOpDesc(), "_logic_stream_id", 1);
  hcom1->GetOpDesc()->SetAttachedStreamIds({3});
  auto relu1 = compute_graph->FindNode("relu1");
  relu1->GetOpDesc()->SetStreamId(0);
  AttrUtils::SetInt(relu1->GetOpDesc(), "_logic_stream_id", 0);
  auto relu2 = compute_graph->FindNode("relu2");
  relu2->GetOpDesc()->SetStreamId(2);
  AttrUtils::SetInt(relu2->GetOpDesc(), "_logic_stream_id", 2);
  std::string split_logic_stream_2_origin_logic_stream = "0:0,1:1,2:2,3:3";
  AttrUtils::SetStr(compute_graph, "_split_logic_stream_2_origin_logic_stream",
                    split_logic_stream_2_origin_logic_stream);
  AttrUtils::SetStr(compute_graph, "_custom_logical_stream_ids", "2");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 4);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);
}
}
class TestCastKernel : public ge::Kernel {
  public:
    Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                  std::vector<ge::GeTensorPtr> &v_output) override {
      auto output = std::make_shared<GeTensor>(std::move(*input[0]));
      output->MutableTensorDesc().SetDataType(DT_FLOAT16);
      v_output.push_back(output);
      return SUCCESS;
    }
};

namespace graph_manager_ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  NodePtr AddNode(const std::string &name, const std::string &type,
                  std::initializer_list<std::string> input_names,
                  std::initializer_list<std::string> output_names,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};

NodePtr GraphBuilder::AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt, Format format,
                              DataType data_type, std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginFormat(format);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }

  return graph_->AddNode(op_desc);
}
void GraphBuilder::AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
}
void GraphBuilder::AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
  GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
}
NodePtr GraphBuilder::AddNode(const string &name, const string &type, std::initializer_list<std::string> input_names,
                              std::initializer_list<std::string> output_names, Format format, DataType data_type,
                              std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginFormat(format);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (auto &input_name : input_names) {
    op_desc->AddInputDesc(input_name, tensor_desc->Clone());
  }
  for (auto &output_name :output_names) {
    op_desc->AddOutputDesc(output_name, tensor_desc->Clone());
  }

  return graph_->AddNode(op_desc);
}

/*                                  -------------------------
*                                  |  partitioncall_0_const1* |
*     partitioncall_0--------------|             |           |
*           |                      |          netoutput      |
*           |                      --------------------------
*           |                       ------------------         -------------
*           |                      |        data      |       |    data     |
*           |                      |          |       |       |     |       |
*     partitioncall_1--------------|        case -----|-------|   squeeze*  |
*                                  |          |       |       |     |       |
*                                  |      netoutput   |       |  netoutput  |
*                                   ------------------         -------------
*/
ComputeGraphPtr BuildGraphPartitionCall() {
  auto root_builder = graph_manager_ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = graph_manager_ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  // 2.build partitioncall_1 sub graph
  auto p2_sub_builder = graph_manager_ut::GraphBuilder("partitioncall_1_sub");
  const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &partitioncall_1_case = p2_sub_builder.AddNode("partitioncall_1_case", "Case", 1, 1);
  const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", NETOUTPUT, 1, 1);
  p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_case, 0);
  p2_sub_builder.AddDataEdge(partitioncall_1_case, 0, partitioncall_1_netoutput, 0);
  const auto &sub_graph2 = p2_sub_builder.GetGraph();
  sub_graph2->SetParentNode(partitioncall_1);
  sub_graph2->SetParentGraph(root_graph);
  partitioncall_1->GetOpDesc()->AddSubgraphName("f");
  partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");

  // 2.1 build case sub graph
  auto case_sub_builder = graph_manager_ut::GraphBuilder("case_sub");
  const auto &case_data = case_sub_builder.AddNode("case_data", DATA, 0, 1);
  AttrUtils::SetInt(case_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &case_squeeze = case_sub_builder.AddNode("case_squeeze", SQUEEZE, 1, 1);
  const auto &case_netoutput = case_sub_builder.AddNode("case_netoutput", NETOUTPUT, 1, 1);
  case_sub_builder.AddDataEdge(case_data, 0, case_squeeze, 0);
  case_sub_builder.AddDataEdge(case_squeeze, 0, case_netoutput, 0);
  const auto &case_sub_graph = case_sub_builder.GetGraph();
  case_sub_graph->SetParentNode(partitioncall_1_case);
  case_sub_graph->SetParentGraph(sub_graph2);
  partitioncall_1_case->GetOpDesc()->AddSubgraphName("branches");
  partitioncall_1_case->GetOpDesc()->SetSubgraphInstanceName(0, "case_sub");

  root_graph->AddSubgraph(case_sub_graph->GetName(), case_sub_graph);
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);
  return root_graph;
}

class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    block_ = new (std::nothrow) MemBlock(*this, &mem, size);
    return block_;
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    block_ = new (std::nothrow) MemBlock(*this, &mem, size);
    advise_cnt++;
    return block_;
  }
  void Free(MemBlock *block) override {
    delete block;
    if (block == block_) {
      block_ = nullptr;
    }
  }
  MemBlock *GetBlockAddr() {
    return block_;
  }
  uint64_t GetAdviseCnt() {
    return advise_cnt;
  }
 private:
  uint64_t mem = 0;
  MemBlock *block_{nullptr};
  uint64_t advise_cnt = 0U;
};
}  // namespace graph_manager_ut

using namespace graph_manager_ut;
const char *const kKernelLibName = "DNN_VM_GE_LOCAL";
class UtestGraphManagerTest : public testing::Test {
 protected:
  const std::string run_data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/st/st_run_data/"});
  static void SetUpTestSuite() {
    SetLocalOmgContext(domi::GetContext());
    GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
  }
  static void TearDownTestSuite() {
  }

  class FakeOpsKernelInfoStore : public OpsKernelInfoStore {
   public:
    FakeOpsKernelInfoStore(){supported_ = true;};
    bool supported_;

   private:
    Status Initialize(const std::map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
      return supported_;
    };
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override {};
  };

  class FakeOpsKernelBuilder : public OpsKernelBuilder {
   public:
    FakeOpsKernelBuilder() = default;
   private:
    Status Initialize(const map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    Status CalcOpRunningParam(Node &node) override {
      return SUCCESS;
    };
    Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
      domi::TaskDef task_def;
      tasks.push_back(task_def);
      return SUCCESS;
    };
  };
  void InitGeLib() {
    map<string, string> options;
    Status ret = ge::GELib::Initialize(options);
    EXPECT_EQ(ret, SUCCESS);
    auto instance_ptr = ge::GELib::GetInstance();
    EXPECT_NE(instance_ptr, nullptr);

    //  SchedulerConf conf;
    SchedulerConf scheduler_conf;
    scheduler_conf.name = kKernelLibName;
    scheduler_conf.cal_engines[kKernelLibName] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines[kKernelLibName]->name = kKernelLibName;
    scheduler_conf.cal_engines[kKernelLibName]->scheduler_id = kKernelLibName;
    map<string, SchedulerConf> scheduler_confs;
    scheduler_confs["scheduler"] = scheduler_conf;
    instance_ptr->DNNEngineManagerObj().schedulers_[kKernelLibName] = scheduler_conf;

    OpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FakeOpsKernelInfoStore>();
    OpsKernelManager::GetInstance().ops_kernel_store_.emplace(kKernelLibName, ops_kernel_info_store_ptr);
    OpsKernelBuilderPtr fake_builder = std::make_shared<FakeOpsKernelBuilder>();
    OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelLibName] = fake_builder;
    OpInfo op_info;
    op_info.engine = kKernelLibName;
    op_info.opKernelLib = kKernelLibName;
    OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
    ops_kernel_manager.ops_kernel_info_[DATA].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[ADD].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[NETOUTPUT].emplace_back(op_info);
  }

  void FinalizeGeLib() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
    OpsKernelBuilderRegistry::GetInstance().Unregister(kKernelLibName);
  }
  void SetUp() override {
    const auto env_ptr = getenv("LD_PRELOAD");
    if (env_ptr != nullptr) {
      env = env_ptr;
      unsetenv("LD_PRELOAD");
    }
  }
  void TearDown() override {
    if (!env.empty()) {
      setenv("LD_PRELOAD", env.c_str(), 1);
    }
  }
  std::string env;
};

class StubExecutor : public Executor {
 public:
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                   const rtStream_t stream = nullptr) override {
    return SUCCESS;
  }

  Status UnloadGraph(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) override {
    return SUCCESS;
  }

  Status PushRunArgs(const std::shared_ptr<RunArgs> &args) override {
    return SUCCESS;
  }

  Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                  const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) override {
    return SUCCESS;
  }

  Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, const rtStream_t stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) override {
    return SUCCESS;
  }

  Status ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                               rtStream_t const stream, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs) override {
    return SUCCESS;
  }

  Status UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base, const size_t size) override {
    (void)graph_node;
    mem_base_ = mem_base;
    mem_base_size_ = size;
    return SUCCESS;
  }

  Status PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                    const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) override {
    return SUCCESS;
  }
  uintptr_t mem_base_;
  size_t mem_base_size_;
};

class StubExecutorFail : public Executor {
 public:
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                   const rtStream_t stream = nullptr) override {
    return FAILED;
  }

  Status UnloadGraph(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) override {
    return FAILED;
  }

  Status PushRunArgs(const std::shared_ptr<RunArgs> &args) override {
    return FAILED;
  }

  Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                  const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) override {
    return SUCCESS;
  }

  Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, const rtStream_t stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) override {
    return FAILED;
  }

  Status ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                               rtStream_t const stream, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs) override {
    return FAILED;
  }

  Status PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                    const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) override {
    return SUCCESS;
  }
};

void SetSubGraph(ComputeGraphPtr &root_graph, NodePtr &parent_node, const std::string &name) {
  auto subgraph = std::make_shared<ComputeGraph>(name);
  subgraph->SetParentGraph(root_graph);
  subgraph->SetParentNode(parent_node);
  auto op_desc = parent_node->GetOpDesc();
  op_desc->AddSubgraphName(name);
  op_desc->SetSubgraphInstanceName(0, name);
  root_graph->AddSubgraph(name, subgraph);
}

void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}

/*      Data
 *       |
 *      Relu       Const
 *       |
 *    Netoutput
 */

ge::ComputeGraphPtr CreateGraphWithIsolatedConst() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto relu = builder.AddNode("addn1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);

  builder.AddDataEdge(data, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 0);
  return builder.GetGraph();
}

/*      Data
 *       |
 *      cast1
 *       |     \
 *       |     cast2
 *       |    /
 *    Netoutput
 */

ge::ComputeGraphPtr CreateGraphWithNullOutput() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto cast1 = builder.AddNode("cast1", "Cast", 1, 2);
  auto cast2 = builder.AddNode("cast2", "Cast", 1, 2);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);

  builder.AddDataEdge(data, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, netoutput, 0);
  builder.AddDataEdge(cast1, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, netoutput, 1);
  return builder.GetGraph();
}

/*
 * netoutput
 *   |
 * aipp_data
 * */
ComputeGraphPtr BuildAippDataGraph() {
  ge::ut::GraphBuilder builder("aipp_graph");
  auto aipp_data = builder.AddNode("aipp_data", ge::AIPPDATA, 1, 1);
  auto netoutput = builder.AddNode("Node_Output", ge::NETOUTPUT, 1, 0);
  builder.AddDataEdge(aipp_data, 0, netoutput, 0);
  aipp_data->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
  aipp_data->GetOpDesc()->SetOutputOffset({1});
  netoutput->GetOpDesc()->SetInputOffset({1});
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithFrameworkOp() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "FrameworkOp", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithConstOutput() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  data->GetOpDesc()->SetOutputOffset({1});
  netoutput->GetOpDesc()->SetInputOffset({1});
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithConstOneAndOutputTwo() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto netoutput1 = builder.AddNode("Node_Output1", "NetOutput", 1, 0);
  auto netoutput2 = builder.AddNode("Node_Outpu2", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput1, 0);
  builder.AddDataEdge(data, 0, netoutput2, 0);
  data->GetOpDesc()->SetOutputOffset({1});
  netoutput1->GetOpDesc()->SetInputOffset({1});
  netoutput2->GetOpDesc()->SetInputOffset({1});
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithConstTwoAndOutputOne() {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto netoutput = builder.AddNode("Node_Output1", "NetOutput", 1, 0);
  builder.AddDataEdge(data1, 0, netoutput, 0);
  builder.AddDataEdge(data2, 0, netoutput, 0);
  data1->GetOpDesc()->SetOutputOffset({1});
  data2->GetOpDesc()->SetOutputOffset({1});
  netoutput->GetOpDesc()->SetInputOffset({1});
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithConstTwoAndoutputTwo() {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto netoutput1 = builder.AddNode("Node_Output1", "NetOutput", 1, 0);
  auto netoutput2 = builder.AddNode("Node_Outpu2", "NetOutput", 1, 0);
  builder.AddDataEdge(data1, 0, netoutput1, 0);
  builder.AddDataEdge(data2, 0, netoutput2, 0);
  data1->GetOpDesc()->SetOutputOffset({1});
  data2->GetOpDesc()->SetOutputOffset({2});
  netoutput1->GetOpDesc()->SetInputOffset({1});
  netoutput2->GetOpDesc()->SetInputOffset({2});
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Variable", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput1() {
  ge::ut::GraphBuilder builder("graph1");
  auto data1 = builder.AddNode("data1", "Variable", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto relu = builder.AddNode("addn1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  builder.AddDataEdge(data2, 0, relu, 0);
  builder.AddDataEdge(data1, 0, netoutput, 0);
  builder.AddDataEdge(relu, 0, netoutput, 1);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput2() {
  ge::ut::GraphBuilder builder("graph1");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  compute_graph->RemoveNode(data);
  return compute_graph;
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput3() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Variable", 1, 1);
  auto hcom = builder.AddNode("hcom", "HcomBroadcast", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, hcom, 0);
  builder.AddDataEdge(hcom, 0, netoutput, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput4() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Variable", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto hcom = builder.AddNode("hcom", "HcomBroadcast", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, hcom, 0);
  builder.AddDataEdge(data2, 0, hcom, 1);
  builder.AddDataEdge(hcom, 0, netoutput, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput5() {
  ge::ut::GraphBuilder builder("graph");
  auto assign = builder.AddNode("assign", "Assign", 1, 1);
  auto data = builder.AddNode("data1", "Variable", 1, 1);
  auto hcom = builder.AddNode("hcom", "HcomBroadcast", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, assign, 0);
  builder.AddDataEdge(assign, 0, hcom, 0);
  builder.AddDataEdge(hcom, 0, netoutput, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput6() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Variable", 1, 1);
  auto cast = builder.AddNode("cast", "Cast", 1, 1);
  data->GetOpDesc()->SetAttr("CheckPointGraphForGetVar", GeAttrValue::CreateFrom<int64_t>(6));
  builder.AddDataEdge(data, 0, cast, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput7() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("const", "Constant", 1, 1);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);
  builder.AddDataEdge(data, 0, relu, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithVariableOutput8() {
  ge::ut::GraphBuilder builder("graph");
  auto cast = builder.AddNode("cast", "Cast", 1, 1);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);
  builder.AddDataEdge(cast, 0, relu, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphNoOutput() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);
  builder.AddDataEdge(data, 0, relu, 0);
  return builder.GetGraph();
}

ge::Status callbackFuncGertTensor1(uint32_t a, const std::map<AscendString, gert::Tensor> &b)
{
  return SUCCESS;
}

ge::Status callbackFuncGertTensor2(uint32_t a, const std::map<AscendString, gert::Tensor> &b)
{
  return SUCCESS;
}

ge::ComputeGraphPtr CreateGraphPipelineParitioned() {
  ge::ut::GraphBuilder builder("root_graph");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 2, 1);
  auto partitioned_call_2 = builder.AddNode("PartitionedCall2", PARTITIONEDCALL, 2, 1);
  auto netoutput = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(data2, 0, partitioned_call_1, 1);
  builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_2, 0);
  builder.AddDataEdge(data2, 0, partitioned_call_2, 1);
  builder.AddDataEdge(partitioned_call_2, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();
  SetSubGraph(root_graph, partitioned_call_1, "subgraph1");
  SetSubGraph(root_graph, partitioned_call_2, "subgraph2");
  return root_graph;
}

TEST_F(UtestGraphManagerTest, test_add_graph_sub) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);
  ComputeGraphPtr compute_graph = std::shared_ptr<ge::ComputeGraph>(GraphUtilsEx::GetComputeGraph(graph));

  Graph root_graph("root_graph");
  CreateGraph(root_graph);
  ComputeGraphPtr root_compute_graph = std::shared_ptr<ge::ComputeGraph>(GraphUtilsEx::GetComputeGraph(root_graph));

  root_compute_graph->AddSubGraph(compute_graph);


  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, root_graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_CheckEngineName) {
  GraphManager graph_manager;
  auto p1 = std::make_shared<GELib>();

  Status status = graph_manager.CheckEngineName("test", "test", std::map<std::string, int>({{"test",1}}));
  EXPECT_EQ(status, ge::PARAM_INVALID);
  status = graph_manager.CheckEngineName("", "test", std::map<std::string, int>());
  EXPECT_EQ(status, ge::GE_GRAPH_OPTIONS_INVALID);
  status = graph_manager.CheckEngineName("test", "test", std::map<std::string, int>());
  EXPECT_EQ(status, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_ParseParallelNum) {
  GraphManager graph_manager;
  const std::string paralled_num1 = "8";
  const std::string paralled_num2 = "0";
  const std::string paralled_num3 = "";
  const std::string paralled_num4 = " ";
  const std::string key = "a";
  int32_t num;
  Status status = graph_manager.ParseParallelNum(paralled_num1, key, num);
  EXPECT_EQ(status, ge::SUCCESS);
  status = graph_manager.ParseParallelNum(paralled_num2, key, num);
  EXPECT_EQ(status, ge::GE_GRAPH_OPTIONS_INVALID);
  status = graph_manager.ParseParallelNum(paralled_num3, key, num);
  EXPECT_EQ(status, ge::GE_GRAPH_OPTIONS_INVALID);
  status = graph_manager.ParseParallelNum(paralled_num4, key, num);
  EXPECT_EQ(status, ge::GE_GRAPH_OPTIONS_INVALID);
}

TEST_F(UtestGraphManagerTest, test_SetAttrForHcomBroadCastOp) {
  GraphManager graph_manager;

  Graph graph("test_graph");
  CreateGraph(graph);

  ComputeGraphPtr compute_graph = std::shared_ptr<ge::ComputeGraph>(GraphUtilsEx::GetComputeGraph(graph));

  EXPECT_NO_THROW(graph_manager.SetAttrForHcomBroadCastOp(compute_graph));
}

TEST_F(UtestGraphManagerTest, test_SubgraphPartitionAndOptimizationFailed) {
  // need build while buildflag is true, var format changed
  GraphId graph_id = 1;
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);

  Graph root_graph("root_graph");
  CreateGraph(root_graph);

  GraphUtilsEx::GetComputeGraph(root_graph)->AddSubGraph(GraphUtilsEx::GetComputeGraph(graph));
  ComputeGraphPtr compute_graph = std::shared_ptr<ge::ComputeGraph>(GraphUtilsEx::GetComputeGraph(root_graph));
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);

  OpsKernelManager::GetInstance().composite_engines_["test"] = std::set<string>({"test"});

  Status status = graph_manager.SubgraphPartitionAndOptimization(graph_node, compute_graph, 1, EnginePartitioner::Mode::kCompositeEnginePartitioning);
  EXPECT_NE(status, ge::SUCCESS);
}

class GraphOptimizerForTest : public GraphOptimizer {
 public:
  ~GraphOptimizerForTest() override {
    Finalize();
  }

  Status Initialize(const std::map<std::string, std::string> &options,
                    OptimizeUtility *const optimize_utility) override {
    return SUCCESS;
  }

  Status Finalize() override {
    return SUCCESS;
  }

  Status OptimizeOriginalGraph(ComputeGraph &graph) override {
    return SUCCESS;
  }

  Status OptimizeFusedGraph(ComputeGraph &graph) override {
    return SUCCESS;
  }

  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    return SUCCESS;
  }

  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    return SUCCESS;
  }
};

class GraphOptimizerForTestPreFail : public GraphOptimizerForTest {
  Status OptimizeSubgraphPreProc(ComputeGraph &graph) override {
    (void)graph;
    return FAILED;
  }
};

class GraphOptimizerForTestPostFail : public GraphOptimizerForTest {
  Status OptimizeSubgraphPostProc(ComputeGraph &graph) override {
    (void)graph;
    return FAILED;
  }
};

TEST_F(UtestGraphManagerTest, test_SubgraphPartitionAndOptimizationPreFail) {
  InitGeLib();
  Graph graph("test_graph");
  CreateGraph(graph);
  auto computer_graph = GraphUtilsEx::GetComputeGraph(graph);
  AttrUtils::SetStr(*computer_graph, ATTR_NAME_SESSION_GRAPH_ID, "test_graph");

  Graph root_graph("root_graph");
  CreateGraph(root_graph);
  auto computer_root_graph = GraphUtilsEx::GetComputeGraph(root_graph);
  AttrUtils::SetStr(*computer_root_graph, ATTR_NAME_SESSION_GRAPH_ID, "root_graph");

  computer_root_graph->AddSubGraph(computer_graph);

  OpsKernelManager &kernel_manager = OpsKernelManager::GetInstance();
  kernel_manager.composite_engines_["test"] = std::set<string>({"test"});
  kernel_manager.atomic_first_optimizers_by_priority_.push_back(make_pair("test", MakeShared<GraphOptimizerForTestPreFail>()));

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1);
  GraphManager graph_manager;
  graph_manager.GetCompilerStages(1).partitioner.GetEnginePlacer().SetComputeGraph(computer_root_graph);
  Status status = graph_manager.SubgraphPartitionAndOptimization(graph_node, computer_root_graph, 1, EnginePartitioner::Mode::kCompositeEnginePartitioning);
  EXPECT_NE(status, SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGraphManagerTest, test_SubgraphPartitionAndOptimizationPostFail) {
  InitGeLib();
  Graph graph("test_graph");
  CreateGraph(graph);
  auto computer_graph = GraphUtilsEx::GetComputeGraph(graph);
  AttrUtils::SetStr(*computer_graph, ATTR_NAME_SESSION_GRAPH_ID, "test_graph");

  Graph root_graph("root_graph");
  CreateGraph(root_graph);
  auto computer_root_graph = GraphUtilsEx::GetComputeGraph(root_graph);
  AttrUtils::SetStr(*computer_root_graph, ATTR_NAME_SESSION_GRAPH_ID, "root_graph");

  computer_root_graph->AddSubGraph(computer_graph);

  OpsKernelManager &kernel_manager = OpsKernelManager::GetInstance();
  kernel_manager.composite_engines_["test"] = std::set<string>({"test"});
  kernel_manager.atomic_first_optimizers_by_priority_.push_back(make_pair("test", MakeShared<GraphOptimizerForTestPostFail>()));

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1);
  GraphManager graph_manager;
  graph_manager.GetCompilerStages(1).partitioner.GetEnginePlacer().SetComputeGraph(computer_root_graph);
  Status status = graph_manager.SubgraphPartitionAndOptimization(graph_node, computer_root_graph, 1, EnginePartitioner::Mode::kCompositeEnginePartitioning);
  EXPECT_NE(status, SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGraphManagerTest, set_and_get_add_graph_flag) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.SetAddGraphCondition(graph_id, 1);
  uint32_t res = graph_manager.GetAddGraphCondition(graph_id);
  EXPECT_EQ(res, 1);
}

TEST_F(UtestGraphManagerTest, test_add_graph_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.SetAddGraphCondition(graph_id, kDoneAdded);
  Graph graph("test_graph");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  OmgContext context;

  std::future<Status> fut1 = std::async(std::launch::async,
      &GraphManager::AddGraph, &graph_manager, graph_id, graph, options, context);
  std::future<Status> fut2 = std::async(std::launch::async,
      &GraphManager::AddGraph, &graph_manager, graph_id, graph, options, context);
  fut1.wait();
  fut2.wait();
  Status status1 = fut1.get();
  Status status2 = fut2.get();
  EXPECT_EQ(status1, ge::SUCCESS);
  EXPECT_EQ(status2, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_4) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  std::map<std::string, std::string> options;
  std::string key_dynamic("ge.dynamicImageSize");
  std::string value_dynamic("416,416");
  options[key_dynamic] = value_dynamic;
  ModelExecutor executor;
  graph_manager.Initialize(options, &executor);

  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)AttrUtils::SetBool(*compute_graph, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, true);

  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_NE(status, ge::SUCCESS);

  Tensor value;
  const std::string test_str = "test";
  const std::string empty_str = "";
  status = graph_manager.GetVariable(empty_str, value);
  EXPECT_EQ(status, ge::GE_GRAPH_EMPTY_STRING_NAME);
  status = graph_manager.GetVariable(test_str, value);
  EXPECT_EQ(status, ge::GE_GRAPH_EMPTY_VARIABLE_TENSOR_TABLE);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_5) {
  Graph graph("test_graph");
  auto data = op::Data("Data").set_attr_index(1);
  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());
  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  graph.SetInputs(inputs).SetOutputs(outputs);

  std::map<std::string, std::string> options = {{"ge.exec.dataInputsShapeRange", "0:[-1]"}};
  OmgContext context;
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  EXPECT_EQ(graph_manager.AddGraph(graph_id, graph, options, context), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_add_graph_6) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_with_copy_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();

  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_manager.graph_map_.insert({1, graph_node});

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraphWithCopy(graph_id, graph, options, context);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_graph_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Status status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::GE_GRAPH_GRAPH_NOT_EXIST);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetRunFlag(true);
  status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_graph_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  StubExecutor stub_executor;
  graph_manager.executor_ = &stub_executor;

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(compute_graph);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.RemoveGraph(graph_id), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_fork_graph) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetOriginGraphId(0);
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.RemoveGraph(graph_id), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_decrease_graph_count) {
  GraphId graph_id = 1;
  GraphId graph_id1 = 2;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Graph graph("test_graph");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_NO_THROW(graph_manager.DecreaseGraphCount(graph_id));
  EXPECT_NO_THROW(graph_manager.DecreaseGraphCount(graph_id1));
}

TEST_F(UtestGraphManagerTest, test_save_checkpoint1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  // Graph graph("test_graph");
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs1;
  Tensor te;
  outputs.push_back(te);
  std::map<std::string, Tensor> var_results = {{"data1", te},};
  std::map<std::string, Tensor> var_results1;
  EXPECT_NE(graph_manager.SaveCheckPointResult(graph, outputs1, var_results1), SUCCESS);
  EXPECT_EQ(graph_manager.SaveCheckPointResult(graph, outputs, var_results), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_save_checkpoint2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput1();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  // Graph graph("test_graph");
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<Tensor> outputs;
  Tensor te;
  outputs.push_back(te);
  std::map<std::string, Tensor> var_results = {{"data1", te},};
  EXPECT_EQ(graph_manager.SaveCheckPointResult(graph, outputs, var_results), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_save_checkpoint3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput4();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  // Graph graph("test_graph");
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<Tensor> outputs;
  Tensor te;
  outputs.push_back(te);
  std::map<std::string, Tensor> var_results = {{"data1", te},};
  EXPECT_EQ(graph_manager.SaveCheckPointResult(graph, outputs, var_results), FAILED);
}

TEST_F(UtestGraphManagerTest, test_save_variable1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<std::string> names;
  std::string str = "test";
  names.push_back(str);
  std::vector<Tensor> outputs;
  Tensor te;
  outputs.push_back(te);
  std::vector<Tensor> var_results;
  EXPECT_EQ(graph_manager.SaveVariables(graph, names, outputs, var_results), FAILED);
}

TEST_F(UtestGraphManagerTest, test_save_variable2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<std::string> names;
  std::vector<std::string> names1;
  std::string str = "data1";
  names.push_back(str);
  std::vector<Tensor> outputs;
  Tensor te;
  outputs.push_back(te);
  std::vector<Tensor> var_results;
  std::vector<Tensor> var_results1;
  EXPECT_EQ(graph_manager.SaveVariables(graph, names1, outputs, var_results1), SUCCESS);
  EXPECT_EQ(graph_manager.SaveVariables(graph, names, outputs, var_results), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_is_checkpoint_graph1) {
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  ComputeGraphPtr compute_graph = nullptr;
  auto compute_graph1 = CreateGraphWithVariableOutput();
  auto compute_graph2 = CreateGraphWithVariableOutput1();
  auto compute_graph3 = CreateGraphWithVariableOutput2();
  EXPECT_FALSE(graph_manager.IsCheckpointGraph(compute_graph));
  EXPECT_TRUE(graph_manager.IsCheckpointGraph(compute_graph1));
  EXPECT_FALSE(graph_manager.IsCheckpointGraph(compute_graph2));
  EXPECT_FALSE(graph_manager.IsCheckpointGraph(compute_graph3));
}

TEST_F(UtestGraphManagerTest, test_is_checkpoint_graph2) {
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput6();
  auto compute_graph1 = CreateGraphWithVariableOutput7();
  auto compute_graph2 = CreateGraphWithVariableOutput8();
  EXPECT_FALSE(graph_manager.IsCheckpointGraph(compute_graph));
  EXPECT_FALSE(graph_manager.IsCheckpointGraph(compute_graph1));
  EXPECT_FALSE(graph_manager.IsCheckpointGraph(compute_graph2));
}

TEST_F(UtestGraphManagerTest, test_get_graph_options) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_NO_THROW(graph_manager.GetGraphOptions(graph_id));
}

TEST_F(UtestGraphManagerTest, test_push_savedate2me_GertTensor) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  gert::Tensor te;
  std::map<std::string, gert::Tensor> save_data;
  save_data["data1"] = std::move(te);
  const std::string summary = "Save";
  EXPECT_EQ(graph_manager.PushSaveData2ME(graph_id, save_data), SUCCESS);
  EXPECT_EQ(graph_manager.PushSaveData2ME(graph_id, save_data), SUCCESS);
  EXPECT_EQ(graph_manager.PushSaveData2ME(graph_id, save_data), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_push_summarydate2me_gert_tensor) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  const std::string summary = "Summary";
  {
    gert::Tensor te;
    std::map<std::string, gert::Tensor> summary_data;
    summary_data["data1"] = std::move(te);
    EXPECT_EQ(graph_manager.PushSummaryData2ME(graph_id, summary_data), FAILED);
  }
  graph_manager.RegisterCallBackFunc(summary, callbackFuncGertTensor2);
  {
    gert::Tensor te;
    std::map<std::string, gert::Tensor> summary_data;
    summary_data["data1"] = std::move(te);
    EXPECT_EQ(graph_manager.PushSummaryData2ME(graph_id, summary_data), SUCCESS);
  }
  graph_manager.RegisterCallBackFunc(summary, callbackFuncGertTensor1);
  {
    gert::Tensor te;
    std::map<std::string, gert::Tensor> summary_data;
    summary_data["data1"] = std::move(te);
    EXPECT_EQ(graph_manager.PushSummaryData2ME(graph_id, summary_data), SUCCESS);
  }
}

TEST_F(UtestGraphManagerTest, test_checkpoint_handle_GertTensor) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph1 = CreateGraphWithVariableOutput();
  Graph graph1 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph1);
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(graph_manager.CheckpointHandle(graph_id, compute_graph1, outputs), PARAM_INVALID);
  auto compute_graph2 = CreateGraphNoOutput();
  Graph graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph2);
  outputs.clear();
  EXPECT_EQ(graph_manager.CheckpointHandle(graph_id, compute_graph2, outputs), FAILED);
  auto compute_graph3 = CreateGraphWithVariableOutput1();
  Graph graph3 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph3);
  outputs.clear();
  EXPECT_EQ(graph_manager.CheckpointHandle(graph_id, compute_graph3, outputs), PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_summary_handle) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(graph_manager.SummaryHandle(graph_id, outputs), FAILED);
}

TEST_F(UtestGraphManagerTest, test_summary_handle_GertTensor) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(graph_manager.SummaryHandle(graph_id, outputs), FAILED);
}

TEST_F(UtestGraphManagerTest, test_setattr_hcombroadcast1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput3();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_NO_THROW(graph_manager.SetAttrForHcomBroadCastOp(compute_graph));
}

TEST_F(UtestGraphManagerTest, test_setattr_hcombroadcast2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;
  auto compute_graph = CreateGraphWithVariableOutput5();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_NO_THROW(graph_manager.SetAttrForHcomBroadCastOp(compute_graph));
}

TEST_F(UtestGraphManagerTest, test_graph_context) {
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  GraphContext graph_context(graph_node);
  EXPECT_EQ(graph_context.SetComputeGraph(graph_node), SUCCESS);
  GraphNodePtr graph_node2 = nullptr;
  EXPECT_EQ(graph_context.SetComputeGraph(graph_node2), GE_GRAPH_PARAM_NULLPTR);
  GraphId graph_id3 = 3;
  GraphNodePtr graph_node3 = MakeShared<ge::GraphNode>(graph_id3);
  ComputeGraphPtr compute_graph3 = MakeShared<ComputeGraph>("test_graph");
  graph_node3->SetComputeGraph(compute_graph3);
  EXPECT_EQ(graph_context.SetComputeGraph(graph_node3), SUCCESS);
  GraphId graph_id4 = 4;
  GraphNodePtr graph_node4 = MakeShared<ge::GraphNode>(graph_id4);
  EXPECT_EQ(graph_context.SetComputeGraph(graph_node4), GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL);
  GradOpList a;
  VariableRecord b("test", a, 0);
  GeTensor c;
  std::pair<VariableRecord, GeTensor> d(b, c);
  graph_context.GetVarNodeTensorTable().push_back(d);
  const std::string var_data_name = "aaa";
  GeTensor res;
  EXPECT_EQ(graph_context.GetVariableTensor(var_data_name, res), GE_GRAPH_VARIABLE_DOES_NOT_EXIST);
}

TEST_F(UtestGraphManagerTest, test_IsGraphNeedRebuild1) {
  GraphId graph_id = 1;
  GraphId graph_id2 = 2;
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithVariableOutput();
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.IsGraphNeedRebuild(graph_id2), true);
}

TEST_F(UtestGraphManagerTest, test_IsGraphNeedRebuild2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithVariableOutput();
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.IsGraphNeedRebuild(graph_id), true);
}

TEST_F(UtestGraphManagerTest, test_AdjustAssignOpData) {
  GraphManager graph_manager;
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  EXPECT_NO_THROW(graph_manager.AdjustAssignOpData(data));
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread) {

  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;

  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;
  RunAsyncCallbackV2 callback;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->graph_id = graph_id;
  arg->session_id = session_id;
  arg->error_context = error_context;
  arg->context = context;
  arg->callback = callback;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  arg->graph_node = graph_node;
  bool ret = graph_manager.prerun_args_v2_q_.Push(arg);
  EXPECT_EQ(ret, true);

  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.PreRunThreadV2();
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread_2) {

  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;

  GraphId graph_id = 1;
  GraphNodePtr graph_node_1 = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node_1);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_node_1->SetBuildFlag(true);
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;
  RunAsyncCallbackV2 callback;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  std::shared_ptr<RunArgs> arg1;
  arg1 = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg1 != nullptr);
  arg1->graph_node = graph_node_1;
  arg1->graph_id = graph_id;
  arg1->session_id = session_id;
  arg1->error_context = error_context;
  arg1->context = context;
  arg1->callback = callback;
  bool ret = graph_manager.prerun_args_v2_q_.Push(arg1);
  EXPECT_EQ(ret, true);
  graph_id = 2;
  GraphNodePtr graph_node_2 = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node_2);
  std::shared_ptr<RunArgs> arg2;
  arg2 = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg2 != nullptr);
  arg2->graph_node = graph_node_2;
  arg2->graph_id = graph_id;
  arg2->session_id = session_id;
  arg2->error_context = error_context;
  arg2->context = context;
  arg2->callback = callback;
  ret = graph_manager.prerun_args_v2_q_.Push(arg2);
  EXPECT_EQ(ret, true);
  graph_manager.PreRunThreadV2();
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread_3) {
  GraphId graph_id1 = 1;
  GraphId graph_id2 = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node1 = MakeShared<ge::GraphNode>(graph_id1);
  GraphNodePtr graph_node2 = MakeShared<ge::GraphNode>(graph_id2);
  graph_manager.AddGraphNode(graph_id1, graph_node1);
  graph_manager.AddGraphNode(graph_id2, graph_node2);
  graph_manager.SetAddGraphCondition(graph_id1, kDoneAdded);
  Graph graph("test_graph");
  CreateGraph(graph);
  std::map<std::string, std::string> options = {
    {"ge.exec.isTailingOptimization", "1"},
    {"ge.streamMaxParallelNum", "a"},
    };
  OmgContext context;
  EXPECT_EQ(graph_manager.Initialize(options), GE_GRAPH_OPTIONS_INVALID);
  std::map<std::string, std::string> options1 = {
    {"ge.exec.isTailingOptimization", "1"},
    {"ge.streamMaxParallelNum", "8"},
    {"ge.streamNum", "0"},
    };
  EXPECT_EQ(graph_manager.Initialize(options1), GE_GRAPH_OPTIONS_INVALID);
  std::map<std::string, std::string> options2 = {
    {"ge.exec.isTailingOptimization", "1"},
    {"ge.streamMaxParallelNum", "8"},
    {"ge.streamNum", "1"},
    {"ge.perfLevel", "-2"},
    };
  EXPECT_EQ(graph_manager.Initialize(options2), GE_GRAPH_OPTIONS_INVALID);
  Status status = graph_manager.AddGraph(graph_id1, graph, options1, context);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(graph_manager.RemoveGraph(graph_id1), SUCCESS);
  graph_node2->SetRunFlag(true);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  graph_node2->SetRunFlag(false);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_ParseOption1) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {
    {"test0", "0"},
    {"test1", "1"},
    {"test2", "2"},
  };
  bool option;
  EXPECT_EQ(graph_manager.ParseOption(options, "test0", option), SUCCESS);
  EXPECT_EQ(graph_manager.ParseOption(options, "test1", option), SUCCESS);
  EXPECT_EQ(graph_manager.ParseOption(options, "test2", option), GE_GRAPH_OPTIONS_INVALID);
}

TEST_F(UtestGraphManagerTest, test_ParseOption2) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {
    {"test0", "a"},
    {"test1", "1"},
  };
  int32_t option;
  EXPECT_EQ(graph_manager.ParseOption(options, "test0", option), GE_GRAPH_OPTIONS_INVALID);
  EXPECT_EQ(graph_manager.ParseOption(options, "test1", option), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_initialize1) {
  auto p1 = std::make_shared<EngineManager>();
  std::map<std::string, DNNEnginePtr> engines_map;
  GetDNNEngineObjs(engines_map);
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {
    {"ge.streamMaxParallelNum", "AIcoreEngine:8"},
  };
  EXPECT_EQ(graph_manager.Initialize(options), GE_GRAPH_OPTIONS_INVALID);
}

TEST_F(UtestGraphManagerTest, test_initialize2) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {};
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  GraphId graph_id1 = 1;
  GraphNodePtr graph_node1 = MakeShared<ge::GraphNode>(graph_id1);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node1->SetGraph(graph_ptr);
  graph_node1->SetRunFlag(true);
  graph_manager.AddGraphNode(graph_id1, graph_node1);
  EXPECT_EQ(graph_manager.Finalize(), GE_GRAPH_GRAPH_IS_RUNNING);
}

TEST_F(UtestGraphManagerTest, test_initialize3) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {};
  StubExecutor executor;
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  graph_manager.executor_ = &executor;
  GraphId graph_id2 = 1;
  GraphNodePtr graph_node2 = MakeShared<ge::GraphNode>(graph_id2);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  //ComputeGraphPtr compute_graph = CreateGraphWithVariableOutput();
  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(compute_graph);
  ge_root_model->SetModelId(1);
  graph_node2->SetGeRootModel(ge_root_model);
  graph_node2->SetLoadFlag(true);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node2->SetGraph(graph_ptr);
  graph_manager.AddGraphNode(graph_id2, graph_node2);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_initialize4) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  graph_manager.executor_ = &executor;
  GraphId graph_id2 = 1;
  GraphNodePtr graph_node2 = MakeShared<ge::GraphNode>(graph_id2);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  //ComputeGraphPtr compute_graph = CreateGraphWithVariableOutput();
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  graph_node2->SetGeRootModel(ge_root_model);
  graph_node2->SetLoadFlag(true);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node2->SetGraph(graph_ptr);
  graph_manager.AddGraphNode(graph_id2, graph_node2);
  EXPECT_EQ(graph_manager.Finalize(), FAILED);
}

TEST_F(UtestGraphManagerTest, test_InitDynamicParams1) {
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithFrameworkOp();
  std::map<std::string, std::string> graph_options;
  EXPECT_EQ(graph_manager.InitDynamicParams(compute_graph, graph_options), FAILED);
}

TEST_F(UtestGraphManagerTest, test_InitDynamicParams_graphOption) {
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithFrameworkOp();
  std::map<std::string, std::string> graph_options;
  graph_options.insert({"ge.inputShape", "data1:-1,2,3;data2:-1,-1,2"});
  graph_options.insert({"ge.dynamicDims", "1,1,1;2,3,4"});
  graph_options.insert({"ge.dynamicNodeType", "1"});
  EXPECT_EQ(graph_manager.InitDynamicParams(compute_graph, graph_options), FAILED);
}

TEST_F(UtestGraphManagerTest, test_InitDynamicParams2) {
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithVariableOutput();
  std::map<std::string, std::string> graph_options;
  graph_manager.options_.dynamic_node_type = 0;
  EXPECT_EQ(graph_manager.InitDynamicParams(compute_graph, graph_options), SUCCESS);
  graph_manager.options_.dynamic_node_type = 1;
  EXPECT_EQ(graph_manager.InitDynamicParams(compute_graph, graph_options), SUCCESS);
  graph_manager.options_.input_shape = {1,2,3,4};
  graph_manager.options_.dynamic_dims = {0};
  EXPECT_EQ(graph_manager.InitDynamicParams(compute_graph, graph_options), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_CheckRepeatAdd1) {
  GraphId graph_id = 1;
  bool is_added = false;
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.CheckRepeatAdd(graph_id, is_added), INTERNAL_ERROR);
}

TEST_F(UtestGraphManagerTest, test_CheckRepeatAdd2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.SetAddGraphCondition(graph_id, kDoneAdded);
  bool is_added = false;
  EXPECT_EQ(graph_manager.CheckRepeatAdd(graph_id, is_added), INTERNAL_ERROR);
}

TEST_F(UtestGraphManagerTest, test_NotifyWaittingGraph) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.NotifyWaittingGraph(graph_id), INTERNAL_ERROR);
}

TEST_F(UtestGraphManagerTest, test_CheckGraphAdded) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  Graph graph;
  EXPECT_EQ(graph_manager.CheckGraphAdded(graph_id, graph), FAILED);
}

TEST_F(UtestGraphManagerTest, CheckPrecisionModeValue_PrecisionMode_SUCCESS) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {std::make_pair(PRECISION_MODE, "force_fp16")};
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  graph_manager.Finalize();
}

TEST_F(UtestGraphManagerTest, CheckPrecisionModeValue_PrecisionMode_FAILED) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {std::make_pair(PRECISION_MODE, "112")};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), GE_GRAPH_OPTIONS_INVALID);
  graph_manager.Finalize();
}

TEST_F(UtestGraphManagerTest, CheckPrecisionModeValue_PrecisionModeV2_SUCCESS) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {std::make_pair(PRECISION_MODE_V2, "fp16")};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  graph_manager.Finalize();
}

TEST_F(UtestGraphManagerTest, CheckPrecisionModeValue_PrecisionModeV2_FAILED) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {std::make_pair(PRECISION_MODE_V2, "111")};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), GE_GRAPH_OPTIONS_INVALID);
  graph_manager.Finalize();
}

TEST_F(UtestGraphManagerTest, CheckPrecisionModeValue_Conflict_FAILED) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {std::make_pair(PRECISION_MODE, "force_fp16"),
                                                std::make_pair(PRECISION_MODE_V2, "fp16")};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), GE_GRAPH_OPTIONS_INVALID);
  graph_manager.Finalize();
}

TEST_F(UtestGraphManagerTest, test_StartForRunGraph) {
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  graph_manager.executor_ = &executor;
  GraphId graph_id2 = 1;
  const GraphNodePtr graph_node2 = MakeShared<ge::GraphNode>(graph_id2);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  graph_node2->SetGeRootModel(ge_root_model);
  graph_node2->SetBuildFlag(true);
  graph_node2->SetLoadFlag(false);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node2->SetGraph(graph_ptr);
  graph_manager.AddGraphNode(graph_id2, graph_node2);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  EXPECT_EQ(graph_manager.StartForRunGraph(graph_node2, inputs, ge_root_model, session_id), FAILED);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

// test autofuse env and input not empty
TEST_F(UtestGraphManagerTest, test_StartForRunGraph_failed) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  GraphManager graph_manager;
  std::map<std::string, std::string> options = {};
  StubExecutorFail executor;
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  EXPECT_EQ(graph_manager.Initialize(options), SUCCESS);
  graph_manager.executor_ = &executor;
  GraphId graph_id2 = 1;
  const GraphNodePtr graph_node2 = MakeShared<ge::GraphNode>(graph_id2);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  graph_node2->SetGeRootModel(ge_root_model);
  graph_node2->SetBuildFlag(true);
  graph_node2->SetLoadFlag(false);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node2->SetGraph(graph_ptr);
  graph_manager.AddGraphNode(graph_id2, graph_node2);
  std::vector<GeTensor> inputs;
  for (const auto &node : compute_graph->GetInputNodes()) {
    const auto &tensor_desc = node->GetOpDesc()->GetOutputDesc(0U);
    inputs.emplace_back(tensor_desc);
  }
  uint64_t session_id = 1;
  EXPECT_EQ(graph_manager.StartForRunGraph(graph_node2, inputs, ge_root_model, session_id), FAILED);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(UtestGraphManagerTest, ResortAndUpdateMultiBatchContext) {
  DEF_GRAPH(dynamic_graph) {
    // dic order: data0 data1 data2 data3
    // topo order: data0 data3 data1 data2
    auto data_0 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_ND, DT_INT32, {-1, 4});
    auto data_3 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, 2});
    auto data_1 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 2)
                      .TensorDesc(FORMAT_ND, DT_INT32, {-1, 4});
    auto data_2 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 3)
                      .TensorDesc(FORMAT_ND, DT_INT32, {-1, 4});
    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
                              .InCnt(4)
                              .OutCnt(1)
                              .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("data_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
    CHAIN(NODE("data_3", data_3)->NODE("fused_op1", fake_type2_op1));
    CHAIN(NODE("data_1", data_1)->NODE("fused_op1", fake_type2_op1));
    CHAIN(NODE("data_2", data_2)->NODE("fused_op1", fake_type2_op1));
    GetLocalOmgContext().user_input_dims = {std::make_pair("a_input", vector<int64_t>{-1, 4}),
                                            std::make_pair("b_input", vector<int64_t>{-1, 4}),
                                            std::make_pair("c_input", vector<int64_t>{-1, -1, 2}),
                                            std::make_pair("d_input", vector<int64_t>{-1, 4})};
  };

  auto root_graph = ToComputeGraph(dynamic_graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  auto graph_node = std::make_shared<GraphNode>(graph_id);
  graph_node->SetComputeGraph(root_graph);
  graph_node->SetGraph(GraphUtilsEx::CreateGraphPtrFromComputeGraph(root_graph));
  GraphManager graph_manager;
  EXPECT_EQ(GetLocalOmgContext().user_input_dims.size(), 4);
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[0].first, "a_input");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[1].first, "b_input");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[2].first, "c_input");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[3].first, "d_input");
  GetLocalOmgContext().batch_shapes.clear();
  EXPECT_EQ(graph_manager.ResortAndUpdateMultiBatchContext(graph_node), SUCCESS);
  GetLocalOmgContext().dynamic_dims = "1,1,1,1,1;8,8,8,8,8";
  EXPECT_EQ(graph_manager.ResortAndUpdateMultiBatchContext(graph_node), SUCCESS);
  GetLocalOmgContext().dynamic_node_type = DATA;
  GetLocalOmgContext().batch_shapes.clear();
  EXPECT_EQ(graph_manager.ResortAndUpdateMultiBatchContext(graph_node), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().user_input_dims.size(), 4);
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[0].first, "data_0");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[1].first, "data_3");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[2].first, "data_1");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[3].first, "data_2");
  EXPECT_EQ(GetLocalOmgContext().batch_shapes.size(), 2);
  GetLocalOmgContext().dynamic_dims.clear();
  GetLocalOmgContext().dynamic_image_size.clear();
  GetLocalOmgContext().dynamic_batch_size.clear();
  GetLocalOmgContext().batch_shapes.clear();
  GetLocalOmgContext().dynamic_node_type.clear();
  std::vector<std::vector<int64_t>> shape;
  std::vector<NodePtr> data_nodes;
  for (const auto &cur_node : root_graph->GetAllNodes()) {
    if (cur_node->GetType() == DATA) {
      data_nodes.emplace_back(cur_node);
    }
  }
  graph_manager.ResortDynamicBatchInput(shape, data_nodes);
}

TEST_F(UtestGraphManagerTest, test_LoadGraph) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(compute_graph);
  graph_manager.options_.run_graph_flag = false;
  EXPECT_EQ(graph_manager.InnerLoadGraph(ge_root_model, graph_node), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_InnerRunGraphWithStream) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  const std::vector<GeTensor> inputs;
  std::vector<GeTensor> outputs;
  rtStream_t stream = nullptr;
  EXPECT_EQ(graph_manager.InnerRunGraphWithStream(graph_node, graph_id, stream, inputs, outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_RunGraphWithStreamAsync1) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = nullptr;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs),
            GE_GRAPH_GRAPH_NODE_NULL);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsync1) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = nullptr;
  dlog_setlevel(0,1,0);
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs),
            GE_GRAPH_GRAPH_NODE_NULL);
  dlog_setlevel(0,3,0);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsync2) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetRunFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = nullptr;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs),
            GE_GRAPH_ALREADY_RUNNING);
}

TEST_F(UtestGraphManagerTest, test_RunGraphWithStreamAsync2) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetRunFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = nullptr;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs),
            GE_GRAPH_ALREADY_RUNNING);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsync3) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 2;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  graph_node->SetGraph(graph_ptr);
  graph_node->SetBuildFlag(true);
  graph_node->SetLoadFlag(true);
  graph_node->SetGeRootModel(ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsync4) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 1;
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = nullptr;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs),
            GE_GRAPH_GRAPH_NOT_EXIST);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsync_without_executor) {
  GraphManager graph_manager;
  graph_manager.executor_ = nullptr;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GraphId graph_id = 2;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  graph_node->SetGraph(graph_ptr);
  graph_node->SetBuildFlag(true);
  graph_node->SetLoadFlag(false);
  graph_node->SetGeRootModel(ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = (void*)0x01;
  graph_manager.options_.run_graph_flag = true;
  // load graph fail
  EXPECT_NE(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_RunGraphWithStreamAsync3) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 2;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  graph_node->SetGraph(graph_ptr);
  graph_node->SetBuildFlag(true);
  graph_node->SetLoadFlag(true);
  graph_node->SetGeRootModel(ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsync) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 2;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  graph_node->SetGraph(graph_ptr);
  graph_node->SetBuildFlag(true);
  graph_node->SetLoadFlag(true);
  graph_node->SetGeRootModel(ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);

  rtStream_t stream = (void*)0x01;
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 0U;
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  //EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
  const std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream,
            gert_inputs, gert_outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_ExecuteGraphWithStreamAsyncHybridModel) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphId graph_id = 2;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  graph_node->SetGraph(graph_ptr);
  graph_node->SetBuildFlag(true);
  graph_node->SetLoadFlag(true);
  graph_node->SetGeRootModel(ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);

  rtStream_t stream = (void*)0x01;
  ModelManager::GetInstance().hybrid_model_map_[0] = std::make_shared<hybrid::HybridDavinciModel>();
  //EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
  const std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream,
            gert_inputs, gert_outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_GenerateInfershapeGraph1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  EXPECT_EQ(graph_manager.GenerateInfershapeGraph(graph_id), GE_GRAPH_GRAPH_NOT_EXIST);
}

TEST_F(UtestGraphManagerTest, test_GenerateInfershapeGraph2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.GenerateInfershapeGraph(graph_id), GE_GRAPH_GRAPH_NODE_NULL);
}

TEST_F(UtestGraphManagerTest, test_GenerateInfershapeGraph3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.GenerateInfershapeGraph(graph_id), GE_GRAPH_NULL_INPUT);
}

TEST_F(UtestGraphManagerTest, test_BuildGraphForUnregisteredOp1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  EXPECT_EQ(graph_manager.BuildGraphForUnregisteredOp(graph_id, inputs, ge_root_model, session_id), GE_GRAPH_GRAPH_NOT_EXIST);
}

TEST_F(UtestGraphManagerTest, test_BuildGraphForUnregisteredOp2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  EXPECT_EQ(graph_manager.BuildGraphForUnregisteredOp(graph_id, inputs, ge_root_model, session_id), GE_GRAPH_GRAPH_NODE_NULL);
}

TEST_F(UtestGraphManagerTest, test_BuildGraphForUnregisteredOp3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  EXPECT_EQ(graph_manager.BuildGraphForUnregisteredOp(graph_id, inputs, ge_root_model, session_id), GE_GRAPH_INIT_FAILED);
}

TEST_F(UtestGraphManagerTest, test_BuildGraph1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  bool async = false;
  EXPECT_EQ(graph_manager.BuildGraph(graph_id, inputs, ge_root_model, session_id, async), GE_GRAPH_GRAPH_NOT_EXIST);
}

TEST_F(UtestGraphManagerTest, test_BuildGraph2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  bool async = false;
  EXPECT_EQ(graph_manager.BuildGraph(graph_id, inputs, ge_root_model, session_id, async), GE_GRAPH_GRAPH_NODE_NULL);
}

TEST_F(UtestGraphManagerTest, test_BuildGraph3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetRunFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  bool async = false;
  EXPECT_EQ(graph_manager.BuildGraph(graph_id, inputs, ge_root_model, session_id, async), GE_GRAPH_ALREADY_RUNNING);
}

TEST_F(UtestGraphManagerTest, test_BuildGraph_with_fileconst) {
  InitGeLib();
  std::map<std::string, string> graph_options;
  graph_options["ge.externalWeight"] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto global_options = back_options;
  global_options["ge.deterministicLevel"] = "1";
  GetThreadLocalContext().SetGlobalOption(global_options);

  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  std::map<std::string, std::string> options;
  options["ge.buildStep"] = BUILD_STEP_AFTER_BUILD;
  graph_node->SetOptions(options);

  auto compute_graph = CreateGraphWithConstOutput();
  compute_graph->SetGraphID(graph_id);
  AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "1");
  GeRootModelPtr ge_root_model = nullptr;
  EXPECT_EQ(graph_manager.BuildGraph(compute_graph, ge_root_model), SUCCESS);

  graph_options["ge.externalWeight"] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);

  int32_t deterministic_level = 0;
  (void)ge::AttrUtils::GetInt(compute_graph, "ge.deterministicLevel", deterministic_level);
  EXPECT_EQ(deterministic_level, 1);
  global_options["ge.deterministicLevel"] = "0";
  GetThreadLocalContext().SetGlobalOption(global_options);
}

ge::ComputeGraphPtr CreateGraphWithHcom() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto hcom1 = builder.AddNode("hcom1", "HcomBroadcast", 1, 1);
  auto hcom2 = builder.AddNode("hcom2", "HcomBroadcast", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  ge::AttrUtils::SetStr(hcom1->GetOpDesc(), "group", "group_a");
  ge::AttrUtils::SetStr(hcom2->GetOpDesc(), "group", "group_a");
  builder.AddDataEdge(data, 0, hcom1, 0);
  builder.AddDataEdge(hcom1, 0, hcom2, 0);
  builder.AddDataEdge(hcom2, 0, netoutput, 0);
  return builder.GetGraph();
}

ge::ComputeGraphPtr CreateGraphWithHcomReverse() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto hcom1 = builder.AddNode("hcom1", "HcomBroadcast", 1, 1);
  auto hcom2 = builder.AddNode("hcom2", "HcomBroadcast", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  ge::AttrUtils::SetStr(hcom1->GetOpDesc(), "group", "group_a");
  ge::AttrUtils::SetStr(hcom2->GetOpDesc(), "group", "group_a");
  builder.AddDataEdge(data, 0, hcom2, 0);
  builder.AddDataEdge(hcom2, 0, hcom1, 0);
  builder.AddDataEdge(hcom1, 0, netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UtestGraphManagerTest, test_SaveOriginCommunicationNodes) {
  InitGeLib();
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  ComputeGraphPtr compute_graph = CreateGraphWithHcom();
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  auto first_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<string, string> options = {{ge::SOC_VERSION, "Ascend910A"}};
  GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(graph_manager.AddGraph(graph_id, *first_graph, options, domi::GetContext()), SUCCESS);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  GeRootModelPtr ge_root_model;
  bool async = false;
  (void)graph_manager.BuildGraph(graph_id, inputs, ge_root_model, session_id, async);
  GraphNodePtr graph_node = nullptr;
  graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  auto hcom_nodes = graph_node->GetCommunicationNodes();
  EXPECT_EQ(hcom_nodes.size(), 1);
  EXPECT_EQ(hcom_nodes["group_a"].size(), 2);
  FinalizeGeLib();
}

TEST_F(UtestGraphManagerTest, test_VerifyCommNodesOrderAfterEngineAssigned) {
  InitGeLib();
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  ComputeGraphPtr compute_graph = CreateGraphWithHcom();
  compute_graph->SetGraphID(graph_id);
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  EXPECT_EQ(graph_manager.SaveOriginCommunicationNodes(compute_graph), domi::SUCCESS);
  auto hcom_nodes = graph_node->GetCommunicationNodes();
  EXPECT_EQ(hcom_nodes.size(), 1);
  EXPECT_EQ(hcom_nodes["group_a"].size(), 2);
  EXPECT_EQ(graph_manager.VerifyCommNodesOrderAfterEngineAssigned(compute_graph), domi::SUCCESS);
  hcom_nodes = graph_node->GetCommunicationNodes();
  EXPECT_EQ(hcom_nodes.size(), 0);
  auto hcom2 = compute_graph->FindNode("hcom2");
  EXPECT_NE(hcom2, nullptr);
  hcom2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHccl);
  EXPECT_EQ(graph_manager.VerifyCommNodesOrderAfterEngineAssigned(compute_graph), domi::SUCCESS);
  hcom_nodes = graph_node->GetCommunicationNodes();
  EXPECT_EQ(hcom_nodes.size(), 1);
  EXPECT_EQ(hcom_nodes["group_a"].size(), 1);

  auto hcom1 = compute_graph->FindNode("hcom1");
  EXPECT_NE(hcom1, nullptr);
  hcom1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHccl);
  EXPECT_EQ(graph_manager.VerifyCommNodesOrderAfterEngineAssigned(compute_graph), domi::SUCCESS);
  hcom_nodes = graph_node->GetCommunicationNodes();
  EXPECT_EQ(hcom_nodes.size(), 1);
  EXPECT_EQ(hcom_nodes["group_a"].size(), 2);
  EXPECT_EQ(hcom_nodes["group_a"][0], "hcom1");
  EXPECT_EQ(hcom_nodes["group_a"][1], "hcom2");

  ComputeGraphPtr compute_graph_reverse = CreateGraphWithHcomReverse();
  compute_graph_reverse->SetGraphID(graph_id);
  hcom1 = compute_graph_reverse->FindNode("hcom1");
  EXPECT_NE(hcom1, nullptr);
  hcom1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHccl);
  hcom2 = compute_graph_reverse->FindNode("hcom2");
  EXPECT_NE(hcom2, nullptr);
  hcom2->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHccl);
  EXPECT_EQ(graph_manager.VerifyCommNodesOrderAfterEngineAssigned(compute_graph_reverse), domi::SUCCESS);
  hcom_nodes = graph_node->GetCommunicationNodes();
  EXPECT_EQ(hcom_nodes.size(), 1);
  EXPECT_EQ(hcom_nodes["group_a"].size(), 2);
  EXPECT_EQ(hcom_nodes["group_a"][0], "hcom2");
  EXPECT_EQ(hcom_nodes["group_a"][1], "hcom1");
  FinalizeGeLib();
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_1) {
  // no need to build
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_1_gerttensor) {
  // no need to build
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  std::vector<gert::Tensor> inputs(1);
  inputs[0].MutableStorageShape().AppendDim(1);
  arg->input_tensor = std::move(inputs);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_2) {
  // need build while buildflag is true, var format changed
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->callback = [](Status, std::vector<gert::Tensor> &) {};
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  graph_node->Lock();
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  graph_manager.graph_rebuild_state_ctrl_->graph_ids_need_rebuild_.insert(graph_id);
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node);
  EXPECT_EQ(status, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_3) {
  // need build while buildflag is false, var format unchanged
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->callback = [](Status, std::vector<gert::Tensor> &) {};
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(false);
  graph_node->Lock();
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_with_copy_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  // create graph
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraphWithCopy(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_with_copy_fail) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  // create graph
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
  status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::SUCCESS);
  status = graph_manager.AddGraphWithCopy(graph_id, graph, options, context);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_prerunthread_failed_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  std::shared_ptr<RunArgs> args;
  args = std::make_shared<RunArgs>();
  ASSERT_TRUE(args != nullptr);
  error_message::ErrorManagerContext error_ctx{1};
  Status st = 0;
  args->graph_id = graph_id;
  args->session_id = 1;
  args->error_context = error_ctx;
  args->context = GetThreadLocalContext();
  args->callback = [&st](Status st_return, std::vector<gert::Tensor> &) { st = st_return; };
  // create graph
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);

  graph_manager.options_.local_fmk_op_flag = false;
  // need build while buildflag is true, var format changed
  graph_node->SetBuildFlag(true);
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  graph_manager.graph_rebuild_state_ctrl_->graph_ids_need_rebuild_.insert(graph_id);

  graph_manager.graph_map_.insert({graph_id, graph_node});
  graph_manager.graph_count_.insert({graph_id, 1});
  graph_node->SetRunFlag(false);
  args->graph_node = graph_node;
  // function return.
  graph_manager.prerun_args_v2_q_.Push(args);
  auto t1 = std::thread(&GraphManager::PreRunThreadV2, &graph_manager);
  if (t1.joinable()) {
    t1.join();
  }
  EXPECT_EQ(st, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_prerunthread_failed_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  std::shared_ptr<RunArgs> args;
  args = std::make_shared<RunArgs>();
  ASSERT_TRUE(args != nullptr);
  error_message::ErrorManagerContext error_ctx{1};
  Status st;
  args->graph_id = graph_id;
  args->session_id = 1;
  args->error_context = error_ctx;
  args->context = GetThreadLocalContext();
  args->callback = [&st, &graph_manager](Status st_return, std::vector<gert::Tensor> &) { st = st_return;
      graph_manager.thread_run_flag_ = false;};
  // create graph
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);

  graph_manager.options_.local_fmk_op_flag = false;
  // need build while buildflag is true, var format changed
  graph_node->SetBuildFlag(true);
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  graph_manager.graph_rebuild_state_ctrl_->graph_ids_need_rebuild_.insert(graph_id);

  graph_manager.graph_map_.insert({graph_id, graph_node});
  graph_manager.graph_count_.insert({graph_id, 1});
  graph_node->SetRunFlag(false);
  args->graph_node = graph_node;
  // function continue
  int ret = setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", 1);
  EXPECT_EQ(ret, 0);
  graph_manager.prerun_args_v2_q_.Push(args);
  auto t1 = std::thread(&GraphManager::PreRunThreadV2, &graph_manager);
  if (t1.joinable()) {
    t1.join();
  }
  EXPECT_EQ(st, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_prerunthread_failed_3) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->graph_id = graph_id;
  arg->session_id = session_id;
  arg->error_context = error_context;
  arg->context = context;
  // create graph
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);
  arg->graph_node = graph_node;

  {
    GraphManager graph_manager;
    graph_manager.thread_run_flag_ = true;
    graph_manager.AddGraphNode(graph_id, graph_node);
    graph_manager.graph_count_.insert({graph_id, 1});
    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };
    arg->callback = callback;
    bool ret = graph_manager.prerun_args_v2_q_.Push(arg);
    EXPECT_EQ(ret, true);
    graph_node->SetRunFlag(true);
    auto t1 = std::thread(&GraphManager::PreRunThreadV2, &graph_manager);
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, GE_GRAPH_ALREADY_RUNNING);
    if (t1.joinable()) {
      t1.join();
    }
  }

  {
    GraphManager graph_manager;
    graph_manager.thread_run_flag_ = true;
    graph_manager.AddGraphNode(graph_id, graph_node);
    graph_manager.graph_count_.insert({graph_id, 1});
    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };
    arg->callback = callback;
    bool ret = graph_manager.prerun_args_v2_q_.Push(arg);
    EXPECT_EQ(ret, true);
    graph_node->SetRunFlag(false);
    graph_node->SetGraph(nullptr);
    auto t1 = std::thread(&GraphManager::PreRunThreadV2, &graph_manager);
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, GE_GRAPH_GRAPH_NODE_NULL);
    if (t1.joinable()) {
      t1.join();
    }
  }
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_prerunthread_success_1) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->graph_id = graph_id;
  arg->session_id = session_id;
  arg->error_context = error_context;
  arg->context = context;
  // create graph
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);
  arg->graph_node = graph_node;
  {
    GraphManager graph_manager;
    graph_manager.thread_run_flag_ = true;
    graph_manager.AddGraphNode(graph_id, graph_node);
    graph_manager.graph_count_.insert({graph_id, 1});
    ModelExecutor executor;
    ASSERT_EQ(executor.Initialize({}, session_id), SUCCESS);
    graph_manager.executor_ = &executor;
    // Callback for execute.
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    };
    arg->callback = callback;
    bool ret = graph_manager.prerun_args_v2_q_.Push(arg);
    EXPECT_EQ(ret, true);
    graph_node->SetRunFlag(false);
    graph_node->SetCompiledFlag(false);
    graph_node->SetBuildFlag(true);
    auto t1 = std::thread(&GraphManager::PreRunThreadV2, &graph_manager);
    sleep(1);  // wait
    EXPECT_NE(executor.run_args_q_.Size(), 0);
    graph_manager.prerun_args_v2_q_.Stop();
    graph_manager.thread_run_flag_ = false;
    t1.join();
    ASSERT_EQ(executor.Finalize(), SUCCESS);
  }

}
// TEST_F(UtestGraphManagerTest, ParseInputsDimsForGetNexNosinkAndData_success) {
//   GraphManager graph_manager;

//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

//   // save1
//   ge::OpDescPtr save_op = std::make_shared<ge::OpDesc>();
//   ge::OpDescUtilsEx::SetType(save_op, "Save");
//   save_op->SetName("Save1");
//   save_op->AddInputDesc(ge::GeTensorDesc());
//   save_op->AddOutputDesc(ge::GeTensorDesc());
//   AttrUtils::SetInt(save_op, ATTR_NAME_INDEX, 1);
//   ge::NodePtr save_node = graph->AddNode(save_op);

//   std::vector<NodePtr> nodes;
//   nodes.emplace_back(save_node);
//   ge::Tensor tensor;
//   std::vector<Tensor> input_tensors;
//   input_tensors.emplace_back(tensor);
//   auto ret = graph_manager.ParseInputsDimsForGetNexNosinkAndData(nodes, input_tensors);
//   EXPECT_EQ(ret, ge::SUCCESS);
// }

TEST_F(UtestGraphManagerTest, ChangeAndDeleteConst_success) {
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;

  auto graph = CreateGraphWithIsolatedConst();
  graph_manager.ChangeConstTypeWhenTraining(graph);
  auto const1 = graph->FindFirstNodeMatchType("Const");
  EXPECT_EQ(const1, nullptr);

  Status status = graph_manager.RemoveIsolatedConstInThisGraph(graph);
  EXPECT_EQ(status, ge::SUCCESS);
  auto all_nodes = graph->GetDirectNode();
  EXPECT_EQ(all_nodes.size(), 3);
}

TEST_F(UtestGraphManagerTest, ProcessNullableOutput_success) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Cast");
  op_impl_func->NullableOutput(1);

  auto graph = CreateGraphWithNullOutput();
  auto cast1_node_raw = graph->FindNode("cast1");
  cast1_node_raw->GetOpDesc()->AppendIrInput("x", IrInputType::kIrInputRequired);
  cast1_node_raw->GetOpDesc()->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  cast1_node_raw->GetOpDesc()->AppendIrOutput("z", IrOutputType::kIrOutputRequired);

  cast1_node_raw->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  cast1_node_raw->GetOpDesc()->MutableAllOutputName() = {{"y", 0}, {"z", 1}};

  auto cast2_node_raw = graph->FindNode("cast2");
  cast2_node_raw->GetOpDesc()->AppendIrInput("x", IrInputType::kIrInputRequired);
  cast2_node_raw->GetOpDesc()->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  cast2_node_raw->GetOpDesc()->AppendIrOutput("z", IrOutputType::kIrOutputRequired);
  cast2_node_raw->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  cast2_node_raw->GetOpDesc()->MutableAllOutputName() = {{"y", 0}, {"z", 1}};

  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.ProcessNullableOutput(graph), ge::SUCCESS);

  bool is_null_output = false;
  bool ret = true;
  auto cast1_node = graph->FindNode("cast1");
  EXPECT_NE(cast1_node, nullptr);
  const auto &tensor_desc_1_0 = cast1_node->GetOpDesc()->GetOutputDesc(0);
  ret = ge::AttrUtils::GetBool(tensor_desc_1_0, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);
  const auto &tensor_desc_1_1 = cast1_node->GetOpDesc()->GetOutputDesc(0);
  ret = ge::AttrUtils::GetBool(tensor_desc_1_1, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);

  auto cast2_node = graph->FindNode("cast2");
  EXPECT_NE(cast2_node, nullptr);
  const auto &tensor_desc_2_0 = cast2_node->GetOpDesc()->GetOutputDesc(0);
  ret = ge::AttrUtils::GetBool(tensor_desc_2_0, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);
  const auto &tensor_desc_2_1 = cast2_node->GetOpDesc()->GetOutputDesc(1);
  ret = ge::AttrUtils::GetBool(tensor_desc_2_1, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(is_null_output, true);
}

TEST_F(UtestGraphManagerTest, ProcessNullableOutput_with_dynamic_output_success) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Cast");
  op_impl_func->NullableOutput(1);

  auto graph = CreateGraphWithNullOutput();
  auto cast1_node_raw = graph->FindNode("cast1");
  cast1_node_raw->GetOpDesc()->AppendIrInput("x", IrInputType::kIrInputRequired);
  cast1_node_raw->GetOpDesc()->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  cast1_node_raw->GetOpDesc()->AppendIrOutput("z", IrOutputType::kIrOutputDynamic);
  cast1_node_raw->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  cast1_node_raw->GetOpDesc()->MutableAllOutputName() = {{"y", 0}, {"z0", 1}};

  auto cast2_node_raw = graph->FindNode("cast2");
  cast2_node_raw->GetOpDesc()->AppendIrInput("x", IrInputType::kIrInputRequired);
  cast2_node_raw->GetOpDesc()->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  cast2_node_raw->GetOpDesc()->AppendIrOutput("z", IrOutputType::kIrOutputDynamic);
  cast2_node_raw->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  cast2_node_raw->GetOpDesc()->MutableAllOutputName() = {{"y", 0}, {"z0", 1}};

  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.ProcessNullableOutput(graph), ge::SUCCESS);

  bool is_null_output = true;
  bool ret = true;
  auto cast1_node = graph->FindNode("cast1");
  EXPECT_NE(cast1_node, nullptr);
  const auto &tensor_desc_1_0 = cast1_node->GetOpDesc()->GetOutputDesc(0);
  ret = ge::AttrUtils::GetBool(tensor_desc_1_0, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);
  const auto &tensor_desc_1_1 = cast1_node->GetOpDesc()->GetOutputDesc(0);
  ret = ge::AttrUtils::GetBool(tensor_desc_1_1, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);

  auto cast2_node = graph->FindNode("cast2");
  EXPECT_NE(cast2_node, nullptr);
  const auto &tensor_desc_2_0 = cast2_node->GetOpDesc()->GetOutputDesc(0);
  ret = ge::AttrUtils::GetBool(tensor_desc_2_0, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);
  const auto &tensor_desc_2_1 = cast2_node->GetOpDesc()->GetOutputDesc(1);
  ret = ge::AttrUtils::GetBool(tensor_desc_2_1, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphManagerTest, test_set_run_context) {
  GraphNodePtr graph_node = MakeShared<GraphNode>(0);
  GraphManager graph_manager;

  GetLocalOmgContext().dynamic_dims = "1;4;8;16";
  GetLocalOmgContext().batch_shapes = {{1},{4},{8},{16}};
  EXPECT_EQ(graph_manager.SetRunContext(graph_node), SUCCESS);
  EXPECT_EQ(graph_node->context_.dynamic_shape_dims.size(), 4);
  EXPECT_EQ(graph_node->context_.dynamic_shape_dims[0], std::vector<int32_t>{1});
  EXPECT_EQ(graph_node->context_.dynamic_shape_dims[1], std::vector<int32_t>{4});
  EXPECT_EQ(graph_node->context_.dynamic_shape_dims[2], std::vector<int32_t>{8});
  EXPECT_EQ(graph_node->context_.dynamic_shape_dims[3], std::vector<int32_t>{16});
}

TEST_F(UtestGraphManagerTest, GraphContext) {
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);

  GraphContext gctx(nullptr);
  GraphContextPtr gcptr = MakeShared<ge::GraphContext>(graph_node);
  EXPECT_NE(gcptr, nullptr);
}

TEST_F(UtestGraphManagerTest, CheckEngineName) {
  InitGeLib();
  GraphManager graph_manager;
  std::string engine_name = "engine";
  std::string key = "key";
  std::map<std::string, int32_t> option;
  EXPECT_EQ(graph_manager.CheckEngineName(engine_name, key, option), SUCCESS);
  option["engine"] = 1;
  EXPECT_EQ(graph_manager.CheckEngineName(engine_name, key, option), GE_GRAPH_OPTIONS_INVALID);
  FinalizeGeLib();
}

TEST_F(UtestGraphManagerTest, ParseParallelNum) {
  GraphManager graph_manager;
  std::string parallel_num = "111111111111";
  std::string key;
  int32_t num = 0;
  EXPECT_EQ(graph_manager.ParseParallelNum(parallel_num, key, num), FAILED);
}

TEST_F(UtestGraphManagerTest, OptimizeSubGraphWithMultiThreads) {
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = BuildGraphPartitionCall();
  ComputeGraphPtr subgraph = compute_graph->GetSubgraph("case_sub");
  Graph2SubGraphInfoList sub_graph_map;
  std::vector<SubGraphInfoPtr> sgi1;
  sub_graph_map[compute_graph] = sgi1;
  std::vector<SubGraphInfoPtr> sgi2;
  auto p = std::make_shared<SubGraphInfo>();
  p->SetSubGraph(subgraph);
  sgi2.push_back(p);
  sub_graph_map[subgraph] = sgi2;
  uint64_t session_id = 0;
  AttrUtils::SetStr(compute_graph, "_op_compile_strategy", "op_compile_strategy");
  EXPECT_EQ(graph_manager.OptimizeSubGraphWithMultiThreads(compute_graph, sub_graph_map, session_id), SUCCESS);
}

TEST_F(UtestGraphManagerTest, TestGetExcludeEngines) {
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(0);
  std::map<std::string, std::string> option;
  option.insert(std::pair<std::string, std::string>(EXCLUDE_ENGINES, "AIcoreEngine"));
  graph_node->SetOptions(option);
  GraphManagerOptions options;
  options.core_type = "VectorCore";
  graph_manager.GetExcludeEngines(graph_node, options);
  const auto iter = options.exclude_engines.find("AIcoreEngine");
  EXPECT_NE(iter, options.exclude_engines.end());
}

TEST_F(UtestGraphManagerTest, test_get_graph_mem_info) {
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(graph);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512));
  EXPECT_NE(ge_model, nullptr);

  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
  GraphManager graph_manager;
  graph_manager.graph_map_.insert({1, graph_node});
  std::map<uint32_t, std::vector<uint64_t>> graphs_mem_info;
  graph_manager.GetGraphsMemInfo(graphs_mem_info);
  for (auto item : graphs_mem_info) {
    auto mem_info = item.second;
    EXPECT_EQ(mem_info[0], 1024);
    EXPECT_EQ(mem_info[1], 512);
  }
}

TEST_F(UtestGraphManagerTest, ComputeHashForConstNodes) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 2, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  (void)AttrUtils::SetTensor(const1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor);
  builder.AddDataEdge(const1, 0, netoutput, 0);
  builder.AddDataEdge(const2, 0, netoutput, 1);
  auto graph = builder.GetGraph();
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.ComputeHashForConstNodes(graph), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_CompileGraph) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);

  graph_manager.AddGraphNode(graph_id, graph_node);
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->Initialize(compute_graph);

  graph_node->SetBuildFlag(false);
  EXPECT_EQ(graph_manager.CompileGraph(graph_id, session_id, std::vector<ge::Tensor>{}), PARAM_INVALID);

  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  graph_node->SetGraph(graph);
  EXPECT_NE(graph_manager.CompileGraph(graph_id, session_id, std::vector<ge::Tensor>{}), SUCCESS);
  graph_node->SetBuildFlag(true);
  EXPECT_EQ(graph_manager.CompileGraph(graph_id, session_id, std::vector<ge::Tensor>{}), SUCCESS); // repeate compile success
}

TEST_F(UtestGraphManagerTest, CompileGraph_Error_StateInvalid) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);

  graph_manager.AddGraphNode(graph_id, graph_node);
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto op_desc = std::make_shared<OpDesc>("i_am_a_variable", VARIABLE);
  compute_graph->AddNode(op_desc);

  graph_node->SetRunFlag(true);
  EXPECT_EQ(graph_manager.CompileGraph(graph_id, session_id, {}), GE_GRAPH_ALREADY_RUNNING);
  graph_node->SetRunFlag(false);
  graph_node->SetBuildFlag(true);
  EXPECT_EQ(graph_manager.CompileGraph(graph_id, session_id, {}), SUCCESS);

  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  graph_node->SetGraph(graph);
  graph_node->SetRunFlag(false);
  graph_node->SetBuildFlag(false);
  graph_manager.options_.local_fmk_op_flag = true;
  EXPECT_NE(graph_manager.CompileGraph(graph_id, session_id, {}), SUCCESS);
  graph_node->SetBuildFlag(true);

  std::shared_ptr<GraphRebuildStateCtrl> rebuild_ctrl = std::make_shared<GraphRebuildStateCtrl>();
  rebuild_ctrl->AddGraph(graph_id, compute_graph);
  rebuild_ctrl->SetStateChanged("i_am_a_variable");
  graph_manager.SetExternalGraphRebuildStateCtrl(rebuild_ctrl);

  gert::GertRuntimeStub stub;
  EXPECT_NE(graph_manager.CompileGraph(graph_id, session_id, {}), SUCCESS);
  auto log_check = stub.GetSlogStub().FindLog(DLOG_ERROR, "need to re-build, you should remove it from GE first, then AddGraph again and re-compile it");
  EXPECT_NE(log_check, -1);
}

void CreateSummaryCompiledModel(GraphNodePtr &graph_node, GeModelPtr &ge_model, bool has_p2p = true) {
  auto compute_graph = CreateGraphWithConstOutput();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  AttrUtils::SetStr(compute_graph, "_split_logic_stream_2_origin_logic_stream", "");
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);
  if (has_p2p) {
    AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 1024);
  }

  uint64_t mem = 0UL;
  std::vector<std::vector<int64_t>> sub_mem_infos;
  std::vector<int64_t> sub_mem_offset;
  sub_mem_offset.emplace_back(0x2U);// mem_type RT_MEMORY_HBM 0x2U
  sub_mem_offset.emplace_back((int64_t)(&mem));// mem_offset_base
  sub_mem_offset.emplace_back(sizeof(mem)); // mem_size
  sub_mem_offset.emplace_back(1UL); // is_fixed_addr_prior
  sub_mem_infos.emplace_back(sub_mem_offset);
  AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  GetThreadLocalContext().SetGraphOption(graph_options);
  graph_node->SetOptions(graph_options);
}

TEST_F(UtestGraphManagerTest, ExecuteGraphWithStreamAsync_graph_not_build) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithIsolatedConst();

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  void* stream = (void *)0x10;
  graph_node->SetBuildFlag(false);
  EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), ge::PARAM_INVALID);
  // // not load
  // graph_node->SetLoadFlag(false);
  // graph_node->SetBuildFlag(true);
  // EXPECT_EQ(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), GE_GRAPH_ALREADY_RUNNING);
}

TEST_F(UtestGraphManagerTest, ExecuteGraphWithStreamAsync_external_allocator_invalid) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithIsolatedConst();
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  void* stream = (void *)0x10;
  ExternalAllocatorManager::SetExternalAllocator(stream, external_allocator);
  graph_manager.LoadGraph(graph_id, {}, stream);
  EXPECT_NE(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, LoadGraph_with_frozen_inputs) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> graph_options_new = ge::GetThreadLocalContext().GetAllGraphOptions();

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.options_.run_graph_flag = true;
  graph_options_new["ge.exec.frozenInputIndexes"] = "0,1111,8";
  GetThreadLocalContext().SetGraphOption(graph_options_new);
  graph_node->GetGeRootModel()->SetRootGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<AscendString, AscendString> ascend_options;
  for (auto &item : graph_options_new) {
    ascend_options[item.first.c_str()] = item.second.c_str();
  }
  EXPECT_EQ(graph_manager.LoadGraph(graph_id, ascend_options, nullptr), SUCCESS);
  const auto root_graph = graph_node->GetGeRootModel()->GetRootGraph();
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      const auto op_desc = node->GetOpDesc();
      int32_t index = -1;
      AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, index);
      if (index == 0) {
        bool frozen_input = false;
        EXPECT_EQ(ge::AttrUtils::GetBool(op_desc, "frozen_input", frozen_input), true);
        EXPECT_EQ(frozen_input, true);
        int64_t device_addr = 0L;
        EXPECT_EQ(ge::AttrUtils::GetInt(op_desc, "addr", device_addr), true);
        EXPECT_EQ(device_addr, 1111);
        int64_t len = 0L;
        EXPECT_EQ(ge::AttrUtils::GetInt(op_desc, "size", len), true);
        EXPECT_EQ(len, 8);
        int64_t placement;
        EXPECT_EQ(ge::AttrUtils::GetInt(op_desc, "placement", placement), true);
        EXPECT_EQ(placement, ge::Placement::kPlacementDevice);
        std::vector<int64_t> shape;
        EXPECT_EQ(ge::AttrUtils::GetListInt(op_desc, "storage_shape", shape), true);
        EXPECT_EQ(ge::AttrUtils::GetListInt(op_desc, "origin_shape", shape), true);
        DataType data_type = DT_UNDEFINED;
        EXPECT_EQ(ge::AttrUtils::GetDataType(op_desc, "dtype", data_type), true);
      }
    }
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestGraphManagerTest, LoadGraph_with_frozen_inputs_dynamic_failed) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> graph_options_new = ge::GetThreadLocalContext().GetAllGraphOptions();

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
    }
  }
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.options_.run_graph_flag = true;
  graph_options_new["ge.exec.frozenInputIndexes"] = "0,1111,8";
  GetThreadLocalContext().SetGraphOption(graph_options_new);
  graph_node->GetGeRootModel()->SetRootGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<AscendString, AscendString> ascend_options;
  for (auto &item : graph_options_new) {
    ascend_options[item.first.c_str()] = item.second.c_str();
  }
  EXPECT_NE(graph_manager.LoadGraph(graph_id, ascend_options, nullptr), SUCCESS);
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestGraphManagerTest, LoadGraph_with_frozen_inputs_index_diff) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> graph_options_new = ge::GetThreadLocalContext().GetAllGraphOptions();

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.options_.run_graph_flag = true;
  graph_options_new["ge.exec.frozenInputIndexes"] = "1,1111,8";
  GetThreadLocalContext().SetGraphOption(graph_options_new);
  graph_node->GetGeRootModel()->SetRootGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<AscendString, AscendString> ascend_options;
  for (auto &item : graph_options_new) {
    ascend_options[item.first.c_str()] = item.second.c_str();
  }
  EXPECT_EQ(graph_manager.LoadGraph(graph_id, ascend_options, nullptr), SUCCESS);
  const auto root_graph = graph_node->GetGeRootModel()->GetRootGraph();
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      const auto op_desc = node->GetOpDesc();
      int32_t index = -1;
      AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, index);
      if (index == 0) {
        bool frozen_input = false;
        EXPECT_EQ(ge::AttrUtils::GetBool(op_desc, "frozen_input", frozen_input), false);
      }
    }
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestGraphManagerTest, LoadGraph_with_frozen_inputs_addr_invalid) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> graph_options_new = ge::GetThreadLocalContext().GetAllGraphOptions();

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.options_.run_graph_flag = true;
  graph_options_new["ge.exec.frozenInputIndexes"] = "1,a,8";
  GetThreadLocalContext().SetGraphOption(graph_options_new);
  graph_node->GetGeRootModel()->SetRootGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<AscendString, AscendString> ascend_options;
  for (auto &item : graph_options_new) {
    ascend_options[item.first.c_str()] = item.second.c_str();
  }
  EXPECT_EQ(graph_manager.LoadGraph(graph_id, ascend_options, nullptr), PARAM_INVALID);
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestGraphManagerTest, LoadGraph_with_frozen_inputs_addr_out_of_range) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> graph_options_new = ge::GetThreadLocalContext().GetAllGraphOptions();

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.options_.run_graph_flag = true;
  graph_options_new["ge.exec.frozenInputIndexes"] = "1,999999999999999999999999999999999999999999999999999999999999999999999999999999999,8";
  GetThreadLocalContext().SetGraphOption(graph_options_new);
  graph_node->GetGeRootModel()->SetRootGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<AscendString, AscendString> ascend_options;
  for (auto &item : graph_options_new) {
    ascend_options[item.first.c_str()] = item.second.c_str();
  }
  EXPECT_EQ(graph_manager.LoadGraph(graph_id, ascend_options, nullptr), PARAM_INVALID);
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestGraphManagerTest, LoadGraph_with_frozen_inputs_number_invalid) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> graph_options_new = ge::GetThreadLocalContext().GetAllGraphOptions();

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  graph_manager.options_.run_graph_flag = true;
  graph_options_new["ge.exec.frozenInputIndexes"] = "0,1111;1";
  GetThreadLocalContext().SetGraphOption(graph_options_new);
  graph_node->GetGeRootModel()->SetRootGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<AscendString, AscendString> ascend_options;
  for (auto &item : graph_options_new) {
    ascend_options[item.first.c_str()] = item.second.c_str();
  }
  EXPECT_NE(graph_manager.LoadGraph(graph_id, ascend_options, nullptr), SUCCESS);
  const auto root_graph = graph_node->GetGeRootModel()->GetRootGraph();
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      const auto op_desc = node->GetOpDesc();
      int32_t index = -1;
      AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, index);
      if (index == 0) {
        bool frozen_input = false;
        EXPECT_EQ(ge::AttrUtils::GetBool(op_desc, "frozen_input", frozen_input), false);
      }
    }
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

void CreateSummaryCompiledModelWithTwoOutputs(GraphNodePtr &graph_node, GeModelPtr &ge_model) {
  auto compute_graph = CreateGraphWithConstOneAndOutputTwo();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  AttrUtils::SetStr(compute_graph, "_split_logic_stream_2_origin_logic_stream", "");
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  GetThreadLocalContext().SetGraphOption(graph_options);
}

void CreateSummaryCompiledModelWithTwoInputs(GraphNodePtr &graph_node, GeModelPtr &ge_model) {
  auto compute_graph = CreateGraphWithConstTwoAndOutputOne();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  AttrUtils::SetStr(compute_graph, "_split_logic_stream_2_origin_logic_stream", "");
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  GetThreadLocalContext().SetGraphOption(graph_options);
}

void CreateSummaryCompiledModelWithTwoInputsAndTwoOutputs(GraphNodePtr &graph_node, GeModelPtr &ge_model) {
  auto compute_graph = CreateGraphWithConstTwoAndoutputTwo();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestGraphManagerTest, RunGraphWithStreamAsync_external_allocator_invalid) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  auto compute_graph = CreateGraphWithIsolatedConst();

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs(1);
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  graph_node->SetAppRefreshFeatureMemoryFlag();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  graph_node->SetGraph(graph_ptr);
  void* stream = (void *)0x10;
  ExternalAllocatorManager::SetExternalAllocator(stream, external_allocator);
  EXPECT_NE(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_Twice_Failed) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  uint64_t mem = 0UL;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), SUCCESS);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), GE_GRAPH_REPEAT_OPERATION);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_AddressNullAndSizeZero_Success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0), SUCCESS);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBaseP2p_AddressNullAndSizeZero_NotSupport) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_P2P, nullptr, 0), GE_GRAPH_UNSUPPORTED);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_AddressNullAndSizeNotZero_Success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 8), PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_MemoryTypeInvalid_Failed) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType(2), nullptr, 0), PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_NoP2pFixedFeature_Success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model, false);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  CompiledGraphSummaryPtr summary = nullptr;
  graph_manager.GetCompiledGraphSummary(graph_id, summary);
  ASSERT_NE(summary, nullptr);
  const auto all_feautre_mem = summary->GetAllFeatureMemoryTypeSize();
  bool has_p2p_fixed_mem = false;
  for (const auto &feature_mem : all_feautre_mem) {
    if ((feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P) && feature_mem->IsFixed()) {
      has_p2p_fixed_mem = true;
    }
  }
  EXPECT_FALSE(has_p2p_fixed_mem);
  uint64_t mem = 0UL;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_P2P, &mem, sizeof(mem)), SUCCESS);
}

TEST_F(UtestGraphManagerTest, GetCompiledGraphSummary_TypeAndSizeCheck_Success) {
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  CompiledGraphSummaryPtr summary = nullptr;
  graph_manager.GetCompiledGraphSummary(graph_id, summary);
  const auto all_feautre_mem = summary->GetAllFeatureMemoryTypeSize();
  EXPECT_EQ(all_feautre_mem.size(), 2U);

  for (const auto &feature_mem : all_feautre_mem) {
    if ((feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT) && (feature_mem->IsFixed())) {
      size_t hbm_fixed_size = 0U;
      summary->GetFixedFeatureMemorySize(hbm_fixed_size);
      EXPECT_EQ(feature_mem->GetSize(), hbm_fixed_size);
    } else if (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P) {
      EXPECT_EQ(feature_mem->GetSize(), 1024);
      EXPECT_TRUE(feature_mem->IsFixed());
    }
  }
}

TEST_F(UtestGraphManagerTest, SetConstMemoryBase_invalid) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, nullptr, 0), GE_GRAPH_NOT_BUILT);

  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, nullptr, 0), PARAM_INVALID);
  uint64_t mem = 0UL;
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, &mem, 0), PARAM_INVALID);

  graph_node->SetConstMemoryBase(&mem, sizeof(mem));
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, &mem, sizeof(mem)), GE_GRAPH_REPEAT_OPERATION);

  graph_node->SetLoadFlag(true);
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, &mem, sizeof(mem)), GE_GRAPH_UNSUPPORTED);

  graph_node->SetRunFlag(true);
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, &mem, sizeof(mem)), GE_GRAPH_GRAPH_IS_RUNNING);
}

TEST_F(UtestGraphManagerTest, SetConstMemoryBase_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  std::vector<uint8_t> mem(512, 0);
  EXPECT_EQ(graph_manager.SetConstMemoryBase(graph_id, mem.data(), 512), SUCCESS);
  auto queryed = graph_node->GetConstMemoryBase();
  EXPECT_EQ(queryed.first, mem.data());
  EXPECT_EQ(queryed.second, 512);
}

TEST_F(UtestGraphManagerTest, UpdateFeatureMemoryBase_invalid) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, nullptr, 0), GE_GRAPH_NOT_BUILT);

  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, nullptr, 0), PARAM_INVALID);
  uint64_t mem = 0;
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, &mem, 0), PARAM_INVALID);

  const auto ge_root_model = graph_node->GetGeRootModel();
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, sizeof(mem), true, false, false, 0U, nullptr}});
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, nullptr, 0), GE_GRAPH_UNSUPPORTED);

  graph_node->SetRunFlag(true);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, &mem, sizeof(mem)), GE_GRAPH_GRAPH_IS_RUNNING);
}

TEST_F(UtestGraphManagerTest, UpdateFeatureMemoryBase_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS); // allow multi update
  auto queryed = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(queryed.first, mem.data());
  EXPECT_EQ(queryed.second, 1024);

  graph_node->SetLoadFlag(true);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ((uintptr_t)mem.data(), executor.mem_base_);
  EXPECT_EQ(1024, executor.mem_base_size_);
}

TEST_F(UtestGraphManagerTest, UpdateFeatureMemoryBase_unrefreshable) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  graph_node->SetOptions(graph_options);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  auto queryed = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(queryed.first, mem.data());
  EXPECT_EQ(queryed.second, 1024);
  EXPECT_NE(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS); // not allow multi update

  graph_node->SetLoadFlag(true);
  EXPECT_NE(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_invalid) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  const auto &ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);

  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0), GE_GRAPH_NOT_BUILT);
  uint64_t mem = 0UL;
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 12), PARAM_INVALID);

  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, sizeof(mem), true, false, false, 0U, nullptr}});
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), GE_GRAPH_REPEAT_OPERATION);

  graph_node->SetFeatureMemoryBase(&mem, sizeof(mem));
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), GE_GRAPH_UNSUPPORTED);

  graph_node->SetLoadFlag(true);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), GE_GRAPH_UNSUPPORTED);

  graph_node->SetRunFlag(true);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), GE_GRAPH_GRAPH_IS_RUNNING);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBaseWithMemoryTypeDefault_Failed_WhenUpdateGraphFeatureMemoryBaseIsCalled) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  uint64_t mem = 0UL;
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  std::vector<uint8_t> mem2(1024, 0);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem2.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, 8), GE_GRAPH_UNSUPPORTED);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBaseWithMemoryTypeP2p_Success_WhenUpdateGraphFeatureMemoryBaseIsCalled) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  std::vector<uint8_t> mem2(1024, 0);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem2.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_P2P, mem2.data(), 1024), SUCCESS);
}

TEST_F(UtestGraphManagerTest, IsNeedMallocFixedFeatureMem_Reture_True) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  CompiledGraphSummaryPtr summary = nullptr;
  graph_manager.GetCompiledGraphSummary(graph_id, summary);
  ASSERT_NE(summary, nullptr);

  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  EXPECT_TRUE(ge_root_model->IsNeedMallocFixedFeatureMem());
}

TEST_F(UtestGraphManagerTest, GetAllFeatureMemoryTypeSize_HbmAndP2pFixedFeature_Success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  CompiledGraphSummaryPtr summary = nullptr;
  graph_manager.GetCompiledGraphSummary(graph_id, summary);
  ASSERT_NE(summary, nullptr);
  const auto &all_feature_memory = summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);

  bool hbm_has_checked = false;
  bool p2p_has_checked = false;
  for (const auto &feature_memory : all_feature_memory) {
    if (feature_memory->GetType() == MemoryType::MEMORY_TYPE_DEFAULT) {
      EXPECT_TRUE(feature_memory->IsFixed());
      EXPECT_EQ(feature_memory->GetSize(), sizeof(uint64_t));
      hbm_has_checked = true;
    }
    if (feature_memory->GetType() == MemoryType::MEMORY_TYPE_P2P) {
      EXPECT_TRUE(feature_memory->IsFixed());
      EXPECT_EQ(feature_memory->GetSize(), 1024U);
      p2p_has_checked = true;
    }
  }
  EXPECT_TRUE(hbm_has_checked);
  EXPECT_TRUE(p2p_has_checked);
}

TEST_F(UtestGraphManagerTest, SetFixedFeatureMemoryBase_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  uint64_t mem = 0UL;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), SUCCESS);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  auto queryed = ge_root_model->GetFixedFeatureMemory();
  const auto hbm_iter = queryed.find(RT_MEMORY_HBM);
  ASSERT_NE(hbm_iter, queryed.end());
  EXPECT_EQ(hbm_iter->second.addr, &mem);
  EXPECT_EQ(hbm_iter->second.size, sizeof(mem));
}

TEST_F(UtestGraphManagerTest, UpdateRefreshableFeatureMemoryBase_invalid) {
  GraphId graph_id = 1;
  uint64_t mem = 0;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, nullptr, 0), GE_GRAPH_NOT_BUILT);

  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, nullptr, 0), PARAM_INVALID);

  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, sizeof(mem), true, false, false, 0U, nullptr}});
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, &mem, 0), PARAM_INVALID);

  graph_node->SetRunFlag(true);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, &mem, sizeof(mem)), GE_GRAPH_GRAPH_IS_RUNNING);
}

TEST_F(UtestGraphManagerTest, UpdateRefreshableFeatureMemoryBase_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  uint64_t fixmem = 0;
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &fixmem, sizeof(fixmem), true, false, false, 0U, nullptr}});

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS); // allow multi update
  auto queryed = graph_node->GetRefreshableFeatureMemoryBase();
  EXPECT_EQ(queryed.first, mem.data());
  EXPECT_EQ(queryed.second, 1024);

  graph_node->SetLoadFlag(true);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ((uintptr_t)mem.data(), executor.mem_base_);
  EXPECT_EQ(1024, executor.mem_base_size_);
}

TEST_F(UtestGraphManagerTest, UpdateRefreshableFeatureMemoryBase_unrefreshable) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  graph_node->SetOptions(graph_options);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  uint64_t fixmem = 0;
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &fixmem, sizeof(fixmem), true, false, false, 0U, nullptr}});
  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  auto queryed = graph_node->GetRefreshableFeatureMemoryBase();
  EXPECT_EQ(queryed.first, mem.data());
  EXPECT_EQ(queryed.second, 1024);
  EXPECT_NE(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS); // not allow multi update

  graph_node->SetLoadFlag(true);
  EXPECT_NE(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
}

TEST_F(UtestGraphManagerTest, UpdateRefreshableFeatureMemoryBase_Failed_WhenSetFixedFeatureMemoryNullptr) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0U), SUCCESS);

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), GE_GRAPH_UNSUPPORTED);

  graph_node->SetLoadFlag(true);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), GE_GRAPH_UNSUPPORTED);
}

TEST_F(UtestGraphManagerTest, UpdateRefreshableFeatureMemoryBase_Failed_WhenUpdateRFeatureMemoryBase) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), GE_GRAPH_UNSUPPORTED);
}

TEST_F(UtestGraphManagerTest, UpdateRFeatureMemoryBase_Failed_WhenUpdateRefreshableFeatureMemoryBase) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, mem.data(), 1024), GE_GRAPH_UNSUPPORTED);
}

TEST_F(UtestGraphManagerTest, GetCompiledGraphSummary_invalid) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), GE_GRAPH_NOT_BUILT);
}

TEST_F(UtestGraphManagerTest, GetCompiledGraphSummary_success) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);

  {
    // mem size check
    size_t size = 0;
    EXPECT_EQ(summary->GetConstMemorySize(size), SUCCESS);
    EXPECT_EQ(size, 512);
    EXPECT_EQ(summary->GetFeatureMemorySize(size), SUCCESS);
    EXPECT_EQ(size, 1024);
  }

  {
    // check refreshable
    bool refreshable = false;
    EXPECT_EQ(summary->GetFeatureMemoryBaseRefreshable(refreshable), SUCCESS);
    EXPECT_EQ(refreshable, true);
  }

  {
    // check event/stream num
    size_t num = 0;
    EXPECT_EQ(summary->GetStreamNum(num), SUCCESS);
    EXPECT_EQ(num, 1);
    EXPECT_EQ(summary->GetEventNum(num), SUCCESS);
    EXPECT_EQ(num, 2);
  }

  {
    //check outputshapes
    std::vector<ge::Shape> shapes;
    EXPECT_EQ(summary->GetOutputShapes(shapes), SUCCESS);
    EXPECT_EQ(summary->GetOutputShapes(shapes), SUCCESS);
    EXPECT_EQ(shapes.size(), 1U);
    std::vector<ge::DataType> dtypes;
    EXPECT_EQ(summary->GetOutputDtypes(dtypes), SUCCESS);
    EXPECT_EQ(dtypes.size(), 1U);
    EXPECT_EQ(dtypes.at(0), 0);
    std::vector<int64_t> expected_dims = {1, 1, 224, 224};
    EXPECT_EQ(shapes[0].GetDims(), expected_dims);
  }

  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 1U);
    EXPECT_EQ(io_indexes[0].first, 0U);
    EXPECT_EQ(io_indexes[0].second, 0U);
  }
}

TEST_F(UtestGraphManagerTest, GetCompiledGraphSummaryWithStreamInfo_KnownGraph_Success) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithStreamInfo(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  std::shared_ptr<StreamAllocationSummary> stream_summary;
  ASSERT_EQ(summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
  auto graph_to_stream_infos = stream_summary->GetAllLogicalStreamInfos();
  auto iter = graph_to_stream_infos.find("g1");
  ASSERT_TRUE(iter != graph_to_stream_infos.end());
  ASSERT_EQ(iter->second.size(), 4U);
  const auto logical_stream_0_info = iter->second[0];
  EXPECT_EQ(logical_stream_0_info.GetLogicalStreamId(), 0);
  EXPECT_NE(logical_stream_0_info.GetAllNodes().size(), 0U);
  EXPECT_EQ(logical_stream_0_info.GetPhysicalStreamNum(), 1);
  const auto logical_stream_1_info = iter->second[1];
  EXPECT_EQ(logical_stream_1_info.GetHcclFollowedStreamNum(), 3U);
  std::vector<int64_t> expect_attached_stream_ids = {3};
  EXPECT_EQ(logical_stream_1_info.GetAttachedStreamIds(), expect_attached_stream_ids);
  std::string expect_user_stream_label = "aaa";
  EXPECT_EQ(logical_stream_1_info.GetUsrStreamLabel().GetString(), expect_user_stream_label);
  std::string expect_stream_info_str =
      "logic_stream_id: 1, user_stream_label: aaa, is_assigned_by_user_stream_pass: false, attached_stream_ids: 3 "
      ", physical_model_stream_num: 1, hccl_followed_stream_num: 3.\n";
  EXPECT_EQ(logical_stream_1_info.ToStringInfo().GetString(), expect_stream_info_str);
  const auto logical_stream_2_info = iter->second[2];
  EXPECT_EQ(logical_stream_2_info.IsAssignedByStreamPass(), true);

  const auto &stream_graphs = stream_summary->ToStreamGraph();
  auto stream_graph_iter = stream_graphs.find("g1");
  ASSERT_TRUE(stream_graph_iter != stream_graphs.end());
  EXPECT_NE(stream_graph_iter->second.GetLength(), 0);

  std::map<AscendString, std::vector<AscendString>> graph_to_string_infos;
  ASSERT_EQ(GEStreamAllocationSummaryGetStringInfos(*summary, graph_to_string_infos), SUCCESS);
  ASSERT_EQ(graph_to_string_infos.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_string_infos.begin()->second[1].GetString(), expect_stream_info_str);

  std::map<AscendString, std::vector<int64_t>> graph_to_logical_stream_ids;
  ASSERT_EQ(GEStreamAllocationSummaryGetLogicalStreamIds(*summary, graph_to_logical_stream_ids), SUCCESS);
  EXPECT_EQ(graph_to_logical_stream_ids.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_logical_stream_ids.begin()->second[1], 1U);

  std::map<AscendString, std::vector<AscendString>> graph_to_user_stream_labels;
  ASSERT_EQ(GEStreamAllocationSummaryGetUsrStreamLabels(*summary, graph_to_user_stream_labels), SUCCESS);
  EXPECT_EQ(graph_to_user_stream_labels.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_user_stream_labels.begin()->second[1].GetString(), expect_user_stream_label);

  std::map<AscendString, std::vector<bool>> graph_to_is_assigned_by_stream_pass;
  ASSERT_EQ(GEStreamAllocationSummaryIsAssignedByStreamPass(*summary, graph_to_is_assigned_by_stream_pass), SUCCESS);
  EXPECT_EQ(graph_to_is_assigned_by_stream_pass.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_is_assigned_by_stream_pass.begin()->second[1], false);

  std::map<AscendString, std::vector<std::vector<int64_t>>> graph_to_attached_stream_ids;
  ASSERT_EQ(GEStreamAllocationSummaryGetAttachedStreamIds(*summary, graph_to_attached_stream_ids), SUCCESS);
  EXPECT_EQ(graph_to_attached_stream_ids.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_attached_stream_ids.begin()->second[1], expect_attached_stream_ids);

  std::map<AscendString, std::vector<int64_t>> graph_to_physical_stream_nums;
  ASSERT_EQ(GEStreamAllocationSummaryGetPhysicalStreamNums(*summary, graph_to_physical_stream_nums), SUCCESS);
  EXPECT_EQ(graph_to_physical_stream_nums.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_physical_stream_nums.begin()->second[1], 1U);

  std::map<AscendString, std::vector<int64_t>> graph_to_hccl_followed_stream_nums;
  ASSERT_EQ(GEStreamAllocationSummaryGetHcclFollowedStreamNums(*summary, graph_to_hccl_followed_stream_nums), SUCCESS);
  EXPECT_EQ(graph_to_hccl_followed_stream_nums.begin()->second.size(), 4U);
  EXPECT_EQ(graph_to_hccl_followed_stream_nums.begin()->second[1], 3U);

  std::map<AscendString, std::vector<std::vector<GNode>>> graph_to_all_nodes;
  ASSERT_EQ(GEStreamAllocationSummaryGetAllNodes(*summary, graph_to_all_nodes), SUCCESS);
  EXPECT_EQ(graph_to_all_nodes.begin()->second.size(), 4U);
  EXPECT_NE(graph_to_all_nodes.begin()->second[1].size(), 0U);

  std::map<AscendString, AscendString> graph_to_stream_graphs;
  ASSERT_EQ(GEStreamAllocationSummaryGetStreamGraphs(*summary, graph_to_stream_graphs), SUCCESS);
  EXPECT_NE(graph_to_stream_graphs.begin()->second.GetLength(), 0U);
}

TEST_F(UtestGraphManagerTest, GetCompiledGraphSummaryWithStreamInfo_UnknownGraph_Success) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithStreamInfo(graph_node, ge_model);
  ge_model->GetGraph()->SetGraphUnknownFlag(true);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  std::shared_ptr<StreamAllocationSummary> stream_summary;
  ASSERT_EQ(summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
  auto graph_to_stream_infos = stream_summary->GetAllLogicalStreamInfos();
  auto iter = graph_to_stream_infos.find("g1");
  ASSERT_TRUE(iter != graph_to_stream_infos.end());
  ASSERT_EQ(iter->second.size(), 4U);
  const auto logical_stream_0_info = iter->second[0];
  EXPECT_EQ(logical_stream_0_info.GetLogicalStreamId(), 0);
  EXPECT_EQ(logical_stream_0_info.GetPhysicalStreamNum(), 1);
  const auto logical_stream_1_info = iter->second[1];
  std::vector<int64_t> expect_attached_stream_ids = {3};
  EXPECT_EQ(logical_stream_1_info.GetAttachedStreamIds(), expect_attached_stream_ids);
  std::string expect_user_stream_label = "aaa";
  EXPECT_EQ(logical_stream_1_info.GetUsrStreamLabel().GetString(), expect_user_stream_label);
  std::string expect_stream_info_str =
      "logic_stream_id: 1, user_stream_label: aaa, is_assigned_by_user_stream_pass: false, attached_stream_ids: 3 "
      ", physical_model_stream_num: 1, hccl_followed_stream_num: 0.\n";
  EXPECT_EQ(logical_stream_1_info.ToStringInfo().GetString(), expect_stream_info_str);
  const auto logical_stream_2_info = iter->second[2];
  EXPECT_EQ(logical_stream_2_info.IsAssignedByStreamPass(), true);

  const auto &stream_graphs = stream_summary->ToStreamGraph();
  auto stream_graph_iter = stream_graphs.find("g1");
  ASSERT_TRUE(stream_graph_iter != stream_graphs.end());
  EXPECT_NE(stream_graph_iter->second.GetLength(), 0);
}

TEST_F(UtestGraphManagerTest, GetCompiledGraphSummary_failed) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  // set GraphUnknownFlag is true, set summary->data_->is_static_ = false
  graph_node->GetComputeGraph()->SetGraphUnknownFlag(true);
  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);

  {
    // mem size check
    size_t size = 0;
    EXPECT_EQ(summary->GetConstMemorySize(size), FAILED);
    EXPECT_TRUE(size == 0);

    EXPECT_EQ(summary->GetFeatureMemorySize(size), FAILED);
    EXPECT_TRUE(size == 0);

    // refreshable feature memorymem size check
    EXPECT_EQ(summary->GetRefreshableFeatureMemorySize(size), FAILED);
    EXPECT_TRUE(size == 0);

    // refreshable feature memorymem size check
    EXPECT_EQ(summary->GetFixedFeatureMemorySize(size), SUCCESS);
  }

  {
    // check refreshable
    bool refreshable = false;
    EXPECT_EQ(summary->GetFeatureMemoryBaseRefreshable(refreshable), FAILED);
    EXPECT_TRUE(refreshable == false);
  }

  {
    // check event/stream num
    size_t num = 0;
    EXPECT_EQ(summary->GetStreamNum(num), domi::SUCCESS);
    EXPECT_TRUE(num == 1);
    EXPECT_EQ(summary->GetEventNum(num), FAILED);
    EXPECT_TRUE(num == 0);
  }

  {
    //check outputshapes
    std::vector<ge::Shape> shapes;
    EXPECT_EQ(summary->GetOutputShapes(shapes), FAILED);
    EXPECT_EQ(shapes.size(), 0U);
    std::vector<ge::DataType> dtypes;
    EXPECT_EQ(summary->GetOutputDtypes(dtypes), FAILED);
    EXPECT_EQ(dtypes.size(), 0U);
  }

  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), FAILED);
    EXPECT_EQ(io_indexes.size(), 0U);
  }
}

TEST_F(UtestGraphManagerTest, SetIOIndexesWithSameAddr_Success_IoOffsetIsInconsistent) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  auto data = graph_node->GetComputeGraph()->FindNode("data1");
  EXPECT_NE(data, nullptr);
  auto netoutput = graph_node->GetComputeGraph()->FindNode("Node_Output");
  EXPECT_NE(netoutput, nullptr);

  CompiledGraphSummaryPtr summary = nullptr;
  data->GetOpDesc()->SetOutputOffset({0});
  netoutput->GetOpDesc()->SetInputOffset({1});
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 0U);
  }
}

TEST_F(UtestGraphManagerTest, SetIOIndexesWithSameAddr_Success_OneInputAndTwoOutputs) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithTwoOutputs(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 2);
    EXPECT_EQ(io_indexes[0].first, 0U);
    EXPECT_EQ(io_indexes[0].second, 0U);
    EXPECT_EQ(io_indexes[1].first, 0U);
    EXPECT_EQ(io_indexes[1].second, 1U);
  }
}

TEST_F(UtestGraphManagerTest, SetIOIndexesWithSameAddr_Success_TwoInputsAndOneOutput) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithTwoInputs(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 0U);
  }
}


TEST_F(UtestGraphManagerTest, SetIOIndexesWithSameAddr_Success_TwoInputsAndTwoOutputs) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithTwoInputsAndTwoOutputs(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 2);
    EXPECT_EQ(io_indexes[0].first, 0U);
    EXPECT_EQ(io_indexes[0].second, 0U);
    EXPECT_EQ(io_indexes[1].first, 1U);
    EXPECT_EQ(io_indexes[1].second, 1U);
  }
}

TEST_F(UtestGraphManagerTest, SetIOIndexesWithSameAddr_Success_InputOffsetIsEmpty) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  auto data = graph_node->GetComputeGraph()->FindNode("data1");
  EXPECT_NE(data, nullptr);

  CompiledGraphSummaryPtr summary = nullptr;
  data->GetOpDesc()->SetOutputOffset({});
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 0U);
  }
}

TEST_F(UtestGraphManagerTest, SetIOIndexesWithSameAddr_Success_OutputOffsetIsEmpty) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  auto netoutput = graph_node->GetComputeGraph()->FindNode("Node_Output");
  EXPECT_NE(netoutput, nullptr);

  CompiledGraphSummaryPtr summary = nullptr;
  netoutput->GetOpDesc()->SetInputOffset({});
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    //check ioindex
    std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
    EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
    EXPECT_EQ(io_indexes.size(), 0U);
  }
}

TEST_F(UtestGraphManagerTest, SetExternalWeightPaths_Success_IoOffsetIsInconsistent) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  auto data = graph_node->GetComputeGraph()->FindNode("data1");
  EXPECT_NE(data, nullptr);
  auto netoutput = graph_node->GetComputeGraph()->FindNode("Node_Output");
  EXPECT_NE(netoutput, nullptr);

  CompiledGraphSummaryPtr summary = nullptr;
  data->GetOpDesc()->SetOutputOffset({0});
  netoutput->GetOpDesc()->SetInputOffset({1});
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    std::vector<ExternalWeightDescPtr> externalWeightPaths;
    EXPECT_EQ(summary->GetExternalWeightPaths(externalWeightPaths), SUCCESS);
    EXPECT_EQ(externalWeightPaths.size(), 0U);
  }
}

TEST_F(UtestGraphManagerTest, SetExternalWeightPaths_Success_OneInputAndTwoOutputs) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithTwoOutputs(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    std::vector<ExternalWeightDescPtr> externalWeightPath;
    EXPECT_EQ(summary->GetExternalWeightPaths(externalWeightPath), SUCCESS);
  }
}

TEST_F(UtestGraphManagerTest, SetExternalWeightPaths_Success_TwoInputsAndOneOutput) {
  GraphManager graph_manager;
  GraphNodePtr graph_node;
  GeModelPtr ge_model;
  CreateSummaryCompiledModelWithTwoInputs(graph_node, ge_model);

  GraphId graph_id = 1;
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  {
    std::vector<ExternalWeightDescPtr> externalWeightPath;
    EXPECT_EQ(summary->GetExternalWeightPaths(externalWeightPath), SUCCESS);
    EXPECT_EQ(externalWeightPath.size(), 0U);
  }
}

TEST_F(UtestGraphManagerTest, SetExternalWeightPaths_Success_SignalFileConstant) {
  auto op_desc = std::make_shared<OpDesc>("FileConstant", FILECONSTANT);
  EXPECT_NE(op_desc, nullptr);
  std::string file_path = "/test_file_path";
  int32_t offset = 1;
  int32_t length = 1;
  std::string file_id = "/test_file_id";
  ASSERT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_path));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, offset));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, length));
  ASSERT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, file_id));
  auto compute_graph = std::make_shared<ComputeGraph>("test");
  auto node = compute_graph->AddNode(op_desc);
  auto ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  std::shared_ptr<CompiledGraphSummary::SummaryData> summary_data = MakeShared<CompiledGraphSummary::SummaryData>();
  auto status = summary_data->SetExternalWeightPaths(ge_model);
  EXPECT_EQ(status, SUCCESS);
  const auto& paths = summary_data->GetExternalWeightPaths();
  EXPECT_EQ(paths.size(), 1U);
  auto external_weight = paths[0];
  EXPECT_NE(external_weight, nullptr);
  EXPECT_EQ(external_weight->GetLocation().GetString(), file_path);
  EXPECT_EQ(external_weight->GetOffset(), static_cast<size_t>(offset));
  EXPECT_EQ(external_weight->GetSize(), static_cast<size_t>(length));
  EXPECT_EQ(external_weight->GetId().GetString(), file_id);
}

TEST_F(UtestGraphManagerTest, test_checkIoReuseMemIndexesOption1) {
  std::map<std::string, std::string> options;
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0xff");

  auto graph1 = gert::ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph1 = GraphUtilsEx::GetComputeGraph(graph1);
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), PARAM_INVALID);

  options.clear();
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0, 0");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), PARAM_INVALID);

  options.clear();
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);

  auto compute_graph2 = gert::ShareGraph::BuildBlockGraph();

  options.clear();
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0xff");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), PARAM_INVALID);

  options.clear();
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0, 0");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), PARAM_INVALID);

  /* OPTION_FEATURE_BASE_REFRESHABLE default set 1 */
  options.clear();
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  GetThreadLocalContext().SetGraphOption(options);
  compute_graph2->SetNeedIteration(false);
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph2, options), SUCCESS);

  options.clear();
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  GetThreadLocalContext().SetGraphOption(options);
  compute_graph2->SetNeedIteration(false);
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph2, options), SUCCESS);
  options.clear();
  options.emplace(ge::OPTION_HOST_SCHEDULING_MAX_THRESHOLD, "-100");
  EXPECT_NE(CheckOptionValidThreshold(options, OPTION_HOST_SCHEDULING_MAX_THRESHOLD), SUCCESS);
  options.clear();
  options.emplace(ge::OPTION_HOST_SCHEDULING_MAX_THRESHOLD, "abc");
  EXPECT_NE(CheckOptionValidThreshold(options, OPTION_HOST_SCHEDULING_MAX_THRESHOLD), SUCCESS);
  options.clear();
  options.emplace(ge::OPTION_HOST_SCHEDULING_MAX_THRESHOLD, "15");
  EXPECT_EQ(CheckOptionValidThreshold(options, OPTION_HOST_SCHEDULING_MAX_THRESHOLD), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_CheckOutputReuseInputMemIndexesOption) {
  auto graph1 = gert::ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph1 = GraphUtilsEx::GetComputeGraph(graph1);

  std::map<std::string, std::string> options;

  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);

  options.clear();
  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,1|2,3|10,20");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), PARAM_INVALID);

  options.clear();
  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);

  options.clear();
  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);

  options.clear();
  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|-1,0");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);

  options.clear();
  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|invalid");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);

  options.clear();
  options.emplace(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, "99999999999999999999999999,1");
  EXPECT_EQ(CheckIoReuseMemIndexesOption(compute_graph1, options), SUCCESS);
}

/**
 *   refdata   data
 *      \      /
 *       conv2d
 */
TEST_F(UtestGraphManagerTest, test_CompileGraph_NormalizeIO_with_input_storage_format) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, {});
  auto tensor_desc = ref_data->GetOpDesc()->MutableInputDesc(0U);
  tensor_desc->SetOriginFormat(FORMAT_NCHW);
  (void)AttrUtils::SetBool(tensor_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  tensor_desc->SetOriginShape(GeShape({1, 2, 4, 5}));
  ref_data->GetOpDesc()->UpdateOutputDesc(0, *tensor_desc);

  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 2, 1, FORMAT_NC1HWC0, DT_FLOAT16, {});
  auto out_tensor_desc = conv2d->GetOpDesc()->MutableOutputDesc(0U);
  out_tensor_desc->SetOriginFormat(FORMAT_NCHW);
  (void)AttrUtils::SetBool(out_tensor_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  out_tensor_desc->SetOriginShape(GeShape({1, 2, 4, 5}));

  builder.AddDataEdge(ref_data, 0, conv2d, 0);
  builder.AddDataEdge(data, 0, conv2d, 1);
  auto compute_graph = builder.GetGraph();
  compute_graph->SetInputSize(2);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  GraphUtilsEx::RecoverGraphOperators(*graph);
  // set output of graph
  Operator conv2d_op;
  EXPECT_EQ(graph->FindOpByName("conv2d", conv2d_op), SUCCESS);

  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs = {{conv2d_op, {0}}};
  graph->SetOutputs(output_indexs);

  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  OmgContext omg_context;
  uint32_t graph_id = 0U;
  SessionId session_id = 0U;

  // init ge and graph_manager
  InitGeLib();
  graph_manager.Initialize({}, &executor);

  // add graph
  GetThreadLocalContext().SetGraphOption({{SOC_VERSION, "Ascend910"}});
  auto ret_add_graph = graph_manager.AddGraph(graph_id, *graph, {{SOC_VERSION, "Ascend910"}}, omg_context);
  EXPECT_EQ(ret_add_graph, SUCCESS);

  // compile graph
  // here can not ensure compileGraph is success, we just need check graph after PrepareRunningFormatRefiner
  EXPECT_NE(graph_manager.CompileGraph(graph_id, session_id, std::vector<ge::Tensor>{}), SUCCESS);

  // check attrs on node of graph
  bool is_refdata_heavy_op = false;
  AttrUtils::GetBool(ref_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_refdata_heavy_op);
  EXPECT_TRUE(is_refdata_heavy_op);
  auto output_desc_refdata = ref_data->GetOpDesc()->GetOutputDescPtr(0);
  int refdata_storage_format = static_cast<int>(FORMAT_RESERVED);
  AttrUtils::GetInt(output_desc_refdata, ATTR_NAME_STORAGE_FORMAT, refdata_storage_format);
  EXPECT_EQ(static_cast<Format>(refdata_storage_format), FORMAT_NC1HWC0);
  EXPECT_EQ(output_desc_refdata->GetFormat(), FORMAT_NC1HWC0);
  std::vector<int64_t> refdata_storage_shape;
  AttrUtils::GetListInt(output_desc_refdata, ATTR_NAME_STORAGE_SHAPE, refdata_storage_shape);
  //EXPECT_STREQ(ToString(refdata_storage_shape).c_str(), "[]");
  EXPECT_STREQ(output_desc_refdata->GetShape().ToString().c_str(), "1,1,4,5,16");

  auto netoutput = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(netoutput, nullptr);
  bool is_netoutput_heavy_op = false;
  AttrUtils::GetBool(netoutput->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_netoutput_heavy_op);
  EXPECT_TRUE(is_netoutput_heavy_op);
  auto output_desc_netoutput = netoutput->GetOpDesc()->GetInputDescPtr(0);
  int netoutput_storage_format = static_cast<int>(FORMAT_RESERVED);
  AttrUtils::GetInt(output_desc_netoutput, ATTR_NAME_STORAGE_FORMAT, netoutput_storage_format);
  EXPECT_EQ(static_cast<Format>(netoutput_storage_format), FORMAT_NC1HWC0);
  EXPECT_EQ(output_desc_netoutput->GetFormat(), FORMAT_NC1HWC0);
  std::vector<int64_t> netoutput_storage_shape;
  AttrUtils::GetListInt(output_desc_netoutput, ATTR_NAME_STORAGE_SHAPE, netoutput_storage_shape);
  //EXPECT_STREQ(ToString(netoutput_storage_shape).c_str(), "[]");
  EXPECT_STREQ(output_desc_netoutput->GetShape().ToString().c_str(), "1,1,4,5,16");

  graph_manager.Finalize();
  FinalizeGeLib();
}
/**
 *  only conv2d has storage format
 *
 *   refdata   data
 *      \      /
 *       conv2d
 *         |
 *        relu
 */
TEST_F(UtestGraphManagerTest, test_CompileGraph_NormalizeIO_with_unexpect_output_size) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {4, 5});
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 2, 1, FORMAT_NC1HWC0, DT_FLOAT16, {});
  auto out_tensor_desc = conv2d->GetOpDesc()->MutableOutputDesc(0U);
  out_tensor_desc->SetOriginFormat(FORMAT_NCHW);
  AttrUtils::SetBool(out_tensor_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  out_tensor_desc->SetOriginShape(GeShape({1, 2, 4, 5}));

  auto relu = builder.AddNode("relu", RELU, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3, 4, 5});

  builder.AddDataEdge(ref_data, 0, conv2d, 0);
  builder.AddDataEdge(data, 0, conv2d, 1);
  builder.AddDataEdge(conv2d, 0, relu, 0);
  auto compute_graph = builder.GetGraph();
  compute_graph->SetInputSize(2);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  GraphUtilsEx::RecoverGraphOperators(*graph);
  // set output of graph
  Operator relu_op;
  EXPECT_EQ(graph->FindOpByName("relu", relu_op), SUCCESS);

  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs = {{relu_op, {0}}};
  graph->SetOutputs(output_indexs);

  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  OmgContext omg_context;
  uint32_t graph_id = 0U;

  // init ge and graph_manager
  InitGeLib();
  graph_manager.Initialize({}, &executor);

  // add graph
  GetThreadLocalContext().SetGraphOption({{SOC_VERSION, "Ascend910"}});
  auto ret_add_graph = graph_manager.AddGraph(graph_id, *graph, {{SOC_VERSION, "Ascend910"}}, omg_context);
  EXPECT_EQ(ret_add_graph, SUCCESS);

  const std::vector<gert::Tensor> inputs((2));
  std::vector<gert::Tensor> outputs(2); // expect 1, got 2

  rtStream_t stream = nullptr;
  EXPECT_NE(graph_manager.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs),
            SUCCESS);

  graph_manager.Finalize();
  FinalizeGeLib();
}

/**
 *  only conv2d has storage format
 *
 *   refdata   data
 *      \      /
 *       conv2d
 *         |
 *        relu
 */
TEST_F(UtestGraphManagerTest, test_CompileGraph_NormalizeIO_with_normal_node_storage_format) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {4, 5});
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 2, 1, FORMAT_NC1HWC0, DT_FLOAT16, {});
  auto out_tensor_desc = conv2d->GetOpDesc()->MutableOutputDesc(0U);
  out_tensor_desc->SetOriginFormat(FORMAT_NCHW);
  AttrUtils::SetBool(out_tensor_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  out_tensor_desc->SetOriginShape(GeShape({1, 2, 4, 5}));

  auto relu = builder.AddNode("relu", RELU, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3, 4, 5});

  builder.AddDataEdge(ref_data, 0, conv2d, 0);
  builder.AddDataEdge(data, 0, conv2d, 1);
  builder.AddDataEdge(conv2d, 0, relu, 0);
  auto compute_graph = builder.GetGraph();
  compute_graph->SetInputSize(2);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  GraphUtilsEx::RecoverGraphOperators(*graph);
  // set output of graph
  Operator relu_op;
  EXPECT_EQ(graph->FindOpByName("relu", relu_op), SUCCESS);

  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs = {{relu_op, {0}}};
  graph->SetOutputs(output_indexs);

  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  OmgContext omg_context;
  uint32_t graph_id = 0U;
  SessionId session_id = 0U;

  // init ge and graph_manager
  InitGeLib();
  graph_manager.Initialize({}, &executor);

  // add graph
  GetThreadLocalContext().SetGraphOption({{SOC_VERSION, "Ascend910"}});
  auto ret_add_graph = graph_manager.AddGraph(graph_id, *graph, {{SOC_VERSION, "Ascend910"}}, omg_context);
  EXPECT_EQ(ret_add_graph, SUCCESS);

  // compile graph
  // here can not ensure compileGraph is success, we just need check graph after PrepareRunningFormatRefiner
  EXPECT_NE(graph_manager.CompileGraph(graph_id, session_id, std::vector<ge::Tensor>{}), SUCCESS);

  // check attrs on node of graph
  bool is_refdata_heavy_op = false;
  AttrUtils::GetBool(ref_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_refdata_heavy_op);
  EXPECT_FALSE(is_refdata_heavy_op);

  auto conv2d_node = compute_graph->FindFirstNodeMatchType(CONV2D);
  EXPECT_NE(conv2d_node, nullptr);
  bool is_conv2d_heavy_op = false;
  AttrUtils::GetBool(conv2d_node->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_conv2d_heavy_op);
  EXPECT_FALSE(is_conv2d_heavy_op);
  auto output_desc_conv2d = conv2d_node->GetOpDesc()->GetOutputDescPtr(0);
  int conv2d_storage_format = static_cast<int>(FORMAT_RESERVED);
  AttrUtils::GetInt(output_desc_conv2d, ATTR_NAME_STORAGE_FORMAT, conv2d_storage_format);
  EXPECT_EQ(static_cast<Format>(conv2d_storage_format), FORMAT_NC1HWC0);
  EXPECT_EQ(output_desc_conv2d->GetFormat(), FORMAT_NCHW);
  std::vector<int64_t> conv2d_storage_shape;
  AttrUtils::GetListInt(output_desc_conv2d, ATTR_NAME_STORAGE_SHAPE, conv2d_storage_shape);
  EXPECT_TRUE(conv2d_storage_shape.empty());

  graph_manager.Finalize();
  FinalizeGeLib();
}

TEST_F(UtestGraphManagerTest, test_SubGraphOnlyHasGelocal_UnfoldDynamicShapeGraph_succ) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto partitionedcall = builder.AddNode("partitionedcall", PARTITIONEDCALL, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(data, 0, partitionedcall, 0);
  auto compute_graph = builder.GetGraph();

  auto builder_sub = ut::GraphBuilder("subgraph1");
  auto const1 = builder_sub.AddNode("const1", CONSTANT, 1, 1, ge::FORMAT_NCHW);
  auto reshape = builder_sub.AddNode("reshape", RESHAPE, 1, 1, ge::FORMAT_NCHW);
  reshape->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  auto subgraph = builder_sub.GetGraph();

  auto partitioncall_node = compute_graph->FindNode("partitionedcall");
  partitioncall_node->GetOpDesc()->AddSubgraphName("subgraph1");
  partitioncall_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph->SetParentNode(partitioncall_node);
  subgraph->SetParentGraph(compute_graph);
  compute_graph->AddSubGraph(subgraph);

  GraphManager graph_manager;

  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  compute_graph->SetGraphUnknownFlag(true);
  auto ret = graph_manager.UnfoldDynamicShapeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 0);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(UtestGraphManagerTest, test_UnfoldDynamicShapeGraph_succ) {
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto partitionedcall = builder.AddNode("partitionedcall", PARTITIONEDCALL, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(data, 0, partitionedcall, 0);
  auto compute_graph = builder.GetGraph();

  auto builder_sub = ut::GraphBuilder("subgraph1");
  auto const1 = builder_sub.AddNode("const1", CONSTANT, 1, 1, ge::FORMAT_NCHW);
  auto subgraph = builder_sub.GetGraph();

  auto partitioncall_node = compute_graph->FindNode("partitionedcall");
  partitioncall_node->GetOpDesc()->AddSubgraphName("subgraph1");
  partitioncall_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph->SetParentNode(partitioncall_node);
  subgraph->SetParentGraph(compute_graph);
  compute_graph->AddSubGraph(subgraph);

  GraphManager graph_manager;
  auto ret = graph_manager.UnfoldDynamicShapeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  compute_graph->SetGraphUnknownFlag(true);
  ret = graph_manager.UnfoldDynamicShapeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  AttrUtils::SetBool(compute_graph, ATTR_SINGLE_OP_SCENE, true);
  ret = graph_manager.UnfoldDynamicShapeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphManagerTest, UnfoldDynamicShapeGraph_Ok_DisableMultiStream) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "0", 0);
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto partitionedcall = builder.AddNode("partitionedcall", PARTITIONEDCALL, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(data, 0, partitionedcall, 0);
  auto compute_graph = builder.GetGraph();
  compute_graph->SetGraphUnknownFlag(true);

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  GraphManager graph_manager;
  auto ret = graph_manager.UnfoldDynamicShapeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "Enable multi-stream in dynamic graph"), -1);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(UtestGraphManagerTest, UnfoldDynamicShapeGraph_Ok_MaxGraphParallelModelNum1) {
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto partitionedcall = builder.AddNode("partitionedcall", PARTITIONEDCALL, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(data, 0, partitionedcall, 0);
  auto compute_graph = builder.GetGraph();
  compute_graph->SetGraphUnknownFlag(true);

  std::map<std::string, std::string> options;
  options["ge.graphMaxParallelModelNum"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  GraphManager graph_manager;
  auto ret = graph_manager.UnfoldDynamicShapeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
}

namespace {
REG_OP(Cast)
.INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                      DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                      DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* input tensor */
.OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                       DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                       DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* output tensor */
.ATTR(dst_type, Int, 0)
.ATTR(truncate, Bool, false)
.OP_END_FACTORY_REG(Cast)
}

TEST_F(UtestGraphManagerTest, Autofuse_BasicGraph) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  GraphId graph_id = 1;
  GraphManager graph_manager;
  auto compute_graph = cg::BuildAddGraph({4, 5, 6}, {4, 5, 6});
  ASSERT_NE(compute_graph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);

  compute_graph->SetGraphID(graph_id);
  AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "1");

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);

  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

// 兼容用户，待原环境变量删除后，删除
TEST_F(UtestGraphManagerTest, Autofuse_BasicGraph_With_old_env) {
  InitGeLib();
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  GraphId graph_id = 1;
  GraphManager graph_manager;
  auto compute_graph = cg::BuildAddGraph({4, 5, 6}, {4, 5, 6});
  ASSERT_NE(compute_graph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);

  compute_graph->SetGraphID(graph_id);
  AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "1");

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);

  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, Autofuse_GraphWithControlGraph) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  GraphId graph_id = 1;
  GraphManager graph_manager;
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

  std::vector<GeTensor> inputs;
  GeTensorDesc td0;
  td0.SetShape((GeShape()));
  td0.SetOriginShape((GeShape()));
  GeTensorDesc td1;
  td1.SetShape((GeShape({4, 5, 6})));
  td1.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td0);
  inputs.emplace_back(td1);

  graph->SetGraphID(graph_id);
  AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "1");
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, AutoFuse_BasicGraph_Shape) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  GraphManager graph_manager;
  auto compute_graph = cg::BuildAddGraph({-1, 5, 6}, {-1, 5, 6});
  ASSERT_NE(compute_graph, nullptr);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({-1, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, AutoFuse_BasicGraph_SingleOp) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  GraphManager graph_manager;
  auto compute_graph = cg::BuildAddGraph({4, 5, 6}, {4, 5, 6});
  ASSERT_NE(compute_graph, nullptr);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);
  AttrUtils::SetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

REG_OP(Abs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Abs);

TEST_F(UtestGraphManagerTest, AutoFuse_AbsAddRelu) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto compute_graph = cg::BuildAbsAddReluReluGraph({4, 5, 6});
  ASSERT_NE(compute_graph, nullptr);

  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, AutoFuse_AbsAddRelu_Shape) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto compute_graph = cg::BuildAbsAddReluReluGraph({-1, 5, 6});
  ASSERT_NE(compute_graph, nullptr);

  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({-1, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, AutoFuse_OpPrecisionHandle) {
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto compute_graph = cg::BuildGraphWithUnConsistantType({4, 5, 6});
  ASSERT_NE(compute_graph, nullptr);

  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, AutoFuse_OpPrecisionHandleWithConstInput) {
  ge::KernelFactory::Instance().creator_map_[kCast] = []() {
    return MakeShared<TestCastKernel>();
  };
  InitGeLib();
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  auto compute_graph = cg::BuildGraphWithUnConsistantTypeWithConstInput({2, 1, 3});
  ASSERT_NE(compute_graph, nullptr);

  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({2, 1, 3})));
  td.SetOriginShape((GeShape({2, 1, 3})));
  inputs.emplace_back(td);
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(compute_graph, inputs), ge::GRAPH_SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestGraphManagerTest, BuildGraph_Ok_GraphWithAippData) {
  InitGeLib();
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  OmgContext omg_context;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetRunFlag(false);
  graph_manager.AddGraphNode(graph_id, graph_node);
  std::map<std::string, std::string> options;
  options["ge.buildStep"] = BUILD_STEP_AFTER_BUILD;
  graph_node->SetOptions(options);

  ComputeGraphPtr compute_graph = BuildAippDataGraph();
  compute_graph->SetGraphID(graph_id);
  AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "1");
  GeRootModelPtr ge_root_model = nullptr;
  EXPECT_EQ(graph_manager.BuildGraph(compute_graph, ge_root_model), ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_BuildGraphWithoutLoad_inputs_empty) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(compute_graph);
  bool async = false;
  EXPECT_EQ(graph_manager.BuildGraphWithoutLoad(graph_id, inputs, ge_root_model, session_id, async), GE_GRAPH_GRAPH_NOT_EXIST);
}

TEST_F(UtestGraphManagerTest, test_BuildGraphWithoutLoad_graphnode_null) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = nullptr;
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(compute_graph);
  bool async = false;
  EXPECT_EQ(graph_manager.BuildGraphWithoutLoad(graph_id, inputs, ge_root_model, session_id, async), GE_GRAPH_GRAPH_NODE_NULL);
}

TEST_F(UtestGraphManagerTest, test_BuildGraphWithoutLoad_is_running) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetRunFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  const std::vector<GeTensor> inputs;
  uint64_t session_id = 1;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(compute_graph);
  bool async = false;
  EXPECT_EQ(graph_manager.BuildGraphWithoutLoad(graph_id, inputs, ge_root_model, session_id, async), GE_GRAPH_ALREADY_RUNNING);
}

TEST_F(UtestGraphManagerTest, test_UpdateInputWithHintShape_hint_shape_empty) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::vector<GeShape> hint_shape;
  std::vector<GeTensor> inputs;
  EXPECT_EQ(graph_manager.UpdateInputWithHintShape(hint_shape, inputs), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_UpdateInputWithHintShape_hint_shape_not_empty_all_inputs_unknown_shape) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
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

TEST_F(UtestGraphManagerTest, test_UpdateInputWithHintShape_hint_shape_not_empty_one_input_unknown_shape) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
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

TEST_F(UtestGraphManagerTest, test_UpdateInputWithHintShape_hint_shape_error) {
  GraphManager graph_manager;
  StubExecutor executor;
  graph_manager.executor_ = &executor;
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
} // namespace ge
