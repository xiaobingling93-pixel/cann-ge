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

#include <mutex>
#include <chrono>

#include "macro_utils/dt_public_scope.h"
#include "ge/ge_api.h"
#include "graph/preprocess/graph_prepare.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/assert/check_utils.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/graph_factory.h"
#include "graph/manager/graph_var_manager.h"
#include "ge_running_env/tensor_utils.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils.h"
#include "utils/synchronizer.h"
#include "graph/ge_global_options.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "common/share_graph.h"
#include "ge_running_env/fake_op.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "common/opskernel/ops_kernel_info_types.h"

#include <stub/gert_runtime_stub.h>

using namespace std;
using namespace ge;
namespace {
const std::string REFDATA = "RefData";
class FakeFormatsWithSubgraphOptimizer : public FakeGraphOptimizer {
 public:
  FakeFormatsWithSubgraphOptimizer &OpFormatByName(std::string name, InferredOpFormat format) {
    op_names_to_format_[std::move(name)] = std::move(format);
    return *this;
  }
  FakeFormatsWithSubgraphOptimizer &OpFormatByType(std::string op_type, InferredOpFormat format) {
    op_types_to_format_[std::move(op_type)] = std::move(format);
    return *this;
  }

   Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &root_graph) override {
     std::queue<NodePtr> nodes;
     std::set<NodePtr> seen_nodes;
     std::vector<ComputeGraphPtr> root_and_all_sub_graphs;
     root_and_all_sub_graphs.emplace_back(root_graph.shared_from_this());
     std::vector<ComputeGraphPtr> subgraphs = root_graph.GetAllSubgraphs();
     root_and_all_sub_graphs.insert(root_and_all_sub_graphs.end(), subgraphs.cbegin(), subgraphs.cend());

     for (const auto &subgraph : subgraphs) {
       for (const auto &node : subgraph->GetDirectNode()) {
         if (node->GetInDataNodes().size() == 0) {
           nodes.emplace(node);
           seen_nodes.insert(node);
         }
       }

       while(!nodes.empty()) {
         auto node = std::move(nodes.front());
         nodes.pop();

         for (auto &src_anchor : node->GetAllOutDataAnchors()) {
           auto src_format = GetSrcFormat(node, src_anchor->GetIdx());
           node->GetOpDesc()->MutableOutputDesc(src_anchor->GetIdx())->SetFormat(src_format.format);
           node->GetOpDesc()->MutableOutputDesc(src_anchor->GetIdx())->SetShape(src_format.shape);

           for (auto &dst_anchor : src_anchor->GetPeerInDataAnchors()) {
             auto dst_node = dst_anchor->GetOwnerNode();
             if (seen_nodes.insert(dst_node).second) {
               nodes.push(dst_node);
             }

             auto dst_format = GetDstFormat(dst_node, dst_anchor->GetIdx());
             dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx())->SetFormat(dst_format.format);
             dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx())->SetShape(dst_format.shape);

             if (dst_format.format != src_format.format) {
               InsertTransdata(*subgraph, src_anchor, src_format, dst_anchor, dst_format);
             }
           }
         }
         for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
           if (seen_nodes.insert(out_ctrl_node).second) {
             nodes.push(out_ctrl_node);
           }
         }
       }
     }
     return SUCCESS;
   }
 private:
  FormatInfo GetSrcFormat(const NodePtr &src_node, int32_t out_index) {
    auto iter = op_names_to_format_.find(src_node->GetName());
    if (iter != op_names_to_format_.end()) {
      return iter->second.output_formats[out_index];
    }
    iter = op_types_to_format_.find(src_node->GetType());
    if (iter != op_types_to_format_.end()) {
      return iter->second.output_formats[out_index];
    }
    auto td = src_node->GetOpDesc()->GetOutputDescPtr(out_index);
    return {td->GetFormat(), td->GetShape()};
  }

  FormatInfo GetDstFormat(const NodePtr &dst_node, int32_t in_index) {
    auto iter = op_names_to_format_.find(dst_node->GetName());
    if (iter != op_names_to_format_.end()) {
      return iter->second.input_formats[in_index];
    }
    iter = op_types_to_format_.find(dst_node->GetType());
    if (iter != op_types_to_format_.end()) {
      return iter->second.input_formats[in_index];
    }
    auto td = dst_node->GetOpDesc()->GetInputDescPtr(in_index);
    return {td->GetFormat(), td->GetShape()};
  }
  void InsertTransdata(ComputeGraph &graph,
                       const OutDataAnchorPtr &src_anchor, const FormatInfo &src_format,
                       const InDataAnchorPtr &dst_anchor, const FormatInfo &dst_format) {
    std::string name = "transdata_" + std::to_string(transdata_index_++);
    auto op_desc = MakeShared<OpDesc>(name, TRANSDATA);
    op_desc->AddInputDesc("src", *src_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src_anchor->GetIdx()));
    op_desc->AddOutputDesc("dst", *dst_anchor->GetOwnerNode()->GetOpDesc()->GetInputDescPtr(dst_anchor->GetIdx()));
    op_desc->MutableInputDesc(0)->SetFormat(src_format.format);
    op_desc->MutableInputDesc(0)->SetShape(src_format.shape);
    op_desc->MutableOutputDesc(0)->SetFormat(dst_format.format);
    op_desc->MutableOutputDesc(0)->SetShape(dst_format.shape);
    AttrUtils::SetStr(op_desc, "src_format", TypeUtils::FormatToSerialString(src_format.format));
    AttrUtils::SetStr(op_desc, "dst_format", TypeUtils::FormatToSerialString(dst_format.format));

    auto node = graph.AddNode(op_desc);
    src_anchor->Unlink(dst_anchor);
    src_anchor->LinkTo(node->GetInDataAnchor(0));
    node->GetOutDataAnchor(0)->LinkTo(dst_anchor);
  }
 private:
  std::map<std::string, InferredOpFormat> op_types_to_format_;
  std::map<std::string, InferredOpFormat> op_names_to_format_;
  std::atomic<int32_t> transdata_index_{0};
};


struct FakeAicoreLibOpsKernelBuilder : FakeOpsKernelBuilder {
 public:
  FakeAicoreLibOpsKernelBuilder(const std::string &kernel_lib_name) : FakeOpsKernelBuilder(kernel_lib_name) {}
  FakeAicoreLibOpsKernelBuilder() : FakeOpsKernelBuilder() {}

 protected:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  Status GenerateTask(const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    GELOGI("Start gen task for %s", node.GetName().c_str());
    tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask());
    return SUCCESS;
  }
};

template <typename T>
void Fake5DNodeEngine(GeRunningEnvFaker &ge_env) {
  auto ffo = MakeShared<T>();
  auto ops_kernel_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AiCoreLib");
  auto aicore_engine_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AIcoreEngine");
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_NHWC, 5));
  ffo->OpFormatByType(
      CONV2D, {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,1,16,16,16}))},
              {dst_format, GeShape(std::vector<int64_t>({4,1,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,1,16,16,16}))}
          }
      });
    ffo->OpFormatByType(
      ASSIGN, {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,1,16,16,16}))},
              {dst_format, GeShape(std::vector<int64_t>({4,1,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,1,16,16,16}))}
          }
      });
  ge_env.InstallDefault();
    ge_env.Install(FakeEngine("AiCoreLib")
                       .GraphOptimizer("FormatOp", ffo)
                       .KernelBuilder(ops_kernel_builder)
                       .KernelBuilder(aicore_engine_builder));
}

/*
 *    out0                out1
 *     |                   |
 *   assign1             assign2
 *    /     \           /       \
 * refdata1  const1   refdata2  const2
 */
void RunAndCheckInitVarGraph(Session &session) {
  DUMP_GRAPH_WHEN("AfterAssignResource");
  auto var_init_graph = GraphFactory::BuildRefDataInitGraph1();
  session.AddGraph(1, var_init_graph);
  auto ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);

  // check load result
  // 1. check offset
  CHECK_GRAPH(AfterAssignResource) {
    std::vector<int64_t> refdata1_output_offsets;
    std::vector<int64_t> assign1_input_offsets;
    std::vector<int64_t> assign1_output_offsets;
    NodePtr assign1 = nullptr;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetName() == "refdata1") {
        refdata1_output_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if (node->GetName() == "assign1") {
        assign1 = node;
        assign1_input_offsets = node->GetOpDesc()->GetInputOffset();
        assign1_output_offsets = node->GetOpDesc()->GetOutputOffset();
      }
    }
    EXPECT_NE(assign1, nullptr);
    std::string ref_var_src_var_name;
    ge::AttrUtils::GetStr(assign1->GetOpDesc()->GetOutputDesc(0), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
    EXPECT_STREQ(ref_var_src_var_name.c_str(), "refdata1");
    EXPECT_EQ(refdata1_output_offsets.size(), 1U);
    EXPECT_EQ(assign1_input_offsets.size(), 2U);
    EXPECT_EQ(assign1_output_offsets.size(), 1U);
    EXPECT_EQ(refdata1_output_offsets[0], assign1_input_offsets[0]);
    EXPECT_EQ(refdata1_output_offsets[0], assign1_output_offsets[0]);
  };
}

/*
 *           out1
 *            |
 *          conv2d2
 *          /    \
 *      assign   data2        
 *      /     \
 *   conv2d1  refdata2
 *    /   \
 *  data1  refdata1
 */
void RunAndCheckTrainGraph(Session &session) {
  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");

  // 3.构图
  auto train_graph = GraphFactory::BuildRefDataTrainGraph1();

  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  rtStream_t stream = 0;// todo
  Synchronizer sync;
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // data2
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2,2,3,2}))); // refdata1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2,2,3,2}))); // refdata2
  g1_outputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // output
  ret = session.RunGraphWithStreamAsync(3, stream, g1_inputs, g1_outputs);
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(5);

  // 6.校验结果
  // 6.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    std::vector<int64_t> refdata_out_offsets;
    std::vector<int64_t> transdata_out_offsets;

    std::vector<int64_t> output_offsets;
    std::vector<int64_t> input_offsets;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetName() == "refdata2") {
        refdata_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if (node->GetName() == "add") {
        // 1.校验连边关系, assign和add之间插入转换算子
        auto transdata = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        EXPECT_EQ(transdata->GetType(), TRANSDATA);
        auto assign = transdata->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        EXPECT_EQ(assign->GetType(), ASSIGN);
        transdata_out_offsets = transdata->GetOpDesc()->GetOutputOffset();
      }

      output_offsets = node->GetOpDesc()->GetOutputOffset();
      input_offsets = node->GetOpDesc()->GetInputOffset();
      for (size_t i = 0; i < input_offsets.size(); ++i) {
        std::cout<<node->GetName().c_str()<< " input offset:" << input_offsets[i] <<std::endl;
      }
      for (size_t i = 0; i < output_offsets.size(); ++i) {
        std::cout<<node->GetName().c_str()<< " output offset:" << output_offsets[i] <<std::endl;
      }
    }

    // 2.校验offset, transdata的输出offset和refdata的offset一致
    EXPECT_EQ(refdata_out_offsets.size(), 1U);
    EXPECT_EQ(transdata_out_offsets.size(), 1U);
    EXPECT_EQ(refdata_out_offsets[0], transdata_out_offsets[0]);
  };
  // 6.2校验args
}

/*
 *           out1
 *            |
 *          conv2d2
 *          /    \
 *      assign   data2
 *      /     \
 *   conv2d1  refdata2
 *    /   \
 *  data1  refdata1
 */
void RunAndCheckTrainGraphWithUserDefinedStorageFormat(Session &session) {
  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");

  // 3.构图
  auto train_graph =
      GraphFactory::BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 2, 4, 5}, "");

  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  Synchronizer sync;
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);


  // 6.校验结果
  // 6.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    // check attrs on node of graph
    auto ref_data = graph->FindNode("refdata1");
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
    EXPECT_STREQ(output_desc_refdata->GetShape().ToString().c_str(), "1,1,4,5,16");
  };
}

void ResetCastNodeInputAndOutputDescDataType(ComputeGraphPtr sub_graph, const std::string &name) {
  auto cast = sub_graph->FindNode(name);
  cast->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  cast->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
}

/*
       refdata               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+

*/
ComputeGraphPtr BuildGraphVarPartitionedCallWithSubgraph(bool main_graph_dynamic_flag, bool partition_dynamic_flag,
                                                         const std::vector<int64_t> &shape) {
  auto variable = OP_CFG(REFDATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("variable");
  auto partition = OP_CFG(PARTITIONEDCALL)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(5)
      .Build("partitioncall");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE(variable)->NODE(partition)->NODE("NetOutput", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");

  auto p_node = main_graph->FindFirstNodeMatchType(PARTITIONEDCALL);

  auto cast1 = OP_CFG(RELU).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("cast1");
  auto cast2 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("cast2");

  auto sub_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE(cast1)->NODE("NetOutput1", "NetOutput"));
                   CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE(cast2)->EDGE(0, 1)->NODE("NetOutput1"));
                   CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE("cast3", CAST)->EDGE(0, 2)->NODE("NetOutput1"));
                   CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE("cast4", CAST)->EDGE(0, 3)->NODE("NetOutput1"));
                   CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE("cast5", CAST)->EDGE(0, 4)->NODE("NetOutput1"));
                 };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub");
  ResetCastNodeInputAndOutputDescDataType(sub_graph, "cast1");
  ResetCastNodeInputAndOutputDescDataType(sub_graph, "cast2");
  ResetCastNodeInputAndOutputDescDataType(sub_graph, "cast3");
  ResetCastNodeInputAndOutputDescDataType(sub_graph, "cast4");
  ResetCastNodeInputAndOutputDescDataType(sub_graph, "cast5");

  auto data_node = sub_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto netoutput_node = sub_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(2), ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(3), ge::ATTR_NAME_PARENT_NODE_INDEX, 3);
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(4), ge::ATTR_NAME_PARENT_NODE_INDEX, 4);

  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(p_node);

  main_graph->AddSubgraph(sub_graph);
  p_node->GetOpDesc()->AddSubgraphName("sub");
  p_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(main_graph_dynamic_flag);
  if (main_graph_dynamic_flag) {
    auto paritioned_call_node = main_graph->FindNode("partitioncall");
    AttrUtils::SetBool(paritioned_call_node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
    AttrUtils::SetBool(main_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  }
  sub_graph->SetGraphUnknownFlag(partition_dynamic_flag);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"partitioncall"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}
void MockGenerateTask() {
  auto aicore_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() == CONSTANT) {
      return SUCCESS;
    }
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char kernel_bin[] = "kernel_bin";
    vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("kernel_bin", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
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

  auto rts_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("RTSLib", rts_func);
}
}  // namespace
class RefDataSt : public testing::Test {
 protected:
  void SetUp() {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
    ge_env.InstallDefault();
    MockGenerateTask();
  }
  void TearDown() {
    ge_env.Reset();
    ge_env.InstallDefault();
  }
  GeRunningEnvFaker ge_env;
};

/*
                     var_init

┌────────┐  (0,1)   ┌─────────┐  (0,0)   ┌──────────┐
│ const1 │ ───────> │ assign1 │ <─────── │ refdata1 │
└────────┘          └─────────┘          └──────────┘
┌────────┐  (0,1)   ┌─────────┐  (0,0)   ┌──────────┐
│ const2 │ ───────> │ assign2 │ <─────── │ refdata2 │
└────────┘          └─────────┘          └──────────┘
 * 用例场景：refdata后面无内部格式，静态模型执行refdata能够正常回写
 * 步骤：
 * step 1. 下发一张refdata初始化图，初始化变量refdata1和refdata2，变量格式为常规ND
 * 期望：refdata可以正常使能零拷贝，与refdata相连的assign可以正常分配内存
 *      1. assign的输入0和输出0，与refdata的输出offset一致
 *      2. assign的输出0上有ref_var_name属性，属性名为refdata
 */
TEST_F(RefDataSt, refdata_without_inner_format_compile) {
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  RunAndCheckInitVarGraph(session);
}

/*
                                         graph

┌───────┐  (0,0)   ┌──────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────┐  (0,1)   ┌───────┐
│ data1 │ ───────> │ conv2d1  │ ───────> │  assign  │ ───────> │ add │ <─────── │ data2 │
└───────┘          └──────────┘          └──────────┘          └─────┘          └───────┘
                     ∧                     ∧
                     │ (0,1)               │ (0,0)
                     │                     │
                   ┌──────────┐          ┌──────────┐
                   │ refdata1 │          │ refdata2 │
                   └──────────┘          └──────────┘
 * 用例场景：ref后面若出现内部格式，在插入转换算子以后，refdata能够正常回写
 * 步骤：
 * step 1. 下发一张训练图，如上所示，refdata1作为权重连接到Conv2D，计算完成后给refdata2赋值。
 * 期望：  （1）conv2d1格式变为私有格式，在refdata2和assign之间插入transdata
 *         (2) assign和add中间插入transdata。该transdata的输出offset和refdata2的输出offset一致。
 *
 */
TEST_F(RefDataSt, refdata_with_inner_format_compile) {
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  RunAndCheckTrainGraph(session);
}

/*
                                body                       cond
 input(refdata)  data             +--------------------+      +---------------------+
       \         /                | data(ref)   data1  |     |      data(0)        |
           while                  |      \      /      |     |       |             |
              |                   |        assign      |     |      cast           |
           netoutput              |         |(0)       |     |       |(0)          |
                                  |      netoutput     |     |      netoutput      |
                                  +--------------------+     +---------------------+
 *
 *
 * 用例场景：refdata connect to while directly, refdata能够正常回写
 * 步骤：
 * step 1. 下发一张训练图，如上所示，refdata1作为权重连接到Conv2D，计算完成后给refdata2赋值。
 * 期望：  （1）a new Refdata named "input" created in subgraph
 *         (2) assign output offset is same as "input" out offset。
 */
TEST_F(RefDataSt, refdata_in_while_subgraph_compile) {
  auto infer_fun = [](Operator &op) -> graphStatus {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
    return GRAPH_SUCCESS;
  };
  ge_env.InstallDefault().Install(FakeOp(CAST).InfoStoreAndBuilder("AicoreLib").InferShape(infer_fun));

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");

  // 2.构图
  auto compute_graph = gert::ShareGraph::BuildGraphRefdataWhile();
  auto train_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // 3.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 4.run graph with stream async
  Synchronizer sync;
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({}))); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // refdata1
  ret = session.BuildGraph(3, g1_inputs);
  EXPECT_EQ(ret, SUCCESS);

  // 5.校验结果
  // 5.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    std::vector<int64_t> root_refdata_out_offsets;
    std::vector<int64_t> sub_refdata_out_offsets;
    std::vector<int64_t> assign_input_offsets;
    std::vector<int64_t> assign_out_offsets;

    std::vector<int64_t> output_offsets;
    std::vector<int64_t> input_offsets;
    for (const auto &node : graph->GetAllNodes()) {
      if ((node->GetName() == "input") && (node->GetOwnerComputeGraphBarePtr()->GetParentNode() == nullptr)) {
        root_refdata_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if ((node->GetName() == "input") && (node->GetOwnerComputeGraphBarePtr()->GetParentNode() != nullptr)) {
        sub_refdata_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if (node->GetName() == "assign") {
        assign_input_offsets = node->GetOpDesc()->GetInputOffset();
        assign_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
    }

    // 校验offset, assign的输出offset和refdata的offset一致
    EXPECT_EQ(root_refdata_out_offsets.size(), 1U);
    EXPECT_EQ(sub_refdata_out_offsets.size(), 1U);
    EXPECT_EQ(assign_input_offsets.size(), 2U);
    EXPECT_EQ(assign_out_offsets.size(), 1U);
    EXPECT_EQ(root_refdata_out_offsets[0], assign_out_offsets[0]);
    EXPECT_EQ(root_refdata_out_offsets[0], sub_refdata_out_offsets[0]);
  };
}
/*
*                             +-------------+  +-----------+
*                             |Then Graph   |  |Else Graph |
*       NetOutput             |             |  |           |
*           |                 | NetOutput   |  | NetOutput |
*          if <----------->   |   |         |  |   |       |
*           |                 | assign      |  |  Cast     |
*           |                 |   |  \      |  |   |       |
*        /    \               | Data const  |  | Data      |
* pred(Data)  input(RefData)  +-------------+  +-----------+

 *
 *
 * 用例场景：refdata connect to while directly, refdata能够正常回写
 * 步骤：
 * step 1. 下发一张训练图，如上所示，refdata1作为权重连接到Conv2D，计算完成后给refdata2赋值。
 * 期望：  （1）a new Refdata named "input" created in subgraph
 *         (2) assign output offset is same as "input" out offset。
 */
TEST_F(RefDataSt, refdata_in_subgraph_compile) {
  auto infer_fun = [](Operator &op) -> graphStatus {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
    return GRAPH_SUCCESS;
  };
  ge_env.InstallDefault().Install(FakeOp(CAST).InfoStoreAndBuilder("AicoreLib").InferShape(infer_fun));
  ge_env.InstallDefault().Install(FakeOp(IDENTITY).InfoStoreAndBuilder("AicoreLib").InferShape(infer_fun));

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");

  // 2.构图
  auto compute_graph = gert::ShareGraph::IfGraphWithRefData();
  auto train_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // 3.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 4.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({}))); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // refdata1
  ret = session.BuildGraph(3, g1_inputs);
  //EXPECT_EQ(ret, SUCCESS);

  // 5.校验结果
  // 5.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    std::vector<int64_t> refdata_out_offsets;
    std::vector<int64_t> assign_input_offsets;
    std::vector<int64_t> assign_out_offsets;

    std::vector<int64_t> output_offsets;
    std::vector<int64_t> input_offsets;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetName() == "input") {
        refdata_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if (node->GetName() == "assign") {
        assign_input_offsets = node->GetOpDesc()->GetInputOffset();
        assign_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
    }

    // 校验offset, assign的输出offset和refdata的offset一致
    EXPECT_EQ(refdata_out_offsets.size(), 1U);
    EXPECT_EQ(assign_input_offsets.size(), 2U);
    EXPECT_EQ(assign_out_offsets.size(), 1U);
    EXPECT_EQ(refdata_out_offsets[0], assign_out_offsets[0]);
  };
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
TEST_F(RefDataSt, refdata_has_transdata_assign_in_subgraph_compile) {
  // setenv("DUMP_GE_GRAPH", "2", 0); // todo delete
  auto infer_fun = [](Operator &op) -> graphStatus {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
    return GRAPH_SUCCESS;
  };
  Fake5DNodeEngine<FakeFormatsWithSubgraphOptimizer>(ge_env);
  ge_env.Install(FakeOp(CAST).InfoStoreAndBuilder("AicoreLib").InferShape(infer_fun))
      .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AicoreLib").InferShape(infer_fun))
      .Install(FakeOp(ASSIGN).InfoStoreAndBuilder("AicoreLib").InferShape(infer_fun));

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");
  // 2.构图
  auto compute_graph = gert::ShareGraph::IfGraphWithRefData();
  auto train_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // 3.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 4.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({}))); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // refdata1
  ret = session.BuildGraph(3, g1_inputs);
  // EXPECT_EQ(ret, SUCCESS); todo

  // 5.校验结果
  // 5.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    std::vector<int64_t> refdata_root_out_offsets;
    std::vector<int64_t> refdata_sub_out_offsets;
    std::vector<int64_t> assign_input_offsets;
    std::vector<int64_t> assign_out_offsets;
    std::vector<int64_t> assign_out_transdata_out_offsets;

    std::vector<int64_t> output_offsets;
    std::vector<int64_t> input_offsets;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetName() == "input" && (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
        refdata_root_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if (node->GetName() == "input" && (node->GetOwnerComputeGraph()->GetName() == "then")) {
        refdata_sub_out_offsets = node->GetOpDesc()->GetOutputOffset();
      }
      if (node->GetName() == "assign") {
        assign_input_offsets = node->GetOpDesc()->GetInputOffset();
        assign_out_offsets = node->GetOpDesc()->GetOutputOffset();
        EXPECT_EQ(node->GetOutDataNodes().at(0)->GetType(), TRANSDATA);
        auto transdata_node = node->GetOutDataNodes().at(0);
        assign_out_transdata_out_offsets = transdata_node->GetOpDesc()->GetOutputOffset();
      }
    }

    // 校验offset, assign的输出offset和refdata的offset一致
    EXPECT_EQ(refdata_root_out_offsets.size(), 1U);
    EXPECT_EQ(refdata_sub_out_offsets.size(), 1U);
    EXPECT_EQ(assign_input_offsets.size(), 2U);
    EXPECT_EQ(assign_out_offsets.size(), 1U);
    EXPECT_EQ(assign_out_transdata_out_offsets.size(), 1U);
    EXPECT_EQ(refdata_root_out_offsets[0], assign_out_transdata_out_offsets[0]);
    EXPECT_EQ(refdata_root_out_offsets[0], refdata_sub_out_offsets[0]);
  };
}

/**
 *    refdata  refdata
 *        \   /
 *         hcom
 *          |
 *       netoutput
 */
TEST_F(RefDataSt, refdata_connect_to_hccl_need_memcpy) {
  auto infer_fun = [](Operator &op) -> graphStatus {
    return GRAPH_SUCCESS;
  };
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  ge_env.Reset();
  // 置fake引擎和算子
  std::vector<FakeEngine> default_engines = {
      FakeEngine("DNN_HCCL").KernelInfoStore(kEngineNameHccl).GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer"),
      FakeEngine("DNN_VM_RTS").KernelInfoStore(kEngineNameRts).GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE"),
      FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore(kEngineNameGeLocal).GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER"),
  };
  for (auto& fake_engine : default_engines) {
    ge_env.Install(fake_engine);
  }

  ge_env.Install(FakeOp("HcomAllReduce").InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_fun));
  ge_env.Install(FakeOp("RefData").InfoStoreAndBuilder(kEngineNameGeLocal).InferShape(infer_fun));
  ge_env.Install(FakeOp("NetOutput").InfoStoreAndBuilder(kEngineNameGeLocal).InferShape(infer_fun));
  ge_env.Install(FakeOp("MemcpyAddrAsync").InfoStoreAndBuilder(kEngineNameRts).InferShape(infer_fun));
  ge_env.Install(FakeOp("Send").InfoStoreAndBuilder(kEngineNameRts).InferShape(infer_fun));
  ge_env.Install(FakeOp("Recv").InfoStoreAndBuilder(kEngineNameRts).InferShape(infer_fun));
  ge_env.Install(FakeOp(IDENTITY).InfoStoreAndBuilder(kEngineNameRts).InferShape(infer_fun));

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");
  // 2.构图
  auto train_graph = gert::ShareGraph::BuildHcomGraphWithRefData();

  // 3.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 4.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({}))); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16}))); // refdata1
  ret = session.BuildGraph(3, g1_inputs);
  EXPECT_EQ(ret, SUCCESS);

  // 5.校验结果
  // 5.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    EXPECT_EQ(
        gert::SummaryChecker(graph).StrictDirectNodeTypes(
            {{"RefData", 2},
             {"Identity", 2},
             {"MemcpyAddrAsync", 2},
             {"HcomAllReduce", 1},
             {"NetOutput", 1}}),
        "success");
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() == "HcomAllReduce") {
        EXPECT_EQ(gert::NodeTopoChecker(node).StrictConnectFrom({{"MemcpyAddrAsync"}, {"MemcpyAddrAsync"}}), "success");
      }
    }
  };
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

/*
                                         graph

┌───────┐  (0,0)   ┌──────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────┐  (0,1)   ┌───────┐
│ data1 │ ───────> │ conv2d1  │ ───────> │  assign  │ ───────> │ add │ <─────── │ data2 │
└───────┘          └──────────┘          └──────────┘          └─────┘          └───────┘
                     ∧                     ∧
                     │ (0,1)               │ (0,0)
                     │                     │
                   ┌──────────┐          ┌──────────┐
                   │ refdata1 │          │ refdata2 │
                   └──────────┘          └──────────┘
 * 用例场景：refdata user defined 内部格式，refdata能够正常回写
 * 步骤：
 * step 1. 下发一张训练图，如上所示，refdata1作为权重连接到Conv2D，计算完成后给refdata2赋值。
 * 期望：  （1）refdata1 has heavy op attr, and its Format is set as StorageFormat
 *
 */
TEST_F(RefDataSt, refdata_with_user_defined_inner_format_compile) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  RunAndCheckTrainGraphWithUserDefinedStorageFormat(session);
}

TEST_F(RefDataSt, refdata_with_unsupport_user_defined_inner_format_compile) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.构图
  auto train_graph =
      GraphFactory::BuildRefDataWithStroageFormatTrainGraph1(FORMAT_FRACTAL_ZN_RNN, FORMAT_NCHW, {1, 2, 4, 5}, "");

  // 2.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 3.run graph with stream async
  Synchronizer sync;
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_FRACTAL_ZN_RNN);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_NE(ret, SUCCESS);
}

/*
                                         graph

┌───────┐  (0,0)   ┌──────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────┐  (0,1)   ┌───────┐
│ data1 │ ───────> │ conv2d1  │ ───────> │  assign  │ ───────> │ add │ <─────── │ data2 │
└───────┘          └──────────┘          └──────────┘          └─────┘          └───────┘
                     ∧                     ∧
                     │ (0,1)               │ (0,0)
                     │                     │
                   ┌──────────┐          ┌──────────┐
                   │ refdata1 │          │ refdata2 │
                   └──────────┘          └──────────┘
 * 用例场景：refdata user defined 内部格式，refdata能够正常回写
 * 步骤：
 * step 1. 下发一张训练图，如上所示，refdata1作为权重连接到Conv2D，计算完成后给refdata2赋值。
 * 期望：  （1）refdata1 has heavy op attr, and its Format is set as StorageFormat
 *
 */
TEST_F(RefDataSt, refdata_with_user_defined_inner_format_compile_and_expand_dims_rule) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("AfterAssignResource");

  // 3.构图
  auto train_graph =
      GraphFactory::BuildRefDataWithStroageFormatTrainGraph1(FORMAT_FRACTAL_Z, FORMAT_NCHW, {3, 8, 8}, "1000");

  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);


  // 6.校验结果
  // 6.1 check offset
  CHECK_GRAPH(AfterAssignResource) {
    // check attrs on node of graph
    auto ref_data = graph->FindNode("refdata1");
    bool is_refdata_heavy_op = false;
    AttrUtils::GetBool(ref_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_refdata_heavy_op);
    EXPECT_TRUE(is_refdata_heavy_op);

    auto output_desc_refdata = ref_data->GetOpDesc()->GetOutputDescPtr(0);
    int refdata_storage_format = static_cast<int>(FORMAT_RESERVED);
    AttrUtils::GetInt(output_desc_refdata, ATTR_NAME_STORAGE_FORMAT, refdata_storage_format);
    EXPECT_EQ(static_cast<Format>(refdata_storage_format), FORMAT_FRACTAL_Z);
    EXPECT_EQ(output_desc_refdata->GetFormat(), FORMAT_FRACTAL_Z);
    std::vector<int64_t> refdata_storage_shape;
    AttrUtils::GetListInt(output_desc_refdata, ATTR_NAME_STORAGE_SHAPE, refdata_storage_shape);
    EXPECT_STREQ(output_desc_refdata->GetShape().ToString().c_str(), "64,1,16,16");
    std::string target_expand_dims_type;
    AttrUtils::GetStr(output_desc_refdata, ATTR_NAME_RESHAPE_INFER_TYPE, target_expand_dims_type);
    EXPECT_STREQ(target_expand_dims_type.c_str(), "CHW");
  };
}
/*
      refdata                +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
TEST_F(RefDataSt, refdata_connect_unknownshape_parittion_skip_split) {
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("PreRunAfterNormalizeGraph");
  // 2.构图
  std::vector<int64_t> shape = {8, -1, 16, 16};  // HWCN
  auto compute_graph = BuildGraphVarPartitionedCallWithSubgraph(false, true, shape);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  auto train_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // 3.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 4.compile graph
  ret = session.CompileGraph(3); // dont care about result

  // 5.校验结果
  // 5.1 check offset
  CHECK_GRAPH(PreRunAfterNormalizeGraph) {
    EXPECT_EQ(graph->GetAllNodesSize(), 10);
  };
}

/* unknown root graph
 *                           known subgraph
      refdata                +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
测试场景：携带私有格式的refdata直连静态子图场景，静态子图中的refdata预期也是私有格式
*/
TEST_F(RefDataSt, refdata_in_dynamic_graph_connect_knownshape_parittion_with_storage_format) {
  auto ge_env = GeRunningEnvFaker();
  gert::GertRuntimeStub runtime_stub;
  auto ops_kernel_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AiCoreLib");
  auto aicore_engine_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AIcoreEngine");
  ge_env.Reset()
       .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeEngine("AIcoreEngine")
       .KernelInfoStore("AiCoreLib")
       .GraphOptimizer("AIcoreEngine")
       .KernelBuilder(ops_kernel_builder)
       .KernelBuilder(aicore_engine_builder))
       .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp("RefData").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp("Identity").InfoStoreAndBuilder("AiCoreLib"))
       .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
       .Install(FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib"))
       .Install(FakeOp(CAST).InfoStoreAndBuilder("AiCoreLib"));

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  // 1.预置校验
  DUMP_GRAPH_WHEN("PrepareAfterUpdateInputOutputByUserOptions");
  // 2.构图
  std::vector<int64_t> shape = {1, 2, 3, 4};  // HWCN
  auto compute_graph = BuildGraphVarPartitionedCallWithSubgraph(true, false, shape);
  auto refdata1_node = compute_graph->FindNode("variable");
  auto refdata1_op_desc = refdata1_node->GetOpDesc();
  refdata1_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  (void)AttrUtils::SetBool(refdata1_op_desc->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  refdata1_op_desc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  refdata1_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_FRACTAL_Z);
  refdata1_op_desc->MutableInputDesc(0)->SetShape(GeShape());
  refdata1_op_desc->UpdateOutputDesc(0, refdata1_op_desc->GetInputDesc(0));

  auto partitioned_call_node = compute_graph->FindNode("partitioncall");
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  auto subgraph = compute_graph->GetAllSubgraphs().at(0);
  auto data_in_subgraph = subgraph->FindNode("data");
  auto data_in_subgraph_op_desc = data_in_subgraph->GetOpDesc();
  data_in_subgraph_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  (void)AttrUtils::SetBool(data_in_subgraph_op_desc->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  data_in_subgraph_op_desc->MutableInputDesc(0)->SetOriginShape(GeShape(shape));
  data_in_subgraph_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_FRACTAL_Z);
  data_in_subgraph_op_desc->MutableInputDesc(0)->SetShape(GeShape());
  data_in_subgraph_op_desc->UpdateOutputDesc(0, data_in_subgraph_op_desc->GetInputDesc(0));

  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  auto train_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // 3.add graph
  uint32_t graph_id = 3u;
  auto ret = session.AddGraph(graph_id, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 4.compile graph
  ret = session.CompileGraph(graph_id); // dont care about result
  EXPECT_EQ(ret, SUCCESS);

  // 6.校验结果
  // 6.1 check offset
  CHECK_GRAPH(PrepareAfterUpdateInputOutputByUserOptions) {
    EXPECT_EQ(graph->GetGraphUnknownFlag(), true);
    // check attrs on node of graph
    auto ref_data_main = graph->FindNode("variable");
    EXPECT_NE(ref_data_main, nullptr);
    auto subgraph = graph->GetSubgraph("sub");
    EXPECT_NE(subgraph, nullptr);
    EXPECT_EQ(subgraph->GetGraphUnknownFlag(), false);

    auto partition_node = graph->FindFirstNodeMatchType(PARTITIONEDCALL);
    auto ref_data_sub = subgraph->FindNode("variable");
    EXPECT_NE(ref_data_sub, nullptr);
    EXPECT_EQ(ref_data_sub->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), FORMAT_FRACTAL_Z);
  };
}
