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
#include <numeric>
#include <string>
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "faker/fake_value.h"
#include "runtime/base.h"
#include "ge/ge_api.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_graph_compile_summary.h"
#include "graph/execute/model_executor.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "stub/gert_runtime_stub.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/taskdef_builder.h"
#include "common/args_checker.h"
#include "graph/load/model_manager/model_manager.h"
#include "init_ge.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "utils/synchronizer.h"
#include "common/global_variables/diagnose_switch.h"
#include "hcom/hcom_topo_info.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"

extern ge::SessionManager *GetSessionManager();
namespace ge {
using namespace gert;
namespace {
class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    void *mem = nullptr;
    (void)rtMalloc(&mem, size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
    malloc_cnt++;
    return new (std::nothrow) MemBlock(*this, mem, size);
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    malloc_advise_cnt++;
    return Malloc(size);
  }
  void Free(MemBlock *block) override {
    if (block != nullptr) {
      rtFree(block->GetAddr());
      delete block;
    }
  }
  uint32_t GetMallocCnt() {
    return malloc_cnt;
  }
  uint32_t GetMallocAdviseCnt() {
    return malloc_advise_cnt;
  }
 private:
  uint32_t malloc_cnt = 0;
  uint32_t malloc_advise_cnt = 0;
};
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
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
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

Status GenerateTaskForMemCopyAyncTsMemory(const Node &node,
                                          RunContext &run_context,
                                          std::vector<domi::TaskDef> &tasks) {
    if ((node.GetType() != MEMCPYASYNC) && (node.GetType() != IDENTITY)) {
      return SUCCESS;
    }

    if (node.GetType() == IDENTITY) {
      std::vector<int64_t> memtype_list = {RT_MEMORY_TS};
      auto op_desc = node.GetOpDesc();
      (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list);
    }

    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    auto kernel_def = task_def.mutable_memcpy_async();
    kernel_def->set_op_index(node.GetOpDesc()->GetId());
    kernel_def->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
    uint8_t *membase =  run_context.dataMemBase;
    kernel_def->set_src((uintptr_t)membase + node.GetOpDesc()->GetInputOffset()[0]);
    if (node.GetType() == IDENTITY) {
      kernel_def->set_dst_max(1000);
    }
    kernel_def->set_dst((uintptr_t)membase + node.GetOpDesc()->GetOutputOffset()[0]);
    tasks.emplace_back(task_def);
    return SUCCESS;
}

class FakeOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
    return true;
  }
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override {}

  Status LoadTask(GETaskInfo &task) override {
    HcclDumpInfo dump_info = {0U, 0U, 0U, (void *)0x01, 1U, (void *)0x02, 1U};
    GETaskKernelHcclInfo kernel_hccl_info;
    task.kernelHcclInfo.emplace_back(kernel_hccl_info);
    task.kernelHcclInfo[0].hccl_dump_info.emplace_back(dump_info);
    return SUCCESS;
  }

  Status UnloadTask(GETaskInfo &task) override {
    return SUCCESS;
  }
};
FakeOpsKernelInfoStore g_fake_ops_kernel_info_store;

Status GenerateTaskForHcomAllReduce(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  std::cout << "======node.GetType():" << node.GetType()  << std::endl;
  if (node.GetType() != "HcomAllReduce") {
      std::cout << "*****return***"<< std::endl;
      return SUCCESS;
  }

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
  task_def.set_stream_id(0);
  const auto op_desc = node.GetOpDesc();
  OpsKernelInfoStore *ptr = &g_fake_ops_kernel_info_store;
  op_desc->SetExtAttr("OpsKernelInfoStorePtr", ptr);
  (void)ge::AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "sum");
  int32_t root_id = 0;
  (void)ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
  auto &kernel_hccl_def = *task_def.mutable_kernel_hccl();
  kernel_hccl_def.set_op_index(op_desc->GetId());
  kernel_hccl_def.set_hccl_type("HcomAllReduce");
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForHcomAllToAll(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  std::cout << "======node.GetType():" << node.GetType()  << std::endl;
  if (node.GetType() != "HcomAllToAll") {
      std::cout << "*****return***"<< std::endl;
      return SUCCESS;
  }

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
  task_def.set_stream_id(0);
  const auto op_desc = node.GetOpDesc();
  OpsKernelInfoStore *ptr = &g_fake_ops_kernel_info_store;
  op_desc->SetExtAttr("OpsKernelInfoStorePtr", ptr);
  int32_t root_id = 0;
  (void)ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
  auto &kernel_hccl_def = *task_def.mutable_kernel_hccl();
  kernel_hccl_def.set_op_index(op_desc->GetId());
  kernel_hccl_def.set_hccl_type("HcomAllToAll");
  tasks.emplace_back(task_def);
  return SUCCESS;
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

Status GenerateTaskForMemCopyAync(const Node &node,
                                  RunContext &run_context,
                                  std::vector<domi::TaskDef> &tasks) {
  if ((node.GetType() != MEMCPYASYNC) && (node.GetType() != IDENTITY)) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  auto kernel_def = task_def.mutable_memcpy_async();
  kernel_def->set_op_index(node.GetOpDesc()->GetId());
  kernel_def->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  uint8_t *membase =  run_context.dataMemBase;
  kernel_def->set_src((uintptr_t)membase + node.GetOpDesc()->GetInputOffset()[0]);
  kernel_def->set_dst((uintptr_t)membase + node.GetOpDesc()->GetOutputOffset()[0]);
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForAicpu(const Node &node,
                            RunContext &run_context,
                            std::vector<domi::TaskDef> &tasks) {
    if (node.GetType() != NEG) {
      return SUCCESS;
    }

    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(0));
    return SUCCESS;
}

Status GenerateTaskForDsa(const Node &node,
                          RunContext &run_context,
                          std::vector<domi::TaskDef> &tasks) {
    if (node.GetType() != "DSARandomNormal") {
      return SUCCESS;
    }

    auto op_desc = node.GetOpDesc();
    op_desc->SetWorkspace({1308, 1458});
    op_desc->SetWorkspaceBytes({150, 150});

    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA));
    domi::DSATaskDef *dsa_task = task_def.mutable_dsa_task();

    dsa_task->set_op_index(op_desc->GetId());
    dsa_task->set_start(1);
    dsa_task->set_sqe_type(1);
    dsa_task->set_distribution_type(1);
    dsa_task->set_data_type(1);
    dsa_task->set_alg_type(1);
    dsa_task->set_input_vld(1);
    dsa_task->set_input_value_addr_flag(0);
    dsa_task->set_input1_value_or_ptr(0);
    dsa_task->set_input2_value_or_ptr(0);
    dsa_task->set_seed_value_or_ptr(0);
    dsa_task->set_random_count_value_or_ptr(0);

    tasks.emplace_back(task_def);
    return SUCCESS;
}

Status GenerateTaskForStreamSwitch(const Node &node,
                          RunContext &run_context,
                          std::vector<domi::TaskDef> &tasks) {
    if (node.GetType() != STREAMSWITCH) {
      return SUCCESS;
    }

    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_SWITCH));
    auto kernel_def = task_def.mutable_stream_switch();
    kernel_def->set_op_index(node.GetOpDesc()->GetId());
    tasks.emplace_back(task_def);

    return SUCCESS;
}

Status GenerateTaskForRts(const Node &node,
                          RunContext &run_context,
                          std::vector<domi::TaskDef> &tasks) {
    Status ret = GenerateTaskForMemCopyAync(node, run_context, tasks);
    ret = GenerateTaskForStreamSwitch(node, run_context, tasks);
    return ret;
}

/**
 *    Data       Const
 *   [2, 3, 4, 5]    [1,2,3,4,5]
 *             \     /
 *              Reshape
 *                 |
 *         Relu[1,2,3,4,5]
 *                 |
 *             NetOutpu
 */
Graph BuildReshapeGraph() {
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec = {1, 2, 3, 4, 5};
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *) data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int32_t));

  auto data = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4, 5}).InCnt(1).OutCnt(1).InNames({"x"})
                      .OutNames({"y"}).Build("data");
  auto const1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {5}).InCnt(1).OutCnt(1).InNames({"x"})
                      .OutNames({"y"}).Weight(data_tensor).Build("const1");
  auto reshape = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).InNames({"x1", "x2"}).OutNames({"y"})
                      .TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5}).Build("reshape");
  auto relu = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5})
                      .InNames({"x"}).OutNames({"y"}).Build("relu");
  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data)
              ->EDGE(0, 0)
              ->NODE(reshape)
              ->NODE(relu)
              ->NODE("netoutput", netoutput));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE("reshape"));
  };
  auto ge_graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(ge_graph);
  compute_graph->FindNode("data")->GetOpDesc()->MutableOutputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({2, 3, 4, 5})));
  compute_graph->FindNode("reshape")->GetOpDesc()->MutableInputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({2, 3, 4, 5})));
  compute_graph->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({1, 2, 3, 4, 5})));
  compute_graph->FindNode("relu")->GetOpDesc()->MutableInputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({1, 2, 3, 4, 5})));
  return ge_graph;
}

void ConstructInputOutputTensor(std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                                size_t output_num = 1U) {
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(1, 0);
  TensorDesc desc_2(Shape({1}));
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
  inputs.emplace_back(input_tensor_2);

  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(96, 0xff);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);
  }
  return;
}

void ConstructInputWithBool(std::vector<ge::Tensor> &inputs) {
  std::vector<int8_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}), FORMAT_ND, DT_BOOL);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int8_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int8_t> input_data_2(1, 0);
  TensorDesc desc_2(Shape({1}), FORMAT_ND, DT_BOOL);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int8_t));
  inputs.emplace_back(input_tensor_2);
}

Status CompileAndGetFixedSize(Session &session, uint32_t graph_id, size_t &hbm_fixed_feature_size,
                              size_t &p2p_fixed_feature_size) {
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  EXPECT_EQ(all_feature_memory.size(), 2U);
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  EXPECT_NE(p2p_fixed_feature_size, 0U);
  EXPECT_NE(hbm_fixed_feature_size, 0U);
  return SUCCESS;
}

Status SetModelVarSize(Session &session, uint32_t graph_id) {
  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto &ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 7));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 1536));
  return SUCCESS;
}

void ConstructIOTensor(std::vector<ge::Tensor> &inputs,
                                std::vector<ge::Tensor> &outputs) {
  std::vector<int64_t> input_data_1(2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2}));
  desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int64_t> input_data_2(2 * 2, 0);
  TensorDesc desc_2(Shape({2, 2}));
  desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_2);

  std::vector<uint8_t> output_data_1(128, 0xff);
  TensorDesc output_desc_1(Shape({2, 2}));
  output_desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);

  std::vector<uint8_t> output_data_2(128, 0xff);
  TensorDesc output_desc_2(Shape({2, 2}));
  output_desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_2{output_desc_2};
  output_tensor_2.SetData(output_data_2.data(), output_data_2.size());
  outputs.emplace_back(output_tensor_2);
}

}

class FmMemoryRefreshTest : public testing::Test {
 protected:
  void SetUp() {
    ModelManager::GetInstance().ClearAicpuSo();
    MockGenerateTask();
  }
  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("RTSLib");
  }
};

/*
 * 用例描述: 使用CompileGraph接口成功编译静态地址可刷新模型

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. dump的buid图中Constant节点被转化为Const
 * 3. dump的buid图中STREAMSWITCH节点后插入MEMCPYASYNC
 */
TEST_F(FmMemoryRefreshTest, constant2const_refreshable) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == STREAMSWITCH) {
        EXPECT_EQ(node->GetInNodesSize(), 2);
        EXPECT_NE(node->GetInDataAnchor(0), nullptr);
        EXPECT_NE(node->GetInDataAnchor(1), nullptr);
        EXPECT_NE(node->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(1)->GetPeerOutAnchor(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode(), nullptr);
        EXPECT_EQ(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
        EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
      }
      if (node->GetName() == "const_2") {
        EXPECT_NE(node->GetType(), CONSTANTOP);
      }
    }
  };

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_GT(weight_size, 512U);
}

/*
 * 用例描述: 使用CompileGraph接口成功编译静态地址不可刷新模型

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 不设可刷新相关的option参数:
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. dump的buid图中Constant节点类型没有改变
 * 3. dump的buid图中STREAMSWITCH节点未插入MEMCPYASYNC
 */
TEST_F(FmMemoryRefreshTest, no_constant2const_unrefreshable) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == STREAMSWITCH) {
        EXPECT_EQ(node->GetInNodesSize(), 3);
        EXPECT_NE(node->GetInDataAnchor(0), nullptr);
        EXPECT_NE(node->GetInDataAnchor(1), nullptr);
        EXPECT_NE(node->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(1)->GetPeerOutAnchor(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode(), nullptr);
        EXPECT_NE(node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), MEMCPYASYNC);
        EXPECT_NE(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), MEMCPYASYNC);
      }
      if (node->GetName() == "const_2") {
        EXPECT_EQ(node->GetType(), CONSTANTOP);
      }
    }
  };

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(weight_size, 1024);
}

/*
 * 用例描述: 使用CompileGraph接口图不符合预期：包含variable

 * 预置条件：
 * 1. 模型中含var节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    MEMORY_OPTIMIZATION_POLICY=GE:kMemoryPriority
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型编译失败
 */
TEST_F(FmMemoryRefreshTest, compile_graph_with_invalid_variable) {
  auto compute_graph = ShareGraph::SimpleVariableAssignGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(graph_id, graph);
  EXPECT_EQ(session.CompileGraph(graph_id), PARAM_INVALID);
}

/*
 *用例描述: 使用CompileGraph接口图不符合预期：输入tensordesc未设置

 * 预置条件：
 * 1. 模型中输入tensordesc未设置
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 模型编译
 *
 * 预期结果：
 * 1. 模型编译失败
 */
TEST_F(FmMemoryRefreshTest, compile_graph_with_invalid_tensor) {
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  // invalid tensor_desc for fail
  auto invalid_td = std::make_shared<GeTensorDesc>();
  invalid_td->SetDataType(DT_UNDEFINED);
  invalid_td->SetFormat(FORMAT_RESERVED);
  compute_graph->FindNode("data_1")->GetOpDesc()->UpdateOutputDesc(0, invalid_td->Clone());
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(graph_id, graph);
  EXPECT_EQ(session.CompileGraph(graph_id), PARAM_INVALID);
}

/*
 * 用例描述: 使用CompileGraph接口入参不符合预期

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE 非 [0|1]
 *    MEMORY_OPTIMIZATION_POLICY 非GE:kMemoryPriority
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 编译报错
 */
TEST_F(FmMemoryRefreshTest, invalid_options) {
  std::map<AscendString, AscendString> options;
  Session ori_session(options);
  auto ori_session_id = ori_session.GetSessionId();

  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  Session session(options);
  EXPECT_EQ(session.GetSessionId(), ori_session_id + 1); // create success with id 0

  options.clear();
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "invalid");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  Session session1(options);
  EXPECT_EQ(session1.GetSessionId(), 0); // create failed and not increase id

  options.clear();
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "2");
  Session session2(options);
  EXPECT_EQ(session2.GetSessionId(), 0); // create failed and not increase id
}
/*
 * 用例描述: option配置地址可刷新但netOutpus不可零拷贝场景

 * 预置条件：
 * 1. 构造模型：一个算子的计算结果给netOutputs的两个输入（此场景输出不可以零拷贝成算子的计算结果，需要强制拷贝）
 * 2. 配置地址可刷新
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 * 3. 模型编译
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. netOutpus两个输入走了强制拷贝流程，data数据一致
 */
TEST_F(FmMemoryRefreshTest, fm_memory_refresh_with_outputs_nozerocopy) {
  GertRuntimeStub runtime_stub;
  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraphWithTwoOutputs();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1(output_desc);

  // output tensor not 32 aligend
  std::vector<uint8_t> output_data_1(96, 0xff);
  output_tensor_1.SetData(output_data_1.data(), 96);
  outputs.emplace_back(output_tensor_1);

  std::vector<uint64_t> invalid_output((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
    outputs[0].GetSize()));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  std::vector<uint64_t> output_data0((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
    outputs[0].GetSize()));
  std::vector<uint64_t> output_data1((uint64_t *)outputs[1].GetData(), (uint64_t *)(outputs[1].GetData() +
    outputs[1].GetSize()));
  EXPECT_NE(output_data0, invalid_output); // 非原始全F内容
  EXPECT_EQ(output_data0, output_data1);
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
}

/*
 * 用例描述: 需要atomic清零的节点，直连netoutput，测试零拷贝功能

 * 预置条件：
 * 1. 构造模型：一个算子的计算结果给netOutputs
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 * 3. 模型编译
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. netOutpus两个输入走了零拷贝流程，data数据一致
 */
TEST_F(FmMemoryRefreshTest, AtomicNodeConnectNetoutput_ZeroCopy) {
  GertRuntimeStub runtime_stub;
  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);


  auto graph = ShareGraph::BuildAtomicNodeConnectNetoutput();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1(output_desc);

  // output tensor not 32 aligend
  std::vector<uint8_t> output_data_1(96, 0xff);
  output_tensor_1.SetData(output_data_1.data(), 96);
  outputs.emplace_back(output_tensor_1);

  std::vector<uint64_t> invalid_output((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
    outputs[0].GetSize()));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  std::vector<uint64_t> output_data0((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
    outputs[0].GetSize()));
  std::vector<uint64_t> output_data1((uint64_t *)outputs[1].GetData(), (uint64_t *)(outputs[1].GetData() +
    outputs[1].GetSize()));
  EXPECT_EQ(output_data0, invalid_output); // 原始全F内容
  EXPECT_EQ(output_data0, output_data1);
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
}

/*
 * 用例描述: 需要atomic清零的节点，经过refnode连接netoutput，测试零拷贝功能

 * 预置条件：
 * 1. 构造模型：一个算子的计算结果给netOutputs
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 * 3. 模型编译
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. netOutpus两个输入走了零拷贝流程，data数据一致
 */
TEST_F(FmMemoryRefreshTest, AtomicNodeConnectNetoutputThroughRefNode_ZeroCopy) {
  GertRuntimeStub runtime_stub;
  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildAtomicNodeConnectNetoutputThroughRefNode();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1(output_desc);

  // output tensor not 32 aligend
  std::vector<uint8_t> output_data_1(96, 0xff);
  output_tensor_1.SetData(output_data_1.data(), 96);
  outputs.emplace_back(output_tensor_1);

  std::vector<uint64_t> invalid_output((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
    outputs[0].GetSize()));

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  std::vector<uint64_t> output_data0((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
    outputs[0].GetSize()));
  std::vector<uint64_t> output_data1((uint64_t *)outputs[1].GetData(), (uint64_t *)(outputs[1].GetData() +
    outputs[1].GetSize()));
  EXPECT_EQ(output_data0, invalid_output); // 原始全F内容
  EXPECT_EQ(output_data0, output_data1);
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
}

/*
 * 用例描述: option配置动态静态图复用时，GE会设置地址可刷新

 * 预置条件：
 * 1. 构造模型：一个算子的计算结果给netOutputs的两个输入
 * 2. 配置动态静态图内存复用
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 * 3. 模型编译
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, DavinciModelGetRefreshableOption_Success_WhenStaticMemoryPolicyIs4) {
  mmSetEnv("GE_USE_STATIC_MEMORY", "4", 1);
  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  std::map<std::string, std::string> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");


  /*
   * 用例设置了GE_USE_STATIC_MEMORY环境变量，需要在用例结束后取消设置。如果不加大括号，那么会先执行取消设置，再执行到Session析构，
   * Session析构会销毁DavinciModel，后者会释放内存，如果获取不到GE_USE_STATIC_MEMORY，
   * DavinciModel::FreeFeatureMapMem会走另外一个分支，无法走到ActiveMemoryAllocator::FreeMemory，
   * 那么ActiveMemoryAllocator对象只能在单例析构时调用rtReleaseMemAddress，而此时so可能已经不在了，出现coredump
   */
  {
    Session session(options);
    auto graph = ShareGraph::BuildSwitchMergeGraphWithTwoOutputs();
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    auto merge_1 = compute_graph->FindNode("merge_1");
    std::vector<std::pair<NodePtr, int32_t>> out_nodes_info = {{merge_1, 0}, {merge_1, 1}};
    compute_graph->SetGraphOutNodesInfo(out_nodes_info);
    uint32_t graph_id = 11221301;
    session.AddGraph(graph_id, graph);

    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    ConstructInputOutputTensor(inputs, outputs);
    inputs.clear();
    ConstructInputWithBool(inputs);
    TensorDesc output_desc(Shape({1, 2, 3, 4}));
    ge::Tensor output_tensor_1(output_desc);

    // output tensor not 32 aligend
    std::vector<uint8_t> output_data_1(96, 0xff);
    output_tensor_1.SetData(output_data_1.data(), 96);
    outputs.emplace_back(output_tensor_1);

    std::vector<uint64_t> invalid_output((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
                                                                                        outputs[0].GetSize()));
    EXPECT_EQ(SUCCESS, session.BuildGraph(graph_id, inputs));
    Synchronizer sync;
    auto ret = session.RunGraphAsync(graph_id, inputs, [&sync](Status run_ret, std::vector<ge::Tensor> &) {
      EXPECT_EQ(run_ret, SUCCESS);
      sync.OnDone();
    });
    EXPECT_EQ(ret, SUCCESS);
    sync.WaitFor(5);
    std::vector<uint64_t> output_data0((uint64_t *)outputs[0].GetData(), (uint64_t *)(outputs[0].GetData() +
                                                                                      outputs[0].GetSize()));
    std::vector<uint64_t> output_data1((uint64_t *)outputs[1].GetData(), (uint64_t *)(outputs[1].GetData() +
                                                                                      outputs[1].GetSize()));
    EXPECT_EQ(output_data0, output_data1);
    unsetenv("GE_DAVINCI_MODEL_PROFILING");
  }
  unsetenv("GE_USE_STATIC_MEMORY");
}

/*
 * 用例描述: 模型第一次run由内部申请feature内存，第二次run由外部传入feature内存

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 第一次run由内部申请feature内存，第二次run由外部传入feature内存
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 * 3. 模型编译
 * 4. 模型执行
 * 5. fm内存申请并刷新
 * 6. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, fm_memory_refresh_with_inner_mem_free) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
}

/*
 * 用例描述: 用户input tensor个数小于计算图输入节点个数

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 用户input tensor个数小于计算图输入节点个数

 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 * 3. 模型编译
 * 4. 模型执行
 * 5. fm内存申请并刷新
 * 6. 输入inputs tensor个数为1
 * 6. 模型执行
 *
 * 预期结果：
 * 1. 模型执行失败
 */
TEST_F(FmMemoryRefreshTest, fm_memory_refresh_with_check_tensor_index) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<uint8_t> output_data_1(96, 0xff);
  TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);

  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
}

/*
 * 用例描述: 用户output tensor的size非法

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 用户input tensor个数小于计算图输入节点个数

 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 * 3. 模型编译
 * 4. 模型执行
 * 5. fm内存申请并刷新
 * 6. 输入outputs tensor的size非法
 * 6. 模型执行
 *
 * 预期结果：
 * 1. 模型执行失败
 */
TEST_F(FmMemoryRefreshTest, fm_memory_refresh_with_check_tensor_size) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);
  std::vector<int32_t> input_data_2(1, 0);
  TensorDesc desc_2(Shape({1}));
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
  inputs.emplace_back(input_tensor_2);

  ge::Tensor output_tensor_1;
  outputs.emplace_back(output_tensor_1);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/* 用例描述: 混用
* 预置条件：
* 1. 构造符合compile接口编译的模型
*
* 测试步骤：
* 1. ir构造计算图
* 2. Compile接口模型编译
* 3. RunGraph/RunGraphAsync/BuildGraph接口重复编译或执行相同的graphid
*
* 预期结果：
* 1. 模型执行失败
*/
TEST_F(FmMemoryRefreshTest, compile_graph_incompatible_with_other_apis) {
  std::map<AscendString, AscendString> options;
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  EXPECT_NE(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);

  Status ret = SUCCESS;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  auto callback = [&] (Status status, std::vector<ge::Tensor> &outputs) {
    std::unique_lock<std::mutex> lk(mu);
    done = true;
    ret = status;
    cv.notify_all();
  };
  session.RunGraphAsync(graph_id, inputs, callback);
  std::unique_lock<std::mutex> lk(mu);
  cv.wait_for(lk, std::chrono::seconds(5), [&]() {
    return done;
  });
  EXPECT_NE(ret, SUCCESS);
}


/*
 * 用例描述: 未进行编译机调用新增的查询summary、设置刷新地址的API

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 外部申请权重/FM/输入输出内存
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE=graph
 * 3. 调用新增的查询summary、设置刷新地址的API
 *
 * 预期结果：
 * 1. API返回失败
 */
TEST_F(FmMemoryRefreshTest, get_summary_without_compiled) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph, options);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_EQ(summary, nullptr);
  size_t weight_size = 512U;
  size_t feature_size = 512U;
  size_t fixed_size = 512U;
  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  std::vector<uint8_t> fixed_feature_mem(feature_size, 0);
  EXPECT_NE(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_NE(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
  EXPECT_NE(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fixed_size));
  EXPECT_NE(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
}

/*
 * 用例描述: UpdateGraphRefreshableFeatureMemoryBase/SetGraphFixedFeatureMemoryBase api接口测试

 * 预期结果：相同graph首次调用UpdateGraphRefreshableFeatureMemoryBase/SetGraphFixedFeatureMemoryBase成功，后面再次调用失败
 */
TEST_F(FmMemoryRefreshTest, fixed_feature_memory_base_test) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);

  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto &ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  EXPECT_NE(ge_model, nullptr);
  uint64_t mem = 0UL;
  std::vector<std::vector<int64_t>> sub_mem_infos;
  std::vector<int64_t> sub_mem_offset;
  sub_mem_offset.emplace_back(0x2U);// mem_type RT_MEMORY_HBM 0x2U
  sub_mem_offset.emplace_back((int64_t)(&mem));// mem_offset_base
  sub_mem_offset.emplace_back(sizeof(mem)); // mem_size
  sub_mem_offset.emplace_back(1UL); // is_fixed_addr_prior
  sub_mem_infos.emplace_back(sub_mem_offset);
  AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);

  size_t refreshable_feature_size = 0U;
  size_t fixed_feature_size = 0U;
  size_t feature_size = 0U;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fixed_feature_size));

  std::vector<uint8_t> feature_mem(feature_size, 0);
  std::vector<uint8_t> fixed_feature_mem(fixed_feature_size, 0);
  std::vector<uint8_t> refreshable_feature_mem(refreshable_feature_size, 0);
  EXPECT_NE(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(), 0));
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, feature_mem.data(), fixed_feature_size));
  EXPECT_NE(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0));
  EXPECT_NE(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(), 0));
  EXPECT_EQ(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(),
      refreshable_feature_size));
  EXPECT_NE(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
  runtime_stub.Clear();
}

/*
 * 用例描述: 输入动态shape模型编译后调用api失败，编译成功且IsStatic返回false

 * 预置条件：
 * 1. 动态shape输入
 *
 * 测试步骤：
 * 1. ir构造计算图，包含动态该shape输入
 * 2. 设option参数并编译
 * 3. 调用新增api接口
 *
 * 预期结果：
 * 1. CompileGraph、GetCompiledGraphSummary返回成功
 * 2. IsStatic返回false
 * 3. 获取以及设置const/feature地址返回失败
 */
TEST_F(FmMemoryRefreshTest, get_summary_unsupport_dynamic_input) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  Session session(options);

  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});
    auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});
    auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});
    CHAIN(NODE("_arg_0", data_0)->NODE("add", add)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", data_1)->NODE("add", add));
  };

  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[1~20,1~30]";
  auto graph = ToGeGraph(dynamic_graph);
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  EXPECT_EQ(summary->IsStatic(), false);

  size_t weight_size = 0U;
  size_t feature_size = 0U;
  size_t fix_size = 0U;
  EXPECT_NE(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_NE(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_NE(SUCCESS, summary->GetRefreshableFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_size));
  EXPECT_EQ(fix_size, 0);
  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  std::vector<uint8_t> fix_feature_mem(20, 0);
  EXPECT_NE(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_NE(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
  /* fix Feature Memor support dynamic */
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fix_feature_mem.data(), 20));
  EXPECT_NE(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
}

/*
 * 用例描述: 多次调用SetGraphConstMemoryBase/SetGraphFixedFeatureMemoryBase失败，不支持重复设置

 * 预期结果：相同graph首次调用etGraphConstMemoryBase成功，后面再次调用失败
 */
TEST_F(FmMemoryRefreshTest, repeated_set_const_base_failed) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  Session session(options);


  auto graph = ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto add_node = compute_graph->FindNode("add_1");
  ASSERT_NE(add_node, nullptr);
  ge::AttrUtils::SetBool(add_node->GetOpDesc(), ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);

  size_t weight_size = 0U;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  std::vector<uint8_t> weight_mem(weight_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_NE(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));

  size_t fixed_mem_size = 0U;
  summary->GetFixedFeatureMemorySize(fixed_mem_size);
  std::vector<uint8_t> fix_feature_mem(fixed_mem_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fix_feature_mem.data(), fixed_mem_size));
  EXPECT_NE(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fix_feature_mem.data(), fixed_mem_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  EXPECT_NE(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  runtime_stub.Clear();
}

/*
 * 用例描述: 静态不可刷新模型，不支持在load后刷新fm地址

 * 预置条件：
 * 1. 静态不可刷新模型
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. graph1不设置可刷新编译选项，graph2设置可刷新编译选项，开始编译
 * 3. 编译玩获取summary，校验summary可刷新是否正确
 * 3. graph1 RunGraphWithStreamAsync后调用api刷新fm地址
 *
 * 预期结果：
 * 1. RunGraphWithStreamAsync执行成功
 * 2. GetFeatureMemoryBaseRefreshable接口返回，graph1不可刷新，graph2可刷新
 * 2. Load后调用刷新fm地址失败
 */
TEST_F(FmMemoryRefreshTest, unrefreshable_graph_update_fm_failed) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options1;
  options1.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  Session session1(options1);


  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 0;
  EXPECT_EQ(session1.AddGraph(graph_id, graph, options1), SUCCESS);
  EXPECT_EQ(session1.CompileGraph(graph_id), SUCCESS);

  std::map<AscendString, AscendString> options2;
  options2.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  EXPECT_EQ(session1.AddGraph(1, BuildReshapeGraph(), options2), SUCCESS);
  EXPECT_EQ(session1.CompileGraph(1), SUCCESS);

  const CompiledGraphSummaryPtr summary1 = session1.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary1, nullptr);
  const CompiledGraphSummaryPtr summary2 = session1.GetCompiledGraphSummary(1);
  EXPECT_NE(summary2, nullptr);

  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary1->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);
  EXPECT_EQ(SUCCESS, summary2->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  size_t feature_size = 0U;
  EXPECT_EQ(SUCCESS, summary1->GetFeatureMemorySize(feature_size));
  std::vector<uint8_t> feature_mem(feature_size, 0);
  // unfreshable graph
  EXPECT_EQ(SUCCESS, session1.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size)); // set success
  EXPECT_NE(SUCCESS, session1.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size)); // repeated
  // Update Graph Feature Memory can not set fix memory
  std::vector<uint8_t> fix_feature_mem(20, 0);
  EXPECT_NE(SUCCESS, session1.SetGraphFixedFeatureMemoryBase(graph_id, fix_feature_mem.data(), 20));
  // freshable graph
  EXPECT_EQ(SUCCESS, session1.UpdateGraphFeatureMemoryBase(1, feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, session1.UpdateGraphFeatureMemoryBase(1, feature_mem.data(), feature_size));
  // Update Graph Feature Memory can not set fix memory
  EXPECT_NE(SUCCESS, session1.SetGraphFixedFeatureMemoryBase(1, fix_feature_mem.data(), 20));

  std::vector<ge::Tensor> inputs, outputs;
  ConstructInputOutputTensor(inputs, outputs);
  EXPECT_EQ(SUCCESS, session1.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  // update failed for unrefreshable
  EXPECT_NE(SUCCESS, session1.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
}

/*
 * 用例描述: 获取netoutput shapes

 * 预置条件：
 * 1. 静态可刷新模型
 *
 * 测试步骤：
 * 1. ir构造计算图并编译
 * 2. 不设置可刷新编译选项，直接编译
 * 3. 调用GetCompiledGraphSummary获取summary。然后调用内部GetOutputShapes获取shapes
 *
 * 预期结果：
 * 1. shapes符合预期
 */
TEST_F(FmMemoryRefreshTest, get_summary_netoutput_shapes) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  std::vector<ge::Shape> netoutput_shapes;
  std::vector<ge::Shape> expected_shapes = {Shape({1, 2, 3, 4})};
  std::vector<ge::DataType> dtypes;
  EXPECT_EQ(SUCCESS, summary->GetOutputShapes(netoutput_shapes));
  EXPECT_EQ(SUCCESS, summary->GetOutputDtypes(dtypes));
  ASSERT_EQ(3, dtypes.at(0));

  size_t shapes_size = 0U;
  for (const auto &shape : netoutput_shapes) {
    shapes_size += shape.GetShapeSize();
  }
  size_t expected_size = 0U;
  for (const auto &shape : expected_shapes) {
    expected_size += shape.GetShapeSize();
  }
  EXPECT_TRUE(shapes_size == expected_size);
}

/*
 * 用例场景: 纯静态图多子图场景，获取netoutput shapes只返回整图的输出，补不应该包含子图的
 */
TEST_F(FmMemoryRefreshTest, get_summary_netoutput_shapes_for_subgraph) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto compute_graph = gert::ShareGraph::BuildGraphRefdataWhile();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  std::vector<ge::Shape> netoutput_shapes;
  std::vector<ge::Shape> expected_shapes = {Shape({8, 3, 16, 16})};
  EXPECT_EQ(SUCCESS, summary->GetOutputShapes(netoutput_shapes));

  EXPECT_EQ(netoutput_shapes.size(), expected_shapes.size());
  for (size_t i = 0U; i < netoutput_shapes.size(); i++) {
    EXPECT_EQ(netoutput_shapes[i].GetDims(), expected_shapes[i].GetDims());
  }
}

/*
 * 用例描述: option配置地址可刷新但入不可零拷贝场景

 * 预置条件：
 * 1. 构造模型：两个data输入给到hcom连续拷贝的节点（此场景输入不可以零拷贝，需要强制拷贝）
 * 2. 配置地址可刷新
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 * 3. 模型编译
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 使用了rtemcpyAsync进行了输入的强制拷贝
 */
TEST_F(FmMemoryRefreshTest, fm_memory_refresh_with_inputs_nozerocopy) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildHcomGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  EXPECT_NE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size(), 0);
}

/********************* input output reuse mem in graph st *********************/

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "0xff"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_input_reuse_mem_indexes_01) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0xff");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_NE(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "0, 0"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_input_reuse_mem_indexes_02) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0, 0");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_NE(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "-3"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_input_reuse_mem_indexes_03) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "-3");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_NE(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "0xff"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_output_reuse_mem_01) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0xff");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_NE(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "-1"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_output_reuse_mem_02) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "-1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_NE(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "0, 0"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_output_reuse_mem_03) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0, 0");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_NE(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES/OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_GRAPH_IO_MEM_ALLOC_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "0"
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "0"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_input_output_reuse_mem_01) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_GRAPH_IO_MEM_ALLOC_MODE, "ByGE");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES/OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "0"
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "0"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. AddGraph成功
 */
TEST_F(FmMemoryRefreshTest, check_input_output_reuse_mem_02) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetNeedIteration(true);
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES/OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_FEATURE_BASE_REFRESHABLE=0
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "0"
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "0"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验成功
 */
TEST_F(FmMemoryRefreshTest, check_input_output_reuse_mem_03) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(SUCCESS, ret);
}

/*
 * 用例描述: 校验OPTION_INPUT_REUSE_MEM_INDEXES/OPTION_OUTPUT_REUSE_MEM_INDEXES 参数的合法性

 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/STREAMSWITCH节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_CONST_LIFECYCLE="graph"
 *    OPTION_FEATURE_BASE_REFRESHABLE=0
 *    OPTION_GRAPH_RUN_MODE = 1
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES = "0"
 *    OPTION_INPUT_REUSE_MEM_INDEXES = "1"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型添加校验失败
 */
TEST_F(FmMemoryRefreshTest, check_input_output_reuse_mem_04) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;

  Status ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(SUCCESS, ret);
}

/**
 * ----REF_DATA reuse mem case -------------------------
 * -------------------------------------------------
 *         ref_data0  // set data reuse mem
 *             \
 *              \    Constant
 *               \   / |  |
 *                Add1 |  |
 *                 \   |  |
 *                  \  |  |
 *                   Add2 |
 *                    \   |
 *                     \  |
 *                      Add3
 *                       \
 *                        |
 *                       NetOutput  // set netoutput reuse mem
 * ------------------------------------------------
 *   data1、add2 reuse mem: offset = 1024 size = 1024
 *   add1、add3 reuse mem : offset = 0, size = 1024
 *   内存分配结果 -----offset ------ size ----------
 *   data1            1024         1024
 *   add1             0            1024   // reuse netoutput
 *   add2             1024         1024   // reuse data1
 *   add3             0            1024
 */
Graph BuildRefDataReuseMemGraph() {
  std::vector<int64_t> shape{1, 4, 8, 8};
  auto data_0 = OP_CFG("RefData")
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  vector<int32_t> data_value(1 * 4 * 8 * 8, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_3");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    ADD_OUTPUT(add_3, 0);
  };

  return ToGeGraph(g1);
}

// When IO allocation mode is ByGE, RefData is not supported in the graph.
TEST_F(FmMemoryRefreshTest, check_ref_data_and_io_allocation_mode)  {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_IO_MEM_ALLOC_MODE, "ByGE");
  Session session(options);
  auto graph = BuildRefDataReuseMemGraph();
  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, PARAM_INVALID);
  runtime_stub.Clear();
}

TEST_F(FmMemoryRefreshTest, ref_data_reuse_mem_01)  {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildRefDataReuseMemGraph();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto data1_op_desc = compute_graph->FindNode("data_0")->GetOpDesc();
  const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
  const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
  const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
  const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

  const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
  const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
  const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

  // check reuse mem
  EXPECT_EQ(data1_output_size_list[0], add2_output_size_list[0]);
  EXPECT_EQ(data1_output_offset_list[0], add2_output_offset_list[0]);

  EXPECT_EQ(add1_output_size_list[0], add3_output_size_list[0]);
  EXPECT_EQ(add1_output_offset_list[0], add3_output_offset_list[0]);

  runtime_stub.Clear();
}

/**
 * ----AIPP_DATA reuse mem case -------------------------
 * -------------------------------------------------
 *         AIPP_data0  // set data reuse mem
 *             \
 *              \    Constant
 *               \   / |  |
 *                Add1 |  |
 *                 \   |  |
 *                  \  |  |
 *                   Add2 |
 *                    \   |
 *                     \  |
 *                      Add3
 *                       \
 *                        |
 *                       NetOutput  // set netoutput reuse mem
 * ------------------------------------------------
 *   data1、add2 reuse mem: offset = 1024 size = 1024
 *   add1、add3 reuse mem : offset = 0, size = 1024
 *   内存分配结果 -----offset ------ size ----------
 *   data1            1024         1024
 *   add1             0            1024   // reuse netoutput
 *   add2             1024         1024   // reuse data1
 *   add3             0            1024
 *
 */
Graph BuildAippDataReuseMemGraph() {
  std::vector<int64_t> shape{1, 4, 8, 8};
  auto data_0 = OP_CFG(AIPP_DATA_TYPE)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  vector<int32_t> data_value(1 * 4 * 8 * 8, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_3");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    ADD_OUTPUT(add_3, 0);
  };

  return ToGeGraph(g1);
}

TEST_F(FmMemoryRefreshTest, aipp_data_reuse_mem_01)  {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildAippDataReuseMemGraph();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto data1_op_desc = compute_graph->FindNode("data_0")->GetOpDesc();
  const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
  const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
  const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
  const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

  const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
  const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
  const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

  // check reuse mem
  EXPECT_EQ(data1_output_size_list[0], add2_output_size_list[0]);
  EXPECT_EQ(data1_output_offset_list[0], add2_output_offset_list[0]);

  EXPECT_EQ(add1_output_size_list[0], add3_output_size_list[0]);
  EXPECT_EQ(add1_output_offset_list[0], add3_output_offset_list[0]);

  runtime_stub.Clear();
}

/**
 * ----DATA reuse mem case -------------------------
 * -------------------------------------------------
 *            Data0  // set data reuse mem
 *             \
 *              \    Constant
 *               \   / |  |
 *                Add1 |  |
 *                 \   |  |
 *                  \  |  |
 *                   Add2 |
 *                    \   |
 *                     \  |
 *                      Add3
 *                       \
 *                        |
 *                       NetOutput  // set netoutput reuse mem
 * ------------------------------------------------
 *   data1、add2 reuse mem: offset = 1024 size = 1024
 *   add1、add3 reuse mem : offset = 0, size = 1024
 *   内存分配结果 -----offset ------ size ----------
 *   data1            1024         1024
 *   add1             0            1024   // reuse netoutput
 *   add2             1024         1024   // reuse data1
 *   add3             0            1024
 */
Graph BuildIoReuseMemGraph() {
  std::vector<int64_t> shape{1, 4, 8, 8};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  vector<int32_t> data_value(1 * 4 * 8 * 8, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_3");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    ADD_OUTPUT(add_3, 0);
  };

  return ToGeGraph(g1);
}


/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的FM地址, 配置复用输入输出内存, 输入输出内存被整段复用
            workspace p2p内存不复用输入输出内存
 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 外部申请权重/FM/输入输出内存
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_INPUT_REUSE_MEM_INDEXES="0"
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES="1"
 * 3. 模型编译
 * 4. 权重内存申请并设置
 * 5. fm内存申请并刷新
 * 6. 模型执行
 * 7. fm内存申请并刷新
 * 8. 模型执行（多次更新）
 *
 * 预期结果：
 * 1. 模型编译成功
 * 2. 模型执行成功
 * 3. 输入输出被复用，FM size = 0
 */
TEST_F(FmMemoryRefreshTest, p2p_workspace_not_reuse_io_01)  {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildIoReuseMemGraph();

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add_1");
  auto op_desc = node->GetOpDesc();
  //op_desc->SetWorkspace({0,0});
  op_desc->SetWorkspaceBytes({32, 32});

  std::vector<int64_t> workspace_memtype_list = {RT_MEMORY_P2P_DDR, RT_MEMORY_HBM};
  vector<int32_t> workspace_no_reuse_scope = { 1, 1 };
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_memtype_list);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  // auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto data1_op_desc = compute_graph->FindNode("data_0")->GetOpDesc();
  const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
  const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
  const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
  const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

  const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
  const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
  const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

  EXPECT_EQ(data1_output_size_list[0], add2_output_size_list[0]);
  EXPECT_EQ(data1_output_offset_list[0], add2_output_offset_list[0]);

  EXPECT_EQ(add1_output_size_list[0], add3_output_size_list[0]);
  EXPECT_EQ(add1_output_offset_list[0], add3_output_offset_list[0]);

  runtime_stub.Clear();
}

/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的FM地址, 配置复用输入输出内存, 输入输出内存被整段复用

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 外部申请权重/FM/输入输出内存
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_INPUT_REUSE_MEM_INDEXES="0"
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES="1"
 * 3. 模型编译
 * 4. 权重内存申请并设置
 * 5. fm内存申请并刷新
 * 6. 模型执行
 * 7. fm内存申请并刷新
 * 8. 模型执行（多次更新）
 *
 * 预期结果：
 * 1. 模型编译成功
 * 2. 模型执行成功
 * 3. 输入输出被复用，FM size = 0
 */
TEST_F(FmMemoryRefreshTest, data_input_output_reuse_mem_01)  {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildIoReuseMemGraph();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto data1_op_desc = compute_graph->FindNode("data_0")->GetOpDesc();
  const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
  const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
  const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
  const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

  const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
  const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
  const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

  EXPECT_EQ(data1_output_size_list[0], add2_output_size_list[0]);
  EXPECT_EQ(data1_output_offset_list[0], add2_output_offset_list[0]);

  EXPECT_EQ(add1_output_size_list[0], add3_output_size_list[0]);
  EXPECT_EQ(add1_output_offset_list[0], add3_output_offset_list[0]);

  runtime_stub.Clear();
}

/**
 * ----DATA reuse mem case -------------------------
 * -------------------------------------------------
 *            Data0  // set data reuse mem
 *             \
 *              \    Constant
 *               \   / |  |
 *                Add1 |  |
 *                 \   |  |
 *                  \  |  |
 *                   Add2 |
 *                    \   |
 *                     \  |
 *                      Add3
 *                       \
 *                        |
 *                       NetOutput  // set netoutput reuse mem
 * ------------------------------------------------
 *   data1、add2、add3 连续输入输出内存
 */
Graph BuildIoReuseMemGraphContinous() {
  std::vector<int64_t> shape{1, 4, 8, 8};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  vector<int32_t> data_value(1 * 4 * 8 * 8, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
      .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
      .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
      .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
      .Build("add_3");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    ADD_OUTPUT(add_3, 0);
  };

  return ToGeGraph(g1);
}

/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的FM地址, 配置复用输入输出内存, 输入输出内存连续

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 外部申请权重/FM/输入输出内存
 *
 * 测试步骤：
 * 1. ir构造计算图, 且图中的节点输入输出连续
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_INPUT_REUSE_MEM_INDEXES="0"
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES="0"
 * 3. 模型编译
 * 4. 权重内存申请并设置
 * 5. fm内存申请并刷新
 * 6. 模型执行
 * 7. fm内存申请并刷新
 * 8. 模型执行（多次更新）
 *
 * 预期结果：
 * 1. 模型编译成功
 * 2. 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, data_input_output_reuse_mem_continous) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildIoReuseMemGraphContinous();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
}

struct FakeFmMemoryOpsKernelBuilder : FakeOpsKernelBuilder {
 public:
  FakeFmMemoryOpsKernelBuilder(const std::string &kernel_lib_name) : FakeOpsKernelBuilder(kernel_lib_name) {}
  FakeFmMemoryOpsKernelBuilder() : FakeOpsKernelBuilder() {}

 protected:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }

  Status CalcOpRunningParam(Node &ge_node) {
    OpDescPtr op_desc = ge_node.GetOpDesc();
    if (op_desc == nullptr) {
      return FAILED;
    }

    bool is_shape_unknown = false;
    if (NodeUtils::GetNodeUnknownShapeStatus(ge_node, is_shape_unknown) == GRAPH_SUCCESS) {
      if (is_shape_unknown) {
        GELOGI("op:%s is unknown shape, does not need to calc output size.", ge_node.GetName().c_str());
        return SUCCESS;
      }
    }

    const string name = ge_node.GetName();
    const string type = ge_node.GetType();
    GELOGD("Calc op[%s:%s] running param, output size=%zu.", name.c_str(), type.c_str(), op_desc->GetOutputsSize());

    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      GeTensorDesc output_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
      Format format = output_tensor.GetFormat();
      DataType data_type = output_tensor.GetDataType();

      int64_t mem_size = 0;
      // If mem size has been set, no need reset.
      if ((TensorUtils::GetSize(output_tensor, mem_size) == GRAPH_SUCCESS) && (mem_size > 0)) {
        GELOGD("Op[%s:%s] out[%zu] mem size has been set, no need calc again, format=%s, data_type=%s, mem_size=%ld.",
              name.c_str(), type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
              TypeUtils::DataTypeToSerialString(data_type).c_str(), mem_size);
        continue;
      }

      int64_t output_mem_size = 0;
      GeShape output_shape = output_tensor.GetShape();
      if ((TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_mem_size) != GRAPH_SUCCESS) ||
          (output_mem_size < 0)) {
        GELOGE(FAILED,
              "[Calc][TensorMemSize] fail for op[%s:%s] out[%zu] mem size, mem_size=%ld, format=%s, data_type=%s.",
              name.c_str(), type.c_str(), i, output_mem_size, TypeUtils::FormatToSerialString(format).c_str(),
              TypeUtils::DataTypeToSerialString(data_type).c_str());
        REPORT_INNER_ERR_MSG(
          "E19999", "CalcTensorMemSize failed for op[%s:%s] out[%zu] mem size, mem_size=%ld, format=%s, data_type=%s.",
          name.c_str(), type.c_str(), i, output_mem_size, TypeUtils::FormatToSerialString(format).c_str(),
          TypeUtils::DataTypeToSerialString(data_type).c_str());
        return FAILED;
      }
      // +32 之后做32字节对齐
      int64_t align_size = (output_mem_size + 32 + 32 - 1UL) / 32 * 32;
      GELOGI("**Calc op[%s:%s] out[%zu] mem size is %ld, align_size=%ld, format=%s, data_type=%s.", name.c_str(),
            type.c_str(), i, output_mem_size, align_size, TypeUtils::FormatToSerialString(format).c_str(),
            TypeUtils::DataTypeToSerialString(data_type).c_str());

      TensorUtils::SetSize(output_tensor, align_size);
      if (op_desc->UpdateOutputDesc(static_cast<uint32_t>(i), output_tensor) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "[Update][OutputDesc] fail for op[%s:%s] out[%zu] desc , format=%s, data_type=%s.", name.c_str(),
              type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
              TypeUtils::DataTypeToSerialString(data_type).c_str());
        REPORT_INNER_ERR_MSG("E19999", "UpdateOutputDesc failed for op[%s:%s] out[%zu] desc , format=%s, data_type=%s.",
                          name.c_str(), type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
                          TypeUtils::DataTypeToSerialString(data_type).c_str());
        return FAILED;
      }
    }

    GELOGD("Calc op[%s:%s] running param success.", name.c_str(), type.c_str());
    return SUCCESS;
  }

  Status GenerateTask(const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    GELOGI("Start gen task for %s", node.GetName().c_str());
    return SUCCESS;
  }
};

/**
 * ----DATA reuse mem case -------------------------
 * -------------------------------------------------
 *            Data0  // set data reuse mem
 *             \
 *              \    Constant
 *               \   / |  |
 *                Add1 |  |
 *                 \   |  |
 *                  \  |  |
 *                   Add2 |
 *                    \   |
 *                     \  |
 *                      Add3
 *                       \
 *                        |
 *                       NetOutput  // set netoutput reuse mem
 * ------------------------------------------------
 *   data1、add2 reuse mem: offset = 1024 size = 1024
 *   add1、add3 reuse mem : offset = 0, size = 1024
 *   内存分配结果 -----offset ------ noalignsize-----real size ----------
 *   data1            0             992             1024
 *   add1             1024          992             1024// reuse netoutput
 *   add2             0             992             1024// reuse data1
 *   add3             1024          992             1024
 */
//[data_0] optype[Data] output[0] offset to [0] size[1024] realsize[1024] noalignsize[992] life time begin[0] life time end[2] child[0:1:0:1:1]
//[add_2] optype[Add] output[0] offset to [0] size[1024] realsize[1024] noalignsize[992] life time begin[3] life time end[4] child[1:1:0:0:1]
//[add_3] optype[Add] output[0] offset to [1024] size[1024] realsize[1024] noalignsize[992] life time begin[4] life time end[4294967295] child[0:1:0:1:1]
//[add_1] optype[Add] output[0] offset to [1024] size[1024] realsize[1024] noalignsize[992] life time begin[2] life time end[3] child[1:1:0:0:1] isref[0]
Graph BuildIoReuseMemGraphzeroCopy() {
  std::vector<int64_t> shape{1, 4, 2, 124};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_UINT8, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  vector<uint8_t> data_value(1 * 4 * 2 * 124, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_UINT8);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(uint8_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_UINT8, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_UINT8, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_UINT8, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_3");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    ADD_OUTPUT(add_3, 0);
  };

  return ToGeGraph(g1);
}

/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的FM地址, 配置复用输入输出内存, noalignsize 和real size不一致场景
             输入输出内存被整段复用

 * 预置条件：
 * 1. 构造用例01中的模型
 * 2. 外部申请权重/FM/输入输出内存
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    OPTION_INPUT_REUSE_MEM_INDEXES="0"
 *    OPTION_OUTPUT_REUSE_MEM_INDEXES="1"
 * 3. 模型编译
 * 4. 权重内存申请并设置
 * 5. fm内存申请并刷新
 * 6. 模型执行
 * 7. fm内存申请并刷新
 * 8. 模型执行（多次更新）
 *
 * 预期结果：
 * 1. 模型编译成功
 * 2. 模型执行成功
 * 3. 输入输出被复用，FM size = 0
 */
TEST_F(FmMemoryRefreshTest, data_input_output_reuse_mem_zero_copy)  {
  GertRuntimeStub runtime_stub;
  /* noalignsize 和real size不一致场景的构造打桩 */
  auto fake_builder1 = std::make_shared<FakeFmMemoryOpsKernelBuilder>("AiCoreLib");
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["AiCoreLib"] = fake_builder1;
  auto fake_builder2 = std::make_shared<FakeFmMemoryOpsKernelBuilder>("AicpuLib");
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["AicpuLib"] = fake_builder2;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildIoReuseMemGraphzeroCopy();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0");
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto data1_op_desc = compute_graph->FindNode("data_0")->GetOpDesc();
  const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
  const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
  const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
  const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

  const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
  const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
  const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

  EXPECT_EQ(data1_output_size_list[0], add2_output_size_list[0]);
  EXPECT_EQ(data1_output_offset_list[0], add2_output_offset_list[0]);

  EXPECT_EQ(add1_output_size_list[0], add3_output_size_list[0]);
  EXPECT_EQ(add1_output_offset_list[0], add3_output_offset_list[0]);

  runtime_stub.Clear();
}

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

class MockIoMemReuse {
 public:
  explicit MockIoMemReuse() {

    auto infer_fun = [](Operator &op) -> graphStatus {
      // info shape实现
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetShape(GeShape({2, 2, 2, 2}));
      return GRAPH_SUCCESS;
    };
    auto pc_infer_fun = [](Operator &op) -> graphStatus {
      // info shape实现
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetShape(GeShape({4, 2, 2, 2}));
      return GRAPH_SUCCESS;
    };
    auto ge_env = GeRunningEnvFaker();
    auto ops_kernel_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AiCoreLib");
    auto aicore_engine_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AIcoreEngine");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));

    ge_env.Reset()
         .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeEngine("AIcoreEngine")
         .KernelInfoStore("AiCoreLib")
         .GraphOptimizer("AIcoreEngine")
         .KernelBuilder(ops_kernel_builder)
         .KernelBuilder(aicore_engine_builder))
         .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("PhonyConcat").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(pc_infer_fun))
         .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("Identity").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun)
                  .AttrsDef(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                  .AttrsDef(ATTR_NAME_KERNEL_BIN_ID, "_mem_fake_id")
                  .ExtAttrsDef(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr));

  }

  virtual ~MockIoMemReuse() {
    auto ge_env = GeRunningEnvFaker();
    ge_env.InstallDefault();
  }
};

/**
 *            Data0   Data01  // set data1 reuse mem
 *             \      /
 *              \    /
 *                Add1
 *                 |    +---Constant_1
 *                 |   /    / |  |
 *                 Add2    /  |  |
 *                 |      /   /  |
 *                  \    /   |   |
 *                   Add3    |   |
 *                    \     /    |
 *                     \   /     |
 *                      Add4    /
 *                       \     /
 *                        \   /
 *                         Add5
 *                         |
 *                         NetOutput
 * ------------------------------------------------
 *  构造图的输入内存比图中的其他op输出内存大的场景，对add进行info shape打桩
 *   内存分配结果 -----offset ------ size ----------
 *   data0            512           16384
 *   data1            16896         16384
 *   add1             0             512
 *   add2             16896         512    //reuse data1
 *   add3             0             512
 *   add4             16896         512    //reuse data1
 *   add5             33280         512
 */
Graph BuildIoReuseMemGraph2() {
  std::vector<int64_t> shape{8, 8, 8, 8};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  auto data_01 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("data_01");

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

  auto add_5 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_5");

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
          ->NODE(add_5)
          );

    CHAIN(NODE(data_01)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_4));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_5));
    ADD_OUTPUT(add_5, 0);
  };

  return ToGeGraph(g1);
}

// add2 add4生命周期不重叠，都从data1的起始地址开始复用
TEST_F(FmMemoryRefreshTest, data_input_output_reuse_mem_02) {
  GertRuntimeStub runtime_stub;
  MockIoMemReuse  mock_io_reuse_mem;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildIoReuseMemGraph2();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "1");
  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  // check reuse mem reusult
  {
    auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
    const auto data1_op_desc = compute_graph->FindNode("data_01")->GetOpDesc();
    const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
    const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

    const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
    const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
    const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

    const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
    const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
    const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

    const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
    const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
    const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

    const auto add4_op_desc = compute_graph->FindNode("add_4")->GetOpDesc();
    const auto add4_output_size_list = ModelUtils::GetOutputSize(add4_op_desc);
    const auto add4_output_offset_list = add4_op_desc->GetOutputOffset();

    bool expect1 = (add2_output_offset_list[0] == data1_output_offset_list[0]);
    EXPECT_EQ(expect1, true);
    bool expect2 = (add4_output_offset_list[0] == data1_output_offset_list[0]);
    EXPECT_EQ(expect2, true);
  }
}

/**
 *            Data0   Data01   // set data1 reuse mem
 *             \      /
 *              \    /
 *                Add1
 *                 |    +---Constant_1
 *                 |   /    / |
 *                 Add2    /  |
 *                 |      /   /
 *                 |\    /   |
 *                 | Add3    |
 *                 |  \     /
 *                 |   \   /
 *                 |    Add4
 *                 |     \
 *                 |      \
 *                 +------Add5
 *                         |
 *                         NetOutput
 * ------------------------------------------------
 *  构造图的输入内存比图中的其他op输出内存大的场景，对add进行info shape打桩
 *   内存分配结果 -----offset ------ size ----------
 *   data0            512           16384
 *   data1            16896         16384
 *   add1             0             512
 *   add2             16896         512    //reuse data1
 *   add3             0             512
 *   add4             17408         512    //reuse data1
 */
Graph BuildIoReuseMemGraph3() {
  std::vector<int64_t> shape{8, 8, 8, 8};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  auto data_01 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("data_01");

  // add1-5
  auto add_1 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_1_fake_id")
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_2_fake_id")
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_3_fake_id")
      .Build("add_3");

  auto add_4 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_4_fake_id")
      .Build("add_4");

  auto add_5 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_5_fake_id")
      .Build("add_5");

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
          ->NODE(add_5)
          );

    CHAIN(NODE(data_01)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_4));
    CHAIN(NODE(add_2)->EDGE(0, 1)->NODE(add_5));
    ADD_OUTPUT(add_5, 0);
  };

  return ToGeGraph(g1);
}

// add2 add4生命周期重叠，都能复用data1, 复用地址相对于data1其实地址的偏移不一样
TEST_F(FmMemoryRefreshTest, data_input_output_reuse_mem_03) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace("ge.exec.hostSchedulingMaxThreshold", "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  MockIoMemReuse  mock_io_reuse_mem;

  auto graph = BuildIoReuseMemGraph3();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "1");
  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  {
    auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
    const auto data1_op_desc = compute_graph->FindNode("data_01")->GetOpDesc();
    const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
    const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

    const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
    const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
    const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

    const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
    const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
    const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

    const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
    const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
    const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

    const auto add4_op_desc = compute_graph->FindNode("add_4")->GetOpDesc();
    const auto add4_output_size_list = ModelUtils::GetOutputSize(add4_op_desc);
    const auto add4_output_offset_list = add4_op_desc->GetOutputOffset();

    bool expect1 = (add2_output_offset_list[0] >= data1_output_offset_list[0])
                   && (add2_output_offset_list[0] < (data1_output_offset_list[0] + data1_output_size_list[0]));
    EXPECT_EQ(expect1, true);
    bool expect2 = (add4_output_offset_list[0] >= data1_output_offset_list[0])
                   && (add4_output_offset_list[0] < (data1_output_offset_list[0] + data1_output_size_list[0]));
    EXPECT_EQ(expect2, true);

    EXPECT_EQ((add2_output_offset_list[0] != add4_output_offset_list[0]), true);
  }

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  // set fm memory base
  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  std::cout << "======weight_size:" << weight_size << ", feature_size:" << feature_size << ", weight_mem:"<< std::hex
      << (uintptr_t)weight_mem.data() << ", feature_mem:"<< std::hex << (uintptr_t)feature_mem.data() << std::endl;

  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  std::vector<int32_t> input_data_0(8 * 8 * 8 * 8, 0);
  TensorDesc desc_0(Shape({8, 8, 8, 8}));
  ge::Tensor input_tensor_0{desc_0};
  input_tensor_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_0);

  std::vector<int32_t> input_data_1(8 * 8 * 8 * 8, 0);
  TensorDesc desc_1(Shape({8, 8, 8, 8}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> output_data_1(2 * 2 * 2 * 2, 0);
  TensorDesc output_desc_1(Shape({2, 2, 2, 2}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(reinterpret_cast<uint8_t *>(output_data_1.data()), output_data_1.size() * sizeof(int32_t));
  outputs.emplace_back(output_tensor_1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // check addr refresh
  uint64_t data1_addr = PtrToValue(input_data_1.data());
  std::cout << "======data1_addr:" << data1_addr << std::endl;
  std::cout << "======GetRtMemcpyRecords size:"
    << runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size() << std::endl;

  std::vector<uint64_t>  task_io_addr;
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    std::cout << "=== host args ===" << std::endl;
    uint64_t *host_args_base = (uint64_t *)args.src_address;
    for (size_t i = 0U; i < (args.copy_len / sizeof(uint64_t)); i++) {
      uint64_t io_addr = host_args_base[i];
      if (io_addr != 0U) {
        task_io_addr.emplace_back(io_addr);
        std::cout << "io_addr: " << io_addr << std::endl;
      }
    }
  }

  // todo: kernel_task_info中老流程的rtMemcpy刪除後這裏可以不用修改
  EXPECT_EQ(task_io_addr.size(), 15);
  /**
   *   内存分配结果 -----offset ------ size ----------
   *   data0            512           16384
   *   data1            16896         16384
   *   add1             0             512
   *   add2             16896         512    //reuse data1
   *   add3             0             512
   *   add4             17408         512    //reuse data1
   *   add5             33280         512
   */

  // get data1 output addr
  uint64_t data1_out_addr = task_io_addr[1];
  // get add2 output addr
  uint64_t add2_out_addr = task_io_addr[5];
  // get add4 ouput addr
  uint64_t add4_out_addr = task_io_addr[11];

EXPECT_EQ(data1_out_addr, add2_out_addr);
EXPECT_EQ(data1_out_addr + 512, add4_out_addr);
runtime_stub.Clear();
}

/**
 * -------------------------------------------------
 *            Data0   Data1
 *             \      /
 *              \    /
 *                Add1
 *                 |    +---Constant_1
 *                 |   /    /     |
 *                 Add2    /      |
 *                 |      /       |
 *                 |\    /        |
 *                 | Add3         |
 *                 |  \           |
 *                 |   \          |
 *                 |----Add4     /
 *                       \      /
 *                        \    /
 *                        mul_1
 *                         |
 *                         NetOutput // set netoutput reuse
 * ------------------------------------------------
 *  构造图的输入内存比图中的其他op输出内存大的场景，对add进行info shape打桩
 *   内存分配结果 -----offset ------ size ----------
 *   data0            16896         512
 *   data1            17408         512
 *   add1             1024          512    // reuse netouput
 *   add2             515           512    // reuse netouput
 *   add3             1024          512    // reuse netouput
 *   add4             0             512
 *   mul_1            512         16384
 *
 */
Graph BuildIoReuseMemGraph4() {
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
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_1_fake_id")
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_2_fake_id")
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_3_fake_id")
      .Build("add_3");

  auto add_4 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_4_fake_id")
      .Build("add_4");

  auto mul_1 = OP_CFG(MUL)
      .InCnt(2)
      .OutCnt(1)
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_mul_1_fake_id")
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

  return ToGeGraph(g1);
}

void ConstructInputOutputTensorForIoReuseMemGraph(std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  inputs.clear();
  outputs.clear();

  std::vector<int32_t> input_data_0(2 * 2 * 2 * 2, 0);
  TensorDesc desc_0(Shape({2, 2, 2, 2}));
  ge::Tensor input_tensor_0{desc_0};
  input_tensor_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_0);

  std::vector<int32_t> input_data_1(2 * 2 * 2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2, 2, 2}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> output_data_1(8 * 8 * 8 * 8, 0);
  TensorDesc output_desc_1(Shape({8, 8, 8, 8}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(reinterpret_cast<uint8_t *>(output_data_1.data()), output_data_1.size() * sizeof(int32_t));
  outputs.emplace_back(output_tensor_1);
}

class MockMulInfoShape {
 public:
  explicit MockMulInfoShape() {

    auto add_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetShape(GeShape({2, 2, 2, 2}));
      return GRAPH_SUCCESS;
    };

    auto mul_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetShape(GeShape({8, 8, 8, 8}));
      return GRAPH_SUCCESS;
    };

    auto ge_env = GeRunningEnvFaker();
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
         .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib").InferShape(add_infer_fun))
         .Install(FakeOp(MUL).InfoStoreAndBuilder("AiCoreLib").InferShape(mul_infer_fun))
         .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AiCoreLib"));
  }

  virtual ~MockMulInfoShape() {
    auto ge_env = GeRunningEnvFaker();
    ge_env.InstallDefault();
  }
};

// 测试net output被复用的场景
TEST_F(FmMemoryRefreshTest, data_input_output_reuse_mem_04) {
  GertRuntimeStub runtime_stub;
  MockMulInfoShape  mock_mul_info_shape;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildIoReuseMemGraph4();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  {
    auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
    const auto data1_op_desc = compute_graph->FindNode("data_1")->GetOpDesc();
    const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
    const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

    const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
    const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
    const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

    const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
    const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
    const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

    const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
    const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
    const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

    const auto add4_op_desc = compute_graph->FindNode("add_4")->GetOpDesc();
    const auto add4_output_size_list = ModelUtils::GetOutputSize(add4_op_desc);
    const auto add4_output_offset_list = add4_op_desc->GetOutputOffset();

    const auto mul1_op_desc = compute_graph->FindNode("mul_1")->GetOpDesc();
    const auto mul1_output_size_list = ModelUtils::GetOutputSize(mul1_op_desc);
    const auto mul1_output_offset_list = mul1_op_desc->GetOutputOffset();

    // check reuse result
    bool expect1 = (add1_output_offset_list[0] >= mul1_output_offset_list[0])
                   && (add1_output_offset_list[0] < (mul1_output_offset_list[0] + mul1_output_size_list[0]));
    EXPECT_EQ(expect1, true);

    bool expect2 = (add2_output_offset_list[0] >= mul1_output_offset_list[0])
                   && (add2_output_offset_list[0] < (mul1_output_offset_list[0] + mul1_output_size_list[0]));
    EXPECT_EQ(expect2, true);

    bool expect3 = (add3_output_offset_list[0] >= mul1_output_offset_list[0])
                   && (add3_output_offset_list[0] < (mul1_output_offset_list[0] + mul1_output_size_list[0]));
    EXPECT_EQ(expect3, true);

    EXPECT_EQ((add1_output_offset_list[0] != add2_output_offset_list[0]), true);
    EXPECT_EQ((add2_output_offset_list[0] != add3_output_offset_list[0]), true);
  }

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  // set fm memory base
  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  std::cout << "======weight_size:" << weight_size << ", feature_size:" << feature_size << ", weight_mem:"<< std::hex
       << (uintptr_t)weight_mem.data() << ", feature_mem:"<< std::hex << (uintptr_t)feature_mem.data() << std::endl;

  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));
  std::vector<uint8_t> fix_feature_mem(20, 0);
  EXPECT_NE(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fix_feature_mem.data(), 20));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  std::vector<int32_t> input_data_0(2 * 2 * 2 * 2, 0);
  TensorDesc desc_0(Shape({2, 2, 2, 2}));
  ge::Tensor input_tensor_0{desc_0};
  input_tensor_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_0);

  std::vector<int32_t> input_data_1(2 * 2 * 2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2, 2, 2}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> output_data_1(8 * 8 * 8 * 8, 0);
  TensorDesc output_desc_1(Shape({8, 8, 8, 8}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(reinterpret_cast<uint8_t *>(output_data_1.data()), output_data_1.size() * sizeof(int32_t));
  outputs.emplace_back(output_tensor_1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // check addr refresh
  uint64_t data1_addr = PtrToValue(input_data_1.data());
  std::cout << "======data1_addr:" << data1_addr << std::endl;
  std::cout << "======GetRtMemcpyRecords size:"
    << runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size() << std::endl;

  std::vector<uint64_t>  task_io_addr;
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    uint64_t *host_args_base = (uint64_t *)args.src_address;
    for (size_t i = 0U; i < (args.copy_len / sizeof(uint64_t)); i++) {
      uint64_t io_addr = host_args_base[i];
      if (io_addr != 0U) {
        task_io_addr.emplace_back(io_addr);
      }
    }
  }

  // check addr refresh
  EXPECT_EQ(task_io_addr.size(), 15);
  /**
   *   内存分配结果 -----offset ------ size ----------
   *   data0            16896         512
   *   data1            17408         512
   *   add1             1024          512    reuse netoutput
   *   add2             515           512    reuse netoutput
   *   add3             1024          512    reuse netoutput
   *   add4             0             512
   *   mul_1            512         16384
   */

  // get mul output addr
  uint64_t mul1_out_addr = task_io_addr[14];
  // get add1 output addr
  uint64_t add1_out_addr = task_io_addr[2];
  // get add2 output addr
  uint64_t add2_out_addr = task_io_addr[5];
  // get add3 output addr
  uint64_t add3_out_addr = task_io_addr[8];

  EXPECT_EQ(add1_out_addr, mul1_out_addr + 0x200);
  EXPECT_EQ(add2_out_addr, mul1_out_addr);
  EXPECT_EQ(add3_out_addr, mul1_out_addr + 0x200);
  runtime_stub.Clear();
}

/**
 * ----ANN_DATA reuse mem case -------------------------
 * -------------------------------------------------
 *         ann_data0  // set data reuse mem
 *             \
 *              \    Constant
 *               \   / |  |
 *                Add1 |  |
 *                 \   |  |
 *                  \  |  |
 *                   Add2 |
 *                    \   |
 *                     \  |
 *                      Add3
 *                       \
 *                        |
 *                       NetOutput  // set netoutput reuse mem
 * ------------------------------------------------
 *   data1、add2 reuse mem: offset = 1024 size = 1024
 *   add1、add3 reuse mem : offset = 0, size = 1024
 *   内存分配结果 -----offset ------ size ----------
 *   data1            1024         1024
 *   add1             0            1024   // reuse netoutput
 *   add2             1024         1024   // reuse data1
 *   add3             0            1024
 */
Graph BuildAnnDataReuseMemGraph() {
  std::vector<int64_t> shape{1, 4, 8, 8};
  auto data_0 = OP_CFG(ANN_DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  vector<int32_t> data_value(1 * 4 * 8 * 8, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");

  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto add_2 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_2");

  auto add_3 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_3");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)
          ->EDGE(0, 0)
          ->NODE(add_1)
          ->EDGE(0, 0)
          ->NODE(add_2)
          ->EDGE(0, 0)
          ->NODE(add_3));

    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    ADD_OUTPUT(add_3, 0);
  };

  return ToGeGraph(g1);
}

TEST_F(FmMemoryRefreshTest, ann_data_reuse_mem_01)  {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildRefDataReuseMemGraph();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0,1"); // "1":for error branch
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0,1"); // "1":for error branch

  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // check reuse mem reusult
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto data1_op_desc = compute_graph->FindNode("data_0")->GetOpDesc();
  const auto data1_output_size_list = ModelUtils::GetOutputSize(data1_op_desc);
  const auto data1_output_offset_list = data1_op_desc->GetOutputOffset();

  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add1_output_size_list = ModelUtils::GetOutputSize(add1_op_desc);
  const auto add1_output_offset_list = add1_op_desc->GetOutputOffset();

  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  const auto add2_output_size_list = ModelUtils::GetOutputSize(add2_op_desc);
  const auto add2_output_offset_list = add2_op_desc->GetOutputOffset();

  const auto add3_op_desc = compute_graph->FindNode("add_3")->GetOpDesc();
  const auto add3_output_size_list = ModelUtils::GetOutputSize(add3_op_desc);
  const auto add3_output_offset_list = add3_op_desc->GetOutputOffset();

  // check reuse mem
  EXPECT_EQ(data1_output_size_list[0], add2_output_size_list[0]);
  EXPECT_EQ(data1_output_offset_list[0], add2_output_offset_list[0]);

  EXPECT_EQ(add1_output_size_list[0], add3_output_size_list[0]);
  EXPECT_EQ(add1_output_offset_list[0], add3_output_offset_list[0]);

  runtime_stub.Clear();
}

/**
 * 用例描述：fm可刷新场景，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，Add节点输入使用data地址段，输出使用fm段
 *      Data    Data
 *        \      /
 *         Switch     Constant
 *          |   \    /   |
 *          |    Add    |
 *          |    |     /
 *          |    |   /
 *          |    Add
 *          |    |
 *          Merge
 *           |
 *        NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，Add不发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, fm_reuse_ok_when_single_execution) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildSwitchMergeGraphWithMultiAddNodes();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // 添加通信域
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  const auto add1_op_desc = compute_graph->FindNode("add_1")->GetOpDesc();
  const auto add2_op_desc = compute_graph->FindNode("add_2")->GetOpDesc();
  AttrUtils::SetListStr(add1_op_desc, "_hccl_group_id_list", {"group0", "group1"});
  AttrUtils::SetListStr(add2_op_desc, "_hccl_group_id_list", {"group1", "group2", "group3"});
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group0", (void*)1), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group1", (void*)2), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group2", (void*)3), GRAPH_SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group0");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group1");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group2");

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm可刷新场景，fm地址段不变，多次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，Add节点输入使用data地址段，输出使用fm段
 *      Data    Data
 *        \      /
 *         Switch     Constant
 *          |   \    /   |
 *          |    Add    |
 *          |    |     /
 *          |    |   /
 *          |    Add
 *          |    |
 *          Merge
 *           |
 *        NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.再执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.二次执行时，argstable的一致性和正确性均为成功，Add不发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, fm_reuse_ok_when_fm_unchanged_and_multiple_executions) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);

  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);


  auto graph = ShareGraph::BuildSwitchMergeGraphWithMultiAddNodes();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"add_1", "add_2"}));

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm可刷新场景，fm地址段发生变化，多次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，Add节点输入使用data地址段，输出使用fm段
 *      Data    Data
 *        \      /
 *         Switch     Constant
 *          |   \    /   |
 *          |    Add    |
 *          |    |     /
 *          |    |   /
 *          |    Add
 *          |    |
 *          Merge
 *           |
 *        NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.重新设置fm地址段
 * 4.再执行计算图，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.二次执行时，argstable的一致性和正确性判断均为成功，Add发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, fm_reuse_ok_when_fm_changed_and_multiple_executions) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildSwitchMergeGraphWithMultiAddNodes();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  // 修改fm地址
  std::vector<uint8_t> feature_mem1(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem1.data(), feature_size));

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem1.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：图的IO复用场景， 单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，node复用图的输入输出
 *-------------------------------------------------
 *           Data0   Data1
 *            \      /
 *             \    /
 *               Add1
 *                |    +---Constant_1
 *                |   /    /     |
 *                Add2    /      |
 *                |      /       |
 *                |\    /        |
 *                | Add3         |
 *                |  \           |
 *                |   \          |
 *                |----Add4     /
 *                      \      /
 *                       \    /
 *                       mul_1
 *                        |
 *                        NetOutput // set netoutput reuse
 *------------------------------------------------
 * Add1  input 0 复用data0，   input1 复用data1， output0复用netoutput
 * Add2  input 0 复用netoutput output0 复用netoutput
 * Add3  input 0 复用netoutput output0 复用netoutput
 * Add4  input 0 复用netoutput input1 复用netoutput，output复用fm
 * mul_1 input 0 复用fm        output0 复用netoutput
 * ----------------------------------------------
 * 测试步骤
 * 1.构造单个计算图1，设置IO地址段以及IO复用标志
 * 2.编译后执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功, Add1~Add4，mul_1发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, data_reuse_ok_when_single_execution) {
  MockMulInfoShape  mock_mul_info_shape;
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildIoReuseMemGraph();

  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  session.AddGraph(graph_id, graph, graph_options);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForIoReuseMemGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
}


/**
 * 用例描述：图的IO复用场景， 图的IO地址段不变，多次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，node复用图的输入输出
 *-------------------------------------------------
 *           Data0   Data1
 *            \      /
 *             \    /
 *               Add1
 *                |    +---Constant_1
 *                |   /    /     |
 *                Add2    /      |
 *                |      /       |
 *                |\    /        |
 *                | Add3         |
 *                |  \           |
 *                |   \          |
 *                |----Add4     /
 *                      \      /
 *                       \    /
 *                       mul_1
 *                        |
 *                        NetOutput // set netoutput reuse
 *------------------------------------------------
 * Add1  input 0 复用data0，   input1 复用data1， output0复用netoutput
 * Add2  input 0 复用netoutput output0 复用netoutput
 * Add3  input 0 复用netoutput output0 复用netoutput
 * Add4  input 0 复用netoutput input1 复用netoutput，output复用fm
 * mul_1 input 0 复用fm        output0 复用netoutput
 * ----------------------------------------------
 * 测试步骤
 * 1.构造单个计算图1，设置IO地址段以及IO复用标志
 * 2.编译后执行计算图1
 * 3.再执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.二次执行时，argstable的一致性和正确性均为成功，Add1~Add4，mul_1不发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, data_reuse_ok_when_data_unchanged_and_multiple_executions) {
  MockMulInfoShape  mock_mul_info_shape;
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildIoReuseMemGraph();

  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  session.AddGraph(graph_id, graph, graph_options);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForIoReuseMemGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"add_1", "add_2", "add_3", "add_4", "mul_1"}));

  runtime_stub.Clear();
}


/**
 * 用例描述：图的IO复用场景， 图的IO地址段发生变化，多次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，node复用图的输入输出
 *-------------------------------------------------
 *           Data0   Data1
 *            \      /
 *             \    /
 *               Add1
 *                |    +---Constant_1
 *                |   /    /     |
 *                Add2    /      |
 *                |      /       |
 *                |\    /        |
 *                | Add3         |
 *                |  \           |
 *                |   \          |
 *                |----Add4     /
 *                      \      /
 *                       \    /
 *                       mul_1
 *                        |
 *                        NetOutput // set netoutput reuse
 *------------------------------------------------
 * Add1  input 0 复用data0，   input1 复用data1， output0复用netoutput
 * Add2  input 0 复用netoutput output0 复用netoutput
 * Add3  input 0 复用netoutput output0 复用netoutput
 * Add4  input 0 复用netoutput input1 复用netoutput，output复用fm
 * mul_1 input 0 复用fm        output0 复用netoutput
 * ----------------------------------------------
 * 测试步骤
 * 1.构造单个计算图1，设置IO地址段以及IO复用标志
 * 2.编译后执行计算图1
 * 3.重新设置IO段地址
 * 4.再执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.二次执行时，argstable的一致性和正确性均为成功，Add1~Add4，mul_1发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, data_reuse_ok_when_data_changed_and_multiple_executions) {
  MockMulInfoShape  mock_mul_info_shape;
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildIoReuseMemGraph();

  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0");
  session.AddGraph(graph_id, graph, graph_options);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForIoReuseMemGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  ConstructInputOutputTensorForIoReuseMemGraph(inputs, outputs);
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
}

/**
 * 用例描述：fm可刷新场景，模型包含aicpu算子（NEG）, 单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，neg0和neg1节点输出使用fm段
 *
 *     NetOutput
 *          |
 *        Merge
 *       /   \
 *      /    NEG
 *     /      |
 *    NEG    shape
 *    F|     T|
 *     Switch1
 *   /       \
 *  Data     Data
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，neg0和neg1发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, fm_reuse_ok_when_aicpu_op_single_execution) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  MockForGenerateTask("AicpuLib", GenerateTaskForAicpu);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildSwitchMergeGraphWithNeg();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

void ConstructInputOutputTensorForAddAndDsaRandomNormalKnownGraph(std::vector<ge::Tensor> &inputs,
                                                                  std::vector<ge::Tensor> &outputs) {
  inputs.clear();
  outputs.clear();

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

  std::vector<int32_t> input_data_3(1 * 2 * 3 * 4, 0);
  TensorDesc desc_3(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_3{desc_3};
  input_tensor_3.SetData(reinterpret_cast<uint8_t *>(input_data_3.data()), input_data_3.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_3);

  std::vector<int32_t> input_data_4(1 * 2 * 3 * 4, 0);
  TensorDesc desc_4(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_4{desc_4};
  input_tensor_4.SetData(reinterpret_cast<uint8_t *>(input_data_4.data()), input_data_4.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_4);

  std::vector<int32_t> input_data_5(1 * 2 * 3 * 4, 0);
  TensorDesc desc_5(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_5{desc_5};
  input_tensor_5.SetData(reinterpret_cast<uint8_t *>(input_data_5.data()), input_data_5.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_5);

  std::vector<int32_t> input_data_6(1 * 2 * 3 * 4, 0);
  TensorDesc desc_6(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_6{desc_6};
  input_tensor_6.SetData(reinterpret_cast<uint8_t *>(input_data_6.data()), input_data_6.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_6);

  std::vector<int32_t> input_data_7(1 * 2 * 3 * 4, 0);
  TensorDesc desc_7(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_7{desc_7};
  input_tensor_7.SetData(reinterpret_cast<uint8_t *>(input_data_7.data()), input_data_7.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_7);

  std::vector<int32_t> input_data_8(1 * 2 * 3 * 4, 0);
  TensorDesc desc_8(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_8{desc_8};
  input_tensor_8.SetData(reinterpret_cast<uint8_t *>(input_data_8.data()), input_data_8.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_8);

  std::vector<uint8_t> output_data_1(96, 0xff);
  TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);
}

/**
 * 用例描述：fm可刷新场景，模型包含dsa算子（randomnormal）, 单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，neg0和neg1节点输出使用fm段
 *
 *      Data    Data Data    Data  Data  Data Data    Data
 *        \      /    \      /       \    /    \      /
 *         add1         add2         add3      add4
 *            \         |             |        /
 *             \        |            /       /
 *               \      |          /       /
 *                 \    |         /      /
 *                    random_normal
 *                       |
 *                     NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，add1~add4, random_normal发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, fm_reuse_ok_when_dsa_op_single_execution) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  MockForGenerateTask("DSAEngine", GenerateTaskForDsa);

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildAddAndDsaRandomNormalKnownGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForAddAndDsaRandomNormalKnownGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6, 7}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm不可刷新场景，模型包含dsa算子（randomnormal）, 不支持paremap
 *
 * 预置条件：
 * 1.构造计算图1，neg0和neg1节点输出使用fm段
 *
 *      Data    Data Data    Data  Data  Data Data    Data
 *        \      /    \      /       \    /    \      /
 *         add1         add2         add3      add4
 *            \         |             |        /
 *             \        |            /       /
 *               \      |          /       /
 *                 \    |         /      /
 *                    random_normal
 *                       |
 *                     NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，add1~add4, random_normal发生args table刷新
 */
TEST_F(FmMemoryRefreshTest, paremap_fm_not_support_refresh_when_dsa_op_single_execution) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  MockForGenerateTask("DSAEngine", GenerateTaskForDsa);

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto compute_graph = ShareGraph::BuildAddAndDsaRandomNormalKnownGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForAddAndDsaRandomNormalKnownGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // not support
  uint64_t fake_pa = 0xaaaaUL;
  EXPECT_EQ(FAILED, session.PaRemapped(reinterpret_cast<uint64_t>(feature_mem.data()), fake_pa, feature_size));
  // cannot recognize
  std::vector<uint8_t> tmp_mem(feature_size, 0);
  EXPECT_EQ(PARAM_INVALID, session.PaRemapped(reinterpret_cast<uint64_t>(tmp_mem.data()), fake_pa, feature_size));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6, 7}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm可刷新场景，rts算子直连data节点，执行一次，rts条件算子的io地址被修改为fix地址
 *
 * 预置条件
 * 1.构造计算图1，rts条件算子直连data节点
 *      Data    Data
 *        \      /
 *         Switch   Constant
 *          |   \    /
 *          |    Add
 *          |    /
 *          Merge
 *           |
 *        NetOutput
 *
 * 测试步骤
 * 1. 构造单个计算图1，fm设置为可刷新
 * 2. 编译后执行计算图1，判断rts条件算子的io是否被修改为fix地址
 *
 * 预期结果
 * 1. 一致性校验成功
*/
TEST_F(FmMemoryRefreshTest, rts_operater_directly_connect_to_data_node) {
  MockForGenerateTask("RTSLib", GenerateTaskForRts);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = ShareGraph::BuildSwitchMergeGraphWithMultiAddNodes();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/*
 * 用例描述: model io变更, 没有task args需要刷新的测试用例
 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. data 和netoutput仅有一条控制边，phony_concat连续内存的输出链接netoutput 不支持零拷贝
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 连续两次rungraph，传不同的model io, 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, no_task_args_associate_user_model_io) {
  GertRuntimeStub runtime_stub;
  MockIoMemReuse  mock_io_reuse_mem;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace("ge.exec.hostSchedulingMaxThreshold", "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::NetoutputNotSupportZeroCopy();

  // comile graph
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  std::vector<int32_t> input_data_0(8 * 8 * 8 * 8, 0);
  TensorDesc desc_0(Shape({8, 8, 8, 8}));
  ge::Tensor input_tensor_0{desc_0};
  input_tensor_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_0);

  std::vector<int32_t> output_data_1(1 * 4 * 2 * 124, 0);
  TensorDesc output_desc_1(Shape({1, 4, 2, 124}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(reinterpret_cast<uint8_t *>(output_data_1.data()), output_data_1.size() * sizeof(int32_t));
  outputs.emplace_back(output_tensor_1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  std::vector<ge::Tensor> inputs_x;
  std::vector<ge::Tensor> outputs_x;

  std::vector<int32_t> input_data_0_x(8 * 8 * 8 * 8, 0);
  TensorDesc desc_0_x(Shape({8, 8, 8, 8}));
  ge::Tensor input_tensor_0_x{desc_0_x};
  input_tensor_0_x.SetData(reinterpret_cast<uint8_t *>(input_data_0_x.data()), input_data_0_x.size() * sizeof(int32_t));
  inputs_x.emplace_back(input_tensor_0_x);

  std::vector<int32_t> output_data_1_x(1 * 4 * 2 * 124, 0);
  TensorDesc output_desc_1_x(Shape({1, 4, 2, 124}));
  ge::Tensor output_tensor_1_x{output_desc_1_x};
  output_tensor_1_x.SetData(reinterpret_cast<uint8_t *>(output_data_1_x.data()), output_data_1_x.size() * sizeof(int32_t));
  outputs_x.emplace_back(output_tensor_1_x);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs_x, outputs_x));
  runtime_stub.Clear();
}

/**
 * 用例描述：fm可刷新+老二包，dsa直连netoutoput，单次执行模型，一致性校验通过
 *
 * 预置条件：
 * 1.构造计算图1，neg0和neg1节点输出使用fm段
 *
 *      Data    Data Data    Data  Data  Data Data    Data
 *        \      /    \      /       \    /    \      /
 *         add1         add2         add3      add4
 *            \         |             |        /
 *             \        |            /       /
 *               \      |          /       /
 *                 \    |         /      /
 *                    random_normal
 *                       |
 *                     NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1，判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性成功
 */
TEST_F(FmMemoryRefreshTest, dsa_operater_directly_connect_to_netoutput) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  MockForGenerateTask("DSAEngine", GenerateTaskForDsa);

  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto compute_graph = ShareGraph::BuildAddAndDsaRandomNormalKnownGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForAddAndDsaRandomNormalKnownGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6, 7}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 *            Data0   Data01   // set data1 reuse mem
 *             \      /
 *              \    /
 *                Add1
 *                 |    +---Constant_1
 *                 |   /    / |
 *                 Add2    /  |
 *                 |      /   /
 *                 |\    /   |
 *                 | Add3    |
 *                 |  \     /
 *                 |   \   /
 *                 |    Add4
 *                 |     \
 *                 |      \
 *                 +------Add5
 *                         |
 *                         NetOutput
 */
// 测试步骤
// 1.构造单个计算图1，配置ge.exec.hostSchedulingMaxThreshold 为20
// 2.编译后执行计算图1，check summary 信息是否走动态调度
// 3.构造单个计算图2，配置ge.exec.hostSchedulingMaxThreshold 为-3,编译对应的计算图
// 预期结果
// 1.图1·编译成功，根据graph summary 判断图是是否走动态调度
// 2.图2·编译失败
TEST_F(FmMemoryRefreshTest, host_scheduling_summary_check) {
  GertRuntimeStub runtime_stub;
  MockIoMemReuse  mock_io_reuse_mem;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.exec.hostSchedulingMaxThreshold", "20");
  Session session(options);

  auto ge_env = GeRunningEnvFaker();
  ge_env.InstallDefault();
  auto graph = BuildIoReuseMemGraph3();

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "1");
  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  EXPECT_EQ(false, summary->IsStatic());

  // mem size check
  size_t size = 0;
  EXPECT_EQ(summary->GetConstMemorySize(size), FAILED);
  EXPECT_TRUE(size == 0);
  EXPECT_EQ(summary->GetFeatureMemorySize(size), FAILED);
  EXPECT_TRUE(size == 0);
  // check refreshable
  bool refreshable = false;
  EXPECT_EQ(summary->GetFeatureMemoryBaseRefreshable(refreshable), FAILED);
  EXPECT_TRUE(refreshable == false);
  // check event/stream num
  size_t num = 0;
  EXPECT_EQ(summary->GetStreamNum(num), SUCCESS);
  EXPECT_TRUE(num == 1);
  EXPECT_EQ(summary->GetEventNum(num), FAILED);
  EXPECT_TRUE(num == 0);
  //check outputshapes
  std::vector<ge::Shape> shapes;
  EXPECT_EQ(summary->GetOutputShapes(shapes), FAILED);
  EXPECT_EQ(shapes.size(), 0U);
  std::vector<ge::DataType> dtypes;
  EXPECT_EQ(summary->GetOutputDtypes(dtypes), FAILED);
  EXPECT_EQ(dtypes.size(), 0U);
  //check ioindex
  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), FAILED);
  EXPECT_EQ(io_indexes.size(), 0U);

  options.clear();
  options.emplace("ge.exec.hostSchedulingMaxThreshold", "-3");
  Session session1(options);

  auto graph2 = ShareGraph::BuildSwitchMergeGraphWithTwoOutputs();
  uint32_t graph_id2 = 2;
  EXPECT_NE(session1.AddGraph(graph_id2, graph2), SUCCESS);
}

void ConstructInputOutputTensorForInputDirectlyConnectedToOutputGraph(std::vector<ge::Tensor> &inputs,
                                                                      std::vector<ge::Tensor> &outputs) {
  inputs.clear();
  outputs.clear();

  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<uint8_t> output_data_1(96, 0xff);
  TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);
}

/*
 *    data1
 *      |
 *    netoutput
 *
 * 测试步骤
 * 1.构造单个计算图1，data1和netoutput直连, 输入输出地址不同
 * 2.编译后执行计算图1，check summary 信息中IoIndexes是否正确，模型是否可以执行成功
 * 预期结果
 * 1.IoIndexes包含输入输出的索引, 且符合预期
 * 2.模型执行成功
 */
TEST_F(FmMemoryRefreshTest, zero_copy_ok_when_input_directly_connected_to_output) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto compute_graph = ShareGraph::BuildInputDirectlyConnectedToOutputGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 1U);
  EXPECT_EQ(io_indexes[0].first, 0U);
  EXPECT_EQ(io_indexes[0].second, 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForInputDirectlyConnectedToOutputGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
}

void ConstructInputOutputTensorWithSameAddrForInputDirectlyConnectedToOutputGraph(std::vector<ge::Tensor> &inputs,
                                                                                  std::vector<ge::Tensor> &outputs) {
  inputs.clear();
  outputs.clear();

  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  outputs = inputs;
}

/*
 *    data1
 *      |
 *    netoutput
 *
 * 测试步骤
 * 1.构造单个计算图1，data1和netoutput直连, 输入输出地址相同
 * 2.编译后执行计算图1，check summary 信息中IoIndexes是否正确，模型是否可以执行成功
 * 预期结果
 * 1.IoIndexes包含输入输出的索引
 * 2.模型执行成功
 */
TEST_F(FmMemoryRefreshTest, zero_copy_ok_when_input_directly_connected_to_output_with_same_addr) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto compute_graph = ShareGraph::BuildInputDirectlyConnectedToOutputGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 1U);
  EXPECT_EQ(io_indexes[0].first, 0U);
  EXPECT_EQ(io_indexes[0].second, 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorWithSameAddrForInputDirectlyConnectedToOutputGraph(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
}

/**
 * -------------------------------------------------
 *            Data0   Data01
 *             \      /
 *              \    /
 *                Add1
 *                 |    +---Constant_1
 *                 |   /    / |  |
 *                 Add2    /  |  |
 *                 |      /   /  |
 *                 |\    /   |   |
 *                 | Add3    |   |
 *                 |  \     /    |
 *                 |   \   /     |
 *                 |    Add4     |
 *                 |     \       id_1
 *                 |      \      |
 *                 +------Add5   |
 *                         |     |
 *                         NetOutput
 */
Graph BuildTsMemGraph() {
  std::vector<int64_t> shape{8, 8, 8, 8};
  auto data_0 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_0");

  auto data_01 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .Build("data_01");

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

  auto add_5 = OP_CFG(ADD)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_5");

  std::vector<int64_t> memtype_list = {RT_MEMORY_TS};
  auto id_1 = OP_CFG("Identity").InCnt(1).OutCnt(1).Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list).Build("id_1");

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
          ->NODE(add_5)
          );

    CHAIN(NODE(data_01)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_2));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_3));
    CHAIN(NODE(const_1)->EDGE(0, 1)->NODE(add_4));
    CHAIN(NODE(add_2)->EDGE(0, 1)->NODE(add_5)->EDGE(0, 0)->NODE("output_1", NETOUTPUT));
    CHAIN(NODE(const_1)->EDGE(0, 0)->NODE(id_1)->EDGE(0, 1)->NODE("output_1", NETOUTPUT));
    //ADD_OUTPUT(add_5, 0);
    //ADD_OUTPUT(id_1, 0);
  };

  auto graph = ToGeGraph(g1);
  return graph;
}

/*
 * 用例描述: memcpy dst是ts内存的场景
 * 预置条件：
 * 1. 模型中不含var/dsa节点
 * 2. 模型输入的tensor_desc描述完备
 * 3. 模型中包含Constant/节点
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, memcpy_dst_is_ts_mem) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAyncTsMemory);

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace("ge.exec.hostSchedulingMaxThreshold", "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto graph = BuildTsMemGraph();

  std::map<AscendString, AscendString> graph_options;
  //graph_options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "1");
  // add graph
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  // comile graph
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int32_t> input_data_0(8 * 8 * 8 * 8, 0);
  TensorDesc desc_0(Shape({8, 8, 8, 8}));
  ge::Tensor input_tensor_0{desc_0};
  input_tensor_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_0);

  std::vector<int32_t> input_data_1(8 * 8 * 8 * 8, 0);
  TensorDesc desc_1(Shape({8, 8, 8, 8}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> output_data_1(8 * 32 * 8 * 32, 0);
  TensorDesc output_desc_1(Shape({8, 32, 8, 32}));
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(reinterpret_cast<uint8_t *>(output_data_1.data()), output_data_1.size() * sizeof(int32_t));
  outputs.emplace_back(output_tensor_1);

  std::vector<int32_t> output_data_2(8 * 32 * 8 * 32, 0);
  TensorDesc output_desc_2(Shape({8, 32, 8, 32}));
  ge::Tensor output_tensor_2{output_desc_2};
  output_tensor_2.SetData(reinterpret_cast<uint8_t *>(output_data_2.data()), output_data_2.size() * sizeof(int32_t));
  outputs.emplace_back(output_tensor_2);

  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
}


/*
                        g1

                               (1,1)
                      ┌───────────────────┐
                      │                   ∨
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌──────────┐
│ data_1 │ ───────> │ hcom_1 │ ───────> │ output_1 │
└────────┘          └────────┘          └──────────┘
                      ∧
                      │ (0,1)
                      │
                    ┌────────┐
                    │ data_2 │
                    └────────┘

*/
Graph BuildHcomGraph1() {
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
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  return graph;
}

/*
                        g1

                               (1,1)
                      ┌───────────────────┐
                      │                   ∨
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌──────────┐
│ data_1 │ ───────> │ hcom_1 │ ───────> │ output_1 │
└────────┘          └────────┘          └──────────┘
                      ∧
                      │ (0,1)
                      │
                    ┌────────┐
                    │ data_2 │
                    └────────┘

*/
Graph BuildHcomAllToAllGraph() {
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data_1");
  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW,  DT_BOOL, {1}).InCnt(1).OutCnt(1).Build("data_2");
  auto hcom_1 = OP_CFG(HCOMALLTOALL)
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
 * set fixed fm_memory_static
 */
TEST_F(FmMemoryRefreshTest, set_fixed_fm_memory_static_0001) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph1();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  std::vector<uint8_t> refreshable_feature_mem(refreshable_feature_size, 0);
  std::cout << "======fix_feature_size:" << fix_feature_size << ", refreshable_feature_size:"
      << refreshable_feature_size << ", feature_size:" << feature_size <<", fixed_feature_mem:"<< std::hex
      << (uintptr_t)fixed_feature_mem.data() << ", refreshable_feature_mem:" << std::hex
      << (uintptr_t)refreshable_feature_mem.data() << std::endl;
  // set fixed fm memory base
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  // update fm memory base
  EXPECT_EQ(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(),
      refreshable_feature_size));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // check addr refresh
  std::cout << "======GetRtMemcpyRecords size:"
    << runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size() << std::endl;

  std::vector<uint64_t>  task_io_addr;
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    std::cout << "=== host args ===" << std::endl;
    uint64_t *host_args_base = (uint64_t *)args.src_address;
    if (host_args_base == nullptr) {
      continue;
    }
    for (size_t i = 0U; i < (args.copy_len / sizeof(uint64_t)); i++) {
      uint64_t io_addr = host_args_base[i];
      if (io_addr != 0U) {
        task_io_addr.emplace_back(io_addr);
        std::cout << "io_addr: " << io_addr << std::endl;
      }
    }
  }
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
                                                 ┌─────────────────────────────────────────┐
                                                 │                                         │
                                                 │                  (1,3)                  │
                      ┌──────────────────────────┘         ┌───────────────────┐           │ (1,1)
                      │                                    │                   ∨           │
┌────────┐  (0,0)   ┌────────┐  (0,1)   ┌─────┐  (0,0)   ┌────────┐  (0,2)   ┌──────────┐  │
│ data_1 │ ───────> │ hcom_1 │ ───────> │     │ ───────> │ hcom_2 │ ───────> │          │ <┘
└────────┘          └────────┘          │     │          └────────┘          │          │
  │                   ∧                 │     │  (0,1)     ∧                 │          │
  │                   │ (0,1)           │ add │ ───────────┘                 │ output_1 │
  │                   │                 │     │                              │          │
  │                 ┌────────┐          │     │                     (0,0)    │          │
  │                 │ data_2 │          │     │ ───────────────────────────> │          │
  │                 └────────┘          └─────┘                              └──────────┘
  │        (0,0)                          ∧
  └───────────────────────────────────────┘

*/
Graph BuildHcomGraphWithP2pAndHbmFixedMemory() {
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
      .Build("hcom_1");
  std::vector<int64_t> p2p_memtype_list = {RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR};
  auto hcom_2 = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(2)
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, p2p_memtype_list)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, p2p_memtype_list)
      .Build("hcom_2");
  auto add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(g1) {
                  CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(add));
                  CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(hcom_1)->EDGE(0, 1)->NODE(add)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
                  CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(hcom_1)->EDGE(1, 1)->NODE("output_1", "NetOutput"));
                  CHAIN(NODE(add)->EDGE(0, 0)->NODE(hcom_2)->NODE("output_1", "NetOutput"));
                  CHAIN(NODE(add)->EDGE(0, 1)->NODE(hcom_2)->NODE("output_1", "NetOutput"));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  {
    auto node = compute_graph->FindNode("hcom_1");
    auto op_desc = node->GetOpDesc();
    op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
    op_desc->SetWorkspace({0, 0});
    op_desc->SetWorkspaceBytes({32, 32});
    ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  }
  {
    auto node = compute_graph->FindNode("hcom_2");
    auto op_desc = node->GetOpDesc();
    op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
    op_desc->SetWorkspace({0, 0});
    op_desc->SetWorkspaceBytes({32, 32});
    ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  }
  return graph;
}

/*

                                       root

┌────────┐  (0,0)   ┌───────────┐  (0,0)   ┌───────────┐  (0,0)   ┌────────────────┐
│ data_1 │ ───────> │   rank    │ ───────> │ known_op1 │ ───────> │ root_netoutput │
└────────┘          └───────────┘          └───────────┘          └────────────────┘
                                                                    ∧
                                                                    │
                                                                    │
┌────────┐  (0,0)   ┌───────────┐  (0,1)                            │
│ data_2 │ ───────> │ known_op2 │ ──────────────────────────────────┘
└────────┘          └───────────┘

                                                sub_1

                                  (0,1)
                         ┌───────────────────┐
                         │                   ∨
┌───────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────┐  (0,0)   ┌────────────┐  (0,0)   ┌─────────────────┐
│ data_sub1 │ ───────> │ hcom_1 │ ───────> │ add │ ───────> │ hcom_1_p2p │ ───────> │ sub_1_netoutput │
└───────────┘          └────────┘          └─────┘          └────────────┘          └─────────────────┘

                                                sub_2

                                  (0,1)
                         ┌───────────────────┐
                         │                   ∨
┌───────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────┐  (0,0)   ┌────────────┐  (0,0)   ┌─────────────────┐
│ data_sub2 │ ───────> │ hcom_2 │ ───────> │ add │ ───────> │ hcom_2_p2p │ ───────> │ sub_2_netoutput │
└───────────┘          └────────┘          └─────┘          └────────────┘          └─────────────────┘
*/
Graph BuildUnknownHcomGraphWithP2pAndHbmFixedMemory() {
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM};
  std::vector<int64_t> shape = {2, 2};  // NCHW
  // sub1
  auto data_sub1 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_sub1");
  data_sub1->SetOutputOffset({32});

  auto hcom_1 = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_MODIFY_INPUT, true)
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
      .Build("hcom_1");

  hcom_1->SetOpKernelLibName("ops_kernel_info_hccl");
  hcom_1->SetWorkspace({0});
  hcom_1->SetWorkspaceBytes({512});
  hcom_1->SetInputOffset({0});
  hcom_1->SetOutputOffset({32});
  ge::AttrUtils::SetBool(hcom_1, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  std::vector<int64_t> memtype_list_p2p = {RT_MEMORY_P2P_DDR};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add1");
  auto hcom_1_p2p = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_MODIFY_INPUT, true)
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list_p2p)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list_p2p)
      .Build("hcom_1_p2p");

  hcom_1_p2p->SetOpKernelLibName("ops_kernel_info_hccl");
  hcom_1_p2p->SetWorkspace({0});
  hcom_1_p2p->SetWorkspaceBytes({512});
  hcom_1_p2p->SetInputOffset({0});
  hcom_1_p2p->SetOutputOffset({32});
  ge::AttrUtils::SetBool(hcom_1_p2p, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  auto sub_1_netoutput = OP_CFG(ge::NETOUTPUT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Build("sub_1_netoutput");
  sub_1_netoutput->SetInputOffset({0});

  // sub2
  auto data_sub2 = OP_CFG("Data")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Attr(ATTR_NAME_INDEX, 0)
      .Build("data_sub2");
  data_sub2->SetOutputOffset({64});

  auto hcom_2 = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_MODIFY_INPUT, true)
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
      .Build("hcom_2");

  hcom_2->SetOpKernelLibName("ops_kernel_info_hccl");
  hcom_2->SetWorkspace({0});
  hcom_2->SetWorkspaceBytes({2048});
  hcom_2->SetInputOffset({16});
  hcom_2->SetOutputOffset({64});
  ge::AttrUtils::SetBool(hcom_2, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  auto hcom_2_p2p = OP_CFG(HCOMALLREDUCE)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_MODIFY_INPUT, true)
      .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list_p2p)
      .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list_p2p)
      .Build("hcom_2_p2p");

  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add2");
  hcom_2_p2p->SetOpKernelLibName("ops_kernel_info_hccl");
  hcom_2_p2p->SetWorkspace({0});
  hcom_2_p2p->SetWorkspaceBytes({2048});
  hcom_2_p2p->SetInputOffset({16});
  hcom_2_p2p->SetOutputOffset({64});
  ge::AttrUtils::SetBool(hcom_2_p2p, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  auto sub_2_netoutput = OP_CFG(ge::NETOUTPUT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
      .Build("sub_2_netoutput");
  sub_2_netoutput->SetInputOffset({16});

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

  auto known_op1 =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("known_op1");

  auto known_op2 =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("known_op2");

  auto root_netoutput =
      OP_CFG(ge::NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("root_netoutput");
  root_netoutput->SetSrcName({"known_op1", "known_op2"});
  root_netoutput->SetSrcIndex({0, 1});

  DEF_GRAPH(sub_1) {
                     CHAIN(NODE(data_sub1)->NODE(hcom_1)->EDGE(0, 0)->NODE(add1)->NODE(hcom_1_p2p)->NODE(sub_1_netoutput));
                     CHAIN(NODE(hcom_1)->EDGE(0, 1)->NODE(add1));
                   };

  DEF_GRAPH(sub_2) {
                     CHAIN(NODE(data_sub2)->NODE(hcom_2)->EDGE(0, 0)->NODE(add2)->NODE(hcom_2_p2p)->NODE(sub_2_netoutput));
                     CHAIN(NODE(hcom_2)->EDGE(0, 1)->NODE(add2));
                   };

  DEF_GRAPH(root) {
                    CHAIN(NODE(data_1)->NODE(rank)->NODE(known_op1)->NODE(root_netoutput));
                    CHAIN(NODE(data_2)->NODE(known_op2)->NODE(root_netoutput));
                  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto sub_graph1 = ToGeGraph(sub_1);
  auto compute_graph1 = ge::GraphUtilsEx::GetComputeGraph(sub_graph1);
  compute_graph1->SetGraphUnknownFlag(false);
  auto net_output1 = compute_graph1->FindNode("sub_1_netoutput");
  net_output1->GetOpDesc()->SetSrcName({"hcom_1"});
  net_output1->GetOpDesc()->SetSrcIndex({0});

  auto sub_graph2 = ToGeGraph(sub_2);
  auto compute_graph2 = ge::GraphUtilsEx::GetComputeGraph(sub_graph2);
  compute_graph2->SetGraphUnknownFlag(false);
  auto net_output2 = compute_graph2->FindNode("sub_2_netoutput");
  net_output2->GetOpDesc()->SetSrcName({"hcom_2"});
  net_output2->GetOpDesc()->SetSrcIndex({0});

  auto known_node1 = root_graph->FindNode("known_op1");
  auto known_node2 = root_graph->FindNode("known_op2");

  SetSubGraph(root_graph, known_node1, compute_graph1);
  SetSubGraph(root_graph, known_node2, compute_graph2);

  AddCompileResult(known_node1, false);
  AddCompileResult(known_node2, false);
  return graph;
}

/*
 * 用例描述: 纯静态图，用户没有设置fixed地址，GE默认申请fixed内存，hbm和p2p

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 调用新增api接口
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p,hbm fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, UserNotSetFixedFeatureMemory_GeMallocHbmAndP2pFixedMemoryByDefault_Success) {

  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_HBM.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * 用例描述:配置静态图复用，多张纯静态图，图中fix优先内存大小不一样，先加载小的，再加载大的，
 * 预期使用session级fix优先内存池，内存大小从小扩展到大

 * 测试步骤：
 * 1. 构造2张静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 编译
 * 3. 设置静态图复用，加载
 *4.卸载
 *
 * 预期结果：
 * 1. 使用session级fix优先内存池，内存从小扩展到大
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中session级fix优先内存池从小扩展到大，卸载时释放了内存
 */
TEST_F(FmMemoryRefreshTest, MultiStaticGraphWithFixedFeatureMemory_staticMemoryPolicy2_Success) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.exec.staticMemoryPolicy", "2");
  Session session(options);
  auto graph1 = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph1);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);

  auto graph2 = BuildHcomGraphWithP2pAndHbmFixedMemory();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph2);
  {
    auto node = compute_graph->FindNode("hcom_1");
    auto op_desc = node->GetOpDesc();
    op_desc->SetWorkspaceBytes({10240, 10240});
  }
  {
    auto node = compute_graph->FindNode("hcom_2");
    auto op_desc = node->GetOpDesc();
    op_desc->SetWorkspaceBytes({10240, 10240});
  }
  uint32_t graph_2_id = 2;
  session.AddGraph(graph_2_id, graph2);
  size_t graph_2_hbm_fixed_feature_size = 0U;
  size_t graph_2_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_2_id, graph_2_hbm_fixed_feature_size, graph_2_p2p_fixed_feature_size), SUCCESS);

  // graph1 run
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_1_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_P2P_DDR, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_HBM, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  // graph2 run
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_2_id, nullptr, inputs, outputs));
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_P2P_DDR, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_HBM, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_2 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  // 加载第二张图，就不会创建物理内存池对象了
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 2");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 17");
  EXPECT_FALSE(find_log > 0);

  // graph1 unload
  EXPECT_EQ(session.RemoveGraph(graph_1_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("free fixed_feature_memory by session allocator, rts memory type: RT_MEMORY_HBM, addr");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("free fixed_feature_memory by session allocator, rts memory type: RT_MEMORY_P2P_DDR, addr");
  EXPECT_TRUE(find_log > 0);

  // graph2 unload
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(session.RemoveGraph(graph_2_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("free fixed_feature_memory by session allocator, rts memory type: RT_MEMORY_HBM, addr");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("free fixed_feature_memory by session allocator, rts memory type: RT_MEMORY_P2P_DDR, addr");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * 用例描述: 配置动静态图复用，多张动态图， 图中fix优先内存大小不一样，先加载大的，再加载小的，预期使用session级fix优先内存池，内存大小从小扩展到大

 * 测试步骤：
 * 1. 构造动态shape静态子图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 编译
 * 3. 设置动静态图复用，加载
 * 4.卸载
 *
 * 预期结果：
 * 1. 使用session级fix优先内存池
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中session级fix优先内存池从小扩展到大，卸载时释放了内存
 */
TEST_F(FmMemoryRefreshTest, MultiDynamicGraphWithFixedFeatureMemory_staticMemoryPolicy4_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace("ge.exec.staticMemoryPolicy", "4");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);
  SetModelVarSize(session, graph_1_id);

  auto graph2 = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph2);
  for (auto &node : compute_graph->GetAllNodesPtr()) {
    if (node->GetName() == "hcom_1") {
      node->GetOpDescBarePtr()->SetWorkspaceBytes({10240});
    }
    if (node->GetName() == "hcom_1_p2p") {
      node->GetOpDescBarePtr()->SetWorkspaceBytes({20480});
    }
  }
  uint32_t graph_2_id = 2;
  session.AddGraph(graph_2_id, graph2);
  size_t graph_2_hbm_fixed_feature_size = 0U;
  size_t graph_2_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_2_id, graph_2_hbm_fixed_feature_size, graph_2_p2p_fixed_feature_size), SUCCESS);
  SetModelVarSize(session, graph_2_id);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructIOTensor(inputs, outputs);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);

  // run graph2
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_2_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_2 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*DavinciModelCreate_.* fixed_feature_memory hbm_addr 0x.*, hbm_size .*, p2p_addr 0x");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*AllocFixedFeatureMemory");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("success to add ctrl edge from davinci_model_finalizer to free node: FreeFixedFeatureMemory_");
  EXPECT_TRUE(find_log > 0);

  // run graph1
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_1_id, nullptr, inputs, outputs));
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*DavinciModelCreate_.* fixed_feature_memory hbm_addr 0x.*, hbm_size .*, p2p_addr 0x");
  EXPECT_TRUE(find_log > 0);
  // 加载第二张图，就不会创建物理内存池对象了
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 2");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 17");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*AllocFixedFeatureMemory");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("success to add ctrl edge from davinci_model_finalizer to free node: FreeFixedFeatureMemory_");
  EXPECT_TRUE(find_log > 0);
  // graph1 unload
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(session.RemoveGraph(graph_1_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*FreeFixedFeatureMemory");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("fixed_base_expandable_memory.* free, block");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("ActiveMemoryAllocator::FreeMemory used_count =");
  EXPECT_TRUE(find_log > 0);

  // graph2 unload
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(session.RemoveGraph(graph_2_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*FreeFixedFeatureMemory");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("fixed_base_expandable_memory.* free, block");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("ActiveMemoryAllocator::FreeMemory used_count =");
  EXPECT_TRUE(find_log > 0);

  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 配置动静态图复用，动态图+静态图， 图中fix优先内存大小不一样，先加载大的，再加载小的，预期使用session级fix优先内存池，内存大小从小扩展到大

 * 测试步骤：
 * 1. 构造动态shape静态子图 静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 编译
 * 3. 设置静静态图复用，加载
 * 4.卸载
 *
 * 预期结果：
 * 1. 使用session级fix优先内存池
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中session级fix优先内存池从小扩展到大，卸载时释放了内存
 */
TEST_F(FmMemoryRefreshTest, DynamicAndStaticGraphWithFixedFeatureMemory_staticMemoryPolicy4_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace("ge.exec.staticMemoryPolicy", "4");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);
  SetModelVarSize(session, graph_1_id);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructIOTensor(inputs, outputs);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);

  // run graph1
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_1_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*DavinciModelCreate_.* fixed_feature_memory hbm_addr 0x.*, hbm_size .*, p2p_addr 0x");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*AllocFixedFeatureMemory");
  EXPECT_TRUE(find_log > 0);

  // run graph2
  auto graph2 = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_2_id = 2;
  session.AddGraph(graph_2_id, graph2);
  size_t graph_2_hbm_fixed_feature_size = 0U;
  size_t graph_2_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_2_id, graph_2_hbm_fixed_feature_size, graph_2_p2p_fixed_feature_size), SUCCESS);

  runtime_stub.GetSlogStub().Clear();
  std::vector<ge::Tensor> graph_2_inputs;
  std::vector<ge::Tensor> graph_2_outputs;
  ConstructInputOutputTensor(graph_2_inputs, graph_2_outputs, 4);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_2_id, nullptr, graph_2_inputs, graph_2_outputs));
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_P2P_DDR, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_HBM, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_2 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  // 加载第二张图，就不会创建物理内存池对象了
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 2");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 17");
  EXPECT_FALSE(find_log > 0);

  // graph1 unload
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(session.RemoveGraph(graph_1_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*FreeFixedFeatureMemory");
  EXPECT_TRUE(find_log > 0);

  // graph2 unload
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(session.RemoveGraph(graph_2_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("free fixed_feature_memory by session allocator, rts memory type: RT_MEMORY_HBM, addr");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("free fixed_feature_memory by session allocator, rts memory type: RT_MEMORY_P2P_DDR, addr");
  EXPECT_TRUE(find_log > 0);

  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 多session用例，配置动静态图复用，动态图+静态图， 预期使用session级fix优先内存池，每个session一个

 * 测试步骤：
 * 1. 构造动态shape静态子图 静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 编译
 * 3. 设置静静态图复用，加载
 * 4.卸载
 *
 * 预期结果：
 * 1. 使用session级fix优先内存池，每个session一个
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中有2个session级fix优先内存池
 */
TEST_F(FmMemoryRefreshTest, MultiSessionDynamicAndStaticGraphWithFixedFeatureMemory_staticMemoryPolicy4_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace("ge.exec.staticMemoryPolicy", "4");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);
  SetModelVarSize(session, graph_1_id);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructIOTensor(inputs, outputs);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);

  // run graph1
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_1_id, nullptr, inputs, outputs));
  std::string reg = "create session allocator, typeid.* session_id: " + std::to_string(session.GetSessionId());
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex(reg.c_str());
  EXPECT_TRUE(find_log > 0);

  // run graph2
  Session session2(options);
  auto graph2 = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_2_id = 2;
  session2.AddGraph(graph_2_id, graph2);
  size_t graph_2_hbm_fixed_feature_size = 0U;
  size_t graph_2_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session2, graph_2_id, graph_2_hbm_fixed_feature_size, graph_2_p2p_fixed_feature_size), SUCCESS);

  runtime_stub.GetSlogStub().Clear();
  std::vector<ge::Tensor> graph_2_inputs;
  std::vector<ge::Tensor> graph_2_outputs;
  ConstructInputOutputTensor(graph_2_inputs, graph_2_outputs, 4);
  EXPECT_EQ(SUCCESS, session2.RunGraphWithStreamAsync(graph_2_id, nullptr, graph_2_inputs, graph_2_outputs));
  reg = "create session allocator, typeid.* session_id: " + std::to_string(session2.GetSessionId());
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex(reg.c_str());
  EXPECT_TRUE(find_log > 0);

  // graph1 unload
  EXPECT_EQ(session.RemoveGraph(graph_1_id), SUCCESS);
  // graph2 unload
  EXPECT_EQ(session2.RemoveGraph(graph_2_id), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 开启动静态复用后，SetGraphFixedFeatureMemoryBase入参addr和size为0报错

 * 测试步骤：
 * 1. 构造动态shape静态子图 静态图，图上有要求fixed地址的算子
 * 2. 编译
 * 3. 调用SetGraphFixedFeatureMemoryBase
 *
 * 预期结果：
 * 1. SetGraphFixedFeatureMemoryBase接口报错
 */
TEST_F(FmMemoryRefreshTest, staticMemoryPolicy4_SetGraphFixedFeatureMemoryBaseAddrSizeNull_Failed) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace("ge.exec.staticMemoryPolicy", "4");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);
  EXPECT_NE(session.SetGraphFixedFeatureMemoryBase(graph_1_id, nullptr, 0), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 配置动静态图复用，多张动态图和静态图，其中一张图用户自己设置了fix优先内存,预期这张图不使用session级fix优先内存池

 * 测试步骤：
 * 1. 构造动态shape静态子图和静态图，图上有要求fixed地址的算子
 * 2. 编译
 * 3. 设置fix优先内存地址，设置静态图复用，加载
 *4.卸载
 *
 * 预期结果：
 * 1. 对于用户设置了fix优先内存地址的图，GE 不再使用session级fix优先内存池
 * 2. 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, staticMemoryPolicy4_UserSetFixedFeatureMemory) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace("ge.exec.staticMemoryPolicy", "4");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);
  SetModelVarSize(session, graph_1_id);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructIOTensor(inputs, outputs);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);

  // run graph1
  runtime_stub.GetSlogStub().Clear();
  std::vector<uint8_t> user_alloc_hbm_fixed_feature_mem(graph_1_hbm_fixed_feature_size, 0);
  session.SetGraphFixedFeatureMemoryBase(graph_1_id, &user_alloc_hbm_fixed_feature_mem[0], graph_1_hbm_fixed_feature_size);
  std::vector<uint8_t> user_alloc_p2p_fixed_feature_mem(graph_1_p2p_fixed_feature_size, 0);
  session.SetGraphFixedFeatureMemoryBaseWithType(graph_1_id, MemoryType::MEMORY_TYPE_P2P, &user_alloc_p2p_fixed_feature_mem[0], graph_1_p2p_fixed_feature_size);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_1_id, nullptr, inputs, outputs));
  std::string reg = "create session allocator, typeid.* session_id: " + std::to_string(session.GetSessionId());
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex(reg.c_str());
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:no need to malloc fixed_feature_memory base. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:no need to malloc fixed_feature_memory base. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  // run graph2
  Session session2(options);
  auto graph2 = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_2_id = 2;
  session2.AddGraph(graph_2_id, graph2);
  size_t graph_2_hbm_fixed_feature_size = 0U;
  size_t graph_2_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session2, graph_2_id, graph_2_hbm_fixed_feature_size, graph_2_p2p_fixed_feature_size), SUCCESS);

  runtime_stub.GetSlogStub().Clear();
  std::vector<ge::Tensor> graph_2_inputs;
  std::vector<ge::Tensor> graph_2_outputs;
  ConstructInputOutputTensor(graph_2_inputs, graph_2_outputs, 4);
  EXPECT_EQ(SUCCESS, session2.RunGraphWithStreamAsync(graph_2_id, nullptr, graph_2_inputs, graph_2_outputs));
  reg = "create session allocator, typeid.* session_id: " + std::to_string(session2.GetSessionId());
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex(reg.c_str());
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_P2P_DDR, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("get fixed_feature_memory success, type: RT_MEMORY_HBM, addr: ");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_2 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  // 加载第二张图，就不会创建物理内存池对象了
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 2");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("CreateAllocator:Create PhysicalMemoryAllocator success device id:0, memory_type: 17");
  EXPECT_FALSE(find_log > 0);

  // graph1 unload
  EXPECT_EQ(session.RemoveGraph(graph_1_id), SUCCESS);
  // graph2 unload
  EXPECT_EQ(session2.RemoveGraph(graph_2_id), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 配置动静态图复用，设置外置外置allocator，多张动态图和静态图，预期使用外置allocator申请fix优先内存

 * 测试步骤：
 * 1. 构造动态shape静态子图和静态图，图上有要求fixed地址的算子
 * 2. 编译
 * 3. 注册外置allocator，设置静态图复用，加载
 * 4. 卸载
 *
 * 预期结果：
 * 1. 使用外置allocator申请fix优先内存，GE 不再使用session级fix优先内存池
 * 2. 模型执行成功
 */
TEST_F(FmMemoryRefreshTest, staticMemoryPolicy4_ExternalAllocator) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace("ge.exec.staticMemoryPolicy", "4");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  rtStream_t stream = (rtStream_t)0x1;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_1_id = 1;
  session.AddGraph(graph_1_id, graph);
  size_t graph_1_hbm_fixed_feature_size = 0U;
  size_t graph_1_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_1_id, graph_1_hbm_fixed_feature_size, graph_1_p2p_fixed_feature_size), SUCCESS);
  SetModelVarSize(session, graph_1_id);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructIOTensor(inputs, outputs);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);

  // run graph1
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_1_id, stream, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*GetUserAllocatorOrFixedBaseAllocator_.*get external allocator");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*GetUserAllocatorOrFixedBaseAllocator_.*ger or create fixed base expandable allocator");
  EXPECT_TRUE(find_log > 0);

  // run graph2
  auto graph2 = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_2_id = 2;
  session.AddGraph(graph_2_id, graph2);
  size_t graph_2_hbm_fixed_feature_size = 0U;
  size_t graph_2_p2p_fixed_feature_size = 0U;
  EXPECT_EQ(CompileAndGetFixedSize(session, graph_2_id, graph_2_hbm_fixed_feature_size, graph_2_p2p_fixed_feature_size), SUCCESS);

  runtime_stub.GetSlogStub().Clear();
  std::vector<ge::Tensor> graph_2_inputs;
  std::vector<ge::Tensor> graph_2_outputs;
  ConstructInputOutputTensor(graph_2_inputs, graph_2_outputs, 4);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_2_id, stream, graph_2_inputs, graph_2_outputs));
  auto reg = "create session allocator, typeid.* session_id: " + std::to_string(session.GetSessionId());
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex(reg.c_str());
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc .* bytes success using external allocator");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_2 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  // graph1 unload
  EXPECT_EQ(session.RemoveGraph(graph_1_id), SUCCESS);
  // graph2 unload
  EXPECT_EQ(session.RemoveGraph(graph_2_id), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 纯静态图，用户只设置了hbm fixed地址，GE默认申请p2p fixed内存

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 调用新增api接口，设置hbm fixed内存
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p  fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, UserOnlySetHbmFixedFeatureMemory_GeMallocP2pFixedMemoryByDefault_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<uint8_t> user_alloc_hbm_fixed_feature_mem(hbm_fixed_feature_size, 0);
  session.SetGraphFixedFeatureMemoryBase(graph_id, &user_alloc_hbm_fixed_feature_mem[0], hbm_fixed_feature_size);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemoryIfNeed:no need to malloc fixed_feature_memory base, type:RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * 用例描述: 动态shape静态子图，用户没有设置fixed地址，GE默认申请fixed内存，hbm和p2p

 * 测试步骤：
 * 1. 构造动态shape静态子图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 调用新增api接口
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p,hbm fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, UnknownGraphUserNotSetFixedFeatureMemory_GeMallocHbmAndP2pFixedMemoryByDefault_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 7));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 1536));


  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int64_t> input_data_1(2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2}));
  desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int64_t> input_data_2(2 * 2, 0);
  TensorDesc desc_2(Shape({2, 2}));
  desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_2);

  std::vector<uint8_t> output_data_1(128, 0xff);
  TensorDesc output_desc_1(Shape({2, 2}));
  output_desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);

  std::vector<uint8_t> output_data_2(128, 0xff);
  TensorDesc output_desc_2(Shape({2, 2}));
  output_desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_2{output_desc_2};
  output_tensor_2.SetData(output_data_2.data(), output_data_2.size());
  outputs.emplace_back(output_tensor_2);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 0, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or inner allocator. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or inner allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*DavinciModelCreate_.* fixed_feature_memory hbm_addr 0x.*, p2p_addr 0x");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 动态shape静态子图，用户只设置hbm fixed地址，GE默认申请p2p fixed内存

 * 测试步骤：
 * 1. 构造动态shape静态子图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 设置hbm fixed内存
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2pfixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, UnknownGraphUserOnlySetHbmFixedFeatureMemory_GeMallocP2pFixedMemoryByDefault_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 7));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 1536));


  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<uint8_t> user_alloc_hbm_fixed_feature_mem(hbm_fixed_feature_size, 0);
  session.SetGraphFixedFeatureMemoryBase(graph_id, &user_alloc_hbm_fixed_feature_mem[0], hbm_fixed_feature_size);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int64_t> input_data_1(2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2}));
  desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int64_t> input_data_2(2 * 2, 0);
  TensorDesc desc_2(Shape({2, 2}));
  desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_2);

  std::vector<uint8_t> output_data_1(128, 0xff);
  TensorDesc output_desc_1(Shape({2, 2}));
  output_desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);

  std::vector<uint8_t> output_data_2(128, 0xff);
  TensorDesc output_desc_2(Shape({2, 2}));
  output_desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_2{output_desc_2};
  output_tensor_2.SetData(output_data_2.data(), output_data_2.size());
  outputs.emplace_back(output_tensor_2);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or inner allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:no need to malloc fixed_feature_memory base. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*DavinciModelCreate_.* fixed_feature_memory hbm_addr 0x.*, p2p_addr 0x");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 动态shape静态子图，用户设置hbm 的fixed地址为nullptr，GE不再默认申请hbm fixed内存，只默认申请p2p fixed内存

 * 测试步骤：
 * 1. 构造动态shape静态子图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 调用api接口将hbm fixed内存设置为nullptr，size为0
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, UnknownGraphUserSetHbmFixedFeatureMemoryNullptr_GeMallocP2pFixedMemoryByDefault_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildUnknownHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 7));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 1536));


  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int64_t> input_data_1(2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2}));
  desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int64_t> input_data_2(2 * 2, 0);
  TensorDesc desc_2(Shape({2, 2}));
  desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_2);

  std::vector<uint8_t> output_data_1(128, 0xff);
  TensorDesc output_desc_1(Shape({2, 2}));
  output_desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);

  std::vector<uint8_t> output_data_2(128, 0xff);
  TensorDesc output_desc_2(Shape({2, 2}));
  output_desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_2{output_desc_2};
  output_tensor_2.SetData(output_data_2.data(), output_data_2.size());
  outputs.emplace_back(output_tensor_2);

  // 用户设置hbm 的fixed地址为nullptr，GE不再默认申请hbm fixed内存
  session.SetGraphFixedFeatureMemoryBase(graph_id, nullptr, 0U);

  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:no need to malloc fixed_feature_memory base. type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("MallocFixedFeatureMemOnInitRootIfNeed:need to malloc fixed_feature_memory base by user allocator or inner allocator. type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("KernelTrace.*DavinciModelCreate_.* fixed_feature_memory hbm_addr .*, hbm_size 0, p2p_addr 0x");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

/*
 * 用例描述: 纯静态图，用户设置了hbm fixed地址为nullptr， size为0，GE默认申请p2p fixed内存

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 调用新增api接口，设置hbm fixed内存为nullptr
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p  fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, UserOnlySetHbmFixedFeatureMemoryNullptr_GeMallocP2pFixedMemoryByDefault_Success) {

  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  session.SetGraphFixedFeatureMemoryBase(graph_id, nullptr, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("IsNeedMallocFixedFeatureMemByType:user set fixed_feature_memory base nullptr, return false, memory type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_HBM.*");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * 用例描述: GE默认申请Fixed feature内存与UpdateGraphFeatureMemoryBase接口的组合场景，
纯静态图，存在fixed feature memory，用户没有设置fixed地址，RunGraph后调用UpdateGraphFeatureMemoryBase接口，然后再次RunGraph

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 图执行
 * 4. UpdateGraphFeatureMemoryBase
 * 5. 图执行
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 调用UpdateGraphFeatureMemoryBase成功
 * 3. 通过info日志校验执行过程中申请了hbm p2p fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, GeMallocFixedMemoryByDefault_UpdateGraphFeatureMemoryBase_AfterLoad_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("IsNeedMallocFixedFeatureMemByType:fixed_feature_memory base is not set by user, return true, memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_HBM.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  std::vector<uint8_t> feature_base(feature_size, 0);
  EXPECT_EQ(session.UpdateGraphFeatureMemoryBase(graph_id, &feature_base[0], feature_size), SUCCESS);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * 用例描述: GE默认申请Fixed feature内存与UpdateGraphFeatureMemoryBase接口的组合场景，
 * 纯静态图，存在fixed feature memory，用户没有设置fixed地址，调用UpdateGraphFeatureMemoryBase接口，然后再RunGraph

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. UpdateGraphFeatureMemoryBase
 * 4. 图执行
 * 5. 图执行
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 调用UpdateGraphFeatureMemoryBase成功
 * 3. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p fixed内存, 没有申请hbm fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, GeMallocFixedMemoryByDefault_UpdateGraphFeatureMemoryBase_BeforeLoad_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();

  std::vector<uint8_t> feature_base(feature_size, 0);
  EXPECT_EQ(session.UpdateGraphFeatureMemoryBase(graph_id, &feature_base[0], feature_size), SUCCESS);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("IsNeedMallocFixedFeatureMemByType:user set fixed_feature_memory base nullptr, return false, memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_HBM.*");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}
/*
 * 用例描述: GE默认申请Fixed feature内存与UpdateGraphRefreshableFeatureMemoryBase接口的组合场景，
纯静态图，存在fixed feature memory，用户没有设置fixed地址，RunGraph后调用UpdateGraphRefreshableFeatureMemoryBase接口，然后再次RunGraph

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 图执行
 * 4. UpdateGraphRefreshableFeatureMemoryBase
 * 5. 图执行
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 调用UpdateGraphRefreshableFeatureMemoryBase成功
 * 3. 通过info日志校验执行过程中申请了hbm p2p fixed内存，执行完成后释放了内存
 */
TEST_F(FmMemoryRefreshTest, GeMallocFixedMemoryByDefault_UpdateGraphRefreshableFeatureMemoryBase_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("IsNeedMallocFixedFeatureMemByType:user set fixed_feature_memory base nullptr, return false, memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_HBM.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  std::vector<uint8_t> feature_base(refreshable_feature_size, 0);
  EXPECT_EQ(session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, &feature_base[0], refreshable_feature_size), SUCCESS);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_HBM");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * 用例描述: SetGraphFixedFeatureMemoryBase 当fixed内存类型为p2p时，无法设置为nullptr和size 0

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 图执行
 *
 * 预期结果：
 * 1. SetGraphFixedFeatureMemoryBase not support
 */
TEST_F(FmMemoryRefreshTest, SetGraphFixedFeatureMemoryBaseWithType_P2pNullptr_NotSupport) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  ASSERT_NE(session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_P2P, nullptr, 0), GE_GRAPH_UNSUPPORTED);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}
/*
 * 用例描述: GE默认申请Fixed feature内存与外置allocator接口的组合场景，
纯静态图，存在fixed feature memory，用户没有设置fixed地址，用户设置了external allocator

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 用户设置了external allocator
 * 4. 图执行
 * 5. 图执行
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p  fixed内存，执行完成后释放了内存。校验使用外置allocator申请了hbm fixed内存，并正常释放
 */
TEST_F(FmMemoryRefreshTest, GeMallocFixedMemoryByDefault_ExternalAllocator_Success) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  rtStream_t stream = (rtStream_t)0x1;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("IsNeedMallocFixedFeatureMemByType:user set fixed_feature_memory base nullptr, return false, memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("bytes success using external allocator");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by external allocator success");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(FmMemoryRefreshTest, GeMallocFixedMemoryByDefault_ExternalAllocator_Success_by_load_executegraph) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  rtStream_t stream = (rtStream_t)0x1;

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  options.emplace("ge.exec.frozenInputIndexes", "1000");
  ret = session.LoadGraph(graph_id, options, stream);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  ge::diagnoseSwitch::DisableDumper();

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(4);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};

  for (size_t i = 0; i < 4; i++) {
    gert_outputs[i] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                                // placement
                          ge::DT_INT32,                              // data type
                          nullptr};
  }
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));

  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  runtime_stub.Clear();
}

TEST_F(FmMemoryRefreshTest, GeMallocFixedMemoryForExecute_ExternalAllocator_Success) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.exec.frozenInputIndexes", "1000");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));

  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1, 0);
  gert_inputs[1] = {{{1}, {1}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  rtStream_t stream = (rtStream_t)0x1;
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  gert_outputs.resize(4);
  gert_outputs[0] = {{{0}, {0}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            nullptr};
  gert_outputs[1] = {{{0}, {0}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            nullptr};
  gert_outputs[2] = {{{0}, {0}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            nullptr};
  gert_outputs[3] = {{{0}, {0}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            nullptr};
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("IsNeedMallocFixedFeatureMemByType:user set fixed_feature_memory base nullptr, return false, memory type: RT_MEMORY_HBM");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("bytes success using external allocator");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by external allocator success");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  runtime_stub.Clear();
}

/*
 * 用例描述: 用户在加载前调用了UpdateGraphFeatureMemoryBase接口，Ge不默认申请hbm类型的fixed内存

 * 测试步骤：
 * 1. 构造静态图，图上有要求fixed地址的算子，同时包含hbm和p2p两种fixed内存类型
 * 2. 设option参数并编译
 * 3. 用户加载前设置了UpdateGraphFeatureMemoryBase
 * 4. 图执行
 *
 * 预期结果：
 * 1. 通过summary拿到的P2p,hbm fixed feature memory size非0
 * 2. 模型执行成功
 * 3. 通过info日志校验执行过程中申请了p2p  fixed内存，执行完成后释放了内存。校验没有申请hbm fixed内存
 */
TEST_F(FmMemoryRefreshTest, GeNotMallocFixedMemory_UserUpdateFeatureMemory) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  std::vector<uint8_t> feature_base(feature_size, 0);
  const auto all_feature_memory =summary->GetAllFeatureMemoryTypeSize();
  ASSERT_EQ(all_feature_memory.size(), 2U);
  size_t hbm_fixed_feature_size = 0U;
  size_t p2p_fixed_feature_size = 0U;
  for (const auto &feature_mem : all_feature_memory) {
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
      hbm_fixed_feature_size = feature_mem->GetSize();
    }
    if (feature_mem->IsFixed() && (feature_mem->GetType() == MemoryType::MEMORY_TYPE_P2P)) {
      p2p_fixed_feature_size = feature_mem->GetSize();
    }
  }
  ASSERT_NE(p2p_fixed_feature_size, 0U);
  ASSERT_NE(hbm_fixed_feature_size, 0U);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 4);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  rtStream_t stream = (rtStream_t)0x1;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  std::vector<gert::Tensor> gert_inputs;
      std::vector<gert::Tensor> gert_outputs;
      gert_inputs.resize(2);
      gert_outputs.resize(4);
      std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
      gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) input_data_1.data()};

      std::vector<int32_t> input_data_2(1, 0);
      gert_inputs[1] = {{{1}, {1}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) input_data_2.data()};
      std::vector<uint8_t> output_data_1(96, 0xFF);
      gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) output_data_1.data()};
      std::vector<uint8_t> output_data_2(96, 0xFF);
      gert_outputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) output_data_2.data()};
      std::vector<uint8_t> output_data_3(96, 0xFF);
      gert_outputs[2] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) output_data_3.data()};
      std::vector<uint8_t> output_data_4(96, 0xFF);
      gert_outputs[3] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) output_data_4.data()};
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, &feature_base[0], feature_size));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_P2P_DDR.*");
  EXPECT_TRUE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("malloc fixed_feature_memory success, type: RT_MEMORY_HBM.*");
  EXPECT_FALSE(find_log > 0);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("graph_1 use p2p_fixed_mem_base");
  EXPECT_TRUE(find_log > 0);

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("FreeFixedFeatureMemoryIfNeed:free fixed_feature_memory by inner allocator success, rts memory type: RT_MEMORY_P2P_DDR");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  runtime_stub.Clear();
}

/*
 * 用例描述: 用户调用了UpdateGraphFeatureMemoryBase接口，再调用UpdateGraphRefreshableFeatureMemoryBase要报错

 * 测试步骤：
 * 1. 构造静态图
 * 2. 设option参数并编译
 * 3. 用户设置了UpdateGraphFeatureMemoryBase
 * 4. 用户设置了UpdateGraphRefreshableFeatureMemoryBase
 *
 * 预期结果：
 * 1. UpdateGraphFeatureMemoryBase 成功
 * 2. UpdateGraphRefreshableFeatureMemoryBase失败
 */

TEST_F(FmMemoryRefreshTest, UpdateGraphRefreshableFeatureMemoryBase_Failed_WhenUpdateGraphFeatureMemoryBase) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);

  size_t feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  std::vector<uint8_t> feature_base(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, &feature_base[0], feature_size));

  size_t refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  std::vector<uint8_t> ref_feature_base(refreshable_feature_size, 0);
  EXPECT_EQ(PARAM_INVALID, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, &ref_feature_base[0], refreshable_feature_size));

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}
/*
 * 用例描述: 用户调用了UpdateGraphRefreshableFeatureMemoryBase接口，再调用UpdateGraphFeatureMemoryBase要报错

 * 测试步骤：
 * 1. 构造静态图
 * 2. 设option参数并编译
 * 3. 用户设置了UpdateGraphRefreshableFeatureMemoryBase
 * 4. 用户设置了UpdateGraphFeatureMemoryBase
 *
 * 预期结果：
 * 1. UpdateGraphRefreshableFeatureMemoryBase 成功
 * 2. UpdateGraphFeatureMemoryBase失败
 */

TEST_F(FmMemoryRefreshTest, UpdateGraphFeatureMemoryBase_Failed_WhenUpdateGraphRefreshableFeatureMemoryBase) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraphWithP2pAndHbmFixedMemory();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);

  size_t refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  std::vector<uint8_t> ref_feature_base(refreshable_feature_size, 0);
  EXPECT_EQ(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, &ref_feature_base[0], refreshable_feature_size));

  size_t feature_size;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  std::vector<uint8_t> feature_base(feature_size, 0);
  EXPECT_EQ(PARAM_INVALID, session.UpdateGraphFeatureMemoryBase(graph_id, &feature_base[0], feature_size));

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}
/*
                                 g1

           (0,0)
  ┌───────────────────────────────────────┐
  │                                       │
  │                                       │     (1,1)
  │                   ┌───────────────────┼────────────────┐
  │                   │                   ∨               ∨
┌────────┐  (0,0)   ┌────────┐  (0,1)   ┌─────┐  (0,0)   ┌──────────┐
│ data_1 │ ───────> │ hcom_1 │ ───────> │ add │ ───────> │ output_1 │
└────────┘          └────────┘          └─────┘          └──────────┘
                      ∧
                      │ (0,1)
                      │
                    ┌────────┐
                    │ data_2 │
                    └────────┘
*/
Graph BuildHcomGraph2() {
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
                    .Build("hcom_1");
  auto add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(hcom_1)->EDGE(0, 1)->NODE(add)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(hcom_1)->EDGE(1, 1)->NODE("output_1", "NetOutput"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
  op_desc->SetWorkspace({0, 0});
  op_desc->SetWorkspaceBytes({32, 32});
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  return graph;
}

/*
 * set fixed fm_memory_static
 */
TEST_F(FmMemoryRefreshTest, set_fixed_fm_memory_static_0002) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph2();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  std::vector<uint8_t> refreshable_feature_mem(refreshable_feature_size, 0);
  std::cout << "======fix_feature_size:" << fix_feature_size << ", refreshable_feature_size:"
      << refreshable_feature_size << ", feature_size:" << feature_size <<", fixed_feature_mem:"<< std::hex
      << (uintptr_t)fixed_feature_mem.data() << ", refreshable_feature_mem:" << std::hex
      << (uintptr_t)refreshable_feature_mem.data() << std::endl;
  // set fixed fm memory base
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  // update fm memory base
  EXPECT_EQ(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(),
      refreshable_feature_size));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
 * set fixed fm_memory_static
 */
TEST_F(FmMemoryRefreshTest, set_fixed_fm_memory_static_0003) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  // memory extend mode
  options.emplace("ge.exec.staticMemoryPolicy", "2");
  Session session(options);

  auto graph = BuildHcomGraph2();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  std::vector<uint8_t> refreshable_feature_mem(refreshable_feature_size, 0);
  std::cout << "======fix_feature_size:" << fix_feature_size << ", refreshable_feature_size:"
      << refreshable_feature_size << ", feature_size:" << feature_size <<", fixed_feature_mem:"<< std::hex
      << (uintptr_t)fixed_feature_mem.data() << ", refreshable_feature_mem:" << std::hex
      << (uintptr_t)refreshable_feature_mem.data() << std::endl;
  // set fixed fm memory base
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(FmMemoryRefreshTest, set_fixed_fm_memory_static_0004) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph1();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  std::vector<uint8_t> refreshable_feature_mem(refreshable_feature_size, 0);
  std::cout << "======fix_feature_size:" << fix_feature_size << ", refreshable_feature_size:"
      << refreshable_feature_size << ", feature_size:" << feature_size <<", fixed_feature_mem:"<< std::hex
      << (uintptr_t)fixed_feature_mem.data() << ", refreshable_feature_mem:" << std::hex
      << (uintptr_t)refreshable_feature_mem.data() << std::endl;
  // set fixed fm memory base
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  // update fm memory base
  EXPECT_EQ(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(),
      refreshable_feature_size));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // check addr refresh
  std::cout << "======GetRtMemcpyRecords size:"
    << runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size() << std::endl;

  std::vector<uint64_t>  task_io_addr;
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    std::cout << "=== host args ===" << std::endl;
    uint64_t *host_args_base = (uint64_t *)args.src_address;
    if (host_args_base == nullptr) {
      continue;
    }
    for (size_t i = 0U; i < (args.copy_len / sizeof(uint64_t)); i++) {
      uint64_t io_addr = host_args_base[i];
      if (io_addr != 0U) {
        task_io_addr.emplace_back(io_addr);
        std::cout << "io_addr: " << io_addr << std::endl;
      }
    }
  }
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

Graph BuildHcomGraph4() {
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
                    .Build("hcom_1");
  auto add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1)
                 .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                 .Build("add");
  auto add_2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_INT32, shape).InCnt(2).OutCnt(1)
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Build("add_2");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(add_2)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(hcom_1)->EDGE(0, 1)->NODE(add)->EDGE(0, 1)->NODE(add_2)->EDGE(0, 0)->NODE("output_1", "NetOutput"));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(hcom_1)->EDGE(1, 1)->NODE("output_1", "NetOutput"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("hcom_1");
  auto op_desc = node->GetOpDesc();
  op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
  op_desc->SetWorkspace({0, 0});
  op_desc->SetWorkspaceBytes({32, 32});
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  return graph;
}

/*
 * set fixed fm_memory_static
 */
TEST_F(FmMemoryRefreshTest, set_fixed_fm_memory_static_0005) {
  GertRuntimeStub runtime_stub;

  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  map<std::string, std::string> graph_options;
  graph_options.emplace(OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1");
  auto graph = BuildHcomGraph4();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph, graph_options);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_EQ(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == true);

  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  std::vector<uint8_t> refreshable_feature_mem(refreshable_feature_size, 0);
  std::cout << "======fix_feature_size:" << fix_feature_size << ", refreshable_feature_size:"
      << refreshable_feature_size << ", feature_size:" << feature_size <<", fixed_feature_mem:"<< std::hex
      << (uintptr_t)fixed_feature_mem.data() << ", refreshable_feature_mem:" << std::hex
      << (uintptr_t)refreshable_feature_mem.data() << std::endl;
  // set fixed fm memory base
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  // update fm memory base
  EXPECT_EQ(SUCCESS, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, refreshable_feature_mem.data(),
      refreshable_feature_size));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

/*
                                       root

┌────────┐  (0,0)   ┌───────────┐  (0,0)   ┌───────────┐  (0,0)   ┌────────────────┐
│ data_1 │ ───────> │   rank    │ ───────> │ known_op1 │ ───────> │ root_netoutput │
└────────┘          └───────────┘          └───────────┘          └────────────────┘
                                                                    ∧
                                                                    │
                                                                    │
┌────────┐  (0,0)   ┌───────────┐  (0,1)                            │
│ data_2 │ ───────> │ known_op2 │ ──────────────────────────────────┘
└────────┘          └───────────┘

                            sub_1

┌───────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────────────────┐
│ data_sub1 │ ───────> │ hcom_1 │ ───────> │ sub_1_netoutput │
└───────────┘          └────────┘          └─────────────────┘

                            sub_2

┌───────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────────────────┐
│ data_sub2 │ ───────> │ hcom_2 │ ───────> │ sub_2_netoutput │
└───────────┘          └────────┘          └─────────────────┘
*/
Graph BuildHcomGraph3() {
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM};
  std::vector<int64_t> shape = {2, 2};  // NCHW
  // sub1
  auto data_sub1 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_sub1");
  data_sub1->SetOutputOffset({32});

  auto hcom_1 = OP_CFG(HCOMALLREDUCE)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_MODIFY_INPUT, true)
                    .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                    .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                    .Build("hcom_1");

  hcom_1->SetOpKernelLibName("ops_kernel_info_hccl");
  hcom_1->SetWorkspace({0});
  hcom_1->SetWorkspaceBytes({512});
  hcom_1->SetInputOffset({0});
  hcom_1->SetOutputOffset({32});
  ge::AttrUtils::SetBool(hcom_1, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  auto sub_1_netoutput = OP_CFG(ge::NETOUTPUT)
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                             .InCnt(1)
                             .OutCnt(1)
                             .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                             .Build("sub_1_netoutput");
  sub_1_netoutput->SetInputOffset({0});

  // sub2
  auto data_sub2 = OP_CFG("Data")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .Build("data_sub2");
   data_sub2->SetOutputOffset({64});

  auto hcom_2 = OP_CFG(HCOMALLREDUCE)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_MODIFY_INPUT, true)
                    .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                    .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                    .Build("hcom_2");

  hcom_2->SetOpKernelLibName("ops_kernel_info_hccl");
  hcom_2->SetWorkspace({0});
  hcom_2->SetWorkspaceBytes({2048});
  hcom_2->SetInputOffset({16});
  hcom_2->SetOutputOffset({64});
  ge::AttrUtils::SetBool(hcom_2, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  auto sub_2_netoutput = OP_CFG(ge::NETOUTPUT)
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                             .InCnt(1)
                             .OutCnt(1)
                             .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Build("sub_2_netoutput");
  sub_2_netoutput->SetInputOffset({16});

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

  auto known_op1 =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("known_op1");

  auto known_op2 =
      OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("known_op2");

  auto root_netoutput =
      OP_CFG(ge::NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("root_netoutput");
  root_netoutput->SetSrcName({"known_op1", "known_op2"});
  root_netoutput->SetSrcIndex({0, 1});

  DEF_GRAPH(sub_1) {
    CHAIN(NODE(data_sub1)->NODE(hcom_1)->NODE(sub_1_netoutput));
  };

  DEF_GRAPH(sub_2) {
    CHAIN(NODE(data_sub2)->NODE(hcom_2)->NODE(sub_2_netoutput));
  };

  DEF_GRAPH(root) {
    CHAIN(NODE(data_1)->NODE(rank)->NODE(known_op1)->NODE(root_netoutput));
    CHAIN(NODE(data_2)->NODE(known_op2)->NODE(root_netoutput));
  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto sub_graph1 = ToGeGraph(sub_1);
  auto compute_graph1 = ge::GraphUtilsEx::GetComputeGraph(sub_graph1);
  compute_graph1->SetGraphUnknownFlag(false);
  auto net_output1 = compute_graph1->FindNode("sub_1_netoutput");
  net_output1->GetOpDesc()->SetSrcName({"hcom_1"});
  net_output1->GetOpDesc()->SetSrcIndex({0});

  auto sub_graph2 = ToGeGraph(sub_2);
  auto compute_graph2 = ge::GraphUtilsEx::GetComputeGraph(sub_graph2);
  compute_graph2->SetGraphUnknownFlag(false);
  auto net_output2 = compute_graph2->FindNode("sub_2_netoutput");
  net_output2->GetOpDesc()->SetSrcName({"hcom_2"});
  net_output2->GetOpDesc()->SetSrcIndex({0});

  auto known_node1 = root_graph->FindNode("known_op1");
  auto known_node2 = root_graph->FindNode("known_op2");

  SetSubGraph(root_graph, known_node1, compute_graph1);
  SetSubGraph(root_graph, known_node2, compute_graph2);

  AddCompileResult(known_node1, false);
  AddCompileResult(known_node2, false);
  return graph;
}

/*
 * set fixed fm_memory_dynamic
 */
TEST_F(FmMemoryRefreshTest, set_fixed_fm_memory_dynamic_0001) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph3();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 5));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 1536));


  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size, refreshable_feature_size;
  EXPECT_NE(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_NE(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  EXPECT_NE(SUCCESS, summary->GetRefreshableFeatureMemorySize(refreshable_feature_size));
  bool is_refreshable = false;
  EXPECT_NE(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));

  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  std::cout << "======fix_feature_size:" << fix_feature_size << ", refreshable_feature_size:"
      << refreshable_feature_size << ", feature_size:" << feature_size <<", fixed_feature_mem:"<< std::hex
      << (uintptr_t)fixed_feature_mem.data() << std::endl;
  // set fixed fm memory base
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<int64_t> input_data_1(2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2}));
  desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int64_t> input_data_2(2 * 2, 0);
  TensorDesc desc_2(Shape({2, 2}));
  desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_2);

  std::vector<uint8_t> output_data_1(128, 0xff);
  TensorDesc output_desc_1(Shape({2, 2}));
  output_desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_1{output_desc_1};
  output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
  outputs.emplace_back(output_tensor_1);

  std::vector<uint8_t> output_data_2(128, 0xff);
  TensorDesc output_desc_2(Shape({2, 2}));
  output_desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor output_tensor_2{output_desc_2};
  output_tensor_2.SetData(output_data_2.data(), output_data_2.size());
  outputs.emplace_back(output_tensor_2);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  OpsKernelBuilderRegistry::GetInstance().Unregister("aicpu_ascend_kernel");
  runtime_stub.Clear();
}

Graph BuildDataNetoutputGraph() {
  auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {64}).Build("_data_0");
  auto output_0 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1)
                      .Build("_output_0");

  DEF_GRAPH(simple_d2o) {
    CHAIN(NODE(data_0)->NODE(output_0));
  };

  auto graph = ToGeGraph(simple_d2o);
  return graph;
}


Status GenerateTaskForMemcpyAddrAsync(const Node &node, RunContext &run_context,
                                      std::vector<domi::TaskDef> &tasks, std::string (*builder)(size_t)) {
  if ((node.GetType() != MEMCPYADDRASYNC)) {
    return SUCCESS;
  }
  const auto base = reinterpret_cast<uintptr_t>(run_context.dataMemBase);
  const auto &op_desc = node.GetOpDesc();
  domi::TaskDef &task_def = tasks.emplace_back();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ADDR_ASYNC));

  auto memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_src(base + op_desc->GetInputOffset()[0]);
  memcpy_async->set_dst(base + op_desc->GetOutputOffset()[0]);

  int64_t isize, osize;
  EXPECT_EQ(GRAPH_SUCCESS, TensorUtils::GetTensorSizeInBytes(op_desc->GetInputDesc(0), isize));
  EXPECT_EQ(GRAPH_SUCCESS, TensorUtils::GetTensorSizeInBytes(op_desc->GetOutputDesc(0), osize));

  memcpy_async->set_dst_max(osize);
  memcpy_async->set_count(isize);
  memcpy_async->set_op_index(op_desc->GetId());
  memcpy_async->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  memcpy_async->set_args_format(builder(isize));
  return SUCCESS;
}

Status GenerateTaskForMemcpyAddrAsync_NonDavid(const Node &node, RunContext &run_context,
                                               std::vector<domi::TaskDef> &tasks) {
  return GenerateTaskForMemcpyAddrAsync(node, run_context, tasks, [](size_t size) {
    ArgsFormatDesc desc;
    desc.Append(AddrType::PLACEHOLDER);
    desc.Append(AddrType::PLACEHOLDER);
    desc.Append(AddrType::INPUT_INSTANCE, 0);
    desc.Append(AddrType::OUTPUT_INSTANCE, 0);
    return desc.ToString();
  });
}

Status GenerateTaskForMemcpyAddrAsync_David(const Node &node, RunContext &run_context,
                                            std::vector<domi::TaskDef> &tasks) {
  return GenerateTaskForMemcpyAddrAsync(node, run_context, tasks, [](size_t size) {
    ArgsFormatDesc desc;
    desc.Append(AddrType::PLACEHOLDER);
    desc.Append(AddrType::PLACEHOLDER);
    desc.Append(AddrType::PLACEHOLDER);
    desc.Append(AddrType::PLACEHOLDER);
    desc.Append(AddrType::INPUT_INSTANCE, 0);
    desc.Append(AddrType::OUTPUT_INSTANCE, 0);
    desc.AppendCustomValue(size, ArgsFormatWidth::BIT32);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT32);
    desc.Append(AddrType::PLACEHOLDER);
    return desc.ToString();
  });
}

void EXPECT_CheckMemcpyAddrAsync(bool update_io) {
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options = {
    { OPTION_BUILD_GRAPH_MODE, "offline" },
  };
  Session session(options);

  EXPECT_EQ(SUCCESS, session.AddGraph(0, BuildDataNetoutputGraph()));
  EXPECT_EQ(SUCCESS, session.CompileGraph(0));
  EXPECT_EQ(SUCCESS, session.LoadGraph(0, options, nullptr));

  std::vector<gert::Tensor> inputs = gert::FakeTensors({64}, 1).Steal();
  std::vector<gert::Tensor> outputs = gert::FakeTensors({64}, 1).Steal();
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(0, nullptr, inputs, outputs));
  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, 0, session.GetSessionId(), runtime_stub);
  };

  auto old_level = runtime_stub.GetSlogStub().GetLevel();
  runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  runtime_stub.GetSlogStub().SetLevel(old_level);

  if (!update_io) {
    return;
  }
  runtime_stub.Clear();
  args_checker->ClearAddrSegments();

  std::vector<gert::Tensor> inputs2 = gert::FakeTensors({64}, 1).Steal();
  std::vector<gert::Tensor> outputs2 = gert::FakeTensors({64}, 1).Steal();
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(0, nullptr, inputs2, outputs2));

  runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0}, inputs2));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs2));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  runtime_stub.GetSlogStub().SetLevel(old_level);
}

TEST_F(FmMemoryRefreshTest, memcpy_async_addr_non_david) {
  auto env_capa = EnvGuard("SET_CAPA_VALUE", "mock_fail");
  auto env_soc = EnvGuard("MOCK_SOC_VERSION", "Ascend910B");
  MockForGenerateTask("RTSLib", GenerateTaskForMemcpyAddrAsync_NonDavid);

  EXPECT_CheckMemcpyAddrAsync(false);
}

TEST_F(FmMemoryRefreshTest, memcpy_async_addr_david) {
  auto env_capa = EnvGuard("SET_CAPA_VALUE", "mock_fail");
  auto env_soc = EnvGuard("MOCK_SOC_VERSION", "Ascend910D");
  MockForGenerateTask("RTSLib", GenerateTaskForMemcpyAddrAsync_David);

  EXPECT_CheckMemcpyAddrAsync(false);
}

TEST_F(FmMemoryRefreshTest, memcpy_async_addr_non_david_update_io) {
  auto env_capa = EnvGuard("SET_CAPA_VALUE", "mock_fail");
  auto env_soc = EnvGuard("MOCK_SOC_VERSION", "Ascend910B");
  MockForGenerateTask("RTSLib", GenerateTaskForMemcpyAddrAsync_NonDavid);

  EXPECT_CheckMemcpyAddrAsync(true);
}

TEST_F(FmMemoryRefreshTest, memcpy_async_addr_david_update_io) {
  auto env_capa = EnvGuard("SET_CAPA_VALUE", "mock_fail");
  auto env_soc = EnvGuard("MOCK_SOC_VERSION", "Ascend910D");
  MockForGenerateTask("RTSLib", GenerateTaskForMemcpyAddrAsync_David);

  EXPECT_CheckMemcpyAddrAsync(true);
}


TEST_F(FmMemoryRefreshTest, hcom_all_to_all_test) {
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllToAll);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomAllToAllGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };
  IMPL_OP(HcomAllToAll).InferShape(infer_shape_func)
      .InferDataType(infer_data_type_func)
      .InferShapeRange(infer_shape_range_func);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("HcomAllToAll");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  auto default_space_registry = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t fix_feature_size = 0U;
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(default_space_registry);
}

}  // namespace ge