/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "base/err_mgr.h"

#include "macro_utils/dt_public_scope.h"
#include "framework/executor/ge_executor.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/helper/model_helper.h"
#include "graph/execute/model_executor.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "ge/ge_api.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/command_handle.h"
#include "common/profiling/profiling_properties.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/ge_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "runtime/subscriber/global_dumper.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/operator_factory_impl.h"
#include "ge_running_env/fake_op.h"
#include "register/node_converter_registry.h"
#include "register/kernel_registry.h"
#include "graph/utils/tensor_adapter.h"
#include "api/aclgrph/option_utils.h"
#include "ge/ge_api_v2.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace std;
using namespace testing;

namespace ge {
bool EnableSliceSchedule() { // 桩函数
  return ((ge::GetAutofuseFlagValue(kAutoFuseEnableOption) == "true") &&
          (ge::GetAutofuseFlagValue(kSliceScheduleOption) == "true"));;
}

class OnlineInferTest : public testing::Test {
 protected:
  void SetUp() {
    GeExecutor::Initialize({});
  }
  void TearDown() {
    GeExecutor::FinalizeEx();
  }
};

static void BuildSampleGraph(ComputeGraphPtr &graph, ComputeGraphPtr &case1, ComputeGraphPtr &case2, uint32_t &mem_offset) {
  // Combine sink and no_sink in one graph.
  DEF_GRAPH(g0) {
    const auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
    const auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
    const auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
    const auto output = OP_CFG(NETOUTPUT).Attr(ATTR_ALL_GEARS_INFO, ";2;4;8").Attr(ATTR_GETNEXT_SINK_DYNMAIC, false);
    CHAIN(NODE("ascend_mbatch_shape_data", data_2)->NODE("ascend_mbatch_shape_mapindex", "MapIndex")->
          NODE("ascend_mbatch_shape_case", CASE)->NODE(NODE_NAME_NET_OUTPUT, output));
    CHAIN(NODE("ascend_mbatch_shape_const", CONSTANT)->EDGE(0, 1)->NODE("ascend_mbatch_shape_mapindex"));
    CHAIN(NODE("_arg_ascend_mbatch_batch_0", data_0)->EDGE(0, 1)->NODE("ascend_mbatch_shape_case"));
    CHAIN(NODE("_arg_ascend_mbatch_batch_1", data_1)->EDGE(0, 2)->NODE("ascend_mbatch_shape_case"));
    CHAIN(NODE("ascend_mbatch_shape_data")->NODE("ascend_mbatch_get_dynamic_dims_node", GETDYNAMICDIMS)->NODE(NODE_NAME_NET_OUTPUT));
  };
  graph = ToComputeGraph(g0);
  SetUnknownOpKernel(graph, mem_offset, true);
  const auto mbatch_case_node = graph->FindNode("ascend_mbatch_shape_case");
  EXPECT_NE(mbatch_case_node, nullptr);
  const auto case_desc = mbatch_case_node->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetBool(case_desc, ATTR_INSERT_BY_MBATCH, true));
  EXPECT_TRUE(AttrUtils::SetInt(case_desc, ATTR_NAME_BATCH_NUM, 2));

  {
    const auto mbatch_batch_node = graph->FindNode("_arg_ascend_mbatch_batch_0");
    EXPECT_NE(mbatch_batch_node, nullptr);
    GeTensorDesc tensor_desc = mbatch_batch_node->GetOpDesc()->GetOutputDesc(0U);
    tensor_desc.SetShape(GeShape({2, -1, 8}));
    mbatch_batch_node->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
    mbatch_batch_node->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  }

  {
    const auto mbatch_output_node = graph->FindNode(NODE_NAME_NET_OUTPUT);
    EXPECT_NE(mbatch_output_node, nullptr);
    const auto &op_desc = mbatch_output_node->GetOpDesc();
    const size_t dynamic_dims_index = op_desc->GetAllInputsSize() - 1U;
    GeTensorDesc tensor_desc = op_desc->GetInputDesc(dynamic_dims_index);
    tensor_desc.SetShape(GeShape({2, 4, 8}));
    op_desc->UpdateInputDesc(dynamic_dims_index, tensor_desc);
  }

  DEF_GRAPH(g1) {
    auto label_switch = OP_CFG(LABELSWITCHBYINDEX).Attr(ATTR_NAME_LABEL_SWITCH_LIST, std::vector<int64_t>{0, 1});
    auto label_set_l0 = OP_CFG(LABELSET).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 0);
    auto label_gotoex = OP_CFG(LABELGOTOEX).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 2);
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    CHAIN(NODE("g1/_arg_0", data_1)->EDGE(0, 0)->NODE("g1/add", ADD)->NODE("g1/output_ascend_mbatch_batch_0", NETOUTPUT));
    CHAIN(NODE("g1/_arg_1", data_2)->EDGE(0, 1)->NODE("g1/add"));

    CHAIN(NODE("g1/_index", data_0)->NODE("g1/switch", label_switch)->CTRL_EDGE()->NODE("label_set_l0", label_set_l0)->CTRL_EDGE()->NODE("g1/_arg_0"));
    CHAIN(NODE("g1/output_ascend_mbatch_batch_0")->CTRL_EDGE()->NODE("label_gotoex", label_gotoex));
  };
  case1 = ToComputeGraph(g1);
  SetUnknownOpKernel(case1, mem_offset);
  AddCaseBranch(graph, "ascend_mbatch_shape_case", case1);
  const auto batch1_output = case1->FindNode("g1/output_ascend_mbatch_batch_0");
  EXPECT_NE(batch1_output, nullptr);
  EXPECT_TRUE(AttrUtils::SetStr(batch1_output->GetOpDesc(), ATTR_NAME_BATCH_LABEL, "batch_label_0"));

  DEF_GRAPH(g2) {
    auto label_set_l1 = OP_CFG(LABELSET).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 1);
    auto label_set_l2 = OP_CFG(LABELSET).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 2);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    CHAIN(NODE("g2/_arg_0", data_1)->EDGE(0, 0)->NODE("g2/add", ADD)->NODE("g2/output_ascend_mbatch_batch_1", NETOUTPUT));
    CHAIN(NODE("g2/_arg_1", data_2)->EDGE(0, 1)->NODE("g2/add"));

    CHAIN(NODE("label_set_l1", label_set_l1)->CTRL_EDGE()->NODE("g2/_arg_0"));
    CHAIN(NODE("g2/output_ascend_mbatch_batch_1")->CTRL_EDGE()->NODE("label_set_l2", label_set_l2));
  };
  case2 = ToComputeGraph(g2);
  SetUnknownOpKernel(case2, mem_offset);
  AddCaseBranch(graph, "ascend_mbatch_shape_case", case2);
  const auto batch2_output = case2->FindNode("g2/output_ascend_mbatch_batch_1");
  EXPECT_NE(batch2_output, nullptr);
  EXPECT_TRUE(AttrUtils::SetStr(batch2_output->GetOpDesc(), ATTR_NAME_BATCH_LABEL, "batch_label_1"));
}

void BuildGraphModel(const ComputeGraphPtr &graph, const ComputeGraphPtr &case1, const ComputeGraphPtr &case2,
                     uint32_t mem_offset, GeModelPtr &ge_model) {
  TBEKernelStore tbe_kernel_store;
  InitConstantNode(graph, "ascend_mbatch_shape_const", 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef(graph, *model_task_def, "ascend_mbatch_shape_mapindex");

  const std::vector<uint32_t> label_idx_list{0, 1};
  InitLabelSwitchDef(case1, *model_task_def, "g1/switch");
  InitLabelSetDef(case1, *model_task_def, "label_set_l0");
  InitKernelTaskDef(case1, *model_task_def, "g1/add");
  InitLabelGotoDef(case1, *model_task_def, "label_gotoex");

  InitLabelSetDef(case2, *model_task_def, "label_set_l1");
  InitKernelTaskDef(case2, *model_task_def, "g2/add");
  InitLabelSetDef(case2, *model_task_def, "label_set_l2");

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));

  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
}

using DynamicAttribute = std::function<void(const uint32_t model_id)>;
static void TestDynamicBatchSize(const uint32_t model_id) {
  GeExecutor ge_executor;
  {
    std::vector<uint64_t> dynamic_dims; std::vector<uint64_t> cur_dynamic_dims;
    EXPECT_EQ(ge_executor.GetCurDynamicDims(model_id, dynamic_dims, cur_dynamic_dims), SUCCESS);
  }

  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = sizeof(uint64_t); uint64_t batch_size = 4U;
    EXPECT_EQ(ge_executor.SetDynamicBatchSize(model_id, &dynamic_input_addr, length, batch_size), SUCCESS);
  }

  {
    std::vector<int64_t> batch_info; int32_t dynamic_type = 0U;
    EXPECT_EQ(ge_executor.GetCurShape(model_id, batch_info, dynamic_type), SUCCESS);
  }

  {
    std::vector<TensorDesc> input_desc; std::vector<TensorDesc> output_desc; bool new_model_desc = false;
    EXPECT_EQ(ge_executor.GetModelDescInfo(model_id, input_desc, output_desc, new_model_desc), SUCCESS);
  }

  {
    std::vector<std::vector<int64_t>> batch_info; int32_t dynamic_type = 0U;
    EXPECT_EQ(ge_executor.GetDynamicBatchInfo(model_id, batch_info, dynamic_type), SUCCESS);
  }

  {
    std::vector<std::vector<int64_t>> batch_info;
    EXPECT_EQ(ge_executor.GetCombinedDynamicDims(model_id, batch_info), SUCCESS);
  }

  {
    std::vector<std::string> user_designate_shape_order;
    EXPECT_EQ(ge_executor.GetUserDesignateShapeOrder(model_id, user_designate_shape_order), SUCCESS);
  }

  {
    std::string op_name; std::string attr_name; std::string attr_value;
    EXPECT_EQ(ge_executor.GetOpAttr(model_id, op_name, attr_name, attr_value), SUCCESS);
  }

  {
    std::vector<std::string> dynamic_output_shape_info;
    EXPECT_EQ(ge_executor.GetModelAttr(model_id, dynamic_output_shape_info), SUCCESS);
  }

  {
    uint32_t max_size = 0U;
    EXPECT_EQ(ge_executor.GetMaxUsedMemory(model_id, max_size), SUCCESS);
  }

  {
    uint32_t device_id = 0U;
    EXPECT_EQ(GeExecutor::GetDeviceIdByModelId(model_id, device_id), SUCCESS);
  }

  {
    size_t shape_count = 0U;
    EXPECT_EQ(ge_executor.GetBatchInfoSize(model_id, shape_count), SUCCESS);
  }

  //{
  //  uint64_t dynamic_input_addr = 0U; uint64_t length = 0U; std::vector<kAippDynamicBatchPara> aipp_batch_para; kAippDynamicPara aipp_parms;
  //  EXPECT_EQ(ge_executor.SetDynamicAippData(model_id, &dynamic_input_addr, length, aipp_batch_para, aipp_parms), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; AippConfigInfo aipp_info;
  //  EXPECT_EQ(ge_executor.GetAIPPInfo(model_id, index, aipp_info), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; InputAippType type; size_t aipp_index = 0U;
  //  EXPECT_EQ(ge_executor.GetAippType(model_id, index, type, aipp_index), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; std::vector<InputOutputDims> input_dims; std::vector<InputOutputDims> output_dims;
  //  EXPECT_EQ(ge_executor.GetAllAippInputOutputDims(model_id, index, input_dims, output_dims), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; OriginInputInfo orig_input_info;
  //  EXPECT_EQ(ge_executor.GetOrigInputInfo(model_id, index, orig_input_info), SUCCESS);
  //}

  // {
  //   MsprofCommandHandle_t command{};
  //   command.profSwitch = 0xFFFFFFFFFFFFFFFF;
  //   command.modelId = model_id;
  //   command.type = 0;
  //   EXPECT_EQ(ProfCtrlHandle(RT_PROF_CTRL_SWITCH, &command, sizeof(command)), RT_ERROR_NONE);
  // }

  // {
  //   MsprofCommandHandle_t command{};
  //   command.profSwitch = 0xFFFFFFFFFFFFFFFF;
  //   command.modelId = model_id;
  //   command.type = 4;
  //   EXPECT_EQ(ProfCtrlHandle(RT_PROF_CTRL_SWITCH, &command, sizeof(command)), RT_ERROR_NONE);
  // }

  // {
  //   MsprofCommandHandle_t command{};
  //   command.profSwitch = 0xFFFFFFFFFFFFFFFF;
  //   command.modelId = model_id;
  //   command.type = 5;
  //   EXPECT_EQ(ProfCtrlHandle(RT_PROF_CTRL_SWITCH, &command, sizeof(command)), RT_ERROR_NONE);
  // }

  // {
  //   MsprofCommandHandle_t command{};
  //   command.profSwitch = 0xFFFFFFFFFFFFFFFF;
  //   command.modelId = model_id;
  //   command.type = 3;
  //   EXPECT_EQ(ProfCtrlHandle(RT_PROF_CTRL_SWITCH, &command, sizeof(command)), RT_ERROR_NONE);
  //   EXPECT_FALSE(ProfilingProperties::Instance().is_load_profiling_);
  //   EXPECT_FALSE(ProfilingProperties::Instance().is_training_trace_);
  // }
}

static void TestDynamicImageSize(const uint32_t model_id) {
  GeExecutor ge_executor;
  {
    std::vector<uint64_t> dynamic_dims; std::vector<uint64_t> cur_dynamic_dims;
    EXPECT_EQ(ge_executor.GetCurDynamicDims(model_id, dynamic_dims, cur_dynamic_dims), SUCCESS);
  }

  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = sizeof(uint64_t); uint64_t image_height = 340U; uint64_t image_width = 560U;
    EXPECT_EQ(ge_executor.SetDynamicImageSize(model_id, &dynamic_input_addr, length, image_height, image_width), SUCCESS);
  }

  {
    std::vector<int64_t> batch_info; int32_t dynamic_type = 0U;
    EXPECT_EQ(ge_executor.GetCurShape(model_id, batch_info, dynamic_type), SUCCESS);
  }

  {
    std::vector<TensorDesc> input_desc; std::vector<TensorDesc> output_desc; bool new_model_desc = false;
    EXPECT_EQ(ge_executor.GetModelDescInfo(model_id, input_desc, output_desc, new_model_desc), SUCCESS);
  }

  {
    std::vector<std::vector<int64_t>> batch_info; int32_t dynamic_type = 0U;
    EXPECT_EQ(ge_executor.GetDynamicBatchInfo(model_id, batch_info, dynamic_type), SUCCESS);
  }

  {
    std::vector<std::vector<int64_t>> batch_info;
    EXPECT_EQ(ge_executor.GetCombinedDynamicDims(model_id, batch_info), SUCCESS);
  }

  {
    std::vector<std::string> user_designate_shape_order;
    EXPECT_EQ(ge_executor.GetUserDesignateShapeOrder(model_id, user_designate_shape_order), SUCCESS);
  }

  {
    std::string op_name; std::string attr_name; std::string attr_value;
    EXPECT_EQ(ge_executor.GetOpAttr(model_id, op_name, attr_name, attr_value), SUCCESS);
  }

  {
    std::vector<std::string> dynamic_output_shape_info;
    EXPECT_EQ(ge_executor.GetModelAttr(model_id, dynamic_output_shape_info), SUCCESS);
  }

  {
    uint32_t max_size = 0U;
    EXPECT_EQ(ge_executor.GetMaxUsedMemory(model_id, max_size), SUCCESS);
  }

  {
    uint32_t device_id = 0U;
    EXPECT_EQ(GeExecutor::GetDeviceIdByModelId(model_id, device_id), SUCCESS);
  }

  {
    size_t shape_count = 0U;
    EXPECT_EQ(ge_executor.GetBatchInfoSize(model_id, shape_count), SUCCESS);
  }

  //{
  //  uint64_t dynamic_input_addr = 0U; uint64_t length = 0U; std::vector<kAippDynamicBatchPara> aipp_batch_para; kAippDynamicPara aipp_parms;
  //  EXPECT_EQ(ge_executor.SetDynamicAippData(model_id, &dynamic_input_addr, length, aipp_batch_para, aipp_parms), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; AippConfigInfo aipp_info;
  //  EXPECT_EQ(ge_executor.GetAIPPInfo(model_id, index, aipp_info), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; InputAippType type; size_t aipp_index = 0U;
  //  EXPECT_EQ(ge_executor.GetAippType(model_id, index, type, aipp_index), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; std::vector<InputOutputDims> input_dims; std::vector<InputOutputDims> output_dims;
  //  EXPECT_EQ(ge_executor.GetAllAippInputOutputDims(model_id, index, input_dims, output_dims), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; OriginInputInfo orig_input_info;
  //  EXPECT_EQ(ge_executor.GetOrigInputInfo(model_id, index, orig_input_info), SUCCESS);
  //}
}

static void TestDynamicDimsSize(const uint32_t model_id) {
  GeExecutor ge_executor;
  {
    std::vector<uint64_t> dynamic_dims{2,4,8}; std::vector<uint64_t> cur_dynamic_dims;
    EXPECT_EQ(ge_executor.GetCurDynamicDims(model_id, dynamic_dims, cur_dynamic_dims), SUCCESS);
  }

  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = sizeof(uint64_t); std::vector<uint64_t> dynamic_dims{2,4,8};
    EXPECT_EQ(ge_executor.SetDynamicDims(model_id, &dynamic_input_addr, length, dynamic_dims), SUCCESS);
  }

  {
    std::vector<int64_t> batch_info; int32_t dynamic_type = 0U;
    EXPECT_EQ(ge_executor.GetCurShape(model_id, batch_info, dynamic_type), SUCCESS);
  }

  {
    std::vector<TensorDesc> input_desc; std::vector<TensorDesc> output_desc; bool new_model_desc = false;
    EXPECT_EQ(ge_executor.GetModelDescInfo(model_id, input_desc, output_desc, new_model_desc), SUCCESS);
  }

  {
    std::vector<std::vector<int64_t>> batch_info; int32_t dynamic_type = 0U;
    EXPECT_EQ(ge_executor.GetDynamicBatchInfo(model_id, batch_info, dynamic_type), SUCCESS);
  }

  {
    std::vector<std::vector<int64_t>> batch_info;
    EXPECT_EQ(ge_executor.GetCombinedDynamicDims(model_id, batch_info), SUCCESS);
  }

  {
    std::vector<std::string> user_designate_shape_order;
    EXPECT_EQ(ge_executor.GetUserDesignateShapeOrder(model_id, user_designate_shape_order), SUCCESS);
  }

  {
    std::string op_name; std::string attr_name; std::string attr_value;
    EXPECT_EQ(ge_executor.GetOpAttr(model_id, op_name, attr_name, attr_value), SUCCESS);
  }

  {
    std::vector<std::string> dynamic_output_shape_info;
    EXPECT_EQ(ge_executor.GetModelAttr(model_id, dynamic_output_shape_info), SUCCESS);
  }

  {
    uint32_t max_size = 0U;
    EXPECT_EQ(ge_executor.GetMaxUsedMemory(model_id, max_size), SUCCESS);
  }

  {
    uint32_t device_id = 0U;
    EXPECT_EQ(GeExecutor::GetDeviceIdByModelId(model_id, device_id), SUCCESS);
  }

  {
    size_t shape_count = 0U;
    EXPECT_EQ(ge_executor.GetBatchInfoSize(model_id, shape_count), SUCCESS);
  }
  //{
  //  uint64_t dynamic_input_addr = 0U; uint64_t length = 0U; std::vector<kAippDynamicBatchPara> aipp_batch_para; kAippDynamicPara aipp_parms;
  //  EXPECT_EQ(ge_executor.SetDynamicAippData(model_id, &dynamic_input_addr, length, aipp_batch_para, aipp_parms), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; AippConfigInfo aipp_info;
  //  EXPECT_EQ(ge_executor.GetAIPPInfo(model_id, index, aipp_info), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; InputAippType type; size_t aipp_index = 0U;
  //  EXPECT_EQ(ge_executor.GetAippType(model_id, index, type, aipp_index), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; std::vector<InputOutputDims> input_dims; std::vector<InputOutputDims> output_dims;
  //  EXPECT_EQ(ge_executor.GetAllAippInputOutputDims(model_id, index, input_dims, output_dims), SUCCESS);
  //}

  //{
  //  uint32_t index = 0U; OriginInputInfo orig_input_info;
  //  EXPECT_EQ(ge_executor.GetOrigInputInfo(model_id, index, orig_input_info), SUCCESS);
  //}
}

// ModelExecutor::RunThread -> ExecuteGraphAsync -> AsyncExecuteModel -> DataInputTensor -> DavinciModel::Push -> Run
Status OnlineInferDynamic(const ComputeGraphPtr &graph, const GeModelPtr &ge_model, const OmeContext &ome_context,
                          const DynamicAttribute &dynamic_callback, const bool sink_dynamic = false) {
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);
  graph_node->SetOmeContext(ome_context);

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<gert::Tensor> inputs(2);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  if (sink_dynamic) { // GETDYNAMICDIMS on output for sink dynamic.
    gert::Tensor tensor;
    TensorCheckUtils::ConstructGertTensor(tensor, {1}, DT_INT64, FORMAT_ND);
    inputs.emplace_back(std::move(tensor));
  }

  std::mutex run_mutex;
  std::condition_variable model_run_cv;
  Status run_status = FAILED;
  std::vector<gert::Tensor> run_outputs;
  const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    std::unique_lock<std::mutex> lock(run_mutex);
    run_status = status;
    run_outputs.swap(outputs);
    model_run_cv.notify_one();
  };

  GEThreadLocalContext context;
  error_message::ErrorManagerContext error_context;
  graph_node->Lock();
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  if (arg == nullptr) {
    return FAILED;
  }
  arg->graph_node = graph_node;
  arg->graph_id = graph_id;
  arg->session_id = 2001;
  arg->error_context = error_context;
  arg->input_tensor = std::move(inputs);
  arg->context = context;
  arg->callback = callback;
  EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

  std::unique_lock<std::mutex> lock(run_mutex);
  EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
  EXPECT_EQ(run_status, SUCCESS);
  EXPECT_EQ(run_outputs.size(), sink_dynamic ? 1U : 2U);  // GETDYNAMICDIMS no output.

  dynamic_callback(ge_root_model->GetModelId());
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  return run_status;
}

static Graph BuildGetNextSinkGraphWithSubgraph() {
  DEF_GRAPH(sub_1) {
    const auto data_0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Build("sub_1_data_0");
    const auto data_1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
        .Build("sub_1_data_1");
    CHAIN(NODE(data_0)->NODE("less", LESS));
    CHAIN(NODE(data_1)->NODE("less"));
  };

  DEF_GRAPH(sub_2) {
    const auto data_0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Build("sub_2_data_0");
    const auto data_1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
        .Build("sub_2_data_1");

    int32_t value = 100;
    GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT32);
    GeTensorPtr const_tensor =
        std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&value), sizeof(int32_t));
    const auto const1 = OP_CFG(CONSTANT).OutCnt(1).Weight(const_tensor).Build("const1");
    const auto const2 = OP_CFG(CONSTANT).OutCnt(1).Weight(const_tensor).Build("const2");
    CHAIN(NODE(const1)->NODE("gen_mask", DROPOUTGENMASK));
    CHAIN(NODE(const2)->NODE("gen_mask", DROPOUTGENMASK));
    CTRL_CHAIN(NODE(data_1)->NODE(const1));
  };

  DEF_GRAPH(g1) {
    const auto iterator_node = OP_CFG(FRAMEWORKOP)
        .Attr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2")
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 3, 16, 16})
        .Build("iterator_node");

    CHAIN(NODE("var", VARIABLE)->NODE("if_node", IF, sub_1, sub_2));
    CHAIN(NODE(iterator_node)->NODE("if_node"));
  };
  return ToGeGraph(g1);
}

static Graph CreateHybridGraph() {
  DEF_GRAPH(test_graph) {
    const auto data0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1,3,3})
        .Build("data0");
    const auto data1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1,3,3})
        .Build("data1");
    auto add0 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {})
        .Build("add0");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}

static Graph CreateDynamicDimsGraph() {
  DEF_GRAPH(test_graph) {
    const auto data0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("data0");
    const auto data1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("data1");
    auto add0 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                    .Build("add0");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .InCnt(2).OutCnt(1).Build("mul1");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(mul1));
    CHAIN(NODE(data0)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(mul1)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(add0)->CTRL_EDGE()->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}
static Graph CreateDynamicDimsGraph1() {
  DEF_GRAPH(test_graph) {
    const auto data0 = OP_CFG(DATA)
                           .Attr(ATTR_NAME_INDEX, 0)
                           .OutCnt(1)
                           .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                           .Build("data0");
    const auto data1 = OP_CFG(DATA)
                           .Attr(ATTR_NAME_INDEX, 1)
                           .OutCnt(1)
                           .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                           .Build("data1");
    auto add0 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                    .Build("add0");
    auto mul1 = OP_CFG(MUL)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                    .InCnt(2).OutCnt(1).Build("mul1");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(mul1));
    CHAIN(NODE(data0)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(mul1)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(add0)->CTRL_EDGE()->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}
static Graph CreateDynamicDimsGraphWithScalarInput() {
  DEF_GRAPH(test_graph) {
    const auto data0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("data0");
    const auto data1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("data1");
    const auto data2 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {})
        .Build("data2");
    auto add0 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                    .Build("add0");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .InCnt(2).OutCnt(1).Build("mul1");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(2)
                          .OutCnt(1)
                          .Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(mul1));
    CHAIN(NODE(data0)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(mul1)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(net_output));
    CHAIN(NODE(add0)->CTRL_EDGE()->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}

static Graph CreateDynamicDimsGraphNoData() {
  DEF_GRAPH(test_graph) {
    const auto const0 = OP_CFG(CONSTANT)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("const0");
    const auto const1 = OP_CFG(CONSTANT)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("const1");
    auto add0 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                    .Build("add0");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .InCnt(2).OutCnt(1).Build("mul1");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Build("net_output");

    CHAIN(NODE(const0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(mul1));
    CHAIN(NODE(const0)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(mul1)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(add0)->CTRL_EDGE()->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}

static Graph CreateDynamicDimsGraphWithMoreData() {
  DEF_GRAPH(test_graph) {
    const auto data0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
        .Build("cat");
    const auto data1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
        .Build("apple");
    const auto data2 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
        .Build("bed");
    auto var0 = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 1, 1}).Build("var0");
    std::vector<int32_t> data_value = {100,100};
    GeTensorDesc data_tensor_desc(GeShape({1, 2, 1, 1}), FORMAT_NCHW, DT_INT32);
    GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), data_value.size() * sizeof(int32_t));
    auto const_1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_1");
    auto const_2 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const_2");
    auto add0 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
                    .Build("add0");

    auto add1 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
                    .Build("add1");
    auto add2 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
                    .Build("add2");
    auto add3 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
                    .Build("add3");

    auto add4 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,1,1})
                    .Build("add4");
    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(add0)->EDGE(0, 1)->NODE(add1));
    CHAIN(NODE(const_1)->NODE("identity1", IDENTITY)->EDGE(0, 0)->NODE("identityn", IDENTITYN)->EDGE(0, 0)->NODE(add2)->EDGE(0, 0)->NODE(add3));
    CHAIN(NODE(const_2)->NODE("identity2", IDENTITY)->EDGE(0, 1)->NODE("identityn", IDENTITYN)->EDGE(1, 1)->NODE(add2));
    CHAIN(NODE(var0)->NODE("identity3", IDENTITY)->EDGE(0, 2)->NODE("identityn", IDENTITYN)->EDGE(2, 1)->NODE(add3));
    CHAIN(NODE(add3)->EDGE(0, 0)->NODE(add4)->NODE(net_output));
    CHAIN(NODE(add1)->EDGE(0, 1)->NODE(add4));
  };
  return ToGeGraph(test_graph);
}

Status OnlineInferDynCompile(const Graph &graph, const uint32_t graph_id,
                             const map<AscendString, AscendString> &options) {
  GeSession session(options);
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);

  // build_graph through session
  ret = session.CompileGraph(graph_id);
  return ret;
}

TEST_F(OnlineInferTest, online_infer_dynamic_execute) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph, case1, case2;
  BuildSampleGraph(graph, case1, case2, mem_offset);
  EXPECT_NE(graph, nullptr);
  const auto mbatch_case_node = graph->FindNode("ascend_mbatch_shape_case");
  EXPECT_NE(mbatch_case_node, nullptr);
  const auto mbatch_batch_node = graph->FindNode("_arg_ascend_mbatch_batch_0");
  EXPECT_NE(mbatch_batch_node, nullptr);
  const auto mbatch_output_node = graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(mbatch_output_node, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, case1, case2, mem_offset, ge_model);
  EXPECT_NE(ge_model, nullptr);

  const auto case_desc = mbatch_case_node->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetInt(case_desc, ATTR_DYNAMIC_TYPE, static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH)));
  EXPECT_TRUE(AttrUtils::SetListStr(case_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, {}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_0", {2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_1", {4}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_0", {2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_1", {4}));
  {
    // Test LoadModelOnline: Online infer for dynamic DATA.
    OmeContext ome_context;
    ome_context.dynamic_node_type = DATA;
    ome_context.user_input_dims = {{"1", {}}, {"2", {}}};
    ome_context.dynamic_shape_dims = {{}, {2}, {4}, {8}};
    ome_context.data_nodes.emplace_back(mbatch_batch_node);
    ome_context.getnext_nosink_nodes.clear();
    EXPECT_EQ(OnlineInferDynamic(graph, ge_model, ome_context, &TestDynamicBatchSize), SUCCESS);   // ParseInputsDimsForData
  }

  EXPECT_TRUE(AttrUtils::SetInt(case_desc, ATTR_DYNAMIC_TYPE, static_cast<int32_t>(DynamicInputType::DYNAMIC_IMAGE)));
  EXPECT_TRUE(AttrUtils::SetListStr(case_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, {}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_0", {295,413}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_1", {340,560}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_0", {0,2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_1", {1,2}));
  {
    // Test LoadModelOnline: Online infer for dynamic DATA+GetNext(no_sink).
    OmeContext ome_context;
    ome_context.dynamic_node_type = DATA;
    ome_context.user_input_dims = {{"1", {}}};
    ome_context.dynamic_shape_dims = {{}, {2}, {4}, {8}};
    ome_context.data_nodes.emplace_back(mbatch_batch_node);
    ome_context.getnext_nosink_nodes.emplace_back(mbatch_batch_node);
    EXPECT_EQ(OnlineInferDynamic(graph, ge_model, ome_context, &TestDynamicImageSize), SUCCESS); // ParseInputsDimsForGetNextNoSinkAndData
  }

  EXPECT_TRUE(AttrUtils::SetInt(case_desc, ATTR_DYNAMIC_TYPE, static_cast<int32_t>(DynamicInputType::DYNAMIC_DIMS)));
  EXPECT_TRUE(AttrUtils::SetListStr(case_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, {"_arg_ascend_mbatch_batch_0"}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_0", {2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_1", {4}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_0", {0,2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_1", {1,2}));
  {
    // Test LoadModelOnline: Online infer for dynamic DATA+GetNext(no_sink).
    OmeContext ome_context;
    ome_context.dynamic_node_type = GETNEXT;
    ome_context.user_input_dims = {{"1", {}}};
    ome_context.dynamic_shape_dims = {{}, {2}, {4}, {8}};
    ome_context.data_nodes.emplace_back(mbatch_batch_node);
    ome_context.getnext_nosink_nodes.emplace_back(mbatch_batch_node);
    EXPECT_EQ(OnlineInferDynamic(graph, ge_model, ome_context, &TestDynamicDimsSize), SUCCESS); // ParseInputsDimsForGetNextNoSinkAndData
  }

  // DavinciModel::GetGetDynamicDimsNodeInfo: is_getnext_sink_dynamic_ = true;
  const auto &op_desc = mbatch_output_node->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, true));
  std::vector<std::string> dynamic_shape_info{ "2,0,2", "1,1,3" };
  EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_shape_info));
  {
    // Test LoadModelOnline: Online infer for dynamic GetNext(no_sink).
    OmeContext ome_context;
    ome_context.dynamic_node_type = GETNEXT;
    ome_context.user_input_dims = {{"1", {}}, {"2", {}}};
    ome_context.dynamic_shape_dims = {{}, {2}, {4}, {8}};
    ome_context.data_nodes.clear();
    ome_context.getnext_nosink_nodes.emplace_back(mbatch_batch_node);
    EXPECT_EQ(OnlineInferDynamic(graph, ge_model, ome_context, &TestDynamicDimsSize, true), SUCCESS); // ParseInputsDimsForData
  }

  {
    // Test LoadModelOnline: Online infer for dynamic GetNext(sink).
    OmeContext ome_context;
    ome_context.dynamic_node_type = GETNEXT;
    ome_context.user_input_dims = {};
    ome_context.dynamic_shape_dims = {{}, {2}, {4}, {8}};
    ome_context.data_nodes.clear();
    ome_context.getnext_nosink_nodes.clear();
    EXPECT_EQ(OnlineInferDynamic(graph, ge_model, ome_context, &TestDynamicDimsSize, true), SUCCESS); // Empty.
  }

  {
    // Test LoadModelOffline
    ModelHelper model_helper;
    model_helper.SetSaveMode(false);  // Save to buffer.
    ModelBufferData model_buffer;
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
    ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
    int64_t arg_0 = 127;
    int64_t arg_1 = 100;
    int64_t arg_2[9] = { 512 };
    RunModelData run_input_data;
    run_input_data.blobs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
    run_input_data.blobs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});
    run_input_data.blobs.emplace_back(DataBuffer{&arg_2, sizeof(arg_2), false, 0});

    int64_t arg_3 = 111;
    RunModelData run_output_data;
    run_output_data.blobs.emplace_back(DataBuffer{&arg_3, sizeof(arg_3), false, 0});

    uint32_t model_id = 0;
    GeExecutor ge_executor;
    model_data.weight_path = "/home";
    EXPECT_NE(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    model_data.weight_path = "";
    EXPECT_EQ(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    EXPECT_EQ(ge_executor.ExecModel(model_id, nullptr, run_input_data, run_output_data, true), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);

    auto ret = ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U);
    EXPECT_EQ(ret, SUCCESS);
    run_input_data.blobs[2].length = 65;
    ret = ge_executor.ExecModel(model_id, nullptr, run_input_data, run_output_data, true);
    EXPECT_EQ(ret, SUCCESS);
    ret = ge_executor.UnloadModel(model_id);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(OnlineInferTest, online_infer_dynamic_execute_invalide_input) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph, case1, case2;
  BuildSampleGraph(graph, case1, case2, mem_offset);
  EXPECT_NE(graph, nullptr);
  const auto mbatch_case_node = graph->FindNode("ascend_mbatch_shape_case");
  EXPECT_NE(mbatch_case_node, nullptr);
  const auto mbatch_batch_node = graph->FindNode("_arg_ascend_mbatch_batch_0");
  EXPECT_NE(mbatch_batch_node, nullptr);
  const auto mbatch_output_node = graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(mbatch_output_node, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, case1, case2, mem_offset, ge_model);
  EXPECT_NE(ge_model, nullptr);

  const auto case_desc = mbatch_case_node->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetInt(case_desc, ATTR_DYNAMIC_TYPE, static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH)));
  EXPECT_TRUE(AttrUtils::SetListStr(case_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, {}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_0", {2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_pred_value_1", {4}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_0", {2}));
  EXPECT_TRUE(AttrUtils::SetListInt(case_desc, "_combined_batch_1", {4}));
  {
    // Test LoadModelOnline: Online infer for dynamic DATA.
    OmeContext ome_context;
    ome_context.dynamic_node_type = DATA;
    ome_context.user_input_dims = {{"1", {}}};
    ome_context.dynamic_shape_dims = {{}, {2}, {4}, {8}};
    ome_context.data_nodes.emplace_back(mbatch_batch_node);
    ome_context.getnext_nosink_nodes.clear();

    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    graph_node->SetOmeContext(ome_context);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

    std::vector<gert::Tensor> inputs(3);
    TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
    TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
    TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    std::vector<gert::Tensor> run_outputs;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      run_outputs.swap(outputs);
      model_run_cv.notify_one();
    };

    GEThreadLocalContext context;
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph_id;
    arg->session_id = 2001;
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_NE(run_status, SUCCESS);

    TestDynamicBatchSize(ge_root_model->GetModelId());
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

ge::graphStatus StubInferShape(ge::Operator &op) {
  auto x_input_desc = op.GetInputDesc(0);
  auto x_shape = x_input_desc.GetShape().GetDims();
  auto x_type = x_input_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  (void)x_input_desc.GetShapeRange(x_shape_range);
  TensorDesc op_output_desc = op.GetOutputDesc(0);
  op_output_desc.SetShape(ge::Shape(x_shape));
  op_output_desc.SetOriginShape(ge::Shape(x_shape));
  op_output_desc.SetDataType(x_type);
  if (!x_shape_range.empty()) {
    op_output_desc.SetShapeRange(x_shape_range);
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  return op_desc->UpdateOutputDesc(0, TensorAdapter::TensorDesc2GeTensorDesc(op_output_desc));
}

ge::graphStatus GetShapeInferShape(ge::Operator &op) {
  std::cout << "Enter infershape getshape" << std::endl;
  std::vector<std::string> tiling_inline_engine;
  tiling_inline_engine.push_back("AIcoreEngine");
  vector<std::string> export_shape_engine;
  export_shape_engine.push_back("AIcoreEngine");
  op.SetAttr("_op_tiling_inline_engine", tiling_inline_engine);
  op.SetAttr("_op_export_shape_engine", export_shape_engine);
  return ge::GRAPH_SUCCESS;
}

void FakeMultiDimsEngine(GeRunningEnvFaker &ge_env) {
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.InstallDefault();
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
}

TEST_F(OnlineInferTest, online_infer_hybrid_mode) {
  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));
  Graph graph = CreateHybridGraph();

  // new session & add graph
  map<AscendString, AscendString> empty_options;
  Session session(empty_options);
  map<AscendString, AscendString> options = {{"ge.compileHybridMode", "1"},
      {"ge.inputShape", "data0:-1,3,3;data1:-1,3,3"}, {"ge.dynamicDims", "1,1;10,10"}, {"ge.dynamicNodeType", "1"}};
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1);
  EXPECT_EQ(session.AddGraph(10, graph, options), FAILED);
  unsetenv("AUTOFUSE_FLAGS");
  auto ret = session.AddGraph(10, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  ge::Tensor tensor;
  TensorDesc tensor_desc(Shape({3, 3, 3}), FORMAT_ND, DT_FLOAT);
  tensor.SetTensorDesc(tensor_desc);
  std::vector<uint8_t> data(27 * 4, 1);
  tensor.SetData(data);
  inputs.emplace_back(tensor);
  inputs.emplace_back(tensor);
  outputs.emplace_back(tensor);
  // build_graph through session
  EXPECT_EQ(session.BuildGraph(10, inputs), SUCCESS);
  EXPECT_FALSE(session.IsGraphNeedRebuild(10));
  EXPECT_EQ(session.RunGraphAsync(10, inputs, nullptr), SUCCESS);

  // dynamic gear execute
  inputs.clear();
  outputs.clear();
  tensor_desc.SetShape(Shape({1,3,3}));
  tensor.SetTensorDesc(tensor_desc);
  inputs.emplace_back(tensor);
  inputs.emplace_back(tensor);
  outputs.emplace_back(tensor);
  EXPECT_EQ(session.RunGraphAsync(10, inputs, nullptr), SUCCESS);

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph) {
  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));
  Graph graph = CreateDynamicDimsGraph();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto add0 = compute_graph->FindNode("add0");
  ASSERT_NE(add0, nullptr);
  auto net_output = compute_graph->FindNode("net_output");
  auto input0 = net_output->GetOpDesc()->MutableInputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.emplace_back(std::make_pair(1, 20));
  input0->SetShapeRange(shape_range);
  input0->GetOriginShapeRange(shape_range);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1;data1:-1"}, {"ge.dynamicDims", "1,1;10,10;20,20"}, {"ge.dynamicNodeType", "1"}};
  Session session(options);
  auto ret = session.AddGraph(10, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.CompileGraph(10);
  ASSERT_EQ(ret, SUCCESS);
  ret = session.LoadGraph(10, {}, nullptr);
  ASSERT_EQ(ret, SUCCESS);
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(10);
  ASSERT_NE(summary, nullptr);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{20};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_helper) {
  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));
  Graph graph = CreateDynamicDimsGraph();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto add0 = compute_graph->FindNode("add0");
  ASSERT_NE(add0, nullptr);
  auto net_output = compute_graph->FindNode("net_output");
  auto input0 = net_output->GetOpDesc()->MutableInputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.emplace_back(std::make_pair(1, 20));
  input0->SetShapeRange(shape_range);
  input0->GetOriginShapeRange(shape_range);
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1;data1:-1"}, {"ge.dynamicDims", "1,1;10,10;20,20"}, {"ge.dynamicNodeType", "1"}};
  GeSession session(options);
  auto ret = session.AddGraph(10, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build_graph through session
  ret = session.CompileGraph(10);
  EXPECT_EQ(ret, SUCCESS);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 4);
    EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  };

  unsetenv("RESOURCE_CONFIG_PATH");
  ge_env.Reset();
  ge_env.InstallDefault();
}
TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_with_running_format) {
  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("Reshape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));
  Graph graph = CreateDynamicDimsGraph1();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data0 = compute_graph->FindNode("data0");
  auto data0_input_desc = data0->GetOpDesc()->MutableInputDesc(0);
  auto data0_output_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetInt(*data0_input_desc, ATTR_NAME_STORAGE_FORMAT, FORMAT_NC1HWC0);
  ge::AttrUtils::SetListInt(*data0_input_desc, ATTR_NAME_STORAGE_SHAPE, {1,1,1,1});
  ge::AttrUtils::SetInt(*data0_output_desc, ATTR_NAME_STORAGE_FORMAT, FORMAT_NC1HWC0);
  ge::AttrUtils::SetListInt(*data0_output_desc, ATTR_NAME_STORAGE_SHAPE, {1,1,1,1});

  auto add0 = compute_graph->FindNode("add0");
  ASSERT_NE(add0, nullptr);
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1;1:-1");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1,1;10,10;20,20");
  auto net_output = compute_graph->FindNode("net_output");
  auto input0 = net_output->GetOpDesc()->MutableInputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.emplace_back(std::make_pair(1, 20));
  input0->SetShapeRange(shape_range);
  input0->GetOriginShapeRange(shape_range);
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1;data1:-1"}, {"ge.dynamicDims", "1,1;10,10;20,20"}, {"ge.dynamicNodeType", "1"}};
  GeSession session(options);
  auto ret = session.AddGraph(10, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build_graph through session
  ret = session.CompileGraph(10);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 4);
    EXPECT_EQ(graph->GetDirectNodesSize(), 4);
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == CASE) {
        EXPECT_EQ(node->GetOpDesc()->GetInputDescPtr(1)->GetFormat(), FORMAT_NC1HWC0);
      }
    }
  };

  unsetenv("RESOURCE_CONFIG_PATH");
  ge_env.Reset();
  ge_env.InstallDefault();
}
TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_with_scalar_input) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  Graph graph = CreateDynamicDimsGraphWithScalarInput();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1;data1:-1;data2:"},
      {"ge.dynamicDims", "10,10;2,2;4,4"},
      {"ge.dynamicNodeType", "1"}, {"ge.runFlag", "0"}};
  OnlineInferDynCompile(graph, 1, options);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 3);
    EXPECT_EQ(graph->GetDirectNodesSize(), 8);
  };
  RuntimeStub::Reset();
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_empty_dynamic_node) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  Graph graph = CreateDynamicDimsGraphNoData();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto add0 = compute_graph->FindNode("add0");
  ASSERT_NE(add0, nullptr);
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1;1:-1");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1,1;10,10;20,20");

  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1;data1:-1"}, {"ge.dynamicDims", "1,1;10,10;20,20"}, {"ge.dynamicNodeType", "1"}};
  EXPECT_NE(OnlineInferDynCompile(graph, 1, options), SUCCESS);
  // check result
  unsetenv("RESOURCE_CONFIG_PATH");
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_three_data) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  Graph graph = CreateDynamicDimsGraphWithMoreData();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "apple:-1,2,-1,-1;bed:-1,2,1,-1;cat:1,2,-1,-1"},
      {"ge.dynamicDims", "1,1,1,1,1,1,1;1,2,2,1,1,1,1;1,3,3,1,1,6,6"},
      {"ge.dynamicNodeType", "1"}, {"ge.runFlag", "0"}};
  OnlineInferDynCompile(graph, 1, options);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 3);
    EXPECT_EQ(graph->GetDirectNodesSize(), 9);
    std::map<std::string, std::vector<int64_t>> expect_shape_map = {
      {"apple_ascend_mbatch_batch_0", {1,2,1,1}},
      {"bed_ascend_mbatch_batch_0", {1,2,1,1}},
      {"cat_ascend_mbatch_batch_0", {1,2,1,1}},
      {"apple_ascend_mbatch_batch_1", {1,2,2,2}},
      {"bed_ascend_mbatch_batch_1", {1,2,1,1}},
      {"cat_ascend_mbatch_batch_1", {1,2,1,1}},
      {"apple_ascend_mbatch_batch_2", {1,2,3,3}},
      {"bed_ascend_mbatch_batch_2", {1,2,1,1}},
      {"cat_ascend_mbatch_batch_2", {1,2,6,6}}
    };
    for (const auto &sub_graph : graph->GetAllSubgraphs()) {
      for (const auto &node : sub_graph->GetAllNodes()) {
        if (node->GetType() == "Data") {
          auto node_name = node->GetName();
          auto output_dims = node->GetOpDesc()->GetOutputDesc(0U).GetShape().GetDims();
          auto iter = expect_shape_map.find(node_name);
          if (iter != expect_shape_map.end()) {
            EXPECT_EQ(output_dims, iter->second);
          }
        }
      }
    }
  };
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_three_data_error_dynamic_dim) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  Graph graph = CreateDynamicDimsGraphWithMoreData();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "apple:-1,2,-1,-1;bed:-1,2,1,-1;cat:1,2,-1,-1"},
      {"ge.dynamicDims", "1,1,1,1,1,1,1:1:3;1,2,2,1,1,1,1;1,3,3,1,1,6,6"},
      {"ge.dynamicNodeType", "1"}, {"ge.runFlag", "0"}};
  EXPECT_NE(OnlineInferDynCompile(graph, 1, options), SUCCESS);
  // check result
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, online_infer_dynamic_dims_graph_three_data_topo_sort_2) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  Graph graph = CreateDynamicDimsGraphWithMoreData();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "apple:-1,2,-1,-1;bed:-1,2,1,-1;cat:1,2,-1,-1"},
      {"ge.dynamicDims", "1,1,1,1,1,1,1;1,2,2,1,1,1,1;1,3,3,1,1,6,6"}, {"ge.dynamicNodeType", "1"},
       {"ge.topoSortingMode", "2"}, {"ge.runFlag", "0"}};
  OnlineInferDynCompile(graph, 1, options);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 3);
    EXPECT_EQ(graph->GetDirectNodesSize(), 9);
    std::map<std::string, std::vector<int64_t>> expect_shape_map = {
      {"apple_ascend_mbatch_batch_0", {1,2,1,1}},
      {"bed_ascend_mbatch_batch_0", {1,2,1,1}},
      {"cat_ascend_mbatch_batch_0", {1,2,1,1}},
      {"apple_ascend_mbatch_batch_1", {1,2,2,2}},
      {"bed_ascend_mbatch_batch_1", {1,2,1,1}},
      {"cat_ascend_mbatch_batch_1", {1,2,1,1}},
      {"apple_ascend_mbatch_batch_2", {1,2,3,3}},
      {"bed_ascend_mbatch_batch_2", {1,2,1,1}},
      {"cat_ascend_mbatch_batch_2", {1,2,6,6}}
    };
    for (const auto &sub_graph : graph->GetAllSubgraphs()) {
      for (const auto &node : sub_graph->GetAllNodes()) {
        if (node->GetType() == "Data") {
          auto node_name = node->GetName();
          auto output_dims = node->GetOpDesc()->GetOutputDesc(0U).GetShape().GetDims();
          auto iter = expect_shape_map.find(node_name);
          if (iter != expect_shape_map.end()) {
            EXPECT_EQ(output_dims, iter->second);
          }
        }
      }
    }
  };
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, OneDataTwoOutputMultiDims) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  DEF_GRAPH(test_graph) {
    const auto data0 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
        .Build("data0");
    auto add0 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                    .Build("add0");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data0)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(net_output));
  };
  Graph graph = ToGeGraph(test_graph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto add0 = compute_graph->FindNode("add0");
  ASSERT_NE(add0, nullptr);
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1;1:-1");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1,1;10,10;20,20");

  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1"}, {"ge.dynamicDims", "1;10;20"}, {"ge.dynamicNodeType", "1"}};
  OnlineInferDynCompile(graph, 1, options);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 4);
    EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  };
  unsetenv("RESOURCE_CONFIG_PATH");
  ge_env.Reset();
  ge_env.InstallDefault();
}
/**
 *     data
 *     |
 *    where
 *     |
 *    netoutput
 */
TEST_F(OnlineInferTest, GraphMultiDimsWithMiddleDynamicShape_ReportError) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);
  DEF_GRAPH(test_graph) {
                          const auto data0 = OP_CFG(DATA)
                              .Attr(ATTR_NAME_INDEX, 0)
                              .OutCnt(1)
                              .TensorDesc(FORMAT_ND, DT_FLOAT, {10})
                              .Build("data0");
                          auto where = OP_CFG(WHERE)
                              .InCnt(1)
                              .OutCnt(1)
                              .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
                              .Build("where");

                          auto net_output = OP_CFG(NETOUTPUT)
                              .InCnt(1)
                              .OutCnt(1)
                              .Build("net_output");

                          CHAIN(NODE(data0)->EDGE(0, 0)->NODE(where));
                          CHAIN(NODE(where)->EDGE(0, 0)->NODE(net_output));
                        };
  Graph graph = ToGeGraph(test_graph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  const auto where_node = compute_graph->FindNode("where");
  const auto where_infer_func = [](Operator &op) {
    op.GetOutputDesc(0).SetShape(Shape({-1, -1, -1, -1}));
    op.GetOutputDesc(0).SetOriginShape(Shape({-1, -1, -1, -1}));
    return GRAPH_SUCCESS;
  };
  where_node->GetOpDesc()->AddInferFunc(where_infer_func);

  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "data0:-1"}, {"ge.dynamicDims", "1;10;20"}};
  EXPECT_NE(OnlineInferDynCompile(graph, 1, options), SUCCESS);

  unsetenv("RESOURCE_CONFIG_PATH");
  ge_env.Reset();
  ge_env.InstallDefault();
}


TEST_F(OnlineInferTest, MinDimsSizeCheck) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  Graph graph = BuildGetNextSinkGraphWithSubgraph();
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "get_next:-1,3,16,16"},
      {"ge.dynamicDims", "2"},
      {"ge.dynamicNodeType", "0"}
  };
  EXPECT_NE(OnlineInferDynCompile(graph, 1, options), SUCCESS);

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(OnlineInferTest, MaxDimsSizeCheck) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  Graph graph = BuildGetNextSinkGraphWithSubgraph();
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  // new session & add graph
  map<AscendString, AscendString> options = {
      {"ge.inputShape", "get_next:-1,3,16,16"},
      {"ge.dynamicDims", "1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;\
        1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;\
        1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;\
        1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;\
        1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8;1;2;4;8"},
      {"ge.dynamicNodeType", "0"}
  };
  EXPECT_NE(OnlineInferDynCompile(graph, 1, options), SUCCESS);

  ge_env.Reset();
  ge_env.InstallDefault();
}
} // namespace ge

