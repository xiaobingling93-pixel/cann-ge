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
#include <vector>
#include <string.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "init_ge.h"
#include "depends/runtime/src/runtime_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "framework/executor/ge_executor.h"
#include "framework/common/types.h"
#include "graph/execute/model_executor.h"
#include "register/hidden_inputs_func_registry.h"
#include "aicpu_task_struct.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "macro_utils/dt_public_unscope.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "common/share_graph.h"
#include "graph/args_format_desc.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/tbe_handle_store/tbe_handle_store.h"

using namespace std;
using namespace testing;
namespace ge {
class StestScatteredCollection : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(StestScatteredCollection, InitAutoMixAicAivCtx_Test) {
  std::vector<uintptr_t> io_addrs;
  std::vector<void *> ext_args;
  std::set<size_t> mode_addr_idx;
  RuntimeParam runtime_param;
  runtime_param.logic_mem_base = 0x10;
  runtime_param.mem_size = 1000;
  runtime_param.mem_base = 0x2000;
  FftsPlusArgsHelper helper(runtime_param);
  FftsPlusProtoTransfer ffpt(0U, &helper, runtime_param, ext_args);
  ffpt.op_desc_ = std::make_shared<ge::OpDesc>("name", "type");

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef &mixctxdef = *mixaicaivctx->mutable_mix_aic_aiv_ctx();

  rtFftsPlusMixAicAivCtx_t ctx;
  uint32_t start_idx = 0;

  // Test: ctx_def.kernel_name_size() == 0
  mixctxdef.set_save_task_addr(1);
  mixctxdef.set_thread_dim(2);
  ctx.threadDim = mixctxdef.thread_dim();
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), ge::FAILED);

  start_idx = 7;
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), ge::FAILED);
}

TEST_F(StestScatteredCollection, mixl2_graph_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto reduce_sum = OP_CFG("ReduceSumD")
                          .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                          .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                          .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(
        NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("reduce_sum", reduce_sum)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("reduce_sum", reduce_sum));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  const auto &data_0 = root_graph->FindNode("_arg_0");
  EXPECT_NE(data_0, nullptr);
  data_0->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
  data_0->GetOpDesc()->SetOutputOffset({100});
  data_0->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  const auto &data_1 = root_graph->FindNode("_arg_1");
  EXPECT_NE(data_1, nullptr);
  data_1->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
  data_1->GetOpDesc()->SetOutputOffset({500});
  data_1->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"reduce_sum"});
  out_node->GetOpDesc()->SetSrcIndex({0});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->SetInputOffset({5000});
  auto reduce_node = root_graph->FindNode("reduce_sum");
  EXPECT_NE(reduce_node, nullptr);
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_mix_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(reduce_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV");
  std::vector<std::string> names_prefix{"_mix_aiv"};
  (void)ge::AttrUtils::SetListStr(reduce_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  (void)AttrUtils::SetStr(reduce_node->GetOpDesc(), "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  // all kernel
  AttrUtils::SetStr(reduce_node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");

  std::vector<char> atomic_bin(64, '\0');
  ge::TBEKernelPtr atomic_kernel = MakeShared<ge::OpKernelBin>("aiv_atomic_test", std::move(atomic_bin));
  (void)AttrUtils::SetStr(reduce_node->GetOpDesc(), ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel->GetName());

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(reduce_node->GetOpDesc()->GetId());
  ffts_plus_task_def->set_addr_size(1024);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  InitMixL2Def(root_graph, *mixaicaivctx, "reduce_sum");
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));

  const domi::FftsPlusMixAicAivCtxDef &ctx_def = mixaicaivctx->mix_aic_aiv_ctx();
  EXPECT_EQ(ctx_def.atm(), 0);
  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  kernel_store.AddTBEKernel(atomic_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
}

TEST_F(StestScatteredCollection, mixl2_with_args_format_graph_load_and_success) {
  auto hcom_hidden_func = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) {
    addr.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_func);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 7)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("ifa", ifa)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("ifa", ifa)->EDGE(3, 3)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 6; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa", "ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1, 2, 3});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(2, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(3, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 10000, 11000, 12000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  ifa_node->GetOpDescBarePtr()->UpdateInputDesc(0, desc);
  ifa_node->GetOpDescBarePtr()->AddDynamicInputDescByIndex("k", 2, 1);
  ifa_node->GetOpDescBarePtr()->UpdateInputDesc(1, desc);
  ifa_node->GetOpDescBarePtr()->UpdateInputDesc(2, desc);
  ifa_node->GetOpDescBarePtr()->AddDynamicInputDescByIndex("value", 2, 3);
  ifa_node->GetOpDescBarePtr()->UpdateInputDesc(3, scalar_desc);
  ifa_node->GetOpDescBarePtr()->UpdateInputDesc(4, scalar_desc);
  ifa_node->GetOpDescBarePtr()->AddInputDesc(5, desc);
  ifa_node->GetOpDescBarePtr()->AddDynamicInputDescByIndex("addition_in", 1, 6);
  ifa_node->GetOpDescBarePtr()->UpdateInputDesc(6, desc);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  ifa_node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  AttrUtils::SetInt(ifa_node->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_STREAM_ID, 0);
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_mix_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV");
  std::vector<std::string> names_prefix{"_mix_aiv"};
  (void)ge::AttrUtils::SetListStr(ifa_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  // all kernel
  AttrUtils::SetStr(ifa_node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");

  std::vector<char> atomic_bin(64, '\0');
  ge::TBEKernelPtr atomic_kernel = MakeShared<ge::OpKernelBin>("aiv_atomic_test", std::move(atomic_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel->GetName());

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(ifa_node->GetOpDesc()->GetId());
  ffts_plus_task_def->set_addr_size(1024);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  InitMixL2DefForIFA(root_graph, *mixaicaivctx, "ifa");
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));

  const domi::FftsPlusMixAicAivCtxDef &ctx_def = mixaicaivctx->mix_aic_aiv_ctx();
  EXPECT_EQ(ctx_def.atm(), 0);
  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  kernel_store.AddTBEKernel(atomic_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  RuntimeStub::Reset();
}

TEST_F(StestScatteredCollection, mixl2_mem_check_success) {
  auto hcom_hidden_func = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) {
    addr.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_func);

  auto root_graph = gert::ShareGraph::IFASingleGraph();
  EXPECT_NE(root_graph, nullptr);
  auto ifa_node = root_graph->FindNode("IncreFlashAttention");
  EXPECT_NE(ifa_node, nullptr);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("6");
  ifa_node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  AttrUtils::SetInt(ifa_node->GetOpDescBarePtr(), ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  AttrUtils::SetStr(ifa_node->GetOpDescBarePtr(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV");
  AttrUtils::SetStr(ifa_node->GetOpDescBarePtr(), TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetInt(ifa_node->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(ifa_node->GetOpDesc(), ATTR_NAME_MAX_TILING_SIZE, 1000);
  AttrUtils::SetBool(ifa_node->GetOpDesc(), "_memcheck", true);
  AttrUtils::SetStr(ifa_node->GetOpDesc(), "op_unique_key", "ifa_key_ffts");
  AttrUtils::SetInt(ifa_node->GetOpDesc(), "ori_op_para_size", 24);
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_mix_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV");
  std::vector<std::string> names_prefix{"_mix_aiv"};
  (void)ge::AttrUtils::SetListStr(ifa_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  // all kernel
  AttrUtils::SetStr(ifa_node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");

  std::vector<char> atomic_bin(64, '\0');
  ge::TBEKernelPtr atomic_kernel = MakeShared<ge::OpKernelBin>("aiv_atomic_test", std::move(atomic_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "_fake_ifa_kernel_bin_id");
  (void)AttrUtils::SetBool(ifa_node->GetOpDesc(), "_mix_with_enhanced_kernel", true);
  ifa_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, atomic_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(ifa_node->GetOpDesc()->GetId());
  ffts_plus_task_def->set_addr_size(1024);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();

  mixaicaivctx->set_op_index(ifa_node->GetOpDesc()->GetId());
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctx_def = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  mixctx_def->set_successor_num(26);
  mixctx_def->set_aten(1);
  mixctx_def->set_prefetch_config(1);
  mixctx_def->set_pred_cnt_init(1);
  mixctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    mixctx_def->add_successor_list(1);  // len = 26
  }
  mixctx_def->set_schem(1);

  mixctx_def->set_prefetch_enable_bitmap(1);
  mixctx_def->set_prefetch_once_bitmap(1);

  mixctx_def->set_pmg(1);
  mixctx_def->set_ns(1);
  mixctx_def->set_part_id(1);
  mixctx_def->set_qos(1);
  mixctx_def->set_atm(0);

  mixctx_def->set_non_tail_block_ratio_n(1);
  mixctx_def->set_tail_block_ratio_n(1);

  mixctx_def->set_thread_id(0);
  mixctx_def->set_thread_dim(1);

  mixctx_def->set_non_tail_block_dim(48);
  mixctx_def->set_tail_block_dim(48);

  mixctx_def->set_aiv_task_param_ptr_offset(32);
  mixctx_def->set_aic_task_param_ptr_offset(32);

  mixctx_def->add_kernel_name("_mix_aivtbeKernel_test");

  mixctx_def->add_task_addr(0);              // custom_value
  mixctx_def->add_task_addr(0xe7ffc67a000);  // ffts_addr
  mixctx_def->add_task_addr(0);              // hidden_input
  mixctx_def->add_task_addr(1000);           // query
  mixctx_def->add_task_addr(2000);           // k0
  mixctx_def->add_task_addr(3000);           // k1
  mixctx_def->add_task_addr(4000);           // k2
  mixctx_def->add_task_addr(5000);           // value0
  mixctx_def->add_task_addr(6000);           // value1
  mixctx_def->add_task_addr(7000);           // value2
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(8000);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(9000);
  mixctx_def->add_task_addr(10000);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(0);
  mixctx_def->add_task_addr(11000);
  mixctx_def->add_task_addr(12000);
  mixctx_def->add_task_addr(0);              // tiling
  mixctx_def->add_task_addr(0);              // hidden_input

  ge::ArgsFormatDesc args_format;
  size_t arg_id = 0;
  args_format.Append(ge::AddrType::INPUT, arg_id++);
  args_format.Append(ge::AddrType::INPUT_DESC, arg_id++, true);
  args_format.Append(ge::AddrType::INPUT_DESC, arg_id++, true);
  for (size_t i = 0; i < 12UL; i++) {
    args_format.Append(ge::AddrType::INPUT, arg_id++);
  }
  args_format.Append(ge::AddrType::OUTPUT, 0);
  auto args_format_str = args_format.ToString();
  mixctx_def->set_args_format(args_format_str);

  mixctx_def->set_input_output_count(3);
  mixctx_def->set_save_task_addr(1);
  for (int j = 0; j < 4; ++j) {
    mixctx_def->add_src_slot(1);  // len = 4, context ID for source data which is out of subgraph
  }
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));

  const domi::FftsPlusMixAicAivCtxDef &ctx_def = mixaicaivctx->mix_aic_aiv_ctx();
  EXPECT_EQ(ctx_def.atm(), 0);
  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  kernel_store.AddTBEKernel(atomic_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  RuntimeStub::Reset();
}

TEST_F(StestScatteredCollection, ifa_aicore_with_args_format_graph_load_and_success) {
  auto hcom_hidden_func = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) {
    addr.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_func);

  DEF_GRAPH(g1) {
                  auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
                  auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
                  auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
                  auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
                  auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
                  auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
                  auto ifa = OP_CFG("IncreFlashAttention_T")
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
                  CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
                  CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
                  CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
                  CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
                  CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
                  CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
                  CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
                };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);

  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = ifa_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  ifa_node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_tbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), "_kernelname", test_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATOMIC_ATTR_TBE_KERNEL_NAME, test_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "_fake_ifa_kernel_bin_id");
  ifa_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);


  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();

  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def.set_stream_id(0);

  task_def.set_stream_id(0);
  auto aicore_kernel = task_def.mutable_kernel();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{t}{ws*}{overflow_addr}");
  aicore_context.set_args_count(9);

  aicore_kernel->set_stub_func("stub_func");
  aicore_kernel->set_args_size(256);
  string args(256, '1');
  aicore_kernel->set_args(args.data(), args.size());

  uint16_t args_offset[9] = {0};
  aicore_context.set_args_offset(args_offset, 9 * sizeof(uint16_t));

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  RuntimeStub::Reset();
}

// mixl2
TEST_F(StestScatteredCollection, ifa_aicore_with_tiling_sink_graph_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr("_mix_aiv_kernel_list_first_name", true);
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = ifa_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});

  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("IncreFlashAttention_T");
  funcs->tiling = nullptr;
  funcs->tiling_parse = nullptr;
  funcs->compile_info_creator = nullptr;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(5), GRAPH_SUCCESS);

  // kernel
  AttrUtils::SetInt(ifa_node->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_STREAM_ID, 0);
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_mix_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV");
  std::vector<std::string> names_prefix{"_mix_aiv"};
  (void)ge::AttrUtils::SetListStr(ifa_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  ge::TBEHandleStore::GetInstance().StoreTBEHandle("_mix_aiv_static_bin", nullptr, nullptr);

  // all kernel
  AttrUtils::SetStr(ifa_node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // aicpu kernel
  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_op_index(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.block_dim}{tiling_context.tiling_key}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);
  // aicore_kenrel

  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(ifa_node->GetOpDesc()->GetId());
  ffts_plus_task_def->set_addr_size(1024);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  InitMixL2DefForIFATilingSink(root_graph, *mixaicaivctx, "ifa");
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  // update_pc_kenrel
  auto &rts_task_def = *model_task_def->add_task();
  rts_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_UPDATE));
  rts_task_def.mutable_update_pc_task()->set_stream_id(1);
  rts_task_def.mutable_update_pc_task()->set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  RuntimeStub::Reset();
}

UINT32 StubTilingMC2(gert::TilingContext *context) {
    context->SetNeedAtomic(false);
    context->SetTilingKey(666U);
    context->SetBlockDim(666U);
    size_t *workspace_size = context->GetWorkspaceSizes(1);
    *workspace_size = 66U;
    return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParseMC2(gert::KernelContext *context) {
    (void)context;
    return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreatorMC2() {
    auto tmp =  ge::MakeUnique<char>();
    return tmp.get();
}

/**
 * 用例描述： 测试包含mc2算子的计算图的runtime tiling
 *
 * 预置条件：
 * 1. 构造包含mc2算子的计算图
 *
 * 测试步骤：
 * 1. 构造计算图
 * 2. 设置runtime tiling策略, 设置args_format
 * 3. 执行
 *
 * 预期结果：
 * 1. 执行成功
 */
TEST_F(StestScatteredCollection, mc2kernel_runtime_tiling_success) {
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("MC2_T");
  funcs->tiling = StubTilingMC2;
  funcs->tiling_parse = StubTilingParseMC2;
  funcs->compile_info_creator = CompileInfoCreatorMC2;
  funcs->compile_info_deleter = nullptr;
  auto hcom_hidden_func = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) {
    addr.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_func);

  DEF_GRAPH(g1) {
          auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // x1
          auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // x2
          auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // bias
          auto mc2 = OP_CFG("MC2_T")
          .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
          .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
          .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
          CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("mc2", mc2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));  // y
          CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("mc2", mc2));
          CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("mc2", mc2));
          CHAIN(NODE("mc2", mc2)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));  // attention_out
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 2; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"mc2", "mc2"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto mc2_node = root_graph->FindNode("mc2");
  EXPECT_NE(mc2_node, nullptr);
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_mix_aictbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(mc2_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  std::vector<std::string> names_prefix{"_mix_aic"};
  (void)ge::AttrUtils::SetListStr(mc2_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  (void)AttrUtils::SetStr(mc2_node->GetOpDesc(), "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  // all kernel
  AttrUtils::SetStr(mc2_node->GetOpDesc(), "_mix_aic_kernel_list_first_name", "aic");

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  mc2_node->GetOpDescBarePtr()->SetInputOffset({1000, 2000, 3000});
  mc2_node->GetOpDescBarePtr()->SetOutputOffset({5000, 6000});
  mc2_node->GetOpDescBarePtr()->SetWorkspace({7000});
  mc2_node->GetOpDescBarePtr()->SetWorkspaceBytes({512});

  AttrUtils::SetInt(mc2_node->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(mc2_node->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, 0);

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();

  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(mc2_node->GetOpDescBarePtr()->GetId());
  aicpu_context.set_op_index(mc2_node->GetOpDescBarePtr()->GetId());
  aicpu_context.set_args_format("{i0}{}{}{o0}{o1}{hi.hcom0*}{ws0}{t}");
  aicpu_context.set_args_count(8);

  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  // mixl2 task
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(mc2_node->GetOpDesc()->GetId());
  ffts_plus_task_def->set_addr_size(1024);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  InitTaskDefForMC2(root_graph, *mixaicaivctx, "mc2");
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  const domi::FftsPlusMixAicAivCtxDef &ctx_def = mixaicaivctx->mix_aic_aiv_ctx();
  EXPECT_EQ(ctx_def.atm(), 0);
  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(mc2_node->GetName(), std::move(kernel_bin));
  ge_model->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
}

TEST_F(StestScatteredCollection, mc2kernel_graph_load_and_success) {
  auto hcom_hidden_func = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) {
    addr.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_func);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // x1
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // x2
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // bias
    auto mc2 = OP_CFG("MC2_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("mc2", mc2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));  // y
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("mc2", mc2));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("mc2", mc2));
    CHAIN(NODE("mc2", mc2)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));  // attention_out
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 2; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"mc2", "mc2"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto mc2_node = root_graph->FindNode("mc2");
  EXPECT_NE(mc2_node, nullptr);
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_mix_aictbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(mc2_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  std::vector<std::string> names_prefix{"_mix_aic"};
  (void)ge::AttrUtils::SetListStr(mc2_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  (void)AttrUtils::SetStr(mc2_node->GetOpDesc(), "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  // all kernel
  AttrUtils::SetStr(mc2_node->GetOpDesc(), "_mix_aic_kernel_list_first_name", "aic");

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  mc2_node->GetOpDescBarePtr()->SetInputOffset({1000, 2000, 3000});
  mc2_node->GetOpDescBarePtr()->SetOutputOffset({5000, 6000});
  mc2_node->GetOpDescBarePtr()->SetWorkspace({7000});
  mc2_node->GetOpDescBarePtr()->SetWorkspaceBytes({512});

  AttrUtils::SetInt(mc2_node->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(mc2_node->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, 0);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  mc2_node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();

  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(mc2_node->GetOpDescBarePtr()->GetId());
  aicpu_context.set_op_index(mc2_node->GetOpDescBarePtr()->GetId());
  aicpu_context.set_args_format("{i0}{}{}{o0}{o1}{hi.hcom0*}{ws0}{t}");
  aicpu_context.set_args_count(8);

  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");

  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  // mixl2 task
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(mc2_node->GetOpDesc()->GetId());
  ffts_plus_task_def->set_addr_size(1024);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  InitTaskDefForMC2(root_graph, *mixaicaivctx, "mc2");
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  const domi::FftsPlusMixAicAivCtxDef &ctx_def = mixaicaivctx->mix_aic_aiv_ctx();
  EXPECT_EQ(ctx_def.atm(), 0);
  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(mc2_node->GetName(), std::move(kernel_bin));
  ge_model->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
}
}  // namespace ge

namespace gert {
class StestRt2ScatteredCollection : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(StestRt2ScatteredCollection, InitAutoMixAicAivCtx_Test) {
  std::vector<uintptr_t> io_addrs;
  std::vector<void *> ext_args;
  std::vector<size_t> mode_addr_idx;
  const ge::RuntimeParam runtime_param;
  FftsPlusProtoTransfer ffpt(0U, io_addrs, runtime_param, ext_args, mode_addr_idx);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef &mixctxdef = *mixaicaivctx->mutable_mix_aic_aiv_ctx();

  rtFftsPlusMixAicAivCtx_t ctx;
  uint32_t start_idx = 0;

  // Test: ctx_def.kernel_name_size() == 0
  mixctxdef.set_save_task_addr(1);
  mixctxdef.set_thread_dim(2);
  ctx.threadDim = mixctxdef.thread_dim();
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), ge::FAILED);

  start_idx = 7;
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), ge::FAILED);
}
}  // namespace gert
