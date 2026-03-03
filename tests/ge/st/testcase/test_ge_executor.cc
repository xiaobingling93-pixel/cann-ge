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
#include "init_ge.h"
#include "utils/bench_env.h"
#include "ge_graph_dsl/graph_dsl.h"
#include <cstdlib>

#include "macro_utils/dt_public_scope.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "hybrid/hybrid_davinci_model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ge_api.h"
#include "framework/executor/ge_executor.h"
#include "graph/execute/model_executor.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/dump/dump_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "graph/load/model_manager/task_info/ge/profiler_trace_task_info.h"
#include "framework/common/ge_model_inout_types.h"
#include "ge/ut/ge/graph/passes/graph_builder_utils.h"
#include "depends/runtime/src/runtime_stub.h"
#include "runtime/v2/kernel/memory/rts_caching_mem_allocator.h"
#include "macro_utils/dt_public_unscope.h"
#include "stub/gert_runtime_stub.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace std;
using namespace testing;
namespace ge {

class MockMemRuntime : public ge::RuntimeStub {
 public:
  rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
    *free = 32UL * 1024UL * 1024UL * 1024UL;
    *total = 32UL * 1024UL * 1024UL * 1024UL;
    return RT_ERROR_NONE;
  }
  rtError_t rtModelCheckCompatibility(const char_t *OmSoCVersion, const char_t *OMArchVersion) {
    if (std::string(OmSoCVersion) == "Ascend310" && std::string(OMArchVersion) == "0") {
      return -1;
    }
    return RT_ERROR_NONE;
  }
};

class GeExecutorTest : public testing::Test {
 protected:
  void SetUp() override {
    ReInitGe();
    BenchEnv::Init();
    actual_info_type.clear();
    ModelManager::GetInstance().cust_aicpu_so_.clear();
    VarManagerPool::Instance().Destory();
  }
  void TearDown() override {
    actual_info_type.clear();
    ModelManager::GetInstance().cust_aicpu_so_.clear();
    VarManagerPool::Instance().Destory();
  }

 public:
  GeExecutor ge_executor_;
};

void BuildHcclSampleGraph(ComputeGraphPtr &graph, uint32_t &mem_offset) {
  DEF_GRAPH(g1) {
    const auto active_s = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{1});
    const auto less_node = OP_CFG(LESS).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    CHAIN(NODE("_arg_0", CONSTANT)->NODE("HcomAllreduce", HCOMALLREDUCE)->NODE("Less", less_node));
    CHAIN(NODE("_arg_1", DATA)->NODE("Less")->NODE("Less_Cast", CAST)->CTRL_EDGE()->NODE("Less_StreamActive", active_s));
    CHAIN(NODE("_arg_2", DATA)->NODE("mul", MUL)->EDGE(0, 1)->NODE("add_n", ADDN)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("_arg_3", DATA)->NODE("aipp", AIPP)->NODE("shape", SHAPE)->EDGE(0, 0)->NODE("add_n"));
    CHAIN(NODE("_cst_string", CONSTANTOP)->NODE("shape", SHAPE));
    CHAIN(NODE("_arg_1")->NODE("mul"));

    const auto switch_f = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_NOT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("Less_Cast")->EDGE(0, 0)->NODE("Less/StreamSwitch_f", switch_f));
    CHAIN(NODE("Less/StreamSwitch_Const_f", CONSTANTOP)->EDGE(0, 1)->NODE("Less/StreamSwitch_f"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_f"));

    const auto switch_t = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("Less_Cast")->EDGE(0, 0)->NODE("Less/StreamSwitch_t", switch_t));
    CHAIN(NODE("Less/StreamSwitch_Const_t", CONSTANTOP)->EDGE(0, 1)->NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_t"));

    const auto active_0 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("cond/pow", POW)->NODE("cond/sub", SUB)->
          NODE("merge_input_0_memcpy", MEMCPYASYNC)->CTRL_EDGE()->
          NODE("merge_input_0_active", active_0)->CTRL_EDGE()->
          NODE("cond/merge", STREAMMERGE)->EDGE(0, 2)->
          NODE("add_n"));
    CHAIN(NODE("merge_input_0_memcpy")->EDGE(0, 0)->NODE("cond/merge"));
    CHAIN(NODE("_arg_1")->EDGE(0, 1)->NODE("cond/pow"));
    CHAIN(NODE("Less/StreamSwitch_f")->CTRL_EDGE()->NODE("cond/pow"));
    CHAIN(NODE("_arg_1")->EDGE(0, 0)->NODE("cond/realdiv", REALDIV)->NODE("cond/sub"));
    CHAIN(NODE("_arg_2")->EDGE(0, 1)->NODE("cond/realdiv"));
    CHAIN(NODE("Less/StreamSwitch_f")->CTRL_EDGE()->NODE("cond/realdiv"));

    const auto active_1 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_1")->EDGE(0, 0)->NODE("cond/mul", MUL)->NODE("cond/add", ADD)->
          NODE("merge_input_1_memcpy", MEMCPYASYNC)->CTRL_EDGE()->
          NODE("merge_input_1_active", active_1)->CTRL_EDGE()->
          NODE("cond/merge"));
    CHAIN(NODE("merge_input_1_memcpy")->EDGE(0, 1)->NODE("cond/merge"));
    CHAIN(NODE("_arg_2")->EDGE(0, 1)->NODE("cond/mul"));
    CHAIN(NODE("Less/StreamSwitch_t")->CTRL_EDGE()->NODE("cond/mul"));
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("cond/square", SQUARE)->NODE("cond/add"));
    CHAIN(NODE("Less/StreamSwitch_t")->CTRL_EDGE()->NODE("cond/square"));
  };
  graph = ToComputeGraph(g1);
  graph->SetSessionID(10086);
  graph->SetGraphID(20061);
  SetUnknownOpKernel(graph, mem_offset, true);

  {
    const auto &node = graph->FindNode(NODE_NAME_NET_OUTPUT);
    EXPECT_NE(node, nullptr);
    GeTensorDesc input_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
  }

  {
    const auto &hccl_node = graph->FindNode("HcomAllreduce");
    EXPECT_NE(hccl_node, nullptr);
    GeTensorDesc input_desc1(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    hccl_node->GetOpDesc()->UpdateInputDesc(0, input_desc1);
    mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    hccl_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHccl);
  }

  {
    const auto &no = graph->FindNode("cond/pow");
    EXPECT_TRUE(AttrUtils::SetListInt(no->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, { RT_MEMORY_L1 }));
    const auto &ni = graph->FindNode("cond/sub");
    EXPECT_TRUE(AttrUtils::SetListInt(ni->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_L1, RT_MEMORY_HBM }));
  }
  {
    const auto &no = graph->FindNode("cond/pow");
  }
  {
    const auto &no = graph->FindNode("Less_Cast");
    EXPECT_TRUE(AttrUtils::SetListInt(no->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, { RT_MEMORY_TS }));
    const auto &nt = graph->FindNode("Less/StreamSwitch_t");
    EXPECT_TRUE(AttrUtils::SetListInt(nt->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_TS, RT_MEMORY_HBM }));
    const auto &nf = graph->FindNode("Less/StreamSwitch_f");
    EXPECT_TRUE(AttrUtils::SetListInt(nf->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_TS, RT_MEMORY_HBM }));
  }
}

void BuildHcclSampleGraphWithQos(ComputeGraphPtr &graph, uint32_t &mem_offset) {
  DEF_GRAPH(g1) {
    const auto active_s = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{1});
    const auto less_node = OP_CFG(LESS).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    CHAIN(NODE("_arg_0", CONSTANT)->NODE("HcomAllreduce", HCOMALLREDUCE)->NODE("Less", less_node));
    CHAIN(NODE("_arg_1", DATA)->NODE("Less")->NODE("Less_Cast", CAST)->CTRL_EDGE()->NODE("Less_StreamActive", active_s));
    CHAIN(NODE("_arg_2", DATA)->NODE("mul", MUL)->EDGE(0, 1)->NODE("add_n", ADDN)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("_arg_3", DATA)->NODE("aipp", AIPP)->NODE("shape", SHAPE)->EDGE(0, 0)->NODE("add_n"));
    CHAIN(NODE("_cst_string", CONSTANTOP)->NODE("shape", SHAPE));
    CHAIN(NODE("_arg_1")->NODE("mul"));

    const auto switch_f = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_NOT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("Less_Cast")->EDGE(0, 0)->NODE("Less/StreamSwitch_f", switch_f));
    CHAIN(NODE("Less/StreamSwitch_Const_f", CONSTANTOP)->EDGE(0, 1)->NODE("Less/StreamSwitch_f"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_f"));

    const auto switch_t = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("Less_Cast")->EDGE(0, 0)->NODE("Less/StreamSwitch_t", switch_t));
    CHAIN(NODE("Less/StreamSwitch_Const_t", CONSTANTOP)->EDGE(0, 1)->NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_t"));

    const auto active_0 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("cond/pow", POW)->NODE("cond/sub", SUB)->
          NODE("merge_input_0_memcpy", MEMCPYASYNC)->CTRL_EDGE()->
          NODE("merge_input_0_active", active_0)->CTRL_EDGE()->
          NODE("cond/merge", STREAMMERGE)->EDGE(0, 2)->
          NODE("add_n"));
    CHAIN(NODE("merge_input_0_memcpy")->EDGE(0, 0)->NODE("cond/merge"));
    CHAIN(NODE("_arg_1")->EDGE(0, 1)->NODE("cond/pow"));
    CHAIN(NODE("Less/StreamSwitch_f")->CTRL_EDGE()->NODE("cond/pow"));
    CHAIN(NODE("_arg_1")->EDGE(0, 0)->NODE("cond/realdiv", REALDIV)->NODE("cond/sub"));
    CHAIN(NODE("_arg_2")->EDGE(0, 1)->NODE("cond/realdiv"));
    CHAIN(NODE("Less/StreamSwitch_f")->CTRL_EDGE()->NODE("cond/realdiv"));

    const auto active_1 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_1")->EDGE(0, 0)->NODE("cond/mul", MUL)->NODE("cond/add", ADD)->
          NODE("merge_input_1_memcpy", MEMCPYASYNC)->CTRL_EDGE()->
          NODE("merge_input_1_active", active_1)->CTRL_EDGE()->
          NODE("cond/merge"));
    CHAIN(NODE("merge_input_1_memcpy")->EDGE(0, 1)->NODE("cond/merge"));
    CHAIN(NODE("_arg_2")->EDGE(0, 1)->NODE("cond/mul"));
    CHAIN(NODE("Less/StreamSwitch_t")->CTRL_EDGE()->NODE("cond/mul"));
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("cond/square", SQUARE)->NODE("cond/add"));
    CHAIN(NODE("Less/StreamSwitch_t")->CTRL_EDGE()->NODE("cond/square"));
  };
  graph = ToComputeGraph(g1);
  graph->SetSessionID(10086);
  graph->SetGraphID(20061);
  SetUnknownOpKernel(graph, mem_offset, true);

  {
    const auto &node = graph->FindNode(NODE_NAME_NET_OUTPUT);
    EXPECT_NE(node, nullptr);
    GeTensorDesc input_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
  }

  {
    const auto &hccl_node = graph->FindNode("HcomAllreduce");
    EXPECT_NE(hccl_node, nullptr);
    GeTensorDesc input_desc1(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    EXPECT_TRUE(AttrUtils::SetInt(hccl_node->GetOpDesc(), "_qos_service_label", 1));
    hccl_node->GetOpDesc()->UpdateInputDesc(0, input_desc1);
    mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
  }

  {
    const auto &no = graph->FindNode("cond/pow");
    EXPECT_TRUE(AttrUtils::SetListInt(no->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, { RT_MEMORY_L1 }));
    const auto &ni = graph->FindNode("cond/sub");
    EXPECT_TRUE(AttrUtils::SetListInt(ni->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_L1, RT_MEMORY_HBM }));
  }
  {
    const auto &no = graph->FindNode("cond/pow");
  }
  {
    const auto &no = graph->FindNode("Less_Cast");
    EXPECT_TRUE(AttrUtils::SetListInt(no->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, { RT_MEMORY_TS }));
    const auto &nt = graph->FindNode("Less/StreamSwitch_t");
    EXPECT_TRUE(AttrUtils::SetListInt(nt->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_TS, RT_MEMORY_HBM }));
    const auto &nf = graph->FindNode("Less/StreamSwitch_f");
    EXPECT_TRUE(AttrUtils::SetListInt(nf->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_TS, RT_MEMORY_HBM }));
  }
}

void BuildSampleGraph(ComputeGraphPtr &graph, uint32_t &mem_offset) {
  DEF_GRAPH(g1) {
    const auto active_s = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{1});
    const auto less_node = OP_CFG(LESS).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC")
                               .Attr(ATTR_NAME_KERNEL_BIN_ID, "_less_fake_kernel_bin_id");
    CHAIN(NODE("_arg_0", DATA)->NODE("HcomAllreduce", HCOMALLREDUCE)->NODE("Less", less_node));
    CHAIN(NODE("_arg_1", DATA)->NODE("Less")->NODE("Less_Cast", CAST)->CTRL_EDGE()->NODE("Less_StreamActive", active_s));
    CHAIN(NODE("_arg_2", DATA)->NODE("mul", MUL)->EDGE(0, 1)->NODE("add_n", ADDN)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("_arg_3", DATA)->NODE("aipp", AIPP)->NODE("shape", SHAPE)->EDGE(0, 0)->NODE("add_n"));
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 2)->NODE("add_n"));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 3)->NODE("add_n"));
    CHAIN(NODE("_cst_string", CONSTANTOP)->NODE("shape", SHAPE));
    CHAIN(NODE("_arg_0")->NODE("cmo1", "Cmo")->Ctrl()->NODE("mul", MUL));
    CHAIN(NODE("_arg_1")->NODE("cmo2", "Cmo")->Ctrl()->NODE("add_n"));
    CHAIN(NODE("_arg_1")->NODE("mul"));

    const auto switch_f = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_NOT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("Less_Cast")->EDGE(0, 0)->
         NODE("switch_f_input0_memcpy", MEMCPYASYNC)->EDGE(0, 0)->    // 增加memcpy
         NODE("Less/StreamSwitch_f", switch_f));
    CHAIN(NODE("Less/StreamSwitch_Const_f", CONSTANTOP)->EDGE(0, 0)->
          NODE("switch_f_input1_memcpy", MEMCPYASYNC)->EDGE(0, 1)->   // 增加memcopy
          NODE("Less/StreamSwitch_f"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_f"));

    const auto switch_t = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("Less_Cast")->EDGE(0, 0)->
          NODE("switch_t_input0_memcpy", MEMCPYASYNC)->EDGE(0, 0)->  // 增加memcpy
          NODE("Less/StreamSwitch_t", switch_t));
    CHAIN(NODE("Less/StreamSwitch_Const_t", CONSTANTOP)->EDGE(0, 0)->
          NODE("switch_t_input1_memcpy", MEMCPYASYNC)->EDGE(0, 1)->  // 增加memcpy
          NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_t"));

    const auto active_0 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("cond/pow", POW)->NODE("cond/sub", SUB)->
          NODE("merge_input_0_memcpy", MEMCPYASYNC)->CTRL_EDGE()->
          NODE("merge_input_0_active", active_0)->CTRL_EDGE()->
          NODE("cond/merge", STREAMMERGE)->EDGE(0, 2)->
          NODE("add_n"));
    CHAIN(NODE("merge_input_0_memcpy")->EDGE(0, 0)->NODE("cond/merge"));
    CHAIN(NODE("_arg_1")->EDGE(0, 1)->NODE("cond/pow"));
    CHAIN(NODE("Less/StreamSwitch_f")->CTRL_EDGE()->NODE("cond/pow"));
    CHAIN(NODE("_arg_1")->EDGE(0, 0)->NODE("cond/realdiv", REALDIV)->NODE("cond/sub"));
    CHAIN(NODE("_arg_2")->EDGE(0, 1)->NODE("cond/realdiv"));
    CHAIN(NODE("Less/StreamSwitch_f")->CTRL_EDGE()->NODE("cond/realdiv"));

    const auto active_1 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_1")->EDGE(0, 0)->NODE("cond/mul", MUL)->NODE("cond/add", ADD)->
          NODE("merge_input_1_memcpy", MEMCPYASYNC)->CTRL_EDGE()->
          NODE("merge_input_1_active", active_1)->CTRL_EDGE()->
          NODE("cond/merge"));
    CHAIN(NODE("merge_input_1_memcpy")->EDGE(0, 1)->NODE("cond/merge"));
    CHAIN(NODE("_arg_2")->EDGE(0, 1)->NODE("cond/mul"));
    CHAIN(NODE("Less/StreamSwitch_t")->CTRL_EDGE()->NODE("cond/mul"));
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("cond/square", SQUARE)->NODE("cond/add"));
    CHAIN(NODE("Less/StreamSwitch_t")->CTRL_EDGE()->NODE("cond/square"));
  };
  graph = ToComputeGraph(g1);
  graph->SetSessionID(100100);
  graph->SetGraphID(20061);
  {
    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    const auto &var1 = graph->FindNode("var1");
    ASSERT_NE(var1, nullptr);
    var1->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var1->GetOpDescBarePtr()->SetOutputOffset({137438953472U});

    const auto &var2 = graph->FindNode("var2");
    ASSERT_NE(var2, nullptr);
    var2->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var2->GetOpDescBarePtr()->SetOutputOffset({137438956472U});

    const auto &_cst_string = graph->FindNode("_cst_string");
    ASSERT_NE(_cst_string, nullptr);
    _cst_string->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*_cst_string->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    _cst_string->GetOpDescBarePtr()->SetOutputOffset({137438959472U});

    const auto &const_f = graph->FindNode("Less/StreamSwitch_Const_f");
    ASSERT_NE(const_f, nullptr);
    const_f->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*const_f->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    const_f->GetOpDescBarePtr()->SetOutputOffset({137438962472U});

    const auto &const_t = graph->FindNode("Less/StreamSwitch_Const_t");
    ASSERT_NE(var2, nullptr);
    const_t->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*const_t->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    const_t->GetOpDescBarePtr()->SetOutputOffset({137438965472U});
  }
  SetUnknownOpKernel(graph, mem_offset, true);
  {
    const auto &node = graph->FindNode(NODE_NAME_NET_OUTPUT);
    EXPECT_NE(node, nullptr);
    GeTensorDesc input_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
  }
  {
    const auto &no = graph->FindNode("cond/pow");
    EXPECT_TRUE(AttrUtils::SetListInt(no->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, { RT_MEMORY_L1 }));
    EXPECT_TRUE(AttrUtils::SetStr(no->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX"));
    EXPECT_TRUE(AttrUtils::SetInt(no->GetOpDesc(), "_task_ratio", 2));
    const auto &ni = graph->FindNode("cond/sub");
    EXPECT_TRUE(AttrUtils::SetListInt(ni->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_L1, RT_MEMORY_HBM }));
  }

  {
    const auto &no = graph->FindNode("Less_Cast");
    EXPECT_TRUE(AttrUtils::SetListInt(no->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, { RT_MEMORY_TS }));
    const auto &nt = graph->FindNode("Less/StreamSwitch_t");
    EXPECT_TRUE(AttrUtils::SetListInt(nt->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_TS, RT_MEMORY_HBM }));
    const auto &nf = graph->FindNode("Less/StreamSwitch_f");
    EXPECT_TRUE(AttrUtils::SetListInt(nf->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, { RT_MEMORY_TS, RT_MEMORY_HBM }));
  }
}
Status CreateFileConstantBin(const std::string &file_path, size_t weight_data_length) {
  vector<int32_t> data(weight_data_length);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }
  std::ofstream out1(file_path, std::ios::binary);
  GE_ASSERT_TRUE(out1.is_open());
  out1.write(reinterpret_cast<char *>(data.data()), sizeof(int32_t) * data.size());
  out1.close();
  return SUCCESS;
}
/*
 *  file_constant1 -> add -> netoutput
 *                    ^
 *  file_constant2 ---+
 */
void BuildFileConstantGraph(ComputeGraphPtr &graph, GeModelPtr &ge_model, uint32_t &mem_offset) {
  DEF_GRAPH(g1) {
    const auto add_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("file_constant1", FILECONSTANT)->NODE("add1", add_node)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("file_constant2", FILECONSTANT)->NODE("add2", add_node)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
  };
  graph = ToComputeGraph(g1);
  graph->SetSessionID(100100);
  graph->SetGraphID(20061);
  {
    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    const auto &var1 = graph->FindNode("file_constant1");
    ASSERT_NE(var1, nullptr);
    var1->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    AttrUtils::SetStr(var1->GetOpDesc(), ATTR_NAME_FILE_PATH, "file_constant1.bin");
    CreateFileConstantBin("file_constant1.bin", 512);

    const auto &var2 = graph->FindNode("file_constant2");
    ASSERT_NE(var2, nullptr);
    var2->GetOpDescBarePtr()->UpdateOutputDesc(0, output_desc);
    ge::TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    AttrUtils::SetStr(var2->GetOpDesc(), ATTR_NAME_FILE_PATH, "file_constant2.bin");
    CreateFileConstantBin("file_constant2.bin", 512);
  }

  SetUnknownOpKernel(graph, mem_offset, true);
  {
    const auto &node = graph->FindNode(NODE_NAME_NET_OUTPUT);
    EXPECT_NE(node, nullptr);
    GeTensorDesc input_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
  }

  TBEKernelStore tbe_kernel_store;

  InitConstantNode(graph, "file_constant1", 1);
  InitConstantNode(graph, "file_constant2", 1);

  const auto model_task_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef(graph, *model_task_def, "add1");
  InitKernelTaskDef(graph, *model_task_def, "add2");
  InitEndGraphDef(graph, *model_task_def, NODE_NAME_NET_OUTPUT);

  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 32));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetListInt(ge_model, ATTR_MODEL_HUGE_STREAM_LIST, {2}));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 256));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 256));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 128));

  // Serialization GeModel for Save offline model.
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, graph->GetSessionID()));
}

void BuildGraphModel(ComputeGraphPtr &graph, GeModelPtr &ge_model, uint32_t mem_offset) {
  TBEKernelStore tbe_kernel_store;
  CustAICPUKernelStore cpu_kernel_store;

  const std::string cst_string("Hello guys. every thing will be ok.");
  InitConstantNode(graph, "_cst_string", GeTensorDesc(GeShape({1}), FORMAT_ND, DT_STRING), cst_string);
  InitConstantNode(graph, "Less/StreamSwitch_Const_f", 0);
  InitConstantNode(graph, "Less/StreamSwitch_Const_t", 1);

  const auto model_task_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef(graph, *model_task_def, "aipp");
  InitKernelTaskDef(graph, *model_task_def, "shape");

  InitKernelTaskDef_TE(graph, *model_task_def, "Less", tbe_kernel_store);
  InitKernelTaskDef_TE(graph, *model_task_def, "Less_Cast", tbe_kernel_store);
  InitKernelTaskDef_CUST_CPU(graph, *model_task_def, "mul", cpu_kernel_store);

  InitStreamActiveDef(graph, *model_task_def, "Less_StreamActive");

  // 构建Switch前的memcpy
  if (graph->FindNode("switch_f_input0_memcpy")) {
    InitMemcpyAsyncDef(graph, *model_task_def, "switch_f_input0_memcpy");
  }
  if (graph->FindNode("switch_f_input1_memcpy")) {
    InitMemcpyAsyncDef(graph, *model_task_def, "switch_f_input1_memcpy");
  }
  if (graph->FindNode("switch_t_input0_memcpy")) {
    InitMemcpyAsyncDef(graph, *model_task_def, "switch_t_input0_memcpy");
  }
  if (graph->FindNode("switch_t_input1_memcpy")) {
    InitMemcpyAsyncDef(graph, *model_task_def, "switch_t_input1_memcpy");
  }

  InitStreamSwitchDef(graph, *model_task_def, "Less/StreamSwitch_f", 1);
  InitStreamSwitchDef(graph, *model_task_def, "Less/StreamSwitch_t", 1);

  InitKernelExTaskDef(graph, *model_task_def, "cond/pow");
  InitKernelExTaskDef_AllShape(graph, *model_task_def, "cond/sub");
  InitKernelExTaskDef_Blocking(graph, *model_task_def, "cond/realdiv");

  InitMemcpyAsyncDef(graph, *model_task_def, "merge_input_0_memcpy");
  InitStreamActiveDef(graph, *model_task_def, "merge_input_0_active");
  InitMemcpyAsyncDef(graph, *model_task_def, "merge_input_1_memcpy");
  InitStreamActiveDef(graph, *model_task_def, "merge_input_1_active");

  InitKernelTaskDef_AI_CPU(graph, *model_task_def, "cond/mul");
  InitKernelTaskDef_CPU_AllShape(graph, *model_task_def, "cond/add");
  InitKernelTaskDef_CPU_Blocking(graph, *model_task_def, "cond/square");

  InitMemcpyAddrAsyncDef(graph, *model_task_def, "cond/merge", 2);
  InitKernelTaskDef(graph, *model_task_def, "add_n");

  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, NODE_NAME_NET_OUTPUT);

  InitHcclTaskDef(graph, *model_task_def, "HcomAllreduce", "HcomBroadcast");
  InitProfilerTaskDef(graph, *model_task_def);
  InitCmoTaskDef(graph, *model_task_def);
  InitCmoAddrTaskDef(graph, *model_task_def, "cmo1");
  InitCmoAddrTaskDef(graph, *model_task_def, "cmo2", 10);
  InitCmoBarrierTaskDef(graph, *model_task_def);

  InitNpuGetFloatStatusTaskDef(graph, *model_task_def, NODE_NAME_NET_OUTPUT);
  InitNpuClearFloatStatusTaskDef(graph, *model_task_def, NODE_NAME_NET_OUTPUT);

  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 32));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 10240));

  EXPECT_TRUE(AttrUtils::SetListInt(ge_model, ATTR_MODEL_HUGE_STREAM_LIST, {2}));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 256));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 256));

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 128));

  // Serialization GeModel for Save offline model.
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_TRUE(cpu_kernel_store.Build());
  ge_model->SetCustAICPUKernelStore(cpu_kernel_store);
  EXPECT_TRUE(AttrUtils::SetListStr(ge_model, "needCheckCpu", { "aicpu_optype_01", "aicpu_optype_02" }));
  EXPECT_TRUE(AttrUtils::SetListStr(ge_model, "needCheckTf", { "aicpu_tf_optype_01", "aicpu_tf_optype_02" }));

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, graph->GetSessionID()));
}

static void ProfileCommandInit(GeExecutor &ge_executor) {
  actual_info_type.clear();
  ProfilingProperties::Instance().ClearProperties();
  {
    Command command{"prof_init", {}, PROF_MODEL_LOAD_MASK | PROF_TRAINING_TRACE_MASK};
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }
}

static void ProfileCommandProf(GeExecutor &ge_executor, const uint32_t model_id) {
  {
    Command command{.cmd_type = "prof_start",
                    .cmd_params = { "devNums", "1", "devIdList", "0", PROFILE_MODEL_ID, std::to_string(model_id) },
                    .module_index = PROF_TRAINING_TRACE_MASK | PROF_OP_DETAIL_MASK };
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }

  {
    Command command{.cmd_type = "prof_start",
                    .cmd_params = { "heterogeneous_host", "1"}};
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }

  {
    Command command{.cmd_type = "prof_model_subscribe",
                    .cmd_params = { PROFILE_MODEL_ID, std::to_string(model_id) },
                    .module_index = PROF_MODEL_LOAD_MASK | PROF_TRAINING_TRACE_MASK | PROF_OP_DETAIL_MASK };
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }
}

static void ProfileCommandFini(GeExecutor &ge_executor, const uint32_t model_id) {
  {
    Command command{.cmd_type = "prof_stop",
                    .cmd_params = { "devNums", "1", "devIdList", "0", PROFILE_MODEL_ID, std::to_string(model_id) },
                    .module_index = PROF_MODEL_LOAD_MASK | PROF_TRAINING_TRACE_MASK | PROF_OP_DETAIL_MASK };
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }

  {
    Command command{.cmd_type = "prof_stop",
                    .cmd_params = { "heterogeneous_host", "1"}};
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }

  {
    Command command{.cmd_type = "prof_model_cancel_subscribe",
                    .cmd_params = { PROFILE_MODEL_ID, std::to_string(model_id) }};
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }

  {
    Command command{.cmd_type = "prof_finalize"};
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }
}

static void DumpCommandInit(GeExecutor &ge_executor) {
  {
    DumpConfig dump_cfg;
    dump_cfg.dump_path = "./dump/";
    dump_cfg.dump_mode = "output";
    dump_cfg.dump_status = "on";
    dump_cfg.dump_op_switch = "on";
    ModelDumpConfig model_dump_config;
    model_dump_config.model_name = "g1_om";
    model_dump_config.layers.emplace_back("Less_Cast");
    model_dump_config.layers.emplace_back("cond/add");
    model_dump_config.layers.emplace_back("cond/mul");
    dump_cfg.dump_list.emplace_back(model_dump_config);
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
  }
}

static void DumpCommandFini(GeExecutor &ge_executor) {
  {
    DumpConfig dump_cfg;
    dump_cfg.dump_path = "./dump/";
    dump_cfg.dump_mode = "output";
    dump_cfg.dump_status = "off";
    dump_cfg.dump_op_switch = "off";
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
  }
}

static void ModelDumpInitCmd(GeExecutor &ge_executor) {
  {
    Command command{.cmd_type = "dump",
                    .cmd_params = {DUMP_STATUS, "on", DUMP_MODEL, "g1_om", DUMP_FILE_PATH, "/tmp", DUMP_MODE, "all"} };
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }
}

static void ModelDumpFiniCmd(GeExecutor &ge_executor) {
  {
    Command command{.cmd_type = "dump",
                    .cmd_params = {DUMP_STATUS, "off", DUMP_MODEL, "g1_om", DUMP_FILE_PATH, "/tmp", DUMP_MODE, "all"} };
    EXPECT_EQ(ge_executor.CommandHandle(command), SUCCESS);
  }
}

void OfflineModelCommand(GeExecutor &ge_executor, const uint32_t model_id) {
  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = sizeof(uint64_t); uint64_t batch_size = 0U;
    ge_executor.SetDynamicBatchSize(model_id, &dynamic_input_addr, length, batch_size);
  }

  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = sizeof(uint64_t); uint64_t image_height = 0U; uint64_t image_width = 0U;
    ge_executor.SetDynamicImageSize(model_id, &dynamic_input_addr, length, image_height, image_width);
  }

  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = sizeof(uint64_t); std::vector<uint64_t> dynamic_dims;
    ge_executor.SetDynamicDims(model_id, &dynamic_input_addr, length, dynamic_dims);
  }

  {
    std::vector<uint64_t> dynamic_dims; std::vector<uint64_t> cur_dynamic_dims;
    ge_executor.GetCurDynamicDims(model_id, dynamic_dims, cur_dynamic_dims);
  }

  {
    std::vector<int64_t> batch_info; int32_t dynamic_type = 0U;
    ge_executor.GetCurShape(model_id, batch_info, dynamic_type);
  }

  {
    uint64_t dynamic_input_addr = 0U; uint64_t length = 0U; std::vector<kAippDynamicBatchPara> aipp_batch_para; kAippDynamicPara aipp_parms;
    ge_executor.SetDynamicAippData(model_id, &dynamic_input_addr, length, aipp_batch_para, aipp_parms);
  }

  {
    std::vector<TensorDesc> input_desc; std::vector<TensorDesc> output_desc; bool new_model_desc = false;
    ge_executor.GetModelDescInfo(model_id, input_desc, output_desc, new_model_desc);
  }

  {
    std::vector<std::vector<int64_t>> batch_info; int32_t dynamic_type = 0U;
    ge_executor.GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  }

  {
    std::vector<std::vector<int64_t>> batch_info;
    ge_executor.GetCombinedDynamicDims(model_id, batch_info);
  }

  {
    std::vector<std::string> user_designate_shape_order;
    ge_executor.GetUserDesignateShapeOrder(model_id, user_designate_shape_order);
  }

  {
    uint32_t index = 0U; AippConfigInfo aipp_info;
    ge_executor.GetAIPPInfo(model_id, index, aipp_info);
  }

  {
    uint32_t index = 0U; InputAippType type; size_t aipp_index = 0U;
    ge_executor.GetAippType(model_id, index, type, aipp_index);
  }

  {
    std::string op_name; std::string attr_name; std::string attr_value;
    ge_executor.GetOpAttr(model_id, op_name, attr_name, attr_value);
  }

  {
    std::vector<std::string> dynamic_output_shape_info;
    ge_executor.GetModelAttr(model_id, dynamic_output_shape_info);
  }

  {
    uint32_t max_size = 0U;
    ge_executor.GetMaxUsedMemory(model_id, max_size);
  }

  {
    uint32_t device_id = 0U;
    GeExecutor::GetDeviceIdByModelId(model_id, device_id) ;
  }

  {
    size_t shape_count = 0U;
    ge_executor.GetBatchInfoSize(model_id, shape_count);
  }

  {
    uint32_t index = 0U; OriginInputInfo orig_input_info;
    ge_executor.GetOrigInputInfo(model_id, index, orig_input_info);
  }

  {
    uint32_t index = 0U; std::vector<InputOutputDims> input_dims; std::vector<InputOutputDims> output_dims;
    ge_executor.GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  }

  {
    uint32_t device_id = 0U; uint32_t stream_id = 0U; uint32_t task_id = 0U; OpDescInfo op_desc_info;
    ge_executor.GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
  }
}

static void BuildDvppGraph(ComputeGraphPtr &root_graph) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
    CHAIN(NODE("_arg_2", DATA)->NODE("PartitionedCall_0"));
  };
  root_graph = ToComputeGraph(g1);
  SetUnknownOpKernel(root_graph, mem_offset, true);
}

TEST_F(GeExecutorTest, dvpp_graph) {
  ComputeGraphPtr root_graph;
  BuildDvppGraph(root_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  EXPECT_NE(root_graph, nullptr);

  InitDvppTaskDef(root_graph, *model_task_def, "PartitionedCall_0");

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(GeExecutorTest, ge_executor_not_inited) {
  GeExecutor::is_inited_ = false;

  std::string path;
  ModelData model_data;
  uint32_t model_id = 0U;
  int32_t dynamic_type = 0;
  size_t mem_size = 0U;
  size_t weight_size = 0U;
  size_t model_size = 1024U;
  uint32_t max_size = 0U;
  {
    std::vector<int64_t> batch_info;
    EXPECT_EQ(ge_executor_.GetCurShape(model_id, batch_info, dynamic_type), ACL_ERROR_GE_EXEC_NOT_INIT);
  }

  {
    std::vector<TensorDesc> input_desc;
    std::vector<TensorDesc> output_desc;
    EXPECT_EQ(ge_executor_.GetModelDescInfo(model_id, input_desc, output_desc, true), ACL_ERROR_GE_EXEC_NOT_INIT);
  }

  std::vector<std::vector<int64_t>> batch_info;
  EXPECT_EQ(ge_executor_.GetDynamicBatchInfo(model_id, batch_info, dynamic_type), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(ge_executor_.GetCombinedDynamicDims(model_id, batch_info), ACL_ERROR_GE_EXEC_NOT_INIT);
  std::vector<std::string> user_designate_shape_order;
  EXPECT_EQ(ge_executor_.GetUserDesignateShapeOrder(model_id, user_designate_shape_order), ACL_ERROR_GE_EXEC_NOT_INIT);

  AippConfigInfo aipp_info;
  EXPECT_EQ(ge_executor_.GetAIPPInfo(model_id, 0U, aipp_info), ACL_ERROR_GE_EXEC_NOT_INIT);
  InputAippType type;
  size_t aipp_index = 0U;
  EXPECT_EQ(ge_executor_.GetAippType(model_id, 0U, type, aipp_index), ACL_ERROR_GE_EXEC_NOT_INIT);

  std::string op_name;
  std::string attr_name;
  std::string attr_value;
  EXPECT_EQ(ge_executor_.GetOpAttr(model_id, op_name, attr_name, attr_value), ACL_ERROR_GE_EXEC_NOT_INIT);
  std::vector<std::string> dynamic_output_shape_info;
  EXPECT_EQ(ge_executor_.GetModelAttr(model_id, dynamic_output_shape_info), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(ge_executor_.GetMaxUsedMemory(model_id, max_size), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(ge_executor_.LoadDataFromFile(path, model_data), ACL_ERROR_GE_EXEC_NOT_INIT);

  EXPECT_EQ(ge_executor_.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), ACL_ERROR_GE_EXEC_NOT_INIT);

  gert::RtSession session(199);
  session.SetExternalVar(nullptr, 0);
  void *external_var = &model_id;
  uint64_t var_size = 99;
  session.GetExternalVar(external_var, var_size);
  EXPECT_EQ(external_var, nullptr);
  EXPECT_EQ(var_size, 0);
  ge::ModelLoadArg load_arg;
  load_arg.dev_ptr = nullptr;
  load_arg.mem_size = 0;
  load_arg.weight_ptr = nullptr;
  load_arg.rt_session = &session;
  EXPECT_EQ(ge_executor_.LoadModelFromDataWithArgs(model_id, model_data, load_arg), ACL_ERROR_GE_EXEC_NOT_INIT);

  std::vector<uint32_t> input_queue_ids;
  std::vector<uint32_t> output_queue_ids;
  EXPECT_EQ(ge_executor_.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), ACL_ERROR_GE_EXEC_NOT_INIT);

  RunModelData run_input_data;
  RunModelData run_output_data;
  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;
  EXPECT_EQ(ge_executor_.ExecModel(model_id, nullptr, run_input_data, input_desc, run_output_data, output_desc, true), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(ge_executor_.GetMemAndWeightSize(path, mem_size, weight_size), ACL_ERROR_GE_EXEC_NOT_INIT);

  EXPECT_EQ(ge_executor_.GetMemAndWeightSize(nullptr, model_size, mem_size, weight_size), ACL_ERROR_GE_EXEC_NOT_INIT);
  OriginInputInfo orig_input_info;
  EXPECT_EQ(ge_executor_.GetOrigInputInfo(model_id, 0U, orig_input_info), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(ge_executor_.GetMemAndWeightSize(nullptr, model_size, mem_size, weight_size), ACL_ERROR_GE_EXEC_NOT_INIT);

  std::vector<InputOutputDims> input_dims;
  std::vector<InputOutputDims> output_dims;
  EXPECT_EQ(ge_executor_.GetAllAippInputOutputDims(model_id, 0U, input_dims, output_dims), ACL_ERROR_GE_EXEC_NOT_INIT);
}

TEST_F(GeExecutorTest, sample_davinci_model_static_memory) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);
  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);
  std::vector<gert::Tensor> input_tensors(4);
  TensorCheckUtils::ConstructGertTensor(input_tensors[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[3], {1}, DT_INT64, FORMAT_ND);

  // Tensor for input.
  std::vector<gert::Tensor> sync_inputs(4);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;
  DumpCommandInit(ge_executor_);
  gert::RtVarManagerPool().Instance().RemoveRtVarManager(graph->GetSessionID());
  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->IncreaseLoadCount();

    // Callback for execute.
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
    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption({{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"},
                            {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"},
                            {"ge.exec.hostInputIndexes", "0"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(input_tensors);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    domi::GetContext().is_online_model = true;
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({{VARIABLE_MEMORY_MAX_SIZE, "12800"}}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);
    model_executor.StartRunThread();

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(run_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());

    EXPECT_EQ(ModelManager::GetInstance().ClearAicpuSo(), SUCCESS);
    EXPECT_TRUE(gert::RtVarManagerPool().Instance().session_id_to_var_manager_.empty());
    // Unload model of graph.
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  // ModelExecutor::RunGraph -> ExecuteGraph -> SyncExecuteModel -> ModelManager::DataInput -> DavinciModel::Push -> Run
  {
    // Test LoadModelOnline: GraphModelListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(false);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 960)); // set max stream

    // Env for load:.ModelManager::CheckAndReleaseStreamEventResource
    GEThreadLocalContext &context = GetThreadLocalContext();
    context.SetGraphOption({{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"},
                            {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});

    // profiling model subscribe on
    ProfilingProperties::Instance().SetSubscribeInfo(0, graph->GetGraphID(), true);

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // Execute Synchronous
    std::vector<gert::Tensor> sync_outputs;
    EXPECT_EQ(model_executor.RunGraph(graph_node, graph->GetGraphID(), sync_inputs, sync_outputs), SUCCESS);
    EXPECT_EQ(sync_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());
    // check reported graph id saved
    EXPECT_TRUE(ProfilingManager::Instance().IsGraphProfReported(graph->GetGraphID()));

    // clear profiling configurations
    ProfilingProperties::Instance().subscribe_count_--;
    ProfilingManager::Instance().ProfFinalize();
    EXPECT_TRUE(gert::RtVarManagerPool().Instance().session_id_to_var_manager_.empty());
    // Unload model of graph(leave as max stream model for follow test)
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  {
    // Test LoadModelOnline: RunGraphWithStream
    const auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    ge_root_model->SetIsSpecificStream(true); // For not start DavinciModel thread.
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 960));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));

    int64_t value_0 = 127;
    int64_t value_1 = 100;
    int64_t value_2 = 258;
    int64_t value_3 = 512;
    // Tensor for input.
    GeTensorDesc sync_tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
    GeTensor sync_tensor_0(sync_tensor_desc, (uint8_t *)&value_0, sizeof(value_0));
    GeTensor sync_tensor_1(sync_tensor_desc, (uint8_t *)&value_1, sizeof(value_1));
    GeTensor sync_tensor_2(sync_tensor_desc, (uint8_t *)&value_2, sizeof(value_2));
    GeTensor sync_tensor_3(sync_tensor_desc, (uint8_t *)&value_3, sizeof(value_3));
    const std::vector<GeTensor> sync_inputs{ sync_tensor_0, sync_tensor_1, sync_tensor_2, sync_tensor_3 };

    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    std::vector<uint8_t> arg_3(512, 0);  // mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    GeTensor nn_tensor_21(output_desc, arg_3.data(), arg_3.size());
    std::vector<GeTensor> nn_outputs{ nn_tensor_21 };

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // NnExecute with stream.
    rtStream_t run_stream = &model_executor;
    EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph->GetGraphID(), run_stream, sync_inputs, nn_outputs), SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());
    EXPECT_TRUE(gert::RtVarManagerPool().Instance().session_id_to_var_manager_.empty());
    // Unload model of graph(leave as max event model for follow test).
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  std::vector<gert::Tensor> input_tensors2(4);
  TensorCheckUtils::ConstructGertTensor(input_tensors2[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors2[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors2[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors2[3], {1}, DT_INT64, FORMAT_ND);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 32));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));
  {
    // Test LoadModelOnline: for SuperKernel
    const uint64_t session_id = graph->GetSessionID();
    DumpProperties dump_properties;
    dump_properties.SetDumpMode("output");
    dump_properties.AddPropertyValue(DUMP_LAYER_OP_MODEL, {"Less"});
    dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"aipp"});
    DumpManager::GetInstance().RemoveDumpProperties(session_id);
    DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->IncreaseLoadCount();

    // Setup SuperKernel
    EXPECT_TRUE(AttrUtils::SetBool(ge_model, ATTR_NAME_SWITCH_FOR_L1_FUSION, true));
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetOpDesc()->GetOpKernelLibName() == "AIcoreEngine") {
        EXPECT_TRUE(AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_FUSION_GROUP_KEY, 1));
      }
    }

    // Callback for execute.
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

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(input_tensors2);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);
    model_executor.StartRunThread();

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(run_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());
    EXPECT_TRUE(gert::RtVarManagerPool().Instance().session_id_to_var_manager_.empty());
    // Unload model of graph.
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
    DumpManager::GetInstance().RemoveDumpProperties(session_id);
  }

  DumpCommandFini(ge_executor_);
  for (const auto &id : model_ids) {
    EXPECT_EQ(ge_executor_.UnloadModel(id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  }
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(GeExecutorTest, sample_davinci_model_recover_single_model) {
  // 构造模型 入图
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 1);
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);
  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetListStr(ge_model, "needCheckCpu", {}));
  EXPECT_TRUE(AttrUtils::SetListStr(ge_model, "needCheckTf", {}));

  // Tensor for input.
  int64_t value_0 = 127;
  int64_t value_1 = 100;
  int64_t value_2 = 258;
  int64_t value_3 = 512;

  GeTensorDesc sync_tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
  GeTensor sync_tensor_0(sync_tensor_desc, (uint8_t *)&value_0, sizeof(value_0));
  GeTensor sync_tensor_1(sync_tensor_desc, (uint8_t *)&value_1, sizeof(value_1));
  GeTensor sync_tensor_2(sync_tensor_desc, (uint8_t *)&value_2, sizeof(value_2));
  GeTensor sync_tensor_3(sync_tensor_desc, (uint8_t *)&value_3, sizeof(value_3));
  const std::vector<GeTensor> sync_inputs{ sync_tensor_0, sync_tensor_1, sync_tensor_2, sync_tensor_3 };

  std::vector<uint32_t> model_ids;
  DumpCommandInit(ge_executor_);
  gert::RtVarManagerPool().Instance().RemoveRtVarManager(graph->GetSessionID());
  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);

  // 模型执行
  {
    // Test LoadModelOnline: RunGraphWithStream
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    ge_root_model->SetIsSpecificStream(true); // For not start DavinciModel thread.

    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 960));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));

    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    std::vector<uint8_t> arg_3(512, 0);  // mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    GeTensor nn_tensor_21(output_desc, arg_3.data(), arg_3.size());
    std::vector<GeTensor> nn_outputs{ nn_tensor_21 };

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // NnExecute with stream.
    rtStream_t run_stream = &model_executor;
    EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph->GetGraphID(), run_stream, sync_inputs, nn_outputs), SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());
    EXPECT_TRUE(gert::RtVarManagerPool().Instance().session_id_to_var_manager_.empty());

   auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
   for (auto stream : all_rt_streams) {
     cout << "stream handle " << ((int64_t)stream) << endl;
   }

    // 调用recover, 内部会校验taskid
   EXPECT_EQ(ge_executor_.RecoverAllModel(0), SUCCESS);

   // 重新执行正确
   EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph->GetGraphID(), run_stream, sync_inputs, nn_outputs), SUCCESS);

    // Unload model of graph(leave as max event model for follow test).
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(GeExecutorTest, sample_davinci_model_lora_format_changed) {
  dlog_setlevel(0, 0, 0);
  VarManagerPool::Instance().Destory();
  int64_t value_0 = 127;
  int64_t value_1 = 100;
  int64_t value_2 = 258;
  int64_t value_3 = 512;
  // Tensor for input.
  TensorDesc tensor_desc(Shape(), FORMAT_ND, DT_INT64);
  Tensor tensor_0(tensor_desc, (uint8_t *)&value_0, sizeof(value_0));
  Tensor tensor_1(tensor_desc, (uint8_t *)&value_1, sizeof(value_1));
  Tensor tensor_2(tensor_desc, (uint8_t *)&value_2, sizeof(value_2));
  Tensor tensor_3(tensor_desc, (uint8_t *)&value_3, sizeof(value_3));
  const std::vector<Tensor> input_tensors{tensor_0, tensor_1, tensor_2, tensor_3};

  // Tensor for input.
  GeTensorDesc sync_tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
  GeTensor sync_tensor_0(sync_tensor_desc, (uint8_t *)&value_0, sizeof(value_0));
  GeTensor sync_tensor_1(sync_tensor_desc, (uint8_t *)&value_1, sizeof(value_1));
  GeTensor sync_tensor_2(sync_tensor_desc, (uint8_t *)&value_2, sizeof(value_2));
  GeTensor sync_tensor_3(sync_tensor_desc, (uint8_t *)&value_3, sizeof(value_3));
  const std::vector<GeTensor> sync_inputs{sync_tensor_0, sync_tensor_1, sync_tensor_2, sync_tensor_3};

  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);
  // model 1 with format FZ
  {
    GeModelPtr ge_model;
    BuildGraphModel(graph, ge_model, mem_offset);
    EXPECT_NE(ge_model, nullptr);
    // Test LoadModelOnline: RunGraphWithStream
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    ge_root_model->SetIsSpecificStream(true);  // For not start DavinciModel thread.
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 960));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));

    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    std::vector<uint8_t> arg_3(512, 0);  // mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    GeTensor nn_tensor_21(output_desc, arg_3.data(), arg_3.size());
    std::vector<GeTensor> nn_outputs{nn_tensor_21};

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // NnExecute with stream.
    rtStream_t run_stream = &model_executor;
    EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph->GetGraphID(), run_stream, sync_inputs, nn_outputs),
              SUCCESS);

    // Unload model of graph(leave as max event model for follow test).
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  // model 2 with format ND
  {
    GeTensorDesc new_tensor(GeShape({2, 4, 8, 2}), FORMAT_ND, DT_FLOAT);
    const auto &var1 = graph->FindNode("var1");
    ASSERT_NE(var1, nullptr);
    var1->GetOpDescBarePtr()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);

    const auto &var2 = graph->FindNode("var2");
    ASSERT_NE(var2, nullptr);
    var2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);

    GeModelPtr ge_model;
    BuildGraphModel(graph, ge_model, mem_offset);
    EXPECT_NE(ge_model, nullptr);
    // Test LoadModelOnline: RunGraphWithStream
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    ge_root_model->SetIsSpecificStream(true);  // For not start DavinciModel thread.
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 960));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));

    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    std::vector<uint8_t> arg_3(512, 0);  // mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    GeTensor nn_tensor_21(output_desc, arg_3.data(), arg_3.size());
    std::vector<GeTensor> nn_outputs{nn_tensor_21};

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // NnExecute with stream.
    rtStream_t run_stream = &model_executor;
    EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph->GetGraphID(), run_stream, sync_inputs, nn_outputs),
              SUCCESS);

    // Unload model of graph(leave as max event model for follow test).
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  dlog_setlevel(3, 3, 0);
}

TEST_F(GeExecutorTest, sample_davinci_model_invalid_input) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> inputs(3);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;
  DumpCommandInit(ge_executor_);
  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);

  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };
    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption({{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"},
                            {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    domi::GetContext().is_online_model = true;
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({{VARIABLE_MEMORY_MAX_SIZE, "12800"}}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);
    model_executor.StartRunThread();

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_NE(run_status, SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph.
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  DumpCommandFini(ge_executor_);
  for (const auto &id : model_ids) {
    EXPECT_EQ(ge_executor_.UnloadModel(id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  }
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(GeExecutorTest, sample_davinci_model_dynamic_memory) {
  dlog_setlevel(-1, 0, 1);
  {
    auto mock_runtime = std::make_shared<MockMemRuntime>();
    ge::RuntimeStub::SetInstance(mock_runtime);
    gert::memory::RtsCachingMemAllocator::GetAllocator(0U, RT_MEMORY_HBM);
    ge::RuntimeStub::Reset();
  }

  shared_ptr<OpsKernelInfoStore> fake_ops_kernel_info_store = std::make_shared<FakeOpsKernelInfoStore>();
  // hccl op goes to AIcoreEngine in this testcase
  OpsKernelExecutorManager::GetInstance().executors_["AIcoreEngine"] = fake_ops_kernel_info_store;
  OpsKernelExecutorManager::GetInstance().executors_[kEngineNameHccl] = fake_ops_kernel_info_store;
  OpsKernelInfoStore *ptr = nullptr;
  EXPECT_EQ(OpsKernelExecutorManager::GetInstance().GetExecutor(kEngineNameHccl, ptr), SUCCESS);
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> inputs(4);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[3], {1}, DT_INT64, FORMAT_ND);

  // Tensor for input.
  std::vector<gert::Tensor> sync_inputs(4);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[3], {1}, DT_INT64, FORMAT_ND);

  int64_t value_0 = 127;
  int64_t value_1 = 100;
  int64_t value_2 = 258;
  int64_t value_3 = 512;
  RunModelData run_input_data;
  run_input_data.blobs.emplace_back(DataBuffer{&value_0, sizeof(value_0), false, 0});
  run_input_data.blobs.emplace_back(DataBuffer{&value_1, sizeof(value_1), false, 0});
  run_input_data.blobs.emplace_back(DataBuffer{&value_2, sizeof(value_2), false, 0});
  run_input_data.blobs.emplace_back(DataBuffer{&value_3, sizeof(value_3), false, 0});

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, graph->GetSessionID()), SUCCESS);
  std::vector<uint32_t> model_ids;
  ModelDumpInitCmd(ge_executor_);

  // ModelExecutor::RunGraph -> ExecuteGraph -> SyncExecuteModel -> ModelManager::DataInput -> DavinciModel::Push -> Run
  {
    // Test LoadModelOnline: GraphModelListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(false);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 256 * 1024 * 1024));
    InitAippNodeRelated(graph, "_arg_3", "_arg_2");

    // Load model of graph
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Execute Synchronous
    std::vector<gert::Tensor> sync_outputs;
    EXPECT_EQ(model_executor.RunGraph(graph_node, graph->GetGraphID(), sync_inputs, sync_outputs), SUCCESS);
    EXPECT_EQ(sync_outputs.size(), 1U);

    // Unload model of graph(leave as max memory model for follow test)
    // EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    CleanAippNodeInfo(graph, "_arg_3");
  }

  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->IncreaseLoadCount();
    const size_t k512MegaBytes = 512 * 1024 * 1024;
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, k512MegaBytes)); // Will unload last model.
    InitAippNodeStatic(graph, "_arg_3");

    // Callback for execute.
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

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    VarManager::Instance(graph->GetSessionID())->UpdateMemoryConfig(k512MegaBytes, k512MegaBytes, k512MegaBytes, k512MegaBytes);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(run_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    CleanAippNodeInfo(graph, "_arg_3");
  }
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  InitAippNodeDynamic(graph, "_arg_3");
  DelStaticForOffline(graph, mem_offset); // Offline model will set new session_id, static var invalid.
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  {
    auto var1 = ge_model->GetGraph()->FindNode("var1");
    auto var2 = ge_model->GetGraph()->FindNode("var2");
    auto var3 = ge_model->GetGraph()->FindNode("_cst_string");
    auto var4 = ge_model->GetGraph()->FindNode("Less/StreamSwitch_Const_f");
    auto var5 = ge_model->GetGraph()->FindNode("Less/StreamSwitch_Const_t");
    ASSERT_NE(var1, nullptr);
    ASSERT_NE(var2, nullptr);
    ge::TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var1->GetOpDescBarePtr()->SetOutputOffset({137438953472U});
    ge::TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var2->GetOpDescBarePtr()->SetOutputOffset({137438954472U});
    ge::TensorUtils::SetSize(*var3->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var3->GetOpDescBarePtr()->SetOutputOffset({137438954472U});
    ge::TensorUtils::SetSize(*var4->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var4->GetOpDescBarePtr()->SetOutputOffset({137438954472U});
    ge::TensorUtils::SetSize(*var5->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var5->GetOpDescBarePtr()->SetOutputOffset({137438954472U});

    // Test LoadModelOffline
    ModelHelper model_helper;
    model_helper.SetSaveMode(true);  // Save to file.
    ModelBufferData model_buffer;
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "sample_offline_model.om", model_buffer), SUCCESS);
    EXPECT_TRUE(AttrUtils::SetStr(*(ge_model.get()), "soc_version", "Ascend910")); // check soc_version success
    EXPECT_TRUE(AttrUtils::SetStr(*(ge_model.get()), "arch_type", "1")); // check arch_type success

    size_t model_mem_size = 0U; size_t model_weight_size = 0U;
    EXPECT_EQ(ge_executor_.GetMemAndWeightSize("sample_offline_model.om", model_mem_size, model_weight_size), SUCCESS);

    ModelData model_data;
    EXPECT_EQ(ge_executor_.LoadDataFromFile("sample_offline_model.om", model_data), SUCCESS);
    model_data.om_name = "g1_om";
    {
      size_t mem_size = 0U; size_t weight_size = 0U;
      EXPECT_EQ(ge_executor_.GetMemAndWeightSize(model_data.model_data, model_data.model_len, mem_size, weight_size), SUCCESS);
      EXPECT_TRUE(model_mem_size == mem_size);
      EXPECT_EQ(model_weight_size, weight_size);
    }

    std::vector<uint8_t> out_0(512, 0);  // mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    RunModelData run_output_data;
    run_output_data.blobs.emplace_back(DataBuffer{out_0.data(), out_0.size(), false, 0});
    {
      uint32_t model_id = 0U;
      ProfileCommandInit(ge_executor_);
      EXPECT_EQ(ge_executor_.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
      ProfileCommandProf(ge_executor_, model_id);
      model_ids.emplace_back(model_id);

      // Run: Asynchronize
      EXPECT_EQ(ge_executor_.ExecModel(model_id, nullptr, run_input_data, run_output_data, true), SUCCESS);

      EXPECT_EQ(actual_info_type.find("id_map_info"), actual_info_type.end());
      for (auto &info : actual_info_type) {
        const static std::set<std::string> expect_info_type{
           "task_desc_info", "tensor_data_info", "model_load_info", "fusion_op_info", "step_info", "model_time_info"
        };
        EXPECT_NE(expect_info_type.find(info.substr(0, info.find("info") + strlen("info"))), expect_info_type.end());
      }

      ProfileCommandFini(ge_executor_, model_id);
      EXPECT_EQ(ge_executor_.UnloadModel(model_id), SUCCESS);
    }

    {
      uint32_t model_id = 0U;
      ProfileCommandInit(ge_executor_);
      uint32_t session_id_tmp = 199U;
      gert::RtSession session(session_id_tmp);
      ge::ModelLoadArg load_arg;
      load_arg.dev_ptr = nullptr;
      load_arg.mem_size = 0;
      load_arg.weight_ptr = nullptr;
      load_arg.rt_session = &session;
      EXPECT_EQ(ge_executor_.LoadModelFromDataWithArgs(model_id, model_data, load_arg), SUCCESS);
      ProfileCommandProf(ge_executor_, model_id);
      model_ids.emplace_back(model_id);

      // Run: Asynchronize
      EXPECT_EQ(ge_executor_.ExecModel(model_id, nullptr, run_input_data, run_output_data, true), SUCCESS);

      EXPECT_EQ(actual_info_type.find("id_map_info"), actual_info_type.end());
      for (auto &info : actual_info_type) {
        const static std::set<std::string> expect_info_type{
            "task_desc_info", "tensor_data_info", "model_load_info", "fusion_op_info", "step_info", "model_time_info"
        };
        EXPECT_NE(expect_info_type.find(info.substr(0, info.find("info") + strlen("info"))), expect_info_type.end());
      }

      ProfileCommandFini(ge_executor_, model_id);
      EXPECT_EQ(ge_executor_.UnloadModel(model_id), SUCCESS);
      EXPECT_TRUE(VarManager::Instance(session_id_tmp)->IsVarExist("var1"));
      session.DestroyResources();
      EXPECT_FALSE(VarManager::Instance(session_id_tmp)->IsVarExist("var1"));
    }


    {
      unsetenv("GE_PROFILING_TO_STD_OUT");
      uint32_t model_id = 0U;
      ProfileCommandInit(ge_executor_);
      EXPECT_EQ(ge_executor_.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
      ProfileCommandProf(ge_executor_, model_id);
      model_ids.emplace_back(model_id);

      // Run: Synchronize, customer stream
      rtStream_t run_stream = &model_id;
      EXPECT_EQ(ge_executor_.ExecModel(model_id, run_stream, run_input_data, run_output_data, false), SUCCESS);

      EXPECT_EQ(actual_info_type.find("id_map_info"), actual_info_type.end());
      for (auto &info : actual_info_type) {
        const static std::set<std::string> expect_info_type{
            "task_desc_info", "tensor_data_info", "model_load_info", "fusion_op_info", "step_info", "model_time_info"
        };
        EXPECT_NE(expect_info_type.find(info.substr(0, info.find("info") + strlen("info"))), expect_info_type.end());
      }

      ProfileCommandFini(ge_executor_, model_id);
      OfflineModelCommand(ge_executor_, model_id);
      auto davinci_model = ModelManager::GetInstance().GetModel(model_id);
      EXPECT_NE(davinci_model, nullptr);
      uint32_t session_id = davinci_model->GetSessionId();
      EXPECT_EQ(ge_executor_.UnloadModel(model_id), SUCCESS);
      EXPECT_FALSE(VarManager::Instance(session_id)->IsVarExist("var1"));
      setenv("GE_PROFILING_TO_STD_OUT", "1", 1); // Reset for it`s set in main.
    }

    {
      // Test LoadModelWithQ
      uint32_t model_id = 0;
      const std::vector<uint32_t> input_queue_ids{ 1001U, 1002U, 1003U, 1004U };
      const std::vector<uint32_t> output_queue_ids{ 2001U };

      EXPECT_EQ(ge_executor_.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), SUCCESS);
      model_ids.emplace_back(model_id);

      EXPECT_EQ(ge_executor_.UnloadModel(model_id), SUCCESS);
    }

    delete [] static_cast<uint8_t *>(model_data.model_data);
  }

  {
    auto mock_runtime = std::make_shared<MockMemRuntime>();
    ge::RuntimeStub::SetInstance(mock_runtime);

    ModelHelper model_helper;
    model_helper.SetSaveMode(true);  // Save to file.
    ModelBufferData model_buffer;
    EXPECT_TRUE(AttrUtils::SetStr(*(ge_model.get()), "soc_version", "Ascend310")); // check soc_version fail
    EXPECT_TRUE(AttrUtils::SetStr(*(ge_model.get()), "arch_type", "0")); // check arch_type fail
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "sample_offline_model.om", model_buffer), SUCCESS);

    ModelData model_data;
    GE_MAKE_GUARD(model_guard, [&model_data]() {
      if (model_data.model_data != nullptr) {
          delete[] static_cast<char_t *>(model_data.model_data);
          model_data.model_data = nullptr;
      }
    });
    EXPECT_EQ(ge_executor_.LoadDataFromFile("sample_offline_model.om", model_data), SUCCESS);
    {
      uint32_t model_id = 0U;
      EXPECT_NE(ge_executor_.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
      model_ids.emplace_back(model_id);
    }
    ge::RuntimeStub::Reset();
  }

  ModelDumpFiniCmd(ge_executor_);

  for (const auto &id : model_ids) {
    EXPECT_EQ(ge_executor_.UnloadModel(id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  }

  dlog_setlevel(-1, 3, 0);
}

/**
 * 用例描述：静态图离线加在场景，图中有FILECONSTANT，用户为FileConstant设置了Device地址，使用用户设置的Device地址
 * 预置条件：
 * 1. 构造包含FILECONSTANT算子的图，并创建对应的权重文件
 * 2. 构造离线模型，落盘生成om 文件
 * 测试步骤：
 * 1. fileconstant算子的权重文件名test_weight.bin
 * 2. 通过计算图生成ModelData
 * 3. 通过ModelData加载执行器
 * 4. 执行器Load，加载入参设置外置权重device地址
 * 预期结果：
 * 1. 如果内存大小正常，加载成功，日志校验正确。file_constant1使用了用户设置的地址
 * 2. 如果内存大小不足，加载失败，带有ERROR日志。
 * 3. 执行器执行输出正确，即tensordata输出为权重文件内容
 */
TEST_F(GeExecutorTest, FileConstant_UserSetDeviceMem) {
  shared_ptr<OpsKernelInfoStore> fake_ops_kernel_info_store = std::make_shared<FakeOpsKernelInfoStore>();
  // hccl op goes to AIcoreEngine in this testcase
  OpsKernelExecutorManager::GetInstance().executors_["AIcoreEngine"] = fake_ops_kernel_info_store;
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  GeModelPtr ge_model;
  BuildFileConstantGraph(graph, ge_model, mem_offset);
  EXPECT_NE(graph, nullptr);
  EXPECT_NE(ge_model, nullptr);

  ModelDumpInitCmd(ge_executor_);

  DelStaticForOffline(graph, mem_offset);  // Offline model will set new session_id, static var invalid.
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  {
    auto var1 = ge_model->GetGraph()->FindNode("file_constant1");
    auto var2 = ge_model->GetGraph()->FindNode("file_constant2");
    ASSERT_NE(var1, nullptr);
    ASSERT_NE(var2, nullptr);
    ge::TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var1->GetOpDescBarePtr()->SetOutputOffset({137438953472U});
    ge::TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var2->GetOpDescBarePtr()->SetOutputOffset({137438954472U});

    // Test LoadModelOffline
    {
      ModelHelper model_helper;
      model_helper.SetSaveMode(true);  // Save to file.
      ModelBufferData model_buffer;
      EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "sample_offline_model.om", model_buffer), SUCCESS);
    }

    ModelData model_data;
    EXPECT_EQ(ge_executor_.LoadDataFromFile("sample_offline_model.om", model_data), SUCCESS);
    model_data.om_name = "g1_om";
    uint32_t model_id = 0U;
    ProfileCommandInit(ge_executor_);
    gert::RtSession session(199);
    ge::ModelLoadArg load_arg;
    load_arg.dev_ptr = nullptr;
    load_arg.mem_size = 0;
    load_arg.weight_ptr = nullptr;
    load_arg.rt_session = &session;
    const size_t user_mem_size = 1024;
    int32_t user_mem[user_mem_size];
    FileConstantMem file_constant_mem{"file_constant1.bin", (void *)user_mem, user_mem_size};
    load_arg.file_constant_mems.emplace_back(file_constant_mem);

    {
      gert::GertRuntimeStub runtime_stub;
      runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
      EXPECT_EQ(ge_executor_.LoadModelFromDataWithArgs(model_id, model_data, load_arg), SUCCESS);
      auto log_ret = runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "FileConstant node file_constant1 found user device memory (addr");
      EXPECT_NE(log_ret, -1);
      log_ret = runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "FileConstant node file_constant2 malloc device memory (addr");
      EXPECT_NE(log_ret, -1);
      ProfileCommandProf(ge_executor_, model_id);
      ProfileCommandFini(ge_executor_, model_id);
      EXPECT_EQ(ge_executor_.UnloadModel(model_id), SUCCESS);
    }
    load_arg.file_constant_mems.back().mem_size = 1U;
    {
      gert::GertRuntimeStub runtime_stub;
      runtime_stub.GetSlogStub().SetLevel(DLOG_ERROR);
      EXPECT_NE(ge_executor_.LoadModelFromDataWithArgs(model_id, model_data, load_arg), SUCCESS);
      auto log_ret = runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, "The device memory size set by the user via aclmdlSetExternalWeightAddress for the external weight file is insufficient. Required: 8 bytes, Provided: 1 bytes. ");
      EXPECT_NE(log_ret, -1);
    }
    delete[] (char *)model_data.model_data;
    ModelDumpFiniCmd(ge_executor_);
  }
  system("rm -rf file_constant1.bin");
  system("rm -rf file_constant2.bin");
}

/**
 * 用例描述：静态图离线加在场景，图中有FILECONSTANT，同一个线程加载2个om，校验第2个om加载时fileconstant会有h2d拷贝
 * 预置条件：
 * 1. 构造包含FILECONSTANT算子的图，并创建对应的权重文件
 * 2. 构造离线模型，落盘生成om 文件
 * 测试步骤：
 * 1. fileconstant算子的权重文件名test_weight.bin
 * 2. 通过计算图生成ModelData
 * 3. 通过ModelData加载执行器
 * 预期结果：
 * 1. 两次加载，session id不同，因此外置权重不能共享同一份device内存，各自有h2d拷贝
 */
TEST_F(GeExecutorTest, FileConstant_OneThreadLoadTwoOm) {
  shared_ptr<OpsKernelInfoStore> fake_ops_kernel_info_store = std::make_shared<FakeOpsKernelInfoStore>();
  // hccl op goes to AIcoreEngine in this testcase
  OpsKernelExecutorManager::GetInstance().executors_["AIcoreEngine"] = fake_ops_kernel_info_store;
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  GeModelPtr ge_model;
  BuildFileConstantGraph(graph, ge_model, mem_offset);
  EXPECT_NE(graph, nullptr);
  EXPECT_NE(ge_model, nullptr);

  ModelDumpInitCmd(ge_executor_);

  DelStaticForOffline(graph, mem_offset);  // Offline model will set new session_id, static var invalid.
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  {
    auto var1 = ge_model->GetGraph()->FindNode("file_constant1");
    auto var2 = ge_model->GetGraph()->FindNode("file_constant2");
    ASSERT_NE(var1, nullptr);
    ASSERT_NE(var2, nullptr);
    ge::TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var1->GetOpDescBarePtr()->SetOutputOffset({137438953472U});
    ge::TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 512);
    var2->GetOpDescBarePtr()->SetOutputOffset({137438954472U});

    // Test LoadModelOffline
    {
      ModelHelper model_helper;
      model_helper.SetSaveMode(true);  // Save to file.
      ModelBufferData model_buffer;
      EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "sample_offline_model1.om", model_buffer), SUCCESS);
    }

    // Test LoadModelOffline
    {
      ModelHelper model_helper;
      model_helper.SetSaveMode(true);  // Save to file.
      ModelBufferData model_buffer;
      EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "sample_offline_model2.om", model_buffer), SUCCESS);
    }

    GeExecutor ge_executor;
    ModelData model_data;
    EXPECT_EQ(ge_executor.LoadDataFromFile("sample_offline_model1.om", model_data), SUCCESS);
    model_data.om_name = "g1_om";
    uint32_t model_id = 0U;
    ge::ModelLoadArg load_arg;

    EXPECT_EQ(ge_executor.LoadModelFromDataWithArgs(model_id, model_data, load_arg), SUCCESS);

    GeExecutor ge_executor2;
    ModelData model_data2;
    EXPECT_EQ(ge_executor2.LoadDataFromFile("sample_offline_model2.om", model_data2), SUCCESS);
    model_data.om_name = "g1_om";
    uint32_t model_id2 = 1U;
    {
      gert::GertRuntimeStub runtime_stub;
      runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
      EXPECT_EQ(ge_executor2.LoadModelFromDataWithArgs(model_id2, model_data2, load_arg), SUCCESS);
      auto log_ret = runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "CopyOneWeightFromFileWithFilehandler");
      EXPECT_NE(log_ret, -1);
    }

    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);

    delete[] (char *)model_data.model_data;
    delete[] (char *)model_data2.model_data;
  }
  system("rm -rf file_constant1.bin");
  system("rm -rf file_constant2.bin");
  system("rm -rf sample_offline_model1.om");
  system("rm -rf sample_offline_model2.om");
}

static void BuildSampleCondGraph(ComputeGraphPtr &graph, uint32_t &mem_offset) {
  DEF_GRAPH(g0) {
    const auto add_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    const auto sub_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    const auto less_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", DATA)->NODE("add", add_node)->NODE("merge", STREAMMERGE)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("const_0", CONSTANT)->NODE("add"));
    CHAIN(NODE("_arg_1", DATA)->NODE("sub", sub_node)->NODE("merge"));
    CHAIN(NODE("const_1", CONSTANT)->NODE("sub"));

    const auto switch_t = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    const auto switch_f = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_NOT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("less", less_node)->EDGE(0, 0)->NODE("Less/StreamSwitch_t", switch_t)->CTRL_EDGE()->NODE("add"));
    CHAIN(NODE("const_0")->EDGE(0, 1)->NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("_arg_1")->EDGE(0, 1)->NODE("less", less_node)->EDGE(0, 0)->NODE("Less/StreamSwitch_f", switch_f)->CTRL_EDGE()->NODE("sub"));
    CHAIN(NODE("const_1")->EDGE(0, 1)->NODE("Less/StreamSwitch_f"));

    const auto active_s = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{1});
    CHAIN(NODE("less")->CTRL_EDGE()->NODE("Less_StreamActive", active_s)->CTRL_EDGE()->NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_f"));

    const auto active_0 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    const auto active_1 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("add")->CTRL_EDGE()->NODE("merge_input_0_active", active_0)->CTRL_EDGE()->NODE("merge"));
    CHAIN(NODE("sub")->CTRL_EDGE()->NODE("merge_input_1_active", active_1)->CTRL_EDGE()->NODE("merge"));
  };
  graph = ToComputeGraph(g0);
  graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(graph, mem_offset, true);
}

void BuildGraphModelRelease(const ComputeGraphPtr &graph, uint32_t mem_offset, GeModelPtr &ge_model, TBEKernelStore &tbe_kernel_store) {
  InitConstantNode(graph, "const_0", 1);
  InitConstantNode(graph, "const_1", 0);

  std::shared_ptr<domi::ModelTaskDef> model_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef_TE(graph, *model_def, "less", tbe_kernel_store);

  InitStreamActiveDef(graph, *model_def, "Less_StreamActive");
  InitStreamSwitchDef(graph, *model_def, "Less/StreamSwitch_f");
  InitStreamSwitchDef(graph, *model_def, "Less/StreamSwitch_t");

  InitKernelTaskDef_TE(graph, *model_def, "add", tbe_kernel_store);
  InitKernelTaskDef_TE(graph, *model_def, "sub", tbe_kernel_store);

  InitStreamActiveDef(graph, *model_def, "merge_input_0_active");
  InitStreamActiveDef(graph, *model_def, "merge_input_1_active");
  InitStreamMergeDef(graph, *model_def, "merge");

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
}

TEST_F(GeExecutorTest, release_single_operator_resource) {
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildSampleCondGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);
  graph->SetGraphUnknownFlag(true);

  GeModelPtr ge_model;
  TBEKernelStore tbe_kernel_store;
  BuildGraphModelRelease(graph, mem_offset, ge_model, tbe_kernel_store);
  EXPECT_NE(ge_model, nullptr);

  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};

  {
    int64_t arg_0 = 127;
    int64_t arg_1 = 100;
    RunModelData run_input_data;
    run_input_data.blobs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
    run_input_data.blobs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});

    int64_t arg_3 = 111;
    RunModelData run_output_data;
    run_output_data.blobs.emplace_back(DataBuffer{&arg_3, sizeof(arg_3), false, 0});

    uint32_t model_id = 0;
    uint32_t device_id = 0;

    EXPECT_EQ(rtSetDevice(device_id), RT_ERROR_NONE);
    EXPECT_EQ(ge_executor_.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    {
      rtStream_t stream = nullptr;
      rtStreamCreate(&stream, 0);
      EXPECT_EQ(ge_executor_.ExecModel(model_id, stream, run_input_data, run_output_data, true), SUCCESS);
      {
        EXPECT_EQ(ge_executor_.ReleaseSingleOpResource(stream), SUCCESS);
        rtStreamDestroy(stream);
      }
    }
    {
      rtStream_t stream = nullptr;
      rtStreamCreate(&stream, 0);
      EXPECT_EQ(ge_executor_.ExecModel(model_id, stream, run_input_data, run_output_data, true), SUCCESS);
      {
        EXPECT_EQ(ge_executor_.ReleaseSingleOpResource(stream), SUCCESS);
        rtStreamDestroy(stream);
      }
    }
    {
      rtStream_t stream = nullptr;
      rtStreamCreate(&stream, 0);
      EXPECT_EQ(ge_executor_.ExecModel(model_id, stream, run_input_data, run_output_data, true), SUCCESS);
      {
        EXPECT_EQ(ge_executor_.ReleaseSingleOpResource(stream), SUCCESS);
        rtStreamDestroy(stream);
      }
    }
    EXPECT_EQ(ge_executor_.UnloadModel(model_id), SUCCESS);
    {
      uint32_t device_id = 256U;
      EXPECT_EQ(ge_executor_.ClearCustomAicpuSo(device_id), SUCCESS);
      EXPECT_EQ(rtDeviceReset(device_id), RT_ERROR_NONE);
    }
  }
}

void BuildComputeGraph(ComputeGraphPtr &graph, int32_t dynamic_type) {
  auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).Build("data");
  auto transdata = OP_CFG(TRANSDATA).InCnt(1).OutCnt(1).Build("transdata");
  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(0).Build("netoutput");

  DEF_GRAPH(g0) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(transdata)->EDGE(0, 0)->NODE(netoutput));
    CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("case", "Case"));
  };
  graph = ToComputeGraph(g0);

  auto op_desc = graph->FindNode("netoutput")->GetOpDesc();
  std::vector<std::string> src_name{"out1", "out2"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0, 1};
  op_desc->SetSrcIndex(src_index);

  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(ge::GeShape({3, 12, 5, 6}));
  std::vector<std::pair<int64_t, int64_t>> range{{1, 5}, {2, 5}, {3, 6}, {4, 7}};
  tensor_desc.SetShapeRange(range);
  op_desc->AddInputDesc("x", tensor_desc);

  NodePtr case_node = graph->FindNode("case");
  OpDescPtr op_desc_case = case_node->GetOpDesc();
  AttrUtils::SetInt(op_desc_case, ATTR_DYNAMIC_TYPE, dynamic_type);
  uint32_t batch_num = 2U;
  AttrUtils::SetInt(op_desc_case, ATTR_NAME_BATCH_NUM, batch_num);
  for (uint32_t i = 0U; i < batch_num; ++i) {
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    std::vector<int64_t> batch_shape = {1, 2};
    AttrUtils::SetListInt(op_desc_case, attr_name, batch_shape);
  }
  std::vector<std::string> user_designate_shape_order = {"data1", "data2"};
  AttrUtils::SetListStr(op_desc_case, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order);
}

TEST_F(GeExecutorTest, get_model_desc_info_from_mem) {
  ComputeGraphPtr graph ;
  int32_t dynamic_type[2] = {ge::DYNAMIC_BATCH, ge::DYNAMIC_DIMS};
  for (int32_t i = 0; i < 2; i++) {
    BuildComputeGraph(graph, dynamic_type[i]);
    GeModelPtr ge_model = std::make_shared<GeModel>();
    std::shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
    model_task_def->add_task()->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_END_GRAPH));
    std::vector<uint64_t> weights_value(64, 1024);
    size_t weight_size = weights_value.size() * sizeof(uint64_t);

    ge_model->SetGraph(graph);
    ge_model->SetModelTaskDef(model_task_def);
    ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
    AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
    AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

    ModelBufferData model_buffer;
    ModelHelper model_helper;
    model_helper.SetSaveMode(false);  // Save to buffer.
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "model_desc_file_prefix", model_buffer), SUCCESS);

    ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
    ModelInOutInfo info;
    EXPECT_EQ(ge_executor_.GetModelDescInfoFromMem(model_data, info), SUCCESS);
  }
}

TEST_F(GeExecutorTest, sample_davinci_model_static_memory_with_qos) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  // Tensor for input.
  std::vector<gert::Tensor> inputs(4);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[3], {1}, DT_INT64, FORMAT_ND);

  // Tensor for input.
  std::vector<gert::Tensor> sync_inputs(4);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(sync_inputs[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;
  DumpCommandInit(ge_executor_);
  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->IncreaseLoadCount();

    // Callback for execute.
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

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption({{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"},
                            {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({{VARIABLE_MEMORY_MAX_SIZE, "12800"}}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);
    model_executor.StartRunThread();

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(run_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph.
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  // ModelExecutor::RunGraph -> ExecuteGraph -> SyncExecuteModel -> ModelManager::DataInput -> DavinciModel::Push -> Run
  {
    // Test LoadModelOnline: GraphModelListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(false);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 960)); // set max stream

    // Env for load:.ModelManager::CheckAndReleaseStreamEventResource
    GEThreadLocalContext &context = GetThreadLocalContext();
    context.SetGraphOption({{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"},
                            {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});

    // profiling model subscribe on
    ProfilingProperties::Instance().SetSubscribeInfo(0, graph->GetGraphID(), true);

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // Execute Synchronous
    std::vector<gert::Tensor> sync_outputs;
    EXPECT_EQ(model_executor.RunGraph(graph_node, graph->GetGraphID(), sync_inputs, sync_outputs), SUCCESS);
    EXPECT_EQ(sync_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());
    // check reported graph id saved
    EXPECT_TRUE(ProfilingManager::Instance().IsGraphProfReported(graph->GetGraphID()));

    // clear profiling configurations
    ProfilingProperties::Instance().subscribe_count_--;
    ProfilingManager::Instance().ProfFinalize();

    // Unload model of graph(leave as max stream model for follow test)
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }


  int64_t value_0 = 127;
  int64_t value_1 = 100;
  int64_t value_2 = 258;
  int64_t value_3 = 512;

  GeTensorDesc sync_tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
  GeTensor sync_tensor_0(sync_tensor_desc, (uint8_t *)&value_0, sizeof(value_0));
  GeTensor sync_tensor_1(sync_tensor_desc, (uint8_t *)&value_1, sizeof(value_1));
  GeTensor sync_tensor_2(sync_tensor_desc, (uint8_t *)&value_2, sizeof(value_2));
  GeTensor sync_tensor_3(sync_tensor_desc, (uint8_t *)&value_3, sizeof(value_3));
  const std::vector<GeTensor> ge_sync_inputs{ sync_tensor_0, sync_tensor_1, sync_tensor_2, sync_tensor_3 };
  {
    // Test LoadModelOnline: RunGraphWithStream
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    ge_root_model->SetIsSpecificStream(true); // For not start DavinciModel thread.
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 960));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));

    GeTensorDesc output_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    std::vector<uint8_t> arg_3(512, 0);  // mem_offset += (2 * 4 * 8 * 2 * sizeof(float));
    GeTensor nn_tensor_21(output_desc, arg_3.data(), arg_3.size());
    std::vector<GeTensor> nn_outputs{ nn_tensor_21 };

    // Load model of graph
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    model_executor.StartRunThread();

    // NnExecute with stream.
    rtStream_t run_stream = &model_executor;
    EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph->GetGraphID(), run_stream, ge_sync_inputs, nn_outputs), SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph(leave as max event model for follow test).
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  std::vector<gert::Tensor> inputs2(4);
  TensorCheckUtils::ConstructGertTensor(inputs2[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs2[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs2[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs2[3], {1}, DT_INT64, FORMAT_ND);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 32));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));
  {
    // Test LoadModelOnline: for SuperKernel
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->IncreaseLoadCount();

    // Setup SuperKernel
    EXPECT_TRUE(AttrUtils::SetBool(ge_model, ATTR_NAME_SWITCH_FOR_L1_FUSION, true));
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetOpDesc()->GetOpKernelLibName() == "AIcoreEngine") {
        EXPECT_TRUE(AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_FUSION_GROUP_KEY, 1));
      }
    }

    // Callback for execute.
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

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs2);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);
    model_executor.StartRunThread();

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(run_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph.
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  DumpCommandFini(ge_executor_);
  for (const auto &id : model_ids) {
    EXPECT_EQ(ge_executor_.UnloadModel(id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  }
  unsetenv(kEnvGeuseStaticMemory.c_str());
}
TEST_F(GeExecutorTest, run_with_fail) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, nullptr);
  model.SetId(1);

  auto data = MakeShared<RunArgs>();
  model.Push(data);
  model.Push(data);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  //task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);

  const char_t * const kEnvRecordPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  unsetenv(kEnvRecordPath);
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(GeExecutorTest, run_with_task_2_iterator) {
  class DModelListener : public ModelListener {
   public:
    DModelListener(){};
    uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result, std::vector<gert::Tensor> &outputs) {
      return 0;
    }
  };

  shared_ptr<ModelListener> g_local_call_back(new DModelListener());
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
    ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 4096, 1024});
    (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
    const auto model_def = MakeShared<domi::ModelTaskDef>();
    ge_model->SetModelTaskDef(model_def);
    DavinciModel model(0, g_local_call_back);
    model.SetId(1);
    model.Assign(ge_model);
    ModelParam param1{};
    EXPECT_EQ(model.Init(param1), SUCCESS);
    model.isGraphLevelSat_ = true;

    auto data = MakeShared<RunArgs>();
    model.Push(data);
    model.Push(data);
    domi::ModelTaskDef model_task_def;
    domi::TaskDef *task = model_task_def.add_task();
    task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
    task->_impl_.stream_id_ = 0;
    rtStream_t stream = nullptr;
    model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model.stream_list_ = { stream };
    TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
    model.task_list_.push_back(task_info);
    model.has_output_node_ = true;
    OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
    std::vector<int64_t> input_offset;
    input_offset.emplace_back(0);
    GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset(input_offset);
    model.InitOutputTensorInfo(op_desc);
    EXPECT_EQ(model.ModelRunStart(), SUCCESS);
    sleep(5);
    EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(GeExecutorTest, run_with_task_0_iterator) {
  class DModelListener : public ModelListener {
   public:
    DModelListener(){};
    uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result, std::vector<gert::Tensor> &outputs) {
      return 0;
    }
  };
  const char *const kVectorcoreNum = "ge.vectorcoreNum";
  shared_ptr<ModelListener> g_local_call_back(new DModelListener());
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  graph_options[AICORE_NUM] = "2";
  graph_options[kVectorcoreNum] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
    ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 4096, 1024});
    (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
    const auto model_def = MakeShared<domi::ModelTaskDef>();
    ge_model->SetModelTaskDef(model_def);
    DavinciModel model(0, g_local_call_back);
    model.SetId(1);
    model.Assign(ge_model);
    model.isGraphLevelSat_ = true;

    domi::ModelTaskDef model_task_def;
    domi::TaskDef *task = model_task_def.add_task();
    task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
    task->_impl_.stream_id_ = 0;
    rtStream_t stream = nullptr;
    model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model.stream_list_ = { stream };
    TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
    model.task_list_.push_back(task_info);
    model.has_output_node_ = true;
    OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
    std::vector<int64_t> input_offset;
    input_offset.emplace_back(0);
    GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset(input_offset);
    model.InitOutputTensorInfo(op_desc);
    model.Init();
    EXPECT_EQ(model.ModelRunStart(), SUCCESS);
    sleep(1);
    EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  graph_options[AICORE_NUM] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(GeExecutorTest, testConstructActiveMemBaseAddrs) {
  const char *const kVectorcoreNum = "ge.vectorcoreNum";
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  graph_options[AICORE_NUM] = "2";
  graph_options[kVectorcoreNum] = "2";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
    ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 4096, 1024});
    (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
    const auto model_def = MakeShared<domi::ModelTaskDef>();
    ge_model->SetModelTaskDef(model_def);
    DavinciModel model(0, nullptr);
    model.SetId(1);
    model.Assign(ge_model);
    model.isGraphLevelSat_ = true;

    domi::ModelTaskDef model_task_def;
    domi::TaskDef *task = model_task_def.add_task();
    task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
    rtStream_t stream = nullptr;
    model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model.stream_list_ = { stream };
    TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
    model.task_list_.push_back(task_info);

    MemAllocation mem_allocation0 = {};
    mem_allocation0.data_size = 25600U + 32U; // random value for test
    mem_allocation0.tensor_size = 25600U; // random value for test
    mem_allocation0.logical_addr = 30902000U; // random value for test
    mem_allocation0.type = MemAllocation::Type::INPUT;
    model.logical_mem_allocations_.emplace_back(mem_allocation0);

    EXPECT_EQ(model.Init(), SUCCESS);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  graph_options[AICORE_NUM] = "";
  graph_options[kVectorcoreNum] = "";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(GeExecutorTest, testHWQ) {
  const char *const kVectorcoreNum = "ge.vectorcoreNum";
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  graph_options[AICORE_NUM] = "2";
  graph_options[kVectorcoreNum] = "2";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
    setenv("HW_QUEUE", "true", 1);
    ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 4096, 1024});
    (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
    const auto model_def = MakeShared<domi::ModelTaskDef>();
    ge_model->SetModelTaskDef(model_def);
    DavinciModel model(0, nullptr);
    model.SetId(1);
    model.Assign(ge_model);
    model.isGraphLevelSat_ = true;

    domi::ModelTaskDef model_task_def;
    domi::TaskDef *task = model_task_def.add_task();
    task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
    rtStream_t stream = nullptr;
    model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model.stream_list_ = { stream };
    TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
    model.task_list_.push_back(task_info);

    MemAllocation mem_allocation0 = {};
    mem_allocation0.data_size = 25600U + 32U; // random value for test
    mem_allocation0.tensor_size = 25600U; // random value for test
    mem_allocation0.logical_addr = 30902000U; // random value for test
    mem_allocation0.type = MemAllocation::Type::INPUT;
    model.logical_mem_allocations_.emplace_back(mem_allocation0);
    QueueAttrs inputQueue1 = {0, 0, 0, 0U};
    QueueAttrs outputQueue1 = {1, 0, 0, 0U};
    model.input_queue_attrs_.emplace_back(inputQueue1);
    model.output_queue_attrs_.emplace_back(outputQueue1);
    ZeroCopyOffset zero_copy_offset;
    std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data;
    virtual_addr_out_data[0x1111].emplace_back(0x1111111);
    zero_copy_offset.outside_addrs_.emplace_back(virtual_addr_out_data);
    model.input_data_info_[0] = zero_copy_offset;
    model.output_data_info_[0] = zero_copy_offset;
    SetMemQueueEntityType(1);
    EXPECT_EQ(model.SetQueueType(), SUCCESS);
    EXPECT_EQ(model.Init(), SUCCESS);
    unsetenv("HW_QUEUE");
    SetMemQueueEntityType(0);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  graph_options[AICORE_NUM] = "";
  graph_options[kVectorcoreNum] = "";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

} // namespace ge

