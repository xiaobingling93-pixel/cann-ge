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
#include <nlohmann/json.hpp>
#include <set>
#include "common/scope_allocator.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "../ub_fusion/builtin_buffer_fusion_pass_test.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"
#include "common/lxfusion_json_util.h"
#include "common/platform_utils.h"
#include "common/fe_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/util/op_info_util.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "./ge_context.h"
#include "./ge_local_context.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"
#include "graph_optimizer/op_compiler/op_compiler_baseline.h"
#include "graph_optimizer/op_compiler/op_compiler_normal.h"
#include "graph_optimizer/op_compiler/op_compiler_optune.h"
#include "graph_optimizer/op_compiler/op_compiler_mstune_before_ub_match.h"
#include "graph/ge_local_context.h"
#include "register/optimization_option_registry.h"
#undef protected
#undef private

using namespace testing;
using namespace fe;
using namespace ge;
using TbeOpStoreAdapterPtr = std::shared_ptr<fe::TbeOpStoreAdapter>;
using FEGraphOptimizerPtr = std::shared_ptr<fe::FEGraphOptimizer>;
using OpStoreAdapterPtr = std::shared_ptr<fe::OpStoreAdapter>;

std::string GetAscendPath() {
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_path = "/mnt/d/Ascend";
  if (ascend_custom_path_ptr != nullptr) {
      ascend_path = fe::GetRealPath(string(ascend_custom_path_ptr));
  } else {
      const char *ascend_home_path_ptr = std::getenv("ASCEND_HOME");
      if (ascend_home_path_ptr != nullptr) {
      ascend_path = fe::GetRealPath(string(ascend_home_path_ptr));
      } else {
      ascend_path = "/mnt/d/Ascend";
      }
  }
  return ascend_path;
}

string GetNetworkPath(const string &network_name) {
  auto custom_path = GetAscendPath();
  custom_path += "/net/";
  return custom_path + network_name;
}

class TestPass : public PatternFusionBasePass {
 protected:

  vector<FusionPattern *> DefinePatterns() override {
    return {};
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    return ge::SUCCESS;
  }
};

using CreateFn = GraphPass *(*)();

fe::GraphPass *CreateFunc() {
  return new(std::nothrow) TestPass();
}

class StubFEKernelInfoStore : public fe::FEOpsKernelInfoStore {
 public:
  StubFEKernelInfoStore(std::string engine_name) :FEOpsKernelInfoStore(engine_name) {}
  bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                              bool realQuery = false) const override {
    ge::AttrUtils::SetInt(opDescPtr, FE_IMPLY_TYPE, 6);
    return true;
  }
};

void RegisterPassFunc(CreateFn create_fn) {
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS2", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS3", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS2", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(SECOND_ROUND_BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS3", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(SECOND_ROUND_BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS4", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "BUILT_IN_PASS3", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "BUILT_IN_PASS4", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS2", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS3", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "BEFORE_QUANT_1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "BEFORE_QUANT_2", create_fn, 0);
}

class OptimizeUtilityUTStub: public ge::OptimizeUtility {
 public:
  OptimizeUtilityUTStub() {}
  virtual ~OptimizeUtilityUTStub() override {}

  ge::Status InferShape(ComputeGraph &compute_graph) override{
    return ge::SUCCESS;
  }

  ge::Status InferShape(const ComputeGraphPtr &compute_graph) override {
    return ge::SUCCESS;
  }
};

bool checkIsRegistered(const te::TbeOpInfo &op_info, bool &val) {
  val = true;
  return true;
}

bool checkIsNotRegistered(const te::TbeOpInfo &op_info, bool &val) {
  val = false;
  return true;
}

bool checkIsRegisteredException(const te::TbeOpInfo &op_info, bool &val) {
  val = false;
  return false;
}

ge::OpKernelBinPtr GetOpKernelBinByKernelName(const std::string &kernel_name) {
  return nullptr;
}

bool teGeneralize(const te::TbeOpInfo &op_info,
    const te::TE_GENERALIZE_TYPE &general_type, const ge::NodePtr &node) {
  std::vector<int64_t> shape_vec;
  auto op_desc = node->GetOpDesc();
  auto tensor_desc_x = op_desc->MutableInputDesc(0);
  if (tensor_desc_x == nullptr) {
    return false;
  }
  shape_vec = tensor_desc_x->GetShape().GetDims();
  if (general_type == te::REGISTER_FUNC) {
    for (auto &i : shape_vec) {
      i = -1;
    }
  } else if (general_type == te::DEFAULT_TBE_OP_INFO) {
    for (int i = 0; i < shape_vec.size()-1; ++i) {
      shape_vec[i] = -1;
    }
  } else {
    shape_vec[0] = -1;
  }
  FE_LOGD("shape:%ld,%ld,%ld,%ld", shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]);
  tensor_desc_x->SetOriginShape(ge::GeShape(shape_vec));
  return true;
}

bool teGeneralizeException(const te::TbeOpInfo &op_info,
    const te::TE_GENERALIZE_TYPE &general_type, const ge::NodePtr &node) {
  return false;
}

tune::Status LxFusionFinalizeFunc1(const ge::ComputeGraph &){
  return tune::SUCCESS;
}

tune::Status LxFusionRecoveryFunc1(ge::ComputeGraph &, const std::vector<ge::NodePtr> &, std::vector<ge::NodePtr> *,
                                   std::vector<ge::NodePtr> *){
  return tune::SUCCESS;
}

class UTEST_fusion_engine_fe_graph_optimizer : public testing::Test
{
public:
    FEOpsKernelInfoStorePtr ops_info_store;
    FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
    SplitNOptimizer split_n_optimizer;
    RefRelationsPtr reflection_builder_ptr_;
    FEGraphOptimizerPtr fe_graph_optimizer_;
    TbeOpStoreAdapterPtr tbe_adapter_ptr;
    shared_ptr<fe::SubOpInfoStore> sub_ops_kernel_ptr;
    shared_ptr<fe::SubOpsStore> sub_ops_store_ptr;
    GraphFusionPtr graph_fusion_ptr_;
    LxFusionOptimizerPtr lx_fusion_optimizer_;
    NodePtr MakeNode(const ComputeGraphPtr &graph, uint32_t in_num, uint32_t out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }
protected:
  static void SetUpTestCase() {
    std::string soc_version = "Ascend310P3";
    PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
    PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
    PlatformUtils::Instance().soc_version_ = soc_version;
    Configuration::Instance(AI_CORE_NAME).InitLibPath();
  }
  void SetUp()
  {
    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
    ops_info_store = std::make_shared<FEOpsKernelInfoStore>();
    sub_ops_store_ptr = make_shared<fe::SubOpsStore>(fe::AI_CORE_NAME);
    tbe_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));

    OptimizeUtilityUTStub *optimize_utility_utub = new OptimizeUtilityUTStub();

    ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    ops_kernel_info_store_ptr_->tbe_info_assembler_ptr_ = std::make_shared<TbeInfoAssembler>();
    ops_kernel_info_store_ptr_->tbe_info_assembler_ptr_->Initialize();
    FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
              fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
    fusion_priority_mgr_ptr_->Initialize();
    lx_fusion_optimizer_ = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr_ptr_, ops_kernel_info_store_ptr_);
    lx_fusion_optimizer_->Initialize();
    graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_,
                                                      fusion_priority_mgr_ptr_);
    graph_fusion_ptr_->SetEngineName(fe::AI_CORE_NAME);
    fe_graph_optimizer_ = make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, fe::AI_CORE_NAME);
    std::map<std::string, std::string> options;
    fe_graph_optimizer_->Initialize(options, optimize_utility_utub);
    fe_graph_optimizer_->graph_fusion_ptr_ = graph_fusion_ptr_;

    FEOpsStoreInfo TBE_OPINFO_STUB = {
            6,
            "tbe-builtin",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/heavy_opinfo",
            ""
    };

    sub_ops_store_ptr->SetSubStoreInfo(TBE_OPINFO_STUB);
    sub_ops_store_ptr->InitializeSubStore();

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(TBE_OPINFO_STUB);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);

    sub_ops_kernel_ptr = std::make_shared<fe::SubOpInfoStore>(TBE_OPINFO_STUB);
    sub_ops_kernel_ptr->Initialize(fe::AI_CORE_NAME);
    OpsKernelManager::Instance(fe::AI_CORE_NAME).sub_ops_kernel_map_.emplace("tbe-builtin", sub_ops_kernel_ptr);

    options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", SHAPE_GENERALIZED));
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    ge::GetThreadLocalContext().SetGlobalOption(options);

    std::map<std::string, std::string> options1;
    OpsKernelManager::Instance(fe::AI_CORE_NAME).Finalize();
    ops_info_store->Initialize(options1);
    ops_kernel_info_store_ptr_->Initialize(options);
  }

  void TearDown()
  {
    sub_ops_store_ptr->FinalizeSubStore();
    sub_ops_store_ptr.reset();
    sub_ops_kernel_ptr->Finalize();
    sub_ops_kernel_ptr.reset();
    ops_info_store->Finalize();

    PlatformUtils::Instance().soc_version_ = "Ascend910B1";
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
  }
  static void CreateConv2dGraph(ComputeGraphPtr graph) {
    OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr reduceSum = std::make_shared<OpDesc>("reduceSum", "ReduceSum");

    // add descriptor
    vector<int64_t> dims = {1, 3, 32, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NHWC);
    in_desc2.SetOriginFormat(FORMAT_NHWC);
    in_desc2.SetDataType(DT_FLOAT16);
    conv2d->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    reduceSum->AddInputDesc("x", in_desc2);

    vector<int64_t> dims1 = {1, 1, 3, 32, 32};
    GeShape shape1(dims1);
    GeTensorDesc out_desc1(shape1);
    out_desc1.SetFormat(FORMAT_NC1HWC0);
    out_desc1.SetOriginFormat(FORMAT_NC1HWC0);
    out_desc1.SetDataType(DT_FLOAT16);
    conv2d->AddOutputDesc("y", out_desc1);
    std::vector<bool> is_in_const_vec = {false};
    conv2d->SetIsInputConst(is_in_const_vec);
    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    reduceSum->AddOutputDesc("y", out_desc2);

    ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(conv2d, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(conv2d);
    NodePtr data_node = graph->AddNode(data);
    NodePtr reduceSum_node = graph->AddNode(reduceSum);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), reduceSum_node->GetInDataAnchor(0));
  }

  static void CreateBatchNormGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);
    uint32_t thread_scope_id = 2;
    (void)ge::AttrUtils::SetInt(bn_op, kThreadScopeId, thread_scope_id);
    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
  }

  ComputeGraphPtr CreateMultiThreadGraph() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateConv2dGraph(graph);
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape, FORMAT_NCHW, DT_FLOAT);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetOriginDataType(DT_FLOAT);
    out_desc.SetOriginShape(shape);
    OpDescPtr relu_op2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr relu_op3 = std::make_shared<OpDesc>("relu3", "Relu");
    OpDescPtr relu_op4 = std::make_shared<OpDesc>("relu4", "Relu");
    OpDescPtr relu_op5 = std::make_shared<OpDesc>("relu5", "Relu");
    OpDescPtr relu_op6 = std::make_shared<OpDesc>("relu6", "Relu");
    OpDescPtr relu_op7 = std::make_shared<OpDesc>("relu7", "Relu");

    relu_op2->AddInputDesc("x", out_desc);
    relu_op2->AddOutputDesc("y", out_desc);
    relu_op3->AddInputDesc("x", out_desc);
    relu_op3->AddOutputDesc("y", out_desc);
    relu_op4->AddInputDesc("x", out_desc);
    relu_op4->AddOutputDesc("y", out_desc);
    relu_op5->AddInputDesc("x", out_desc);
    relu_op5->AddOutputDesc("y", out_desc);
    relu_op6->AddInputDesc("x", out_desc);
    relu_op6->AddOutputDesc("y", out_desc);
    relu_op7->AddInputDesc("x", out_desc);
    relu_op7->AddOutputDesc("y", out_desc);

    NodePtr relu_node2 = graph->AddNode(relu_op2);
    NodePtr relu_node3 = graph->AddNode(relu_op3);
    NodePtr relu_node4 = graph->AddNode(relu_op4);
    NodePtr relu_node5 = graph->AddNode(relu_op5);
    NodePtr relu_node6 = graph->AddNode(relu_op6);
    NodePtr relu_node7 = graph->AddNode(relu_op7);
    return graph;
  }

  static void CreateSubGraph(ComputeGraphPtr graph, ComputeGraphPtr subgraph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    subgraph->SetParentNode(bn_node);
    subgraph->SetParentGraph(graph);
    graph->AddSubgraph(subgraph->GetName(), subgraph);
  }
  static void CreateSimpleGraph(ComputeGraphPtr graph) {
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv");

    int64_t int_value = 1;
    float float_value = 2.0;
    bool bool_value = false;
    string str_value = "abc";
    vector<int64_t> int_vec{1, 2, 3};
    vector<int64_t> rint_vec;
    vector<float> float_vec{4.0, 5.0, 6.0};
    vector<float> rfloat_vec;
    vector<bool> bool_vec{false, true, true};
    vector<bool> rbool_vec;
    std::vector<string> str_vec{"a", "b", "c"};
    AttrUtils::SetInt(op_desc_ptr, "transposX", int_value);
    AttrUtils::SetFloat(op_desc_ptr, "transposY", float_value);
    AttrUtils::SetBool(op_desc_ptr, "attrBool", bool_value);
    AttrUtils::SetStr(op_desc_ptr, "attrStr", str_value);
    AttrUtils::SetListInt(op_desc_ptr, "attrListInt", int_vec);
    AttrUtils::SetListFloat(op_desc_ptr, "attrListFloat", float_vec);
    AttrUtils::SetListBool(op_desc_ptr, "attrListBool", bool_vec);
    AttrUtils::SetListStr(op_desc_ptr, "attrListStr", str_vec);

    ge::DataType set_dtype = ge::DT_FLOAT16;
    std::vector<int64_t> shape_vec{256, 256, 512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    vector<std::pair<int64_t, int64_t>> range({{1, 512}, {256, 256}, {512, 512}});

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    input0_desc_ptr->SetOriginShape(shape_desc);
    input0_desc_ptr->SetOriginShapeRange(range);
    input0_desc_ptr->SetValueRange(range);
    op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetOriginShape(shape_desc);
    input1_desc_ptr->SetShape(shape_desc);
    input1_desc_ptr->SetOriginShapeRange(range);
    input1_desc_ptr->SetValueRange(range);
    op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

    std::vector<bool> is_input_const;
    is_input_const.emplace_back(false);
    is_input_const.emplace_back(true);
    op_desc_ptr->SetIsInputConst(is_input_const);

    shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
    output_desc_ptr->SetDataType(set_dtype);
    output_desc_ptr->SetShape(shape_desc);
    output_desc_ptr->SetOriginShape(shape_desc);
    output_desc_ptr->SetOriginShapeRange(range);
    output_desc_ptr->SetValueRange(range);
    op_desc_ptr->AddOutputDesc("z", output_desc_ptr->Clone());

    AttrUtils::SetInt(op_desc_ptr, "imply_type", EN_IMPL_HW_TBE);
    NodePtr conv_node = graph->AddNode(op_desc_ptr);
    op_desc_ptr->SetName("conv2");
    NodePtr conv_next_node = graph->AddNode(op_desc_ptr);
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv_next_node->GetInDataAnchor(0));
  }

  static void CreateSingleNodeGraph(ComputeGraphPtr graph) {
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    shared_ptr<ge::GeTensorDesc> in_desc1 = make_shared<ge::GeTensorDesc>();
    in_desc1->SetDataType(DT_FLOAT16);
    in_desc1->SetFormat(FORMAT_NCHW);
    in_desc1->SetShape(shape);
    relu_op->AddInputDesc("x", in_desc1->Clone());
    data->AddOutputDesc("x", in_desc1->Clone());
    data->AddInputDesc("x", in_desc1->Clone());

    shared_ptr<ge::GeTensorDesc> out_desc1 = make_shared<ge::GeTensorDesc>();
    out_desc1->SetDataType(DT_FLOAT16);
    out_desc1->SetFormat(FORMAT_NCHW);
    out_desc1->SetShape(shape);
    relu_op->AddOutputDesc("y", out_desc1->Clone());

    ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateSingleNodeGraph2(ComputeGraphPtr graph) {
    OpDescPtr max_pool_op = std::make_shared<OpDesc>("maxpool", "MaxPoolV3");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    shared_ptr<ge::GeTensorDesc> in_desc1 = make_shared<ge::GeTensorDesc>();
    in_desc1->SetDataType(DT_FLOAT16);
    in_desc1->SetFormat(FORMAT_NCHW);
    in_desc1->SetShape(shape);
    max_pool_op->AddInputDesc("x", in_desc1->Clone());
    data->AddOutputDesc("x", in_desc1->Clone());
    data->AddInputDesc("x", in_desc1->Clone());

    shared_ptr<ge::GeTensorDesc> out_desc1 = make_shared<ge::GeTensorDesc>();
    out_desc1->SetDataType(DT_FLOAT16);
    out_desc1->SetFormat(FORMAT_NCHW);
    out_desc1->SetShape(shape);
    max_pool_op->AddOutputDesc("y", out_desc1->Clone());

    ge::AttrUtils::SetInt(max_pool_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    NodePtr relu_node = graph->AddNode(max_pool_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateTwoOpDescGraph(ComputeGraphPtr graph, bool set_fusion_scope_flag = false) {
      OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
      OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
      OpDescPtr max_op = std::make_shared<OpDesc>("max", "Maximum");
      OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");

      // add descriptor
      vector<int64_t> dims = {1,2,3,4};
      GeShape shape(dims);

      GeTensorDesc in_desc1(shape);
      in_desc1.SetFormat(FORMAT_NCHW);
      in_desc1.SetDataType(DT_FLOAT16);
      relu_op->AddInputDesc("x", in_desc1);

      GeTensorDesc out_desc1(shape);
      out_desc1.SetFormat(FORMAT_HWCN);
      out_desc1.SetDataType(DT_FLOAT16);
      relu_op->AddOutputDesc("y", out_desc1);

      GeTensorDesc in_desc2(shape);
      in_desc2.SetFormat(FORMAT_FRACTAL_Z);
      in_desc2.SetDataType(DT_FLOAT16);
      bn_op->AddInputDesc("x", in_desc2);

      GeTensorDesc out_desc2(shape);
      out_desc2.SetFormat(FORMAT_NHWC);
      out_desc2.SetDataType(DT_FLOAT16);
      bn_op->AddOutputDesc("y", out_desc2);
      std::vector<bool> is_in_const_vec = {false};
      bn_op->SetIsInputConst(is_in_const_vec);

      GeTensorDesc in_desc3(shape);
      in_desc3.SetFormat(FORMAT_FRACTAL_Z);
      in_desc3.SetDataType(DT_FLOAT16);
      max_op->AddInputDesc("x", in_desc3);

      GeTensorDesc in_desc4(shape);
      in_desc4.SetFormat(FORMAT_FRACTAL_Z);
      in_desc4.SetDataType(DT_FLOAT16);
      max_op->AddInputDesc("y", in_desc4);

      GeTensorDesc out_desc3(shape);
      out_desc3.SetFormat(FORMAT_NHWC);
      out_desc3.SetDataType(DT_FLOAT16);
      max_op->AddOutputDesc("z", out_desc3);

      GeTensorDesc out_desc4(shape);
      out_desc4.SetFormat(FORMAT_NHWC);
      out_desc4.SetDataType(DT_FLOAT16);
      const_op->AddOutputDesc("z", out_desc4);

      ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
      ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
      ge::AttrUtils::SetInt(max_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

      NodePtr bn_node = graph->AddNode(bn_op);
      NodePtr relu_node = graph->AddNode(relu_op);
      NodePtr const_node = graph->AddNode(const_op);
      NodePtr max_node = graph->AddNode(max_op);

      GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(1));
      if (set_fusion_scope_flag) {
        ge::AttrUtils::SetInt(bn_op, "fusion_scope", -1);
        ge::AttrUtils::SetInt(relu_op, "fusion_scope", -2);
        ge::AttrUtils::SetInt(max_op, "fusion_scope", -3);
      }
  }

  static void CreateTwoOpDescGraph2(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {0,2,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {1,2,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {1,2,3,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {1,2,3,4};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateUnknownShapeGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {0,-1,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {1,-1,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {1,2,-1,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {1,2,3,-1};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateTwoOpDescGraph3(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {1,2,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {0,2,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {1,2,3,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {1,2,3,4};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateTwoOpDescGraph4(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {1,2,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {1,2,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {0,2,3,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {1,2,3,4};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateTwoOpDescGraph5(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {1,2,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {1,2,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {1,2,3,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {0,2,3,4};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateTwoOpDescGraph6(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {1,2,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {1,0,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {1,2,3,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {1,2,3,4};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateTwoOpDescGraph7(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    // add descriptor
    vector<int64_t> dims1 = {1,2,3,4};
    GeShape shape1(dims1);
    vector<int64_t> dims2 = {1,2,3,4};
    GeShape shape2(dims2);
    vector<int64_t> dims3 = {1,2,3,4};
    GeShape shape3(dims3);
    vector<int64_t> dims4 = {1,0,3,4};
    GeShape shape4(dims4);

    GeTensorDesc in_desc1(shape1);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x1", in_desc1);

    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x2", in_desc2);

    GeTensorDesc out_desc1(shape3);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape4);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y2", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
  }

  static void CreateSplitOpDescGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr split_op = std::make_shared<OpDesc>("split", "SplitD");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    // add descriptor
    vector<int64_t> dims = {1, 2};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetOriginShape(shape);
    in_desc1.SetDataType(DT_FLOAT16);
    split_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetOriginShape(shape);
    out_desc1.SetDataType(DT_FLOAT16);
    split_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginShape(shape);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);


    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetOriginShape(shape);
    in_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_HWCN);
    out_desc4.SetOriginShape(shape);
    out_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc4);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(split_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(split_op, SPLIT_DIM, -4);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr split_node = graph->AddNode(split_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        split_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
  }

  static void CreateConstSplitOpDescGraph(ComputeGraphPtr graph) {
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    OpDescPtr split_op = std::make_shared<OpDesc>("split", "SplitD");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    // add descriptor
    vector<int64_t> dims = {1, 2};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetOriginShape(shape);
    in_desc1.SetDataType(DT_FLOAT16);
    split_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetOriginShape(shape);
    out_desc1.SetDataType(DT_FLOAT16);
    split_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetOriginShape(shape);
    out_desc2.SetDataType(DT_FLOAT16);
    const_op->AddOutputDesc("y", out_desc2);


    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetOriginShape(shape);
    in_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_NCHW);
    out_desc4.SetOriginShape(shape);
    out_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc4);

    ge::AttrUtils::SetInt(const_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(split_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(split_op, SPLIT_DIM, 0);
    NodePtr const_node = graph->AddNode(const_op);
    NodePtr split_node = graph->AddNode(split_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                        split_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
  }

  static void CreateDataSplitOpDescGraph(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("data", DATA);
    OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);
  
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
    tensor_desc1.SetOriginShape(shape1);
    data->AddOutputDesc(tensor_desc1);
    split->AddInputDesc(tensor_desc1);
  
    ge::GeShape shape2({1,4,9,16});
    GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
    tensor_desc2.SetOriginShape(shape2);
    split->AddOutputDesc(tensor_desc2);
    split->AddOutputDesc(tensor_desc2);
    relu1->AddInputDesc(tensor_desc2);
    relu2->AddInputDesc(tensor_desc2);
  
    (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 0);
    (void)ge::AttrUtils::SetInt(relu1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetInt(relu2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  
    NodePtr data_node = graph->AddNode(data);
    NodePtr split_node = graph->AddNode(split);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
  
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            split_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0),
                            relu1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1),
                            relu2_node->GetInDataAnchor(0));
  }

  static void CreateConcatOpDescGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    // add descriptor
    vector<int64_t> dims = {1, 2};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetOriginShape(shape);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetOriginShape(shape);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetOriginShape(shape);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginShape(shape);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetOriginShape(shape);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetOriginShape(shape);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetOriginShape(shape);
    in_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_HWCN);
    out_desc4.SetOriginShape(shape);
    out_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc4);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, -4);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    NodePtr relu_node = graph->AddNode(relu_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
  }

  static void CreateConcatOpDescGraph2(ComputeGraphPtr graph) {
    OpDescPtr placeholder_op =
        std::make_shared<OpDesc>("placeholder", "PlaceHolder");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc1.SetOriginShape(shape);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc1.SetOriginShape(shape);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    placeholder_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    placeholder_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    placeholder_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(placeholder_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 1);
    NodePtr placeholder_node = graph->AddNode(placeholder_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(placeholder_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph3(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph4(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_CONTINUOUS_OUTPUT, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph5(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_REFERENCE, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph6(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_HWCN);
    out_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc4);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
  }

  static void CreateConcatOpDescGraph7(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);
    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_HWCN);
    out_desc4.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc4);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
  }

  static void CreateConcatOpDescGraph8(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 1);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph9(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    vector<int64_t> output_index;
    output_index.push_back(0);
    (void)ge::AttrUtils::SetListInt(bn_op, ge::ATOMIC_ATTR_OUTPUT_INDEX,
                                    output_index);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph10(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
  }

  static void CreateConcatOpDescGraph11(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(shape_op, ge::ATTR_NAME_REFERENCE, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }

  static void CreateConcatOpDescGraph12(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc in_desc111(shape);
    in_desc111.SetFormat(FORMAT_NCHW);
    in_desc111.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("w", in_desc111);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(2));
  }
  static void CreateConcatOpDescGraph13(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
  }
  static void CreateConcatOpDescGraph14(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(shape_node->GetOutControlAnchor(),
                        concat_node->GetInControlAnchor());
  }
  static void CreateConcatOpDescGraph15(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
    OpDescPtr end_op = std::make_shared<OpDesc>("end", "End");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_Z);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_HWCN);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetDataType(DT_FLOAT16);
    end_op->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_HWCN);
    out_desc4.SetDataType(DT_FLOAT16);
    end_op->AddOutputDesc("y", out_desc4);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    ge::AttrUtils::SetStr(end_op, "parentOpType", "NetOutput");
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    NodePtr end_node = graph->AddNode(end_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                        end_node->GetInDataAnchor(0));
  }
  static void CreateConcatOpDescGraph16(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
    OpDescPtr reshape_op1 = std::make_shared<OpDesc>("reshape1", "Reshape");
    OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
    OpDescPtr reshape_op2 = std::make_shared<OpDesc>("reshape2", "Reshape");
    OpDescPtr end_op = std::make_shared<OpDesc>("end", "End");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("x", in_desc1);

    GeTensorDesc in_desc11(shape);
    in_desc11.SetFormat(FORMAT_NCHW);
    in_desc11.SetDataType(DT_FLOAT16);
    concat_op->AddInputDesc("z", in_desc11);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    concat_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddInputDesc("x", in_desc3);
    reshape_op1->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_NCHW);
    out_desc3.SetDataType(DT_FLOAT16);
    shape_op->AddOutputDesc("y", out_desc3);
    reshape_op1->AddOutputDesc("y", out_desc3);

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetDataType(DT_FLOAT16);
    end_op->AddInputDesc("x", in_desc4);
    reshape_op2->AddInputDesc("x", in_desc4);

    GeTensorDesc out_desc4(shape);
    out_desc4.SetFormat(FORMAT_NCHW);
    out_desc4.SetDataType(DT_FLOAT16);
    end_op->AddOutputDesc("y", out_desc4);
    reshape_op2->AddOutputDesc("y", out_desc4);

    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(shape_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(reshape_op1, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(reshape_op2, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
    ge::AttrUtils::SetStr(end_op, "parentOpType", "NetOutput");
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr concat_node = graph->AddNode(concat_op);
    NodePtr shape_node = graph->AddNode(shape_op);
    NodePtr reshape_node1 = graph->AddNode(reshape_op1);
    NodePtr end_node = graph->AddNode(end_op);
    NodePtr reshape_node2 = graph->AddNode(reshape_op2);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0),
                        reshape_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(reshape_node1->GetOutDataAnchor(0),
                        concat_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                        reshape_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(reshape_node2->GetOutDataAnchor(0),
                        end_node->GetInDataAnchor(0));
  }
  static ComputeGraphPtr CreateCastReluCastGraph6() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Cast");
    OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("loss_scale/gradients/fp32_vars/conv2d_15/Conv2D_grad/Conv2DBackpropInput_dilation", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("loss_scale/gradients/fp32_vars/conv2d_15/Conv2D_grad/Conv2DBackpropInput_dilation", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_cast3->AddInputDesc(tensor_desc_c);
    op_desc_cast3->AddOutputDesc(tensor_desc_d);

    op_desc_cast4->AddInputDesc(tensor_desc_c);
    op_desc_cast4->AddOutputDesc(tensor_desc_c);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_c);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
    NodePtr node_cast4 = graph->AddNode(op_desc_cast4);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);
    (void)ge::AttrUtils::SetInt(node_cast1->GetOpDesc(), kThreadScopeId, 1);
    (void)ge::AttrUtils::SetInt(node_cast3->GetOpDesc(), kThreadScopeId, 2);
    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(2));

    return graph;
  }
  static void CreateConv2dFixpipeGraph(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
    OpDescPtr fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
    OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 3};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NHWC);
    in_desc2.SetOriginFormat(FORMAT_NHWC);
    in_desc2.SetDataType(DT_FLOAT16);
    data->AddOutputDesc("x", in_desc2);
    conv2d->AddInputDesc("x", in_desc2);
    conv2d->AddOutputDesc("y", in_desc2);
    fixpipe->AddInputDesc("x", in_desc2);
    fixpipe->AddOutputDesc("y", in_desc2);
    out->AddInputDesc("x", in_desc2);

    ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(fixpipe, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    NodePtr data_node = graph->AddNode(data);
    NodePtr conv2d_node = graph->AddNode(conv2d);
    NodePtr fixpipe_node = graph->AddNode(fixpipe);
    NodePtr out_node = graph->AddNode(out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }
  static void CreateCMOMultiStreamGraph(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr opdesc_a = std::make_shared<OpDesc>("A", "A"); opdesc_a->SetId(0); opdesc_a->SetStreamId(1);
    OpDescPtr opdesc_b = std::make_shared<OpDesc>("B", "B"); opdesc_b->SetId(1); opdesc_b->SetStreamId(1);
    OpDescPtr opdesc_c = std::make_shared<OpDesc>("C", "C"); opdesc_c->SetId(2); opdesc_c->SetStreamId(1);
    OpDescPtr opdesc_d = std::make_shared<OpDesc>("D", "D"); opdesc_d->SetId(3); opdesc_d->SetStreamId(1);
    OpDescPtr opdesc_e = std::make_shared<OpDesc>("E", "E"); opdesc_e->SetId(0); opdesc_e->SetStreamId(2);
    OpDescPtr opdesc_f = std::make_shared<OpDesc>("F", "F"); opdesc_f->SetId(1); opdesc_f->SetStreamId(2);
    OpDescPtr opdesc_g = std::make_shared<OpDesc>("G", "G"); opdesc_g->SetId(2); opdesc_g->SetStreamId(2);
    OpDescPtr opdesc_h = std::make_shared<OpDesc>("H", "H"); opdesc_h->SetId(3); opdesc_h->SetStreamId(2);
    OpDescPtr opdesc_j = std::make_shared<OpDesc>("J", "J"); opdesc_j->SetId(4); opdesc_j->SetStreamId(2);
    OpDescPtr opdesc_send = std::make_shared<OpDesc>("send", "Send"); opdesc_send->SetId(4); opdesc_send->SetStreamId(1);
    OpDescPtr opdesc_recv = std::make_shared<OpDesc>("recv", "Recv"); opdesc_recv->SetId(4); opdesc_recv->SetStreamId(2);
    OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");
    AttrUtils::SetInt(opdesc_a, ATTR_NAME_OP_READ_WRITE_INDEX, 0);
    AttrUtils::SetInt(opdesc_b, ATTR_NAME_OP_READ_WRITE_INDEX, 1);
    AttrUtils::SetInt(opdesc_c, ATTR_NAME_OP_READ_WRITE_INDEX, 2);
    AttrUtils::SetInt(opdesc_d, ATTR_NAME_OP_READ_WRITE_INDEX, 3);
    AttrUtils::SetInt(opdesc_e, ATTR_NAME_OP_READ_WRITE_INDEX, 0);
    AttrUtils::SetInt(opdesc_f, ATTR_NAME_OP_READ_WRITE_INDEX, 1);
    AttrUtils::SetInt(opdesc_g, ATTR_NAME_OP_READ_WRITE_INDEX, 2);
    AttrUtils::SetInt(opdesc_h, ATTR_NAME_OP_READ_WRITE_INDEX, 3);
    AttrUtils::SetInt(opdesc_j, ATTR_NAME_OP_READ_WRITE_INDEX, 4);
    // add descriptor
    vector<int64_t> dims = {1, 16, 16, 32};
    GeShape shape(dims);
    GeTensorDesc in_desc2(shape);

    data->AddOutputDesc("x", in_desc2);
    opdesc_a->AddInputDesc("x", in_desc2);
    opdesc_a->AddOutputDesc("y", in_desc2);
    opdesc_b->AddInputDesc("x", in_desc2);
    opdesc_b->AddOutputDesc("y", in_desc2);
    opdesc_c->AddInputDesc("x", in_desc2);
    opdesc_c->AddOutputDesc("y", in_desc2);
    opdesc_d->AddInputDesc("x", in_desc2);
    opdesc_d->AddOutputDesc("y", in_desc2);
    opdesc_e->AddInputDesc("x", in_desc2);
    opdesc_e->AddOutputDesc("y", in_desc2);
    opdesc_f->AddInputDesc("x", in_desc2);
    opdesc_f->AddOutputDesc("y", in_desc2);
    opdesc_g->AddInputDesc("x", in_desc2);
    opdesc_g->AddOutputDesc("y", in_desc2);
    opdesc_h->AddInputDesc("x", in_desc2);
    opdesc_h->AddOutputDesc("y", in_desc2);
    opdesc_j->AddInputDesc("x", in_desc2);
    opdesc_j->AddOutputDesc("y", in_desc2);
    out->AddInputDesc("x1", in_desc2);
    out->AddInputDesc("x2", in_desc2);

    ge::AttrUtils::SetInt(opdesc_a, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_b, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_c, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_e, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_f, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_g, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_h, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_j, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(opdesc_send, "event_id", 1);
    ge::AttrUtils::SetInt(opdesc_recv, "event_id", 1);
    NodePtr data_node = graph->AddNode(data);
    NodePtr node_a = graph->AddNode(opdesc_a);
    NodePtr node_b = graph->AddNode(opdesc_b);
    NodePtr node_c = graph->AddNode(opdesc_c);
    NodePtr node_d = graph->AddNode(opdesc_d);
    NodePtr node_e = graph->AddNode(opdesc_e);
    NodePtr node_f = graph->AddNode(opdesc_f);
    NodePtr node_g = graph->AddNode(opdesc_g);
    NodePtr node_h = graph->AddNode(opdesc_h);
    NodePtr node_j = graph->AddNode(opdesc_j);
    NodePtr out_node = graph->AddNode(out);
    NodePtr send = graph->AddNode(opdesc_send);
    NodePtr recv = graph->AddNode(opdesc_recv);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_b->GetOutControlAnchor(), send->GetInControlAnchor());
    GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    GraphUtils::AddEdge(recv->GetOutControlAnchor(), node_e->GetInControlAnchor());
    GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_j->GetOutDataAnchor(0), out_node->GetInDataAnchor(1));
  }
  static void CreateSwitchMergeFixpipeGraph(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
    OpDescPtr switch_op = std::make_shared<OpDesc>("switch", "Switch");
    OpDescPtr merge = std::make_shared<OpDesc>("merge", "Merge");
    OpDescPtr fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
    OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 3};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NHWC);
    in_desc2.SetOriginFormat(FORMAT_NHWC);
    in_desc2.SetDataType(DT_FLOAT16);
    data->AddOutputDesc("x", in_desc2);
    conv2d->AddInputDesc("x", in_desc2);
    conv2d->AddOutputDesc("y", in_desc2);
    switch_op->AddInputDesc("x", in_desc2);
    switch_op->AddOutputDesc("y", in_desc2);
    merge->AddInputDesc("x", in_desc2);
    merge->AddOutputDesc("y", in_desc2);
    fixpipe->AddInputDesc("x", in_desc2);
    fixpipe->AddOutputDesc("y", in_desc2);
    out->AddInputDesc("x", in_desc2);

    ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(fixpipe, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(switch_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(merge, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    NodePtr data_node = graph->AddNode(data);
    NodePtr conv2d_node = graph->AddNode(conv2d);
    NodePtr switch_node = graph->AddNode(switch_op);
    NodePtr merge_node = graph->AddNode(merge);
    NodePtr fixpipe_node = graph->AddNode(fixpipe);
    NodePtr out_node = graph->AddNode(out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  ge::ComputeGraphPtr CreateInceptionV3NetGraph() {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("inceptionv3");
    string network_path = GetNetworkPath("inceptionv3_aipp_int8_16batch.txt");
    (void)ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
    return graph;
  }

 static void CreateSwitchMergeFixpipeGraph2(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
    OpDescPtr switch_op = std::make_shared<OpDesc>("switch", "Switch");
    OpDescPtr merge = std::make_shared<OpDesc>("merge", "Merge");
    OpDescPtr fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
    OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");
    OpDescPtr quant = std::make_shared<OpDesc>("quant", "AscendQuant");
    OpDescPtr bias = std::make_shared<OpDesc>("bias", "QuantBiasOptimization");
    OpDescPtr const_op = std::make_shared<OpDesc>("cosnt", "Const");
    OpDescPtr transdata = std::make_shared<OpDesc>("trans", "TransData");
    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 3};
    GeShape shape(dims);
    vector<int64_t> dims1 = {1, 2, 3, 3, 1};
    GeShape shape1(dims1);

    GeTensorDesc in_desc1(shape1);
    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NHWC);
    in_desc2.SetOriginFormat(FORMAT_NHWC);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc1.SetFormat(FORMAT_NC1HWC0);
    in_desc1.SetFormat(FORMAT_NHWC);
    in_desc1.SetDataType(DT_FLOAT16);
    out->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    conv2d->AddInputDesc("x1", in_desc2);
    conv2d->AddInputDesc("x2", in_desc2);
    conv2d->AddInputDesc("x3", in_desc2);
    conv2d->AddOutputDesc("y", in_desc2);
    switch_op->AddInputDesc("x", in_desc2);
    switch_op->AddOutputDesc("y", in_desc1);
    merge->AddInputDesc("x", in_desc2);
    merge->AddOutputDesc("y", in_desc2);
    fixpipe->AddInputDesc("x", in_desc2);
    fixpipe->AddOutputDesc("y", in_desc2);
    quant->AddInputDesc("x", in_desc2);
    quant->AddOutputDesc("y", in_desc2);
    bias->AddInputDesc("x", in_desc2);
    bias->AddOutputDesc("y", in_desc2);
    const_op->AddOutputDesc("y", in_desc2);
    transdata->AddInputDesc("x", in_desc1);
    transdata->AddOutputDesc("y", in_desc2);

    ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(fixpipe, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(switch_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(merge, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(transdata, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(bias, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(quant, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    NodePtr data_node = graph->AddNode(data);
    NodePtr conv2d_node = graph->AddNode(conv2d);
    NodePtr switch_node = graph->AddNode(switch_op);
    NodePtr merge_node = graph->AddNode(merge);
    NodePtr fixpipe_node = graph->AddNode(fixpipe);
    NodePtr out_node = graph->AddNode(out);
    NodePtr quant_node = graph->AddNode(quant);
    NodePtr bias_node = graph->AddNode(bias);
    NodePtr const_node = graph->AddNode(const_op);
    NodePtr transdata_node = graph->AddNode(transdata);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), quant_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(quant_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));    
    GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }
};

namespace {
std::string GetGeContextBuildModeOptionValue(Configuration *This, const std::string &key)
{
  std::string value = "tuning";
  return value;
}

std::string GetGeContextBuildStepOptionValue(Configuration *This, const std::string &key)
{
  std::string value = "tuning";
  return value;
}
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, multi_thread_judge)
{
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
  std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  auto graph = CreateInceptionV3NetGraph();
  Status ret = fe_graph_optimizer_ptr->op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  ret = fe_graph_optimizer_ptr->format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto graph2 = CreateMultiThreadGraph();
  ret = fe_graph_optimizer_ptr->format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph2);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, get_attributes_success)
{
  GraphOptimizerAttribute attrs;
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);

  Status status = fe_graph_optimizer_ptr->GetAttributes(attrs);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, init_opcompiler)
{
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->InitializeAllOpCompiler();

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, finalize_success1)
{
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->Finalize();

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, finalize_success2)
{
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  Status status = fe_graph_optimizer_ptr->Finalize();

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, finalize_session_info_success1)
{
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  auto graph = std::make_shared<ComputeGraph>("test");
  Status status = fe_graph_optimizer_ptr->FinalizeSessionInfo(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_failed)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraph(*graph);

  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->graph_fusion_ptr_ = graph_fusion_ptr_;
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraph(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, concat_split_optimizer_success1)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  bool need_set_virtual_op = true;
  Status status = fe_graph_optimizer_ptr->SplitOptimizer(*graph, need_set_virtual_op);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, concat_split_optimizer_success2)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  bool need_set_virtual_op = false;
  Status status = fe_graph_optimizer_ptr->SplitOptimizer(*graph, need_set_virtual_op);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, post_process_after_compiling_op_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  BufferFusionPtr buffer_fusion_ptr_ = std::make_shared<BufferFusion>(graph_comm_ptr,
                                                                      fusion_priority_mgr_ptr_, nullptr);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  Status status = fe_graph_optimizer_ptr->PostProcessAfterCompilingOp(*graph, buffer_fusion_ptr_);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, alloc_resouce_test1)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  ge::NodePtr bn_node = graph->FindNode("batchnormal");
  ge::NodePtr relu = graph->FindNode("relu");
  graph->SetGraphUnknownFlag(false);
  ge::AttrUtils::SetStr(bn_node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->AllocMixResource(*graph);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_STREAM_INFO), true);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_NOTIFY_INFO), true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, alloc_resouce_test2)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  ge::NodePtr bn_node = graph->FindNode("batchnormal");
  ge::NodePtr relu = graph->FindNode("relu");
  graph->SetGraphUnknownFlag(true);
  graph->SetParentNode(bn_node);
  ge::AttrUtils::SetStr(bn_node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  ge::AttrUtils::SetBool(bn_node->GetOpDesc(), ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE, true);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->AllocMixResource(*graph);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_STREAM_INFO), false);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_NOTIFY_INFO), false);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ATTR_NAME_DISABLE_MIX_VECTOR_CORE), true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_fused_graph_after_graph_slice_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraphAfterGraphSlice(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_fused_graph_after_graph_slice_with_compiled_fusionop_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();

  for (auto& node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_COMPIED_FUSION_OP, true);
    break;
  }

  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraphAfterGraphSlice(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_fused_graph_after_graph_slice_with_tuneformat_fail)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  for (auto& node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_COMPIED_FUSION_OP, true);
    break;
  }
  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraphAfterGraphSlice(*graph);

  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_whole_graph_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeWholeGraph(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
  auto options_bk = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.buildMode"] = "tuning";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  auto graph_2 = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph_2);
  status = fe_graph_optimizer_ptr->OptimizeWholeGraph(*graph_2);
  EXPECT_EQ(fe::SUCCESS, status);
  ge::GetThreadLocalContext().SetGraphOption(options_bk);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphBeforeBuild_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr_;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphBeforeBuild(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphBeforeBuild_del_fusion_scope)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph, true);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr_;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphBeforeBuild(*graph);
  for (auto& node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetOpDesc()->HasAttr("fusion_scope"), false);
  }
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphPrepare_failed1)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateTwoOpDescGraph(graph);
    auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
    fe_graph_optimizer_ptr->init_flag_ = false;
    Status status = fe_graph_optimizer_ptr->OptimizeGraphPrepare(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_atomic_add_info_success)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateTwoOpDescGraph(graph);

    auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
    //fe_graph_optimizer_ptr->init_flag_ = true;
    for (auto node : graph->GetDirectNode()) {
        string op_type = node->GetType();
        if (op_type == OP_TYPE_PLACE_HOLDER ||
            op_type == OP_TYPE_END) {
            continue;
        }
        ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
        if (!ge::AttrUtils::HasAttr(op_desc_ptr, FE_IMPLY_TYPE)) {
            continue;
        }
        int tmp_imply_type = -1;
        ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type);
        OpImplType op_impl_type = (OpImplType)tmp_imply_type;
        if (op_desc_ptr->GetName() == "batchnormal") {
            std::vector<uint32_t> tmp_output_index {1, 0, 0};
            bool output_index = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);

            std::vector<int64_t> tmp_wk_index {1, 1, 1};
            bool atomic = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_INDEX, tmp_wk_index);
            op_desc_ptr->SetWorkspaceBytes({32, 32, 32});
            EXPECT_EQ(output_index, true);
            EXPECT_EQ(atomic, true);
        }
        if (op_desc_ptr->GetName() == "relu") {
            ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
            std::vector<uint32_t> tmp_output_index {1};
            bool output_index2 = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);

            std::vector<int64_t> tmp_wk_index {1, 1, 1};
            bool atomic2 = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_INDEX, tmp_wk_index);
            op_desc_ptr->SetWorkspaceBytes({32, 32, 32});
            EXPECT_EQ(output_index2, true);
            EXPECT_EQ(atomic2, true);
        }
    }
    vector<ge::NodePtr> atomic_node_vec;
    string atomic_clean_policy = "0";
    Status status = fe_graph_optimizer_ptr->GetAndPreProcessForAtomicNodes(*(graph.get()), atomic_node_vec, atomic_clean_policy);
    EXPECT_EQ(fe::SUCCESS, status);
    atomic_clean_policy = "1";
    status = fe_graph_optimizer_ptr->GetAndPreProcessForAtomicNodes(*(graph.get()), atomic_node_vec, atomic_clean_policy);
    EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_atomic_add_info_success2)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateTwoOpDescGraph(graph);

    auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
        //fe_graph_optimizer_ptr->init_flag_ = true;
        for (auto node : graph->GetDirectNode()) {
        string op_type = node->GetType();
        if (op_type == OP_TYPE_PLACE_HOLDER ||
        op_type == OP_TYPE_END) {
        continue;
        }
        ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
        if (!ge::AttrUtils::HasAttr(op_desc_ptr, FE_IMPLY_TYPE)) {
            continue;
        }
        int tmp_imply_type = -1;
        ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type);
        OpImplType op_impl_type = (OpImplType)tmp_imply_type;

        if (op_desc_ptr->GetName() == "batchnormal") {
            std::vector<int64_t> tmp_wk_index {0, 0};
            bool atomic = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_FLAG, tmp_wk_index);
            op_desc_ptr->SetWorkspaceBytes({32, 32});
            EXPECT_EQ(atomic, true);
        }
        if (op_desc_ptr->GetName() == "relu") {
            ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
            std::vector<uint32_t> tmp_output_index {0, 1};
            bool output_index = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);
            EXPECT_EQ(output_index, true);
        }
    }
    vector<ge::NodePtr> atomic_node_vec;
    string atomic_clean_policy = "0";
    Status status = fe_graph_optimizer_ptr->GetAndPreProcessForAtomicNodes(*(graph.get()), atomic_node_vec, atomic_clean_policy);
    EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_judge_c04_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
  std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ =
  std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
      fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ =
  std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
                                                                            ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 1;
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 0;
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_judge_insert_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
      std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ =
      std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
      fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ =
      std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
                                                                            ops_kernel_info_store_ptr_,
                                                                            fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_judge_insert_success1)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    std::string subgraph_name = "subgraph";
    ge::ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
    CreateSubGraph(graph, subgraph);
    auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
    fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
    std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
    fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ = 
    std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
    fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = 
    std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
    FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
          fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
    fusion_priority_mgr_ptr_->Initialize();
    fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
          ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
    fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
    fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
    Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);
    status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_judge_with_aclnn_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bm_aclnn_only = std::make_shared<OpDesc>("bm_aclnn_only", "BatchMatMulAclnnOnly");

  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(bm_aclnn_only, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr bm_aclnn_only_node = graph->AddNode(bm_aclnn_only);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
      fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
                                                                            ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 1;
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  // EXPECT_EQ(fe::FAILED, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 0;
  // EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_fusion_virtual_op_success1_split) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSplitOpDescGraph(graph);
  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == "SplitD") {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_fusion_virtual_op_success2_split) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConstSplitOpDescGraph(graph);
  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == "SplitD") {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, single_op_scene_notask_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)AttrUtils::SetBool(graph, "_single_op_scene", true);
  (void)AttrUtils::SetBool(graph, "_memory_discontiguous_allocation", false);

  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

  ge::GeShape shape1({1,4,9,16});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  relu1->AddOutputDesc(tensor_desc1);
  relu2->AddOutputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);

  ge::GeShape shape2({2,4,9,16});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  concat->AddOutputDesc(tensor_desc2);
  netoutput->AddInputDesc(tensor_desc2);

  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 0);
  (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  NodePtr concat_node = graph->AddNode(concat);
  NodePtr netoutput_node = graph->AddNode(netoutput);

  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                          netoutput_node->GetInDataAnchor(0));
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->SplitOptimizer(*graph, true);
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == CONCATD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, single_op_scene_check_export_compile_stat_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)AttrUtils::SetBool(graph, "_single_op_scene", true);
  (void)AttrUtils::SetBool(graph, "_memory_discontiguous_allocation", false);

  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

  ge::GeShape shape1({1,4,9,16});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  relu1->AddOutputDesc(tensor_desc1);
  relu2->AddOutputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);

  ge::GeShape shape2({2,4,9,16});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  concat->AddOutputDesc(tensor_desc2);
  netoutput->AddInputDesc(tensor_desc2);

  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 0);
  (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  NodePtr concat_node = graph->AddNode(concat);
  NodePtr netoutput_node = graph->AddNode(netoutput);

  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                          concat_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                          netoutput_node->GetInDataAnchor(0));
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  EXPECT_EQ(fe_graph_optimizer_ptr->CheckExportFusionResCond(*graph), false);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, not_after_compile_compile_check_export_compile_stat_failed) {
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ExportCompileStat)] = 1;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  EXPECT_EQ(fe_graph_optimizer_ptr->CheckExportFusionResCond(*graph), false);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, data_split_notask_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);
  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, true);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_root_graph_unknown_notask_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);

  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  ge::AttrUtils::SetBool(root_graph, "_graph_unknown_flag", true);
  graph->SetParentGraph(root_graph);

  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, true);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_owner_graph_unknown_notask_fail1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);
  ge::AttrUtils::SetBool(graph, "_dynamic_shape_partitioned", true);

  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_owner_graph_unknown_notask_fail2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);
  ge::AttrUtils::SetBool(graph, "_graph_unknown_flag", true);

  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_stage1_case1) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr transdata = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_4d_fp32(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp32(shape, FORMAT_NCHW, DT_FLOAT);

  data->AddOutputDesc(tenosr_desc_4d_fp32);
  transdata->AddInputDesc(tenosr_desc_4d_fp32);
  transdata->AddOutputDesc(tenosr_desc_5d_fp32);
  cast->AddInputDesc(tenosr_desc_5d_fp32);
  cast->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  PlatformUtils::Instance().soc_version_ = "Ascend310";
  Status ret = fe_graph_optimizer_->OptimizeAfterStage1(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == "TransData") {
      EXPECT_EQ(node->GetOpDesc()->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(node->GetOpDesc()->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_stage1_case2) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr transdata = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_4d_fp32(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp32(shape, FORMAT_NCHW, DT_FLOAT);

  data->AddOutputDesc(tenosr_desc_4d_fp32);
  cast->AddInputDesc(tenosr_desc_4d_fp32);
  cast->AddOutputDesc(tenosr_desc_4d_fp16);
  transdata->AddInputDesc(tenosr_desc_4d_fp16);
  transdata->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  Status ret = fe_graph_optimizer_->OptimizeAfterStage1(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == "TransData") {
      EXPECT_EQ(node->GetOpDesc()->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(node->GetOpDesc()->MutableOutputDesc(0)->GetDataType(), DT_FLOAT16);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_nonfuzzy)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_precise"));
  ge::GetThreadLocalContext().SetGlobalOption(options);

  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  std::map<std::string, std::string> options1;
  options1.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", SHAPE_GENERALIZED));
  ge::GetThreadLocalContext().SetGlobalOption(options1);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fuzzy)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
  ge::GetThreadLocalContext().SetGlobalOption(options);

  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, feed_node_general_info_from_op_store){
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", "Matmul");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);

  matmul->AddInputDesc(tenosr_desc_4d_fp16);
  matmul->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr matmul_node = graph->AddNode(matmul);
  NodeGeneralInfoPtr node_info_ptr = std::make_shared<NodeGeneralInfo>();
  Status ret = tbe_adapter_ptr->FeedNodeGeneralInfoFromOpStore(matmul_node, node_info_ptr);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr transdata = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast1");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_4d_fp32(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp32(shape, FORMAT_NCHW, DT_FLOAT);

  data->AddOutputDesc(tenosr_desc_4d_fp32);
  cast->AddInputDesc(tenosr_desc_4d_fp32);
  cast->AddOutputDesc(tenosr_desc_4d_fp16);
  transdata->AddInputDesc(tenosr_desc_4d_fp16);
  transdata->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  std::map<ge::NodePtr, std::pair<bool, TbeOpInfoPtr>> node_info;

  Status ret = fe_graph_optimizer_->ShapeAndValueGeneralize(*graph);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_success1)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateConcatOpDescGraph10(graph);
    vector<int64_t> dims = {1, 2, 3, 32};
    vector<int64_t> shape_vec;
    tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
    tbe_adapter_ptr->TeGeneralize = teGeneralize;
    Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
    for (auto node : graph->GetDirectNode()) {
      auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() != "ConcatD") {
        continue;
      }
      auto tensor_desc = op_desc->MutableInputDesc("x");
      shape_vec = tensor_desc->GetShape().GetDims();
    }
    EXPECT_EQ(shape_vec, dims);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail1)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateSimpleGraph(graph);
    std::vector<int64_t> dims = {256, 256, 512};
    std::vector<int64_t> shape_vec;
    tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
    tbe_adapter_ptr->TeGeneralize = teGeneralize;

    Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
    for (auto node : graph->GetDirectNode()) {
      auto op_desc = node->GetOpDesc();
      auto tensor_desc_x = op_desc->MutableInputDesc("x");
      auto tensor_desc_y = op_desc->MutableInputDesc("y");
      shape_vec = tensor_desc_x->GetShape().GetDims();
      EXPECT_EQ(shape_vec, dims);
      shape_vec = tensor_desc_y->GetShape().GetDims();
      EXPECT_EQ(shape_vec, dims);
    }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail2)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateSingleNodeGraph(graph);
    std::vector<int64_t> dims = {1, 2, 3, 4};
    std::vector<int64_t> shape_vec;
    tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
    tbe_adapter_ptr->TeGeneralize = teGeneralize;
    Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
    for (auto node : graph->GetDirectNode()) {
      if (node->GetType() == fe::DATA) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      auto tensor_desc_x = op_desc->MutableInputDesc("x");
      shape_vec = tensor_desc_x->GetShape().GetDims();
    }
    EXPECT_EQ(shape_vec, dims);
    auto graph2 = std::make_shared<ComputeGraph>("test");
    CreateSingleNodeGraph2(graph2);

    status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph2.get()));
    EXPECT_EQ(fe::FAILED, status);
    for (auto node : graph2->GetDirectNode()) {
      if (node->GetType() == fe::DATA) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      auto tensor_desc_x = op_desc->MutableInputDesc("x");
      shape_vec = tensor_desc_x->GetShape().GetDims();
    }
    EXPECT_EQ(shape_vec, dims);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail3)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateBatchNormGraph(graph);
    vector<int64_t> shape_vec;
    tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
    tbe_adapter_ptr->TeGeneralize = teGeneralize;
    Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail4)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateBatchNormGraph(graph);
    vector<int64_t> shape_vec;
    tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
    tbe_adapter_ptr->TeGeneralize = teGeneralizeException;
    Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, test1)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;
  Status status = fe_graph_optimizer_->graph_fusion_ptr_->Fusion(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  ComputeGraphPtr parent_graph = std::make_shared<ComputeGraph>("parent_graph");
  auto parent_const = MakeNode(parent_graph, 0, 1, "parent_const", "Const");
  auto parent_case = MakeNode(parent_graph, 3, 1, "parent_case", "Case");
  auto parent_output = MakeNode(parent_graph, 1, 0, "parent_output", "NetOutput");

  GeTensorDesc tensor_desc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);

  parent_const->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(1, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(2, tensor_desc);
  parent_case->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(0));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(1));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(2));
  GraphUtils::AddEdge(parent_case->GetOutDataAnchor(0), parent_output->GetInDataAnchor(0));

  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  auto data0 = MakeNode(sub_graph, 1, 1, "data0", "Data");
  data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data1 = MakeNode(sub_graph, 1, 1, "data1", "Data");
  data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data2 = MakeNode(sub_graph, 1, 1, "data2", "Data");
  data2->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  (void)AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  (void)AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);

  sub_graph->SetParentNode(parent_case);
  sub_graph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);

  graph_fusion_ptr_->Fusion(*(parent_graph.get()));
  graph_fusion_ptr_->JudgeQuantMode(*(parent_graph.get()));
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, get_op_compiler_fail) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  OpCompilerPtr op_compiler;
  Status status = fe_graph_optimizer_ptr->GetOpCompiler(op_compiler);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue1) {
  FEOpsKernelInfoStorePtr stub_ops_kernel_info_store_ptr = std::make_shared<StubFEKernelInfoStore>(fe::AI_CORE_NAME);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(stub_ops_kernel_info_store_ptr, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->InsertClipByValue(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 6);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->InsertClipByValue(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 6);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", CONSTANT);
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", CONSTANT);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(const1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(const2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  const1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  const2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(const1);
  ge::NodePtr pld2 = graph->AddNode(const2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_,  AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->InsertClipByValue(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 9);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->CreateClipByValue(*graph, pld1, false);
  EXPECT_EQ(graph->GetDirectNode().size(), 3);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, convert_ext_attr2json) {
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr mul_node = graph->AddNode(mul);
  std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp =
      std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>();
  op_attrs_maps_tmp->insert({"mul",{{"Mul", "Mul"}}});
  mul_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  fe_graph_optimizer_ptr->ConvertExtAttr2Json(*graph, true);
  bool res = mul_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP);
  EXPECT_EQ(res, false);
  (void)ge::AttrUtils::SetListStr(mul_node->GetOpDesc(), kPassNameAttr, {"mul"});
  fe_graph_optimizer_ptr->ConvertJson2ExtAttr(*graph);
  std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp_check =
      std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>();
  op_attrs_maps_tmp_check = mul_node->GetOpDesc()->TryGetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp_check);
  auto iter = (*op_attrs_maps_tmp_check).begin();
  std::string pass_name = iter->first;
  auto op_attrs_vec = iter->second[0];
  EXPECT_EQ(pass_name, "mul");
  EXPECT_EQ(op_attrs_vec[0], "name:Mul");
  EXPECT_EQ(op_attrs_vec[1], "type:Mul");
  fe_graph_optimizer_ptr->ConvertExtAttr2Json(*graph, true);
  std::string json_str;
  (void)ge::AttrUtils::GetStr(mul_node->GetOpDesc(), ge::ATTR_NAME_ORIGIN_OP_ATTRS_IN_FUSION_PROCESS, json_str);
  EXPECT_EQ(json_str, "{\"mul\":[[\"name:Mul\",\"type:Mul\"]]}");
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "";
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, convert_ext_attr2json_fail) {
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr mul_node = graph->AddNode(mul);
  std::unordered_map<std::string, std::vector<std::string>> op_attrs_maps_tmp;
  op_attrs_maps_tmp.insert({"mul",{"qqqq"}});
  mul_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  (void)ge::AttrUtils::SetListStr(mul_node->GetOpDesc(), kPassNameAttr, {"mul"});
  fe_graph_optimizer_ptr->ConvertExtAttr2Json(*graph, false);
  fe_graph_optimizer_ptr->ConvertJson2ExtAttr(*graph);
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "";
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fused_sub_graph_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->lx_fusion_optimizer_ptr_ = lx_fusion_optimizer_;
  fe_graph_optimizer_ptr->InitializeAllOpCompiler();
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr_;
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->init_flag_ = true;
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_OFF_OPTIMIZE);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  OpCompilerPtr op_compiler_baseline_ptr =
      make_shared<OpCompilerBaseline>("baseline", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_baseline_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);

  OpCompilerPtr op_compiler_normal_ptr =
      make_shared<OpCompilerNormal>("normal", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_normal_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);

  OpCompilerPtr op_compiler_optune_ptr =
      make_shared<OpCompilerMstuneBeforeUbMatch>("optune", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_optune_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);

  OpCompilerPtr op_compiler_mstune_ptr = make_shared<OpCompilerOpTune>("mstune", AI_CORE_NAME, lx_fusion_optimizer_, nullptr);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_mstune_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_graph_normalization_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  Status status = fe_graph_optimizer_ptr->OptimizeAfterGraphNormalization(graph);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_graph_normalization_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  Status status = fe_graph_optimizer_->OptimizeAfterGraphNormalization(graph);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_blocked_some_process)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
  std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ =
  std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
      fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
  std::string allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ =
  std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
      ops_kernel_info_store_ptr_, fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);

  std::map<std::string, std::string> context_maps;
  std::string fusion_switch_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(fusion_switch_file_path).empty()) {
    fusion_switch_file_path = "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", fusion_switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);

  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (node->GetType() == CONV2D) {
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetOriginFormat());
      vector<int64_t> right_in_shape {1, 3, 32, 32};
      EXPECT_EQ(right_in_shape, op_desc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(op_desc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(op_desc->GetOutputDesc(0).GetOriginFormat()));
      vector<int64_t> right_out_shape {1, 1, 3, 32, 32};
      EXPECT_EQ(right_out_shape, op_desc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());
    }

    if (node->GetType() == "ReduceSum") {
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetOriginFormat());
      vector<int64_t> right_in_shape {1, 3, 32, 32};
      EXPECT_EQ(right_in_shape, op_desc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());

      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetOutputDesc(0).GetOriginFormat());
      vector<int64_t> right_out_shape {1, 3, 32, 32};
      EXPECT_EQ(right_out_shape, op_desc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_stage1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr mul_node = graph->AddNode(mul);

  std::string subgraph_name = "subgraph";
  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
  subgraph->SetParentNode(mul_node);
  subgraph->SetParentGraph(graph);
  graph->AddSubgraph(subgraph_name, subgraph);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp =
      std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>();
  op_attrs_maps_tmp->insert({"mul",{{"Mul", "Mul"}}});
  mul_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  Status ret = fe_graph_optimizer_->OptimizeAfterStage1(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dFixpipeGraph(graph);

  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(3, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);
  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(4, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op1)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dFixpipeGraph(graph);

  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(3, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);

  std::vector<ge::ComputeGraphPtr> sub_graphs = graph->GetAllSubgraphs();
  for (auto subgraph : sub_graphs) {
    for (auto node : subgraph->GetDirectNode()) {
      if (node->GetType() == "Conv2D") {
        auto tmpgraph = std::make_shared<ComputeGraph>("tmp_graph");
        node->GetOpDesc()->AddSubgraphName("tmp");
        ge::NodeUtils::SetSubgraph(*(node.get()), 0, tmpgraph);
      }
    }
  }
  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(4, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op2)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSwitchMergeFixpipeGraph(graph);

  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(3, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);
  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(6, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op4)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSwitchMergeFixpipeGraph2(graph);
  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(6, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);

  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(10, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op_sub_graph)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("node1", "PartitionCalled");
  ge::NodePtr node = graph->AddNode(opdesc);
  auto sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  CreateConv2dFixpipeGraph(sub_graph);
  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(node);
  graph->AddSubgraph(sub_graph->GetName(), sub_graph);

  sub_graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op_sub_graph2)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dFixpipeGraph(graph);

  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_subgraph_of_precompiledOp_case) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr transdata = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_4d_fp32(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp32(shape, FORMAT_NCHW, DT_FLOAT);

  data->AddOutputDesc(tenosr_desc_4d_fp32);
  transdata->AddInputDesc(tenosr_desc_4d_fp32);
  transdata->AddOutputDesc(tenosr_desc_5d_fp32);
  cast->AddInputDesc(tenosr_desc_5d_fp32);
  cast->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  Status ret = fe_graph_optimizer_->OptimizeSubgraphOfPrecompiledOp(*graph, GetOpKernelBinByKernelName);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_multi_stream_01)
{
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ReuseMemory)] = 0;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateCMOMultiStreamGraph(graph);
  ge::NodePtr a_node = graph->FindNode("A");
  ge::NodePtr b_node = graph->FindNode("B");
  ge::NodePtr h_node = graph->FindNode("H");
  vector<int32_t> data_visit_dist_vec = {2};
  auto input_desc = b_node->GetOpDesc()->MutableInputDesc(0);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info =
          {{"output0", {{h_node, MemType::OUTPUT_MEM, 0}}}};
  (void)ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  a_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->graph_optimizer_attr_.engineName = AI_CORE_NAME;
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>();
  Status status = fe_graph_optimizer_ptr->OptimizeStreamedWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = b_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = h_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 1);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_multi_stream_02)
{
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ReuseMemory)] = 0;
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 3;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateCMOMultiStreamGraph(graph);
  OpDescPtr opdesc_send = std::make_shared<OpDesc>("send1", "Send"); opdesc_send->SetStreamId(1);
  OpDescPtr opdesc_recv = std::make_shared<OpDesc>("recv1", "Recv"); opdesc_recv->SetStreamId(2);
  ge::NodePtr send = graph->AddNode(opdesc_send);
  ge::NodePtr recv = graph->AddNode(opdesc_recv);
  ge::AttrUtils::SetInt(opdesc_send, "event_id", 1);
  ge::AttrUtils::SetInt(opdesc_recv, "event_id", 1);
  ge::NodePtr a_node = graph->FindNode("A");
  ge::NodePtr b_node = graph->FindNode("B");
  ge::NodePtr d_node = graph->FindNode("D");
  ge::NodePtr e_node = graph->FindNode("E");
  ge::NodePtr h_node = graph->FindNode("H");
  GraphUtils::AddEdge(d_node->GetOutControlAnchor(), send->GetInControlAnchor());
  GraphUtils::AddEdge(recv->GetOutControlAnchor(), e_node->GetInControlAnchor());
  vector<int32_t> data_visit_dist_vec = {2};
  auto input_desc = b_node->GetOpDesc()->MutableInputDesc(0);
  (void)ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info =
          {{"output0", {{h_node, MemType::OUTPUT_MEM, 0}}}};
  a_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->graph_optimizer_attr_.engineName = AI_CORE_NAME;
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>();
  Status status = fe_graph_optimizer_ptr->OptimizeStreamedWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = b_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = h_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 1);
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 2;
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_graph_attr)
{
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  auto graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ =
          std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  Status status = fe_graph_optimizer_ptr->OptimizeGraphBeforeBuild(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  bool op_need_multi_task = false;
  (void)ge::AttrUtils::GetBool(graph, "_op_need_multi_task", op_need_multi_task);
  EXPECT_EQ(op_need_multi_task, true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_multi_stream_03)
{
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ReuseMemory)] = 0;
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 3;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateCMOMultiStreamGraph(graph);
  ge::NodePtr a_node = graph->FindNode("A");
  ge::NodePtr b_node = graph->FindNode("B");
  ge::NodePtr f_node = graph->FindNode("F");
  vector<int32_t> data_visit_dist_vec = {2};
  auto input_desc = b_node->GetOpDesc()->MutableInputDesc(0);
  (void)ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info =
          {{"output0", {{f_node, MemType::OUTPUT_MEM, 0}}}};
  a_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->graph_optimizer_attr_.engineName = AI_CORE_NAME;
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>();
  Status status = fe_graph_optimizer_ptr->OptimizeStreamedWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 2;
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphInit_pass)
{
  RegisterPassFunc(CreateFunc);
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
  std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ =
  std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fusion_rule_mgr_ptr_->init_flag_ = true;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ =
          std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
  std::string allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ =
  std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
      ops_kernel_info_store_ptr_, fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);

  std::map<std::string, std::string> context_maps;
  std::string fusion_switch_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(fusion_switch_file_path).empty()) {
    fusion_switch_file_path = "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", fusion_switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);

  fe_graph_optimizer_ptr->init_flag_ = true;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphInit(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphInit_fail_01)
{
    auto graph = std::make_shared<ComputeGraph>("test");
    CreateTwoOpDescGraph(graph);
    auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
    fe_graph_optimizer_ptr->init_flag_ = false;
    Status status = fe_graph_optimizer_ptr->OptimizeGraphInit(*(graph.get()));
    EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphInit_fail_02)
{
  RegisterPassFunc(CreateFunc);
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
  std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ =
  std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ =
          std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).lib_path_ =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
  std::string allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ =
  std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
      ops_kernel_info_store_ptr_, fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);

  std::map<std::string, std::string> context_maps;
  std::string fusion_switch_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(fusion_switch_file_path).empty()) {
    fusion_switch_file_path = "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", fusion_switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);

  fe_graph_optimizer_ptr->init_flag_ = true;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphInit(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, clear_same_memset)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
  OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
  ffts::ThreadSliceMapPtr slice_info_ptr1 = std::make_shared<ffts::ThreadSliceMap>();
  ffts::ThreadSliceMapPtr slice_info_ptr2 = std::make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr1->same_atomic_clean_nodes = {"cast1", "case2"};
  slice_info_ptr2->same_atomic_clean_nodes = {"cast1", "case2"};
  op_desc_cast1->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr1);
  op_desc_cast2->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr2);
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_OUTPUT_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_WORKSPACE_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_DTYPES, {0, 2});
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_INT64_VALUES, {1, 1});
  ge::AttrUtils::SetListFloat(op_desc_cast1, TBE_OP_ATOMIC_FLOAT_VALUES, {1.1, 2.2});

  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_OUTPUT_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_WORKSPACE_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_DTYPES, {0, 2});
  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_INT64_VALUES, {1, 1});
  ge::AttrUtils::SetListFloat(op_desc_cast2, TBE_OP_ATOMIC_FLOAT_VALUES, {1.1, 2.2});
  auto op_node_case1 = graph->AddNode(op_desc_cast1);
  auto op_node_case2 = graph->AddNode(op_desc_cast2);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->ClearSameMemSet(*graph);
  
  bool has_attr = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_OUTPUT_INDEX);
  EXPECT_FALSE(has_attr);
  bool has_attr_work = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_WORKSPACE_INDEX);
  EXPECT_FALSE(has_attr_work);
  bool has_attr_dtypes = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_DTYPES);
  EXPECT_FALSE(has_attr_dtypes);
  bool has_attr_int64 = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_INT64_VALUES);
  EXPECT_FALSE(has_attr_int64);
  bool has_attr_float = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_FLOAT_VALUES);
  EXPECT_FALSE(has_attr_float);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, CheckNeedSetSliceInfo)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = false;
  bool bres = fe_graph_optimizer_ptr->CheckNeedSetSliceInfo(*(graph.get()));
  EXPECT_EQ(bres, false);
  (void)ge::AttrUtils::GetBool(*(graph.get()), "need_set_slice_info", bres);
  EXPECT_EQ(bres, false);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, op_tiling) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", CONSTANT);
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", CONSTANT);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  OpDescPtr reduce_sum = std::make_shared<OpDesc>("sum", "ReduceSumD");
  const1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  const2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_DOUBLE));

  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(mul, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(mul, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");

  ge::NodePtr pld1 = graph->AddNode(const1);
  ge::NodePtr pld2 = graph->AddNode(const2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ge::NodePtr reduce_sum_node = graph->AddNode(reduce_sum);

  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), reduce_sum_node->GetInDataAnchor(0));

  FEOpsKernelInfoStorePtr ops_info_store = std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  Status ret = fe_graph_optimizer_ptr->OptimizeGraphForTiling(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);

  (void)ge::AttrUtils::SetBool(mul_node->GetOpDesc(), kDynamicTilingDependOp, true);
  graph->SetGraphUnknownFlag(true);
  ret = fe_graph_optimizer_ptr->OptimizeGraphForTiling(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, op_tiling_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", CONSTANT);
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", CONSTANT);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  OpDescPtr reduce_sum = std::make_shared<OpDesc>("sum", "ReduceSumD");
  const1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  const2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_DOUBLE));

  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(mul, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(mul, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  ge::AttrUtils::SetBool(mul, kAttrTileFwkOpStr, true);

  ge::NodePtr pld1 = graph->AddNode(const1);
  ge::NodePtr pld2 = graph->AddNode(const2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ge::NodePtr reduce_sum_node = graph->AddNode(reduce_sum);

  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), reduce_sum_node->GetInDataAnchor(0));

  FEOpsKernelInfoStorePtr ops_info_store = std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  Status ret = fe_graph_optimizer_ptr->OptimizeGraphForTiling(*graph);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, match_skp_case1) {
  PlatformUtils::Instance().short_soc_version_ = "Ascend035";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::SpecifiedMemBase)] = 1;
  Configuration::Instance(AI_CORE_NAME).content_map_["superkernel_plus.enable"] = "true";
  vector<int64_t> dims = {3, 4, 5, 6};
  ge::GeShape shape(dims);
  ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
  OpDescPtr softmax_op = std::make_shared<OpDesc>("softmax", "SoftmaxV2");
  OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
  OpDescPtr slice_op = std::make_shared<OpDesc>("strided_sliced", "StridedSliceD");

  data1_op->AddOutputDesc(tensor_desc);
  const_op->AddOutputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddOutputDesc(tensor_desc);
  relu_op->AddInputDesc(tensor_desc);
  relu_op->AddOutputDesc(tensor_desc);
  sigmoid_op->AddInputDesc(tensor_desc);
  sigmoid_op->AddOutputDesc(tensor_desc);
  softmax_op->AddInputDesc(tensor_desc);
  softmax_op->AddOutputDesc(tensor_desc);
  slice_op->AddInputDesc(tensor_desc);
  slice_op->AddOutputDesc(tensor_desc);

  AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(sigmoid_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(softmax_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(slice_op, "_fe_imply_type", 6);

  AttrUtils::SetInt(conv_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(relu_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(sigmoid_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(softmax_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(slice_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data1_node = graph->AddNode(data1_op);
  NodePtr const_node = graph->AddNode(const_op);
  NodePtr conv_node = graph->AddNode(conv_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
  NodePtr softmax_node = graph->AddNode(softmax_op);
  NodePtr slice_node = graph->AddNode(slice_op);
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0), slice_node->GetInDataAnchor(0));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  fe_graph_optimizer_ptr->MatchSuperkernelPlusNodes(*graph);

  set<int64_t> scope_set;
  for (auto &node : graph->GetDirectNode()) {
    int64_t scope_id = -1;
    if (ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), scope_id)) {
      scope_set.emplace(scope_id);
    }
  }
  EXPECT_EQ(scope_set.size(), 1);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, match_skp_case2) {
  PlatformUtils::Instance().short_soc_version_ = "Ascend035";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::SpecifiedMemBase)] = 1;
  Configuration::Instance(AI_CORE_NAME).content_map_["superkernel_plus.enable"] = "true";
  vector<int64_t> dims = {3, 4, 5, 6};
  ge::GeShape shape(dims);
  ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
  OpDescPtr softmax_op = std::make_shared<OpDesc>("softmax", "SoftmaxV2");
  OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
  OpDescPtr slice_op = std::make_shared<OpDesc>("strided_sliced", "StridedSliceD");

  data1_op->AddOutputDesc(tensor_desc);
  const_op->AddOutputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddOutputDesc(tensor_desc);
  relu_op->AddInputDesc(tensor_desc);
  relu_op->AddOutputDesc(tensor_desc);
  sigmoid_op->AddInputDesc(tensor_desc);
  sigmoid_op->AddOutputDesc(tensor_desc);
  softmax_op->AddInputDesc(tensor_desc);
  softmax_op->AddOutputDesc(tensor_desc);
  slice_op->AddInputDesc(tensor_desc);
  slice_op->AddOutputDesc(tensor_desc);

  AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(sigmoid_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(softmax_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(slice_op, "_fe_imply_type", 6);

  AttrUtils::SetInt(conv_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(relu_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(sigmoid_op, ge::TVM_ATTR_NAME_BLOCKDIM, 2);
  AttrUtils::SetInt(softmax_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  AttrUtils::SetInt(slice_op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data1_node = graph->AddNode(data1_op);
  NodePtr const_node = graph->AddNode(const_op);
  NodePtr conv_node = graph->AddNode(conv_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
  NodePtr softmax_node = graph->AddNode(softmax_op);
  NodePtr slice_node = graph->AddNode(slice_op);
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0), slice_node->GetInDataAnchor(0));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  fe_graph_optimizer_ptr->MatchSuperkernelPlusNodes(*graph);

  set<int64_t> scope_set;
  for (auto &node : graph->GetDirectNode()) {
    int64_t scope_id = -1;
    if (ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), scope_id)) {
      scope_set.emplace(scope_id);
    }
  }
  EXPECT_EQ(scope_set.size(), 2);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, StaticMultiKernelTest) {
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.deterministic"] = "true";
  option_tmp["ge.exec.allow_hf32"] = "true";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  auto graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
  // add descriptor
  vector<int64_t> dims = {1, 3, 32, 32};
  GeShape shape(dims);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc("x", in_desc2);

  vector<int64_t> dims1 = {1, 1, 3, 32, 32};
  GeShape shape1(dims1);
  GeTensorDesc out_desc1(shape1);
  out_desc1.SetFormat(FORMAT_NC1HWC0);
  out_desc1.SetOriginFormat(FORMAT_NC1HWC0);
  out_desc1.SetDataType(DT_FLOAT16);
  conv2d->AddOutputDesc("y", out_desc1);

  graph->AddNode(conv2d);
  graph->SetGraphUnknownFlag(true);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  auto ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetBool(conv2d, ATTR_NAME_FALLBACK_ACLNN, false);
  ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool aclnn_flag = false;
  (void)ge::AttrUtils::GetBool(conv2d, ATTR_NAME_FALLBACK_ACLNN, aclnn_flag);
  EXPECT_EQ(aclnn_flag, false);
  conv2d->DelAttr(ATTR_NAME_FALLBACK_ACLNN);
  ge::AttrUtils::SetBool(conv2d, "_unknown_shape", false);
  ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  (void)ge::AttrUtils::GetBool(conv2d, ATTR_NAME_FALLBACK_ACLNN, aclnn_flag);
  EXPECT_EQ(aclnn_flag, true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, StaticMultiKernelAutoFuseTest) {
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.deterministic"] = "true";
  option_tmp["ge.exec.allow_hf32"] = "true";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  auto graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "AscBackend");
  // add descriptor
  vector<int64_t> dims = {1, 3, 32, 32};
  GeShape shape(dims);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc("x", in_desc2);

  vector<int64_t> dims1 = {1, 1, 3, 32, 32};
  GeShape shape1(dims1);
  GeTensorDesc out_desc1(shape1);
  out_desc1.SetFormat(FORMAT_NC1HWC0);
  out_desc1.SetOriginFormat(FORMAT_NC1HWC0);
  out_desc1.SetDataType(DT_FLOAT16);
  conv2d->AddOutputDesc("y", out_desc1);

  graph->AddNode(conv2d);
  graph->SetGraphUnknownFlag(true);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  auto ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, compile_level_heavy_prop_test1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  auto reflection_builder_ptr = std::make_shared<ge::RefRelations>();
  HeavyFormatPropagationPtr heavy_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr);

  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_[fe::kComLevelO1Opt] = fe::kStrFalse;

  auto ret = fe_graph_optimizer_ptr->HeavyFormatPropagate(*graph, heavy_format_propagator);
  EXPECT_EQ(ret, fe::SUCCESS);


  mul->SetType(ASCEND_QUANT);
  ret = fe_graph_optimizer_ptr->HeavyFormatPropagate(*graph, heavy_format_propagator);
  EXPECT_EQ(ret, fe::SUCCESS);
  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_.clear();
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, test_lxfusion_recovery) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetStr(mul, "fusion_op_build_options", "1111111");
  ge::NodePtr mul_node = graph->AddNode(mul);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.emplace_back(mul_node);
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  std::vector<ge::NodePtr> buff_fus_to_del_nodes;
  lx_fusion_optimizer_->LxFusionRecovery(*(graph.get()), buff_fus_compile_failed_nodes, buff_fus_rollback_nodes, buff_fus_to_del_nodes);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_stream_graph_success)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->l2_optimize_ptr_ = std::make_shared<L2Optimizer>(AI_CORE_NAME);
  ge::RunContext context_;
  Status status = fe_graph_optimizer_ptr->OptimizeStreamGraph(*graph, context_);
  EXPECT_EQ(fe::SUCCESS, status);
}