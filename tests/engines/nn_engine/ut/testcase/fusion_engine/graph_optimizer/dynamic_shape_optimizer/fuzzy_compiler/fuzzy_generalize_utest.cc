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
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/configuration.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/op_info_util.h"
#include "all_ops_stub.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include <graph_optimizer/fe_graph_optimizer.h>
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_store/binary_kernel_info.h"
#include "ops_store/ops_kernel_utils.h"
#include <fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h>
#include "graph_optimizer/graph_fusion/graph_replace.h"
#include "./ge_context.h"
#include "./ge_local_context.h"
#include "ge/ge_api_types.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "common/lxfusion_json_util.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/graph/fe_graph_utils.h"
#include "common/unknown_shape_util.h"
#include "common/fe_inner_attr_define.h"
#include "common/platform_utils.h"
#include "common/config_parser/modify_mixlist_config_parser.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/node.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/fuzzy_generalize.h"
#include "graph_constructor.h"

#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace fe;


using FEGraphOptimizerPtr = std::shared_ptr<fe::FEGraphOptimizer>;
using OpStoreAdapterPtr = std::shared_ptr<fe::OpStoreAdapter>;

class OptimizeUtilityStub: public ge::OptimizeUtility {
 public:
  OptimizeUtilityStub() {}
  virtual ~OptimizeUtilityStub() override {}

  ge::Status InferShape(ComputeGraph &compute_graph) override{
    return ge::SUCCESS;
  }

  ge::Status InferShape(const ComputeGraphPtr &compute_graph) override {
    return ge::SUCCESS;
  }
};

bool CheckIsRegistered(const te::TbeOpInfo &op_info, bool &val) {
  val = true;
  return true;
}

bool CheckIsNotRegistered(const te::TbeOpInfo &op_info, bool &val) {
  val = false;
  return true;
}

bool CheckIsRegisteredException(const te::TbeOpInfo &op_info, bool &val) {
  val = false;
  return false;
}

std::string GetCurpath() {
    Dl_info dl_info;
    if (dladdr((void*) GetCurpath, &dl_info) == 0) {
        return "";
    } else {
        std::string so_path = dl_info.dli_fname;
        char resoved_path[4096] = {0x00};
        realpath(so_path.c_str(), resoved_path);
        so_path = resoved_path;
        std::string real_dir_file_path = so_path.substr(0, so_path.rfind('/') + 1);
        return real_dir_file_path;
    }
}

bool TeGeneralizeStub(const te::TbeOpInfo &op_info,
    const te::TE_GENERALIZE_TYPE &general_type, const ge::NodePtr &node) {
  std::vector<int64_t> shape_vec;
  auto op_desc = node->GetOpDesc();
  auto tensor_desc_x = op_desc->MutableInputDesc("x");
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
    for (auto &ele : shape_vec) {
      ele = -1;
    }
  }
  tensor_desc_x->SetShape(ge::GeShape(shape_vec));
  tensor_desc_x->SetOriginShape(ge::GeShape(shape_vec));
  return true;
}

bool TeGeneralizeStubDynamicRank(const te::TbeOpInfo &op_info,
    const te::TE_GENERALIZE_TYPE &general_type, const ge::NodePtr &node) {
  std::vector<int64_t> shape_vec;
  auto op_desc = node->GetOpDesc();
  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  shape_vec = tensor_desc_x->GetShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  if (general_type == te::REGISTER_FUNC) {
    for (auto &i : shape_vec) {
      shape_range.emplace_back(std::make_pair(1, -1));
      i = -1;
    }
  } else if (general_type == te::DEFAULT_TBE_OP_INFO) {
    for (int i = 0; i < shape_vec.size()-1; ++i) {
      shape_range.emplace_back(std::make_pair(1, -1));
      shape_vec[i] = -1;
    }
  } else {
    shape_vec[0] = -1;
  }
  tensor_desc_x->SetShape(ge::GeShape(shape_vec));
  tensor_desc_x->SetOriginShape(ge::GeShape(shape_vec));
  tensor_desc_x->SetShapeRange(shape_range);
  tensor_desc_x->SetOriginShapeRange(shape_range);
  return true;
}

bool TeGeneralizeTrue(const te::TbeOpInfo &op_info,
    const te::TE_GENERALIZE_TYPE &general_type, const ge::NodePtr &node) {
  return true;
}

bool TeGeneralizeException(const te::TbeOpInfo &op_info,
    const te::TE_GENERALIZE_TYPE &general_type, const ge::NodePtr &node) {
  return false;
}

bool GetOpSpecificInfoLimitStep(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  opSpecificInfo = "limited";
  return true;
}

bool GetOpSpecificInfoUnlimitStep(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  opSpecificInfo = "unlimited";
  return true;
}

bool GetOpSpecificInfoDynamicStep(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  opSpecificInfo = "dynamic";
  return true;
}

bool GetOpSpecificInfoFailStep(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  return false;
}

bool DynamicShapeRangeCheckNotSupportUpperStep(const te::TbeOpInfo &tbeOpInfo, bool &isSupported,
    std::vector<size_t> &upperLimitedInputIndexs,
    std::vector<size_t> &lowerLimitedInputIndexs) {
  upperLimitedInputIndexs.emplace_back(0);
  isSupported = false;
  return true;
}

bool DynamicShapeRangeCheckNotSupportLowerStep(const te::TbeOpInfo &tbeOpInfo, bool &isSupported,
    std::vector<size_t> &upperLimitedInputIndexs,
    std::vector<size_t> &lowerLimitedInputIndexs) {
  lowerLimitedInputIndexs.emplace_back(0);
  isSupported = false;
  return true;
}

bool DynamicShapeRangeCheckSupportStep(const te::TbeOpInfo &tbeOpInfo, bool &isSupported,
    std::vector<size_t> &upperLimitedInputIndexs,
    std::vector<size_t> &lowerLimitedInputIndexs) {
  isSupported = true;
  return true;
}

class TestGeneralizationPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern1 = new (std::nothrow) FusionPattern("TestGenPattern1");
    FE_CHECK(pattern1 == nullptr, FE_LOGE("New a pattern1 object failed."),  return patterns);
    pattern1->AddOpDesc("TestGen", {"TestGen"})
        .SetOutput("TestGen");
    patterns.push_back(pattern1);
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    return fe::SUCCESS;
  }
};

REG_PASS("TestGeneralizationPass", BUILT_IN_GRAPH_PASS,
         TestGeneralizationPass, SINGLE_SCENE_OPEN | FE_PASS | ALWAYS_GENERALIZE);

class UTEST_fusion_engine_fuzzy_generalize : public testing::Test {
public:
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  FuzzyGeneralizePtr fuzzy_ptr;
  RefRelationsPtr reflection_builder_ptr_;
  FEGraphOptimizerPtr fe_graph_optimizer_;
  TbeOpStoreAdapterPtr tbe_op_store_adapter_;
  shared_ptr<fe::SubOpInfoStore> sub_ops_kernel_ptr;
  FusionAttrManagerPtr fusion_attr_mgr_;
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;
  shared_ptr<fe::SubOpsStore> sub_ops_store_ptr;
  GraphFusionPtr graph_fusion_ptr_;
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
    std::string soc_version = "Ascend910B1";
    PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
    PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
    PlatformUtils::Instance().soc_version_ = soc_version;
  }
  void SetUp()
  {
    ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    ops_kernel_info_store_ptr_->tbe_info_assembler_ptr_ = std::make_shared<TbeInfoAssembler>();
    ops_kernel_info_store_ptr_->tbe_info_assembler_ptr_->Initialize();
    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
    FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    FusionPriorityMgrPtr fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
              fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
     graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
        ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
    graph_fusion_ptr_->SetEngineName(fe::AI_CORE_NAME);

    OptimizeUtilityStub *optimize_utility_stub = new OptimizeUtilityStub();
    fe_graph_optimizer_ = make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, fe::AI_CORE_NAME);
    fe_graph_optimizer_->optimize_utility_ = optimize_utility_stub;
    fe_graph_optimizer_->graph_fusion_ptr_ = graph_fusion_ptr_;
    fuzzy_ptr = std::make_shared<fe::FuzzyGeneralize>(nullptr, ops_kernel_info_store_ptr_, nullptr);
    fuzzy_ptr->optimize_utility_ = optimize_utility_stub;

    FEOpsStoreInfo TBE_OPINFO_STUB = {
            6,
            "tbe-builtin",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/heavy_opinfo",
            ""
    };
    sub_ops_store_ptr = make_shared<fe::SubOpsStore>(fe::AI_CORE_NAME);
    sub_ops_store_ptr->SetSubStoreInfo(TBE_OPINFO_STUB);
    sub_ops_store_ptr->InitializeSubStore();

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(TBE_OPINFO_STUB);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);

    sub_ops_kernel_ptr = std::make_shared<fe::SubOpInfoStore>(TBE_OPINFO_STUB);
    sub_ops_kernel_ptr->Initialize(fe::AI_CORE_NAME);
    OpsKernelManager::Instance(fe::AI_CORE_NAME).sub_ops_kernel_map_.emplace("tbe-builtin", sub_ops_kernel_ptr);

    std::map<std::string, std::string> options;
    options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", SHAPE_GENERALIZED));
    ge::GetThreadLocalContext().SetGlobalOption(options);

    std::map<std::string, std::string> options1;
    OpsKernelManager::Instance(fe::AI_CORE_NAME).Finalize();
    ops_kernel_info_store_ptr_->Initialize(options1);

    tbe_op_store_adapter_ = std::dynamic_pointer_cast<TbeOpStoreAdapter>(
        OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));

    FusionRuleManagerPtr fusion_rule_mgr_ptr = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
        fe::AI_CORE_NAME, fusion_rule_mgr_ptr);
    fusion_attr_mgr_ = std::make_shared<FusionAttrManager>(fusion_priority_mgr_ptr_);
    fe::FusionPassRegistry::PassDesc pass_desc = {SINGLE_SCENE_OPEN | FE_PASS | ALWAYS_GENERALIZE,
                                                  []() -> ::fe::GraphPass * { return new (std::nothrow) TestGeneralizationPass(); }};
    fusion_priority_mgr_ptr_->sorted_graph_fusion_single_scene_map_[FusionPriorityManager::GetCurrentHashedKey()].emplace_back(
        "TestGeneralizationPass", BUILT_IN_GRAPH_PASS, "", 1, pass_desc);
    fusion_attr_mgr_->Initialize();
  }

  void TearDown() {
    sub_ops_store_ptr->FinalizeSubStore();
    sub_ops_store_ptr.reset();
    sub_ops_kernel_ptr->Finalize();
    sub_ops_kernel_ptr.reset();
    ops_kernel_info_store_ptr_->Finalize();
    delete fe_graph_optimizer_->optimize_utility_;
  }

  static void CreateBatchNormGraphDynamicRank(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    vector<std::pair<int64_t, int64_t>> range = {std::make_pair(1, -1), std::make_pair(1, -1), std::make_pair(1, -1), std::make_pair(1, -1)};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetOriginShapeRange(range);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    out_desc2.SetShapeRange(range);
    out_desc2.SetOriginShapeRange(range);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
  }

  static void CreateBatchNormGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}, {1, 200}, {1, 200}});
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetOriginShapeRange(range);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    out_desc2.SetShapeRange(range);
    out_desc2.SetOriginShapeRange(range);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
  }

  static void CreateBatchNormGraph1(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {1, 2};
    vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}});
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetOriginFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetOriginShapeRange(range);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    out_desc2.SetShapeRange(range);
    out_desc2.SetOriginShapeRange(range);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
  }

  static void CreateBatchNormGeneralizedGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {-1, -1, -1, -1};
    vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}, {1, 200}, {1, 200}});
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetOriginShapeRange(range);
    AttrUtils::SetInt(in_desc2, ge::ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(ge::FORMAT_NC1HWC0));
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

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
  }

  static void CreateBatchNormGeneralizedGraph1(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {-1, -1};
    vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}});
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetOriginFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    in_desc2.SetOriginShape(shape);
    in_desc2.SetOriginShapeRange(range);
    AttrUtils::SetInt(in_desc2, ge::ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(ge::FORMAT_NC1HWC0));
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

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
  }

  static void CreateDynamicShapeGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {-1, 2, 3, 32};
    vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}, {1, 200}, {1, 200}});
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetShapeRange(range);
    in_desc2.SetOriginShapeRange(range);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    out_desc2.SetShapeRange(range);
    out_desc2.SetOriginShapeRange(range);
    bn_op->AddOutputDesc("y", out_desc2);
    std::vector<bool> is_in_const_vec = {false};
    bn_op->SetIsInputConst(is_in_const_vec);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE,
                          static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));

    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    GeTensorDesc in_desc1(shape);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc1);

    NodePtr relu_node = graph->AddNode(relu_op);
    ge::GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateDynamicShapeGraphWithoutRange(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

    // add descriptor
    vector<int64_t> dims = {-1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc2);
    data->AddOutputDesc("x", in_desc2);
    data->AddInputDesc(in_desc2);

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
  }

  static void CreateSingleNode(ComputeGraphPtr graph, const string &op_type) {
    OpDescPtr test_gen = std::make_shared<OpDesc>("test_gen", op_type);
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    GeTensor tensor(in_desc1);
    ge::AttrUtils::SetTensor(in_desc1, ATTR_NAME_VALUE, tensor);

    test_gen->AddInputDesc("x", in_desc1);
    data->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    test_gen->AddOutputDesc("y", out_desc1);

    ge::AttrUtils::SetInt(test_gen, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
    NodePtr relu_node = graph->AddNode(test_gen);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateSingleNodeGraph2(ComputeGraphPtr graph) {
    OpDescPtr cosh_op = std::make_shared<OpDesc>("cosh", "Cosh");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    cosh_op->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    cosh_op->AddOutputDesc("y", out_desc1);

    ge::AttrUtils::SetInt(cosh_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    NodePtr cosh_node = graph->AddNode(cosh_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cosh_node->GetInDataAnchor(0));
  }

  static void CreateComplexGraph(ComputeGraphPtr graph, string data_type) {
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "Convolution");

    vector<ge::GeTensorPtr> weights;
    vector<int64_t> dim_weight = {256, 256, 512};
    uint32_t data_size = 256 * 256 * 512;
    GeShape shape_weight(dim_weight);
    GeTensorDesc weight_desc(shape_weight);
    GeTensor tensor(weight_desc);
    if (data_type == "float") {
      weight_desc.SetDataType(DT_FLOAT);
      vector<float> weight(data_size, 0.01);
      vector<uint8_t> buffer(data_size * 4);
      memcpy(buffer.data(), weight.data(), data_size * 4);
      tensor.SetData(std::move(buffer));
    } else if (data_type == "int32") {
      weight_desc.SetDataType(DT_INT32);
      vector<int32_t> weight(data_size, 1);
      vector<uint8_t> buffer(data_size * 4);
      memcpy(buffer.data(), weight.data(), data_size * 4);
      tensor.SetData(std::move(buffer));
    }
    GeTensorPtr tensor_ptr = make_shared<GeTensor>(tensor);
    weights.emplace_back(tensor_ptr);

    OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", "Const");
    weight_op_desc1->AddInputDesc("x", weight_desc);
    weight_op_desc1->AddOutputDesc("z", weight_desc);
    std::vector<bool> w_input_const;
    w_input_const.emplace_back(false);
    weight_op_desc1->SetIsInputConst(w_input_const);

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

    AttrUtils::SetInt(weight_op_desc1, "transposX", int_value);
    AttrUtils::SetFloat(weight_op_desc1, "transposY", float_value);
    AttrUtils::SetBool(weight_op_desc1, "attrBool", bool_value);
    AttrUtils::SetStr(weight_op_desc1, "attrStr", str_value);
    AttrUtils::SetListInt(weight_op_desc1, "attrListInt", int_vec);
    AttrUtils::SetListFloat(weight_op_desc1, "attrListFloat", float_vec);
    AttrUtils::SetListBool(weight_op_desc1, "attrListBool", bool_vec);
    AttrUtils::SetListStr(weight_op_desc1, "attrListStr", str_vec);

    ge::DataType set_dtype = ge::DT_FLOAT;
    std::vector<int64_t> shape_vec{256, 256, 512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    if (data_type == "float") {
      input1_desc_ptr->SetDataType(DT_FLOAT);
    } else if (data_type == "int32") {
      input1_desc_ptr->SetDataType(DT_INT32);
    }
    input1_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

    std::vector<bool> is_input_const;
    is_input_const.emplace_back(false);
    is_input_const.emplace_back(true);
    op_desc_ptr->SetIsInputConst(is_input_const);

    shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
    output_desc_ptr->SetDataType(set_dtype);
    output_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddOutputDesc("z", output_desc_ptr->Clone());

    AttrUtils::SetInt(op_desc_ptr, "imply_type", EN_IMPL_HW_TBE);
    NodePtr weight_node1 = graph->AddNode(weight_op_desc1);
    NodePtr conv_node = graph->AddNode(op_desc_ptr);
    op_desc_ptr->SetName("conv2");
    NodePtr conv_next_node = graph->AddNode(op_desc_ptr);
    GraphUtils::AddEdge(weight_node1->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(weight_node1->GetOutDataAnchor(0), conv_next_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv_next_node->GetInDataAnchor(0));
    OpDescUtils::SetWeights(*conv_node, weights);
    OpDescUtils::SetWeights(*conv_next_node, weights);
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

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

    std::vector<bool> is_input_const;
    is_input_const.emplace_back(false);
    is_input_const.emplace_back(true);
    op_desc_ptr->SetIsInputConst(is_input_const);

    shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
    output_desc_ptr->SetDataType(set_dtype);
    output_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddOutputDesc("z", output_desc_ptr->Clone());

    AttrUtils::SetInt(op_desc_ptr, "imply_type", EN_IMPL_HW_TBE);
    NodePtr conv_node = graph->AddNode(op_desc_ptr);
    op_desc_ptr->SetName("conv2");
    NodePtr conv_next_node = graph->AddNode(op_desc_ptr);
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv_next_node->GetInDataAnchor(0));
  }

  static void CreateTwoOpDescGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 4};
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

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_FRACTAL_Z);
    in_desc4.SetDataType(DT_FLOAT16);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_NHWC);
    out_desc3.SetDataType(DT_FLOAT16);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_feedInputsRootSet) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  // CorrectCAxisByOriginalFormat
  EXPECT_EQ(fuzzy_ptr->CorrectCAxisByOriginalFormat(ge::FORMAT_FRACTAL_Z, bn_node, data_node, "input_name"), fe::FAILED);

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();

  NodeGeneralInfoPtr node_info = std::make_shared<NodeGeneralInfo>();
  NodeGeneralInfoPtr node_info_data = std::make_shared<NodeGeneralInfo>();
  node_info_data->disjoint_root_set.emplace(data_node);
  fuzzy_ptr->node_info_map_.insert(std::make_pair(data_node, node_info_data));
  fuzzy_ptr->FeedInputsRootSet(bn_node, node_info);
  EXPECT_EQ(node_info->disjoint_root_set, node_info_data->disjoint_root_set);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_check_is_external) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "batchnormal") {
      EXPECT_EQ(fuzzy_ptr->CheckIsExternalNode(node), false);
    } else {
      EXPECT_EQ(fuzzy_ptr->CheckIsExternalNode(node), true);
    }
  }
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_check_and_update_limited_nodes) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDynamicShapeGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.emplace("DATA0", data_node);

  bool flag = false;
  vector<int64_t> dims = {1, 2, 3, 32};
  vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}, {1, 200}, {1, 200}});
  const std::vector<NodePtr> nodes = {bn_node};
  NodeGeneralInfoPtr node_info_bn = std::make_shared<NodeGeneralInfo>();;
  node_info_bn->disjoint_root_set.emplace(data_node);
  node_info_bn->inputs_root_map.insert(std::make_pair(bn_node->GetOpDesc()->MutableInputDesc(0),
      node_info_bn->disjoint_root_set));
  fuzzy_ptr->node_info_map_.insert(std::make_pair(bn_node, node_info_bn));

  tbe_op_store_adapter_->DynamicShapeRangeCheck = DynamicShapeRangeCheckSupportStep;
  (void)fuzzy_ptr->CheckAndUpdateLimitedNodes(tbe_op_store_adapter_, nodes, flag);
  vector<pair<int64_t, int64_t>> range_temp;
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_temp);
  EXPECT_EQ(range_temp, range);

  fuzzy_ptr->node_info_map_.insert(std::make_pair(bn_node, node_info_bn));
  fuzzy_ptr->decent_times_count_.insert(make_pair(data_node->GetName(), 1));
  tbe_op_store_adapter_->DynamicShapeRangeCheck = DynamicShapeRangeCheckNotSupportUpperStep;
  (void)fuzzy_ptr->CheckAndUpdateLimitedNodes(tbe_op_store_adapter_, nodes, flag);
  range_temp.clear();
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_temp);
  EXPECT_NE(range_temp, range);

  tbe_op_store_adapter_->DynamicShapeRangeCheck = DynamicShapeRangeCheckNotSupportLowerStep;
  (void)fuzzy_ptr->CheckAndUpdateLimitedNodes(tbe_op_store_adapter_, nodes, flag);
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_temp);
  EXPECT_NE(range_temp, range);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_update_dynamic_shape_to_original) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDynamicShapeGraph(graph);
  ge::NodePtr data_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    }
  }

  ge::ComputeGraphPtr graph_1 = std::make_shared<ComputeGraph>("test1");
  CreateDynamicShapeGraph(graph_1);
  ge::NodePtr data1_node;
  for (auto &node : graph_1->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data1_node = node;
    }
  }

  vector<std::pair<int64_t, int64_t>> range1({{1, 100}, {1, 100}, {1, 100}, {1, 100}});
  data1_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShapeRange(range1);

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();

  fuzzy_ptr->external_input_nodes_.emplace(data1_node);
  fuzzy_ptr->UpdateDynamicShapeToNewBakGraph(*graph_1);
  std::vector<std::pair<int64_t, int64_t>> src_shape_range;
  data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(src_shape_range);
  EXPECT_NE(range1, src_shape_range);

  ge::ComputeGraphPtr graph_2 = std::make_shared<ComputeGraph>("test2");
  CreateDynamicShapeGraphWithoutRange(graph_2);
  ge::NodePtr data2_node;
  for (auto &node : graph_2->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data2_node = node;
    }
  }
  fuzzy_ptr->external_input_nodes_.emplace(data2_node);
  fuzzy_ptr->UpdateDynamicShapeToNewBakGraph(*graph_2);

  ge::ComputeGraphPtr graph_3 = std::make_shared<ComputeGraph>("test2");
  CreateBatchNormGraph(graph_3);
  ge::NodePtr data3_node;
  for (auto &node : graph_3->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data3_node = node;
    }
  }
  fuzzy_ptr->external_input_nodes_.emplace(data3_node);
  fuzzy_ptr->UpdateDynamicShapeToNewBakGraph(*graph_3);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_graph_preprocessing) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();

  ge::ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  CreateSingleNodeGraph2(graph1);
  fe::Status res = fuzzy_ptr->GraphPreprocessing(*graph1, tbe_op_store_adapter_);
  EXPECT_EQ(res, fe::SUCCESS);
//
//  tbe_op_store_adapter_->GetOpSpecificInfo = GetOpSpecificInfoLimitStep;
//  res = fuzzy_ptr->GraphPreprocessing(*graph, tbe_op_store_adapter_);
//  EXPECT_EQ(fuzzy_ptr->external_input_nodes_.empty(), false);
//  EXPECT_EQ(fuzzy_ptr->limited_range_nodes_.empty(), false);
//  EXPECT_EQ(res, fe::SUCCESS);
//
//  tbe_op_store_adapter_->GetOpSpecificInfo = GetOpSpecificInfoUnlimitStep;
//  res = fuzzy_ptr->GraphPreprocessing(*graph, tbe_op_store_adapter_);
//  EXPECT_EQ(res, fe::SUCCESS);
//
//  tbe_op_store_adapter_->GetOpSpecificInfo = GetOpSpecificInfoDynamicStep;
//  res = fuzzy_ptr->GraphPreprocessing(*graph, tbe_op_store_adapter_);
//  EXPECT_EQ(res, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_update_input_nodes) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDynamicShapeGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  vector<std::pair<int64_t, int64_t>> range({{1, 200}, {1, 200}, {1, 200}, {1, 200}});
  vector<std::pair<int64_t, int64_t>> range_last;
  vector<std::pair<int64_t, int64_t>> range_cur;
  vector<std::pair<int64_t, int64_t>> range_dst({{1,145}, {1,146}, {1,146}, {1,153}});
  fuzzy_ptr->original_input_nodes_.emplace("DATA0", data_node);
  vector<size_t> limited_input_indexs = {0};
  NodeGeneralInfoPtr node_info_bn = std::make_shared<NodeGeneralInfo>();;
  std::unordered_set<ge::NodePtr> root_set{data_node};
  node_info_bn->inputs_root_map.insert(std::make_pair(bn_node->GetOpDesc()->MutableInputDesc(0), root_set));
  fuzzy_ptr->node_info_map_.insert(std::make_pair(bn_node, node_info_bn));

  fuzzy_ptr->decent_times_count_.insert(std::make_pair("DATA0", 1));
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_cur);
  bool is_range_out = false;
  fe::Status res = fuzzy_ptr->Downgrades(bn_node, true, limited_input_indexs, is_range_out);
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_last);
  EXPECT_NE(range_last, range_cur);
  range_last.clear();
  range_cur.clear();

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();

  fuzzy_ptr->decent_times_count_["DATA0"] = 4;
  fuzzy_ptr->decent_steps_.insert(pair<string, vector<double>>("DATA0", {5.0, 5.0, 5.0, 5.0}));
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_cur);
  res = fuzzy_ptr->Downgrades(bn_node, true, limited_input_indexs, is_range_out);
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_last);

  EXPECT_NE(range_last, range_cur);
  EXPECT_EQ(range_last, range_dst);
  range_last.clear();
  range_cur.clear();
  fuzzy_ptr->decent_times_count_["DATA0"] = 5;
  res = fuzzy_ptr->Downgrades(bn_node, true, limited_input_indexs, is_range_out);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_TRUE(is_range_out);

  fuzzy_ptr->decent_times_count_["DATA0"] = 5;
  res = fuzzy_ptr->Downgrades(bn_node, true, limited_input_indexs, is_range_out);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_TRUE(is_range_out);

  fuzzy_ptr->decent_times_count_["DATA0"] = 1;
  res = fuzzy_ptr->Downgrades(bn_node, false, limited_input_indexs, is_range_out);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_TRUE(is_range_out);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_calculate_decent_steps) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDynamicShapeGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  vector<std::pair<int64_t, int64_t>> range_last;
  vector<std::pair<int64_t, int64_t>> range_cur;
  NodeGeneralInfoPtr node_info_bn = std::make_shared<NodeGeneralInfo>();;
  std::unordered_set<ge::NodePtr> root_set{data_node};
  node_info_bn->inputs_root_map.insert(std::make_pair(bn_node->GetOpDesc()->MutableInputDesc(0), root_set));

  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_cur);
  fuzzy_ptr->CalDecentSteps(data_node, data_node->GetOpDesc());
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(range_last);
  EXPECT_EQ(fuzzy_ptr->decent_steps_.empty(), false);

  vector<std::pair<int64_t, int64_t>> range({{1024, -1}, {1, 200}, {1, 200}, {1, 200}});
  (void)data_node->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(range);
  fuzzy_ptr->CalDecentSteps(data_node, data_node->GetOpDesc());
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_init_range_decentinfos) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDynamicShapeGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->InitOriginalGraphInfos(*graph);
  EXPECT_EQ(fuzzy_ptr->decent_steps_.empty(), false);
  EXPECT_EQ(fuzzy_ptr->decent_times_count_.empty(), false);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_generalize_graph) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.emplace("DATA0", data_node);

  tbe_op_store_adapter_->GetOpSpecificInfo = GetOpSpecificInfoDynamicStep;
  tbe_op_store_adapter_->DynamicShapeRangeCheck = DynamicShapeRangeCheckSupportStep;
  tbe_op_store_adapter_->CheckIsTbeGeneralizeFuncRegistered = CheckIsRegistered;
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeStub;
  Status res = fuzzy_ptr->GeneralizeGraph(*graph);
  EXPECT_EQ(res, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_C_axis_corrent_001) {
  ge::ComputeGraphPtr generalized_graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGeneralizedGraph(generalized_graph);
  ge::ComputeGraphPtr original_graph = std::make_shared<ComputeGraph>("test1");
  CreateBatchNormGraph(original_graph);
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.clear();
  ge::NodePtr node;
  for (auto &cur_node : generalized_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->external_input_nodes_.emplace(cur_node);
      node = cur_node;
    }
    if (cur_node->GetType() == "BatchNorm") {
        NodeGeneralInfoPtr node_info_ptr = std::make_shared<NodeGeneralInfo>();
        node_info_ptr->op_kernel = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", "BatchNorm");
        fuzzy_ptr->node_info_map_.insert(std::make_pair(cur_node, node_info_ptr));
    }
  }
  for (auto &cur_node : original_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->original_input_nodes_.emplace(std::make_pair(cur_node->GetName(), cur_node));
    }
  }

  Status res = fuzzy_ptr->UpdateDynamicShapeToOriginalGraph(*generalized_graph);
  EXPECT_EQ(res, fe::SUCCESS);

  std::vector<int64_t> shape = node->GetOpDesc()->MutableInputDesc(0)->GetOriginShape().GetDims();
  EXPECT_EQ(shape, (std::vector<int64_t>{-1,2,-1,-1}));
}


TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_C_axis_corrent_002) {
  ge::ComputeGraphPtr generalized_graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGeneralizedGraph1(generalized_graph);
  ge::ComputeGraphPtr original_graph = std::make_shared<ComputeGraph>("test1");
  CreateBatchNormGraph1(original_graph);
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.clear();
  ge::NodePtr node;
  for (auto &cur_node : generalized_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->external_input_nodes_.emplace(cur_node);
      node = cur_node;
    }
    if (cur_node->GetType() == "BatchNorm") {
      NodeGeneralInfoPtr node_info_ptr = std::make_shared<NodeGeneralInfo>();
      node_info_ptr->op_kernel = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", "BatchNorm");
      fuzzy_ptr->node_info_map_.insert(std::make_pair(cur_node, node_info_ptr));
    }
  }
  for (auto &cur_node : original_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->original_input_nodes_.emplace(std::make_pair(cur_node->GetName(), cur_node));
    }
  }

  Status res = fuzzy_ptr->UpdateDynamicShapeToOriginalGraph(*generalized_graph);
  EXPECT_EQ(res, fe::SUCCESS);

  std::vector<int64_t> shape = node->GetOpDesc()->MutableInputDesc(0)->GetOriginShape().GetDims();
  EXPECT_EQ(shape, (std::vector<int64_t>{ -1, 2}));
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_C_axis_corrent_003) {
  ge::ComputeGraphPtr generalized_graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGeneralizedGraph1(generalized_graph);
  ge::ComputeGraphPtr original_graph = std::make_shared<ComputeGraph>("test1");
  CreateBatchNormGraph1(original_graph);
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.clear();
  ge::NodePtr node;
  for (auto &cur_node : generalized_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->external_input_nodes_.emplace(cur_node);
      node = cur_node;
    }
    if (cur_node->GetType() == "BatchNorm") {
      auto op_desc = cur_node->GetOpDesc();
      ge::OpDescUtilsEx::SetType(op_desc, "Cosh");
      NodeGeneralInfoPtr node_info_ptr = std::make_shared<NodeGeneralInfo>();
      node_info_ptr->op_kernel = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", "BatchNorm");
      fuzzy_ptr->node_info_map_.insert(std::make_pair(cur_node, node_info_ptr));
    }
  }
  for (auto &cur_node : original_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->original_input_nodes_.emplace(std::make_pair(cur_node->GetName(), cur_node));
    }
    if (cur_node->GetType() == "BatchNorm") {
      auto op_desc = cur_node->GetOpDesc();
      ge::OpDescUtilsEx::SetType(op_desc, "Cosh");
    }
  }

  Status res = fuzzy_ptr->UpdateDynamicShapeToOriginalGraph(*generalized_graph);
  EXPECT_EQ(res, fe::SUCCESS);

  std::vector<int64_t> shape = node->GetOpDesc()->MutableInputDesc(0)->GetOriginShape().GetDims();
  EXPECT_EQ(shape, (std::vector<int64_t>{-1, 2}));
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, fuzzy_generalize_C_axis_corrent_004) {
  ge::ComputeGraphPtr generalized_graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGeneralizedGraph(generalized_graph);
  ge::ComputeGraphPtr original_graph = std::make_shared<ComputeGraph>("test1");
  CreateBatchNormGraph(original_graph);
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.clear();
  ge::NodePtr node;
  for (auto &cur_node : generalized_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->external_input_nodes_.emplace(cur_node);
      node = cur_node;
    }
  }
  for (auto &cur_node : original_graph->GetDirectNode()) {
    if (cur_node->GetType() == fe::DATA) {
      fuzzy_ptr->original_input_nodes_.emplace(std::make_pair(cur_node->GetName(), cur_node));
    }
  }
  Status res = fuzzy_ptr->UpdateDynamicShapeToOriginalGraph(*generalized_graph);
  EXPECT_EQ(res, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, get_subgraphs_by_curnode_001) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = true;
  fuzzy_ptr->node_info_map_.clear();
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
      fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  std::vector<ge::ComputeGraphPtr> cur_node_subgraph = input_node_generalize.GetSubgraphsByCurNode(data_node);
  EXPECT_EQ(cur_node_subgraph.empty(), true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, get_subgraphs_by_curnode_002) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = true;
  fuzzy_ptr->node_info_map_.clear();
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  std::vector<ge::ComputeGraphPtr> cur_node_subgraph = input_node_generalize.GetSubgraphsByCurNode(data_node);
  EXPECT_EQ(cur_node_subgraph.empty(), true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, unlimited_node_generalize) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = true;
  fuzzy_ptr->node_info_map_.clear();
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  NodeGeneralInfoPtr node_info_bn = std::make_shared<NodeGeneralInfo>();;
  std::unordered_set<ge::NodePtr> root_set{data_node};
  node_info_bn->inputs_root_map.insert(std::make_pair(bn_node->GetOpDesc()->MutableInputDesc(0), root_set));
  node_info_bn->is_found_in_opstore = true;
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeException;
  Status res = input_node_generalize.UnlimitedNodeGeneralize(bn_node, node_info_bn);
  EXPECT_EQ(res, fe::FAILED);

  node_info_bn->is_found_in_opstore = false;
  res = input_node_generalize.UnlimitedNodeGeneralize(bn_node, node_info_bn);
  EXPECT_EQ(res, fe::FAILED);

  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeTrue;
  res = input_node_generalize.UnlimitedNodeGeneralize(bn_node, node_info_bn);
  EXPECT_EQ(res, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, GeneralizeFirstNodeOfGraph) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = false;
  fuzzy_ptr->node_info_map_.clear();

  NodeGeneralInfoPtr node_info_bn = std::make_shared<NodeGeneralInfo>();;
  std::unordered_set<ge::NodePtr> root_set{data_node};
  node_info_bn->inputs_root_map.insert(std::make_pair(bn_node->GetOpDesc()->MutableInputDesc(0), root_set));
  node_info_bn->is_found_in_opstore = false;
  fuzzy_ptr->node_info_map_.insert(std::make_pair(bn_node, node_info_bn));
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeException;
  Status res = input_node_generalize.GeneralizeFirstNodeOfGraph(bn_node);
  EXPECT_EQ(res, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, downgrades_test) {
  bool is_upper_limited;
  bool is_range_out = false;
  std::vector<size_t> limited_input_indexs;
  limited_input_indexs.emplace_back(0);
  limited_input_indexs.emplace_back(1);
  limited_input_indexs.emplace_back(2);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op1 = std::make_shared<OpDesc>("test", "test");
  vector<int64_t> dim_input({4, 33, 12, 16, 64});
  GeShape shape(dim_input);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_ND);
  tensor_desc.SetDataType(DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op1->AddOutputDesc("y", tensor_desc);
  ge::NodePtr cur_node = graph->AddNode(op1);
  std::map<ge::NodePtr, NodeGeneralInfoPtr> node_info_map_;
  NodeGeneralInfoPtr node_info = std::make_shared<NodeGeneralInfo>();
  fuzzy_ptr->node_info_map_.insert(std::make_pair(cur_node, node_info));
  Status ret = fuzzy_ptr->Downgrades(cur_node, is_upper_limited, limited_input_indexs, is_range_out);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, merge_range_with_upper_limit_max_test) {
  std::pair<int64_t, int64_t> upper_limit_max_range{3, 4};
  std::pair<int64_t, int64_t> range{1, 2};
  size_t dim_index;
  std::vector<std::pair<int64_t, int64_t>> dst_shape_range;

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = true;
  fuzzy_ptr->node_info_map_.clear();
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);

  Status ret = input_node_generalize.MergeRangeWithUpperLimitMax(upper_limit_max_range, range, dim_index, dst_shape_range);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, set_value_depend_flag_to_input_nodes) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  NodeGeneralInfoPtr node_info = std::make_shared<NodeGeneralInfo>();
  Status ret = input_node_generalize.SetValueDependFlagToInputNodes(bn_node, node_info);
  EXPECT_EQ(ret, fe::SUCCESS);
  OpKernelInfoPtr info_ptr_act = std::make_shared<OpKernelInfo>("Activation");
  node_info->op_kernel = info_ptr_act;
  ret = input_node_generalize.SetValueDependFlagToInputNodes(bn_node, node_info);
  EXPECT_EQ(ret, fe::SUCCESS);

  InputOrOutputInfoPtr in_desc_ptr = std::make_shared<fe::InputOrOutputInfo>("x");
  in_desc_ptr->op_const_value_depend_ = CONST_OPTIONAL;
  info_ptr_act->input_infos_.emplace_back(in_desc_ptr);
  node_info->op_kernel = info_ptr_act;
  ret = input_node_generalize.SetValueDependFlagToInputNodes(bn_node, node_info);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool is_value_depend = false;
  (void)ge::AttrUtils::GetBool(data_node->GetOpDesc(), ge::ATTR_NAME_VALUE_DEPEND, is_value_depend);
  EXPECT_EQ(is_value_depend, true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, update_subGraph_input_to_rootGraph) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }
  (void)ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);

  ge::ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  CreateBatchNormGraph(graph1);
  ge::NodePtr data_node1;
  ge::NodePtr bn_node1;
  for (auto &node : graph1->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node1 = node;
    } else {
      bn_node1 = node;
    }
  }

  graph->SetParentNode(bn_node1);
  Status ret = input_node_generalize.UpdateSubGraphInputToRootGraph(fuzzy_ptr->external_input_nodes_, graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  (void)ge::AttrUtils::SetBool(data_node->GetOpDesc(), ge::ATTR_NAME_VALUE_DEPEND, true);
  ret = input_node_generalize.UpdateSubGraphInputToRootGraph(fuzzy_ptr->external_input_nodes_, graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool is_value_depend = false;
  (void)ge::AttrUtils::GetBool(data_node1->GetOpDesc(), ge::ATTR_NAME_VALUE_DEPEND, is_value_depend);
  EXPECT_EQ(is_value_depend, true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, update_dynamic_shape_to_newInputNode) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  std::map<std::string, ge::NodePtr> new_input_nodes;
  new_input_nodes.insert(std::pair<std::string, ge::NodePtr>("DATA0", data_node));
  Status ret = fuzzy_ptr->UpdateDynamicShapeToNewInputNode(fuzzy_ptr->external_input_nodes_, new_input_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
  (void)ge::AttrUtils::SetBool(data_node->GetOpDesc(), ge::ATTR_NAME_VALUE_DEPEND, true);
  ret = fuzzy_ptr->UpdateDynamicShapeToNewInputNode(fuzzy_ptr->external_input_nodes_, new_input_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, merge_range) {
  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  std::vector<std::pair<int64_t, int64_t>> src_range;
  std::vector<std::pair<int64_t, int64_t>> dst_range;
  std::vector<std::pair<int64_t, int64_t>> res_range;
  dst_range.push_back(std::pair<int64_t, int64_t>(1, 110));
  Status ret = input_node_generalize.MergeRange(src_range, dst_range);
  EXPECT_EQ(ret, fe::FAILED);

  src_range.push_back(std::pair<int64_t, int64_t>(10, 100));
  res_range.push_back(std::pair<int64_t, int64_t>(10, 100));
  ret = input_node_generalize.MergeRange(src_range, dst_range);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(dst_range == res_range, true);

  dst_range[0].second = 90;
  res_range[0].second = 90;
  ret = input_node_generalize.MergeRange(src_range, dst_range);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(dst_range == res_range, true);

  dst_range[0].first = 20;
  res_range[0].first = 20;
  ret = input_node_generalize.MergeRange(src_range, dst_range);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(dst_range == res_range, true);

  src_range[0].second = -1;
  res_range[0].second = 90;
  ret = input_node_generalize.MergeRange(src_range, dst_range);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(dst_range == res_range, true);

  dst_range[0].second = -1;
  res_range[0].second = -1;
  ret = input_node_generalize.MergeRange(src_range, dst_range);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(dst_range == res_range, true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, update_op_attrs) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim0 = {-1, 1, 1, 1};
  vector<int64_t> dim1 = {1, 1, 1, 1};
  GeShape shape0(dim0);
  GeTensorDesc in_desc0(shape0);
  GeShape shape1(dim1);
  GeTensorDesc in_desc1(shape1);
  in_desc0.SetDataType(DT_FLOAT16);
  in_desc1.SetDataType(DT_FLOAT16);
  op_desc0->AddInputDesc(in_desc0);
  op_desc0->AddOutputDesc(in_desc0);
  op_desc1->AddInputDesc(in_desc1);
  op_desc1->AddOutputDesc(in_desc1);
  NodePtr node_0 = graph->AddNode(op_desc0);
  NodePtr node_1 = graph->AddNode(op_desc1);
  ge::GeTensorDescPtr cur_input_tensor_desc = op_desc0->MutableInputDesc(0);
  cur_input_tensor_desc->SetOriginShape(GeShape(dim0));
  ge::GeTensorDescPtr cur_output_tensor_desc = op_desc0->MutableOutputDesc(0);
  cur_output_tensor_desc->SetOriginShape(GeShape(dim0));
  ge::GeTensorDescPtr ori_input_tensor_desc = op_desc1->MutableInputDesc(0);
  (void)ge::AttrUtils::SetBool(ori_input_tensor_desc, "_value", true);
  ori_input_tensor_desc->SetOriginShape(GeShape(dim1));
  ge::GeTensorDescPtr ori_output_tensor_desc = op_desc1->MutableOutputDesc(0);
  (void)ge::AttrUtils::SetBool(ori_output_tensor_desc, "_value", true);
  ori_output_tensor_desc->SetOriginShape(GeShape(dim1));

  fuzzy_ptr->UpdateOpAttrs(cur_input_tensor_desc, ori_input_tensor_desc, op_desc1, true);
  fuzzy_ptr->UpdateOpAttrs(cur_output_tensor_desc, ori_output_tensor_desc, op_desc1, false);
  bool res = (!ge::AttrUtils::HasAttr(ori_input_tensor_desc, "_value") &&
              !ge::AttrUtils::HasAttr(ori_output_tensor_desc, "_value") &&
              ge::AttrUtils::HasAttr(op_desc1, "_is_op_generalized"));
  EXPECT_EQ(res, true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, is_shape_generalized) {
  std::map<std::string, std::string> options_map = ge::GetThreadLocalContext().GetAllOptions();
  options_map.insert(std::pair<std::string, std::string>("ge.shape_generalized", "1"));
  ge::GetThreadLocalContext().SetGlobalOption(options_map);
  bool bres = IsShapeGeneralizedMode();
  EXPECT_EQ(bres, true);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, coverage_increase) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = true;
  fuzzy_ptr->node_info_map_.clear();
  NodeGeneralInfoPtr node_info = std::make_shared<NodeGeneralInfo>();
  fuzzy_ptr->node_info_map_.insert(std::make_pair(bn_node, node_info));

  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};
  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, nullptr);
  std::unordered_set<ge::NodePtr> sub_graph_input_nodes;
  sub_graph_input_nodes.emplace(data_node);
  input_node_generalize.UpdateSubGraphInputToRootGraph(sub_graph_input_nodes, graph);

  OpDescPtr tmp_op = std::make_shared<OpDesc>("Test_tmp", "BatchNorm");
  OpDescPtr data_op = std::make_shared<OpDesc>("DATA_tmp", fe::DATA);
  NodePtr tmp_node = graph->AddNode(tmp_op);
  NodePtr data_tmp_node = graph->AddNode(data_op);
  input_node_generalize.GeneralizeFirstNodeOfGraph(tmp_node);
  fuzzy_ptr->node_info_map_.clear();
  fuzzy_ptr->node_info_map_.insert(std::make_pair(tmp_node, node_info));

  std::vector<ge::NodePtr> limited_nodes;
  limited_nodes.emplace_back(bn_node);
  bool btmp = false;
  fuzzy_ptr->CheckAndUpdateLimitedNodes(tbe_op_store_adapter_, limited_nodes, btmp);
  std::vector<size_t> limited_input_indexs;
  fuzzy_ptr->Downgrades(bn_node, false, limited_input_indexs, btmp);
  EXPECT_EQ(limited_input_indexs.size(), 0);
  std::map<std::string, ge::NodePtr> new_input_nodes;
  new_input_nodes.insert(std::make_pair("nnnnnnnn", bn_node));
  fuzzy_ptr->UpdateDynamicShapeToNewInputNode(sub_graph_input_nodes, new_input_nodes);
  fuzzy_ptr->UpdateDynamicShapeToNewBakGraph(*graph);
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->InitOriginalGraphInfos(*graph);

  fuzzy_ptr->original_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.insert(std::make_pair("nnnnnnnn", bn_node));
  uint32_t tmp_int;
  fuzzy_ptr->RangeDecent(tmp_node, tmp_int);
  fuzzy_ptr->original_input_nodes_.clear();
  fuzzy_ptr->node_info_map_.clear();
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, further_generalize) {
  std::string path = GetCurpath() + "../../../../../tests/engines/nn_engine/config/op_impl/built-in/ai_core/tbe/kernel/config/Ascend910B/op_info_config.json";
  char resoved_path[500] = {0x00};
  realpath(path.c_str(), resoved_path);
  path = resoved_path;
  Configuration::Instance(fe::AI_CORE_NAME).bin_cfg_file_ = path;
  BinaryKernelInfo &bin_kernel_info = BinaryKernelInfo::Instance();
  Status init_flag = bin_kernel_info.Initialize(path);
  EXPECT_EQ(init_flag, fe::SUCCESS);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  CreateBatchNormGraphDynamicRank(graph);
  ge::NodePtr data_node;
  ge::NodePtr bn_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      bn_node = node;
    }
  }

  fuzzy_ptr->decent_steps_.clear();
  fuzzy_ptr->decent_times_count_.clear();
  fuzzy_ptr->external_input_nodes_.clear();
  fuzzy_ptr->original_input_nodes_.emplace("DATA0", data_node);

  tbe_op_store_adapter_->GetOpSpecificInfo = GetOpSpecificInfoDynamicStep;
  tbe_op_store_adapter_->DynamicShapeRangeCheck = DynamicShapeRangeCheckSupportStep;
  tbe_op_store_adapter_->CheckIsTbeGeneralizeFuncRegistered = CheckIsRegistered;
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeStubDynamicRank;
  Status res = fuzzy_ptr->GeneralizeGraph(*graph);
  bool bres = bin_kernel_info.IsBinSupportDynamicRank("Add");
  EXPECT_EQ(bres, true);
  bres = bin_kernel_info.IsBinSupportDynamicRank("Sqrt");
  EXPECT_EQ(bres, false);

  auto input_desc = bn_node->GetOpDesc()->MutableInputDesc("x");
  auto shape_dims = input_desc->GetShape().GetDims();
  std::vector<int64_t> shape_dims_std = {1, 2, 3, 32};
  EXPECT_EQ(shape_dims, shape_dims_std);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, test_generalization_when_pass_registered_as_always_generalization) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateSingleNode(graph, "TestGen");
  ge::NodePtr data_node;
  ge::NodePtr gen_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "DATA0") {
      data_node = node;
    } else {
      gen_node = node;
    }
  }
  auto data_input_desc = data_node->GetOpDesc()->MutableInputDesc(0);
  auto data_output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  auto gen_node_input_desc = gen_node->GetOpDesc()->MutableInputDesc(0);

  fuzzy_ptr->external_input_nodes_.emplace(data_node);
  fuzzy_ptr->is_range_limited_graph_ = false;
  fuzzy_ptr->node_info_map_.clear();
  NodeGeneralInfoPtr node_info_gen = std::make_shared<NodeGeneralInfo>();;
  node_info_gen->is_found_in_opstore = true;
  fuzzy_ptr->node_info_map_.insert(std::make_pair(gen_node, node_info_gen));

  GraphType graph_type{fuzzy_ptr->is_range_limited_graph_, fuzzy_ptr->is_single_op_graph_};

  InputNodeGeneralize input_node_generalize(fuzzy_ptr->external_input_nodes_, graph_type,
                                            fuzzy_ptr->node_info_map_, tbe_op_store_adapter_, fusion_attr_mgr_);

  /* 1. Test when generalize function does not work. */
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeException;
  Status res = input_node_generalize.GeneralizeFirstNodeOfGraph(gen_node);
  EXPECT_EQ(res, fe::FAILED);
  ge::GeShape original_shape({1,2,3,4});
  EXPECT_EQ(data_output_desc->GetShape(), original_shape);
  EXPECT_EQ(gen_node_input_desc->GetShape(), original_shape);
  EXPECT_EQ(data_input_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(data_output_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(gen_node_input_desc->HasAttr(ge::ATTR_NAME_VALUE), true);

  /* 2. Test when the op is not found in ops kernel store and generalization function
   * works in single op scene but InputNodeGeneralize' single op scene is set as false. */
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeStub;
  node_info_gen->is_found_in_opstore = false;
  input_node_generalize.graph_type_.is_single_op_graph = false;
  res = input_node_generalize.GeneralizeFirstNodeOfGraph(gen_node);
  EXPECT_EQ(res, fe::SUCCESS);
  ge::GeShape generalized_shape({-1, -1, -1, -1});
  EXPECT_EQ(data_output_desc->GetShape().GetDims(), original_shape.GetDims());
  EXPECT_EQ(gen_node_input_desc->GetShape().GetDims(), generalized_shape.GetDims());
  EXPECT_EQ(data_output_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(data_input_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(gen_node_input_desc->HasAttr(ge::ATTR_NAME_VALUE), true);

  /* 3. Test when the op is not found in ops kernel store and generalization function
   * works in single op scene and InputNodeGeneralize' single op scene is set as true.
   * But Op type is not matched. */
  gen_node->GetOpDesc()->SetType("TestGenFailed");
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeStub;
  node_info_gen->is_found_in_opstore = false;
  input_node_generalize.graph_type_.is_single_op_graph = true;
  res = input_node_generalize.GeneralizeFirstNodeOfGraph(gen_node);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_EQ(data_output_desc->GetShape().GetDims(), original_shape.GetDims());
  EXPECT_EQ(gen_node_input_desc->GetShape().GetDims(), generalized_shape.GetDims());
  EXPECT_EQ(data_output_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(gen_node_input_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(data_input_desc->HasAttr(ge::ATTR_NAME_VALUE), true);

  /* 4. Test when the op is not found in ops kernel store and generalization function
   * works in single op scene and InputNodeGeneralize' single op scene is set as true. */
  gen_node->GetOpDesc()->SetType("TestGen");
  tbe_op_store_adapter_->TeGeneralize = TeGeneralizeStub;
  node_info_gen->is_found_in_opstore = false;
  input_node_generalize.graph_type_.is_single_op_graph = true;
  res = input_node_generalize.GeneralizeFirstNodeOfGraph(gen_node);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_EQ(data_output_desc->GetShape().GetDims(), generalized_shape.GetDims());
  EXPECT_EQ(gen_node_input_desc->GetShape().GetDims(), generalized_shape.GetDims());
  EXPECT_EQ(data_output_desc->HasAttr(ge::ATTR_NAME_VALUE), true);
  EXPECT_EQ(data_input_desc->HasAttr(ge::ATTR_NAME_VALUE), false);
  EXPECT_EQ(gen_node_input_desc->HasAttr(ge::ATTR_NAME_VALUE), false);
}

TEST_F(UTEST_fusion_engine_fuzzy_generalize, test_read_json_object) {
  nlohmann::json json_obj;
  std::string file = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/invalid.json";
  Status status = ReadJsonObject(file, json_obj);
  EXPECT_EQ(status, fe::FAILED);
  ModifyMixlistConfigParser parser;
  status = parser.ReadMixlistJson(file, json_obj);
  EXPECT_EQ(status, fe::FAILED);
}