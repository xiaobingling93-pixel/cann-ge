/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <string>
using StringVector = std::vector<std::string>;
#include "graph_optimizer/fe_graph_optimizer.h"

#include <cfloat>
#include <memory>
#include <mutex>
#include <iostream>
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/platform_utils.h"
#include "common/math/math_util.h"
#include "common/fe_context_utils.h"
#include "common/fe_type_utils.h"
#include "common/fe_report_error.h"
#include "common/scope_allocator.h"
#include "common/fe_graph_common.h"
#include "common/util/trace_manager/trace_manager.h"
#include "framework/common/ge_types.h"
#include "ops_store/op_kernel_info.h"
#include "ops_store/ops_kernel_manager.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/tuning_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/ir_definitions_recover.h"
#include "common/weight_compress_utils.h"
#include "graph_optimizer/op_compiler/op_format_tune.h"
#include "graph_optimizer/weight_prefetch/weight_prefetch_utils.h"
#include "trace_handle_manager/trace_handle_manager.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "register/optimization_option_registry.h"

namespace fe {
namespace {
constexpr int64_t kInvalidStreamId = -1;
thread_local static OpFormatDtypeJudgePtr op_format_dtype_judge_ptr;
thread_local static ReflectionBuilderPtr reflection_builder_ptr;

bool NeedDisableVector(const ge::NodePtr node_ptr) {
  if (Configuration::Instance(AI_CORE_NAME).IsEnableVirtualType()) {
    return true;
  }
  if (ge::AttrUtils::HasAttr(node_ptr->GetOpDesc(), ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE)) {
    FE_LOGD("Node [%s] has been set to disabled.", node_ptr->GetNamePtr());
    return true;
  }
  return false;
}
}  // namespace

const string kStageInit = "[GraphOpt][Init]";
const string kStagePrepare = "[GraphOpt][Prepare]";
const string kStageBeforeQuant = "[GraphOpt][BeforeQuant]";
const string kStageOrigin = "[GraphOpt][Origin]";
const string kStageAftFmtSlct = "[GraphOpt][AftFmtSlct]";
const string kStageJudgeInsert = "[GraphOpt][JdgInst]";
const string kStageSetOpSlc = "[SubGraphOpt][SetOpSlc]";
const string kStagePreCompile = "[SubGraphOpt][PreComp]";
const string kStageParseCompRst = "[SubGraphOpt][ParseCompRst]";
const string kStageLx = "[SubGraphOpt][Lx]";
const string kStageCompile = "[SubGraphOpt][Compile]";
const string kStageAfterMultiDims = "[GraphOpt][AfterMultiDims]";
const string kStageAfterOptimizeStage1 = "[GraphOpt][AfterOptimizeStage1]";
const string kStageFused = "[SubGraphOpt]";
const char* kGroupPolicy = "group";
const char* kVectorGroup = "vector";
const char* kAttachNotifyNumKey = "_attached_notify_num";
const char* kFusionVirtualOpSetSwitch = "FusionVirtualOpSetSwitch";
FEGraphOptimizer::FEGraphOptimizer(FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr, std::string engine_name)
    : ops_kernel_info_store_ptr_(fe_ops_kernel_info_store_ptr),
      op_setter_ptr_(nullptr),
      op_impl_type_judge_ptr_(nullptr),
      format_dtype_setter_ptr_(nullptr),
      space_size_calculator_ptr_(nullptr),
      l2_optimize_ptr_(nullptr),
      lx_fusion_optimizer_ptr_(nullptr),
      graph_fusion_ptr_(nullptr),
      fusion_pass_mgr_ptr_(nullptr),
      fusion_rule_mgr_ptr_(nullptr),
      fusion_priority_mgr_ptr_(nullptr),
      graph_optimizer_attr_({engine_name, ge::ENGINE}),
      init_flag_(false),
      optimize_utility_(nullptr) {}

FEGraphOptimizer::~FEGraphOptimizer() {}

template <typename T>
Status FEGraphOptimizer::InitialzeOneCompiler(const string &compiler_name) {
  std::shared_ptr<T> op_compiler;
  FE_MAKE_SHARED(op_compiler = std::make_shared<T>(compiler_name, graph_optimizer_attr_.engineName,
                                                   lx_fusion_optimizer_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  Status ret = op_compiler->Initialize();
  if (ret != SUCCESS) {
    FE_LOGE("[GraphOpt][InitOneComp] Failed to Initialize %s.", compiler_name.c_str());
    return FAILED;
  }

  op_compiler_ptr_.emplace_back(op_compiler);
  return SUCCESS;
}

template <typename T>
Status FEGraphOptimizer::InitialzeOpTuneCompiler(const string &compiler_name) {
  std::shared_ptr<T> op_compiler;
  FE_MAKE_SHARED(op_compiler = std::make_shared<T>(compiler_name, graph_optimizer_attr_.engineName,
                                                   lx_fusion_optimizer_ptr_, ops_kernel_info_store_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  Status ret = op_compiler->Initialize();
  if (ret != SUCCESS) {
    FE_LOGE("[GraphOpt][InitOneComp] Failed to Initialize %s.", compiler_name.c_str());
    return FAILED;
  }

  op_compiler_ptr_.emplace_back(op_compiler);
  return SUCCESS;
}

Status FEGraphOptimizer::InitializeAllOpCompiler() {
  if (InitialzeOneCompiler<OpCompiler>("Op Compiler") != SUCCESS) {
    return FAILED;
  }

  if (InitialzeOneCompiler<OpCompilerBaseline>("Baseline Op Compiler") != SUCCESS) {
    return FAILED;
  }

  if (InitialzeOneCompiler<OpCompilerNormal>("Normal mode Op Compiler") != SUCCESS) {
    return FAILED;
  }

  if (InitialzeOpTuneCompiler<OpCompilerOpTune>("Op-Tune Op Compiler") != SUCCESS) {
    return FAILED;
  }

  if (InitialzeOneCompiler<OpCompilerMstuneBeforeUbMatch>("Before Ub Match Compiler") != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FEGraphOptimizer::Initialize(const std::map<string, string>& options,
                                    ge::OptimizeUtility *const optimize_utility) {
  (void)options;
  // if graph optimizer has been initialized, return success
  FE_TIMECOST_START(FEGraphOptimizerInitialize);
  if (init_flag_) {
    FE_LOGW("FEGraphOptimizer has been initialized.");
    FE_TIMECOST_END(FEGraphOptimizerInitialize, "FEGraphOptimizer.Initialize");
    return SUCCESS;
  }

  init_flag_ = true;
  optimize_utility_ = optimize_utility;
  FE_LOGD("Begin to init FEGraphOptimizer in engine[%s].", graph_optimizer_attr_.engineName.c_str());
  // initialize op compiler
  FE_CHECK(ops_kernel_info_store_ptr_ == nullptr, FE_LOGE("[GraphOpt][Init] opsKernelInfoStorePtr_ is NULL."),
           return FAILED);
  ops_kernel_info_store_ptr_->SetGeneralizeRelatedParam(optimize_utility, fusion_priority_mgr_ptr_);

  FE_MAKE_SHARED(op_impl_type_judge_ptr_ =
                     std::make_shared<OpImplTypeJudge>(graph_optimizer_attr_.engineName, ops_kernel_info_store_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_MAKE_SHARED(op_setter_ptr_ = std::make_shared<OpSetter>(graph_optimizer_attr_.engineName),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_MAKE_SHARED(format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(graph_optimizer_attr_.engineName),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_MAKE_SHARED(space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  FE_MAKE_SHARED(op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(graph_optimizer_attr_.engineName),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_MAKE_SHARED(l2_optimize_ptr_ = std::make_shared<L2Optimizer>(graph_optimizer_attr_.engineName),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_MAKE_SHARED(generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  FE_MAKE_SHARED(fe_lock_ptr_ = std::make_shared<std::mutex>(), return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  // init pass mgr ptr
  FE_MAKE_SHARED(fusion_pass_mgr_ptr_ = std::make_shared<FusionPassManager>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (fusion_pass_mgr_ptr_->Initialize(graph_optimizer_attr_.engineName) != SUCCESS) {
    FE_LOGE("[GraphOpt][Init] PassMngr initialize failed.");
    return FAILED;
  }

  // init rule mgr ptr
  FE_MAKE_SHARED(fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (fusion_rule_mgr_ptr_->Initialize(graph_optimizer_attr_.engineName) != SUCCESS) {
    FE_LOGE("[GraphOpt][Init] RuleMngr initialize failed.");
    return FAILED;
  }

  // init priority mgr ptr
  FE_MAKE_SHARED(fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
                     graph_optimizer_attr_.engineName, fusion_rule_mgr_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (fusion_priority_mgr_ptr_->Initialize() != SUCCESS) {
    FE_LOGE("[GraphOpt][Init] FusionPriorityMgr initialize failed.");
    return FAILED;
  }

  // init lx fusion optimizer
  FE_MAKE_SHARED(lx_fusion_optimizer_ptr_ = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr_ptr_,
                                                                                ops_kernel_info_store_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (lx_fusion_optimizer_ptr_->Initialize() != SUCCESS) {
    FE_LOGE("[GraphOpt][Init] LxFusionOptimizer initialize failed.");
    return FAILED;
  }

  // init graph fusion ptr
  FE_MAKE_SHARED(graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_,
                                                                   fusion_priority_mgr_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  graph_fusion_ptr_->SetEngineName(graph_optimizer_attr_.engineName);

  if (InitializeAllOpCompiler() != SUCCESS) {
    FE_TIMECOST_END(FEGraphOptimizerInitialize, "FEGraphOptimizer.Initialize");
    return FAILED;
  }
  FusionStatisticWriter::Instance().ClearHistoryFile();
  if (op_setter_ptr_->InitializeQuerier() != SUCCESS) {
    return FAILED;
  }
  FE_LOGI("Initialize success.");
  FE_TIMECOST_END(FEGraphOptimizerInitialize, "FEGraphOptimizer.Initialize");
  return SUCCESS;
}

Status FEGraphOptimizer::Finalize() {
  if (!init_flag_) {
    FE_LOGW("FEGraphOptimizer finalize is not allowed, initialize first is necessary.");
    return SUCCESS;
  }

  if (generate_cmo_type_manager_ptr_ != nullptr) {
    (void) generate_cmo_type_manager_ptr_->Finalize();
  }

  Status ret1 = SUCCESS;
  for (auto& compiler : op_compiler_ptr_) {
    if (compiler->Finalize() != SUCCESS) {
      FE_LOGE("[GraphOpt][Finalize] Failed to finalize %s.", compiler->GetCompilerName().c_str());
      ret1 = FAILED;
    }
  }

  Status ret2 = SUCCESS;
  if (fusion_pass_mgr_ptr_ != nullptr) {
    ret2 = fusion_pass_mgr_ptr_->Finalize();
    FE_LOGE_IF(ret2 != SUCCESS, "Pass Manager finalize failed.");
  }

  Status ret3 = SUCCESS;
  if (fusion_rule_mgr_ptr_ != nullptr) {
    ret3 = fusion_rule_mgr_ptr_->Finalize();
    FE_LOGE_IF(ret3 != SUCCESS, "Rule Manager finalize failed.");
  }

  Status ret4 = SUCCESS;
  if (lx_fusion_optimizer_ptr_ != nullptr) {
    ret4 = lx_fusion_optimizer_ptr_->Finalize();
    FE_LOGE_IF(ret4 != SUCCESS, "LxFusion optimizer finalize failed.");
  }

  if ((ret1 != SUCCESS) || (ret2 != SUCCESS) || (ret3 != SUCCESS)) {
    FE_LOGW("FE graph optimizer finalization failed!");
    return FAILED;
  }

  if (Configuration::Instance(AI_CORE_NAME).GetExportCompileStat() != ExportCompileStatType::NONE) {
    FusionStatisticWriter::Instance().WriteAllFusionInfoToJsonFile();
  }
  init_flag_ = false;
  FE_LOGD("Finalized successfully.");

  return SUCCESS;
}

Status FEGraphOptimizer::FinalizeSessionInfo(ge::ComputeGraph& graph) {
  if (ge::GraphUtils::IsSingleOpScene(graph.shared_from_this())) {
    FE_LOGI("[GraphOpt] _single_op_scene is true, skip FinalizeSessionInfo, graph name=%s.",
            graph.GetName().c_str());
    return SUCCESS;
  }

  std::string session_graph_id = "";
  if (ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) == false) {
    FE_LOGW("[GraphOpt] get session graph id failed, graph name=%s.", graph.GetName().c_str());
    return SUCCESS;
  }
  OpStoreAdapterPtr op_store_adapter = OpStoreAdapterManager::Instance(
                    graph_optimizer_attr_.engineName).GetOpStoreAdapter(EN_IMPL_HW_TBE);
  FE_CHECK_NOTNULL(op_store_adapter);
  Status res = op_store_adapter->FinalizeSessionInfo(session_graph_id);
  if (res != SUCCESS) {
    FE_LOGW("FinalizeSessionInfo failed!");
    return FAILED;
  }
  
  if (Configuration::Instance(AI_CORE_NAME).GetExportCompileStat() != ExportCompileStatType::NONE) {
    FusionStatisticWriter::Instance().WriteAllFusionInfoToJsonFile(session_graph_id);
  }
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph& graph) {
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageOrigin, graph.GetName());
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][init] FEGraphOptimizer has not been initialized.");
    return FAILED;
  }

  ops_kernel_info_store_ptr_->SetCheckSupportedStaticFlag(false);
  if (op_setter_ptr_ != nullptr) {
    op_setter_ptr_->SetOpImplMode(graph);
  }
  FE_TIMECOST_START(OptimizeOriginalGraph);
  FE_LOGD("Begin to optimize the original graph [%s] in engine [%s], with node size: %zu.", graph.GetName().c_str(),
          graph_optimizer_attr_.engineName.c_str(), graph.GetAllNodesSize());

  FE_TIMECOST_START(PruningPassFusion);
  Status ret = graph_fusion_ptr_->FusionPruningPass(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Pruning] Failed to execute pruning pass fusion for graph [%s].", graph.GetName().c_str());
    return ret;
  }
  FE_TIMECOST_END(PruningPassFusion, "FEGraphOptimizer::PruningPassFusion");

  FE_TIMECOST_START(BeforeQuantFusion);
  ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStageBeforeQuant, graph,
                                                    BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][BeforeQuant] Failed to execute before-quant-opt-graph-fusion for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }
  FE_TIMECOST_END(BeforeQuantFusion, "FEGraphOptimizer::BeforeQuantFusion");

  FE_TIMECOST_START(OptimizeQuantGraph);
  ret = graph_fusion_ptr_->FusionQuantOp(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][OptQuant] Quant optimization failed for graph [%s]", graph.GetName().c_str());
    return ret;
  }

  ret = graph.TopologicalSorting();
  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][BeforeFusion]Failed to do topological sorting before graph fusion for graph %s",
                    graph.GetName().c_str());
    return FAILED;
  }
  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeQuantGraph_FeGraphFusionAfter");

  FE_LOGI("Quant optimization successful, graph [%s].", graph.GetName().c_str());
  FE_TIMECOST_END(OptimizeQuantGraph, "FEGraphOptimizer::OptimizeQuantGraph");

  ret = ops_kernel_info_store_ptr_->SetDynamicCustomOpStoreInfo(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][BeforeFusion] Failed to set dynamic custom op store info for graph %s. ErrNo: %u.",
                    graph.GetName().c_str(), ret);
    return ret;
  }
  FE_LOGI("Graph[%s]: set dynamic custom op store info successfully.", graph.GetName().c_str());

  FE_TIMECOST_START(FusionGraph);
  ret = graph_fusion_ptr_->Fusion(graph);

  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][AfterFusion]Failed to perform graph fusion for graph %s. ErrNo: %u.",
                    graph.GetName().c_str(), ret);
    return ret;
  }

  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeOriginalGraph_FeGraphFusionAfter");
  FeGraphUtils::DumpSubGraphAndOnnx(graph, "OptimizeOriginalGraph_FeGraphFusionAfter_Subgraph");
  FE_TIMECOST_END(FusionGraph, "GraphFusion::Fusion during FEGraphOptimizer::OptimizeOriginalGraph");

  ret = graph.TopologicalSorting();

  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][AfterFusion] Failed to perform topological sorting after graph fusion for graph %s.",
                    graph.GetName().c_str());
    return FAILED;
  }

  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeOriginalGraph_FeTopoSortingAfter");
  FeGraphUtils::DumpSubGraphAndOnnx(graph, "OptimizeOriginalGraph_FeTopoSortingAfter_Subgraph");

  AddAssignMemAttr(graph);
  FE_LOGI("Optimize original graph[%s] successfully, node_size:%zu.", graph.GetName().c_str(), graph.GetAllNodesSize());
  FE_TIMECOST_END(OptimizeOriginalGraph, "FEGraphOptimizer::OptimizeOriginalGraph");
  ops_kernel_info_store_ptr_->SetCheckSupportedStaticFlag(true);
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeOriginalGraphOpJudgeAndFormatDtypeSetter(ge::ComputeGraph& graph) const {
  Status ret;
  // set the highest prior imply type for op
  ret = op_impl_type_judge_ptr_->MultiThreadJudge(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][Judge] Judge the op implementation failed, graph[%s].", graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("Optimizing original graph[%s] judge op implementation successfully.", graph.GetName().c_str());

  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][SetSupportFormat] Set the support format and dtype information failed, graph[%s].",
        graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("Optimizing original graph[%s] set the support format and dtype information successfully.",
          graph.GetName().c_str());
  return SUCCESS;
}

Status FEGraphOptimizer::InsertTransNodesForAllGraph(ge::ComputeGraph& graph,
                                                     TransNodeManagerPtr& trans_node_mgr_ptr) const {
  Status ret;
  // insert format and data type transfer op
  FE_MAKE_SHARED(trans_node_mgr_ptr = std::make_shared<TransNodeManager>(ops_kernel_info_store_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (trans_node_mgr_ptr->Initialize() != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Init] Failed to initialize transNodeMgrPtr for graph %s.", graph.GetName().c_str());
    return FAILED;
  }

  ret = trans_node_mgr_ptr->InsertAndMergeTransNodes(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Insert] Failed to insert format and dtype transformation operation for graph %s.",
                    graph.GetName().c_str());
    return ret;
  }

  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeOriginalGraph_FeInsertTransNodeAfter");

  FE_LOGI("Successfully inserted format and dtype transfer operation into the original graph, graph[%s].", graph.GetName().c_str());

  for (auto& subgraph : graph.GetAllSubgraphs()) {
    ret = trans_node_mgr_ptr->InsertAndMergeTransNodes(*(subgraph.get()));
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Trans][Insert] Failed to insert format and dtype transfer op for subgraph %s.",
                      subgraph->GetName().c_str());
      return ret;
    }
    FeGraphUtils::DumpGraphAndOnnx(*(subgraph.get()), "OptimizeOriginalGraph_FeInsertTransNodeAfter_Subgraph");
    FE_LOGI("Successfully inserted format and dtype transfer operation into subgraph, subgraph[%s].", subgraph->GetName().c_str());
  }
  return SUCCESS;
}

Status FEGraphOptimizer::GraphFusionBeforeTransnodesInsertion(ge::ComputeGraph& graph) const {
  if (graph_fusion_ptr_->SetContinuousDtypeForOutput(graph) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][GraphFusion][SetContinuousDtype] Failed to set continuous dtype for graph: %s.",
                    graph.GetName().c_str());
    return FAILED;
  }

  for (const auto& sub_graph : graph.GetAllSubgraphs()) {
    if (graph_fusion_ptr_->SetContinuousDtypeForOutput(*sub_graph) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][GraphFusion][SetContinuousDtype] Failed to set continuous dtype for sub-graph: %s.",
          sub_graph->GetName().c_str());
      return FAILED;
    }
  }

  Status ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStageJudgeInsert, graph,
                                                           BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][GraphFusion][Run] Fail to run graph fusion for graph[%s] before trans-nodes insertion.",
        graph.GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

#define FGO_RUN_AND_DUMP_WITH_EVENT(need_dump, name, func, ...)                                              \
  do {                                                                                                       \
    if (IsProcessBlockedByFusionSwitchCfg(name)) {                                                           \
      FE_LOGI("Process [%s] has been blocked by FusionSwitchCfg.", name);                                     \
    } else {                                                                                                 \
      if (need_dump) {                                                                                       \
        FE_RUN_AND_DUMP(FEGraphOptimizer, "OptimizeOriginalGraph_" name "After", func, __VA_ARGS__);         \
      } else {                                                                                               \
        FE_RUN(FEGraphOptimizer, func, __VA_ARGS__);                                                         \
      };                                                                                                     \
      FE_LOGI("Run %s on graph %s successfully.", name, graph.GetName().c_str());                            \
    }                                                                                                        \
  } while (0)

bool FEGraphOptimizer::IsProcessBlockedByFusionSwitchCfg(const std::string &process_name) {
  std::string build_inner_model = kStrTrue;
  ge::graphStatus status = ge::GetContext().GetOption("ge.build_inner_model", build_inner_model);
  if (status != ge::GRAPH_SUCCESS || build_inner_model == kStrTrue) {
    FE_LOGD("Get build_inner_model value is %s, status: %u.", build_inner_model.c_str(), status);
    return false;
  }

  return (!fusion_priority_mgr_ptr_->GetFusionSwitchByName(process_name, GRAPH_FUSION));
}

Status FEGraphOptimizer::GraphFusionAfterJudge(ge::ComputeGraph& graph) const {
  Status ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStageJudgeInsert, graph,
                                                           BUILT_IN_AFTER_OP_JUDGE);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][GraphFusionAfterJudge] Failed to run graph fusion for graph[%s] after op judge.",
        graph.GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

Status FEGraphOptimizer::HeavyFormatPropagate(ge::ComputeGraph &graph,
                                              HeavyFormatPropagationPtr heavy_format_prop_ptr) {
  std::string opt_val;
  (void)ge::GetThreadLocalContext().GetOo().GetValue(kComLevelO1Opt, opt_val);
  if (opt_val == kStrFalse) {
    bool is_quant_scene = false;
    for (const ge::NodePtr &node : graph.GetDirectNode()) {
      if (node->GetType() == ASCEND_QUANT || node->GetType() == OP_TYPE_QUANTIZE) {
        is_quant_scene = true;
        break;
      }
    }
    FE_LOGI("The O1 level quantization scene flag is [%d].", is_quant_scene);
    if (!is_quant_scene) {
      return SUCCESS;
    }
  }
  FGO_RUN_AND_DUMP_WITH_EVENT(true, "FeDistHeavyFormat", heavy_format_prop_ptr->PropagateHeavyFormat, graph);
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeOriginalGraphJudgeInsert(ge::ComputeGraph& graph) {
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageJudgeInsert, graph.GetName());
  FE_TIMECOST_START(OriginalGraphJudgeInsert);
  FeGraphUtils::DumpGraphAndOnnx(graph, "BeforeOptimizeOriginalGraphJudgeInsert");
  FeGraphUtils::DumpSubGraphAndOnnx(graph, "BeforeOptimizeOriginalGraphJudgeInsert_Subgraph");
  ClearUnknowShapeAttr(graph);
  FE_MAKE_SHARED(reflection_builder_ptr = std::make_shared<ge::RefRelations>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  ops_kernel_info_store_ptr_->SetCheckSupportedStaticFlag(false);
  FE_LOGD("Begin to judge the insertion of graph [%s] in engine [%s].", graph.GetName().c_str(),
          graph_optimizer_attr_.engineName.c_str());

  FE_MAKE_SHARED(op_format_dtype_judge_ptr = std::make_shared<OpFormatDtypeJudge>(
                     graph_optimizer_attr_.engineName, reflection_builder_ptr),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (op_format_dtype_judge_ptr->Initialize() != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][Init] Failed to initialize op_format_dtype_judge_ptr for graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }

  fe::PrecisionMode precision_mode = fe::PrecisionMode::ENUM_UNDEFINED;
  (void)FEContextUtils::GetPrecisionMode(precision_mode);
  (void)ge::AttrUtils::SetInt(graph, "graph_precision_mode", static_cast<int>(precision_mode));
  op_format_dtype_judge_ptr->SetPrecisionMode(precision_mode);
  (void)IsSingleOpGraphWithCache(graph);

  FGO_RUN_AND_DUMP_WITH_EVENT(false, "FeOptimizeOriginalGraphOpJudgeAndFormatDtypeSetter",
      OptimizeOriginalGraphOpJudgeAndFormatDtypeSetter, graph);

  // set the format and data type of the input and output desc of op
  (void)reflection_builder_ptr->Clear();
  auto status = reflection_builder_ptr->BuildRefRelations(graph);
  if (status != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][BuildRef] Failed to build reflection relations for main and subgraph %s.",
                    graph.GetName().c_str());
    return FAILED;
  }

  FGO_RUN_AND_DUMP_WITH_EVENT(true, "FeOpDtypeJudge", op_format_dtype_judge_ptr->Judge, graph);

  FE_TIMECOST_END(OriginalGraphJudgeInsert, "FEGraphOptimizer::OriginalGraphJudgeInsert");
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeOriginalGraphJudgeFormatInsert(ge::ComputeGraph& graph) {
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageJudgeInsert, graph.GetName());
  FE_LOGD("[GraphOptJdgFmtInst] Begin to judge format of graph[%s], in engine[%s].", graph.GetName().c_str(),
         graph_optimizer_attr_.engineName.c_str());
  FE_TIMECOST_START(OriginalGraphJudgeFormatInsert);
  FGO_RUN_AND_DUMP_WITH_EVENT(true, "FeOpFormatJudge", op_format_dtype_judge_ptr->SetFormat, graph);
  Status ret = GraphFusionAfterJudge(graph);
  if (ret != SUCCESS) {
    return ret;
  }
  HeavyFormatPropagationPtr heavy_format_propagation_ptr = nullptr;
  FE_MAKE_SHARED(heavy_format_propagation_ptr = std::make_shared<HeavyFormatPropagation>(
                     graph_optimizer_attr_.engineName, reflection_builder_ptr),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (heavy_format_propagation_ptr->Initialize() != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgFmtInst][Init] Failed to initialize heavy_format_propagation_ptr_ for graph[%s].",
                    graph.GetName().c_str());
    return FAILED;
  }
  FE_CHECK(HeavyFormatPropagate(graph, heavy_format_propagation_ptr) != SUCCESS, , return FAILED);

  FGO_RUN_AND_DUMP_WITH_EVENT(false, "FeUpdateAxis", op_axis_update_desc_ptr_->UpdateAxis, graph);
  FE_TIMECOST_START(SetFallbackAttr);
  fe::PrecisionMode precision_mode = fe::PrecisionMode::ENUM_UNDEFINED;
  (void)FEContextUtils::GetPrecisionMode(precision_mode);
  bool need_update_stream_core_limit = false;
  op_setter_ptr_->SetFallbackAttr(graph, precision_mode, need_update_stream_core_limit);
  if (need_update_stream_core_limit) {
    FE_LOGI("Graph attribute need_set_stream_core_limits is set true.");
    ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", true);
  }
  FE_TIMECOST_END(SetFallbackAttr, "OriginalGraphJudgeFormatInsert.SetFallbackAttr");
  ret = GraphFusionBeforeTransnodesInsertion(graph);
  if (ret != SUCCESS) {
    return ret;
  }

  TransNodeManagerPtr trans_node_mgr_ptr;
  FGO_RUN_AND_DUMP_WITH_EVENT(false, "FeInsertTransNodes", InsertTransNodesForAllGraph, graph, trans_node_mgr_ptr);
  FGO_RUN_AND_DUMP_WITH_EVENT(false, "FeSwitchTransDataAndCast", graph_fusion_ptr_->SwitchTransDataAndCast, graph,
      trans_node_mgr_ptr->GetOptimizableCast());

  ret = WeightCompressJudge::CompressTypeJudge(optimize_utility_, graph);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStageJudgeInsert, graph, SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgFmtInst][Run] Failed to execute the second round of fusion for graph [%s].",
                   graph.GetName().c_str());
    return ret;
  }
  StridedOptimize(graph);
  if (PlatformUtils::Instance().GetCubeVecState() == CubeVecStateNew::CUBE_VEC_SPLIT &&
      PlatformUtils::Instance().GetFftsMode() == FFTS_MODE_FFTS_PLUS) {
    // for new op insert between SECOND_ROUND_BUILT_IN_GRAPH_PASS
    ret = op_impl_type_judge_ptr_->MultiThreadJudge(graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgFmtInst][JudgeAfterSecondRoundPass] Judge Op implementation failed, graph[%s].",
                      graph.GetName().c_str());
      return ret;
    }

    // set the op information
    FE_TIMECOST_START(OriginalGraphJudgeInsertSetOpInfo);
    ret = op_setter_ptr_->MultiThreadSetOpInfo(graph);
    FE_TIMECOST_END(OriginalGraphJudgeInsertSetOpInfo, "OriginalGraphJudgeFormatInsert.SetOpInfo");
    if (ret != SUCCESS) {
      return ret;
    }
  }

  // calculate the input and output size of op.
  ret = space_size_calculator_ptr_->CalculateAICoreRunningParams(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgFmtInst][Calculate] Failed to calculate running parameters for graph [%s]",
                    graph.GetName().c_str());
    return ret;
  }

  ConvertExtAttr2Json(graph, false);
  FE_LOGI("[GraphOptJdgFmtInst] Optimizing original graph[%s] set op information successfully.",
          graph.GetName().c_str());

  (void)InsertClipByValue(graph);
  for (const auto &sub_graph : graph.GetAllSubgraphs()) {
    (void)InsertClipByValue(*sub_graph);
  }
  FE_CHECK(ConvertPartitionCalledOp(graph) != SUCCESS,
           REPORT_FE_ERROR("[GraphOptJdgFmtInst][TrsFuncOp] Failed to convert fixpipe to func op in graph [%s]",
                           graph.GetName().c_str()), return FAILED);

  ClearUnknowShapeAttr(graph);
  FE_LOGI("%s Graph[%s]: Successfully judged the format and data type of the operation and inserted the transformation operation. The node size is %zu.",
          kStageJudgeInsert.c_str(), graph.GetName().c_str(), graph.GetAllNodesSize());
  FE_TIMECOST_END(OriginalGraphJudgeFormatInsert, "FEGraphOptimizer::OriginalGraphJudgeFormatInsert");
  ops_kernel_info_store_ptr_->SetCheckSupportedStaticFlag(true);
  
  op_format_dtype_judge_ptr.reset();
  reflection_builder_ptr.reset();
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeAfterStage1(ge::ComputeGraph &graph) {
  FE_LOGD("Begin optimizing graph [%s] after stage 1.", graph.GetName().c_str());
  FE_CHECK_NOTNULL(graph_fusion_ptr_);
  Status status = graph_fusion_ptr_->RunGraphFusionPassByType(kStageAfterOptimizeStage1, graph,
                                                              BUILT_IN_AFTER_OPTIMIZE_STAGE1);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] Failed to run after-multi-dims for graph[%s].", graph.GetName().c_str());
    return status;
  }

  return SUCCESS;
}

Status FEGraphOptimizer::ShapeAndValueGeneralize(ge::ComputeGraph &graph) const {
  if (!IsFuzzBuild() && !IsShapeGeneralizedMode()) {
    FE_LOGD("[GraphOpt][Prepare][Generalize] No need to generalize the current graph [%s].", graph.GetName().c_str());
    return SUCCESS;
  }
  FE_LOGI("[GraphOpt][Prepare][Generalize] Begin to generalize graph[%s].", graph.GetName().c_str());
  (void)graph.TopologicalSorting();

  FuzzyGeneralizePtr fuzzy_generalize_ptr = nullptr;
  FE_MAKE_SHARED(fuzzy_generalize_ptr = std::make_shared<FuzzyGeneralize>(
    optimize_utility_, ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_),
    return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  Status res = fuzzy_generalize_ptr->GeneralizeGraph(graph);
  bool bres = (res == SUCCESS) ? true : false;
  FE_LOGI("[GraphOpt][Prepare][Generalize] End to generalize graph[%s], generalize result is [%d].",
          graph.GetName().c_str(), bres);
  return res;
}

void FEGraphOptimizer::AddAssignMemAttr(ge::ComputeGraph &graph) const {
  if (PlatformUtils::Instance().IsEnableL2CacheCmo()) {
    ge::AttrUtils::SetBool(graph, ge::ATTR_NAME_MEM_RELEASE_FIRST_REUSE_FIRST, true);
    FE_LOGI("[GraphOpt][Optimize] platform support cmo, assign mem FirstReleaseFirstReuse.");
  }
}

Status FEGraphOptimizer::OptimizeAfterGraphNormalization(const ge::ComputeGraphPtr &graph) {
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageAfterMultiDims, graph->GetName());
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] FEGraphOptimizer has not been initialized.");
    return FAILED;
  }
  Status ret = optimize_utility_->MultiDimsProcess(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] Failed to run multi-dimension process for graph [%s].", graph->GetName().c_str());
    return ret;
  }
  FE_LOGI("Optimize graph[%s] after multi dims successfully.", graph->GetName().c_str());

  ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStageAfterMultiDims, *graph, BUILT_IN_AFTER_MULTI_DIMS_PASS);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] Failed to run after-multi-dims for graph[%s].", graph->GetName().c_str());
    return ret;
  }

  FE_LOGI("Optimize graph [%s] successfully after changing constant.", graph->GetName().c_str());
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeGraphInit(ge::ComputeGraph& graph) {
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageInit, graph.GetName());
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][Init] FEGraphOptimizer has not been initialized.");
    return FAILED;
  }

  FE_TIMECOST_START(SortFusionPassByPriority);
  {
    std::lock_guard <std::mutex> lock_guard(sort_lock_);
    if (Configuration::Instance(graph_optimizer_attr_.engineName).RefreshParameters() != SUCCESS) {
      FE_LOGE("[GraphOpt][RefreshConfig] Failed to refresh parameters of configuration.");
      return FAILED;
    }

    if (fusion_priority_mgr_ptr_->Initialize() != SUCCESS) {
      FE_LOGE("[GraphOpt][InitFusionPrioMgr] Failed to init fusion priority manager.");
      return FAILED;
    }

    if (fusion_priority_mgr_ptr_->SortGraphFusion() != SUCCESS) {
      FE_LOGE("[GraphOpt][SortGraphFusion] Failed to sort graph fusion by priority.");
      return FAILED;
    }
    if (fusion_priority_mgr_ptr_->SortBufferFusion() != SUCCESS) {
      FE_LOGE("[GraphOpt][SortBufferFusion] Failed to sort buffer fusion by priority.");
      return FAILED;
    }
  }

  FE_TIMECOST_END(SortFusionPassByPriority, "FEGraphOptimizer::SortFusionPassByPriority");
  FE_LOGI("Sort fusion passes by priority graph[%s] successfully.", graph.GetName().c_str());
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeGraphPrepare(ge::ComputeGraph& graph) {
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStagePrepare, graph.GetName());
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] FEGraphOptimizer has not been initialized.");
    return FAILED;
  }
  OpSetter::SetOpDebugAttr(graph);
  OpSetter::SetQuantDumpableAttr(graph);
  if (ShapeAndValueGeneralize(graph) != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare] Graph[%s]: failed to generalize the graph.", graph.GetName().c_str());
  }

  FE_TIMECOST_START(OptimizeTagNoConstFoldingGraph);

  Status ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStagePrepare, graph,
                                                           BUILT_IN_TF_TAG_NO_CONST_FODING_GRAPH_PASS);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] Failed to apply no constant folding tag to graph %s", graph.GetName().c_str());
    return ret;
  }

  ret = graph_fusion_ptr_->RunGraphFusionPassByType(kStagePrepare, graph,
                                                    BUILT_IN_PREPARE_GRAPH_PASS);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Prepare] Failed to run prepare-graph-fusion for graph[%s].", graph.GetName().c_str());
    return ret;
  }

  FE_TIMECOST_END(OptimizeTagNoConstFoldingGraph, "FEGraphOptimizer::OptimizeTagNoConstFoldingGraph");
  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeGraph_TagNoConstFoldingAfter");
  FeGraphUtils::DumpSubGraphAndOnnx(graph, "OptimizeGraph_TagNoConstFoldingAfter_Subgraph");
  FE_LOGI("Optimize tag: no const folding for graph [%s] succeeded.", graph.GetName().c_str());
  return SUCCESS;
}

Status FEGraphOptimizer::ConvertPartitionCalledOp(ge::ComputeGraph& graph) {
  std::shared_ptr<fe::GraphComm> graph_comm_ptr = nullptr;
  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<fe::GraphComm>(AI_CORE_NAME, fe_lock_ptr_), return FAILED);
  FE_CHECK(graph_comm_ptr == nullptr, FE_LOGE("graphCommPtr is null."), return FAILED);
  if (graph_comm_ptr->ConvertFixpipePartitionCalledOp(graph, PlatformUtils::Instance().GetFftsMode()) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ConvertPartitionCalledOp] Failed to transform fixpipe to func op, graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FEGraphOptimizer::CheckAndSetAtomicAttr(const ge::OpDescPtr &op_desc, bool &atomic_node_flag) const {
  std::vector<uint32_t> output_index;
  std::map<string, std::map<int64_t, int64_t>> sub_node_workspace_info;
  // only process when get output_index success
  std::vector<uint32_t> tmp_output_index;
  if (ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index)) {
    uint32_t output_size = tmp_output_index.size();
    for (uint32_t i = 0; i < output_size; i++) {
      if (tmp_output_index[i] == 1) {
        output_index.push_back(i);
      }
    }
    if (!output_index.empty()) {
      (void)ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
      atomic_node_flag = true;
    }
    FE_LOGD("Finish setting tbe op [%s] output_index atomic info.", op_desc->GetName().c_str());
  }
  // process with workspace info
  std::vector<int64_t> tmp_workspace_index;
  std::vector<int64_t> workspace_index;
  if (ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_WORKSPACE_INDEX, tmp_workspace_index)) {
    for (size_t i = 0; i < tmp_workspace_index.size(); i++) {
      if (tmp_workspace_index[i] == 1) {
        workspace_index.push_back(i);
      }
    }
    std::map<int64_t, int64_t> workspace_info;
    std::vector<int64_t> workspace_bytes_vec = op_desc->GetWorkspaceBytes();
    if (!workspace_index.empty()) {
      for (int64_t index : workspace_index) {
        if (index >= static_cast<int64_t>(workspace_bytes_vec.size())) {
          continue;
        }
        workspace_info.insert(std::make_pair(index, workspace_bytes_vec[index]));
      }
      sub_node_workspace_info.insert(std::make_pair(op_desc->GetName(), workspace_info));
      if (!op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, sub_node_workspace_info)) {
        REPORT_FE_ERROR("[GraphOpt][SetAttr][SetExtAttr] Failed to set op [%s] workspace atomic info!",
                        op_desc->GetName().c_str());
        return fe::FAILED;
      }
      FE_LOGD("Finish setting tbe op [%s] workspace atomic info.", op_desc->GetName().c_str());
      atomic_node_flag = true;
    } else {
      FE_LOGD("Operation [%s] has no associated workspace atomic information.", op_desc->GetName().c_str());
    }
  }
  return fe::SUCCESS;
}

Status FEGraphOptimizer::HandleCompressOp(const ge::ComputeGraph &graph) const {
  WEIGHCOMPRESSINNERFLAG enable_sparsity = JudgeIsSparsityFlag();
  FE_LOGD("Handle compression before compile. Enable sparsity: [%d]", static_cast<int32_t>(enable_sparsity));
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (kCubeCompressOpList.count(node->GetType()) == 0) {
      continue;
    }
    FE_LOGD("Trying to link anchor for compress index input of node [%s, %s].",
            node->GetName().c_str(), node->GetType().c_str());
    ge::InDataAnchorPtr compress_index_in_anchor = node->GetInDataAnchor(TENSOR_INDEX_COMPRESS_INDEX);
    FE_CHECK_NOTNULL(compress_index_in_anchor);
    if (compress_index_in_anchor->GetPeerOutAnchor() != nullptr) {
      FE_LOGD("Compress index input is already linked.");
      continue;
    }

    ge::InDataAnchorPtr weight_in_anchor = node->GetInDataAnchor(TENSOR_INDEX_FILTER_COMPRESS);
    FE_CHECK_NOTNULL(weight_in_anchor);
    ge::OutDataAnchorPtr weight_peer_out_anchor = weight_in_anchor->GetPeerOutAnchor();
    if (weight_peer_out_anchor != nullptr) {
      if (ge::GraphUtils::AddEdge(weight_peer_out_anchor, compress_index_in_anchor) != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][CmprsOp][AddEdge] Failed to add edge for node [%s, %s]'s compress index input.",
                        node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
      FE_LOGD("Input compression for node [%s, %s] has been linked to the weight node.",
              node->GetName().c_str(), node->GetType().c_str());
    }

    if (enable_sparsity == WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG) {
      if (RefreshConvShapeForSpasity(node->GetOpDesc()) != SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][CmprsOp][AddEdge] Failed to refresh shape for node [%s, %s].",
                        node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status FEGraphOptimizer::HandleAclnnOp(ge::ComputeGraph &graph) const {
  if (!FeGraphCommon::IsUnknownGraph(graph.shared_from_this())) {
    FE_LOGD("Graph [%s] is not dynamic; no processing needed.", graph.GetName().c_str());
    return SUCCESS;
  }
  fe::PrecisionMode precision_mode = fe::PrecisionMode::ENUM_UNDEFINED;
  (void)FEContextUtils::GetPrecisionMode(precision_mode);
  std::string deterministic_str = "";
  (void)ge::GetThreadLocalContext().GetOption(ge::DETERMINISTIC, deterministic_str);
  std::string allow_hf32_str = "";
  (void)ge::GetThreadLocalContext().GetOption(ge::ALLOW_HF32, allow_hf32_str);
  for (const auto &node : graph.GetDirectNode()) {
    if (node->GetOpDesc()->HasAttr(ATTR_NAME_FALLBACK_ACLNN)) {
      continue;
    }
    int64_t imply_type = -1;
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), FE_IMPLY_TYPE, imply_type)) {
      continue;
    }
    OpImplType op_impl_type = static_cast<OpImplType>(imply_type);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(graph_optimizer_attr_.engineName).
                                         GetOpKernelInfoByOpType(op_impl_type, node->GetType());
    if (op_kernel_info_ptr == nullptr || !op_kernel_info_ptr->IsMultiKernelSupport()) {
      continue;
    }
    FE_LOGD("Node[%s] force to execute in aclnn.", node->GetName().c_str());
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FALLBACK_ACLNN, true);
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ge::kAttrLowingFunc, kAclnnLoweringFunc);
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kPrecisionModeEnum, static_cast<int>(precision_mode));
    if (!deterministic_str.empty()) {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ge::DETERMINISTIC, deterministic_str);
    }
    if (!allow_hf32_str.empty()) {
      allow_hf32_str = Configuration::Instance(AI_CORE_NAME).EmplaceHf32ModeForAclnn(allow_hf32_str);
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ALLOW_HF32, allow_hf32_str);
    }
  }
  return SUCCESS;
}

Status FEGraphOptimizer::SplitOptimizer(ge::ComputeGraph& graph,
                                              const bool& need_set_virtual_op) const {
  Status ret = FAILED;
  bool is_memory_discontinuous = false;
  (void)ge::AttrUtils::GetBool(graph, ge::ATTR_NAME_MEMORY_DISCONTIGUOUS_ALLOCATION, is_memory_discontinuous);
  if (ge::GraphUtils::IsSingleOpScene(graph.shared_from_this())) {
    FE_LOGI("Single op scene has no need to do concat/split optimize.");
    return SUCCESS;
  }
  if (need_set_virtual_op && !is_memory_discontinuous) {
    FE_LOGD("split_optimizer to set FusionVirtualOp.");
    ret = split_n_optimizer_ptr_->SetFusionVirtualOp(graph);
    if (ret != SUCCESS) {
      FE_LOGD("OptimizeOriginalGraphJudgeInsertSplit failed to set fusion_virtual_op, graph [%s].",
              graph.GetName().c_str());
      return ret;
    }
    FE_LOGD("Using split_c_to_n_optimizer to configure FusionVirtualOp.");
    (void)split_c_to_n_optimizer_ptr_->SetFusionVirtualOp(graph);
  } else {
    FE_LOGD("Fusion virtual op set switch is off for graph %s, is_memory_discontinuous:%d.",
            graph.GetName().c_str(), is_memory_discontinuous);
  }
  return SUCCESS;
}

Status FEGraphOptimizer::InsertClipByValue(ge::ComputeGraph& graph) const {
  if (!PlatformUtils::Instance().IsDCSoc()) {
    FE_LOGD("[SubGraphOpt][OptFusGraph][InstClip] No need to insert ClipByValue node.");
    return SUCCESS;
  }

  auto nodes = graph.GetDirectNode();
  for (auto &node : nodes) {
    if (node->GetType() == CONSTANT) {
      FE_CHECK_NOTNULL(node->GetOpDesc()->GetOutputDescPtr(0));
      ge::DataType out_dtype = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
      bool need_insert_clip = (out_dtype == ge::DT_FLOAT || out_dtype == ge::DT_DOUBLE) &&
                              (!node->GetOutDataNodes().empty());
      if (need_insert_clip) {
        // need insert clip by value
        CreateClipByValue(graph, node, true);
      }
    } else if (node->GetType() == OP_TYPE_PLACE_HOLDER) {
      std::string parent_op_engine_name;
      ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, parent_op_engine_name);
      FE_CHECK_NOTNULL(node->GetOpDesc()->GetOutputDescPtr(0));
      ge::DataType out_dtype = node->GetOpDesc()->GetOutputDescPtr(0)->GetDataType();
      bool need_insert_clip = (parent_op_engine_name == "DNN_VM_AICPU_ASCEND" ||
                               parent_op_engine_name == "DNN_VM_AICPU") &&
                              (out_dtype == ge::DT_FLOAT || out_dtype == ge::DT_DOUBLE) &&
                              (!node->GetOutDataNodes().empty());
      if (need_insert_clip) {
        // need insert clip by value
        CreateClipByValue(graph, node, false);
      }
    }
  }
  return SUCCESS;
}

Status FEGraphOptimizer::CreateClipByValue(ge::ComputeGraph& graph, const ge::NodePtr& node,
                                           const bool& const_input) const {
  FE_CHECK_NOTNULL(node);
  FE_CHECK_NOTNULL(node->GetOpDesc()->GetOutputDescPtr(0));
  ge::GeTensorDesc tensor_desc = node->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc const_desc;
  const_desc.SetDataType(tensor_desc.GetDataType());
  const_desc.SetOriginDataType(tensor_desc.GetOriginDataType());
  ge::OpDescPtr clip_desc = nullptr;
  std::string op_name = "clip_by_value_" + std::to_string(GetAtomicId());
  FE_MAKE_SHARED(clip_desc = std::make_shared<ge::OpDesc>(op_name, "ClipByValue"), return FAILED);
  clip_desc->AddInputDesc("x", tensor_desc);
  clip_desc->AddInputDesc("clip_value_min", const_desc);
  clip_desc->AddInputDesc("clip_value_max", const_desc);
  clip_desc->AddOutputDesc("y", tensor_desc);
  ge::NodePtr clip_node = graph.AddNode(clip_desc);
  FE_CHECK_NOTNULL(clip_node);
  ge::NodePtr min_node = CreateSalcarConst(graph, op_name, const_desc, true);
  FE_CHECK_NOTNULL(min_node);
  ge::NodePtr max_node = CreateSalcarConst(graph, op_name, const_desc, false);
  FE_CHECK_NOTNULL(max_node);
  (void)ge::GraphUtils::AddEdge(min_node->GetOutDataAnchor(0), clip_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(max_node->GetOutDataAnchor(0), clip_node->GetInDataAnchor(2));
  const auto out_anchor = node->GetOutDataAnchor(0);
  FE_CHECK_NOTNULL(out_anchor);
  auto peer_in_data_anchors = out_anchor->GetPeerInDataAnchors();
  for (const auto &peer_anchor : peer_in_data_anchors) {
    if (peer_anchor == nullptr || peer_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    auto op_desc = peer_anchor->GetOwnerNode()->GetOpDesc();
    if (!ge::AttrUtils::HasAttr(op_desc, FE_IMPLY_TYPE)) {
      std::string un_supported_reason;
      (void)ops_kernel_info_store_ptr_->CheckAccuracySupported(op_desc, un_supported_reason, true);
    }
    if (const_input && !ge::AttrUtils::HasAttr(op_desc, FE_IMPLY_TYPE)) {
      continue;
    }
    (void)ge::GraphUtils::RemoveEdge(out_anchor, peer_anchor);
    (void)ge::GraphUtils::AddEdge(clip_node->GetOutDataAnchor(0), peer_anchor);
  }
  (void)ge::GraphUtils::AddEdge(out_anchor, clip_node->GetInDataAnchor(0));
  if (clip_node->GetOutNodes().empty()) {
    FE_LOGD("[SubGraphOpt][OptFusGraph][CrtClip] Remove node[%s] from graph:%s.", clip_node->GetName().c_str(),
            graph.GetName().c_str());
    graph.RemoveNode(clip_node);
  }
  return SUCCESS;
}

ge::NodePtr FEGraphOptimizer::CreateSalcarConst(ge::ComputeGraph& graph, const std::string &clip_name,
                                                const ge::GeTensorDesc &tensor_desc, const bool min_value) const {
  ge::OpDescPtr op_desc = nullptr;
  std::string op_name = clip_name;
  if (min_value) {
    op_name += "/min";
  } else {
    op_name += "/max";
  }
  FE_MAKE_SHARED(op_desc = std::make_shared<ge::OpDesc>(op_name, CONSTANT), return nullptr);
  op_desc->AddOutputDesc(tensor_desc);
  ge::NodePtr node = graph.AddNode(op_desc);
  ge::GeTensor tensor;
  if (tensor_desc.GetDataType() == ge::DT_FLOAT) {
    float value = FLT_MAX;
    if (min_value) {
      value = -FLT_MAX;
    }
    std::unique_ptr<float> data(new (std::nothrow) float(value));
    tensor.SetTensorDesc(tensor_desc);
    tensor.SetData(reinterpret_cast<uint8_t *>(data.get()), sizeof(float));
  } else {
    double value = DBL_MAX;
    if (min_value) {
      value = -DBL_MAX;
    }
    std::unique_ptr<double> data(new (std::nothrow) double(value));
    tensor.SetTensorDesc(tensor_desc);
    tensor.SetData(reinterpret_cast<uint8_t *>(data.get()), sizeof(float));
  }
  const int32_t origin_element_num = 1;
  // this attr is used for constant_folding_pass, which set to fuse constant
  ge::AttrUtils::SetInt(tensor.MutableTensorDesc(), "origin_element_num", origin_element_num);
  if (node == nullptr) { FE_LOGE("Node is a nullptr."); return nullptr; }
  ge::AttrUtils::SetTensor(node->GetOpDesc(), "value", tensor);
  return node;
}

Status FEGraphOptimizer::CompileMemSetOp(ge::ComputeGraph& graph) const {
  string memset_policy_str;
  (void)ge::GetThreadLocalContext().GetOption(ge::ATOMIC_CLEAN_POLICY, memset_policy_str);
  if (memset_policy_str.empty()) {
    FE_LOGW("[SubGraphOpt][PostProcess][CompileMemSetOp] Graph[%s], get MEMSET_POLICY from ge is null.",
            graph.GetName().c_str());
    memset_policy_str = "0";
  }
  if (memset_policy_str != "0" && memset_policy_str != "1") {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {memset_policy_str, ge::ATOMIC_CLEAN_POLICY,
                               "The atomic clean policy value must be 0 or 1"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  FE_LOGI("[SubGraphOpt][PostProcess][CompileMemSetOp] Graph[%s]: memset policy is %s.", graph.GetName().c_str(),
          memset_policy_str.c_str());

  ClearSameMemSet(graph);
  vector<ge::NodePtr> memset_node_vec;
  if (GetAndPreProcessForAtomicNodes(graph, memset_node_vec, memset_policy_str) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompileMemSetOp] Failed to get memset node for graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }
  if (!memset_node_vec.empty() && ops_kernel_info_store_ptr_->CompileMemSet(memset_node_vec)) {
    return FAILED;
  }
  return SUCCESS;
}

void FEGraphOptimizer::ClearSameMemSet(ge::ComputeGraph &graph) const {
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  std::map<std::vector<std::string>, std::vector<ge::NodePtr>> same_memset_nodes_map;
  for (auto &node : graph.GetDirectNode()) {
    slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr == nullptr) {
      continue;
    }
    if (slice_info_ptr->same_atomic_clean_nodes.empty()) {
      continue;
    }
    auto same_memset_nodes = slice_info_ptr->same_atomic_clean_nodes;
    FE_LOGD("same_memset_nodes: %s, node: %s.", StringUtils::StrVecToString(same_memset_nodes).c_str(),
            node->GetName().c_str());
    if (same_memset_nodes_map.find(same_memset_nodes) == same_memset_nodes_map.end()) {
      same_memset_nodes_map[same_memset_nodes] = {node};
    } else {
      same_memset_nodes_map[same_memset_nodes].emplace_back(node);
    }
  }

  for (auto iter : same_memset_nodes_map) {
    for (size_t index = 1; index < iter.second.size(); ++index) {
      iter.second[index]->GetOpDesc()->DelAttr(TBE_OP_ATOMIC_OUTPUT_INDEX);
      iter.second[index]->GetOpDesc()->DelAttr(TBE_OP_ATOMIC_WORKSPACE_INDEX);
      iter.second[index]->GetOpDesc()->DelAttr(TBE_OP_ATOMIC_DTYPES);
      iter.second[index]->GetOpDesc()->DelAttr(TBE_OP_ATOMIC_INT64_VALUES);
      iter.second[index]->GetOpDesc()->DelAttr(TBE_OP_ATOMIC_FLOAT_VALUES);
    }
  }
}

Status FEGraphOptimizer::GetAndPreProcessForAtomicNodes(ge::ComputeGraph& graph,
    vector<ge::NodePtr> &atomic_node_vec, const string &memset_policy) const {
  for (auto &node : graph.GetDirectNode()) {
    FE_CHECK_NOTNULL(node);
    /* if the node is atomic node and connected to netoutput,
      then add it into atomic_node_vec and continue to next node */
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);

    bool is_tbe_op = false;
    bool atomic_node_flag = false;
    is_tbe_op = IsTbeOp(op_desc);
    if (is_tbe_op) {
      FE_LOGD("[SubGraphOpt][PostProcess][CompileAtomic] Start to set atomic attr for op[%s].",
          node->GetName().c_str());
      if (CheckAndSetAtomicAttr(op_desc, atomic_node_flag) != fe::SUCCESS) {
        FE_LOGW("[SubGraphOpt][PostProcess][CompileAtomic] Setting TBE op [%s] atomic info failed.",
                op_desc->GetName().c_str());
        return FAILED;
      }
      FE_LOGD("[SubGraphOpt][PostProcess][CompileAtomic] Node[%s]: end to set atomic attr. The atomic node flag is %d.",
              node->GetName().c_str(), atomic_node_flag);
    }

    if (!UnknownShapeUtils::IsUnknownShapeOp(*(op_desc.get())) &&
        (memset_policy.compare(kAtomicCleanTogetherMode) == 0)) {
      FE_LOGD("[SubGraphOpt][PostProcess][CompileAtomic] Op:%s is not of unknown shape, moving to the next one.",
          node->GetName().c_str());
      continue;
    }
    FE_LOGD("[SubGraphOpt][PostProcess][CompileAtomic] Op:%s is unknown shape, start to set atomic attr for it.",
        node->GetName().c_str());

    if (atomic_node_flag) {
      atomic_node_vec.emplace_back(node);
    }
  }
  return SUCCESS;
}

void FEGraphOptimizer::AllocMixResource(ge::ComputeGraph& graph) const {
  bool is_in_dyn_graph = (graph.GetParentNode() != nullptr) && graph.GetGraphUnknownFlag();
  for (auto &node : graph.GetDirectNode()) {
    std::string core_type;
    ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
    if (core_type != kCoreTypeMixVectorCore && core_type != kCoreTypeMixAICore) {
      continue;
    }
    if (NeedDisableVector(node)) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_DISABLE_MIX_VECTOR_CORE, true);
      continue;
    }
    FE_LOGD("%s Node [%s] is in the %s graph flag state.", core_type.c_str(), node->GetNamePtr(),
            is_in_dyn_graph ? "dynamic" : "static");
    ge::AttrUtils::SetStr(node->GetOpDesc(), kGroupPolicy, kVectorGroup);
    ge::GeAttrValue::NAMED_ATTRS stream_attrs;
    ge::AttrUtils::SetStr(stream_attrs, ge::ATTR_NAME_ATTACHED_STREAM_KEY, "mix_stream");
    ge::AttrUtils::SetStr(stream_attrs, ge::ATTR_NAME_ATTACHED_STREAM_POLICY, "group");
    (void)ge::AttrUtils::SetNamedAttrs(node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO, stream_attrs);
    ge::GeAttrValue::NAMED_ATTRS notify_attrs;
    ge::AttrUtils::SetStr(notify_attrs, ge::ATTR_NAME_ATTACHED_NOTIFY_KEY, "mix_notify");
    ge::AttrUtils::SetStr(notify_attrs, ge::ATTR_NAME_ATTACHED_NOTIFY_POLICY, "group");
    ge::AttrUtils::SetInt(notify_attrs, kAttachNotifyNumKey, kMixNotifyIdNum);
    (void)ge::AttrUtils::SetNamedAttrs(node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_NOTIFY_INFO, notify_attrs);
  }
  return;
}

Status FEGraphOptimizer::PostProcessAfterCompilingOp(ge::ComputeGraph& graph,
                                                     const BufferFusionPtr& buffer_fusion_ptr) {
  if (WeightPrefetchUtils::HandleWeightPrefetch(graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][WeightPrefetch] Failed to handle weight prefetch.");
    return FAILED;
  }

  FE_TIMECOST_START(FusionGraph);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  if (buffer_fusion_ptr->BuildFusionGraph(graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][BuildFusion] Failed to build fusion graph, graph[%s].",
                    graph.GetName().c_str());
    return FAILED;
  }
  FE_TIMECOST_END(FusionGraph, "BuildFusionGraph during FEGraphOptimizer::OptimizeFusedGraph");

  // calculate the input and output size of op.
  FE_CHECK(space_size_calculator_ptr_ == nullptr,
           REPORT_FE_ERROR("[GraphOpt][PostProcess][CalcuRunPara] The spaceSizeCalculatorPtr is null. Please check the initialization process."),
           return FAILED);
  Status ret = space_size_calculator_ptr_->CalculateRunningParams(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CalcuRunPara] Failed to calculate running parameters for graph [%s]",
                    graph.GetName().c_str());
    return ret;
  }
  AllocMixResource(graph);
  if (OptimizeGraphForTiling(graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][OptimizeGraphForTiling] Tiling failed for graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }
  // set atomic info to op
  if (CompileMemSetOp(graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CompileMemSetOp] Compilation of atomic operation failed for graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }

  // set task fusion scopeid
  MatchSuperkernelPlusNodes(graph);

  FE_LOGI("Optimize fused graph successfully.");
  return SUCCESS;
}

void FEGraphOptimizer::MatchSuperkernelPlusNodes(const ge::ComputeGraph& graph) const {
  if (!Configuration::Instance(graph_optimizer_attr_.engineName).IsEnableSuperkernelPlus()) {
    return;
  }
  std::vector<ge::NodePtr> match_nodes;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    // ignore non tbe op
    if (!IsTbeOp(node->GetOpDesc())) {
      continue;
    }
    if (IsNoTaskOp(node)) {
      continue;
    }
    int32_t block_dim = 0;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
    if (block_dim == 1) {
      match_nodes.push_back(node);
    } else {
      SetSkpScopeAttr(match_nodes);
      match_nodes.clear();
    }
  }
  SetSkpScopeAttr(match_nodes);
}

void FEGraphOptimizer::SetSkpScopeAttr(const std::vector<ge::NodePtr> &match_nodes) {
  if (match_nodes.size() > 1) {
    int64_t scope_id = ScopeAllocator::Instance().AllocateSkpScopeId();
    bool is_first_node = true;
    for (const ge::NodePtr &match_node : match_nodes) {
      if (is_first_node) {
        is_first_node = false;
      } else {
        (void)ge::AttrUtils::SetBool(match_node->GetOpDesc(), ge::ATTR_NAME_NOTASK, true);
      }
      (void)ScopeAllocator::SetSkpScopeAttr(match_node->GetOpDesc(), scope_id);
    }
    FE_LOGD("Set skp scope id [%ld] for nodes [%s], and set no task flag for all matched nodes except the first one.",
            scope_id, GetFusionNodesDescStr(match_nodes).c_str());
  }
}

Status FEGraphOptimizer::GetOpCompiler(OpCompilerPtr& op_compiler) const {
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  FE_LOGD("[SubGraphOpt][Compile][GetCompiler] Get build status from option: build_mode [%s], step [%s].",
          build_mode_value.c_str(), step_mode_value.c_str());
  auto size = op_compiler_ptr_.size();
  if (size != static_cast<size_t>(OpCompilerIndex::OP_COMPILER_BOTTOM)) {
    FE_LOGE("[SubGraphOpt][Compile][GetCompiler] Op compiler's size(%zu) is less than %u.",
            size, static_cast<uint32_t>(OpCompilerIndex::OP_COMPILER_BOTTOM));
    return FAILED;
  }

  if (build_mode_value != ge::BUILD_MODE_TUNING) {
    /* Baseline(using l2 buffer) or atc normal mode(using l2 fusion). */
    if (build_mode_value == ge::BUILD_MODE_BASELINE) {
      op_compiler = op_compiler_ptr_[static_cast<uint32_t>(OpCompilerIndex::BASELINE)];
    } else {
      op_compiler = op_compiler_ptr_[static_cast<uint32_t>(OpCompilerIndex::NORMAL)];
    }
  } else {
    if (step_mode_value == ge::BUILD_STEP_AFTER_BUILDER || step_mode_value == ge::BUILD_STEP_AFTER_BUILDER_SUB) {
      /* Op-tune scenario. */
      op_compiler = op_compiler_ptr_[static_cast<uint32_t>(OpCompilerIndex::OPTUNE)];
    } else if (step_mode_value == ge::BUILD_STEP_BEFORE_UB_MATCH) {
      /* Ms-tune stage 2.1: BUILD_STEP_BEFORE_UB_MATCH.
       * Try to compile first and roll back all nodes which cannot be compiled in a fusion scope to single node.
       * Single node can always be compiled successfully.
       * The purpose is to make sure all the fusion scope can be correctly compiled before ms-tuing.
       * */
      op_compiler = op_compiler_ptr_[static_cast<uint32_t>(OpCompilerIndex::MSTUNE_BEFORE_UB_MATCH)];
    } else if (step_mode_value.empty()) {
      op_compiler = op_compiler_ptr_[static_cast<uint32_t>(OpCompilerIndex::BASELINE)];
    } else {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][GetCompiler] Step %s is invalid in build mode %s.",
                      step_mode_value.c_str(), build_mode_value.c_str());
      return FAILED;
    }
  }
  FE_CHECK_NOTNULL(op_compiler);
  FE_LOGI("The compiler for mode [%s] step [%s] is %s.", build_mode_value.c_str(), step_mode_value.c_str(),
          op_compiler->GetCompilerName().c_str());
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeFusedCompileOpAndCalcTensorSize(const OpCompilerPtr &op_compiler,
                                                                 const BufferFusionPtr &buffer_fusion_ptr,
                                                                 ge::ComputeGraph &graph) {
  // compile tbe op
  Status ret = op_compiler->CompileOp(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][SubGraphOptAfterSlice][Compile] CompileOp failed for graph with name: %s.",
                    graph.GetName().c_str());
    return ret;
  }

  FE_LOGD("Optimizing fused graph: compile op successfully.");
  return PostProcessAfterCompilingOp(graph, buffer_fusion_ptr);
}

Status FEGraphOptimizer::BufferFusionMatch(ge::ComputeGraph& graph,
                                           const std::shared_ptr<FusionCycleDetector> &cycle_detector,
                                           const BufferFusionPtr& buffer_fusion_ptr) const {
  FE_TIMECOST_START(FusionMatch);
  // find sub-graphs that match UB fusion pattern
  if (buffer_fusion_ptr->MatchFusionPatternFromGraph(graph) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionMatch] Fusion pattern match failed for graph [%s].", graph.GetName().c_str());
    return FAILED;
  }
  FE_TIMECOST_END(FusionMatch, "FEGraphOptimizer::OptimizeFusedGraph.MatchFusionPattern");
  FE_TIMECOST_START(BufferFusionRun);
  std::unique_ptr<ConnectionMatrix> connection_matrix;
  /* Here, because we do not need to use cycle detector anymore,
   * we do not set connection matrix to cycle detector. */
  cycle_detector->GetConnectionMatrix(connection_matrix);
  AutomaticBufferFusionPtr auto_buffer_fusion_ptr = nullptr;
  FE_MAKE_SHARED(auto_buffer_fusion_ptr =
                 std::make_shared<AutomaticBufferFusion>(std::move(connection_matrix)),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (fusion_priority_mgr_ptr_->GetFusionSwitchByName("AutomaticUbFusion", UB_FUSION)) {
    if (auto_buffer_fusion_ptr->Run(graph) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionMatch] Failed to perform automatic buffer fusion for graph [%s].",
                      graph.GetName().c_str());
      return FAILED;
    }
  } else {
    FE_LOGI("Automatic buffer fusion is off for graph %s", graph.GetName().c_str());
  }
  FE_TIMECOST_END(BufferFusionRun, "FEGraphOptimizer::OptimizeFusedGraph.BufferFusionRun");
  return SUCCESS;
}

Status FEGraphOptimizer::SubGraphCompile(ge::ComputeGraph& graph, const OpCompilerPtr &op_compiler,
                                         const BufferFusionPtr &buffer_fusion_ptr) {
  Status ret = op_compiler->RunCompileProcess(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusedGraph][RunCompile] Failed to compile graph with compiler %s",
                    op_compiler->GetCompilerName().c_str());
    return ret;
  }

  if (!op_compiler->IsNeedPostProcess()) {
    std::string build_mode_value = FEContextUtils::GetBuildMode();
    std::string step_mode_value = FEContextUtils::GetBuildStep();
    FE_LOGD("In build mode %s and at build step %s, post processing is not required.", build_mode_value.c_str(),
            step_mode_value.c_str());
    return SUCCESS;
  }
  FE_LOGI("Optimizing fused graph: compile op successfully.");
  return PostProcessAfterCompilingOp(graph, buffer_fusion_ptr);
}

Status FEGraphOptimizer::PrecompileAndUbMatch(ge::ComputeGraph& graph, GraphCommPtr &graph_comm_ptr,
                                              OpCompilerPtr &op_compiler, BufferFusionPtr &buffer_fusion_ptr) const {
  if (GetOpCompiler(op_compiler) != SUCCESS) {
    return FAILED;
  }
  // pre compile tbe op
  Status ret = op_compiler->UpdateCompileParams(graph);
  if (ret != SUCCESS) {
    return ret;
  }

  ge::TraceOwnerGuard guard_precompile(FE_MODULE_NAME, kStagePreCompile, graph.GetName());
  ret = op_compiler->PreCompileOp(graph);
  if (ret != SUCCESS) {
    return ret;
  }
  FE_LOGI("Optimizing fused graph name = %s: precompile op successfully.", graph.GetName().c_str());
  // Do topo sort before cycle detection Init, otherwise connection matrix will be wrong
  FE_CHECK(graph.TopologicalSorting() != ge::GRAPH_SUCCESS,
           REPORT_FE_ERROR("[GraphOpt][FusedGraph][TopoSort] failed before ub match"), return FAILED);

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(), return FAILED);
  cycle_detector->Initialize(graph);

  OpStoreAdapterPtr op_store_adapter = OpStoreAdapterManager::Instance(
                    graph_optimizer_attr_.engineName).GetOpStoreAdapter(EN_IMPL_HW_TBE);
  FE_CHECK_NOTNULL(op_store_adapter);

  FE_MAKE_SHARED(buffer_fusion_ptr = std::make_shared<BufferFusion>(
                 graph_comm_ptr, fusion_priority_mgr_ptr_, op_store_adapter, cycle_detector),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  buffer_fusion_ptr->SetEngineName(graph_optimizer_attr_.engineName);

  ret = BufferFusionMatch(graph, cycle_detector, buffer_fusion_ptr);
  if (ret != SUCCESS) {
    return ret;
  }

  return SUCCESS;
}

bool FEGraphOptimizer::CheckNeedSetSliceInfo(ge::ComputeGraph& graph) const {
  bool sgt_sliced = false;
  for (const auto &node : graph.GetDirectNode()) {
    FE_CHECK(node == nullptr, FE_LOGE("Node is a nullptr."), return false);
    if (!IsValidOp(node)) {
      continue;
    }
    auto op_desc_ptr = node->GetOpDesc();
    uint32_t thread_scope_id = 0xFFFFFFFF;
    (void)ge::AttrUtils::GetInt(op_desc_ptr, kThreadScopeId, thread_scope_id);
    if (thread_scope_id != 0xFFFFFFFF) {
      FE_LOGD("Operation %s belongs to the ffts-subgraph; there is no need to set slice information for the current graph %s.",
              node->GetName().c_str(), graph.GetName().c_str());
      sgt_sliced = true;
    }
    break;
  }
  bool no_need_lx = (PlatformUtils::Instance().GetCubeVecState() == CubeVecStateNew::CUBE_VEC_SPLIT &&
                     PlatformUtils::Instance().GetFftsMode() == FFTS_MODE_FFTS_PLUS) && FEContextUtils::IsTrainMode();
  bool bres = !sgt_sliced && !no_need_lx;
  (void)ge::AttrUtils::SetBool(graph, kNeedSetSliceInfo, bres);
  FE_LOGD("Set attribute need_set_slice_info [%d] for current graph %s.",
          bres, graph.GetName().c_str());
  return bres;
}

Status FEGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph& graph) {
  FE_LOGD("Begin optimizing fused graph in engine [%s].", graph_optimizer_attr_.engineName.c_str());
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageFused, graph.GetName());
  if (!init_flag_) {
    FE_LOGW("OptimizeFusedGraph is not allowed; initialize first.");
    return FAILED;
  }
  FE_TIMECOST_START(OptimizeFusedGraph);
  TraceHandleManager::Instance().AddSubGraphTraceHandle();

  GraphCommPtr graph_comm_ptr = nullptr;
  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<GraphComm>(graph_optimizer_attr_.engineName, fe_lock_ptr_),
      return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  if (graph_comm_ptr->Initialize() != SUCCESS) {
    FE_LOGW("GraphComm initialize failed");
    return FAILED;
  }
  // UnfoldSubGraph
  FE_CHECK(graph_comm_ptr->UnfoldFuncOp(graph) != SUCCESS,
           REPORT_FE_ERROR("[GraphOpt][FusedGraph] UnfoldFuncOp failed, graph[%s].", graph.GetName().c_str()),
           return FAILED);

  (void)InsertClipByValue(graph);

  /* set the highest prior imply type for op which is inserted between original graph
  optimization and fused graph optimization */
  FE_CHECK(op_impl_type_judge_ptr_ == nullptr, REPORT_FE_ERROR("[GraphOpt][FusedGraph] opImplTypeJudgePtr_ is null."),
           return FAILED);
  Status ret = op_impl_type_judge_ptr_->JudgeInSubGraph(graph);
  if (ret != SUCCESS) {
    return ret;
  }
  FE_LOGI("Optimizing fused graph: %s judge op succeeded.", graph.GetName().c_str());

  // set the op information
  FE_TIMECOST_START(SetOpInfo);
  if (CheckNeedSetSliceInfo(graph)) {
    ret = op_setter_ptr_->SetOpInfo(graph);
  } else {
    ret = op_setter_ptr_->OnlySetOpDescAttr(graph);
  }
  FE_TIMECOST_END(SetOpInfo, "OptimizedFusedGraph.SetOpInfo");
  if (ret != SUCCESS) {
    return ret;
  }
  FE_LOGI("Optimizing fused graph:%s set the op information successfully.", graph.GetName().c_str());

  // handle compress op
  ret = HandleCompressOp(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusedGraph][InsCmprsOP] Failed to insert compression operation for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }

  // handle aclnn op
  ret = HandleAclnnOp(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusedGraph][InsCmprsOP] Failed to process aclnn op for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }

  OpCompilerPtr op_compiler;
  BufferFusionPtr buffer_fusion_ptr;
  ret = PrecompileAndUbMatch(graph, graph_comm_ptr, op_compiler, buffer_fusion_ptr);
  if (ret != SUCCESS) {
    return ret;
  }

  ConvertJson2ExtAttr(graph);
  ge::TraceOwnerGuard guard_compile(FE_MODULE_NAME, kStageCompile, graph.GetName());
  ret = SubGraphCompile(graph, op_compiler, buffer_fusion_ptr);
  // count match & effect times
  buffer_fusion_info_collecter_lock_.lock();
  if (buffer_fusion_info_collecter_ptr_ == nullptr) {
    FE_MAKE_SHARED(buffer_fusion_info_collecter_ptr_ = std::make_shared<BufferFusionInfoCollecter>(), return FAILED);
  }
  buffer_fusion_info_collecter_lock_.unlock();
  buffer_fusion_info_collecter_ptr_->CountBufferFusionTimes(graph);
  if (ret != SUCCESS) {
    return FAILED;
  }
  ConvertExtAttr2Json(graph, true);
  FE_TIMECOST_END(OptimizeFusedGraph, "FEGraphOptimizer::OptimizeFusedGraph");
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   optimize fused graph for LXfusion
 *  @param   [in|out] graph   compute graph
 *  @return  SUCCESS or FAILED
 */
Status FEGraphOptimizer::OptimizeFusedGraphAfterGraphSlice(ge::ComputeGraph& graph) {
  FE_LOGD("Begin optimizing the fused graph for the second stage in engine [%s].", graph_optimizer_attr_.engineName.c_str());
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageAftFmtSlct, graph.GetName());
  FE_TIMECOST_START(OptimizeFusedGraphAfterGraphSlice);
  if (!init_flag_) {
    FE_LOGW("OptimizeFusedGraphAfterGraphSlice is not permitted; please initialize first.");
    return FAILED;
  }
  TraceHandleManager::Instance().AddSubGraphTraceHandle();

  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  FE_LOGD("Optimizing fused graph in the second stage, build_mode is [%s], and build_step is [%s].",
          build_mode_value.c_str(), step_mode_value.c_str());
  Status ret;
  // handle compress op
  ret = HandleCompressOp(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][CompileAfterSlice][InsCmprss] Failed to insert compress op for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }

  OpCompilerPtr op_compiler = op_compiler_ptr_[static_cast<size_t>(OpCompilerIndex::COMMON)];
  if (op_compiler == nullptr) {
    return FAILED;
  }

  OpCompilerFormatTunePtr op_compiler_format_tune_ptr = nullptr;
  FE_MAKE_SHARED(op_compiler_format_tune_ptr = std::make_shared<OpCompilerFormatTune>(graph_optimizer_attr_.engineName),
                 return fe::OP_COMPILER_MAKE_SHARED_FAILED);
  FE_CHECK_NOTNULL(op_compiler_format_tune_ptr);
  ret = op_compiler_format_tune_ptr->UpdateTuneFormatByNodeAttr(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][CompileAfterSlice][FormatTune] Failed to update tuneformat using node attributes for graph %s.",
                    graph.GetName().c_str());
    return ret;
  }

  ret = op_compiler->UpdateCompileParams(graph);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = op_compiler->PreCompileOp(graph);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG(EM_INNER_ERROR.c_str(),
                       "[SubGraphOpt][CompileAfterSlice][Pre-Comp] Failed to precompile graph %s after slice.",
                       graph.GetName().c_str());
    return ret;
  }
  FE_LOGD("Optimizing fused graph name = %s: precompile op successfully.", graph.GetName().c_str());

  GraphCommPtr graph_comm_ptr = nullptr;
  BufferFusionPtr buffer_fusion_ptr;

  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<GraphComm>(graph_optimizer_attr_.engineName, fe_lock_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (graph_comm_ptr->Initialize() != SUCCESS) {
    FE_LOGW("GraphComm initialize failed");
    return FAILED;
  }

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(), return FAILED);
  cycle_detector->Initialize(graph);

  OpStoreAdapterPtr op_store_adapter = OpStoreAdapterManager::Instance(
                    graph_optimizer_attr_.engineName).GetOpStoreAdapter(EN_IMPL_HW_TBE);
  FE_CHECK_NOTNULL(op_store_adapter);

  FE_MAKE_SHARED(buffer_fusion_ptr = std::make_shared<BufferFusion>(
                 graph_comm_ptr, fusion_priority_mgr_ptr_, op_store_adapter, cycle_detector),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  buffer_fusion_ptr->SetEngineName(graph_optimizer_attr_.engineName);
  ret = OptimizeFusedCompileOpAndCalcTensorSize(op_compiler, buffer_fusion_ptr, graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][CompileAfterSlice][FusComp] Failed to optimize the fused graph after UB match for graph [%s].",
        graph.GetName().c_str());
    return ret;
  }
  FE_TIMECOST_END(OptimizeFusedGraphAfterGraphSlice, "FEGraphOptimizer::OptimizeFusedGraphAfterGraphSlice");
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   optimize stream graph (now only for single stream graph)
 */
Status FEGraphOptimizer::OptimizeStreamGraph(ge::ComputeGraph& stream_graph, const ge::RunContext& context) {
  FE_LOGI("FEGraphOptimizer start optimizing stream graph.");
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStageLx, stream_graph.GetName());
  string session_graph_id = "";
  if (ge::AttrUtils::GetStr(stream_graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) &&
      !session_graph_id.empty()) {
    FE_LOGD("stream session_graph_id=%s", session_graph_id.c_str());
  }
  // write fusion info to file
  string context_graph_id = "";
  FeGraphUtils::GetGraphIdFromAttr(stream_graph, context_graph_id);
  FE_LOGD("Stream session id %lu graph id %s.", context.sessionId, context_graph_id.c_str());

  if (!init_flag_) {
    FE_LOGW("OptimizeStreamGraph is not allowed. Initialization must be performed first.");
    return FAILED;
  }
  int64_t stream_id = kInvalidStreamId;
  // if find any unknown shape node, return success
  for (auto &node_ptr : stream_graph.GetDirectNode()) {
    bool unknown_shape = true;
    if (ge::NodeUtils::GetNodeUnknownShapeStatus(*(node_ptr.get()), unknown_shape) == ge::GRAPH_SUCCESS) {
      if (unknown_shape) {
        FE_LOGI("Optimize stream graph encountered unknown_shape node: %s", stream_graph.GetName().c_str());
        return SUCCESS;
      }
    }
    if (node_ptr->GetOpDesc()->GetStreamId() != kInvalidStreamId) {
      stream_id = node_ptr->GetOpDesc()->GetStreamId();
    }
  }
  // choose L2 cache or L2 buffer mode, if L2 buffer mode, dynamic batch judge
  FE_CHECK(l2_optimize_ptr_ == nullptr, REPORT_FE_ERROR("[GraphOpt][OptStream] l2OptimizePtr_ is null."),
           return fe::FAILED);
  Status ret = l2_optimize_ptr_->GetL2DataAlloc(stream_graph,
                                                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context.dataMemBase)),
                                                stream_id);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][OptStream][GetL2DataAlloc] GetL2DataAlloc failed for graph [%s].",
                    stream_graph.GetName().c_str());
    return ret;
  }

  FE_LOGI("Optimize stream graph successfully.");
  return SUCCESS;
}

void FEGraphOptimizer::FeedStreamCtrlMap(ge::NodePtr &node, int64_t &event_id, StreamCtrlMap &stream_ctrls) const {
  if (node == nullptr) {
    return;
  }
  std::string node_type = node->GetType();
  if (node_type == kSend) {
    auto in_control_nodes = node->GetInControlNodes();
    if (in_control_nodes.empty()) {
      return;
    }
    ge::NodePtr src_node = in_control_nodes.at(0);
    int64_t stream_id = src_node->GetOpDesc()->GetStreamId();
    stream_ctrls[event_id].src_stream_id = stream_id;
    stream_ctrls[event_id].src_ctrl_node = src_node;
  } else {
    auto out_control_nodes = node->GetOutControlNodes();
    if (out_control_nodes.empty()) {
      return;
    }
    ge::NodePtr dst_node = out_control_nodes.at(0);
    int64_t stream_id = dst_node->GetOpDesc()->GetStreamId();
    stream_ctrls[event_id].dst_stream_id = stream_id;
    stream_ctrls[event_id].dst_ctrl_node = dst_node;
  }
  return;
}

void FEGraphOptimizer::GenerateStreamCtrlMap(ge::ComputeGraph &graph, StreamCtrlMap &stream_ctrls) const {
  for (auto &node : graph.GetDirectNode()) {
    std::string node_type = node->GetType();
    if (node_type == kSend || node_type == kRecv) {
      int64_t event_id = -1;
      (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::SEND_ATTR_EVENT_ID, event_id);
      if (event_id == -1) {
        continue;
      }
      FeedStreamCtrlMap(node, event_id, stream_ctrls);
    }
  }
}

Status FEGraphOptimizer::OptimizeStreamedWholeGraph(ge::ComputeGraph &graph) {
  if (!PlatformUtils::Instance().IsEnableL2CacheCmo()) {
    FE_LOGI("FEGraphOptimizer does not use CMO mode.");
    return SUCCESS;
  }

  FE_LOGI("FEGraphOptimizer start optimize streamed whole graph.");
  if (!init_flag_) {
    FE_LOGW("Initializing first is necessary as OptimizeStreamedWholeGraph is not allowed without it.");
    return FAILED;
  }

  StreamCtrlMap stream_ctrls;
  GenerateStreamCtrlMap(graph, stream_ctrls);
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  generate_cmo_type_manager_ptr_->Initialize();
  for (const auto &node_ptr : graph.GetDirectNode()) {
    generate_cmo_type_manager_ptr_->GenerateType(node_ptr, stream_ctrls, prefetch_cache_map, stream_node_map);
  }
  FE_LOGI("FEGraphOptimizer end optimize streamed whole graph.");
  return SUCCESS;
}

Status FEGraphOptimizer::GetAttributes(ge::GraphOptimizerAttribute& attrs) const {
  attrs = graph_optimizer_attr_;
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph& graph) {
  std::string build_mode = FEContextUtils::GetBuildMode();
  if (build_mode != ge::BUILD_MODE_TUNING) {
    FE_LOGD("Got build mode: %s, no need to restore ext_attr and sub_node_workspace_info.",
            build_mode.c_str());
    return SUCCESS;
  }
  // 恢复扩展属性sub_node_workspace_info
  // 解决问题: aoe sgat会跳过子图优化，导致memset扩展属性丢失，未作集中清零、精度异常。
  bool atomic_node_flag = false;
  for (auto &node : graph.GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (ops_kernel_info_store_ptr_->SetMemSetOpWorkspaceInfo(op_desc, atomic_node_flag) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FEGraphOptimizer::OptimizeGraphForTiling(ge::ComputeGraph& graph) const {
  FE_TIMECOST_START(OptimizeGraphForTiling);
  if (ge::RecoverIrDefinitions(graph.shared_from_this()) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  FE_TIMECOST_END(OptimizeGraphForTiling, "RecoverIrDefinitions");
  for (auto &node : graph.GetAllNodes()) {
    if (!IsStaticOrAutoFuseReuseBinaryOp(node)) {
      FE_LOGD("Node %s is not a reusable operation.", node->GetName().c_str());
      continue;
    }
    if (TilingForOneNode(node) != SUCCESS) {
      return FAILED;
    }
    if (UpdateTileFwkKernelInfo(node->GetOpDesc()) != SUCCESS) {
      FE_LOGE("Node[%s, %s]: failed to update tile fwk kernel info.", node->GetNamePtr(), node->GetTypePtr());
      return FAILED;
    }
  }
  return SUCCESS;
}

bool FEGraphOptimizer::CheckExportFusionResCond(ge::ComputeGraph &graph) const {
  if (Configuration::Instance(AI_CORE_NAME).GetExportCompileStat() != ExportCompileStatType::AFTER_COMPILE_COMPLITE) {
    return false;
  }
  if (FEContextUtils::IsOpTuneMode()) {
    return false;
  }
  if (ge::GraphUtils::IsSingleOpScene(graph.shared_from_this())) {
    return false;
  }
  return true;
}


Status FEGraphOptimizer::OptimizeGraphBeforeBuild(ge::ComputeGraph& graph) {
  if (PlatformUtils::Instance().IsEnableL2CacheCmo()) {
    ge::AttrUtils::SetBool(graph, "_op_need_multi_task", true);
    FE_LOGI("[GraphOpt][Optimize] Platform support CMO, set attribute _op_need_multi_task for graph[%s].",
            graph.GetName().c_str());
  }

  // set fusion_virtual_op info to op
  bool need_set_virtual_op =
          fusion_priority_mgr_ptr_->GetFusionSwitchByName(kFusionVirtualOpSetSwitch, UB_FUSION);
  if (SplitOptimizer(graph, need_set_virtual_op) != SUCCESS) {
    FE_LOGD("SplitOptimizer unsuccessful, graph [%s].", graph.GetName().c_str());
    return FAILED;
  }

  for (const auto &subgraph : graph.GetAllSubgraphs()) {
    FE_CHECK(SplitOptimizer(*(subgraph.get()), need_set_virtual_op) != SUCCESS,
             FE_LOGD("SplitOptimizer subgraph unsuccess, subgraph [%s].",
                     subgraph->GetName().c_str()), return FAILED);
  }
  op_setter_ptr_->DeleteFusionScope(graph);
  if (CheckExportFusionResCond(graph)) {
    FusionStatisticWriter::Instance().WriteAllFusionInfoToJsonFile();
  }
  return SUCCESS;
}

REG_PASS_OPTION(kFusionVirtualOpSetSwitch).LEVELS(ge::OoLevel::kO2);

void FEGraphOptimizer::StridedOptimize(ge::ComputeGraph& graph) const {
  std::shared_ptr<StridedWriteOptimizer> stridedwrite_optimizer = nullptr;
  FE_MAKE_SHARED(stridedwrite_optimizer = std::make_shared<fe::StridedWriteOptimizer>(), return);
  std::shared_ptr<StridedReadOptimizer> stridedread_optimizer = nullptr;
  FE_MAKE_SHARED(stridedread_optimizer = std::make_shared<fe::StridedReadOptimizer>(), return);
  stridedwrite_optimizer->DoOptimizeForConcat(graph);
  stridedread_optimizer->DoOptimizeForSplit(graph);
  return;
}

void FEGraphOptimizer::ClearUnknowShapeAttr(const ge::ComputeGraph& graph) const {
  for (auto& node : graph.GetAllNodes()) {
    (void)node->GetOpDesc()->DelAttr(fe::ATTR_NAME_UNKNOWN_SHAPE);
  }
}

void FEGraphOptimizer::ConvertExtAttr2Json(const ge::ComputeGraph& graph, bool need_delete_ext_attr) const {
  if (Configuration::Instance(graph_optimizer_attr_.engineName).GetDumpGeGraph().empty()) {
    return;
  }

  const auto &nodes = graph.GetAllNodes();
  const std::string name_str = "name:";
  const std::string type_str = "type:";
  try {
    for (const auto &node : nodes) {
      std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp =
          nullptr;
      FE_MAKE_SHARED((op_attrs_maps_tmp =
                     std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>()),
                     return);
      const ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      op_attrs_maps_tmp = op_desc_ptr->TryGetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
      if (op_attrs_maps_tmp == nullptr || op_attrs_maps_tmp->empty()) {
        continue;
      }
      for (auto &it : (*op_attrs_maps_tmp)) {
        for (auto &op_attrs_vec : it.second) {
          if (op_attrs_vec[0].find(name_str) == std::string::npos) {
            op_attrs_vec[0] = name_str + op_attrs_vec[0];
            op_attrs_vec[1] = type_str + op_attrs_vec[1];
          }
        }
      }
      nlohmann::json json_object(*op_attrs_maps_tmp);
      (void)ge::AttrUtils::SetStr(op_desc_ptr, ge::ATTR_NAME_ORIGIN_OP_ATTRS_IN_FUSION_PROCESS, json_object.dump());
      if (need_delete_ext_attr) {
        op_desc_ptr->DelAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP);
      }
    }
  } catch (const std::exception &e) {
    FE_LOGW("Parsing JSON string failed, message is: %s, please check ExtAttr.", e.what());
  }
}

void FEGraphOptimizer::ConvertJson2ExtAttr(const ge::ComputeGraph& graph) const {
  if (Configuration::Instance(graph_optimizer_attr_.engineName).GetDumpGeGraph().empty()) {
    return;
  }

  const auto &nodes = graph.GetAllNodes();
  try {
    for (const auto &node : nodes) {
      std::vector<std::string> pass_names;
      const ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      (void)ge::AttrUtils::GetListStr(op_desc_ptr, kPassNameAttr, pass_names);
      if (pass_names.empty()) {
        continue;
      }
      std::string json_str;
      (void)ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_ORIGIN_OP_ATTRS_IN_FUSION_PROCESS, json_str);
      if (json_str.empty()) {
        continue;
      }
      std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> origin_op_attrs_map =
          nullptr;
      FE_MAKE_SHARED((origin_op_attrs_map =
                     std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>()),
                     return);
      for (const auto &pass_name : pass_names) {
        std::vector<std::vector<std::string>> origin_op_attrs_tmp;
        const nlohmann::json &json_value = nlohmann::json::parse(json_str);
        json_value.at(pass_name).get_to(origin_op_attrs_tmp);
        origin_op_attrs_map->insert(std::pair<std::string,
                                    std::vector<std::vector<std::string>>>(pass_name, origin_op_attrs_tmp));
      }
      (void)op_desc_ptr->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, origin_op_attrs_map);
    }
  } catch (const std::exception &e) {
    FE_LOGW("Parsing JSON string failed, message is: %s, please check ExtAttr.", e.what());
  }
}

Status FEGraphOptimizer::OptimizeSubgraphOfPrecompiledOp(ge::ComputeGraph &graph, const KernelLookup &lookup) {
  FE_LOGD("Begin to recover node info in om graph[%s] in engine[%s].",
          graph.GetName().c_str(), graph_optimizer_attr_.engineName.c_str());
  ge::TraceOwnerGuard guard(FE_MODULE_NAME, kStagePreCompile, graph.GetName());
  FE_TIMECOST_START(OptimizeSubgraphOfPrecompiledOp);
  // copy origin node info to om sub graph and recover node attrs info in om subgraph
  ModelBinaryCompilerPtr model_binary_compiler_ptr = nullptr;
  FE_MAKE_SHARED(model_binary_compiler_ptr = std::make_shared<ModelBinaryCompiler>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (model_binary_compiler_ptr->UpdateNodeInfoInOmSubGraph(graph, lookup) != SUCCESS) {
    REPORT_FE_ERROR("[OptimizeWholeGraph][ModelBinary] Failed to recover node information in OM sub-graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }

  FE_TIMECOST_END(OptimizeSubgraphOfPrecompiledOp, "FEGraphOptimizer::OptimizeSubgraphOfPrecompiledOp");
  return SUCCESS;
}
}  // namespace fe
