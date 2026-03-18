/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_graph_optimizer.h"

#include "platform/platform_info.h"
#include "common/util/trace_manager/trace_manager.h"
#include "config/config_file.h"
#include "engine/base_engine.h"
#include "error_code/error_code.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "graph_optimizer_utils.h"
#include "optimizer.h"
#include "runtime/dev.h"
#include "util/log.h"
#include "util/util.h"
#include "util/constant.h"
#include "pass/concat_from_sequence_pass.h"

using namespace ge;
using namespace std;

namespace {
// need delete when update libgraph.so in blue zone
const std::string AICPU_ATTR_NAME_OP_TILING_INLINE_ENGINE = "_op_tiling_inline_engine";
const std::string AICPU_ATTR_NAME_OP_EXPORT_SHAPE_ENGINE = "_op_export_shape_engine";
const std::string AICPU_ATTR_NAME_UNKNOWN_SHAPE_TYPE = "_aicpu_unknown_shape_type";
const std::string AICPU_NODE_NAME_NETOUTPUT = "NetOutput";
const std::string AICPU_NODE_NAME_DATA = "Data";
std::mutex g_cust_mutex;
const std::string kOpsParallel = "aicpu_ops_parallel";
}

namespace aicpu {
ge::Status AicpuGraphOptimizer::Initialize(const map<string, string> &options,
                                           [[maybe_unused]] ge::OptimizeUtility *const optimize_utility) {
  // initial optimizers
  auto iter = options.find(ge::SOC_VERSION);
  AICPU_IF_BOOL_EXEC(iter == options.end(),
      AICPU_REPORT_INNER_ERR_MSG(
          "cannot find [%s] in param of optimizer initialize function.",
          ge::SOC_VERSION.c_str());
      return INPUT_PARAM_VALID)
  soc_version_ = iter->second;
  std::string auto_cast_mode;
  if (ConfigFile::GetInstance().GetValue(kAutoCastMode, auto_cast_mode)) {
    auto_cast_mode_ = kAutoCastModeOff;
    if (StringToNum(auto_cast_mode, auto_cast_mode_).state != ge::SUCCESS) {
      AICPUE_LOGW("Tran auto_cast_mode[%s] to integer failed. default value is [%d].",
                  auto_cast_mode.c_str(), auto_cast_mode_);
    }
  } else {
    AICPUE_LOGW("Get Value[AutoCastMode] failed");
  }
  AICPU_CHECK_RES(Finalize())
  AICPU_CHECK_RES(GetOpsInfo(all_op_info_));
  std::string optimizers_str;
  std::string optimizers_config = Stringcat(engine_name_, "GraphOptimizer");
  AICPU_IF_BOOL_EXEC(
      !ConfigFile::GetInstance().GetValue(optimizers_config, optimizers_str),
      AICPU_REPORT_INNER_ERR_MSG("[%s] not exist.", optimizers_config.c_str());
      return LOAD_OPTIMIZER_CONFIG_FAILED)
  vector<string> optimizers;
  ConfigFile::GetInstance().SplitValue(optimizers_str, optimizers);
  for (auto optimizer : optimizers) {
    FACTORY_GRAPH_OPTIMIZER::FactoryType optimizers_ptr =
        FACTORY_GRAPH_OPTIMIZER::Produce(optimizer);
    AICPU_IF_BOOL_EXEC(optimizers_ptr == nullptr,
        AICPU_REPORT_INNER_ERR_MSG("[%s] instantiate failed", optimizer.c_str());
        return GRAPH_OPTIMIZER_INSTANCE_FAILED)
    optimizer_map_[optimizer] = optimizers_ptr;
    AICPU_CHECK_RES(optimizers_ptr->Initialize());
    if (optimizer == "AICPUOptimizer") {
      optimizers_ptr->SetCustUserInfos(cust_op_infos_);
      AICPUE_LOGD("cust_op_infos_ size is %zu", cust_op_infos_.size());
    }
  }

  GetGetOpsParallelOoLevel();

  if (is_op_parallel_oo_enable_) {
    // get ops parallel rule info
    GetOpsParallelRule();
  }

  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::Finalize() {
  optimizer_map_.clear();
  ops_parallel_rule_infos_.clear();
  cust_op_infos_.clear();
  all_op_info_.clear();
  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::GetAttributes(
    GraphOptimizerAttribute &attrs) const {
  attrs.engineName = engine_name_;
  return SUCCESS;
}

void AicpuGraphOptimizer::GetGetOpsParallelOoLevel() {                                 
  std::string opt_value;
  auto status = GetThreadLocalContext().GetOo().GetValue(kOpsParallel, opt_value);
  AICPUE_LOGI("Get option[%s], opt_value[%s], status[%u].",
               kOpsParallel.c_str(), opt_value.c_str(), status);
  // 注册时未指定对应级别value或者获取到与注册不相等Value
  if (opt_value == "false") {
    AICPUE_LOGI("ops parallel disable in current level.");
    is_op_parallel_oo_enable_ = false;
  } else {
    is_op_parallel_oo_enable_ = true;
  }
  return;
}

ge::Status AicpuGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph) {
  ConcatFromSequencePass cocat_from_sequence_pass;
  cocat_from_sequence_pass.Run(graph);

  ge::TraceManager::GetInstance().SetTraceOwner(kModuleName, kTraceOriginalOptimizer, graph.GetName());
  if (IsEmptyGraph(graph)) {
    return SUCCESS;
  }

  auto ret = AddAicpuSupportedNodeAttr(graph);
  if (ret == SUCCESS) {
    ret = AddDtStringUnknownAttr(graph);
  }

  for (auto optimizer : optimizer_map_) {
    AICPU_CHECK_RES(optimizer.second->OptimizeOriginalGraph(graph, all_op_info_));
  }
  ge::TraceManager::GetInstance().ClearTraceOwner();
  return ret;
}

ge::Status AicpuGraphOptimizer::AddTilingModeAttr(const ge::OpDescPtr &op_desc_ptr, const std::string &op_type) const {
  vector<string> engines;
  (void)AttrUtils::GetListStr(op_desc_ptr, AICPU_ATTR_NAME_OP_TILING_INLINE_ENGINE, engines);
  engines.emplace_back(engine_name_);
  bool ret = AttrUtils::SetListStr(op_desc_ptr, AICPU_ATTR_NAME_OP_TILING_INLINE_ENGINE, engines);
  AICPU_IF_BOOL_EXEC(!(ret),
                      AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetListStr failed to set attr[%s], op[%s]",
                                              AICPU_ATTR_NAME_OP_TILING_INLINE_ENGINE.c_str(),
                                              op_desc_ptr->GetName().c_str());
                      return FAILED)
  engines.clear();
  (void)AttrUtils::GetListStr(op_desc_ptr, AICPU_ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engines);
  engines.emplace_back(engine_name_);
  ret = AttrUtils::SetListStr(op_desc_ptr, AICPU_ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, engines);
  AICPU_IF_BOOL_EXEC(!(ret),
                      AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetListStr failed to set attr[%s], op[%s]",
                                              AICPU_ATTR_NAME_OP_EXPORT_SHAPE_ENGINE.c_str(),
                                              op_desc_ptr->GetName().c_str());
                      return FAILED)
  AICPUE_LOGD("Add no tiling engine[%s] for op type[%s], attrs[%s, %s].", engine_name_.c_str(), op_type.c_str(),
              AICPU_ATTR_NAME_OP_TILING_INLINE_ENGINE.c_str(), AICPU_ATTR_NAME_OP_EXPORT_SHAPE_ENGINE.c_str());
  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::AddAicpuSupportedNodeAttr(ge::ComputeGraph &graph) {
  for (const NodePtr &curr_node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(curr_node)
    OpDescPtr curr_op_desc_ptr = curr_node->GetOpDesc();
    AICPU_CHECK_NOTNULL(curr_op_desc_ptr)
    std::string op_type;
    AICPU_CHECK_RES(GetOriginalType(curr_op_desc_ptr, op_type));

    auto const iter = all_op_info_.find(op_type);
    if (iter != all_op_info_.end()) {
      auto op_info = iter->second;
      bool no_tiling = op_info.noTiling;
      AICPUE_LOGD("Op type[%s], no tiling[%d], unknown_shape_type[%d].", op_type.c_str(), no_tiling, op_info.shapeType);
      if (no_tiling) {
        AICPU_CHECK_RES(AddTilingModeAttr(curr_op_desc_ptr, op_type));
      }
      if (op_info.shapeType == ge::DEPEND_SHAPE_RANGE || op_info.shapeType == ge::DEPEND_COMPUTE) {
        (void)ge::AttrUtils::SetInt(curr_op_desc_ptr, AICPU_ATTR_NAME_UNKNOWN_SHAPE_TYPE, op_info.shapeType);
      }
      // Set attr[_is_blocking_op] for asyn ops, except Assert.
      // Because op Assert's flagAsyn is false in aicpu kernel lib, but is true in tf kernel lib.
      if (op_info.flagAsync && op_type.compare("Assert") != 0) {
        AICPUE_LOGD("Set aicpu blocking op:%s, attribute(is_blocking_op):true", curr_op_desc_ptr->GetName().c_str());
        (void)ge::AttrUtils::SetBool(curr_op_desc_ptr, ge::ATTR_NAME_IS_BLOCKING_OP, true);
        SetAicpuAsyncOpTimeout(curr_op_desc_ptr, op_type);
      }
    }
  }
  return SUCCESS;
}

bool AicpuGraphOptimizer::NodeExistStringOutput(const ge::NodePtr &node) const {
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  for (const ge::GeTensorDesc &output_tensor_desc : op_desc_ptr->GetAllOutputsDesc()) {
    if (output_tensor_desc.GetDataType() == DT_STRING) {
      return true;
    }
  }
  return false;
}

bool AicpuGraphOptimizer::IsSetMaxSize(const ge::NodePtr &node) const {
  const auto &anchor = node->GetOutDataAnchor(0);
  if ((anchor != nullptr) && (anchor->GetFirstPeerAnchor() != nullptr) &&
      (anchor->GetFirstPeerAnchor()->GetOwnerNode() != nullptr)) {
    const auto &node_desc = anchor->GetFirstPeerAnchor()->GetOwnerNode()->GetOpDesc();
    if (AttrUtils::HasAttr(node_desc, "_op_max_size")) {
      return true;
    }
  }
  return false;
}

ge::Status AicpuGraphOptimizer::AddDtStringUnknownAttr(ge::ComputeGraph &graph) const {
  for (const NodePtr &curr_node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(curr_node)
    // set input dtstring node atrr
    if (curr_node->GetType() == AICPU_NODE_NAME_DATA) {
      OpDescPtr op_desc_ptr = curr_node->GetOpDesc();
      AICPU_CHECK_NOTNULL(op_desc_ptr)
      auto input_tensor_desc = op_desc_ptr->GetInputDesc(0);
      if (input_tensor_desc.GetDataType() == DT_STRING && !IsSetMaxSize(curr_node)) {
        (void)ge::AttrUtils::SetBool(op_desc_ptr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
        (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrNameInputOutputDtString, true);
        AICPUE_LOGI("Op [%s], data_type is DT_STRING, set attr[%s] and attr[%s] success.",
                    op_desc_ptr->GetName().c_str(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE.c_str(),
                    kAttrNameInputOutputDtString.c_str());
        SetInputDtStringNodeAttr(curr_node);
      }
    } else if (curr_node->GetType() == AICPU_NODE_NAME_NETOUTPUT) { // set output dtstring node atrr
      OpDescPtr op_desc_ptr = curr_node->GetOpDesc();
      AICPU_CHECK_NOTNULL(op_desc_ptr)
      for (auto &tensor_desc : op_desc_ptr->GetAllInputsDesc()) {
        if (tensor_desc.GetDataType() == DT_STRING) {
          (void)ge::AttrUtils::SetBool(op_desc_ptr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
          (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrNameInputOutputDtString, true);
          AICPUE_LOGI("Op [%s], data_type is DT_STRING, set attr[%s] and attr[%s] success.",
                      op_desc_ptr->GetName().c_str(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE.c_str(),
                      kAttrNameInputOutputDtString.c_str());
          SetOutputDtStringNodeAttr(curr_node);
          break;
        }
      }
    }
  }
  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::SetInputDtStringNodeAttr(const ge::NodePtr &node) const {
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    AICPU_CHECK_NOTNULL(out_data_anchor)
    for (auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      const auto &peer_node = peer_in_anchor->GetOwnerNode();
      OpDescPtr op_desc_ptr = peer_node->GetOpDesc();
      GeTensorDesc input_tensor_desc = op_desc_ptr->GetInputDesc(peer_in_anchor->GetIdx());
      if (input_tensor_desc.GetDataType() == DT_STRING) {
        (void)ge::AttrUtils::SetBool(op_desc_ptr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
        if (NodeExistStringOutput(peer_node)) {
          (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrNameInputOutputDtString, true);
          SetInputDtStringNodeAttr(peer_node);
        }
      }
    }
  }
  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::SetOutputDtStringNodeAttr(const ge::NodePtr &node) const {
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    AICPU_CHECK_NOTNULL(in_data_anchor)
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &peer_node = peer_out_anchor->GetOwnerNode();
    OpDescPtr op_desc_ptr = peer_node->GetOpDesc();
    GeTensorDesc output_tensor_desc = op_desc_ptr->GetOutputDesc(peer_out_anchor->GetIdx());
    if (output_tensor_desc.GetDataType() == DT_STRING && peer_node->GetType() != kNodeTypeConst &&
        peer_node->GetType() != kNodeTypeConstant) {
      (void)ge::AttrUtils::SetBool(op_desc_ptr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
      (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrNameInputOutputDtString, true);
      SetOutputDtStringNodeAttr(peer_node);
    }
  }
  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::OptimizeOriginalGraphJudgeInsert(
    ComputeGraph &graph) {
  if (IsEmptyGraph(graph)) {
    return SUCCESS;
  }

  for (auto optimizer : optimizer_map_) {
    AICPU_CHECK_RES(
        optimizer.second->OptimizeOriginalGraphJudgeInsert(graph, all_op_info_))
  }
  return SUCCESS;
}

ge::Status AicpuGraphOptimizer::OptimizeFusedGraph(ComputeGraph &graph) {
  std::string graph_name = graph.GetName();
  ge::TraceManager::GetInstance().SetTraceOwner(kModuleName, kTraceFusedOptimizer, graph_name);
  AICPUE_LOGI("begin to optimizer graph[%s]", graph_name.c_str());
  if (IsEmptyGraph(graph)) {
    ge::TraceManager::GetInstance().ClearTraceOwner();
    return SUCCESS;
  }
  // vertify placehold and end node make sure format equals client format
  AICPU_CHECK_RES(GraphOptimizerUtils::VerifyPldAndEndNode(graph))

  std::string suffix = "Before_Aicpu_Optimized";
  GraphOptimizerUtils::DumpGraph(graph, suffix);

  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos opt_infos;
  AICPU_IF_BOOL_EXEC(
      fe::PlatformInfoManager::Instance().GetPlatformInfos(soc_version_,
          platform_infos, opt_infos) != ge::GRAPH_SUCCESS,
      AICPU_REPORT_INNER_ERR_MSG("Call fe::PlatformInfoManager::GetPlatformInfos "
          "function failed. soc version[%s]", soc_version_.c_str());
      ge::TraceManager::GetInstance().ClearTraceOwner();
      return ge::FAILED)
  // cache coherence need to be guaranteed
  std::string aicpu_cache_enable;
  AICPU_IF_BOOL_EXEC(
      !platform_infos.GetPlatformRes("CPUCache", "AICPUSyncBySW", aicpu_cache_enable),
      AICPU_REPORT_INNER_ERR_MSG("Call fe::PlatFormInfos::GetPlatformRes failed to"
          " get aicpu cache synchronous status");
      ge::TraceManager::GetInstance().ClearTraceOwner();
      return ge::FAILED)
  AICPUE_LOGI("aicpu cache enable flag[%s]", aicpu_cache_enable.c_str());
  if (aicpu_cache_enable.find("1") != string::npos) {
    AICPU_CHECK_RES_WITH_LOG(CacheGraph::GenerateNoCacheGraph(graph),
        "Call GenerateCacheGraph function failed, graph[%s].", graph_name.c_str())
    suffix = "After_Insert_Cache_op";
    GraphOptimizerUtils::DumpGraph(graph, suffix);
  }

  if (auto_cast_mode_ == kAutoCastModeOn) {
    AICPU_CHECK_RES_WITH_LOG(AutoCastGraph::GenerateAutoCastGraph(graph, all_op_info_), "AutoCast failed");
    suffix = "After_Auto_Cast";
    GraphOptimizerUtils::DumpGraph(graph, suffix);
  }

  if (is_op_parallel_oo_enable_) {
    SetStreamLabelForOpsParallel(graph);
  }

  for (auto optimizer : optimizer_map_) {
    AICPU_CHECK_RES(optimizer.second->OptimizeFusedGraph(graph, all_op_info_))
  }

  suffix = "After_Aicpu_Optimized";
  GraphOptimizerUtils::DumpGraph(graph, suffix);

  AICPUE_LOGI("optimizer fused graph[%s] success.", graph_name.c_str());
  ge::TraceManager::GetInstance().ClearTraceOwner();
  return ge::SUCCESS;
}

bool AicpuGraphOptimizer::IsEmptyGraph(const ComputeGraph &graph) const {
  std::string graph_name = graph.GetName();
  if (graph.GetDirectNodesSize() == 0) {
    AICPUE_LOGW("No ge node exists in graph[%s].", graph_name.c_str());
    return true;
  }
  return false;
}

ge::Status AicpuGraphOptimizer::GetOpsInfo(
    map<string, OpFullInfo> &all_op_info) {
  const std::lock_guard<std::mutex> lock(g_cust_mutex);
  FACTORY_ENGINE::FactoryType engine_ptr = FACTORY_ENGINE::Produce(engine_name_);
  AICPU_CHECK_NOTNULL(engine_ptr)
  AicpuOpsKernelInfoStorePtr aicpu_ops_kernel_info_store_ptr =
      engine_ptr->GetAicpuOpsKernelInfoStore();
  AICPU_CHECK_NOTNULL(aicpu_ops_kernel_info_store_ptr)
  aicpu_ops_kernel_info_store_ptr->GetAllOpsFullKernelInfo(all_op_info);
  if (engine_name_ == "DNN_VM_AICPU_ASCEND") {
    aicpu_ops_kernel_info_store_ptr->GetCustUserNameInfo(cust_op_infos_);
  }

  return ge::SUCCESS;
}

bool AicpuGraphOptimizer::ReadOpsParallelRuleFromJsonFile()
{
  std::string realConfigFilePath = GetSoPath(reinterpret_cast<void *>(&AicpuGraphOptimizer::Initialize));
  AICPUE_LOGD("realConfigFilePath is %s.", realConfigFilePath.c_str());
  std::string opsParallelRuleFilePath = realConfigFilePath + kAiCpuOpsParallelRuleFileRelativePath;
  AICPUE_LOGD("opsParallelRuleFilePath is %s.", opsParallelRuleFilePath.c_str());
  return OpsParallelRuleJsonFile::Instance().ParseUnderPath(opsParallelRuleFilePath,
                                                            ops_parallel_rule_json_file_).state == ge::SUCCESS;
}

ge::Status AicpuGraphOptimizer::GetOpsParallelRule()
{
  if (!ReadOpsParallelRuleFromJsonFile()) {
    AICPU_REPORT_INNER_ERR_MSG("Call Aicpu ops_parallel_rule_file_path to read ops paralle info from json file failed.");
    return LOAD_CONFIG_JSON_FILE_FAILED;
  }

  AICPU_IF_BOOL_EXEC(
    (ops_parallel_rule_json_file_.find(kOpsParallelRule) == ops_parallel_rule_json_file_.end()),
    AICPUE_LOGW("Json file does not have ops parallel rule infos"); return SUCCESS)

  try {
    RuleInfoDesc ops_parallel_info = ops_parallel_rule_json_file_;
    AICPUE_LOGI("Read json file, support parallel ops size is:%lu.",
        ops_parallel_info.rule_info.ops_list.size());
    return FillOpsParallelRuleInfos(ops_parallel_info);
  } catch (const nlohmann::json::exception &e) {
    AICPU_REPORT_INNER_ERR_MSG("Parse ops parallel rule json file[%s] failed, %s.",
        ops_parallel_rule_json_file_.dump().c_str(), e.what());
      return LOAD_CONFIG_JSON_FILE_FAILED;
  }
}

ge::Status AicpuGraphOptimizer::FillOpsParallelRuleInfos(RuleInfoDesc &ops_parallel_info)
{
  if (ops_parallel_info.rule_name.size() == 0) {
    AICPUE_LOGW("Json file does not have ops parallel rule info.");
    return ge::SUCCESS;
  }

  for (const auto &op : ops_parallel_info.rule_info.ops_list) {
    ops_parallel_rule_infos_.push_back(op);
    AICPUE_LOGD("Insert a op[%s] success.", op.c_str());
  }
  AICPUE_LOGD("Read json file, rule name[%s].", ops_parallel_info.rule_name.c_str());
  return ge::SUCCESS;
}

ge::Status AicpuGraphOptimizer::GetOpsParallelInfo(std::unordered_set<string> &ops_parallel_info) const
{
  if (ops_parallel_rule_infos_.size() == 0) {
    AICPUE_LOGW("ops parallel rule infos size is 0.");
    return ge::FAILED;
  }
  for (const auto &op : ops_parallel_rule_infos_) {
    ops_parallel_info.insert(op);
    AICPUE_LOGD("Insert a op[%s] in set success.", op.c_str());
    if (ops_parallel_info.size() >= kMaxOpsParallelNum) {
      AICPUE_LOGW("ops parallel rule support the max num is %zu.", ops_parallel_info.size());
      break;
    }
  }
  return ge::SUCCESS;
}

ge::Status AicpuGraphOptimizer::SetStreamLabel(const ge::NodePtr &node, const std::string &label) const
{
  AICPU_CHECK_NOTNULL(node);
  const OpDescPtr tmp_desc = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(tmp_desc);

  if (!AttrUtils::SetStr(tmp_desc, "_stream_label", label)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:fail for op:%s(%s)",
        node->GetName().c_str(), node->GetType().c_str());
    AICPUE_LOGE("[Set][Attr] fail for op:%s(%s)", node->GetName().c_str(),
        node->GetType().c_str());
    return ge::FAILED;
  }

  return ge::SUCCESS;
}

ge::Status AicpuGraphOptimizer::SetStreamLabelForOpsParallel(ge::ComputeGraph &graph) const
{
  unordered_set<string> ops_parallel_info;
  AICPU_CHECK_RES(GetOpsParallelInfo(ops_parallel_info));
  for (ge::NodePtr &node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(node);
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    AICPU_CHECK_NOTNULL(op_desc_ptr);
    std::string op_type = op_desc_ptr->GetType();

    auto iter = ops_parallel_info.find(op_type);
    if (iter == ops_parallel_info.end()) {
      AICPUE_LOGD("Current op type [%s]. Don't exist in ops parallel rule list.", op_type.c_str());
      continue;
    }

    const std::string lable = node->GetName();
    auto status = SetStreamLabel(node, lable);
    if (status != ge::SUCCESS) {
      AICPUE_LOGW("[Set][Streamlabel] %s to op:%s(%s) failed.", lable.c_str(), node->GetName().c_str(),
          op_type.c_str());
      return ge::FAILED;
    }
    AICPUE_LOGD("[Set][Streamlabel] %s to op:%s(%s) success.", lable.c_str(), node->GetName().c_str(),
        op_type.c_str());
  }
  return ge::SUCCESS;
}

void AicpuGraphOptimizer::SetAicpuAsyncOpTimeout(const ge::OpDescPtr &op_desc_ptr, std::string &op_type) {
  static unordered_map<std::string, int64_t> async_ops_timeout = {{"OutfeedEnqueueOpV2", 0}};
  auto op_iter = async_ops_timeout.find(op_type);
  if (op_iter != async_ops_timeout.end()) {
    (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_BLOCKING_OP_TIMEOUT, op_iter->second);
    AICPUE_LOGI("Set op:%s timeout:%d", op_type.c_str(), op_iter->second);
  }
}

REG_OPTION(kOpsParallel)
    .LEVELS(ge::OoLevel::kO0)
    .DEFAULT_VALUES({{ge::OoLevel::kO0, "false"}, {ge::OoLevel::kO1, "false"},
                     {ge::OoLevel::kO2, "true"}, {ge::OoLevel::kO3, "true"}});

}  // namespace aicpu
