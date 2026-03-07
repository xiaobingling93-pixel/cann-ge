/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/inner_pp_loader.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "dflow/base/model/model_relation.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "dflow/base/model/flow_model_om_loader.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"

namespace ge {
namespace {
const std::string kModelPpFusionInputs = "INVOKED_MODEL_FUSION_INPUTS";
}
Status InnerPpLoader::LoadProcessPoint(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                                       const NodePtr &node) {
  const auto &pp_extend_attrs = process_point.pp_extend_attrs();
  const auto &find_ret = pp_extend_attrs.find(dflow::INNER_PP_CUSTOM_ATTR_INNER_TYPE);
  if (find_ret == pp_extend_attrs.cend()) {
    GELOGE(FAILED, "custom inner pp type attr[%s] does not exist.", dflow::INNER_PP_CUSTOM_ATTR_INNER_TYPE);
    return FAILED;
  }
  const auto &inner_type = find_ret->second;
  if (inner_type == dflow::INNER_PP_TYPE_MODEL_PP) {
    return LoadModelPp(process_point, data_flow_graph, node);
  } else {
    GELOGE(FAILED, "load inner pp failed, inner pp type[%s] is unknown, node=%s.", inner_type.c_str(),
           node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status InnerPpLoader::LoadModelPp(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                                  const NodePtr &node) {
  const auto &pp_extend_attrs = process_point.pp_extend_attrs();
  const auto &find_ret = pp_extend_attrs.find(dflow::INNER_PP_CUSTOM_ATTR_MODEL_PP_MODEL_PATH);
  if (find_ret == pp_extend_attrs.cend()) {
    GELOGE(FAILED, "cannot find extend attr[%s], node=%s, pp name=%s.", dflow::INNER_PP_CUSTOM_ATTR_MODEL_PP_MODEL_PATH,
           node->GetName().c_str(), process_point.name().c_str());
    return FAILED;
  }
  const std::string &model_path = find_ret->second;
  const std::string &pp_name = process_point.name();
  const std::string &compile_file = process_point.compile_cfg_file();
  std::function<Status()> load_task = [&data_flow_graph, model_path, pp_name, node, compile_file]() {
    FlowModelPtr flow_model = nullptr;
    GE_CHK_STATUS_RET(LoadModel(data_flow_graph, model_path, flow_model), "load model failed, model_path=%s.",
                      model_path.c_str());
    flow_model->SetModelName(pp_name);
    const auto &root_graph = flow_model->GetRootGraph();
    if (flow_model->GetModelRelation() == nullptr) {
      auto model_relation = MakeUnique<ModelRelation>();
      GE_CHECK_NOTNULL(model_relation);
      GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*root_graph, *model_relation),
                        "Failed to build model relation for graph[%s].", pp_name.c_str());
      flow_model->SetModelRelation(std::shared_ptr<ModelRelation>(model_relation.release()));
    }
    if (!compile_file.empty()) {
      CompileConfigJson::ModelPpConfig model_pp_cfg = {};
      GE_CHK_STATUS_RET(CompileConfigJson::ReadModelPpConfigFromJsonFile(compile_file, model_pp_cfg),
                        "Failed to read process point[%s] compile config.", compile_file.c_str());
      if (!model_pp_cfg.invoke_model_fusion_inputs.empty()) {
        (void)AttrUtils::SetStr(root_graph, kModelPpFusionInputs, model_pp_cfg.invoke_model_fusion_inputs);
      }
      GELOGI("Parse compile config file [%s] success. fusion inputs config [%s]", compile_file.c_str(),
             model_pp_cfg.invoke_model_fusion_inputs.c_str());
    }
    GE_CHK_STATUS_RET(data_flow_graph.AddLoadedModel(node->GetName(), pp_name, flow_model),
                      "Failed to add loaded model, pp_name:%s", pp_name.c_str());
    return SUCCESS;
  };
  GE_CHK_STATUS_RET(data_flow_graph.CommitPreprocessTask(pp_name, load_task),
                    "Failed to commit load model pp task[%s].", pp_name.c_str());
  return SUCCESS;
}

Status InnerPpLoader::LoadModel(const DataFlowGraph &data_flow_graph, const std::string &model_path,
                                FlowModelPtr &flow_model) {
  flow_model = nullptr;
  GE_CHK_STATUS_RET(FlowModelHelper::LoadToFlowModel(model_path, flow_model),
                    "load to flow model failed, model_path=%s", model_path.c_str());

  const auto &root_graph = data_flow_graph.GetRootGraph();
  uint64_t session_id = root_graph->GetSessionID();
  uint32_t graph_id = root_graph->GetGraphID();
  std::string file_constant_weight_dir;
  GE_ASSERT_SUCCESS(FileConstantUtils::GetExternalWeightDirFromOmPath(model_path, file_constant_weight_dir));
  GE_CHK_STATUS_RET(
      FlowModelOmLoader::RefreshModel(flow_model, file_constant_weight_dir, session_id, graph_id),
      "Failed to assign constant mem for model");
  // change to offline session graph id to avoid conflit with other model pp.
  const std::string offline_session_graph_id = "-1_" + std::to_string(graph_id);
  GE_CHK_STATUS_RET(FlowModelHelper::UpdateSessionGraphId(flow_model, offline_session_graph_id),
                    "Failed to update flow model session graph id, session_graph_id:%s",
                    offline_session_graph_id.c_str());
  return SUCCESS;
}
}  // namespace ge
