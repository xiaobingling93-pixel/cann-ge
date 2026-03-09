/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_model_om_loader.h"
#include <regex>
#include "framework/common/helper/model_helper.h"
#include "common/util/mem_utils.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/thread_pool/thread_pool.h"
#include "common/math/math_util.h"
#include "model_deploy_resource.h"
#include "model_relation.h"
#include "common/helper/model_parser_base.h"
#include "endpoint.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/model.h"
#include "serialized_model.h"
#include "proto/flow_model.pb.h"
#include "debug/ge_attr_define.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "common/model/ge_root_model.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"

namespace ge {
namespace {
constexpr size_t kFlowModelPartitionsModeDefIdx = 0UL;
constexpr size_t kFlowModelPartitionsFlowModelIdx = 1UL;
constexpr size_t kFlowModelPartitionsFlowSubModelStartIdx = 2UL;

void ConvertModelQueueInfo(const flow_model::proto::ModelRelationDef_ModelEndpointInfo &proto_model_endpoint_info,
                           ModelRelation::ModelEndpointInfo &model_endpoint_info) {
  model_endpoint_info.model_name = proto_model_endpoint_info.model_name();
  model_endpoint_info.input_endpoint_names.assign(proto_model_endpoint_info.input_endpoint_name().cbegin(),
                                                  proto_model_endpoint_info.input_endpoint_name().cend());
  model_endpoint_info.output_endpoint_names.assign(proto_model_endpoint_info.output_endpoint_name().cbegin(),
                                                   proto_model_endpoint_info.output_endpoint_name().cend());

  model_endpoint_info.external_input_queue_names.assign(proto_model_endpoint_info.external_input_queue_name().cbegin(),
                                                        proto_model_endpoint_info.external_input_queue_name().cend());

  model_endpoint_info.external_output_queue_names.assign(
      proto_model_endpoint_info.external_output_queue_name().cbegin(),
      proto_model_endpoint_info.external_output_queue_name().cend());

  model_endpoint_info.invoke_model_keys.assign(proto_model_endpoint_info.invoke_model_key().cbegin(),
                                               proto_model_endpoint_info.invoke_model_key().cend());
}

void ConvertModelRealtion(const flow_model::proto::ModelRelationDef &model_relation_def,
                          ModelRelation &model_relation) {
  for (const auto &proto_endpoint : model_relation_def.endpoint()) {
    Endpoint endpoint(proto_endpoint.name(), static_cast<EndpointType>(proto_endpoint.endpoint_type()));
    (void)endpoint.Deserialize(proto_endpoint);
    model_relation.endpoints.emplace_back(endpoint);
  }

  for (const auto &proto_submodel_queue_info : model_relation_def.submodel_endpoint_info()) {
    ModelRelation::ModelEndpointInfo model_endpoint_info;
    ConvertModelQueueInfo(proto_submodel_queue_info.second, model_endpoint_info);
    model_relation.submodel_endpoint_infos[proto_submodel_queue_info.first] = model_endpoint_info;
  }

  for (const auto &proto_invoked_model_queue_iter : model_relation_def.invoked_model_queue_info()) {
    const auto &proto_invoked_model_queue = proto_invoked_model_queue_iter.second;
    ModelRelation::InvokedModelQueueInfo invoked_model_queue;
    invoked_model_queue.input_queue_names.assign(proto_invoked_model_queue.input_queue_name().cbegin(),
                                                 proto_invoked_model_queue.input_queue_name().cend());
    invoked_model_queue.output_queue_names.assign(proto_invoked_model_queue.output_queue_name().cbegin(),
                                                  proto_invoked_model_queue.output_queue_name().cend());

    model_relation.invoked_model_queue_infos[proto_invoked_model_queue_iter.first] = invoked_model_queue;
  }

  const auto &proto_root_model_queue_info = model_relation_def.root_model_endpoint_info();
  ConvertModelQueueInfo(proto_root_model_queue_info, model_relation.root_model_endpoint_info);
}

Status ParseModeldata(const flow_model::proto::SubmodelDef &flow_submodel_def, const ge::ModelData &model_data,
                        PneModelPtr &pne_model) {
  ComputeGraphPtr root_graph;
  auto ret = FlowModelOmLoader::TransModelDataToComputeGraph(model_data, root_graph);
  GE_CHK_STATUS_RET(ret, "Failed to trans model data to compute graph");
  GE_CHECK_NOTNULL(root_graph, "[ParseModelData] load root graph is null");
  GraphModelPtr graph_model_ptr = MakeShared<ge::GraphModel>(root_graph);
  GE_ASSERT_NOTNULL(graph_model_ptr);
  GE_CHK_STATUS_RET(graph_model_ptr->Init(model_data), "Failed to init graph model.");
  pne_model = graph_model_ptr;

  if (flow_submodel_def.has_deploy_info()) {
    const auto &deploy_info = flow_submodel_def.deploy_info();
    GE_CHK_STATUS_RET(pne_model->SetLogicDeviceId(deploy_info.logic_device_id()),
                      "set ge root model logic device id failed, model name=%s, logic_device_id=%s",
                      flow_submodel_def.model_name().c_str(), deploy_info.logic_device_id().c_str());
  }
  if (flow_submodel_def.has_redundant_deploy_info()) {
    const auto &redundant_deploy_info = flow_submodel_def.redundant_deploy_info();
    GE_CHK_STATUS_RET(pne_model->SetRedundantLogicDeviceId(redundant_deploy_info.redundant_logic_device_id()),
                      "set ge root model redundant logic device id failed, model name=%s, " \
                      "redundant_logic_device_id=%s", flow_submodel_def.model_name().c_str(),
                      redundant_deploy_info.redundant_logic_device_id().c_str());
  }

  GELOGD("load ge root model success, model name=%s.", flow_submodel_def.model_name().c_str());
  return SUCCESS;
}

Status LoadModeldata(const flow_model::proto::SubmodelDef &flow_submodel_def,
                       const std::string &split_om_data_base_dir, PneModelPtr &pne_model) {
  ge::ModelData model;
  const auto &om_data_file_name = flow_submodel_def.om_data_file_path();
  if (!om_data_file_name.empty()) {
    GE_ASSERT_TRUE(!split_om_data_base_dir.empty(), "Split om data base can not be empty while data file path exist.");
    const auto submodel_file_path = split_om_data_base_dir + om_data_file_name;
    GELOGI("Load submodel data by file %s.", submodel_file_path.c_str());
    GE_CHK_STATUS_RET(ModelParserBase::LoadFromFile(submodel_file_path.c_str(), 0, model),
        "Load model from file[%s] failed.", submodel_file_path.c_str());
    GE_MAKE_GUARD(model_guard, [&model]() {
    if (model.model_data != nullptr) {
        delete[] static_cast<char_t *>(model.model_data);
        model.model_data = nullptr;
      }
    });
    GE_CHK_STATUS_RET(ParseModeldata(flow_submodel_def, model, pne_model), "Failed to parse ge root model");
    pne_model->SetSavedModelPath(submodel_file_path);
    return SUCCESS;
  } else {
    const auto &om_data = flow_submodel_def.om_data();
    model.model_len = om_data.size();
    model.model_data = const_cast<char_t *>(om_data.c_str());
    return ParseModeldata(flow_submodel_def, model, pne_model);
  }
  return SUCCESS;
}

std::string GetUdfModelNameByFileName(const std::string &om_file) {
  if (om_file.empty()) {
    return "";
  }
  std::string dir_path;
  std::string file_name;
  SplitFilePath(om_file, dir_path, file_name);
  const auto pos = file_name.find_first_of(".");
  if (pos == std::string::npos) {
    return "";
  }
  return file_name.substr(0, pos);
}

Status LoadSerializedModel(flow_model::proto::SubmodelDef &flow_submodel_def, const std::string &split_om_data_base_dir,
                           PneModelPtr &pne_model) {
  ModelSerializeImp serialize_imp;
  ComputeGraphPtr graph;
  if (!serialize_imp.UnserializeGraph(graph, *(flow_submodel_def.mutable_graph()))) {
    GELOGE(FAILED, "UnserializeGraph failed, model_name=%s, model_type=%s.", flow_submodel_def.model_name().c_str(),
           flow_submodel_def.model_type().c_str());
    return FAILED;
  }
  const auto serialized_model = MakeShared<SerializedModel>(graph);
  GE_CHECK_NOTNULL(serialized_model, ", make SerializedModel failed");

  ge::ModelData model;
  ModelBufferData model_buff;
  if (flow_submodel_def.is_builtin_udf()) {
    // builtin udf
    const auto &om_data = flow_submodel_def.om_data();
    serialized_model->SetIsBuiltinModel(true);
    std::shared_ptr<uint8_t> data_buf(new (std::nothrow) uint8_t[om_data.size()], std::default_delete<uint8_t[]>());
    GE_CHECK_NOTNULL(data_buf, ", make data buf failed, size=%zu", om_data.size());
    if (memcpy_s(data_buf.get(), om_data.size(), om_data.c_str(), om_data.size()) != EOK) {
      GELOGE(FAILED, "copy data failed, size=%zu.", om_data.size());
      return FAILED;
    }
    model_buff.data = data_buf;
    model_buff.length = om_data.size();
    GE_CHK_STATUS_RET(serialized_model->UnSerializeModel(model_buff), "UnSerializeModel failed, size=%zu",
                      om_data.size());
    GELOGI("Load builtin model success.");
  }
  else {
    const auto &om_data_file_name = flow_submodel_def.om_data_file_path();
    GE_ASSERT_TRUE(!om_data_file_name.empty(), "Missing om_data_file_path or is_builtin_udf in cache. "
        "Please generate new cache base on current version.");
    auto submodel_file_path = split_om_data_base_dir + om_data_file_name;
    GE_ASSERT_TRUE(FlowModelOmLoader::CheckFilePathValid(split_om_data_base_dir, submodel_file_path),
        "Submodel file path[%s] is not in base dir[%s]", submodel_file_path.c_str(), split_om_data_base_dir.c_str());
    // user define function :deployder will find saved path and set to request directly
    serialized_model->SetSavedModelPath(submodel_file_path);
    std::string normalize_model_name = GetUdfModelNameByFileName(submodel_file_path);
    GE_ASSERT_TRUE(!normalize_model_name.empty(), "Get normalized name faield by path[%s]",
                   submodel_file_path.c_str());
    serialized_model->SetNormalizedModelName(normalize_model_name);
    GELOGI("set udf model cache file path: %s, normalize name: %s.",
           submodel_file_path.c_str(), normalize_model_name.c_str());
  }

  if (flow_submodel_def.has_deploy_resource()) {
    const auto &deploy_resource_proto = flow_submodel_def.deploy_resource();
    const auto deploy_resource = MakeShared<ModelDeployResource>();
    GE_CHECK_NOTNULL(deploy_resource);
    deploy_resource->resource_type = deploy_resource_proto.resource_type();
    deploy_resource->is_heavy_load = deploy_resource_proto.is_heavy_load();
    // deploy resource other field is not support now.
    serialized_model->SetDeployResource(deploy_resource);
  }
  if (flow_submodel_def.has_deploy_info()) {
    const auto &deploy_info = flow_submodel_def.deploy_info();
    (void)serialized_model->SetLogicDeviceId(deploy_info.logic_device_id());
  }
  if (flow_submodel_def.has_redundant_deploy_info()) {
    const auto &redundant_deploy_info = flow_submodel_def.redundant_deploy_info();
    (void)serialized_model->SetRedundantLogicDeviceId(redundant_deploy_info.redundant_logic_device_id());
  }
  pne_model = serialized_model;
  GELOGD("load serialized model success, model name=%s.", flow_submodel_def.model_name().c_str());
  return SUCCESS;
}
}  // namespace

Status FlowModelOmLoader::LoadToFlowModelDesc(const ge::ModelData &model_data, const FlowModelPtr &flow_model) {
  OmFileLoadHelper om_file_load_helper;
  GE_CHK_STATUS_RET(om_file_load_helper.Init(model_data), "Om file load helper init failed.");
  const auto &model_partitions = om_file_load_helper.GetModelPartitions(0);
  GE_CHK_STATUS_RET(CheckModelPartitions(model_partitions), "Check model partitions failed.");
  std::vector<string> submodel_names;
  GE_CHK_STATUS_RET(LoadFlowModelPartition(model_partitions[kFlowModelPartitionsFlowModelIdx],
                                           flow_model, submodel_names),
                    "Load flow model partition failed.");
  return SUCCESS;
}

bool FlowModelOmLoader::CheckFilePathValid(const std::string &base_dir, const std::string &check_dir) {
  const auto real_check_dir = RealPath(check_dir.c_str());
  return real_check_dir.find(base_dir) == 0UL;
}

Status FlowModelOmLoader::TransModelDataToComputeGraph(const ge::ModelData &model_data, ge::ComputeGraphPtr &root_graph) {
  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.LoadRootModel(model_data), "[Load][RootModel] failed.");
  root_graph = model_helper.GetGeRootModel()->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  return SUCCESS;
}

Status FlowModelOmLoader::LoadToFlowModel(const ge::ModelData &model_data, FlowModelPtr &flow_model,
                                          const std::string &split_om_data_path) {
  OmFileLoadHelper om_file_load_helper;
  auto ret = om_file_load_helper.Init(model_data);
  GE_CHK_STATUS_RET(ret, "om file load helper init failed.");
  const auto &model_partitions = om_file_load_helper.GetModelPartitions(0);
  ret = CheckModelPartitions(model_partitions);
  GE_CHK_STATUS_RET(ret, "check model partitions failed.");
  // load Root Graph
  const auto root_graph = LoadRootGraph(model_partitions[kFlowModelPartitionsModeDefIdx]);
  GE_CHECK_NOTNULL(root_graph, ", load root graph is null");
  const auto tmp_flow_model = MakeShared<FlowModel>(root_graph);
  GE_CHECK_NOTNULL(tmp_flow_model, ", load root graph is null");
  std::vector<string> submodel_names;
  ret = LoadFlowModelPartition(model_partitions[kFlowModelPartitionsFlowModelIdx], tmp_flow_model, submodel_names);
  GE_CHK_STATUS_RET(ret, "load flow model partition failed.");
  std::map<std::string, PneModelPtr> flow_submodels;
  ret = LoadFlowSubmodelPartition(model_partitions, split_om_data_path, flow_submodels);
  GE_CHK_STATUS_RET(ret, "load flow submodel partition failed.");
  for (const auto &submodel_name : submodel_names) {
    const auto find_ret = flow_submodels.find(submodel_name);
    if (find_ret == flow_submodels.cend()) {
      GELOGE(FAILED, "flow model with submodel name=%s, but not found in submodel partition", submodel_name.c_str());
      return FAILED;
    }
    const auto &submodel = find_ret->second;
    ret = tmp_flow_model->AddSubModel(submodel, submodel->GetModelType());
    GE_CHK_STATUS_RET(ret, "add sub model failed, model_name=%s, model_type=%s.", submodel->GetModelName().c_str(),
                      submodel->GetModelType().c_str());
  }
  flow_model = tmp_flow_model;
  GELOGI("load to flow model success, model name=%s.", flow_model->GetModelName().c_str());
  return SUCCESS;
}

Status FlowModelOmLoader::CheckModelPartitions(const std::vector<ModelPartition> &model_partitions) {
  if (model_partitions.size() < kFlowModelPartitionsFlowSubModelStartIdx) {
    GELOGE(FAILED, "flow model partitions must has 2 partitions[MODEL_DEF, FLOW_MODEL], but size=%zu.",
           model_partitions.size());
    return FAILED;
  }
  // the 0th is model def
  const auto &model_def_partition = model_partitions[kFlowModelPartitionsModeDefIdx];
  if (model_def_partition.type != MODEL_DEF) {
    GELOGE(FAILED, "flow model [0]th partition type must be MODEL_DEF[%d], but %d.", MODEL_DEF,
           model_def_partition.type);
    return FAILED;
  }

  // the 1th partion is flow model.
  const auto &flow_model_partition = model_partitions[kFlowModelPartitionsFlowModelIdx];
  if (flow_model_partition.type != FLOW_MODEL) {
    GELOGE(FAILED, "flow model [1]th partition type must be FLOW_MODEL[%d], but %d.", FLOW_MODEL,
           flow_model_partition.type);
    return FAILED;
  }
  for (size_t idx = kFlowModelPartitionsFlowSubModelStartIdx; idx < model_partitions.size(); ++idx) {
    const auto &flow_submodel_partition = model_partitions[idx];
    if (flow_submodel_partition.type != FLOW_SUBMODEL) {
      GELOGE(FAILED, "flow model [%zu]th partition type must be FLOW_SUBMODEL[%d], but %d.", idx, FLOW_SUBMODEL,
             flow_submodel_partition.type);
      return FAILED;
    }
  }
  return SUCCESS;
}

ComputeGraphPtr FlowModelOmLoader::LoadRootGraph(const ModelPartition &model_def_partition) {
  Model model;
  const auto status = Model::Load(model_def_partition.data, model_def_partition.size, model);
  if (status != GRAPH_SUCCESS) {
    GELOGE(status, "load model def failed, size=%lu.", model_def_partition.size);
    return nullptr;
  }
  return model.GetGraph();
}

Status FlowModelOmLoader::LoadFlowModelPartition(const ModelPartition &flow_model_partition,
                                                 const FlowModelPtr &flow_model, std::vector<string> &submodel_names) {
  flow_model::proto::FlowModelDef flow_model_def;
  if (!flow_model_def.ParseFromArray(flow_model_partition.data, static_cast<int32_t>(flow_model_partition.size))) {
    GELOGE(FAILED, "parse flow model partition def failed.");
    return FAILED;
  }
  flow_model->SetModelName(flow_model_def.model_name());
  submodel_names.assign(flow_model_def.submodel_name().cbegin(), flow_model_def.submodel_name().cend());

  if (flow_model_def.has_relation()) {
    const auto model_relation = MakeShared<ModelRelation>();
    GE_CHECK_NOTNULL(model_relation, ", make shared for model relation failed, model_name %s",
                     flow_model_def.model_name().c_str());
    ConvertModelRealtion(flow_model_def.relation(), *model_relation);
    flow_model->SetModelRelation(model_relation);
  }

  const auto &proto_models_esched_priority = flow_model_def.models_esched_priority();
  if (!proto_models_esched_priority.empty()) {
    std::map<std::string, std::map<std::string, int32_t>> models_esched_priority;
    for (const auto &proto_models_esched_priority_pair : proto_models_esched_priority) {
      const auto &proto_esched_priority = proto_models_esched_priority_pair.second.esched_priority();
      std::map<std::string, int32_t> esched_priority(proto_esched_priority.cbegin(), proto_esched_priority.cend());
      models_esched_priority[proto_models_esched_priority_pair.first] = std::move(esched_priority);
    }
    flow_model->SetModelsEschedPriority(models_esched_priority);
  }
  const auto compile_resource = MakeShared<ModelCompileResource>();
  GE_CHECK_NOTNULL(compile_resource);
  const auto &proto_compile_resource = flow_model_def.compile_resource();
  compile_resource->host_resource_type = proto_compile_resource.host_resource_type();
  for (const auto &dev_to_type : proto_compile_resource.logic_device_id_to_resource_type()) {
    compile_resource->logic_dev_id_to_res_type[dev_to_type.first] = dev_to_type.second;
  }

  const auto &proto_dev_to_res_list = proto_compile_resource.dev_to_resource_list();
  if (!proto_dev_to_res_list.empty()) {
    for (const auto &dev_to_res_list : proto_dev_to_res_list) {
      const auto &running_res_list = dev_to_res_list.second.running_resource();
      if (running_res_list.empty()) {
        continue;
      }
      auto &compile_res_list = compile_resource->dev_to_resource_list[dev_to_res_list.first];
      for (const auto &running_res : running_res_list) {
        compile_res_list.emplace_back(std::make_pair(running_res.type(), running_res.value()));
      }
    }
  }

  if (compile_resource->IsEmpty()) {
    // old version
    GELOGI("All compile resource in om model are empty.");
  } else {
    flow_model->SetCompileResource(compile_resource);
    GELOGI("Set valid compile resource info success. Host resource type %s, device number :%zu",
           compile_resource->host_resource_type.c_str(), compile_resource->logic_dev_id_to_res_type.size());
  }
  return SUCCESS;
}

Status FlowModelOmLoader::LoadFlowSubmodelPartition(const std::vector<ModelPartition> &model_partitions,
                                                    const std::string &split_om_data_base_dir,
                                                    std::map<std::string, PneModelPtr> &flow_submodels) {
  std::vector<PneModelPtr> submodels(model_partitions.size());
  std::vector<std::future<Status>> task_futures;
  constexpr uint32_t default_thread_num = 16;
  uint32_t thread_num = model_partitions.size() > default_thread_num ? default_thread_num : model_partitions.size();
  ThreadPool thread_pool("ge_dpl_omld", thread_num, false);
  for (size_t idx = kFlowModelPartitionsFlowSubModelStartIdx; idx < model_partitions.size(); ++idx) {
    const auto &flow_submodel_partition = model_partitions[idx];
    auto &submodel = submodels[idx];
    auto task = [&split_om_data_base_dir, &flow_submodel_partition, &submodel]() -> Status {
      flow_model::proto::SubmodelDef flow_submodel_def;
      if (!flow_submodel_def.ParseFromArray(flow_submodel_partition.data,
                                            static_cast<int32_t>(flow_submodel_partition.size))) {
        GELOGE(FAILED, "parse flow submodel partition def failed, size=%lu, type=%d.", flow_submodel_partition.size,
               flow_submodel_partition.type);
        return FAILED;
      }
      if (flow_submodel_def.model_type() == PNE_ID_UDF) {
        GE_CHK_STATUS_RET(LoadSerializedModel(flow_submodel_def, split_om_data_base_dir, submodel),
                          "LoadSerializedModel failed, model_name=%s, model_type=%s",
                          flow_submodel_def.model_name().c_str(), flow_submodel_def.model_type().c_str());
      } else {
        GE_CHK_STATUS_RET(LoadModeldata(flow_submodel_def, split_om_data_base_dir, submodel),
                          "LoadModeldata failed, model_name=%s, model_type=%s",
                          flow_submodel_def.model_name().c_str(), flow_submodel_def.model_type().c_str());
      }
      GE_CHECK_NOTNULL(submodel, ", load flow submodel failed, model_name=%s, model_type=%s.",
                       flow_submodel_def.model_name().c_str(), flow_submodel_def.model_type().c_str());

      submodel->SetModelName(flow_submodel_def.model_name());
      submodel->SetModelType(flow_submodel_def.model_type());
      GELOGI("load flow submodel success, submodel name=%s.", flow_submodel_def.model_name().c_str());
      return SUCCESS;
    };
    auto f = thread_pool.commit(task);
    GE_CHK_BOOL_RET_STATUS(f.valid(), FAILED, "Failed to load flow submodel, idx=%zu.", idx);
    task_futures.emplace_back(std::move(f));
  }
  Status result = SUCCESS;
  for (size_t idx = 0; idx < task_futures.size(); ++idx) {
    const auto ret = task_futures[idx].get();
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to get submodel future result, idx=%zu", idx);
      result = ret;
      continue;
    }
    auto &submodel = submodels[idx + kFlowModelPartitionsFlowSubModelStartIdx];
    flow_submodels[submodel->GetModelName()] = submodel;
  }
  return result;
}

Status FlowModelOmLoader::RefreshModel(const FlowModelPtr &flow_model, const std::string &model_path,
                                               const uint64_t session_id, const uint32_t graph_id) {
  std::set<ComputeGraph *> refreshed_graphs;
  for (const auto &submodel_iter : flow_model->GetSubmodels()) {
    const auto &model_name = submodel_iter.first;
    const auto &submodel = submodel_iter.second;
    GE_CHECK_NOTNULL(submodel, ", submodel is null, submodel_name=%s", model_name.c_str());
    // udf model has no file constant now
    if (submodel->GetModelType() == PNE_ID_UDF) {
      continue;
    }
    const auto &submodel_graph = submodel->GetRootGraph();
    GE_CHECK_NOTNULL(submodel_graph, ", submodel root graph is null, submodel_name=%s", model_name.c_str());

    GE_CHK_STATUS_RET(FileConstantUtils::SetExternalPath(submodel_graph, model_path), "Failed to set external path:%s.",
                      model_path.c_str());
    submodel_graph->SetSessionID(session_id);
    submodel_graph->SetGraphID(graph_id);
  }
  return SUCCESS;
}
}  // namespace ge