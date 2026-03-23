/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_model_om_saver.h"
#include <fstream>
#include <regex>
#include "mmpa/mmpa_api.h"
#include "graph/model.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "model_deploy_resource.h"
#include "endpoint.h"
#include "common/util.h"
#include "common/checker.h"
#include "proto/flow_model.pb.h"
#include "graph/utils/graph_utils.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
namespace ge {
namespace {
void ConvertModelQueueInfo(const ModelRelation::ModelEndpointInfo &model_endpoint_info,
                           flow_model::proto::ModelRelationDef_ModelEndpointInfo &proto_model_queue_info) {
  proto_model_queue_info.set_model_name(model_endpoint_info.model_name);
  for (const auto &input_endpoint_name : model_endpoint_info.input_endpoint_names) {
    proto_model_queue_info.add_input_endpoint_name(input_endpoint_name);
  }
  for (const auto &output_endpoint_name : model_endpoint_info.output_endpoint_names) {
    proto_model_queue_info.add_output_endpoint_name(output_endpoint_name);
  }
  for (const auto &external_input_queue_name : model_endpoint_info.external_input_queue_names) {
    proto_model_queue_info.add_external_input_queue_name(external_input_queue_name);
  }
  for (const auto &external_output_queue_name : model_endpoint_info.external_output_queue_names) {
    proto_model_queue_info.add_external_output_queue_name(external_output_queue_name);
  }
  for (const auto &invoke_model_key : model_endpoint_info.invoke_model_keys) {
    proto_model_queue_info.add_invoke_model_key(invoke_model_key);
  }
}

void ConvertModelRealtion(const ModelRelation &model_relation,
                          flow_model::proto::ModelRelationDef &model_relation_def) {
  for (const auto &endpoint : model_relation.endpoints) {
    auto *proto_endpoint = model_relation_def.add_endpoint();
    (void)endpoint.Serialize(proto_endpoint);
  }
  auto *proto_submodel_endpoint_info = model_relation_def.mutable_submodel_endpoint_info();
  for (const auto &submodel_queue_info : model_relation.submodel_endpoint_infos) {
    const auto &model_queue = submodel_queue_info.second;
    flow_model::proto::ModelRelationDef_ModelEndpointInfo proto_model_queue_info;
    ConvertModelQueueInfo(model_queue, proto_model_queue_info);
    (*proto_submodel_endpoint_info)[submodel_queue_info.first] = proto_model_queue_info;
  }

  auto *proto_invoked_model_queue_info = model_relation_def.mutable_invoked_model_queue_info();
  for (const auto &invoked_model_queue_info : model_relation.invoked_model_queue_infos) {
    const auto &invoked_model_queue = invoked_model_queue_info.second;
    flow_model::proto::ModelRelationDef_InvokedModelQueueInfo proto_invoked_model_queue;
    for (const auto &input_queue_name : invoked_model_queue.input_queue_names) {
      proto_invoked_model_queue.add_input_queue_name(input_queue_name);
    }
    for (const auto &output_queue_name : invoked_model_queue.output_queue_names) {
      proto_invoked_model_queue.add_output_queue_name(output_queue_name);
    }
    (*proto_invoked_model_queue_info)[invoked_model_queue_info.first] = proto_invoked_model_queue;
  }

  auto *proto_root_model_endpoint_info = model_relation_def.mutable_root_model_endpoint_info();
  ConvertModelQueueInfo(model_relation.root_model_endpoint_info, *proto_root_model_endpoint_info);
}

Status AddFlowModelCompileResource(const FlowModelPtr &flow_model, flow_model::proto::FlowModelDef &flow_model_def) {
  auto *const execution_runtime = ExecutionRuntime::GetInstance();
  auto *const compile_res_info = flow_model_def.mutable_compile_resource();
  auto *const running_res_list_proto = compile_res_info->mutable_dev_to_resource_list();
  const auto &submodels = flow_model->GetSubmodels();
  std::set<std::string> logic_dev_lists;
  for (const auto &submodel_iter : submodels) {
    const auto &submodel = submodel_iter.second;
    const auto &logic_device_id = submodel->GetLogicDeviceId();
    const auto &deploy_resource = submodel->GetDeployResource();
    GELOGI("Get logic device id: %s from submodel", logic_device_id.c_str());
    const auto vec_after_split = StringUtils::Split(logic_device_id, ',');
    if ((!logic_device_id.empty()) && (!vec_after_split.empty())) {
      (void)logic_dev_lists.insert(vec_after_split.cbegin(), vec_after_split.cend());
    }
    if ((vec_after_split.empty()) || (deploy_resource == nullptr)) {
      continue;
    }
    for (const auto &dev_id : vec_after_split) {
      for (const auto &res : deploy_resource->resource_list) {
        auto *const running_res_proto = (*running_res_list_proto)[dev_id].add_running_resource();
        running_res_proto->set_type(res.first);
        running_res_proto->set_value(res.second);
      }
    }
  }
  if (execution_runtime != nullptr) {
    const auto &host_res_type = execution_runtime->GetCompileHostResourceType();
    const auto &logic_dev_id_to_res_type = execution_runtime->GetCompileDeviceInfo();
    if (host_res_type.empty() && logic_dev_id_to_res_type.empty()) {
      GELOGI("Needn't to record resource info result of compile resource empty");
    } else if ((!host_res_type.empty()) && (!logic_dev_id_to_res_type.empty())) {
      compile_res_info->set_host_resource_type(host_res_type);
      auto *const proto_dev_to_type = compile_res_info->mutable_logic_device_id_to_resource_type();
      for (const auto &dev_to_type : logic_dev_id_to_res_type) {
        // In load balance mode: logic device id is empty. Record all compile resource
        if ((!logic_dev_lists.empty()) && (logic_dev_lists.count(dev_to_type.first) == 0UL)) {
          GELOGD("Logic device id %s is not assign to any submodel.", dev_to_type.first.c_str());
          continue;
        }
        (*proto_dev_to_type)[dev_to_type.first] = dev_to_type.second;
      }
      GELOGI("Save compile info : host resource type %s, device resource number %zu in offline model success.",
             host_res_type.c_str(), logic_dev_id_to_res_type.size());
    } else {
      GELOGW("Host resource type %s is empty or device resource number %zu is zero",
             host_res_type.c_str(), logic_dev_id_to_res_type.size());
    }
  }
  return SUCCESS;
}

Status SaveOmDataToFile(const std::shared_ptr<PneModel> &submodel,
                        flow_model::proto::SubmodelDef &submodel_def,
                        ModelBufferData &serialize_buff,
                        const std::string &split_om_data_base_dir) {
  GE_CHECK_NOTNULL(submodel);
  const auto root_graph = submodel->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  const std::string &normalize_name = submodel->GetNormalizedModelName();
  const std::string &graph_name = root_graph->GetName();
  const std::string file_name = normalize_name.empty() ? (graph_name + ".om") :
                               (normalize_name  + ".om");
  // file path is graph_name/file_name.om  base dir is ./cache_dir/graph_key
  submodel_def.set_om_data_file_path(graph_name + "/" + file_name);
  const std::string split_om_data_dir = split_om_data_base_dir + graph_name;
  GE_ASSERT_TRUE((CreateDir(split_om_data_dir) == 0),
                  "Create directory failed, path: %s.", split_om_data_dir.c_str());
  const std::string om_file_name = split_om_data_dir + "/" + file_name;
  // UDF model(not builtin) cp tar.gz to om ; UDF(builtin) or NPU CPU model save from memory
  if ((submodel->GetModelType() == PNE_ID_UDF) && (!submodel->GetIsBuiltinModel())) {
    const std::string release_pkg = submodel->GetSavedModelPath();
    if (mmAccess(release_pkg.c_str()) != EN_OK) {
      GELOGE(FAILED, "Can not find release pkg file by path:%s.", release_pkg.c_str());
      return FAILED;
    }
    if (release_pkg != om_file_name) {
      // for subgraph cache mode, release_pkg is equal to om file name
      std::regex dir_pattern(R"([A-Za-z0-9./+\-_]+)");
      std::smatch match_result;
      GE_CHK_BOOL_RET_STATUS(std::regex_match(om_file_name, match_result, dir_pattern), PARAM_INVALID,
                             "Invalid target om file path: %s", om_file_name.c_str());
      GELOGI("Copy release pkg: %s to cache file: %s.", release_pkg.c_str(), om_file_name.c_str());
      const std::string cmd = "cp " + release_pkg + " " +  om_file_name;
      GE_CHK_BOOL_RET_STATUS(system(cmd.c_str()) == 0, FAILED, "Failed to execute cmd[%s].", cmd.c_str());
    }
  } else {
    GELOGI("Write om data to file: %s. Set submodel def file name: %s", om_file_name.c_str(), file_name.c_str());
    GE_ASSERT_GRAPH_SUCCESS(SaveBinToFile(reinterpret_cast<char_t *>(serialize_buff.data.get()),
        serialize_buff.length, om_file_name), "Failed to save model data to file %s.", om_file_name.c_str());
  }
  return SUCCESS;
}
}  // namespace

Status FlowModelOmSaver::SaveToOm(const std::string &output_file, const std::string &split_om_data_base_dir) {
  GE_ASSERT_SUCCESS(AddModelDefPartition(), "[Add][ModelPartition] failed.");
  GE_ASSERT_SUCCESS(AddFlowModelPartition(), "[Add][FlowModelPartition] failed.");
  GE_ASSERT_SUCCESS(AddFlowSubModelPartitions(split_om_data_base_dir), "[Add][FlowSubModelPartition] failed.");
  GE_ASSERT_SUCCESS(UpdateModelHeader(), "[Update][Header] failed.");
  GE_ASSERT_SUCCESS(SaveFlowModelToFile(output_file), "[Save][FlowModelToBuffer] failed.");
  buffers_.clear();
  GELOGI("save to om success, output_file=%s.", output_file.c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::SaveToModelData(ModelBufferData &model_buff) {
  GE_ASSERT_SUCCESS(AddModelDefPartition(), "[Add][ModelPartition] failed.");
  GE_ASSERT_SUCCESS(AddFlowModelPartition(), "[Add][FlowModelPartition] failed.");
  GE_ASSERT_SUCCESS(AddFlowSubModelPartitions(), "[Add][FlowSubModelPartition] failed.");
  GE_ASSERT_SUCCESS(UpdateModelHeader(), "[Update][Header] failed.");
  GE_ASSERT_SUCCESS(SaveFlowModelToDataBuffer(model_buff), "[Save][FlowModelToBuffer] failed.");
  buffers_.clear();
  GELOGI("save to model data buffer success.");
  return SUCCESS;
}

Status FlowModelOmSaver::AddModelDefPartition() {
  const auto &root_graph = flow_model_->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  ComputeGraphPtr graph_for_save = MakeShared<ComputeGraph>(root_graph->GetName());
  GE_CHECK_NOTNULL(graph_for_save);
  const auto graph_filter = [](const Node &, const char_t *, const ComputeGraphPtr &) -> bool {
    return false;
  };
  (void)GraphUtils::CopyComputeGraph(root_graph, nullptr, graph_filter, nullptr, graph_for_save);
  graph_for_save = (graph_for_save == nullptr) ? root_graph : graph_for_save;
  FixNonStandardGraph(graph_for_save);
  ge::Model ge_model;
  ge_model.SetName(graph_for_save->GetName());
  ge_model.SetGraph(graph_for_save);
  ge::Buffer model_buffer;
  (void)ge_model.Save(model_buffer);
  GELOGD("MODEL_DEF size is %zu", model_buffer.GetSize());
  if (model_buffer.size() == 0UL) {
    GELOGE(FAILED, "save model def failed, as save size is 0.");
    return FAILED;
  }
  const auto ret = AddPartition(model_buffer, MODEL_DEF);
  GE_CHK_STATUS_RET(ret, "[Add][ModelDefPartition]Failed, partition size %zu", model_buffer.size());
  return SUCCESS;
}

Status FlowModelOmSaver::AddFlowModelPartition() {
  flow_model::proto::FlowModelDef flow_model_def;
  flow_model_def.set_model_name(flow_model_->GetModelName());
  // add model relation.
  const auto &model_relation = flow_model_->GetModelRelation();
  if (model_relation != nullptr) {
    auto *proto_relation = flow_model_def.mutable_relation();
    ConvertModelRealtion(*model_relation, *proto_relation);
  }

  const auto &submodels = flow_model_->GetSubmodels();
  for (const auto &submodel : submodels) {
    flow_model_def.add_submodel_name(submodel.second->GetModelName());
  }

  for (const auto &models_esched_priority : flow_model_->GetModelsEschedPriority()) {
    auto *const proto_models_esched_priority = flow_model_def.mutable_models_esched_priority();
    flow_model::proto::FlowModelDef_EschedPriority proto_esched_priority;
    auto *const proto_esched_priority_map = proto_esched_priority.mutable_esched_priority();
    for (const auto &esched_priority : models_esched_priority.second) {
      (*proto_esched_priority_map)[esched_priority.first] = esched_priority.second;
    }
    (*proto_models_esched_priority)[models_esched_priority.first] = proto_esched_priority;
  }

  GE_CHK_STATUS_RET(AddFlowModelCompileResource(flow_model_, flow_model_def), "[Add][CompileResource] to flow model failed.");
  GE_CHK_STATUS_RET(AddPartition(flow_model_def, FLOW_MODEL), "[Add][FlowModelDef]Failed, model=%s",
                    flow_model_->GetModelName().c_str());
  return SUCCESS;
}


Status FlowModelOmSaver::AddFlowSubModelPartitions(const std::string &split_om_data_base_dir) {
  const auto &submodels = flow_model_->GetSubmodels();
  for (const auto &submodel_iter : submodels) {
    const auto &submodel = submodel_iter.second;
    if (!submodel->GetSubmodels().empty()) {
      GELOGE(FAILED, "flow model is not flatten, sub model[%s] has [%zu] submodels", submodel->GetModelName().c_str(),
             submodel->GetSubmodels().size());
      return FAILED;
    }
    flow_model::proto::SubmodelDef submodel_def;
    submodel_def.set_model_name(submodel->GetModelName());
    submodel_def.set_model_type(submodel->GetModelType());
    ModelBufferData serialize_buff{};
    const auto ret = submodel->SerializeModel(serialize_buff);
    GE_CHK_STATUS_RET(ret, "[Serialize][Model]Failed, model name=%s, model_type=%s", submodel->GetModelName().c_str(),
                      submodel->GetModelType().c_str());
    submodel_def.set_is_builtin_udf(submodel->GetIsBuiltinModel());
    if (!split_om_data_base_dir.empty() && (!submodel->GetIsBuiltinModel())) {
      GE_CHK_STATUS_RET(SaveOmDataToFile(submodel, submodel_def, serialize_buff, split_om_data_base_dir),
                        "Save om data to file failed.");
    } else {
      submodel_def.set_om_data(serialize_buff.data.get(), serialize_buff.length);
      GELOGI("Save om data to buffer in memory.");
    }
    
    const auto logic_device_id = submodel->GetLogicDeviceId();
    if (!logic_device_id.empty()) {
      auto *deploy_info = submodel_def.mutable_deploy_info();
      deploy_info->set_logic_device_id(logic_device_id);
    }
    const auto redundant_logic_device_id = submodel->GetRedundantLogicDeviceId();
    auto *redundant_deploy_info = submodel_def.mutable_redundant_deploy_info();
    redundant_deploy_info->set_redundant_logic_device_id(redundant_logic_device_id);
    if (submodel->GetModelType() == PNE_ID_UDF) {
      auto *const udf_graph = submodel_def.mutable_graph();
      const ModelSerializeImp serialize_imp;
      if (!serialize_imp.SerializeGraph(submodel->GetRootGraph(), udf_graph, false)) {
        GELOGE(FAILED, "serialize udf graph failed, model name=%s", submodel->GetModelName().c_str());
        return FAILED;
      }
      const auto deploy_resource = submodel->GetDeployResource();
      if (deploy_resource != nullptr) {
        auto *const deploy_resource_proto = submodel_def.mutable_deploy_resource();
        deploy_resource_proto->set_resource_type(deploy_resource->resource_type);
        deploy_resource_proto->set_is_heavy_load(deploy_resource->is_heavy_load);
        // deploy resource other field is not support now
      }
    }
    GE_CHK_STATUS_RET(AddPartition(submodel_def, FLOW_SUBMODEL),
                      "[Add][FlowSubModelPartition]Failed, model=%s, model_type=%s", submodel->GetModelName().c_str(),
                      submodel->GetModelType().c_str());
    GELOGD("add flow submodel partition end, model=%s, model_type=%s", submodel->GetModelName().c_str(),
           submodel->GetModelType().c_str());
  }
  return SUCCESS;
}

Status FlowModelOmSaver::UpdateModelHeader() {
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper_.GetModelFileHeader();
  // just 1 model.
  model_header.model_num = 1U;
  model_header.modeltype = MODEL_TYPE_FLOW_MODEL;

  const auto &model_name = flow_model_->GetModelName();
  size_t name_len = model_name.length();
  name_len = (name_len > (MODEL_NAME_LENGTH - 1U)) ? (MODEL_NAME_LENGTH - 1U) : name_len;
  const auto err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, model_name.c_str(), name_len);
  if (err != EOK) {
    GELOGW("[Save][Model]Failed copy model name for model %s, err %d", model_name.c_str(), err);
  }
  GELOGD("Model name save:%s", model_name.c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::SaveFlowModelToFile(const std::string &output_file) {
  ModelBufferData model{};
  const auto ret = om_file_save_helper_.SaveModel(output_file.c_str(), model, true);
  GE_CHK_STATUS_RET(ret, "save model to file failed, filename:%s.", output_file.c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::SaveFlowModelToDataBuffer(ModelBufferData &model_buff) {
  const std::string output_file;
  const auto ret = om_file_save_helper_.SaveModel(output_file.c_str(), model_buff, false);
  GE_ASSERT_SUCCESS(ret, "save model to model buffer failed.");
  return SUCCESS;
}

Status FlowModelOmSaver::AddPartition(const google::protobuf::Message &partition_msg,
                                      ModelPartitionType partition_type) {
  Buffer buffer(partition_msg.ByteSizeLong());
  if (buffer.GetData() == nullptr) {
    GELOGE(FAILED, "alloc buffer failed, size=%zu, partition_type=%d.", partition_msg.ByteSizeLong(), partition_type);
    return FAILED;
  }
  if (!partition_msg.SerializePartialToArray(buffer.GetData(), static_cast<int32_t>(buffer.GetSize()))) {
    GELOGE(FAILED, "SerializePartialToArray failed, size=%zu, partition_type=%d", partition_msg.ByteSizeLong(),
           partition_type);
    return FAILED;
  }
  return AddPartition(buffer, partition_type);
}

Status FlowModelOmSaver::AddPartition(Buffer &buffer, ModelPartitionType partition_type) {
  ModelPartition partition;
  partition.data = buffer.data();
  partition.size = static_cast<uint64_t>(buffer.size());
  partition.type = partition_type;
  const auto ret = om_file_save_helper_.AddPartition(partition, 0UL);
  GE_CHK_STATUS_RET(ret, "[Add][Partition]Failed, partition size %zu, partition_type=%d", buffer.size(),
                    partition_type);
  buffers_.emplace_back(buffer);
  return SUCCESS;
}

void FlowModelOmSaver::FixNonStandardGraph(const ComputeGraphPtr &graph) {
  // remove invalid output nodes.
  const auto output_nodes = graph->GetOutputNodes();
  for (const auto &node : output_nodes) {
    if (node->GetOwnerComputeGraph() != graph) {
      (void)graph->RemoveOutputNode(node);
    }
  }
}
}  // namespace ge
