/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_EXECUTOR_CONTEXT_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_EXECUTOR_CONTEXT_H_

#include <vector>
#include <map>
#include <memory>
#include "ge/ge_api_error_codes.h"
#include "common/model/ge_model.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/model.h"
#include "framework/common/types.h"
#include "framework/common/helper/om_file_helper.h"
#include "executor/dynamic_model_executor.h"
#include "executor/proxy_dynamic_model_executor.h"
#include "proto/deployer.pb.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "external/ge/ge_ir_build.h"
#include "acl/acl.h"

namespace ge {
class ExecutorContext {
 public:
  ExecutorContext() = default;
  virtual ~ExecutorContext() = default;
  static ExecutorContext &LocalContext();

  class ModelHandle {
   public:
    struct LoadParam {
      uint32_t replica_num = 1U;
      uint32_t replica_idx = 0U;
      std::vector<QueueAttrs> input_queues;
      std::vector<QueueAttrs> output_queues;
      std::vector<int32_t> input_fusion_offsets;
      QueueAttrs status_output_queue;
      uint32_t model_uuid = 0U;
      bool is_dynamic_sched = false;
      bool need_report_status = false;
      bool is_head = false;
      InputAlignAttrs input_align_attrs{};
    };

    ModelHandle() = default;
    virtual ~ModelHandle();
    GE_DELETE_ASSIGN_AND_COPY(ModelHandle);

    virtual Status ParseModel(const std::string &model_path);
    virtual Status LoadModel(const LoadParam &param);
    Status UnloadModel();
    void SetExecuteTimes(int32_t execute_times);
    void SetEschedPriority(int32_t esched_process_priority, int32_t esched_event_priority);
    void SetIsDynamicProxyControlled(const bool is_dynamic_proxy_controlled);
    void SetScope(const std::string &scope);
    const std::string &GetScope() const;
    bool IsInvokedNN() const;
    void SetEnableExceptionCatch(bool enable_exception_catch);
    bool IsEnableExceptionCatch() const;
    Status GetModelData(ModelData &model_data);
    void SetModelData(const ModelData &model_data);
    Status GetRootGraph(ComputeGraphPtr &root_graph);
    void SetRootGraph(const ComputeGraphPtr &root_graph);
    virtual Status GetModelRuntimeIdOrHandle(std::vector<uint32_t> &davinci_model_runtime_ids,
                                             std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles);
    virtual Status ClearModel(const int32_t clear_type);
    virtual Status ExceptionNotify(uint32_t type, uint64_t trans_id);
   protected:
    virtual Status DoLoadModel(const ModelData &model_data,
                               const ComputeGraphPtr &root_graph,
                               const LoadParam &params);
    virtual Status DoLoadModelWithQ(const ModelData &model_data,
                                    const ComputeGraphPtr &root_graph,
                                    const LoadParam &params);

    virtual Status DoUnloadModel(uint32_t model_id);
    virtual std::unique_ptr<DynamicModelExecutor> CreateDynamicModelExecutor(bool is_host);
    virtual std::unique_ptr<ProxyDynamicModelExecutor> CreateProxyDynamicModelExecutor();

   private:
    static Status CheckAicpuAlignTask(const InputAlignAttrs &input_align_attrs);
    uint32_t inner_model_id_ = UINT32_MAX;
    std::unique_ptr<DynamicModelExecutor> dynamic_model_executor_;
    bool loaded_ = false;
    int32_t esched_process_priority_ = -1;  // -1 is user unset eshced priority
    int32_t esched_event_priority_ = -1;
    ModelData model_data_;
    bool model_data_from_cache_ = false;
    ComputeGraphPtr root_graph_;
    int32_t execute_times_ = -1; // execute times
    bool is_dynamic_proxy_controlled_ = false;  // control dynamic model execution by proxy process
    bool is_invoked_nn_ = false;
    bool enable_exception_catch_ = false;
    std::string scope_;  // data flow exception scope
    aclmdlConfigHandle *handle_ = nullptr;
  };

  Status Initialize() const;

  void Finalize() const;

  void SetBaseDir(const std::string &base_dir);

  virtual Status GetModel(uint32_t root_model_id, std::map<uint32_t, std::unique_ptr<ModelHandle>> *&submodel_map);

  Status SyncSharedVarManager(const deployer::ExecutorRequest &request) const;

  Status ParseModel(const deployer::ExecutorRequest_LoadModelRequest &request);

  static Status AttachQueues(const deployer::ExecutorRequest_LoadModelRequest &request);

  Status LoadModel(const deployer::ExecutorRequest_LoadModelRequest &request);

  PneModelPtr GetLocalModel(uint32_t root_model_id, uint32_t model_id);
  void AddLocalModel(uint32_t root_model_id, uint32_t model_id, const PneModelPtr &model);
  void RemoveLocalModel(uint32_t root_model_id);

  static Status SetOpTimeout();

  static Status SetDeviceSatMode();

  static void UpdateOptions(const deployer::Options &options);

  static Status UpdateProfInfo(const deployer::ExecutorRequest &request);

 protected:
  virtual std::unique_ptr<std::istream> CreateInputStream(const std::string &path) const;
  virtual ModelHandle *GetOrCreateModelHandle(uint32_t root_model_id, uint32_t model_id);

 private:
  Status ParseModel(uint32_t root_model_id, uint32_t model_id, const std::string &model_path);
  Status ParseModelEschedPriority(const deployer::ExecutorRequest_LoadModelRequest &request, ModelHandle &handle) const;
  static Status ParseInputAlignAttrs(const deployer::ExecutorRequest_LoadModelRequest &request,
                                     InputAlignAttrs &input_align_attrs);
  static void UpdateGraphOptions(const std::string &key, const std::string &value);

  // root_model_id, model_id, model_handle
  std::mutex mu_;
  std::map<uint32_t, std::map<uint32_t, std::unique_ptr<ModelHandle>>> model_handles_;
  std::map<uint32_t, std::map<uint32_t, PneModelPtr>> models_;
  std::string base_dir_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_EXECUTOR_CONTEXT_H_
