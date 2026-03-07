/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_HETEROGENEOUS_MODEL_EXECUTOR_H_
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_HETEROGENEOUS_MODEL_EXECUTOR_H_

#include <thread>
#include "common/thread_pool/thread_pool.h"
#include "common/blocking_queue.h"
#include "ge/ge_data_flow_api.h"
#include "dflow/base/deploy/exchange_service.h"
#include "graph/ge_tensor.h"
#include "dflow/base/deploy/model_deployer.h"
#include "graph/ge_local_context.h"
#include "heterogeneous_model_io_helper.h"
#include "inner_process_msg_forwarding.h"
#include "data_flow_exception_handler.h"
#include "data_flow_data_aligner.h"

namespace ge {
class HeterogeneousModelExecutor {
 public:
  /// @ingroup ge
  /// @brief Constructor
  /// @param [in]  root_model        Root model
  /// @param [in]  deploy_result     Model deployment info
  HeterogeneousModelExecutor(const FlowModelPtr &flow_model, const DeployResult &deploy_result);

  virtual ~HeterogeneousModelExecutor();

  /// @ingroup ge
  /// @brief Initialize executor
  /// @return SUCCESS success / others failure
  Status Initialize();

  /// @ingroup ge
  /// @brief Execute model
  /// @param [in]  inputs     inputs
  /// @param [out] outputs    outputs
  /// @return SUCCESS success / others failure
  Status Execute(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);

  /// @ingroup ge
  /// @brief Execute distributed model
  /// @param [in]  graph_id   graph_id
  /// @param [in]  device_to_inputs     device to inputs
  /// @param [out] device_to_outputs    device to outputs
  /// @return SUCCESS success / others failure
  Status Execute(uint32_t graph_id, const std::map<int32_t, std::vector<Tensor>> &device_to_inputs,
                 std::map<int32_t, std::vector<Tensor>> &device_to_outputs);

  /// @ingroup ge
  /// @brief Execute model async
  /// @param [in]  inputs     inputs
  /// @param [out] callback   callback function
  /// @return SUCCESS success / others failure
  Status ExecuteAsync(const std::vector<Tensor> &inputs, const RunAsyncCallback &callback);

  /// @ingroup ge
  /// @brief create model std::thread,
  /// @brief start to execute Model
  /// @return Status create model thread and execute result
  Status ModelRunStart();

  /// @ingroup ge
  /// @brief call API provided by data inputer and destroy model Thread
  /// @return Status Destroy result
  Status ModelRunStop();

  /// @ingroup ge
  /// @brief Get Id of the DeployedModel
  /// @param [in]  listener    listener
  uint32_t GetDeployedModelId() const;

  void SetModelId(const uint32_t model_id);

  Status FeedData(const std::vector<uint32_t> &indexes, const std::vector<GeTensor> &inputs, const DataFlowInfo &info,
                  int32_t timeout);

  Status FetchData(const std::vector<uint32_t> &indexes, std::vector<GeTensor> &outputs, DataFlowInfo &info,
                   int32_t timeout);

  Status FeedFlowMsg(const std::vector<uint32_t> &indexes, const std::vector<FlowMsgPtr> &inputs, int32_t timeout);

  Status FetchFlowMsg(const std::vector<uint32_t> &indexes, std::vector<FlowMsgPtr> &outputs, int32_t timeout);

  Status FeedRawData(const std::vector<RawData> &raw_data_list, uint32_t index,
                     const DataFlowInfo &info, int32_t timeout);

  FlowModelPtr GetFlowModel() const { return flow_model_; }

  static inline uint64_t Now() {
    static auto zero = std::chrono::system_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now() - zero).count();
    return uint64_t(us);
  }
 private:
  struct RunAsyncRequest {
    RunAsyncCallback callback;
    const void *shared_buffer = nullptr;
  };

  struct ModelIndices {
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    int32_t devicd_id;
  };

  struct QueueStatus {
    int32_t queue_depth;
    uint32_t model_uuid;
    int32_t device_id;
    int32_t device_type;
  };

  Status WrapSingleModel();
  Status ParseInputTensorInfo();
  Status ParseOutputTensorInfo();
  Status BuildInputTensorDescMapping(std::map<std::string, GeTensorDescPtr> &mapping);
  Status BuildOutputTensorDescMapping(std::map<std::string, GeTensorDescPtr> &mapping);
  Status SetTensorInfo(std::map<std::string, GeTensorDescPtr> &mapping,
                       const std::vector<std::string> &queue_names,
                       const bool is_input);
  static Status GetTimeoutFromOption(int32_t &timeout);
  Status EnqueueInputTensors(const std::vector<GeTensor> &inputs);
  Status EnqueueInputTensors(const std::vector<GeTensor> &inputs, const size_t replica_num);
  Status DequeueOutputTensors(std::vector<GeTensor> &outputs);
  Status DequeueControlOutputs(const size_t replica_num, const int32_t timeout);
  Status DoDequeue(GeTensor &output_tensor, std::shared_ptr<AlignedPtr> &aligned_ptr,
                   ExchangeService::ControlInfo &control_info, const size_t output_index);
  Status DoDequeue(FlowMsgBasePtr &flow_msg, ExchangeService::ControlInfo &control_info, size_t output_index);
  Status DoDequeueOnce(GeTensor &output_tensor, std::shared_ptr<AlignedPtr> &aligned_ptr,
                       ExchangeService::ControlInfo &control_info, const size_t output_index);
  Status DoDequeueOnce(FlowMsgBasePtr &flow_msg, ExchangeService::ControlInfo &control_info, size_t output_index);
  Status ValidateInputTensors(const vector<GeTensor> &inputs);
  void Run();
  Status StatusRun();
  Status SchedRun(uint32_t index);
  Status FindValidGroupEntry(uint32_t uuid, int32_t logic_queue_id, int32_t logic_group_id,
                             DeployPlan::DstGroupInfo **group_entry_ptr);
  void ProcAfterFindGroupEntry(DeployPlan::DstGroupInfo &group_info, int32_t group_entry_index);
  bool FindGroupEntryIndexInSingleInstance(DeployPlan::DstGroupInfo &group_info,
                                           std::pair<int32_t, std::string> &group_entry_index_and_name);
  bool FindGroupEntryIndexFromCache(DeployPlan::DstGroupInfo &group_info,
                                    std::pair<int32_t, std::string> &group_entry_index_and_name,
                                    uint64_t trans_id,
                                    uint32_t route_label);
  void UpdateQueueDefaultStatus(const DeployPlan::DstGroupInfo &group_info, int32_t device_id, int32_t device_type,
                                uint32_t i);
  void FindGroupEntryIndexBySchedule(DeployPlan::DstGroupInfo &group_info, int32_t device_id, int32_t device_type,
                                     std::pair<int32_t, std::string> &group_entry_index_and_name);
  void DeleteInvalidCache();
  Status DynamicSchedQueueInitialize(const bool is_dynamic_sched);
  void UpdateAbnormalInstanceList(RootModelId2SubmodelName &abnormal_submodel_instances_name);
  void UpdateAbnormalInstanceForTrimmingModel(const uint32_t root_model_id, const std::string &abnormal_name);
  void AbnormalStatusCallbackInit();
  template<typename T>
  Status GetQueueInfoByDequeueMbuf(const int32_t device_id, const uint32_t queue_id, T &info,
                                   const int32_t time_out = 0) const;
  Status DynamicSchedProc(const domi::FlowgwRequest &flowgw_request,
                          int32_t queue_infos_index,
                          domi::FlowgwResponse &flowgw_response);
  Status FlowgwResponseEnqueue(int32_t device_id, int32_t datagw_input_index, domi::FlowgwResponse &flowgw_response);
  void DynamicSchedDurationStart();
  void DynamicSchedDurationEnd();
  void UpdateQueueStatusInfo(const domi::SubmodelStatus &submodel_status, int32_t queue_status_index);
  void DynamicSchedInfoClear();
  Status FeedEmptyEosData(ExchangeService::ControlInfo &control_info) const;

  Status BuildFusionInputTensorMapping();

  Status FillFusionInput(const std::vector<GeTensor> &fusion_inputs, void *const buffer, const size_t size) const;

  Status EnqueueFusionInputs(const std::map<DeployQueueAttr, std::vector<GeTensor>> &fusion_inputs,
                             ExchangeService::ControlInfo &control_info) const;

  static Status WaitInputTensorsUnbuild(const void *const inputs_buffer);

  static Status GetIndicesToTensorDesc(const ComputeGraphPtr &compute_graph,
                                       std::map<int64_t, GeTensorDescPtr> &indices_to_tensor_descs);
  bool IsRedeployStart(const uint32_t abnormal_status_operation_type) const;
  bool IsDynamicSched(const uint32_t abnormal_status_operation_type) const;
  bool IsRedeployFailed(const uint32_t abnormal_status_operation_type) const;
  bool IsModelInstanceAbnormal(const std::string &submodel_instance_name);
  void DynamicSchedClear();
  void ClearFeedData();
  void ClearFetchData();
  Status GetRedeployStatus();
  void ModelIndexInfoUpdate();
  void ModelIndexGroupInfoUpdate(DeployPlan::DstGroupInfo &group_info);

  Status AlignFetchData(const std::vector<uint32_t> &fetch_indexes, std::vector<GeTensor> &outputs, DataFlowInfo &info,
                        const int32_t timeout);

  Status GetOrCreateDataAligner(const std::vector<uint32_t> &fetch_indexes,
                                std::shared_ptr<DataFlowDataAligner> &data_aligner);

  FlowModelPtr flow_model_;
  std::shared_ptr<const ModelRelation> model_relation_;
  uint32_t deployed_model_id_;
  uint32_t model_id_ = 0U;
  ExchangeService *exchange_service_ = nullptr;
  std::vector<DeployQueueAttr> input_queue_attrs_;
  std::vector<DeployQueueAttr> control_input_queue_attrs_;
  std::vector<DeployQueueAttr> output_queue_attrs_;
  std::vector<DeployQueueAttr> control_output_queue_attrs_;
  std::function<Status(void)> dev_abnormal_callback_;
  HeterogeneousModelIoHelper io_helper_;
  DataFlowExceptionHandler exception_handler_;
  size_t replica_num_ = 1U;
  std::vector<GeTensorDescPtr> input_tensor_desc_;
  std::vector<GeTensorDescPtr> output_tensor_desc_;
  std::vector<bool> input_is_no_tiling_;
  std::vector<bool> is_fusion_input_;
  std::vector<bool> output_is_no_tiling_;
  std::vector<int64_t> input_tensor_sizes_;
  std::vector<int64_t> output_tensor_sizes_;
  std::vector<int64_t> input_tensor_raw_sizes_;
  std::vector<int64_t> output_tensor_raw_sizes_;
  // {key: input_queue_attr, value: { input tensor index }
  std::map<DeployQueueAttr, std::vector<size_t>> fusion_input_queue_attrs_;
  std::mutex mu_;
  std::atomic_bool run_flag_{false};
  GEThreadLocalContext run_context_;
  std::string input_model_name_;
  std::map<std::string, ModelIndices> model_indices_;
  std::mutex output_mutex_;
  std::map<int32_t, DeployQueueAttr> datagw_rqt_to_rsp_; // datagw input indice to sched app output queueid
  std::map<int32_t, std::pair<QueueStatus, uint64_t>> queue_status_info_; // key:logic_queue_id;data:QueueStatus, 决策时间点
  // cache dynamic route trans id and route label, key=trans id, value route labels
  std::map<uint64_t, std::set<uint32_t>> cached_trans_ids_;
  // routelabel_cache_info_: transid routelabel, group size, group_entry_index
  std::map<std::pair<uint64_t, uint32_t>, std::map<int32_t, std::pair<int32_t, std::string>>> routelabel_cache_info_;

  std::mutex cache_mu_;
  thread_local static uint64_t duration_total_;
  thread_local static uint64_t cnt_total_;
  thread_local static uint64_t duration_max_;
  thread_local static uint64_t duration_size_;
  thread_local static uint64_t call_;
  std::vector<DeployQueueAttr> status_input_queue_attrs_; // 状态接受队列
  std::vector<DeployQueueAttr> sched_input_queue_attrs_; // app datagw response 队列
  std::vector<DeployQueueAttr> sched_output_queue_attrs_; // app datagw request 队列
  std::map<int32_t, int32_t> sched_input_cnt_; // <app datagw response 队列index, 调度结果enqueue次数统计>
  DeployPlan::DynamicSchedIndex model_index_info_;
  std::map<int32_t, int32_t> datagw_request_bindings_; // app output 到datagw input 逻辑queue id 映射
  bool is_dynamic_sched_ = false;
  bool is_exception_catch_ = false;
  bool contains_n_mapping_node_ = false;
  std::thread status_run_thread_;
  std::vector<std::thread> sched_threads_;
  std::mutex queue_status_mu_;
  std::mutex abnormal_name_mu_;
  RootModelId2SubmodelName abnormal_submodel_instances_name_;
  std::atomic<uint32_t> deploy_state_ {0U};
  DeployPlan::AbnormalStatusCallbackInfo *abnormal_status_callback_info_ = nullptr;
  // 内层集合是同一个device上裁边场景具有连接关系的模型集合
  std::vector<std::unordered_set<std::string>> model_trimming_edges_model_instances_;
  BlockingQueue<domi::SubmodelStatus> status_messages_queue_;

  InnerProcessMsgForwarding process_forwarding_;

  // model io align attr is used for output
  InputAlignAttrs align_attrs_;
  // key is fetch indexes.
  std::map<std::vector<uint32_t>, std::shared_ptr<DataFlowDataAligner>> data_aligner_map_;
  std::map<uint32_t, std::shared_ptr<DataFlowDataAligner>> input_to_data_aligner_map_;
  std::mutex data_aligner_mu_;
};
}  // namespace ge
#endif  // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_HETEROGENEOUS_MODEL_EXECUTOR_H_
