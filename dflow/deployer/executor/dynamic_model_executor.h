/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_DYNAMIC_MODEL_EXECUTOR_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_DYNAMIC_MODEL_EXECUTOR_H_

#include <thread>
#include <vector>
#include "graph/compute_graph.h"
#include "ge_common/ge_api_error_codes.h"
#include "common/blocking_queue.h"
#include "framework/common/runtime_tensor_desc.h"
#include "executor/cpu_sched_model.h"
#include "executor/cpu_id_resource_manager.h"
#include "common/ge_common/ge_types.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
// for rtMbufPtr_t
#include "runtime/rt.h"

namespace ge {
class DynamicModelExecutor {
 public:  
  explicit DynamicModelExecutor(bool is_host);
  virtual ~DynamicModelExecutor() noexcept;
  virtual Status Initialize();
  virtual void Finalize();
  virtual Status LoadModel(const ModelData &model_data,
                           const ComputeGraphPtr &root_graph,
                           const ModelQueueParam &model_queue_param);

  virtual Status ExecuteAsync(const std::function<void(Status, void *, void *)> &callback,
                              void *req_mbuf = nullptr, void *resp_mbuf = nullptr);
  Status ExecuteInternal();
  virtual Status ExecuteDirectly();
  virtual void UnloadModel();
  void SetModelEschedPriority(int32_t esched_process_priority, int32_t esched_event_priority);
  void SetModelExecuteTimes(int32_t execute_times);
  virtual Status ClearModel(const int32_t clear_type);
  virtual Status ExceptionNotify(uint32_t type, uint64_t trans_id);
  Status CheckLocalAicpuSupportExceptionNotify() const;
  static Status GenerateLoadConfig(const ModelData &model_data, const std::vector<FileConstantMem> &external_weight_mem_data, aclmdlConfigHandle *handle);
  static Status InitExternalWeightMem(const ComputeGraphPtr &root_graph, std::vector<FileConstantMem> &external_weight_mem_data);
  struct ModelExecuteParam {
    std::function<void(Status, void *, void *)> callback;
    void *req_mbuf;
    void *resp_mbuf;
  };
 protected:
  void Run();
  void DestroyDatasetResource();
  void Stop();
  Status AllocEventIOBuffer(const ComputeGraphPtr &root_graph) const;
  Status FreeEventIOBuffer();
  virtual Status DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph);
  virtual Status DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs);
  Status GetGlobalStepAddr();
  Status ParseModelDesc(const ComputeGraphPtr &root_graph);
  virtual Status GetInputAndOutputNum(const ComputeGraphPtr &root_graph, const ModelQueueParam &model_queue_param);
  virtual Status LoadWithAicpuSd(const ComputeGraphPtr &root_graph, const ModelQueueParam &model_queue_param);
  virtual Status UnloadFromAicpuSd();
  static Status UpdateTensorDesc(const RuntimeTensorDesc &runtime_tensor_desc, GeTensorDesc &tensor_desc);
  static Status UpdateRuntimeTensorDesc(const GeTensorDesc &tensor_desc, RuntimeTensorDesc &runtime_tensor_desc);
  static Status UpdateRuntimeShape(const GeShape &shape, int64_t (&shape_buffer)[33]);
  virtual Status PrepareInputs(std::vector<DataBuffer> &model_inputs);
  Status UpdateBufferDataAddr(size_t index, void *&buffer_data, uint64_t buffer_size) const;
  virtual Status PrepareOutputs(std::vector<DataBuffer> &model_outputs);
  virtual Status UpdateOutputs(std::vector<DataBuffer> &model_outputs);
  static Status GetTensorSize(const GeTensorDesc &tensor_desc, int64_t &tensor_size);
  static Status CopyMbufHead(rtMbufPtr_t src, rtMbufPtr_t dst);
  Status CheckAndGetAlignAttr(uint32_t &align_interval, std::vector<uint32_t> &align_offsets);
  virtual Status CheckInputs();
  virtual Status PublishOutputWithoutExecute();
  virtual void PublishErrorOutput(Status ret);
  virtual void ClearOutputs();
  void FreeOutputs();
  bool IsEventInput(const int64_t index) const;
  bool IsEventOutput(const int64_t index) const;
  virtual void UpdateFusionInputsAddr();
  Status ReportStatus();
  Status ClearModelInner(const int32_t clear_type);
  Status StopSchedule();
  Status ClearAndRestart();
  Status AicpuClearModel(const int32_t clear_type);
  bool StopAndWaitRestart();
  Status CreateFakeAicpuModelAndStream();
  Status CheckAicpuKernelSupported(const std::string &kernel_name, bool &is_supported) const;
  Status ParseModelOutputToTensorDesc(const aclTensorDesc *acl_tensor_desc, GeTensorDesc &tensor_desc) const;
  Status CreateInputDataset(const std::vector<DataBuffer> &inputs);
  Status CreateOutputDataset(const std::vector<DataBuffer> &outputs);
 private:
  void FinalizeInternal();
 protected:
  std::thread run_thread_;
  BlockingQueue<ModelExecuteParam> task_queue_;
  size_t num_inputs_ = 0U;
  size_t num_outputs_ = 0U;
  aclmdlDataset *input_dataset_ = nullptr;
  aclmdlDataset *output_dataset_ = nullptr;
  aclmdlDesc *model_desc_ = nullptr;
  aclmdlConfigHandle *handle_ = nullptr;
  std::vector<aclTensorDesc *> acl_tensor_desc_;
  std::vector<aclDataBuffer *> output_data_buffer_;
  std::vector<aclDataBuffer *> input_data_buffer_;
  // send/recv num
  size_t input_events_num_ = 0U;
  size_t output_events_num_ = 0U;
  // queue num
  size_t input_queues_num_ = 0U;
  size_t output_queues_num_ = 0U;
  std::vector<QueueAttrs> input_queue_attrs_;
  std::vector<QueueAttrs> output_queue_attrs_;
  std::vector<int32_t> input_fusion_offsets_;
  std::vector<void *> input_mbuf_addresses_;
  std::vector<void *> output_mbuf_addresses_;
  std::vector<void *> input_buf_addresses_;
  std::vector<void *> output_buf_addresses_;
  std::vector<GeTensorDesc> input_tensor_descs_;
  std::vector<GeTensorDesc> output_tensor_descs_;
  std::vector<int64_t> input_tensor_sizes_;
  std::vector<int64_t> output_tensor_sizes_;
  std::vector<RuntimeTensorDesc> output_runtime_tensor_descs_;
  std::vector<RuntimeTensorDesc> output_static_runtime_tensor_descs_;
  std::vector<bool> is_input_dynamic_;
  std::vector<bool> is_output_dynamic_;
  uint32_t model_id_ = UINT32_MAX;
  uint32_t aicpu_model_id_ = UINT32_MAX;
  aclmdlRI aicpu_model_handle_ = nullptr;
  aclrtStream aicpu_stream_ = nullptr;
  int32_t aicpu_stream_id_ = -1;
  int32_t device_id_ = 0;
  bool is_host_ = false;
  aclrtStream stream_ = nullptr;
  aclrtContext rt_context_ = nullptr;
  CpuSchedModel model_;
  std::map<uint32_t, NamedAttrs> align_attrs_;
  uint64_t global_step_ = 0UL;
  int32_t esched_process_priority_ = -1;  // -1 is user unset eshced priority
  int32_t esched_event_priority_ = -1;
  bool is_need_execute_model_ = true;
  int32_t data_ret_code_ = 0;
  void *aicpu_handle_ = nullptr;
  int32_t execute_times_ = -1;
  void *new_allocated_global_step_ = nullptr;
  bool is_need_alloc_output_mbuf_ = true;  // no need alloc mbuf for output in ps model
  ModelExecuteParam model_execute_param_;   // record current model execute param
  int32_t status_output_queue_device_id_ = 0; // status queue device id
  uint32_t status_output_queue_id_ = 0U; // output queue for report status
  int32_t model_uuid_ = 0U; // model uuid
  bool need_report_status_ = false; // whether to report status
  uint32_t input_consume_num_ = 0U; // num of times when report status fail
  std::atomic<bool> stop_schedule_flag_{false};
  std::atomic<bool> has_stop_schedule_{false};
  std::vector<uint32_t> aicpu_model_ids_;
  InputAlignAttrs input_align_attrs_{};
  std::vector<FileConstantMem> external_weight_mem_data_{};
  bool exec_with_mutex_ = false;
  static std::mutex exec_mutex_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_DYNAMIC_MODEL_EXECUTOR_H_
