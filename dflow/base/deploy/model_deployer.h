/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_RUNTIME_DEPLOY_MODEL_DEPLOYER_H_
#define BASE_RUNTIME_DEPLOY_MODEL_DEPLOYER_H_

#include <map>
#include <string>
#include <vector>
#include "dflow/inc/data_flow/model/flow_model.h"
#include "common/plugin/ge_make_unique_util.h"
#include "ge/ge_api_error_codes.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
constexpr uint32_t kCallbackStartRedeploy = 1U;
constexpr uint32_t kCallbackDynamicSched = 2U;
constexpr uint32_t kCallbackFailedRedeploy = 3U;
constexpr uint32_t kCallbackRedeployDone = 4U;
struct UserExceptionNotify {
  uint32_t type = 0;  // 0: exception occurred, 1: exception expired
  int32_t exception_code = 0;
  uint64_t trans_id = 0;
  uint64_t user_context_id = 0;
  std::string scope;
  const void *exception_context = nullptr;
  uint32_t exception_context_len = 0;
};

struct DeployResult {
  uint32_t model_id;
  std::vector<DeployQueueAttr> input_queue_attrs;
  std::vector<DeployQueueAttr> output_queue_attrs;
  std::vector<DeployQueueAttr> control_input_queue_attrs;
  std::vector<DeployQueueAttr> control_output_queue_attrs;
  std::vector<std::vector<DeployQueueAttr>> broadcast_input_queue_attrs;
  std::function<Status(void)> dev_abnormal_callback;
  // exception notify function.
  std::function<void(const UserExceptionNotify&)> exception_notify_callback;
  size_t replica_num = 1U;
  std::string input_model_name;
  std::vector<DeployQueueAttr> status_output_queue_attrs;
  std::vector<DeployQueueAttr> sched_input_queue_attrs;
  std::vector<DeployQueueAttr> sched_output_queue_attrs;
  DeployPlan::DynamicSchedIndex model_index_info;
  std::map<int32_t, int32_t> datagw_request_bindings;
  bool is_dynamic_sched = false;
  bool is_exception_catch = false;
  bool contains_n_mapping_node = false;
  DeployPlan::AbnormalStatusCallbackInfo *abnormal_status_callback_info = nullptr;
  // 内层集合是同一个device上裁边场景具有连接关系的模型集合
  std::vector<std::unordered_set<std::string>> model_trimming_edges_model_instances;
  InputAlignAttrs input_align_attrs = {};
};

class ModelDeployer {
 public:
  ModelDeployer() = default;
  GE_DELETE_ASSIGN_AND_COPY(ModelDeployer);
  virtual ~ModelDeployer() = default;

  /// Deploy model to devices
  /// @param model                models to deploy
  /// @param deploy_result        deploy result
  /// @return                     SUCCESS if deployed successfully, otherwise returns appropriate error code
  virtual Status DeployModel(const FlowModelPtr &flow_model, DeployResult &deploy_result) = 0;

  /// Undeploy model
  /// @param model_id             id of the deployed model
  /// @return                     SUCCESS if undeployed successfully, otherwise returns appropriate error code
  virtual Status Undeploy(const uint32_t model_id) = 0;

  virtual Status UpdateProfilingInfo(const bool) {return SUCCESS; };

  /// Get local device node mesh index
  /// @return                     empty means not support
  virtual Status GetDeviceMeshIndex(const int32_t, std::vector<int32_t> &)  { return UNSUPPORTED; };

  /// Get valid logic device id str
  virtual Status GetValidLogicDeviceId(std::string &) { return UNSUPPORTED; };
};
}  // namespace ge

#endif  // BASE_RUNTIME_DEPLOY_MODEL_DEPLOYER_H_
