/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_ENGINE_DAEMON_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_ENGINE_DAEMON_H_

#include <string>
#include <atomic>
#include "acl/acl.h"
#include "framework/common/debug/ge_log.h"
#include "framework/executor/ge_executor.h"
#include "common/utils/rts_api_utils.h"
#include "common/message_handle/message_server.h"
#include "executor/executor_context.h"
#include "executor/event_handler.h"

namespace ge {
class EngineDaemon {
 public:
  explicit EngineDaemon(bool is_host_cpu = false);

  Status InitializeWithArgs(int32_t argc, char_t *argv[]);
  void Finalize();
  Status LoopEvents();

 private:
  Status InitializeExecutor();
  Status InitializeGeExecutor();
  Status InitializeMaintenanceFromOption();
  Status FinalizeMaintenance();
  Status InitDumpFromOption(std::map<std::string, std::string> &options);
  Status InitProfilingFromOption(const std::map<std::string, std::string> &options_all) const;
  Status ParseCmdLineArgs(int32_t argc, char_t *argv[]);
  static void GetGlobalEnvOptions(std::map<std::string, std::string> &env_option);

  Status HandleEvent(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);

  Status NotifyInitialized() const;

  Status InitializeWithKVArgs();

  void FinalizeThread();
  static void PrintLogLevel();
  static void SignalHandler(int32_t sig_num);

  template<typename T>
  Status ToNumber(const char_t *num_str, T &value) const {
    GE_CHECK_NOTNULL(num_str);
    std::stringstream ss(num_str);
    ss >> value;
    if (ss.fail()) {
      GELOGE(PARAM_INVALID, "Failed to convert [%s] to number", num_str);
      return PARAM_INVALID;
    }
    if (!ss.eof()) {
      GELOGE(PARAM_INVALID, "Failed to convert [%s] to number", num_str);
      return PARAM_INVALID;
    }
    return SUCCESS;
  }
  void TransArray2ArgsOption(const int32_t start, const int32_t end, char_t **argv);

  GeExecutor ge_executor_;
  EventHandler event_handler_;
  std::string mem_group_name_;
  int32_t device_id_ = -1;
  int32_t msg_queue_device_id_ = -1;
  aclrtContext rt_context_ = nullptr;
  uint32_t req_msg_queue_id_ = UINT32_MAX;
  uint32_t rsp_msg_queue_id_ = UINT32_MAX;
  bool is_host_cpu_ = false;
  bool is_dump_inited_ = false;
  std::map<std::string, std::string> args_option_;
  std::condition_variable cond_var_;
  std::mutex single_mutex_;
  std::atomic<bool> is_finish_{false};
  std::string base_dir_;
  std::shared_ptr<MessageServer> executor_message_server_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_ENGINE_DAEMON_H_
