/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_COMMON_SUBPROCESS_SUBPROCESS_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_COMMON_SUBPROCESS_SUBPROCESS_MANAGER_H_

#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <functional>
#include "ge/ge_api_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/thread_pool/thread_pool.h"

namespace ge {
constexpr uint32_t kAbnormalTypeNode = 1U;
constexpr uint32_t kAbnormalTypeDevice = 2U;
constexpr uint32_t kAbnormalTypeModelInstance = 3U;
enum class ProcStatus : int32_t {
  NORMAL = 0,
  EXITED,
  STOPPED,
  INVALID
};
class SubprocessManager {
 public:
  struct SubprocessConfig {
    std::string process_type;
    int32_t death_signal = 0;
    std::map<std::string, std::string> envs;
    std::vector<std::string> args;
    std::map<std::string, std::string> kv_args;
    std::vector<std::string> unset_envs;
  };
  GE_DELETE_ASSIGN_AND_COPY(SubprocessManager);

  static SubprocessManager& GetInstance();

  Status Initialize();
  Status ForkSubprocess(const SubprocessConfig &subprocess_config, pid_t &pid);
  void NotifySubprocessShutdown(const pid_t &pid);
  Status ShutdownSubprocess(const pid_t &pid, const uint32_t wait_time_in_sec = 3U);
  void RegExcptHandleCallback(pid_t pid, std::function<void(const ProcStatus &)> callback);
  void UnRegExcptHandleCallback(pid_t pid);
  void Finalize();
  static Status HasFlowGw(bool &has_flowgw);

  static std::vector<std::string> FormatArgs(const SubprocessConfig &subprocess_config);
 private:
  SubprocessManager();
  ~SubprocessManager() {
    Finalize();
  }

  static Status Execute(const std::string &path, const SubprocessConfig &subprocess_config, char_t *const argv[]);
  static Status ToCmdlineArgs(const std::vector<std::string> &args_strings, char_t *var_args[]);
  static void FormatKvArgs(const std::map<std::string, std::string> &kv_args,
                           std::vector<std::string> &out_args);
  static Status GetBinDir(std::string &bin_dir);
  static Status GetFlowGwBinDir(const std::string &bin_dir, std::string &flowgw_bin_dir);
  static bool FileExist(const std::string &file_path);
  void MonitorSubprocess();
  // key: subprocess type
  std::map<std::string, std::string> executable_paths_;
  std::atomic<bool> run_flag_;
  std::thread watch_sub_proc_thread_;
  std::map<pid_t, std::function<void(const ProcStatus &)>> excpt_handle_callbacks_;
  std::mutex callbacks_mu_;
  ThreadPool pool_{"ge_dpl_subp", 1, false};
  std::map<pid_t, bool> planned_shutdown_;
  std::map<std::string, ProcStatus> process_status_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_COMMON_SUBPROCESS_SUBPROCESS_MANAGER_H_
