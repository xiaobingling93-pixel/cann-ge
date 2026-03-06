/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
#define GE_COMMON_DUMP_EXCEPTION_DUMPER_H_

#include <vector>
#include <mutex>

#include "graph/op_desc.h"
#include "framework/common/ge_types.h"
#include "common/dump/dump_properties.h"
#include "exe_graph/runtime/dfx_info_filler.h"
#include "common/dump/kernel_tracing_utils.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/mem.h"
#include "graph_metadef/common/ge_common/util.h"

#include <set>
#include <utility>

namespace ge {
struct ExtraOpInfo {
  std::string node_info;
  std::string tiling_data;
  std::string args_before_execute;
  uint32_t tiling_key{0U};
  uintptr_t args{0U};
  size_t args_size{0UL};
  std::map<uint64_t, uint64_t> cust_to_relevant_offset_;
  std::vector<void *> input_addrs;
  std::vector<void *> output_addrs;
  std::vector<uint64_t> input_sizes;
  std::vector<uint64_t> output_sizes;
  bool is_host_args{false};
  std::vector<std::pair<uintptr_t, int64_t>> workspace_info{};
  void DebugLogString() {
    if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
      return;
    }
    std::stringstream ss;
    ss << "Node Info: " << node_info;
    GELOGD("%s", ss.str().c_str());
    ss.str("");
    ss << "Tiling key: " << tiling_key;
    ss << " data: " << tiling_data;
    GELOGD("%s", ss.str().c_str());
    ss.str("");
    ss << "Args before execute: " << args_before_execute;
    ss << " addr: " << args;
    ss << " size: " << args_size;
    GELOGD("%s", ss.str().c_str());
    for (const auto &ele : workspace_info) {
      ss.str("");
      ss << "Workspace addr: " << ele.first;
      ss << " size: " << ele.second;
      GELOGD("%s", ss.str().c_str());
    }
  }

  void RecordArgsBefore() {
    if ((args != 0U) && (args_size != 0UL)) {
      uint8_t *host_addr = nullptr;
      aclError rt_ret =
          aclrtMallocHost(PtrToPtr<uint8_t *, void *>(&host_addr), args_size);
      if (rt_ret != ACL_SUCCESS) {
        GELOGW("[Call][RtMallocHost] failed, size:%zu, ret:0x%X", args_size, rt_ret);
        return;
      }
      GE_MAKE_GUARD_RTMEM(host_addr);
      rt_ret = aclrtMemcpy(host_addr, static_cast<uint64_t>(args_size), reinterpret_cast<void *>(args),
          static_cast<uint64_t>(args_size), ACL_MEMCPY_DEVICE_TO_HOST);
      if (rt_ret != ACL_SUCCESS) {
        GELOGW("[Call][aclrtMemcpy] failed, size:%zu, ret:0x%X", args_size, rt_ret);
        return;
      }
      std::stringstream ss;
      ss << "args before execute: ";
      gert::PrintHex(reinterpret_cast<void **>(host_addr), args_size / sizeof(void *), ss);
      args_before_execute = ss.str();
    }
  }
};

class ExceptionDumper {
 public:
  // auto_register = true, register the dumper to the global dumper
  // and the dumper will be called on aic exception occur.
  explicit ExceptionDumper(bool auto_register = true);
  ~ExceptionDumper();

  // is_dynamic  dynamic op or single op,
  // if true, OpInfo will be saved in aging mode: the later ones may overwrite the earlier ones.
  void SaveDumpOpInfo(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id, bool is_dynamic);
  void SaveInputOutputInfo(const bool is_input, const OpDescPtr &op, OpDescInfo &op_desc_info) const;
  void LogExceptionTvmOpInfo(const OpDescInfo &op_desc_info) const;
  void LogExceptionArgs(const OpDescInfo &op_desc_info) const;
  bool GetOpDescInfo(const OpDescInfoId &op_id, OpDescInfo &op_desc_info);
  OpDescInfo *MutableOpDescInfo(const uint32_t task_id, const uint32_t stream_id);

  static Status DumpDevMem(const ge::char_t * const file, const void * const addr, const uint64_t size);

  static void Reset(ExtraOpInfo &extra_op_info);

  Status DumpNodeInfo(const OpDescInfo &op_desc_info, const std::string &file_path,
                      const bool is_exception, const bool is_ffts_plus,
                      const ge::DumpProperties &dump_properties) const;

  void Clear();

 private:
  void RefreshAddrs(OpDescInfo &op_desc_info) const;
  void SaveOpDescInfo(const OpDescPtr &op, const OpDescInfoId &id, OpDescInfo &op_desc_info) const;
  Status DumpExceptionInput(const OpDescInfo &op_desc_info, const std::string &dump_file,
                            const bool is_exception, const ge::DumpProperties &dump_properties) const;
  Status DumpExceptionOutput(const OpDescInfo &op_desc_info, const std::string &dump_file,
                             const bool is_exception, const ge::DumpProperties &dump_properties) const;
  Status DumpExceptionWorkspace(const OpDescInfo &op_desc_info, const std::string &dump_file,
                                const bool is_exception, const ge::DumpProperties &dump_properties) const;

  // is_dynamic: dynamic op or single op,
  // if true, OpInfo will be saved in aging mode: the later ones may overwrite the earlier ones.
  void SaveDumpOpInfoLocal(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id,
                           bool is_dynamic);
  void SaveOpInfoToAdump(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id,
                         bool is_dynamic = false);

  std::mutex mutex_;
  std::vector<OpDescInfo> op_desc_info_;
  size_t op_desc_info_idx_{0UL};
  bool is_registered_{false};

  // device_id and stream_id of op saved by adump, for Clear() to notify adump to release op infos
  std::set<std::pair<int32_t, uint32_t>> devid_stream_saved_;
};
}  // namespace ge

#endif // GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
