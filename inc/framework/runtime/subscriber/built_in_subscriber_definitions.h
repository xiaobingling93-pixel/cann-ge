/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_BUILT_IN_SUBSCRIBER_DEFINITIONS_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_BUILT_IN_SUBSCRIBER_DEFINITIONS_H_
#include <type_traits>
#include <vector>
#include "graph/gnode.h"
#include "common/ge_types.h"
#include "framework/common/ge_visibility.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "graph/anchor.h"
#include "framework/common/profiling_definitions.h"
#include "acl/acl_rt.h"
#include "graph/fast_graph/execute_graph.h"

namespace ge {
class GeRootModel;
}
namespace gert {
constexpr size_t kProfilingDataCap = 10UL * 1024UL * 1024UL;
constexpr size_t kInitSize = 10UL * 1024UL;
constexpr size_t kDouble = 2UL;
using SymbolsToValue = std::unordered_map<uint64_t, AsyncAnyValue *>;
static_assert(kInitSize > static_cast<uint64_t>(gert::profiling::kProfilingIndexEnd),
              "The max init size is less than kProfilingIndexEnd.");
enum class BuiltInSubscriberType {
  kGeProfiling,
  kDumper,
  kTracer,
  kCannProfilerV2,
  kCannHostProfiler,
  kMemoryProfiler,
  kHostDumper,
  kNum
};

enum class ProfilingType {
  kCannHost = 0,  // 打开Host侧调度的profiling
  kDevice = 1,
  kGeHost = 2,  // 打开GE Host侧调度的profiling
  kTrainingTrace = 3,
  kTaskTime = 4,
  kMemory = 5,
  kCannHostL1 = 6,
  kNum,
  kAll = kNum
};
static_assert(static_cast<size_t>(ProfilingType::kNum) < sizeof(uint64_t) * static_cast<size_t>(8),
              "The max num of profiling type must less than the width of uint64");

enum class DumpType {
  kDataDump = 0,
  kExceptionDump = 1,
  kOverflowDump = 2,
  kHostDump = 3,
  // kNeedSubscribe为分界线，枚举值小于kNeedSubscribe的
  // 枚举量表示此Dump需要通过rt2.0中的OnEvent来做dump动作
  // 大于kNeedSubscribe的枚举量表示不需要通过OnEvent来做dump
  // 动作的dump
  kNeedSubscribe = 4,
  kLiteExceptionDump = 4,
  kNum = 5,
  kAll = kNum
};
static_assert(static_cast<size_t>(DumpType::kNum) < sizeof(uint64_t) * static_cast<size_t>(8),
              "The max num of dumper type must less than the width of uint64");
class ModelV2Executor;
struct TraceAttr {
  bool is_fp = false;
  bool is_bp = false;
  int64_t start_log_id = -1;
  int64_t logic_stream_id = 0;
};
// todo : 需要整改，root_model等需要删除
struct SubscriberExtendInfo {
  SubscriberExtendInfo(ModelV2Executor *out_executor, const ge::ExecuteGraphPtr &out_exe_graph,
                       const ge::ComputeGraphPtr &out_root_graph, const ge::ModelData &out_model_data,
                       const std::shared_ptr<ge::GeRootModel> &out_root_model,
                       const SymbolsToValue &out_symbols_to_value, const uint32_t out_model_id,
                       const std::string &out_model_name, const aclrtStream out_stream,
                       const std::unordered_map<std::string, TraceAttr> &out_node_names_to_attrs)
      : executor(out_executor),
        exe_graph(out_exe_graph),
        root_graph(out_root_graph),
        model_data(out_model_data),
        root_model(out_root_model),
        symbols_to_value(out_symbols_to_value),
        model_id(out_model_id),
        model_name(out_model_name),
        stream(out_stream),
        node_names_to_attrs(out_node_names_to_attrs) {}
  SubscriberExtendInfo()
      : executor(nullptr),
        exe_graph(nullptr),
        root_graph(nullptr),
        root_model(nullptr),
        model_id(0U),
        stream(nullptr) {}
  ModelV2Executor *executor;
  ge::ExecuteGraphPtr exe_graph;
  ge::ComputeGraphPtr root_graph;
  ge::ModelData model_data;
  std::shared_ptr<ge::GeRootModel> root_model;
  SymbolsToValue symbols_to_value;
  uint32_t model_id;
  std::string model_name;
  aclrtStream stream;
  std::unordered_map<std::string, TraceAttr> node_names_to_attrs;
};

class VISIBILITY_EXPORT BuiltInSubscriberUtil {
 public:
  template <typename T,
            typename std::enable_if<(std::is_same<T, ProfilingType>::value) || (std::is_same<T, DumpType>::value),
                                    int>::type = 0>
  constexpr static uint64_t EnableBit(T et) {
    return 1UL << static_cast<size_t>(et);
  }

  template <typename T,
            typename std::enable_if<(std::is_same<T, ProfilingType>::value) || (std::is_same<T, DumpType>::value),
                                    int>::type = 0>
  static uint64_t BuildEnableFlags(const std::vector<T> &enable_types) {
    uint64_t flag = 0UL;
    for (const auto &et : enable_types) {
      if (et == T::kAll) {
        return EnableBit(T::kNum) - 1UL;
      }
      flag |= EnableBit(et);
    }
    return flag;
  }
};
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_BUILT_IN_SUBSCRIBER_DEFINITIONS_H_
