/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_TYPES_H_
#define INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>
#include "graph/buffer.h"
#include "runtime/rt_model.h"

namespace ge {
/*lint -e148*/
struct RunContext {
  uint64_t sessionId;
  uint64_t dataMemSize;
  uint8_t *dataMemBase;
  std::map<int64_t, uint64_t> mem_type_data_mem_size;
  std::map<int64_t, uint8_t *> mem_type_data_mem_base;
  uint64_t weightMemSize;
  uint8_t *weightMemBase;
  ge::Buffer weightsBuffer;
};

/*lint +e148*/
struct Task {
  uint32_t id;
  uint16_t type;
  void *stream;
  void *event;
};

struct OpInfo {
  std::string engine;  // which engin
  /*lint -e148*/
  std::string opKernelLib;  // which opsKernelStore
  int32_t computeCost;     // compute cost
  bool flagPartial;    // whether to support is related to shape
  bool flagAsync;      // Whether to support asynchronous
  bool isAtomic;       // whether to support atomic addr clean
  std::string opFileName;   // op file name
  std::string opFuncName;   // op function name
};

enum class CheckSupportFlag : uint32_t {
  kDefault = 0,
  kNotSupportDynamicShape
};

enum class ModelTaskType : uint32_t {
  MODEL_TASK_KERNEL = 0,
  MODEL_TASK_EVENT_RECORD,
  MODEL_TASK_EVENT_WAIT,
  MODEL_TASK_FUSION_START,
  MODEL_TASK_FUSION_END,
  MODEL_TASK_KERNEL_EX,
  MODEL_TASK_HCCL,
  MODEL_TASK_STREAM_SWITCH,
  MODEL_TASK_STREAM_ACTIVE,
  MODEL_TASK_LABEL_SET,
  MODEL_TASK_LABEL_SWITCH,
  MODEL_TASK_LABEL_GOTO,
  MODEL_TASK_PROFILER_TRACE,
  MODEL_TASK_MEMCPY_ASYNC,
  MODEL_TASK_NOTIFY_RECORD,
  MODEL_TASK_NOTIFY_WAIT,
  MODEL_TASK_REDUCE_ASYNC,
  MODEL_TASK_RDMA_SEND,
  MODEL_TASK_EVENT_RESET,
  MODEL_TASK_END_GRAPH,
  MODEL_TASK_STREAM_SWITCH_N,
  MODEL_TASK_RDMA_DB_SEND,
  MODEL_TASK_MEMCPY_ADDR_ASYNC,
  MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX,
  MODEL_TASK_STREAM_LABEL_GOTO,
  MODEL_TASK_MODEL_EXIT,
  MODEL_TASK_ALL_KERNEL,
  MODEL_TASK_PROFILER_TRACE_EX,
  MODEL_TASK_FFTS,
  MODEL_TASK_FFTS_PLUS,
  MODEL_TASK_DSA,
  MODEL_TASK_CMO,
  MODEL_TASK_BARRIER,
  MODEL_TASK_NPU_GET_FLOAT_STATUS,
  MODEL_TASK_NPU_CLEAR_FLOAT_STATUS,
  MODEL_TASK_DVPP,
  MODEL_TASK_NPU_GET_DEBUG_FLOAT_STATUS,
  MODEL_TASK_NPU_CLEAR_DEBUG_FLOAT_STATUS,
  MODEL_TASK_CMO_ADDR,
  MODEL_TASK_VECTOR_KERNEL,
  MODEL_TASK_VECTOR_ALL_KERNEL,
  MODEL_TASK_UPDATE,
  MODEL_TASK_NOP,
  MODEL_TASK_PREPROCESS_KERNEL,
  MODEL_TASK_SUPER_KERNEL,
  MODEL_TASK_MEM_EVENT_RECORD,
  MODEL_TASK_MEM_EVENT_WAIT,
  MODEL_TASK_FUSION_KERNEL,
  MODEL_TASK_KERNEL_LAUNCH_V2,
  MODEL_TASK_CCU_KERNEL,
  MODEL_TASK_CUSTOM_KERNEL
};
}  // namespace ge

#endif  // INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_TYPES_H_
