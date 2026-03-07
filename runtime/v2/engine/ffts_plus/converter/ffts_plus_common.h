/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_FFTS_PLUS_FFTS_PLUS_COMMON_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_FFTS_PLUS_FFTS_PLUS_COMMON_H_
#include "common/checker.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/value_holder_generator.h"
#include "engine/node_converter_utils.h"
#include "register/ffts_node_calculater_registry.h"
#include "register/ffts_node_converter_registry.h"
#include "lowering/graph_converter.h"
#include "lowering/placement/placed_lowering_result.h"
namespace gert {
inline bool IsFftsSubgraphNode(const ge::NodePtr &node) {
  return node->GetOpDesc()->HasAttr(ge::ATTR_NAME_FFTS_SUB_GRAPH) ||
         node->GetOpDesc()->HasAttr(ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH);
}
inline bool IsNoNeedCalcSize(const ge::NodePtr &node) {
  static std::unordered_set<std::string> no_need_calc_type = {"Data", "NetOutput", "Variable", "Const", "Constant",
    "PhonyConcat", "StreamActive", "PhonyReduce", "Identity"};
  bool no_need_flag = no_need_calc_type.find(node->GetType()) != no_need_calc_type.end();
  no_need_flag |= (node->GetOpDesc()->GetOpKernelLibName() == ge::kEngineNameGeLocal);
  no_need_flag &= (node->GetType() != "PartitionedCall");
  GELOGD("Node[%s/%s] need calc mem size flag:%d.", node->GetName().c_str(), node->GetType().c_str(), no_need_flag);
  return no_need_flag;
}

constexpr const char *kAtomicCtxIdList = "_atomic_context_id_list";

struct MemPrePara {
  size_t offset{0};
  size_t size{0};
  size_t pre_size{0};
  bg::ValueHolderPtr args_data{nullptr};
};

struct MemGuard{
  void* guard_ptr{nullptr};
  int64_t guard_val{0};
};

struct FFTSAllMemPara {
  std::unordered_map<std::string, MemPrePara> node_to_args_para;
  size_t total_size{0};
  const domi::FftsPlusTaskDef *ffts_plus_task_def;
  bg::ValueHolderPtr dev_addr_base{nullptr};
  bg::ValueHolderPtr host_addr_base{nullptr};
  bg::ValueHolderPtr mem_guarder{nullptr};
  bg::ValueHolderPtr task_data{nullptr};
};
#define ADDR_ALIGN_NUM_128 128
inline uintptr_t AddrAlignBy128(const uintptr_t addr) {
  return (((addr + ADDR_ALIGN_NUM_128) - 1U) / ADDR_ALIGN_NUM_128) * ADDR_ALIGN_NUM_128;
}

inline void FFTSAddNodeMemPara(FFTSAllMemPara &all_mem_para, size_t mem_size, size_t pre_data_size,
    std::unique_ptr<uint8_t[]> pre_data, std::string cfg_key = "") {
  MemPrePara node_para;
  node_para.size = mem_size;
  node_para.offset = all_mem_para.total_size;
  node_para.pre_size = pre_data_size;
  // todo :if size 0 ok??
  if (pre_data) {
    node_para.args_data = bg::ValueHolder::CreateConst(pre_data.get(), mem_size);
  } else {
    node_para.args_data = bg::ValueHolder::CreateConst(&pre_data, sizeof(pre_data));
  }
  GELOGD("Add node mem size:%zu.", mem_size);
  all_mem_para.total_size += (mem_size + sizeof(int64_t));
  all_mem_para.node_to_args_para[cfg_key] = std::move(node_para);
}
enum class MemParaInKey {
  PRE_PARA = 0,
  PRE_DATA = 1,
  DEV_ADDR = 2,
  HOST_ADDR = 3,
  kNUM
};
enum class MemParaOutKey {
  NODE_PARA = 0,
  MEM_GUARD = 1,
  kNUM
};
FFTSNodeCalculaterRegistry::NodeCalculater GetNodeCalculater(const ge::NodePtr &node);
ge::Status CreateMemoryGuard(FFTSAllMemPara &all_mem_para);
bg::ValueHolderPtr CreateNodeMemParam(const ge::NodePtr &node, FFTSAllMemPara &all_mem_para, std::string key = "");

enum class TaskPreOutKey {
  NODE_PARA = 0,
  TASK_INFO = 1,
  kNUM
};
enum class TaskProcKey {
  H2D_COPY = 0,
  TASK_LAUNCH = 1,
  kNUM
};

struct FFTSLuanchArg {
  ge::NodePtr node;
  LoweringGlobalData *global_data;
  bool do_copy;
  bg::ValueHolderPtr need_launch;
  bg::ValueHolderPtr workspaces_addr;
};
std::vector<bg::ValueHolderPtr> FFTSTaskAndArgsLaunch(FFTSLuanchArg launch_arg, FFTSAllMemPara &all_mem_para,
                                                      std::vector<bg::ValueHolderPtr> &task_info_vec);
ge::Status LoweringGraphPostProc(const LowerResult *graph_result, const std::vector<bg::ValueHolderPtr> &task_ret,
    const std::vector<bg::ValueHolderPtr> &free_vec, const std::vector<bg::ValueHolderPtr> &alloc_vec);
std::vector<bg::ValueHolderPtr> RedirectLaunchArgs(const bg::ValueHolderPtr args_para);
}
#endif  // AIR_CXX_RUNTIME_V2_ENGINE_FFTS_PLUS_FFTS_PLUS_COMMON_H_
