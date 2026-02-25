/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_LAUNCH_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_LAUNCH_H_
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "register/node_converter_registry.h"
namespace gert {
namespace bg {
struct CommonLaunchArg {
  ValueHolderPtr stream;
  ValueHolderPtr bin_addr;
  ValueHolderPtr block_dim;
  ValueHolderPtr schedule_mode;
  ValueHolderPtr local_mem_size;
  ValueHolderPtr workspace_addrs;
  ValueHolderPtr shapebuffer_addr;
  ValueHolderPtr qos;
  std::vector<ValueHolderPtr> input_shapes;
  std::vector<ValueHolderPtr> output_shapes;
  ge::NodePtr node;
  LoweringGlobalData *global_data;
  ValueHolderPtr dfx_holder;
  ValueHolderPtr rt_arg;
};

struct AtomicLoweringArg {
  ValueHolderPtr workspaces_size;
  DevMemValueHolderPtr &workspaces_addrs;
  std::vector<ValueHolderPtr> output_sizes;
  std::vector<DevMemValueHolderPtr> output_addrs;
};

ValueHolderPtr LaunchKernelWithHandle(const CommonLaunchArg &launch_arg,
                                      const ValueHolderPtr &stub_func,
                                      const ValueHolderPtr &node_info,
                                      const std::vector<DevMemValueHolderPtr> &input_addrs,
                                      const std::vector<DevMemValueHolderPtr> &output_addrs);
ValueHolderPtr LaunchKernelWithFlag(const CommonLaunchArg &launch_arg,
                                    const std::vector<DevMemValueHolderPtr> &input_addrs,
                                    const std::vector<DevMemValueHolderPtr> &output_addrs);
ValueHolderPtr AtomicLaunchKernelWithHandle(const CommonLaunchArg &launch_arg,
                                            const ValueHolderPtr &stub_func,
                                            const ValueHolderPtr &clean_workspace_indexes,
                                            const std::vector<DevMemValueHolderPtr> &clean_output_addrs,
                                            const ValueHolderPtr &clean_workspace_addrs);
ValueHolderPtr AtomicLaunchKernelWithFlag(const CommonLaunchArg &launch_arg,
                                          const ValueHolderPtr &clean_workspace_indexes,
                                          const std::vector<DevMemValueHolderPtr> &clean_output_addrs,
                                          const ValueHolderPtr &clean_workspace_addrs);
ValueHolderPtr LaunchFFTSPlusTaskNoCopy(const ValueHolderPtr &stream, bg::ValueHolderPtr task_info_para,
    bg::ValueHolderPtr need_launch, bg::ValueHolderPtr dfx_holder, bg::ValueHolderPtr workspaces_addr);
ValueHolderPtr LaunchStarsKernel(const ValueHolderPtr &sqe_addr,
                                 const ValueHolderPtr &sqe_len,
                                 const ValueHolderPtr &stream);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_LAUNCH_H_
