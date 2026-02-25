/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_kernel_launch.h"
#include "kernel/kernel_log.h"
#include "graph/debug/ge_attr_define.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "engine/aicore/fe_rt2_common.h"
namespace gert {
namespace bg {
namespace {
const size_t kMixNotifyIdNum = 2;

std::vector<ValueHolderPtr> BuildLaunchCommonHead(const CommonLaunchArg &launch_arg,
                                                  const size_t &io_num) {
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(launch_arg.stream);
  inputs.emplace_back(launch_arg.bin_addr);
  inputs.emplace_back(launch_arg.block_dim);
  inputs.emplace_back(launch_arg.workspace_addrs);
  inputs.emplace_back(launch_arg.shapebuffer_addr);
  inputs.emplace_back(launch_arg.qos);
  inputs.emplace_back(ValueHolder::CreateConst(&io_num, sizeof(io_num)));
  inputs.emplace_back(launch_arg.schedule_mode);
  inputs.emplace_back(launch_arg.dfx_holder);
  inputs.emplace_back(launch_arg.rt_arg);
  inputs.emplace_back(launch_arg.local_mem_size);
  return inputs;
}
enum class CoreNumType {
  ALL_CORE,
  AI_CORE,
  VECTOR_CORE,
  TYPE_NUM
};

bool GetResource(const ge::OpDescPtr op_desc, const CommonLaunchArg &launch_arg, std::vector<ValueHolderPtr> &inputs) {
  auto sub_stream_id = op_desc->GetAttachedStreamId();
  if (sub_stream_id < 0) {
    GELOGW("Node[%s] received an invalid sub-stream ID [%ld].", op_desc->GetNamePtr(), sub_stream_id);
    return false;
  }
  auto sub_stream = launch_arg.global_data->GetStreamById(sub_stream_id);
  FE_ASSERT_NOTNULL(sub_stream);
  inputs.emplace_back(sub_stream);

  std::vector<int32_t> notify_id_v;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::RECV_ATTR_NOTIFY_ID, notify_id_v);
  if (notify_id_v.size() != kMixNotifyIdNum) {
    GELOGW("Node[%s] get invalid notify size[%zu].", op_desc->GetNamePtr(), notify_id_v.size());
    return false;
  }
  GELOGD("Get sub stream id [%ld], notify id [%ld][%ld].", sub_stream_id, notify_id_v[0], notify_id_v[1]);
  auto notify_holder_1 = launch_arg.global_data->GetNotifyById(notify_id_v[0]);
  FE_ASSERT_NOTNULL(notify_holder_1);
  auto notify_holder_2 = launch_arg.global_data->GetNotifyById(notify_id_v[1]);
  FE_ASSERT_NOTNULL(notify_holder_2);
  inputs.emplace_back(notify_holder_1);
  inputs.emplace_back(notify_holder_2);
  return true;
}

bool BuildLaunchCommonTail(const CommonLaunchArg &launch_arg, std::vector<ValueHolderPtr> &inputs) {
  auto op_desc = launch_arg.node->GetOpDesc();
  kernel::MixCoreArgs mix_args;
  string core_type;
  (void)ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  if (core_type == kCoreTypeMixVectorCore) {
    mix_args.mix_type = kernel::MixType::MIX_VECTOR_CORE;
  } else if (core_type == kCoreTypeMixAICore) {
    mix_args.mix_type = kernel::MixType::MIX_AICORE;
  } else {
    return false;
  }
  if (ge::AttrUtils::HasAttr(op_desc, kDisableMixVectorCore)) {
    GELOGI("Node [%s] has disabled the vector core.", op_desc->GetNamePtr());
    return false;
  }
  std::vector<uint32_t> core_num_v;
  (void)ge::AttrUtils::GetListInt(op_desc, kMixCoreNumVec, core_num_v);
  if (core_num_v.size() != static_cast<size_t>(CoreNumType::TYPE_NUM)) {
    return false;
  }
  mix_args.all_core_num = core_num_v[static_cast<size_t>(CoreNumType::ALL_CORE)];
  mix_args.ai_core_num = core_num_v[static_cast<size_t>(CoreNumType::AI_CORE)];
  mix_args.vec_core_num = core_num_v[static_cast<size_t>(CoreNumType::VECTOR_CORE)];
  if (mix_args.all_core_num == 0 || mix_args.ai_core_num == 0 || mix_args.vec_core_num == 0) {
    GELOGW("Node [%s] core number [%u][%u][%u] is invalid.", op_desc->GetNamePtr(), mix_args.all_core_num, mix_args.ai_core_num,
           mix_args.vec_core_num);
    return false;
  }
  inputs.emplace_back(ValueHolder::CreateConst(&mix_args, sizeof(mix_args)));
  if (launch_arg.global_data->IsSingleStreamScene()) {
    GELOGW("Node[%s] in single stream scene, go normal mode.", op_desc->GetNamePtr());
    return false;
  }
  if (!GetResource(op_desc, launch_arg, inputs)) {
    return false;
  }
  (void)ge::AttrUtils::SetBool(op_desc, kEnableMixVectorCore, true);
  return true;
}

}  // namespace
ValueHolderPtr LaunchKernelWithHandle(const CommonLaunchArg &launch_arg,
                                      const ValueHolderPtr &stub_func,
                                      const ValueHolderPtr &node_info,
                                      const std::vector<DevMemValueHolderPtr> &input_addrs,
                                      const std::vector<DevMemValueHolderPtr> &output_addrs) {
  size_t io_num = input_addrs.size() + output_addrs.size();
  std::vector<ValueHolderPtr> inputs = BuildLaunchCommonHead(launch_arg, io_num);
  inputs.emplace_back(stub_func);
  inputs.emplace_back(node_info);
  inputs.insert(inputs.cend(), input_addrs.cbegin(), input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  inputs.insert(inputs.cend(), launch_arg.input_shapes.cbegin(), launch_arg.input_shapes.cend());
  inputs.insert(inputs.cend(), launch_arg.output_shapes.cbegin(), launch_arg.output_shapes.cend());
  auto ret = BuildLaunchCommonTail(launch_arg, inputs);
  if (ret) {
    return ValueHolder::CreateVoid<ValueHolder>("LaunchMixKernelWithHandle", inputs);
  } else {
    return ValueHolder::CreateVoid<ValueHolder>("LaunchKernelWithHandle", inputs);
  }
}

ValueHolderPtr LaunchKernelWithFlag(const CommonLaunchArg &launch_arg,
                                    const std::vector<DevMemValueHolderPtr> &input_addrs,
                                    const std::vector<DevMemValueHolderPtr> &output_addrs) {
  size_t io_num = input_addrs.size() + output_addrs.size();
  std::vector<ValueHolderPtr> inputs = BuildLaunchCommonHead(launch_arg, io_num);
  inputs.insert(inputs.cend(), input_addrs.cbegin(), input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  inputs.insert(inputs.cend(), launch_arg.input_shapes.cbegin(), launch_arg.input_shapes.cend());
  inputs.insert(inputs.cend(), launch_arg.output_shapes.cbegin(), launch_arg.output_shapes.cend());
  auto ret = BuildLaunchCommonTail(launch_arg, inputs);
  if (ret) {
    return ValueHolder::CreateVoid<ValueHolder>("LaunchMixKernelWithFlag", inputs);
  } else {
    return ValueHolder::CreateVoid<ValueHolder>("LaunchKernelWithFlag", inputs);
  }
}

ValueHolderPtr AtomicLaunchKernelWithHandle(const CommonLaunchArg &launch_arg,
                                            const ValueHolderPtr &stub_func,
                                            const ValueHolderPtr &clean_workspace_indexes,
                                            const std::vector<DevMemValueHolderPtr> &clean_output_addrs,
                                            const ValueHolderPtr &clean_workspace_addrs) {
  size_t io_num = clean_output_addrs.size(); // only clean output addrs
  std::vector<ValueHolderPtr> inputs = BuildLaunchCommonHead(launch_arg, io_num);
  inputs.emplace_back(stub_func);
  inputs.emplace_back(clean_workspace_indexes);
  inputs.insert(inputs.cend(), clean_output_addrs.cbegin(), clean_output_addrs.cend());
  inputs.emplace_back(clean_workspace_addrs);
  return ValueHolder::CreateVoid<ValueHolder>("AtomicLaunchKernelWithHandle", inputs);
}

ValueHolderPtr AtomicLaunchKernelWithFlag(const CommonLaunchArg &launch_arg,
                                          const ValueHolderPtr &clean_workspace_indexes,
                                          const std::vector<DevMemValueHolderPtr> &clean_output_addrs,
                                          const ValueHolderPtr &clean_workspace_addrs) {
  size_t io_num = clean_output_addrs.size();  // only clean output addrs
  std::vector<ValueHolderPtr> inputs = BuildLaunchCommonHead(launch_arg, io_num);
  inputs.emplace_back(clean_workspace_indexes);
  inputs.insert(inputs.cend(), clean_output_addrs.cbegin(), clean_output_addrs.cend());
  inputs.emplace_back(clean_workspace_addrs);
  return ValueHolder::CreateVoid<ValueHolder>("AtomicLaunchKernelWithFlag", inputs);
}

ValueHolderPtr LaunchFFTSPlusTaskNoCopy(const ValueHolderPtr &stream, bg::ValueHolderPtr task_info_para,
    bg::ValueHolderPtr need_launch, bg::ValueHolderPtr dfx_holder, bg::ValueHolderPtr workspaces_addr) {
  std::vector<ValueHolderPtr> inputs = {stream, task_info_para, need_launch, dfx_holder};
  if (workspaces_addr != nullptr) {
    inputs.emplace_back(workspaces_addr);
  }
  return ValueHolder::CreateVoid<ValueHolder>("LaunchFFTSPlusTaskNoCopy", inputs);
}

ValueHolderPtr LaunchStarsKernel(const ValueHolderPtr &sqe_addr,
                                 const ValueHolderPtr &sqe_len,
                                 const ValueHolderPtr &stream) {
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(sqe_addr);
  inputs.emplace_back(sqe_len);
  inputs.emplace_back(stream);
  return ValueHolder::CreateSingleDataOutput("StarsTaskLaunchKernel", inputs);
}
}  // namespace bg
}  // namespace gert
