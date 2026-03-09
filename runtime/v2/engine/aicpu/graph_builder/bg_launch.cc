/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_launch.h"
#include <mutex>
#include "framework/common/taskdown_common.h"
#include "register/kernel_registry.h"
#include "common/debug/ge_log.h"
#include "common/omg_util/omg_util.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/value_holder_generator.h"
#include "engine/aicpu/converter/aicpu_callback.h"
#include "graph_builder/bg_condition.h"
#include "graph/utils/node_utils.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
namespace bg {
namespace {
std::mutex cust_op_mutex;
const char *const kSmallShapeHostcpu = "SmallShapeHostcpu";

using CustAICPUKernelPtr = std::shared_ptr<ge::OpKernelBin>;

// 申请index输出的内存
DevMemValueHolderPtr AllocHostCpuOutputMemory(const ge::NodePtr &node, const IoInfo &io_info, const size_t index,
                                              LoweringGlobalData &global_data) {
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(io_info.output_sizes[index]);
  auto allocator_holder = global_data.GetOrCreateAllocator({kOnHost, AllocatorUsage::kAllocNodeWorkspace});
  inputs.emplace_back(allocator_holder);
  // 这里需要input shape/addr 的连接保证控制关系，和后续compute结点的输入相同，使内存申请结点和后续的compute结点是同折叠init、或同不折叠的
  // 若内存申请被折叠，compute不折叠，且compute被冻结，则此时会有精度问题
  inputs.insert(inputs.cend(), io_info.input_shapes.cbegin(), io_info.input_shapes.cend());
  inputs.insert(inputs.cend(), io_info.input_addrs.cbegin(), io_info.input_addrs.cend());
  auto output = DevMemValueHolder::CreateSingleDataOutput("AllocHostCpuOutputMemory", inputs,
                                                          node->GetOpDescBarePtr()->GetStreamId());
  output->SetPlacement(kOnHost);
  return output;
}

// 申请hostcpu结点的所有输出内存
const std::vector<DevMemValueHolderPtr> AllocHostCpuOutputsMemory(const ge::NodePtr &node, const IoInfo &io_info,
                                                                  LoweringGlobalData &global_data) {
  bool is_small_shape = false;
  const auto &op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::GetBool(op_desc, kSmallShapeHostcpu, is_small_shape);
  if (is_small_shape) {
    GELOGI("Op[%s] is small shape host node, alloc mem independent.", node->GetName().c_str());
    std::vector<DevMemValueHolderPtr> output_addrs(io_info.output_sizes.size(), nullptr);
    std::map<size_t, size_t> ref_map;
    if (GetNodeRefMap(node, ref_map) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "Node %s get ref map failed", node->GetName().c_str());
      return output_addrs;
    }

    for (size_t i = 0U; i < io_info.output_sizes.size(); i++) {
      auto iter = ref_map.find(i);
      if (iter == ref_map.end()) {
        output_addrs[i] = AllocHostCpuOutputMemory(node, io_info, i, global_data);
        continue;
      }
      auto ref_input_index = iter->second;
      if (ref_input_index >= io_info.input_addrs.size()) {
        GELOGE(ge::FAILED, "Node %s output %zu ref from input %zu exceed input addrs num %zu",
               node->GetName().c_str(), i, ref_input_index, io_info.input_addrs.size());
        return output_addrs;
      }
      output_addrs[i] = io_info.input_addrs[ref_input_index];
    }
    return output_addrs;
  } else {
    return bg::AllocOutputMemory(kOnHost, node, io_info.output_sizes, io_info.input_addrs, global_data);
  }
}
} // namespace

ValueHolderPtr UpdateAicpuIoAddr(const ValueHolderPtr &args_handler,
                                 const std::vector<DevMemValueHolderPtr> input_addrs,
                                 const std::vector<ValueHolderPtr> output_addrs) {
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(args_handler);
  inputs.insert(inputs.cend(), input_addrs.cbegin(), input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  return ValueHolder::CreateSingleDataOutput("UpdateAicpuIoAddr", inputs);
}

ValueHolderPtr AicpuTfLaunchKernel(const ValueHolderPtr &args_handler,
                                   const ValueHolderPtr &stream,
                                   const ValueHolderPtr &bin_handler,
                                   const ge::NodePtr node) {
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(args_handler);
  inputs.emplace_back(stream);
  inputs.emplace_back(bin_handler);
  const auto &node_type = node->GetType();
  inputs.emplace_back(ValueHolder::CreateConst(node_type.c_str(), node_type.size() + 1U, true));
  return ValueHolder::CreateSingleDataOutput("AicpuLaunchTfKernel", inputs);
}

ValueHolderPtr AicpuCCLaunchKernel(const ValueHolderPtr &args_handler, const ValueHolderPtr &stream,
                                   const ValueHolderPtr &block_dim, const domi::KernelDef &kernel_def,
                                   const ge::OpDescPtr &op_desc, const ValueHolderPtr &ext_info_handler,
                                   const ValueHolderPtr &bin_handle, const ge::NodePtr node) {
  const uint32_t kernel_type = kernel_def.context().kernel_type();	
  // for cust aicpu task
  if (static_cast<ge::ccKernelType>(kernel_type) == ge::ccKernelType::CUST_AI_CPU) {
    const std::lock_guard<std::mutex> lk(cust_op_mutex);
    bool has_so_loaded = false;
    const auto &so_name = kernel_def.so_name();
    AicpuResourceManager::GetInstance().HasLoadedCustAicpuSo(so_name, has_so_loaded);
    if (!has_so_loaded) {
      GELOGI("aicpu cc launch kernel in, op name %s, so name %s", op_desc->GetName().c_str(), so_name.c_str());
      auto launch_cust = FrameSelector::OnInitRoot([&so_name, &op_desc]() -> std::vector<ValueHolderPtr> {
        auto so_name_holder = bg::ValueHolder::CreateConst(so_name.c_str(), so_name.size() + 1, true);
        const CustAICPUKernelPtr aicpu_kernel_ptr =
            op_desc->TryGetExtAttr(ge::OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
        ge::OpKernelBin *aicpu_kernel = aicpu_kernel_ptr.get();
        GE_ASSERT_NOTNULL(aicpu_kernel);
        auto aicpu_kernel_holder = bg::ValueHolder::CreateConst(&aicpu_kernel, sizeof(uintptr_t));
        auto launch_cust_local =
            bg::ValueHolder::CreateSingleDataOutput("LaunchAicpuCustKernel", {aicpu_kernel_holder, so_name_holder});
        bg::ValueHolder::CreateVoidGuarder("ClearAicpuCustKernel", launch_cust_local, {});
        return {launch_cust_local};
      });
    }
  }  
  (void)op_desc;
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(args_handler);
  inputs.emplace_back(stream);
  inputs.emplace_back(block_dim);
  inputs.emplace_back(ValueHolder::CreateConst(&kernel_type, sizeof(kernel_type)));
  // This is for GE dump log
  inputs.emplace_back(ext_info_handler);
  inputs.emplace_back(bin_handle);
  const auto &node_type = node->GetType();
  inputs.emplace_back(ValueHolder::CreateConst(node_type.c_str(), node_type.size() + 1U, true));
  auto launch_cc =  ValueHolder::CreateSingleDataOutput("AicpuLaunchCCKernel", inputs);
  return launch_cc;
}

ValueHolderPtr AicpuHostComputeByCpuKernel(const ge::NodePtr &node, const AicpuArgs &args,
                                           const IoInfo &io_info, LoweringGlobalData &global_data,
                                           std::vector<DevMemValueHolderPtr> &output_addrs) {
  // update ext_info and compute
  const auto &op_desc = node->GetOpDesc();
  auto update_ext = UpdateExtInfo(op_desc, {io_info.input_shapes, io_info.output_shapes}, args.ext_info_handler,
                                  global_data.GetStream());
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(args.args_handler);
  inputs.insert(inputs.cend(), io_info.input_addrs.cbegin(), io_info.input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  ValueHolderPtr compute_holder = ValueHolder::CreateSingleDataOutput("AicpuHostCompute", inputs);
  ValueHolder::AddDependency(args.ext_info_handler, compute_holder);
  if (update_ext != nullptr) {
    ValueHolder::AddDependency(update_ext, compute_holder);
  }
  NodeOutput node_output = {io_info.input_shapes, io_info.output_shapes};
  AicpuCallback(node, args.ext_info_handler, compute_holder, global_data, node_output);
  return compute_holder;
}

ValueHolderPtr AicpuHostExecFuncProcess(const AicpuHostProcFunc &func,
                                        const IoInfo &io_info,
                                        const std::vector<DevMemValueHolderPtr> &output_addrs) {
  std::vector<ValueHolderPtr> inputs;
  inputs.insert(inputs.cend(), io_info.input_shapes.cbegin(), io_info.input_shapes.cend());
  inputs.insert(inputs.cend(), io_info.input_addrs.cbegin(), io_info.input_addrs.cend());
  // func固定为输入最后一个，加参数的话，在前面加。
  auto func_holder = ValueHolder::CreateConst(&func, sizeof(AicpuHostProcFunc));
  inputs.emplace_back(func_holder);
  size_t output_num = io_info.output_shapes.size() + output_addrs.size();
  const std::string exec_func = "AicpuHostExecFunc";
  auto output = ValueHolder::CreateDataOutput(exec_func.c_str(), inputs, output_num);
  size_t idx = 0U;
  for (const auto &shape : io_info.output_shapes) {
    output[idx]->RefFrom(shape);
    ValueHolder::AddDependency(shape, output[idx]);
    ++idx;
  }
  for (const auto &addr : output_addrs) {
    output[idx]->RefFrom(addr);
    ValueHolder::AddDependency(addr, output[idx]);
    ++idx;
  }
  ValueHolderPtr compute_holder = nullptr;
  // almost inevitable
  if (output_num > 0U) {
    compute_holder = output[0];
  }
  return compute_holder;
}

ValueHolderPtr AicpuHostCompute(const ge::NodePtr &node, const AicpuArgs &args, const IoInfo &io_info,
                                LoweringGlobalData &global_data, std::vector<DevMemValueHolderPtr> &output_addrs) {
  output_addrs = AllocHostCpuOutputsMemory(node, io_info, global_data);

  std::string type;
  ge::GetOriginalType(node, type);
  auto aicpu_host_find_func = AicpuResourceManager::GetInstance().GetAicpuHostFindFunc();
  GE_ASSERT_NOTNULL(aicpu_host_find_func);

  AicpuHostProcFunc func = aicpu_host_find_func(type);
  if (func != nullptr) {
    return AicpuHostExecFuncProcess(func, io_info, output_addrs);
  } else {
    return AicpuHostComputeByCpuKernel(node, args, io_info, global_data, output_addrs);
  }
}
}  // namespace bg
}  // namespace gert
