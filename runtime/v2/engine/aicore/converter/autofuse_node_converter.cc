/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_node_converter.h"

#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_tensor.h"
#include "graph_builder/converter_checker.h"
#include "lowering/placement/placed_lowering_result.h"
#include "engine/aicore/fe_rt2_common.h"
#include "engine/aicore/graph_builder/bg_aicore_memory.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/lowering/frame_selector.h"
#include "aicore_compile_results.h"
#include "bg_kernel_launch.h"
#include "aicore_node_converter.h"
#include "node_converter_utils.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "graph_builder/bg_tiling.h"
#include "graph_builder/bg_platform.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/op_so_bin.h"

namespace gert {
namespace {
constexpr char const *kAutofuseLoweringFunc = "kAutoFuseLoweringFunc";

bg::ValueHolderPtr AutofuseLaunch(const ge::NodePtr &node,
                                  const LowerInput &lower_input,
                                  const std::vector<bg::ValueHolderPtr> &tiling_results,
                                  const domi::TaskDef *task_def,
                                  const std::vector<bg::ValueHolderPtr> &output_shapes,
                                  const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
                                  const bg::DevMemValueHolderPtr& workspace_addr) {
  auto global_data = lower_input.global_data;
  // sink bin
  auto node_bin = SinkBinForAicore(node, global_data->FindCompiledResult(node));
  // shapebuffer_addr
  auto shapebuffer_addr = bg::AllocShapeBufferMem(kOnDeviceHbm, node, *(global_data));
  // get qos info
  bg::ValueHolderPtr qos = nullptr;
  if (GetQosInfo(qos) != ge::SUCCESS) {
    return {};
  }
  auto node_info = task_def->kernel_with_handle().node_info() + "/";
  auto node_info_holder = bg::ValueHolder::CreateConst(node_info.c_str(), node_info.size() + 1, true);
  DfxExeArg dfx_exe_arg = GetOpDfxExeArg(node);
  auto dfx_holder = bg::ValueHolder::CreateConst(&dfx_exe_arg, sizeof(dfx_exe_arg));
  auto launch_arg_ref = bg::LaunchKernelWithHandle(
      {
          global_data->GetStream(),
          node_bin,
          tiling_results[static_cast<size_t>(TilingContext::kOutputBlockDim)],
          tiling_results[TilingContext::kOutputScheduleMode],
          tiling_results[TilingContext::kOutputLocalMemorySize],
          workspace_addr,
          shapebuffer_addr,
          qos,
          lower_input.input_shapes,
          output_shapes,
          node,
          global_data,
          dfx_holder,
          tiling_results[static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)],
      },
      tiling_results[TilingContext::kOutputTilingKey],
      node_info_holder,
      lower_input.input_addrs,
      output_addrs);
  FE_ASSERT_NOTNULL(launch_arg_ref);
  return launch_arg_ref;
}

ge::Status CreateSoPathHolder(const ge::NodePtr &node, bg::ValueHolderPtr &so_path_holder) {
  std::string so_path;
  if (!ge::AttrUtils::GetStr(node->GetOpDesc(), "bin_file_path", so_path)) {
    GELOGE(ge::FAILED, "Get 'bin_file_path' from node %s failed.", node->GetName().c_str());
    return ge::FAILED;
  }
  so_path_holder = bg::ValueHolder::CreateConst(so_path.c_str(), so_path.size() + 1, true);

  // 从扩展属性获取bin_file_buffer
  auto graph = ge::NodeUtils::FindRootGraph(*node);
  GE_ASSERT_NOTNULL(graph);
  auto bin_file_buffer = graph->GetExtAttr<std::map<std::string, ge::OpSoBinPtr>>("bin_file_buffer");
  if (bin_file_buffer == nullptr) {
    // bin_file_buffer不存在，走so_path流程
    GELOGD("Not exist bin_file_buffer.");
    return ge::SUCCESS;
  }
  auto buffer = bin_file_buffer->find(so_path);
  if (buffer == bin_file_buffer->end()) {
    GELOGE(ge::FAILED, "Not exist autofuse so in bin_file_buffer, key:%s.", so_path.c_str());
    return ge::FAILED;
  }
  GE_ASSERT_NOTNULL(buffer->second.get());
  uint32_t bin_len = buffer->second.get()->GetBinDataSize();
  if (mmAccess(so_path.c_str()) != EN_OK) {
    // AutofuseSo存在则不重复创建
    std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[bin_len]);
    GE_ASSERT_EOK(memcpy_s(so_bin.get(), bin_len, buffer->second->GetBinData(), bin_len));
    GE_ASSERT_GRAPH_SUCCESS(ge::SaveBinToFile(so_bin.get(), bin_len, so_path.c_str()),
                            "Failed to save bin to file, so_path %s.", so_path.c_str());
  }
  return ge::SUCCESS;
}
}

bg::ValueHolderPtr AutofuseNodeConveter::GetAutofuseHandle(LoweringGlobalData &global_data, const ge::NodePtr &node, GetAutofuseFuncsOutput index) {
  auto builder = [&node]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&node]() -> std::vector<bg::ValueHolderPtr> {
      bg::ValueHolderPtr so_path_holder;
      if (CreateSoPathHolder(node, so_path_holder) != ge::SUCCESS) {
        GELOGE(ge::FAILED, "Failed to create value holder.");
        return {};
      }
      auto so_handle = bg::ValueHolder::CreateDataOutput("GetAutofuseFuncs", {so_path_holder}, static_cast<size_t>(GetAutofuseFuncsOutput::kNum));
      return so_handle;
    });
  };
  auto handles = global_data.GetOrCreateUniqueValueHolder(node->GetName() + "AutofuseSoHanldes", builder);
  GE_ASSERT_TRUE(handles.size() > static_cast<size_t>(index));
  return handles[static_cast<size_t>(index)];
}

std::vector<bg::ValueHolderPtr> AutofuseNodeConveter::GetSymbolInputsWithSize(
    LoweringGlobalData &global_data, const std::vector<bg::ValueHolderPtr> &input_shapes,
    const std::string &graph_name) {
    auto holder_name = graph_name + "_input_shapes_with_size";
    auto builder = [&input_shapes]() -> std::vector<bg::ValueHolderPtr> {
      std::vector<bg::ValueHolderPtr> holders;
      const auto input_num = input_shapes.size();
      auto input_num_holder = bg::ValueHolder::CreateConst(&input_num, sizeof(input_num));
      holders.emplace_back(input_num_holder);
      holders.insert(holders.end(), input_shapes.begin(), input_shapes.end());
      return holders;
    };
    return global_data.GetOrCreateUniqueValueHolder(holder_name, builder);
}

bg::ValueHolderPtr AutofuseNodeConveter::GetAllSymbolNumHolder(
    LoweringGlobalData &global_data, const ge::NodePtr &node) {
  // all_sym_num_handle, 同一个图的all_sym_num相同，可以用一个节点
  // 可能是子图上的融合节点，需要找到根图再获取all_sym_num
  auto graph = ge::NodeUtils::FindRootGraph(*node);
  GE_ASSERT_NOTNULL(graph);
  auto all_sym_num_handle_builder = [&graph]() -> bg::ValueHolderPtr {
    size_t all_sym_num;
    GE_ASSERT_TRUE(ge::AttrUtils::GetInt(graph, "_all_symbol_num", all_sym_num));
    return bg::ValueHolder::CreateConst(&all_sym_num, sizeof(all_sym_num));
  };
  return global_data.GetOrCreateUniqueValueHolder(
      graph->GetName() + "_all_symbol_num_handle", all_sym_num_handle_builder);
}

LowerResult LoweringAutofuseNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LOWER_REQUIRE_NOTNULL(node);
  GELOGD("Begin to do lowering for autofuse node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
  LOWER_REQUIRE_HYPER_SUCCESS(CheckLowerInput(lower_input));

  auto global_data = lower_input.global_data;
  LOWER_REQUIRE_NOTNULL(global_data);
  auto compile_result = global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICore);
  if (task_def == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Not autofuse find AI core task def")),
            {}, {}, {}};
  }
  // infershape
  const auto root_graph = ge::NodeUtils::FindRootGraph(*node);
  LOWER_REQUIRE_NOTNULL(root_graph);
  const auto input_data_shape_handlers = GetOrCreateInputFeeds(global_data, root_graph);
  auto output_shapes = bg::InferStorageShape(node, input_data_shape_handlers, *global_data);
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, lower_input.input_addrs, *(global_data));
  // tiling
  auto assembled_platform_info_holders = bg::AppendCoreTypeToPlatform(node, global_data);
  GE_ASSERT_TRUE(assembled_platform_info_holders.size() == static_cast<size_t>(bg::AssemblePlatformInfoIndex::kNums));
  auto platform_info =
      assembled_platform_info_holders[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)];
  auto launch_arg = bg::AllocRtArg(node, task_def->kernel_with_handle(), bg::kMaxTilingSize);
  CONVERTER_CHECK_HOLDERS_ALL_OK(launch_arg, static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  auto tiling_results =
      bg::Tiling(node, input_data_shape_handlers, output_shapes,
                 {platform_info, *global_data, launch_arg[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)]});
  CONVERTER_CHECK_HOLDERS_ALL_OK(tiling_results, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));

  // launch
  auto workspace_addr = bg::AllocWorkspaceMem(kOnDeviceHbm,
                                              tiling_results[static_cast<size_t>(TilingContext::kOutputWorkspace)],
                                              *global_data);
  auto launch_arg_ref = AutofuseLaunch(node, lower_input, tiling_results,
                                       task_def, output_shapes, output_addrs, workspace_addr);
  for (size_t i = 0; i < lower_input.input_addrs.size(); ++i) {
    auto guarder = lower_input.input_addrs[i]->GetGuarder();
    if (guarder != nullptr) {
      GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(launch_arg_ref, guarder));
    }
  }
  auto free_workspace_holder = bg::FreeWorkspaceMem(kOnDeviceHbm, workspace_addr);
  GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(launch_arg_ref, free_workspace_holder));
  return {HyperStatus::Success(), {launch_arg_ref}, output_shapes, output_addrs};
}

REGISTER_NODE_CONVERTER_PLACEMENT(kAutofuseLoweringFunc, kOnDeviceHbm, LoweringAutofuseNode);
}
