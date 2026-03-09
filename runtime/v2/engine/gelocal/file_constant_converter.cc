/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include "graph/node.h"
#include "graph_builder/converter_checker.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/checker.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_rt_session.h"
#include "lowering/frame_selector.h"
#include "graph_builder/bg_checker.h"
#include "graph_builder/bg_model_desc.h"
#include "engine/node_converter_utils.h"

namespace gert {
using namespace bg;
namespace {
ge::Status GetFileConstantStorageShapeFromOpdesc(const ge::OpDescPtr &file_constant_opdesc,
                                                 StorageShape &storage_shape) {
  // outputshape需要通过属性获取,实现infershape功能,获取outputshape
  std::vector<int64_t> out_shape;
  std::vector<int64_t> original_out_shape;
  GE_ASSERT_TRUE(ge::AttrUtils::GetListInt(file_constant_opdesc, "shape", out_shape));
  GE_ASSERT_TRUE(ge::AttrUtils::GetListInt(file_constant_opdesc, "original_shape", original_out_shape));

  storage_shape.MutableStorageShape().SetDimNum(out_shape.size());
  for (size_t i = 0U; i < out_shape.size(); i++) {
    storage_shape.MutableStorageShape().SetDim(i, out_shape[i]);
  }
  storage_shape.MutableOriginShape().SetDimNum(original_out_shape.size());
  for (size_t i = 0U; i < original_out_shape.size(); ++i) {
    storage_shape.MutableOriginShape().SetDim(i, original_out_shape[i]);
  }
  return ge::SUCCESS;
}

std::vector<bg::DevMemValueHolderPtr> FileConstantConverter(const ge::NodePtr &node, const LowerInput &lower_input,
                                                            const std::string &file_name,
                                                            const bg::ValueHolderPtr &output_shape) {
  auto builder = [&node, &lower_input, &file_name, &output_shape]() -> std::vector<bg::ValueHolderPtr> {
    auto init_outputs = bg::FrameSelector::OnInitRoot(
        [&node, &lower_input, &file_name, &output_shape]() -> std::vector<bg::ValueHolderPtr> {
          // 在线场景走var mgr內存
          const auto var_name = node->GetName();
          const auto var_id = bg::ValueHolder::CreateConst(var_name.c_str(), var_name.size() + 1U, true);
          const auto rt_session = bg::HolderOnInit(bg::GetRtSession(*lower_input.global_data));
          GE_ASSERT_NOTNULL(var_id);
          GE_ASSERT_NOTNULL(rt_session);

          // 离线场景自行申请内存
          const auto output_sizes = bg::CalcTensorSize(node, {output_shape});
          GE_ASSERT_TRUE(output_sizes.size() == 1U);
          const auto allocator_holder =
              lower_input.global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
          const auto stream_holder = lower_input.global_data->GetStream();
          GE_ASSERT_NOTNULL(allocator_holder);
          GE_ASSERT_NOTNULL(stream_holder);

          // 离线场景路径拼接
          const auto file_constant_weight_dir_holder =
              bg::HolderOnInit(bg::GetFileConstantWeightDir(*lower_input.global_data));
          const auto file_name_holder = bg::ValueHolder::CreateConst(file_name.c_str(), file_name.size() + 1U, true);
          GE_ASSERT_NOTNULL(file_constant_weight_dir_holder);
          GE_ASSERT_NOTNULL(file_name_holder);

          auto out_addr = bg::DevMemValueHolder::CreateSingleDataOutput(
              "FileConstantKernel",
              {rt_session, var_id, allocator_holder, stream_holder, output_sizes[0], file_constant_weight_dir_holder,
               file_name_holder},
              node->GetOpDescBarePtr()->GetStreamId());
          GE_ASSERT_NOTNULL(out_addr);
          out_addr->SetPlacement(kOnDeviceHbm);

          GE_ASSERT_NOTNULL(ValueHolder::CreateVoidGuarder("FreeMemory", out_addr, {}));

          return {out_addr};
        });

    GE_ASSERT_TRUE(init_outputs.size() == 1U);
    return init_outputs;
  };

  int64_t offset_attr = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), "offset", offset_attr);

  auto file_const_holders = lower_input.global_data->GetOrCreateUniqueValueHolder(file_name + ":" + std::to_string(offset_attr), builder);
  std::vector<bg::DevMemValueHolderPtr> return_holder;
  for (const auto &addr : file_const_holders) {
    return_holder.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(addr));
  }
  return return_holder;
}

std::vector<bg::ValueHolderPtr> GetUserDeviceMemoryKernel(LoweringGlobalData *const global_data,
                                                          const ge::NodePtr &node,
                                                          const void *const user_mem, size_t mem_size) {
  auto builder = [&node, user_mem, mem_size]() -> std::vector<bg::ValueHolderPtr> {
    auto init_outputs = bg::FrameSelector::OnInitRoot(
        [&node, user_mem, mem_size]() -> std::vector<bg::ValueHolderPtr> {
          const auto user_mem_holder = bg::ValueHolder::CreateConst(&user_mem, sizeof(void *));
          const auto mem_size_holder = bg::ValueHolder::CreateConst(&mem_size, sizeof(size_t));
          GE_ASSERT_NOTNULL(user_mem_holder);
          GE_ASSERT_NOTNULL(mem_size_holder);

          auto out_addr = bg::DevMemValueHolder::CreateSingleDataOutput(
              "FileConstantUserMemKernel",
              {user_mem_holder, mem_size_holder}, node->GetOpDescBarePtr()->GetStreamId());
          GE_ASSERT_NOTNULL(out_addr);
          return {out_addr};
        });

    GE_ASSERT_TRUE(init_outputs.size() == 1U);
    return init_outputs;
  };
  return global_data->GetOrCreateUniqueValueHolder(node->GetName(), builder);
}

ge::Status GetUserDeviceAddress(const ge::NodePtr &node, LoweringGlobalData *const global_data,
    const std::string &file_path, size_t offset, size_t length, std::vector<bg::DevMemValueHolderPtr> &out_addrs) {
  if (!global_data->IsUserSetFileConstantMem()) {
    return ge::SUCCESS;
  }
  std::string file_constant_file_name;
  std::string file_dir;
  ge::SplitFilePath(file_path, file_dir, file_constant_file_name);
  (void)file_dir;
  const auto user_device_mem = global_data->GetFileConstantMem(file_constant_file_name);
  if (user_device_mem == nullptr) {
    GELOGI("No user device memory found for FileConstant node %s. File name: %s", node->GetNamePtr(),
           file_constant_file_name.c_str());
    return ge::SUCCESS;
  }
  // It's unlikely, since aclmdlSetExternalWeightAddress has already verified the device_mem.
  GE_ASSERT_NOTNULL(user_device_mem->device_mem, "Error: The address set by the user via aclmdlSetExternalWeightAddress"
                    " for the file %s is a null pointer.", file_constant_file_name.c_str());
  const ge::OpDescPtr &op_desc = node->GetOpDesc();
  const auto tensor_desc = op_desc->GetOutputDesc(0);
  int64_t tensor_size = 0;
  GE_ASSERT_SUCCESS(ge::TensorUtils::GetTensorSizeInBytes(tensor_desc, tensor_size),
                    "FileConstant %s get tensor size failed.", node->GetNamePtr());
  const auto weights_size = static_cast<size_t>(tensor_size);

  // The offset is non-zero only when multiple FileConstants share one weight file,
  // which is a rarely used pattern
  GE_ASSERT_TRUE(user_device_mem->mem_size > offset, "mem_size: %zu, offset: %zu", user_device_mem->mem_size, offset);
  if (user_device_mem->mem_size - offset < weights_size) {
    std::string reason =
        "The device memory size set by the user via "
        "aclmdlSetExternalWeightAddress for the external weight file is insufficient. "
        "Required: " +
        std::to_string(weights_size) + " bytes, Provided: " + std::to_string(user_device_mem->mem_size - offset) +
        " bytes. External weight - Shape: [" + tensor_desc.GetShape().ToString().c_str() + "], Data type: [" +
        ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str() + "], Offset: [" +
        std::to_string(offset) + "], File name: [" + file_constant_file_name + "], Node name: [" +
        op_desc->GetNamePtr() + "].";
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"aclmdlSetExternalWeightAddress",
                                                         std::to_string(weights_size).c_str(), reason.c_str()}));

    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param] The device memory size set by the user via "
           "aclmdlSetExternalWeightAddress for the external weight file is insufficient. "
           "Required: %zu bytes, Provided: %zu bytes. External weight - Shape: [%s], Data type: [%s], Offset: [%zu], "
           "File name: [%s], Node name: [%s].", weights_size, user_device_mem->mem_size - offset,
           tensor_desc.GetShape().ToString().c_str(),
           ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(), offset,
           file_constant_file_name.c_str(), op_desc->GetNamePtr());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const auto user_mem = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(user_device_mem->device_mem) + offset);
  GELOGI("FileConstant node %s found user device memory (addr: %p, size: %zu). file name: %s, offset: %zu, length: %zu,"
      " weight size: %zu", op_desc->GetNamePtr(), user_mem, user_device_mem->mem_size,
      file_constant_file_name.c_str(), offset, length, weights_size);

  const auto user_device_mem_holders =
      GetUserDeviceMemoryKernel(global_data, node, user_mem, user_device_mem->mem_size);
  for (const auto &addr : user_device_mem_holders) {
    out_addrs.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(addr));
  }
  return ge::SUCCESS;
}
}  // namespace

LowerResult LoweringFileConstantNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LOWER_REQUIRE_NOTNULL(lower_input.global_data);
  LOWER_REQUIRE_NOTNULL(node);
  const ge::OpDescPtr &op_desc = node->GetOpDesc();
  LOWER_REQUIRE_NOTNULL(op_desc);
  StorageShape storage_shape;
  LOWER_REQUIRE_SUCCESS(GetFileConstantStorageShapeFromOpdesc(op_desc, storage_shape));

  auto output_shape_init_ouput =
      bg::FrameSelector::OnInitRoot([&storage_shape, op_desc]() -> std::vector<bg::ValueHolderPtr> {
        const auto storage_shape_holder =
            NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), storage_shape);
        return {storage_shape_holder};
      });
  CONVERTER_CHECK_HOLDERS_ALL_OK(output_shape_init_ouput, 1U);

  std::string tmp_file_path;
  size_t offset = 0U;
  size_t length = 0U;
  std::map<std::string, std::string> file_id_and_path_map;
  LOWER_REQUIRE_SUCCESS(ge::FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map),
                        "Failed to get FILE_CONSTANT_PATH option.");
  LOWER_REQUIRE_SUCCESS(
      ge::FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, tmp_file_path, offset, length),
      "Failed to get file path.");
  std::vector<bg::DevMemValueHolderPtr> init_outputs;
  LOWER_REQUIRE_SUCCESS(GetUserDeviceAddress(node, lower_input.global_data, tmp_file_path, offset, length,
    init_outputs), "lowering FileConstant node %s failed.", node->GetNamePtr());
  if (init_outputs.empty()) {
    init_outputs =
        FileConstantConverter(node, lower_input, tmp_file_path, bg::HolderOnInit(output_shape_init_ouput[0U]));
  }
  CONVERTER_CHECK_HOLDERS_ALL_OK(init_outputs, 1U);
  std::vector<bg::ValueHolderPtr> order_holders(init_outputs.cbegin(), init_outputs.cend());
  return {HyperStatus::Success(), order_holders, output_shape_init_ouput, init_outputs};
}

REGISTER_NODE_CONVERTER("FileConstant", LoweringFileConstantNode);
}  // namespace gert
