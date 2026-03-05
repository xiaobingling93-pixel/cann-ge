/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/om2_package_helper.h"
#include "framework/common/helper/model_save_helper_factory.h"
#include "common/ge_common/ge_types.h"
#include "common/ge_common/string_util.h"
#include "common/helper/om2/zip_archive.h"
#include "common/helper/om2/om2_package_contants.h"
#include "common/helper/om2/json_file.h"
#include "common/om2/codegen/om2_codegen.h"
#include "common/om2/codegen/om2_codegen_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace {
constexpr auto kAttrKernelName = "_kernelname";
struct ModelIoNodes {
  std::map<uint32_t, OpDescPtr> input_ops;
  std::vector<OpDescPtr> output_ops;
};

struct ModelMetaExtraInfo {
  JsonFile::json dynamic_output_shape = JsonFile::json::array();
  JsonFile::json dynamic_batch_info = JsonFile::json::array();
  JsonFile::json user_designate_shape_order = JsonFile::json::array();
  int32_t dynamic_type = 0;
};

Status GetDynamicBatchInfo(const OpDescPtr &op_desc, JsonFile::json &batch_info,
                           JsonFile::json &user_designate_shape_order, int32_t &dynamic_type) {
  uint32_t batch_num = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGI("Not multi-batch Node: %s", op_desc->GetName().c_str());
    return SUCCESS;
  }
  batch_info.clear();

  (void)AttrUtils::GetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type);
  std::vector<std::string> user_designate_shape_order_vec;
  (void)AttrUtils::GetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order_vec);
  for (const auto &s : user_designate_shape_order_vec) {
    user_designate_shape_order.push_back(s);
  }
  for (uint32_t i = 0U; i < batch_num; ++i) {
    std::vector<int64_t> batch_shape;
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::GetListInt(op_desc, attr_name, batch_shape)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) fail", attr_name.c_str(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) fail", attr_name.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      batch_info.clear();
      return FAILED;
    }
    batch_info.push_back(batch_shape);
  }
  return SUCCESS;
}

Status FillTensorInfo(JsonFile &tensor_info, const ConstGeTensorDescPtr &tensor_desc, const int64_t tensor_size) {
  tensor_info.Set("shape", tensor_desc->GetShape().GetDims())
      .Set("data_type", TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()))
      .Set("format", TypeUtils::FormatToSerialString(tensor_desc->GetFormat()))
      .Set("size", tensor_size);
  std::vector<std::pair<int64_t, int64_t>> range;
  if (tensor_desc->GetShapeRange(range) == SUCCESS) {
    tensor_info.Set("shape_range", range);
  }
  return SUCCESS;
}

Status CollectModelIoNodes(const ComputeGraphPtr &graph, ModelIoNodes &io_nodes) {
  uint32_t data_index = 0U;
  const std::set<std::string> kDataOpTypes{DATA, AIPPDATA, ANN_DATA};
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (kDataOpTypes.count(op_desc->GetType()) > 0U) {
      uint32_t tmp_index = data_index++;
      if (AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, tmp_index)) {
        GELOGD("Get new data index %u, old index is %u", tmp_index, data_index - 1U);
      }
      io_nodes.input_ops[tmp_index] = op_desc;
      GELOGD("Find input node [%s], index [%u]", node->GetNamePtr(), tmp_index);
      continue;
    }
    if (op_desc->GetType() == NETOUTPUT) {
      io_nodes.output_ops.push_back(op_desc);
      GELOGD("Find output node [%s]", node->GetNamePtr());
    }
  }
  return SUCCESS;
}

Status BuildInputJsonArray(const std::map<uint32_t, OpDescPtr> &input_ops, JsonFile::json &input_json_array) {
  for (const auto &[index, op_desc] : input_ops) {
    const auto &tensor_desc = op_desc->GetInputDescPtr(0);
    GE_ASSERT_NOTNULL(tensor_desc);
    int64_t input_size = 0;
    if (AttrUtils::GetInt(*op_desc->GetOutputDescPtr(0U), ATTR_NAME_SPECIAL_INPUT_SIZE, input_size) &&
        (input_size > 0)) {
      GELOGI("data[%s] output has special size [%" PRId64 "]", op_desc->GetName().c_str(), input_size);
    } else {
      GE_CHK_STATUS_RET(TensorUtils::GetSize(*tensor_desc, input_size), "[Get][InputSize] failed in op: %s.",
                        op_desc->GetName().c_str());
    }
    JsonFile input_info;
    input_info.Set("name", op_desc->GetName()).Set("index", index);
    GE_ASSERT_SUCCESS(FillTensorInfo(input_info, tensor_desc, input_size));
    std::vector<int64_t> model_input_dims;
    if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
      (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    } else {
      model_input_dims = tensor_desc->GetShape().GetDims();
    }
    input_info.Set("shape_v2", model_input_dims);
    input_json_array.push_back(input_info.Raw());
  }
  return SUCCESS;
}

Status BuildOutputJsonArray(const std::vector<OpDescPtr> &output_ops, const std::vector<std::string> &out_node_name,
                            JsonFile::json &output_json_array, ModelMetaExtraInfo &extra_info) {
  for (const auto &op_desc : output_ops) {
    const auto out_size = op_desc->GetInputsSize();
    const auto src_name = op_desc->GetSrcName();
    const auto src_index = op_desc->GetSrcIndex();
    GE_ASSERT_TRUE(src_name.size() >= out_size && src_index.size() >= out_size);
    for (size_t i = 0UL; i < out_size; ++i) {
      JsonFile output_info;
      std::string output_name;
      if (out_size == out_node_name.size()) {
        const bool contains_colon = out_node_name[i].find(':') != std::string::npos;
        output_name = contains_colon ? out_node_name[i] : (out_node_name[i] + ":" + std::to_string(src_index[i]));
      } else {
        output_name =
            std::string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
      }
      const auto &tensor_desc = op_desc->GetInputDescPtr(i);
      GE_ASSERT_NOTNULL(tensor_desc);
      int64_t tensor_size = 0;
      if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size) && (tensor_size > 0)) {
        GELOGI("netoutput[%s] [%zu]th input has special size [%" PRId64 "]", op_desc->GetName().c_str(), i, tensor_size);
      } else {
        (void)TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
      }
      output_info.Set("name", output_name).Set("index", i);
      GE_ASSERT_SUCCESS(FillTensorInfo(output_info, tensor_desc, tensor_size));
      output_json_array.push_back(output_info.Raw());
    }
    std::vector<std::string> shape_info;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      for (const auto &s : shape_info) {
        extra_info.dynamic_output_shape.push_back(s);
      }
    }
    if (op_desc->GetType() == CASE) {
      GE_ASSERT_SUCCESS(
          GetDynamicBatchInfo(op_desc, extra_info.dynamic_batch_info, extra_info.user_designate_shape_order,
                              extra_info.dynamic_type));
    }
  }
  return SUCCESS;
}

void FillModelMetaInfo(const GeModelPtr &ge_model, const JsonFile::json &input_json_array,
                       const JsonFile::json &output_json_array, const ModelMetaExtraInfo &extra_info,
                       JsonFile &model_meta_info) {
  model_meta_info.Set("inputs", input_json_array);
  model_meta_info.Set("outputs", output_json_array);
  model_meta_info.Set("dynamic_output_shape", extra_info.dynamic_output_shape);
  model_meta_info.Set("dynamic_batch_info", extra_info.dynamic_batch_info);
  model_meta_info.Set("user_designate_shape_order", extra_info.user_designate_shape_order);
  model_meta_info.Set("dynamic_type", extra_info.dynamic_type);
  model_meta_info.Set("name", ge_model->GetName());
}
}  // namespace
Status Om2PackageHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                           ModelBufferData &model, const bool is_unknown_shape) {
  GE_ASSERT_NOTNULL(ge_root_model, "[OM2] ge_root_model is nullptr");
  GE_ASSERT_TRUE(!output_file.empty(), "[OM2] Empty path of output file is invalid");
  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_ASSERT_TRUE(!name_to_ge_model.empty(), "[OM2] No subgraphs found in ge_root_model");

  if (!is_unknown_shape) {
    auto &model_root = name_to_ge_model.begin()->second;
    return SaveToOmModel(model_root, output_file, model, ge_root_model);
  }

  // todo 动态 shape 场景暂时不支持
  GELOGE(FAILED, "[OM2] Unknown shape models are not supported for .om2 format conversion");
  REPORT_INNER_ERR_MSG("E19999", "[OM2] Unknown shape models are not supported for .om2 format conversion");
  return FAILED;
}

Status Om2PackageHelper::SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                                       ModelBufferData &model, const GeRootModelPtr &ge_root_model) {
  (void)model;
  GE_ASSERT_NOTNULL(ge_model, "ge_model is nullptr");
  GE_ASSERT_TRUE(!output_file.empty(), "[OM2] Empty path of the output file is invalid");

  auto zip_writer = MakeShared<ZipArchiveWriter>(output_file);
  GE_ASSERT_NOTNULL(zip_writer);
  GE_ASSERT_TRUE(zip_writer->IsMemFileOpened());

  // 1. Codegen and shared library
  GE_ASSERT_SUCCESS(SaveCodegenArtifacts(zip_writer, ge_model, 0UL));
  // 2. Save constants/weights
  GE_ASSERT_SUCCESS(SaveConstants(zip_writer, ge_model, 0UL));
  // 3. Save TBE kernels
  GE_ASSERT_SUCCESS(SaveTbeKernels(zip_writer, ge_model));
  // 4. Save meta infos of the compiled model
  GE_ASSERT_SUCCESS(SaveModelInfo(zip_writer, ge_model, 0UL));
  // 5. Save archive manifest
  GE_ASSERT_SUCCESS(SaveManifest(zip_writer, ge_root_model));

  // Complete packaging
  GE_ASSERT_TRUE(zip_writer->SaveModelDataToFile());
  GELOGI("[OM2] Successfully created OM2 model");

  return SUCCESS;
}

void Om2PackageHelper::SetSaveMode(const bool val) {
  is_offline_ = val;
}

Status Om2PackageHelper::SaveConstants(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                                       const size_t model_index) {
  GELOGI("[OM2] Begin to save model constants");
  if (ge_model->GetWeightSize() > 0) {
    const auto constant_file_name = FormatOm2Path("%s%s%zu", OM2_CONSTANTS_DIR, OM2_CONSTANTS_FILE_PREFIX, model_index);
    GE_ASSERT_TRUE(
        zip_writer->WriteBytes(constant_file_name, ge_model->GetWeightData(), ge_model->GetWeightSize(), false));
  }

  JsonFile json_file;
  json_file.Set("weight_size", ge_model->GetWeightSize());
  const std::string constants_json_str = json_file.Dump();
  const auto constants_config_path =
      FormatOm2Path(OM2_CONSTANTS_CONFIG_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(zip_writer->WriteBytes(constants_config_path, constants_json_str.data(), constants_json_str.size(), false));

  GELOGI("[OM2] Successfully saved model constants, total size = %zu bytes", ge_model->GetWeightSize());
  return SUCCESS;
}

Status Om2PackageHelper::SaveTbeKernels(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model) {
  GELOGI("[OM2] Begin to save TBE kernels");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);
  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  const auto kernel_bin_dir = FormatOm2Path(OM2_KERNELS_DIR_FORMAT, "npu_arch");
  // todo 这里可以直接解析tbe_kernel_store.Data(), 无需遍历节点
  std::unordered_set<std::string> added_kernels;
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    std::string kernel_name;
    (void)AttrUtils::GetStr(node->GetOpDesc(), kAttrKernelName, kernel_name);
    const auto node_name = node->GetName();
    auto kernel_bin = tbe_kernel_store.FindKernel(kernel_name);
    if ((kernel_bin != nullptr) && (added_kernels.count(kernel_name) == 0)) {
      GELOGD("[OM2] Save kernel for node [%s], kernel name is [%s]", node_name.c_str(), kernel_name.c_str());
      const auto entry_path = kernel_bin_dir + Om2CodegenUtils::GetKernelNameWithExtension(kernel_name);
      GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_path, kernel_bin->GetBinData(), kernel_bin->GetBinDataSize(), false));
      added_kernels.insert(kernel_name);
    }
  }
  GELOGI("[OM2] Successfully saved all TBE kernels");
  return SUCCESS;
}

Status Om2PackageHelper::SaveModelInfo(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                                       const size_t model_index) {
  GELOGI("Begin to saved model manifest");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);
  ModelIoNodes io_nodes;
  GE_ASSERT_SUCCESS(CollectModelIoNodes(graph, io_nodes));

  JsonFile model_meta_info;
  auto input_json_array = JsonFile::json::array();
  GE_ASSERT_SUCCESS(BuildInputJsonArray(io_nodes.input_ops, input_json_array));

  auto output_json_array = JsonFile::json::array();
  std::vector<std::string> out_node_name;
  (void)AttrUtils::GetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, out_node_name);
  ModelMetaExtraInfo extra_info;
  GE_ASSERT_SUCCESS(BuildOutputJsonArray(io_nodes.output_ops, out_node_name, output_json_array, extra_info));
  FillModelMetaInfo(ge_model, input_json_array, output_json_array, extra_info, model_meta_info);

  const auto model_meta_info_str = model_meta_info.Dump();
  const auto model_meta_entry_path = FormatOm2Path(OM2_MODEL_META_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(zip_writer->WriteBytes(model_meta_entry_path, model_meta_info_str.data(), model_meta_info_str.size(), false));
  GELOGI("Successfully saved model manifest");
  return SUCCESS;
}

Status Om2PackageHelper::SaveManifest(std::shared_ptr<ZipArchiveWriter> &zip_writer,
                                      const GeRootModelPtr &ge_root_model) {
  JsonFile json_file;
  json_file.Set<std::string>(OM2_ARCHIVE_VERSION, OM2_ARCHIVE_VERSION_VALUE);
  json_file.Set(OM2_MODEL_NUM, ge_root_model->GetSubgraphInstanceNameToModel().size());
  json_file.Set(OM2_ATC_COMMAND, domi::GetContext().atc_cmdline);
  const auto manifest_str = json_file.Dump();
  GE_ASSERT_TRUE(zip_writer->WriteBytes(OM2_MANIFEST_PATH, manifest_str.data(), manifest_str.size(), false));
  return SUCCESS;
}

Status Om2PackageHelper::SaveCodegenArtifacts(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                                              const size_t model_index) {
  GELOGI("[OM2] Begin to save codegen artifacts");
  Om2Codegen codegen;

  std::vector<std::string> output_files;
  GE_ASSERT_SUCCESS(codegen.Om2CodegenAndCompile(ge_model, output_files));
  GE_ASSERT_TRUE(!output_files.empty());
  const std::string artifacts_base_dir = FormatOm2Path(OM2_RUNTIME_DIR_FORMAT, std::to_string(model_index).c_str());
  for (const auto &src_file_path : output_files) {
    const std::string entry_name = artifacts_base_dir + StringUtils::GetFileName(src_file_path);
    GE_ASSERT_TRUE(zip_writer->WriteFile(entry_name, src_file_path, true), "Failed to write file [%s]",
                   src_file_path.c_str());
  }
  GELOGI("[OM2] Successfully saved all codegen artifacts");
  return SUCCESS;
}

REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_OM2, Om2PackageHelper);
}  // namespace ge
