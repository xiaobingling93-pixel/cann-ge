/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/file_constant_utils.h"
#include <sys/file.h>
#include <iomanip>
#include <fstream>
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "common/helper/file_saver.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/math/math_util.h"
#include "common/checker.h"
#include "common/thread_pool.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "runtime/mem.h"
#include "mmpa/mmpa_api.h"
#include "base/err_mgr.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr int64_t kBlockSize = 10485760;
constexpr int64_t kDefaultOffset = 0;
constexpr int32_t kFirstElementIndex = 0;
constexpr int32_t kIndentWidth = 2;
const std::string kBinFileValues = "value_bins";
const std::string kBinIdValue = "value_bin_id";
const std::string kBinFilePathValue = "value_bin_file";
const std::string kTmpWeightDir = "tmp_weight_";
const std::string kMetaFileName = "meta.json";
const std::string kAttrDtype = "dtype";
const std::string kAttrShape = "shape";
constexpr int32_t kIndentationLen = 4;
constexpr int64_t kDefaultThreadNum = 8;

Status ReadJsonFile(const std::string &file_path, nlohmann::json &json_obj) {
  std::ifstream file_stream(file_path);
  GE_CHK_BOOL_RET_STATUS(file_stream.is_open(), FAILED, "Failed to open json file[%s]", file_path.c_str());
  try {
    file_stream >> json_obj;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read json file[%s], err msg: %s", file_path.c_str(), e.what());
    return FAILED;
  }
  GE_CHK_BOOL_RET_STATUS(file_stream.good(), FAILED, "Failed to read json file[%s]", file_path.c_str());
  return SUCCESS;
}

Status WriteJsonFile(const std::string &file_path, const nlohmann::json &json_obj) {
  std::ofstream out_stream(file_path);
  GE_CHK_BOOL_RET_STATUS(out_stream.is_open(), FAILED, "Failed to open json file:%s", file_path.c_str());
  try {
    out_stream << std::setw(kIndentationLen) << json_obj;
  } catch (const std::exception &e) {
    GELOGE(FAILED, "Failed to write json file[%s], err msg: %s", file_path.c_str(), e.what());
    return FAILED;
  }
  GE_CHK_BOOL_RET_STATUS(out_stream.good(), FAILED, "Failed to write json file[%s]", file_path.c_str());
  return SUCCESS;
}
}

void from_json(const nlohmann::json &j, FileIdToFilePath &info) {
  const auto id = j.find(kBinIdValue);
  if (id != j.end()) {
    info.value_bin_file_id = id->get<std::string>();
  }

  const auto file_path = j.find(kBinFilePathValue);
  if (file_path != j.end()) {
    info.value_bin_file_path = file_path->get<std::string>();
  }
}

void from_json(const nlohmann::json &j, OptionInfo &option_info) {
  const auto it = j.find(kBinFileValues);
  if (it != j.end()) {
    option_info = it->get<OptionInfo>();
  }
}

Status FileConstantUtils::GetFileIdToPathMapFromOption(std::map<std::string, std::string> &file_id_to_path_map) {
  std::string opt;
  (void)GetContext().GetOption(FILE_CONSTANT_PATH, opt);
  if (opt.empty()) {
    GELOGW("[Check][Param] Failed to get file constant path.");
    return SUCCESS;
  }
  GELOGI("source string = %s.", opt.c_str());

  nlohmann::json options;
  try {
    options = nlohmann::json::parse(opt);
  } catch (nlohmann::json::exception &ex) {
    const auto readable_name = ge::GetContext().GetReadableName(FILE_CONSTANT_PATH);
    std::string reason = "it is not a valid json string, exception: " + std::string(ex.what());
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", 
        std::vector<const char *>({"value", "parameter", "reason"}),
        std::vector<const char *>({opt.c_str(), readable_name.c_str(), reason.c_str()})
    );
    GELOGE(FAILED, "Failed to parse option FILE_CONSTANT_PATH, which [%s] is invalid, err:%s", opt.c_str(), ex.what());
    return FAILED;
  }

  for (const nlohmann::json &single_json : options) {
    GELOGD("Parsing op[%d], jsonStr = %s.", kFirstElementIndex, single_json.dump(kIndentWidth).c_str());
    std::vector<FileIdToFilePath> multi_info;
    multi_info = single_json.get<std::vector<FileIdToFilePath>>();
    for (const auto &single_info : multi_info) {
      GELOGD("get single info, file id is %s, file path is %s.", single_info.value_bin_file_id.c_str(),
             single_info.value_bin_file_path.c_str());
      (void)file_id_to_path_map.insert(
        std::pair<std::string, std::string>(single_info.value_bin_file_id, single_info.value_bin_file_path));
    }
  }
  return SUCCESS;
}

Status FileConstantUtils::CopyOneWeightFromFileWithFilehandler(const void *const curr_dev_ptr, const std::string &file_path,
                                                              const size_t offset, const size_t file_constant_size,
                                                              size_t &left_size, std::ifstream &ifs) {
  GE_CHECK_GE(left_size, file_constant_size);
  ifs.clear();
  (void)ifs.seekg(static_cast<int64_t>(offset), ifs.beg);
  size_t used_memory = 0U;
  std::string compress_nodes;
  compress_nodes.reserve(static_cast<size_t>(kBlockSize));
  Status ret = SUCCESS;
  while ((!ifs.eof()) && (used_memory != file_constant_size)) {
    (void)ifs.read(&compress_nodes[0U], kBlockSize);
    auto copy_len_once = static_cast<size_t>(ifs.gcount());
    if ((file_constant_size - used_memory) < copy_len_once) {
      copy_len_once = file_constant_size - used_memory;
    }
    if (left_size < (used_memory + copy_len_once)) {
      GELOGE(GRAPH_FAILED, "copy failed for lack memory, free size is %zu, need memroy is %zu.", left_size,
             used_memory + copy_len_once);
      REPORT_INNER_ERR_MSG("E19999", "copy failed for lack memory, free size is %zu, need memroy is %zu.", left_size,
                        used_memory + copy_len_once);
      ret = FAILED;
      break;
    }

    GELOGI("copy %zu bytes to memory, path = %s.", copy_len_once, file_path.c_str());
    void *const cur_dev_ptr = reinterpret_cast<void *>(PtrToValue(curr_dev_ptr) + used_memory);
    const rtError_t rts_error =
      rtMemcpy(cur_dev_ptr, left_size - used_memory, &compress_nodes[0U], copy_len_once, RT_MEMCPY_HOST_TO_DEVICE);
    if (rts_error != RT_ERROR_NONE) {
      GELOGE(GRAPH_FAILED, "copy failed, result code = %d.", rts_error);
      REPORT_INNER_ERR_MSG("E19999", "copy failed, result code = %d.", rts_error);
      ret = RT_ERROR_TO_GE_STATUS(rts_error);
      break;
    }
    used_memory += copy_len_once;
  }
  left_size -= used_memory;
  GELOGI("used memory is %zu.", used_memory);
  return ret;
}

Status FileConstantUtils::CopyOneWeightFromFile(const void *const curr_dev_ptr, const std::string &file_path,
                                                const size_t offset, const size_t file_constant_size,
                                                size_t &left_size) {
  GE_CHECK_GE(left_size, file_constant_size);
  const std::string real_path = RealPath(file_path.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  if (!ifs.is_open()) {
    GELOGE(FAILED, "[Open][File] %s failed.", file_path.c_str());
    (void)REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                       std::vector<const char *>({file_path.c_str(), "Open file failed"}));
    return FAILED;
  }
  const Status ret = CopyOneWeightFromFileWithFilehandler(curr_dev_ptr, real_path, offset, file_constant_size, left_size, ifs);
  ifs.close();
  return ret;
}

Status FileConstantUtils::GetFilePath(const OpDescPtr &op_desc,
                                      const std::map<std::string, std::string> &file_id_to_path_map,
                                      std::string &file_path, size_t &offset, size_t &length) {
  const auto &fileconstant_info = GetFileConstantInfo(op_desc);
  if (!fileconstant_info.weight_path.empty()) {
    file_path = fileconstant_info.weight_path;
    offset = fileconstant_info.weight_offset;
    length = fileconstant_info.weight_length;
    return SUCCESS;
  }
  offset = 0U;
  length = 0U;
  file_path = "";
  const std::string* file_path_ptr = AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_PATH);
  if (file_path_ptr != nullptr && !file_path_ptr->empty()) {
    file_path = *file_path_ptr;
    return SUCCESS;
  }
  const std::string* file_id_ptr = AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID);
  GE_CHK_BOOL_RET_STATUS(file_id_ptr != nullptr, FAILED, "Failed to get filed id from attr");
  GE_CHK_BOOL_RET_STATUS(!file_id_ptr->empty(), FAILED, "The file path and file id are empty.");
  const auto it = file_id_to_path_map.find(*file_id_ptr);
  if (it == file_id_to_path_map.end()) {
    GELOGW("Failed to get file path of file id:%s", file_id_ptr->c_str());
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(!(it->second.empty()), FAILED, "File path is empty.");
  file_path = it->second;
  return SUCCESS;
}

FileConstantInfo FileConstantUtils::GetFileConstantInfo(const OpDescPtr &op_desc) {
  FileConstantInfo fileconstant_info;
  const std::string* weight_path_ptr = AttrUtils::GetStr(op_desc, ATTR_NAME_LOCATION);
  if (weight_path_ptr != nullptr) {
    fileconstant_info.weight_path = *weight_path_ptr;
  }
  int64_t attr_value = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_OFFSET, attr_value);
  fileconstant_info.weight_offset = static_cast<size_t>(attr_value);
  int64_t attr_length = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_LENGTH, attr_length);
  fileconstant_info.weight_length = static_cast<size_t>(attr_length);
  return fileconstant_info;
}

Status FileConstantUtils::GetExternalWeightDir(const ge::ModelData &model_data, string &file_constant_weight_dir) {
  if (!model_data.weight_path.empty()) {
    std::string weight_real_path = ge::RealPath(model_data.weight_path.c_str());
    GE_ASSERT_TRUE(!weight_real_path.empty());
    file_constant_weight_dir = weight_real_path + "/";
    GELOGI("Get external weight path from model_data.weight_path: %s", file_constant_weight_dir.c_str());
    return SUCCESS;
  }
  return GetExternalWeightDirFromOmPath(model_data.om_path, file_constant_weight_dir);
}

Status FileConstantUtils::GetExternalWeightDirFromOmPath(const std::string &om_path, string &file_constant_weight_dir) {
  if (om_path.empty()) {
    GELOGW("Om path is empty, thus file constant weight dir is empty.");
    return ge::SUCCESS;
  }
  const std::string om_real_path = ge::RealPath(om_path.c_str());
  GELOGD("Get OM path[%s], real path[%s].", om_path.c_str(), om_real_path.c_str());
  std::string om_dir;
  std::string om_name;
  ge::SplitFilePath(om_real_path, om_dir, om_name);
  GELOGD("OM dir is[%s], om name is[%s].", om_dir.c_str(), om_name.c_str());
  GE_ASSERT_TRUE(!om_name.empty());
  file_constant_weight_dir = om_dir.append("/weight/");
  GELOGI("Get external weight path %s from model_data.om_path: %s",
    file_constant_weight_dir.c_str(), om_path.c_str());
  return ge::SUCCESS;
}

Status FileConstantUtils::SetExternalPath(const OpDescPtr &op_desc, const std::string &weight_dir) {
  const auto &fileconstant_info = GetFileConstantInfo(op_desc);
  std::string file_name = fileconstant_info.weight_path;
  const size_t offset = fileconstant_info.weight_offset;
  const size_t length = fileconstant_info.weight_length;
  if (file_name.empty()) {
    return SUCCESS;
  }
  if (file_name.rfind('/') != std::string::npos) {
    file_name = file_name.substr(file_name.rfind('/') + 1UL);
  }
  const std::string file_path = weight_dir + file_name;
  // refresh file constant location to new weight_dir
  SetFileConstantPath(op_desc, file_path, static_cast<int64_t>(offset), static_cast<int64_t>(length));
  GELOGD("Set external path success, file path:%s", file_path.c_str());
  return SUCCESS;
}

Status FileConstantUtils::SetExternalPath(const ComputeGraphPtr &compute_graph, const std::string &weight_dir) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetType() != FILECONSTANT) {
      continue;
    }
    GE_ASSERT_SUCCESS(SetExternalPath(op_desc, weight_dir), "Op: %s set external path failed.",
                      op_desc->GetName().c_str());
  }
  return SUCCESS;
}

void FileConstantUtils::GetValidFullPath(const std::string &dir_name, const std::string &file_name,
                                         std::string &full_name) {
  full_name = dir_name;
  if (!dir_name.empty() && (dir_name.back() != '/')) {
    full_name = full_name.append("/");
  }
  full_name = full_name.append(file_name);
}

Status FileConstantUtils::ReadExternalWeightFromFile(const std::string &file_path, const size_t offset,
                                                     const size_t file_length, char_t *const bin_buff) {
  const std::string real_path = RealPath(file_path.c_str());
  GE_CHK_BOOL_RET_STATUS(!real_path.empty(), FAILED, "Failed to get real path of %s", file_path.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  GE_CHK_BOOL_RET_STATUS(ifs.is_open(), FAILED, "Read file %s failed.", real_path.c_str());
  (void)ifs.seekg(0, std::ifstream::end);
  const size_t act_file_len = static_cast<size_t>(ifs.tellg());
  const size_t pos = offset + file_length;
  GE_CHECK_LE(pos, act_file_len);
  ifs.clear();
  (void)ifs.seekg(static_cast<int64_t>(offset), ifs.beg);
  (void)ifs.read(bin_buff, static_cast<int64_t>(file_length));
  GE_CHK_BOOL_RET_STATUS(ifs.good(), FAILED, "read file %s failed.", real_path.c_str());
  ifs.close();
  return SUCCESS;
}

Status FileConstantUtils::ConvertFileConstToConst(const NodePtr &node) {
  GE_ASSERT_TRUE(node->GetType() == FILECONSTANT, "%s[%s] is not fileconstant", node->GetName().c_str(),
                 node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto output_desc = op_desc->MutableOutputDesc(0U);
  GE_CHECK_NOTNULL(output_desc);
  DataType out_type = ge::DT_UNDEFINED;
  (void)AttrUtils::GetDataType(op_desc, kAttrDtype, out_type);
  output_desc->SetDataType(out_type);
  int64_t weight_size = 0;
  GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*output_desc, weight_size), "Failed to get weight size");
  const auto &fileconstant_info = GetFileConstantInfo(op_desc);
  std::string file_path = fileconstant_info.weight_path;
  if (file_path.empty()) {
    const std::string* file_path_ptr = AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_PATH);
    if (file_path_ptr == nullptr || file_path_ptr->empty()) {
      return SUCCESS;
    }
    file_path = *file_path_ptr;
  }
  const size_t offset = fileconstant_info.weight_offset;
  const size_t length = fileconstant_info.weight_length;
  const size_t file_length = (length == 0U ? static_cast<size_t>(weight_size) : length);
  GE_CHK_STATUS_RET(CheckUint64AddOverflow(offset, file_length), "offset add length overlow uint64");
  const auto bin_buff = MakeUnique<char_t[]>(file_length);
  GE_CHECK_NOTNULL(bin_buff);
  GE_CHK_STATUS_RET(ReadExternalWeightFromFile(file_path, offset, file_length, bin_buff.get()),
                    "Failed to read external weight from file:%s", file_path.c_str());
  const auto &weight = MakeShared<GeTensor>(*output_desc, reinterpret_cast<uint8_t *>(bin_buff.get()), file_length);
  GE_CHECK_NOTNULL(weight);
  OpDescUtilsEx::SetType(op_desc, CONSTANT);
  GE_ASSERT_TRUE(AttrUtils::SetShareTensor(op_desc, ATTR_NAME_WEIGHTS, *weight));
  (void)op_desc->DelAttr(ATTR_NAME_OFFSET);
  (void)op_desc->DelAttr(ATTR_NAME_LENGTH);
  (void)op_desc->DelAttr(ATTR_NAME_LOCATION);
  GELOGI("Convert node:%s from file constant to const success.", node->GetName().c_str());
  return SUCCESS;
}

Status FileConstantUtils::ConvertFileConstToConst(const ComputeGraphPtr &compute_graph) {
  ThreadPool thread_pool("ge_fcst2cst", static_cast<uint32_t>(kDefaultThreadNum), false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() != FILECONSTANT) {
      continue;
    }
    const auto &error_manager_context = error_message::GetErrMgrContext();
    auto fut = thread_pool.commit([node, error_manager_context]() -> Status {
      error_message::SetErrMgrContext(error_manager_context);
      GE_CHK_STATUS_RET_NOLOG(ConvertFileConstToConst(node));
      return SUCCESS;
    });
    (void)fut_rets.emplace_back(std::move(fut));
  }
  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to convert fileconstant to const, graph:%s", compute_graph->GetName().c_str());
  }
  return SUCCESS;
}

ConstNodeWeightHashMap FileConstantUtils::GetAllConstNodesAndWeightHash(const ComputeGraphPtr &compute_graph) {
  ConstNodeWeightHashMap const_to_weight_hash_map;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (!NodeUtils::IsConst(*node)) {
      continue;
    }
    std::string hash_value;
    const std::string* hash_value_ptr = AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256);
    if (hash_value_ptr == nullptr || hash_value_ptr->empty()) {
      continue;
    }
    hash_value = *hash_value_ptr;
    const auto &weights = OpDescUtils::MutableWeights(node);
    GE_ASSERT_TRUE(!weights.empty(), "Failed to get weight of const node:%s", node->GetName().c_str());
    const auto &weight = weights[0];
    GE_ASSERT_NOTNULL(weight, "Weight is null, node:%s", node->GetName().c_str());
    const_to_weight_hash_map[node] = std::make_pair(weight, hash_value);
  }
  return const_to_weight_hash_map;
}

Status FileConstantUtils::SaveWeightToFileWithReuse(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                                    const std::string &weight_dir, FileConstantMeta &meta) {
  errno = 0;
  const char_t *value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_MAX_COMPILE_CORE_NUMBER, value);
  const int64_t thread_num = ((value != nullptr) && (value[0U] != '\0')) ?
    std::strtol(value, nullptr, 10) : kDefaultThreadNum;
  GE_ASSERT_TRUE(errno == 0, "strtol failed, value: %s", value);
  GELOGI("Start to save weight to file, thread num[%ld]", thread_num);
  ThreadPool thread_pool("ge_savweigh", static_cast<uint32_t>(thread_num), false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &const_and_weight_hash : const_to_weight_hash_map) {
    const auto &const_name = const_and_weight_hash.first->GetName();
    const auto &weight_hash = const_and_weight_hash.second.second;
    const std::string file_name = "weight_" + weight_hash;
    std::string weight_path;
    GetValidFullPath(weight_dir, file_name, weight_path);
    const bool weight_file_exist = (mmAccess(weight_path.c_str()) == EN_OK);
    if ((meta.hash_to_weight_file.count(weight_hash) > 0U) && weight_file_exist) {
      GELOGI("Reuse existing weight file: [%s]. No need to save again.", weight_path.c_str());
      continue;
    }

    const auto &weight = const_and_weight_hash.second.first;
    const auto &data = weight->GetData().GetData();
    const auto size = weight->GetData().GetSize();
    meta.hash_to_weight_file[weight_hash] = weight_path;
    const auto &error_manager_context = error_message::GetErrMgrContext();
    auto fut = thread_pool.commit([const_name, weight_path, data, size, error_manager_context]() -> Status {
      error_message::SetErrMgrContext(error_manager_context);
      GE_ASSERT_SUCCESS(FileSaver::SaveToFile(weight_path, data, size),
                        "Failed to save weight to file, node[%s], weight size:%zu", const_name.c_str(), size);
      return SUCCESS;
    });
    (void)fut_rets.emplace_back(std::move(fut));
  }
  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to save weight to file");
  }
  return SUCCESS;
}

// 使用流式写入避免全部权重加载到内存，支持大模型场景
Status FileConstantUtils::SaveWeightToOneFileWithReuse(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                                    const std::string &weight_dir, FileConstantMeta &meta) {
  // 预检查：计算需要写入的新权重数量
  size_t new_weights_count = 0;
  ComputeGraphPtr root_graph = nullptr;
  for (const auto &const_and_weight_hash : const_to_weight_hash_map) {
    const auto &weight_hash = const_and_weight_hash.second.second;
    if (root_graph == nullptr && const_and_weight_hash.first != nullptr) {
      root_graph = NodeUtils::FindRootGraph(*const_and_weight_hash.first);
    }
    if (meta.hash_to_weight_file.find(weight_hash) == meta.hash_to_weight_file.end()) {
      ++new_weights_count;
    }
  }

  if (new_weights_count == 0) {
    GELOGI("All %zu weights already exist in meta, skip writing", const_to_weight_hash_map.size());
    return SUCCESS;
  }

  GELOGI("Converting %zu const nodes to file constants (all-in-one mode), %zu are new weights",
         const_to_weight_hash_map.size(), new_weights_count);

  std::string model_file_name_prefix;
  std::string graph_name;
  if (root_graph != nullptr) {
    graph_name = root_graph->GetName();
    (void)AttrUtils::GetStr(root_graph, ATTR_MODEL_FILE_NAME_PREFIX, model_file_name_prefix);
  }

  std::string om_name = StringUtils::GetFileName(model_file_name_prefix);
  auto pos = om_name.rfind('.');
  if (pos != std::string::npos) {
    om_name = om_name.substr(0, pos);
  }
  const std::string weight_file_name = om_name.empty()
                                         ? (graph_name.empty() ? "weight_combined" : (graph_name + "_weight_combined"))
                                         : (om_name + "_weight_combined");
  std::string weight_path;
  GetValidFullPath(weight_dir, weight_file_name, weight_path);

  std::ofstream ofs(weight_path, std::ios::binary | std::ios::app | std::ios::out);
  if (!ofs.is_open()) {
    GELOGE(FAILED, "Failed to open weight file for writing: %s", weight_path.c_str());
    return FAILED;
  }

  (void)ofs.seekp(0, std::ios::end);
  size_t offset = static_cast<size_t>(ofs.tellp());

  size_t new_weights_written = 0;
  size_t reused_weights_count = 0;
  GE_CHK_STATUS_RET(SaveWeightsAndMetadata(const_to_weight_hash_map, meta, ofs, weight_path, offset,
                                            new_weights_written, reused_weights_count));
  ofs.close();
  GELOGI("Successfully saved %zu new weights to single file (reused: %zu), total file offset: %zu bytes",
         new_weights_written, reused_weights_count, offset);

  return SUCCESS;
}

Status FileConstantUtils::SaveWeightsAndMetadata(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                                  FileConstantMeta &meta, std::ofstream &ofs,
                                                  const std::string &weight_path, size_t &offset,
                                                  size_t &new_weights_written, size_t &reused_weights_count) {
  new_weights_written = 0UL;
  reused_weights_count = 0UL;
  for (const auto &const_and_weight_hash : const_to_weight_hash_map) {
    const auto &weight = const_and_weight_hash.second.first;
    const auto &weight_hash = const_and_weight_hash.second.second;

    if (meta.hash_to_weight_file.find(weight_hash) != meta.hash_to_weight_file.end()) {
      ++reused_weights_count;
      continue;
    }

    const uint8_t *weight_data_ptr = weight->GetData().data();
    const size_t size = weight->GetData().size();

    // 记录元数据
    meta.hash_to_weight_file[weight_hash] = weight_path;
    meta.hash_to_weight_offset[weight_hash] = offset;

    // 写入权重数据和填充
    GE_CHK_STATUS_RET(WriteWeightWithPadding(ofs, weight_data_ptr, size, offset));

    ++new_weights_written;
  }
  return SUCCESS;
}

Status FileConstantUtils::WriteWeightWithPadding(std::ofstream &ofs, const uint8_t *weight_data_ptr,
                                                 const size_t size, size_t &offset) {
  // 写入权重数据
  (void)ofs.write(reinterpret_cast<const char*>(weight_data_ptr), static_cast<std::streamsize>(size));
  if (!ofs.good()) {
    GELOGE(FAILED, "Failed to write weight data to file");
    return FAILED;
  }

  // 写入填充数据以对齐512字节边界
  constexpr uint aligns = 512;
  const size_t padding = MemSizeAlign(size, aligns) - size;
  if (padding > 0) {
    static constexpr std::array<char, 512> zeros = {'\0'};  // 预分配512字节的零值缓冲区
    const size_t padding_blocks = padding / aligns;
    const size_t padding_remainder = padding % aligns;

    // 批量写入512字节块
    size_t i = 0;
    while (i < padding_blocks) {
      (void)ofs.write(zeros.data(), static_cast<std::streamsize>(aligns));
      ++i;
    }
    // 写入剩余字节
    if (padding_remainder > 0) {
      (void)ofs.write(zeros.data(), static_cast<std::streamsize>(padding_remainder));
    }

    if (!ofs.good()) {
      GELOGE(FAILED, "Failed to write padding data to file");
      return FAILED;
    }
  }

  // 更新偏移量（当前写入位置）
  if (offset > (std::numeric_limits<size_t>::max() - size - padding)) {
    GELOGE(FAILED, "Failed to update offset");
    return FAILED;
  }
  offset += size + padding;
  return SUCCESS;
}

Status FileConstantUtils::ConvertToFileConstants(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                                 const std::string &weight_dir, FileConstantMeta &meta, const bool all_in_one) {
  if (all_in_one) {
    GE_CHK_STATUS_RET(SaveWeightToOneFileWithReuse(const_to_weight_hash_map, weight_dir, meta),
                      "Failed to save weight to one file");
  } else {
    GE_CHK_STATUS_RET(SaveWeightToFileWithReuse(const_to_weight_hash_map, weight_dir, meta),
                      "Failed to save weight to file");
  }
  for (const auto &const_and_weight_hash : const_to_weight_hash_map) {
    const auto &const_node = const_and_weight_hash.first;
    auto const_op = const_node->GetOpDesc();
    const auto &weight = const_and_weight_hash.second.first;
    const auto weight_length = weight->GetData().GetSize();
    const auto &weight_hash = const_and_weight_hash.second.second;
    const std::string new_name = const_node->GetName() + "_" + weight_hash;
    const_op->SetName(new_name);
    weight->ClearData();
    OpDescUtilsEx::SetType(const_op, FILECONSTANT);
    (void)const_op->DelAttr(ATTR_NAME_WEIGHTS);
    const auto &output_desc = const_op->GetOutputDesc(0U);
    const auto &weight_path = meta.hash_to_weight_file.at(weight_hash);
    const int64_t offset = all_in_one ? static_cast<int64_t>(meta.hash_to_weight_offset.at(weight_hash)) : kDefaultOffset;
    SetFileConstantPath(const_op, weight_path, offset, static_cast<int64_t>(weight_length));
    (void)AttrUtils::SetDataType(const_op, kAttrDtype, output_desc.GetDataType());
    (void)AttrUtils::SetListInt(const_op, kAttrShape, output_desc.GetShape().GetDims());
    (void)AttrUtils::SetListInt(const_op, "original_shape", output_desc.GetOriginShape().GetDims());
    GELOGI("Convert node:%s from const to file constant success, save path[%s], offset[%d]",
      const_node->GetName().c_str(), weight_path.c_str(), offset);
  }
  return SUCCESS;
}

Status FileConstantUtils::ConvertConstToFileConst(const ComputeGraphPtr &compute_graph, bool all_in_one) {
  const auto &const_to_weight_hash_map = GetAllConstNodesAndWeightHash(compute_graph);
  if (const_to_weight_hash_map.empty()) {
    GELOGI("Can not find valid const nodes on graph:%s, skip conversion", compute_graph->GetName().c_str());
    return SUCCESS;
  }
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  GE_CHECK_NOTNULL(external_weight_manager);
  const std::string &file_const_dir = external_weight_manager->GetWeightPath();
  GE_CHK_BOOL_RET_STATUS(!file_const_dir.empty(), FAILED, "file constant dir is empty.");
  GE_CHK_STATUS_RET(external_weight_manager->CreateWeightPath(), "Failed to create directory:%s.",
                    file_const_dir.c_str());
  std::string meta_path;
  GetValidFullPath(file_const_dir, kMetaFileName, meta_path);
  int32_t fd = -1;
  bool is_file_locked = false;
  const ScopeGuard fd_guard([&fd, &is_file_locked]() {
    if (is_file_locked) {
      (void)flock(fd, LOCK_UN);
    }
    if (fd != -1) {
      (void)mmClose(fd);
      fd = -1;
    }
  });
  const bool meta_file_exist = (mmAccess(meta_path.c_str()) == EN_OK);
  constexpr int32_t flags = M_RDWR | M_CREAT;
  constexpr MODE mode = static_cast<MODE>(M_UMASK_USRREAD | M_UMASK_USRWRITE);
  fd = mmOpen2(meta_path.c_str(), flags, mode);
  GE_CHK_BOOL_RET_STATUS(fd != -1, FAILED, "open file[%s] failed, error=%d, error msg=%s.", meta_path.c_str(),
                         mmGetErrorCode(), GetErrorNumStr(mmGetErrorCode()).c_str());
  const int32_t file_lock_ret = flock(fd, LOCK_EX);
  GE_CHK_BOOL_RET_STATUS(file_lock_ret == 0, FAILED, "lock file[%s] failed, ret=%d", meta_path.c_str(), file_lock_ret);
  is_file_locked = true;
  auto &meta = external_weight_manager->MutableMetaFile();
  if (meta_file_exist) {
    nlohmann::json meta_json;
    GE_CHK_STATUS_RET(ReadJsonFile(meta_path, meta_json), "read meta json failed.");
    meta = meta_json.get<FileConstantMeta>();
  }

  GE_CHK_STATUS_RET_NOLOG(ConvertToFileConstants(const_to_weight_hash_map, file_const_dir, meta, all_in_one));
  const nlohmann::json out_json(meta);
  GE_CHK_STATUS_RET(WriteJsonFile(meta_path, out_json), "save file constant meta failed.");
  return SUCCESS;
}

Status FileConstantUtils::ConvertConstToFileConst(const NodePtr &node) {
  GE_ASSERT_TRUE(NodeUtils::IsConst(*node), "%s[%s] is not fileconstant", node->GetName().c_str(),
                 node->GetType().c_str());
  const auto &weights = OpDescUtils::MutableWeights(node);
  if (weights.empty() || weights[0]->GetTensorDesc().GetShape().IsEmptyTensor()) {
    GELOGW("Node:%s weight is null or empty tensor", node->GetName().c_str());
    return SUCCESS;
  }
  const auto &tensor = weights[0];
  const auto &data = tensor->GetData().GetData();
  const auto size = tensor->GetData().GetSize();
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  GE_CHECK_NOTNULL(external_weight_manager);
  const std::string &weight_dir = external_weight_manager->GetWeightPath();
  GE_CHK_STATUS_RET(external_weight_manager->CreateWeightPath(), "Failed to create directory:%s.", weight_dir.c_str());
  std::string weight_path;
  GetValidFullPath(weight_dir, GetRegulatedName(node->GetName()), weight_path);
  GE_ASSERT_SUCCESS(FileSaver::SaveToFile(weight_path, data, size),
                    "Failed to save weight to file, node[%s], weight size:%zu", node->GetName().c_str(), size);
  tensor->ClearData();
  auto const_op = node->GetOpDesc();
  OpDescUtilsEx::SetType(const_op, FILECONSTANT);
  (void)const_op->DelAttr(ATTR_NAME_WEIGHTS);
  const auto &output_desc = const_op->GetOutputDesc(0U);
  SetFileConstantPath(const_op, weight_path, kDefaultOffset, static_cast<int64_t>(size));
  (void)AttrUtils::SetDataType(const_op, kAttrDtype, output_desc.GetDataType());
  (void)AttrUtils::SetListInt(const_op, kAttrShape, output_desc.GetShape().GetDims());
  (void)AttrUtils::SetListInt(const_op, "original_shape", output_desc.GetOriginShape().GetDims());
  GELOGI("Convert node:%s from const to file constant success.", node->GetName().c_str());
  return SUCCESS;
}

Status FileConstantUtils::ChangeFilePath(const ComputeGraphPtr &compute_graph, const std::string &om_path) {
  std::map<std::string, std::string> old_file_to_new_file;
  GE_ASSERT_SUCCESS(ChangeFilePathAttr(compute_graph, om_path, old_file_to_new_file),
                    "Chnage file path attribute for graph %s failed.", compute_graph->GetName().c_str());
  return MoveFilePath(old_file_to_new_file);
}

std::string FileConstantUtils::GetTmpWeightDir(const int32_t pid, const uint64_t session_id) {
  std::string dir = kTmpWeightDir + std::to_string(pid) + "_" + std::to_string(session_id) + "/";
  std::string ascend_work_path;
  (void)GetAscendWorkPath(ascend_work_path);
  if (!ascend_work_path.empty()) {
    dir = ascend_work_path + "/" + dir;
  }
  return dir;
}

void FileConstantUtils::SetFileConstantPath(const OpDescPtr &op_desc, const std::string &file_path,
                                            const int64_t offset, const int64_t length) {
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, offset);
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, length);
  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_path);
}

Status FileConstantUtils::ChangeFilePathAttr(const ComputeGraphPtr &compute_graph, const std::string &om_path,
                                             std::map<std::string, std::string> &old_file_to_new_file) {
  std::string origin_dir;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == FILECONSTANT) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const auto &fileconstant_info = GetFileConstantInfo(op_desc);
      const auto &file_path = fileconstant_info.weight_path;
      const size_t offset = fileconstant_info.weight_offset;
      const size_t length = fileconstant_info.weight_length;
      if (file_path.empty()) {
        continue;
      }
      const std::string real_path = RealPath(file_path.c_str());
      GE_CHK_BOOL_RET_STATUS(!real_path.empty(), FAILED, "Failed to get real path of %s", file_path.c_str());
      auto pos = real_path.find(kTmpWeightDir);
      if (pos == std::string::npos) {
        continue;
      }
      if (origin_dir.empty()) {
        pos = real_path.find('/', pos);
        GE_CHK_BOOL_RET_STATUS(pos != std::string::npos, FAILED, "File path:%s is invalid.", real_path.c_str());
        origin_dir = real_path.substr(0, pos);
      }
      std::string file_name = StringUtils::GetFileName(real_path);
      GE_CHK_BOOL_RET_STATUS(!file_name.empty(), FAILED, "The file name is empty.");
      std::string path = om_path;
      const char_t *const om_dir = mmDirName(&path[0]);
      GE_CHECK_NOTNULL(om_dir);
      const std::string om_weight_path = std::string(om_dir) + "/weight/" + file_name;
      SetFileConstantPath(op_desc, file_name, static_cast<int64_t>(offset), static_cast<int64_t>(length));
      old_file_to_new_file[real_path] = om_weight_path;
      GELOGD("Node:%s changes file path attribute success.", node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status FileConstantUtils::MoveFilePath(const std::map<std::string, std::string> &old_file_to_new_file) {
  std::set<std::string> old_dirs;
  for (const auto &file_path : old_file_to_new_file) {
    const auto &old_path = file_path.first;
    const auto &new_path = file_path.second;
    GE_ASSERT_TRUE(!old_path.empty(), "Old fileconstant path can not be null");
    GE_ASSERT_TRUE(!new_path.empty(), "New fileconstant path can not be null");

    auto pos = old_path.rfind('/');
    GE_ASSERT_TRUE(pos != std::string::npos, "File path:%s is invalid.", old_path.c_str());
    const std::string origin_dir = old_path.substr(0, pos);
    (void)old_dirs.insert(origin_dir);
    pos = new_path.rfind('/');
    GE_ASSERT_TRUE(pos != std::string::npos, "File path:%s is invalid.", new_path.c_str());
    const std::string om_weight_dir = new_path.substr(0, pos);
    GE_ASSERT_TRUE(CreateDirectory(om_weight_dir) == 0, "Failed to create directory:%s.", om_weight_dir.c_str());
    GE_ASSERT_TRUE(std::rename(old_path.c_str(), new_path.c_str()) == 0,
                   "Failed to change path from %s to %s.", old_path.c_str(), new_path.c_str());
    GELOGD("Change file %s to %s success.", old_path.c_str(), new_path.c_str());
  }
  for (const auto &old_dir : old_dirs) {
    GE_ASSERT_TRUE(mmRmdir(old_dir.c_str()) == 0, "Failed to remove dir:%s.", old_dir.c_str());
  }
  return SUCCESS;
}

Status FileConstantUtils::RefreshRelativePath(const ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == FILECONSTANT) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const auto &fileconstant_info = GetFileConstantInfo(op_desc);
      if (!fileconstant_info.weight_path.empty()) {
        const std::string &file_name = StringUtils::GetFileName(fileconstant_info.weight_path);
        (void)AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_name);
        GELOGI("Success to refresh relative file path:%s for fileconstant node:%s.", file_name.c_str(),
               node->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge