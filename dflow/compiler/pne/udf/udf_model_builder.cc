/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_model_builder.h"

#include <regex>
#include <cstdio>
#include <string>
#include <sstream>
#include <cctype>
#include <sys/file.h>
#include <openssl/sha.h>
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "dflow/base/model/model_deploy_resource.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "udf_attr_utils.h"
#include "dflow/base/utils/process_utils.h"
#include "graph/utils/op_type_utils.h"
#include "mmpa/mmpa_api.h"

namespace {
const std::string kUdfAttrNameBinPath = "bin_path";
const std::string kUdfAttrNameFuncName = "func_name";
const std::string kUdfAttrNameProcessorType = "_processor_type";
const std::string kUdfAttrNameOsVersion = "_os_version";
const std::string kUdfAttrNameReleaseLib = "_dflow_process_point_release_pkg";
const std::string kUdfAttrNameFinalLocation = "_dflow_final_location";
const std::set<std::string> kUdfBaseAttrNames = {kUdfAttrNameBinPath, kUdfAttrNameFuncName, kUdfAttrNameProcessorType,
                                                 kUdfAttrNameOsVersion};
const std::string kUdfOpTypeFlowFunc = "FlowFunc";
const std::string kUdfBuildInBinName = "libbuilt_in_flowfunc.so";
const std::string kUdfBuildInFuncNamePrefix = "_BuiltIn_";
constexpr const char *kAttrNameDataFlowHeavyLoad = "_dflow_heavy_load";
constexpr const char *kCpuNumAttrName = "__cpu_num";
constexpr const char *kBufferConfig = "_user_buf_cfg";

std::string GetUdfModelNameByFileName(const std::string &om_file) {
  const auto start_idx = om_file.find_last_of("/") + 1;
  const auto end_idx = om_file.find_last_of(".");
  if (start_idx >= end_idx) {
    return "";
  }
  return om_file.substr(start_idx, end_idx - start_idx);
}
}  // namespace

namespace ge {
Status UdfModelBuilder::Build(UdfModel &udf_model) const {
  auto graph = udf_model.GetRootGraph();
  GE_CHECK_NOTNULL(graph);
  GE_DUMP(graph, "Before_UDF_Build");
  udf_model.SetModelName(graph->GetName());
  udf_model.SetModelType(PNE_ID_UDF);
  bool has_udf = false;
  const bool no_tiling = true;
  bool is_heavy_load = false;
  (void)AttrUtils::GetBool(graph, kAttrNameDataFlowHeavyLoad, is_heavy_load);
  GELOGD("model[%s] attr[%s]=%d.", graph->GetName().c_str(), kAttrNameDataFlowHeavyLoad,
         static_cast<int32_t>(is_heavy_load));
  auto buffer_configs =graph->TryGetExtAttr<std::vector<CompileConfigJson::BufCfg>>(kBufferConfig, {});
  for (const NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const std::string &op_type = op_desc->GetType();
    if (OpTypeUtils::IsDataNode(op_type)) {
      (void)AttrUtils::SetBool(op_desc->MutableOutputDesc(0), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling);
      continue;
    }
    if (op_type == NETOUTPUT) {
      for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
        (void)AttrUtils::SetBool(op_desc->MutableInputDesc(i), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling);
      }
      continue;
    }
    GE_CHK_BOOL_RET_STATUS(op_type == FLOWFUNC, FAILED, "Unsupport this op[%s], only support op[%s].", op_type.c_str(),
                           FLOWFUNC);
    GE_CHK_BOOL_RET_STATUS(!has_udf, FAILED, "The graph[%s] has more than one udf op, only support one udf op.",
                           graph->GetName().c_str());
    has_udf = true;
    GE_CHK_STATUS_RET(BuildFlowFuncOp(op_desc, is_heavy_load, graph, buffer_configs, udf_model),
                      "Build flow func op failed.");
  }
  GE_DUMP(graph, "After_UDF_Build");
  return SUCCESS;
}

Status UdfModelBuilder::BuildFlowFuncOp(const OpDescPtr &op_desc, const bool is_heavy_load,
                                        const ComputeGraphPtr &graph,
                                        const std::vector<CompileConfigJson::BufCfg> &buffer_configs,
                                        UdfModel &udf_model) const {
  std::string resource_type;
  if (ge::AttrUtils::GetStr(op_desc, kUdfAttrNameFinalLocation, resource_type)) {
    GELOGI("Get final location[%s] from Node[%s].", resource_type.c_str(), op_desc->GetName().c_str());
  } else {
    (void)AttrUtils::GetStr(op_desc, kUdfAttrNameProcessorType, resource_type);
  }
  GE_CHK_STATUS_RET(SetDeployResource(op_desc, udf_model, is_heavy_load, resource_type),
                    "Failed to set deploy resource for graph[%s]", graph->GetName().c_str());
  udf::UdfModelDef &udf_model_def = udf_model.MutableUdfModelDef();
  udf::UdfDef *udf_def = udf_model_def.add_udf_def();
  GE_CHECK_NOTNULL(udf_def);
  for (const auto &cfg : buffer_configs) {
    auto *cfg_proto = udf_def->add_user_buf_cfg();
    GE_CHECK_NOTNULL(cfg_proto);
    cfg_proto->set_total_size(cfg.total_size);
    cfg_proto->set_blk_size(cfg.blk_size);
    cfg_proto->set_max_buf_size(cfg.max_buf_size);
    cfg_proto->set_page_type(cfg.page_type);
    GELOGI("Set user buffer config for UdfDef from op[%s].{ total_size:%u, blk_size:%u, max_buf_size:%u, page_type:%s }",
        op_desc->GetName().c_str(), cfg.total_size, cfg.blk_size, cfg.max_buf_size, cfg.page_type.c_str());
  }
  GE_CHK_STATUS_RET(BuildUdfDef(op_desc, *udf_def), "Failed to build UdfDef from op[%s].",
                    op_desc->GetName().c_str());
  if (op_desc->HasAttr(dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS)) {
    std::vector<std::string> invoke_keys;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetListStr(op_desc, dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS, invoke_keys),
                           FAILED, "Failed to get attr[%s] from op[%s].", dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS,
                           op_desc->GetName().c_str());
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(graph, dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS, invoke_keys), FAILED,
                           "Failed to set attr[%s] to graph[%s].", dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS,
                           graph->GetName().c_str());
    }
  GE_CHK_STATUS_RET(GenReleasePackage(udf_model, op_desc, resource_type, graph), "Set release lib to udf failed.");
  return SUCCESS;
}

Status UdfModelBuilder::SetDeployResource(const OpDescPtr &op_desc, UdfModel &udf_model, bool is_heavy_load,
                                          const std::string &resource_type) const {
  std::shared_ptr<ModelDeployResource> resource = MakeShared<ModelDeployResource>();
  GE_CHECK_NOTNULL(resource);
  resource->resource_type = resource_type;
  resource->is_heavy_load = is_heavy_load;
  int32_t cpu_num = -1;
  if (AttrUtils::GetInt(op_desc, kCpuNumAttrName, cpu_num)) {
    resource->resource_list[kCpuNumAttrName] = static_cast<int64_t>(cpu_num);
  }
  udf_model.SetDeployResource(resource);
  return SUCCESS;
}

Status UdfModelBuilder::BuildUdfDef(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const {
  udf_def.set_name(op_desc->GetName());
  GE_CHK_STATUS_RET(SetFuncNameAndInputOutputMaps(op_desc, udf_def), "Failed to set func_name and inputs/outputs map "
      "for UdfDef from op[%s].", op_desc->GetName().c_str());
  std::string release_pkg_path;
  if (AttrUtils::GetStr(op_desc, kUdfAttrNameReleaseLib, release_pkg_path)) {
    if (release_pkg_path.empty()) {
      const auto &func_inputs_map = udf_def.func_inputs_map();
      if (func_inputs_map.empty()) {
        const auto &func_name = udf_def.func_name();
        GE_CHK_BOOL_RET_STATUS(func_name.find("_BuiltIn_") == 0, FAILED, "func[%s] release pkg path is empty, "
            "but not built in func, op[%s].", func_name.c_str(), op_desc->GetName().c_str());
      } else {
        for (const auto &func_input_map : func_inputs_map) {
          const auto &func_name = func_input_map.first;
          GE_CHK_BOOL_RET_STATUS(func_name.find("_BuiltIn_") == 0, FAILED, "func[%s] release pkg path is empty, "
              "but not built in func, op[%s].", func_name.c_str(), op_desc->GetName().c_str());
        }
      }
      udf_def.set_bin_name(kUdfBuildInBinName);
    } else {
      GELOGD("Generate release info later.");
    }
  } else {
    GELOGE(FAILED, "Failed to get release lib from op[%s].", op_desc->GetName().c_str());
    return FAILED;
  }

  const auto attrs_map = op_desc->GetAllAttrs();
  auto *udf_attrs = udf_def.mutable_attrs();
  GE_CHECK_NOTNULL(udf_attrs);
  for (const auto &iter : attrs_map) {
    if (kUdfBaseAttrNames.find(iter.first) != kUdfBaseAttrNames.end()) {
      continue;
    }
    GE_CHK_STATUS_RET(SetAttr(iter.first, iter.second, *udf_attrs), "Failed to set %s attr for UdfDef from op[%s].",
                      iter.first.c_str(), op_desc->GetName().c_str());
  }
  return SUCCESS;
}

Status UdfModelBuilder::SetBin(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const {
  std::string bin_path;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetStr(op_desc, kUdfAttrNameBinPath, bin_path), FAILED,
                         "Failed to get %s attr from op[%s].", kUdfAttrNameBinPath.c_str(), op_desc->GetName().c_str());
  char_t *bin_buff = nullptr;
  int32_t length = 0;
  GE_CHK_BOOL_RET_STATUS(ReadBytesFromBinaryFile(bin_path.c_str(), &bin_buff, length), FAILED,
                         "Failed to read bin from %s.", bin_path.c_str());
  udf_def.set_bin(bin_buff, length);
  delete[] bin_buff;
  return SUCCESS;
}

Status UdfModelBuilder::SetBinName(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const {
  std::string bin_path;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetStr(op_desc, kUdfAttrNameBinPath, bin_path), FAILED,
                         "Failed to get %s attr from op[%s].", kUdfAttrNameBinPath.c_str(), op_desc->GetName().c_str());
  auto const pos = bin_path.find_last_of('/');
  if (pos == std::string::npos) {
    udf_def.set_bin_name(bin_path);
  } else {
    udf_def.set_bin_name(bin_path.substr(pos + 1));
  }
  return SUCCESS;
}

Status UdfModelBuilder::SetMultiFuncInputOutputMaps(const std::string &op_name,
                                                    const std::vector<NamedAttrs> &funcs_attr,
                                                    udf::UdfDef &udf_def) const {
  auto func_inputs_map = udf_def.mutable_func_inputs_map();
  auto func_outputs_map = udf_def.mutable_func_outputs_map();
  std::vector<int64_t> inputs_idx;
  std::vector<int64_t> outputs_idx;

  for (const auto &func_attr : funcs_attr) {
    const auto &multi_func_name = func_attr.GetName();
    // maybe no input, but udf use input map to load flow func, so must add key to input map.
    auto &func_map_inputs = (*func_inputs_map)[multi_func_name];
    // input index map
    inputs_idx.clear();
    GE_CHK_STATUS_RET(
        func_attr.GetItem(dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX).GetValue<std::vector<int64_t>>(inputs_idx),
        "Failed to get attr[%s] from FlowFunc[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX, op_name.c_str());
    for (auto input_index : inputs_idx) {
      func_map_inputs.add_num(input_index);
    }
    GELOGD("Set func name[%s]'s input index[%s] for pp[%s] success.", multi_func_name.c_str(),
           ToString(inputs_idx).c_str(), op_name.c_str());
    // output index
    outputs_idx.clear();
    (void)func_attr.GetItem(dflow::ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX).GetValue<std::vector<int64_t>>(outputs_idx);
    if (!outputs_idx.empty()) {
      for (auto output_index : outputs_idx) {
        (*func_outputs_map)[multi_func_name].add_num(output_index);
      }
      GELOGD("Set func name[%s]'s output index[%s] for pp[%s] success.", multi_func_name.c_str(),
             ToString(outputs_idx).c_str(), op_name.c_str());
    } else {
      GELOGI("FlowFunc[%s] has no attr[%s]", op_name.c_str(), dflow::ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX);
    }
  }
  return SUCCESS;
}

Status UdfModelBuilder::SetStreamInputFuncNames(const std::string &op_name, const std::vector<NamedAttrs> &funcs_attr,
                                                udf::UdfDef &udf_def) const {
  for (const auto &func_attr : funcs_attr) {
    if (!func_attr.HasAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT)) {
      continue;
    }
    bool stream_input = false;
    const auto &func_name = func_attr.GetName();
    GE_CHK_STATUS_RET(func_attr.GetItem(dflow::ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT).GetValue<bool>(stream_input),
                      "Failed to get attr[%s] from pp[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT,
                      op_name.c_str());
    if (stream_input) {
      udf_def.add_stream_input_func_name(func_name);
      GELOGI("Set stream input func name[%s] for pp[%s] success.", func_name.c_str(), op_name.c_str());
    }
  }
  return SUCCESS;
}

Status UdfModelBuilder::SetFuncNameAndInputOutputMaps(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const {
  std::string func_name;

  if (op_desc->HasAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST)) {
    std::vector<NamedAttrs> funcs_attr;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetListNamedAttrs(op_desc, dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST, funcs_attr),
                           FAILED, "Failed to get attr[%s] from op[%s].", dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST,
                           op_desc->GetName().c_str());
    GE_CHK_BOOL_RET_STATUS(funcs_attr.size() >= 1, FAILED, "The op[%s] should has at least one func.",
                           op_desc->GetName().c_str());
    func_name = funcs_attr[0].GetName();
    if (funcs_attr.size() > 1) {
      GE_CHK_STATUS_RET(SetMultiFuncInputOutputMaps(op_desc->GetName(), funcs_attr, udf_def),
                        "Set multi funcs input/output maps for pp[%s] failed.", op_desc->GetName().c_str());
    }
    GE_CHK_STATUS_RET(SetStreamInputFuncNames(op_desc->GetName(), funcs_attr, udf_def),
                      "Set multi funcs stream input func names for pp[%s] failed.", op_desc->GetName().c_str());
  }
  (void)AttrUtils::GetStr(op_desc, kUdfAttrNameFuncName, func_name);
  GE_CHK_BOOL_RET_STATUS(!func_name.empty(), FAILED, "Failed to get %s attr from op[%s].", kUdfAttrNameFuncName.c_str(),
                         op_desc->GetName().c_str());
  udf_def.set_func_name(func_name);
  return SUCCESS;
}

Status UdfModelBuilder::GetAndCheckAttrs(const OpDescPtr &op_desc, const ComputeGraphPtr &graph,
    std::string &release_pkg_path, std::string &cache_release_info, std::string &om_model_file) const {
  (void)AttrUtils::GetStr(op_desc, kUdfAttrNameReleaseLib, release_pkg_path);
  GE_CHK_BOOL_RET_STATUS(!release_pkg_path.empty(), PARAM_INVALID,
                         "release pkg path can not be empty in user define udf.");
  std::regex dir_pattern(R"([A-Za-z0-9./+\-_]+)");
  std::smatch match_result;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(release_pkg_path, match_result, dir_pattern), PARAM_INVALID,
                          "Invalid release path: %s", release_pkg_path.c_str());
  GE_CHECK_NOTNULL(graph);
  cache_release_info = graph->TryGetExtAttr<std::string>("_cache_graph_info_for_data_flow_cache", "");
  om_model_file = graph->TryGetExtAttr<std::string>("_cache_graph_udf_om_file", "");
  if (!cache_release_info.empty() && om_model_file.empty()) {
    GELOGE(FAILED, "Current cache maybe old version cache. Please generate cache base on current compiler package.");
    return FAILED;
  }
  return SUCCESS;
}

Status UdfModelBuilder::GenReleasePackageForUserDefineFunc(UdfModel &udf_model, const OpDescPtr &op_desc,
    const std::string &resource_type, const ComputeGraphPtr &graph) const {
  std::string release_pkg_path;
  std::string cache_release_info;
  std::string om_model_file;
  GE_CHK_STATUS_RET(GetAndCheckAttrs(op_desc, graph, release_pkg_path, cache_release_info, om_model_file),
      "Check and get op/graph attrs failed.");
  bool skip_check = false;
  skip_check = graph->TryGetExtAttr<bool>("_cache_skip_release_info_check", false);
  if (skip_check) {
    return ProcessForCache(udf_model, om_model_file);
  }
  const std::string compile_lock = release_pkg_path + "/../build/compile_lock";
  int32_t fd = mmOpen(compile_lock.c_str(), M_WRONLY);
  if (fd == -1) {
    const int32_t error_code = mmGetErrorCode();
    GELOGE(FAILED, "Failed to open file[%s], error msg[%s].", compile_lock.c_str(),
           GetErrorNumStr(error_code).c_str());
    return FAILED;
  }
  {
    bool is_locked = false;
    ScopeGuard auto_close([&fd, &is_locked] {
      if (is_locked) {
        (void)flock(fd, LOCK_UN);
      }
      (void)close(fd);
      fd = -1;
    });
    // lock multiprocessing build
    if (flock(fd, LOCK_EX) != 0) {
      const int32_t error_code = mmGetErrorCode();
      GELOGE(FAILED, "Failed to lock file[%s], error msg[%s].", compile_lock.c_str(),
             GetErrorNumStr(error_code).c_str());
      return FAILED;
    }
    is_locked = true;
    std::string release_info;
    GE_CHK_STATUS_RET(GetReleaseInfo(release_pkg_path, release_info), "Failed to get path[%s] info.",
                      release_pkg_path.c_str());
    // 老版本缓存文件格式不匹配，需要重新生成tar.gz。在线编译在release_pkg_path目录下生成tar.gz
    if ((cache_release_info != release_info) || (om_model_file.empty())) {
      const std::string normalize_name = GenNormalizeModelName(udf_model.GetModelName());
      GE_CHK_STATUS_RET(SaveModelToFile(udf_model, release_pkg_path, normalize_name),
                        "Failed to save model to release pkg for op[%s], release_pkg_path=%s",
                        op_desc->GetName().c_str(), release_pkg_path.c_str());
      GE_CHK_STATUS_RET(PackRelease(release_pkg_path, resource_type, normalize_name),
                        "Failed to pack release pkg for op[%s], release_pkg_path=%s", op_desc->GetName().c_str(),
                          release_pkg_path.c_str());
      GE_CHK_STATUS_RET(GetReleaseInfo(release_pkg_path, release_info), "Failed to get path[%s] info.",
                        release_pkg_path.c_str());
      GE_CHK_BOOL_RET_STATUS(graph->SetExtAttr("_graph_info_for_data_flow_cache", release_info), FAILED,
                              "Failed to set graph info, graph[%s].", graph->GetName().c_str());
      const std::string real_pkg_path = RealPath(release_pkg_path.c_str());
      GE_ASSERT_TRUE(!real_pkg_path.empty(), "Real path can not be empty.");
      udf_model.SetSavedModelPath(real_pkg_path + "/" + normalize_name + ".tar.gz");
      udf_model.SetNormalizedModelName(normalize_name);
      GE_CHK_BOOL_RET_STATUS(graph->SetExtAttr("_udf_om_file_for_data_flow_cache",
          graph->GetName() + "/" + normalize_name + ".om"), FAILED, "Failed to set cache graph info for graph[%s].",
          graph->GetName().c_str());
      GELOGI("Set release lib[%s] to UdfModel from op[%s] success.", real_pkg_path.c_str(),
           op_desc->GetName().c_str());
    } else {
      GE_CHK_STATUS_RET(ProcessForCache(udf_model, om_model_file), "Process for udf cache failed.");
    }
  }
  return SUCCESS;
}

Status UdfModelBuilder::ProcessForCache(UdfModel &udf_model, const std::string &om_model_file) const {
  udf_model.SetSavedModelPath(om_model_file);
  std::string normalize_model_name = GetUdfModelNameByFileName(om_model_file);
  GE_ASSERT_TRUE(!normalize_model_name.empty(), "Get normalized name faield by path[%s]", om_model_file.c_str());
  udf_model.SetNormalizedModelName(normalize_model_name);
  GELOGI("Set cache file path [%s] and normalized name [%s] to UdfModel success.",
    om_model_file.c_str(), normalize_model_name.c_str());
  return SUCCESS;
}

Status UdfModelBuilder::GenReleasePackage(UdfModel &udf_model, const OpDescPtr &op_desc,
    const std::string &resource_type, const ComputeGraphPtr &graph) const {
  udf::UdfModelDef &udf_model_def = udf_model.MutableUdfModelDef();
  const auto udf_def_size = udf_model_def.udf_def_size();
  GE_ASSERT_TRUE(udf_def_size == 1, "udf def size should be 1");
  udf::UdfDef *udf_def = udf_model_def.mutable_udf_def(0);
  GE_CHECK_NOTNULL(udf_def);
  if (udf_def->bin_name() == kUdfBuildInBinName) {
    GELOGD("Builtin udf need to set release package.");
    // builtin model need not be genenrate tar.gz
    udf_model.SetIsBuiltinModel(true);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(SetBinName(op_desc, *udf_def), "Failed to set bin_name for UdfDef from op[%s].",
                    op_desc->GetName().c_str());
  return GenReleasePackageForUserDefineFunc(udf_model, op_desc, resource_type, graph);
}

Status UdfModelBuilder::SaveModelToFile(UdfModel &udf_model, const std::string &release_pkg_path,
                                        const std::string &normalize_name) const {
  ModelBufferData serialize_buff{};
  GE_CHK_STATUS_RET(udf_model.SerializeModelDef(serialize_buff), "[Serialize][Model]Failed, model name=%s",
                    udf_model.GetModelName().c_str());
  std::string om_data_file_name = release_pkg_path + "/" + normalize_name + ".om";
  uint32_t data_len = static_cast<uint32_t>(serialize_buff.length);
  GE_CHK_GRAPH_STATUS_RET(WriteBinToFile(om_data_file_name, reinterpret_cast<char_t *>(serialize_buff.data.get()),
      data_len), "Wrtie data to file[%s] failed.", om_data_file_name.c_str());
  GELOGI("Save udf model %s to file %s success. size is %u.",
          udf_model.GetModelName().c_str(), om_data_file_name.c_str(), data_len);
  return SUCCESS;
}

Status UdfModelBuilder::SetAttr(const std::string &attr_name, const AnyValue &value, UdfAttrMap &udf_attrs) const {
  AnyValue::ValueType value_type = value.GetValueType();
  const auto iter = UdfAttrUtils::set_attr_funcs_.find(value_type);
  if (iter == UdfAttrUtils::set_attr_funcs_.cend()) {
    GELOGW("Set attr[%s] failed, unsupport value type %d.", attr_name.c_str(), value_type);
    return SUCCESS;
  }
  udf::AttrValue attr;
  GE_CHK_STATUS_RET(iter->second(value, attr), "Failed to set %s attr.", attr_name.c_str());
  GE_CHK_BOOL_RET_STATUS(udf_attrs.insert(UdfAttrMap::value_type(attr_name, attr)).second, FAILED,
                         "Failed to insert %s attr for UdfDef.attrs.", attr_name.c_str());
  return SUCCESS;
}

Status UdfModelBuilder::GetReleaseInfo(const std::string &release_path, std::string &release_info) {
  const std::string cmd = "ls -Rl --full-time --ignore=*release.tar.gz " + release_path + " 2>&1";
  FILE *pipe = popen(cmd.c_str(), "r");
  GE_CHECK_NOTNULL(pipe);
  release_info = "";
  constexpr int32_t buffer_len = 128;
  char buffer[buffer_len];
  while (fgets(buffer, buffer_len, pipe) != nullptr) {
    release_info += buffer;
  }
  const auto ret = pclose(pipe);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED, "Failed to get release info, ret[%d], errmsg[%s]", ret,
                         release_info.c_str());
  return SUCCESS;
}

Status UdfModelBuilder::PackRelease(const std::string &release_pkg_path, const std::string &resource_type,
                                    const std::string &normalize_name) {
  GE_ASSERT_TRUE(!normalize_name.empty(), "Udf model name should not be empty.");
  if (resource_type == kResourceTypeAscend) {
    return PackReleaseWithHash(release_pkg_path, normalize_name);
  } else {
    return PackReleaseWithoutHash(release_pkg_path, normalize_name);
  }
  return SUCCESS;
}

void UdfModelBuilder::GenerateTarCmd(const std::string &release_pkg_path, const std::string &normalize_name,
                                     const bool with_hash, std::string &pack_cmd) {
  const std::string om_dir = normalize_name + ".om_dir";
  pack_cmd = R"(set -e
release_dir=")";
  pack_cmd.append(release_pkg_path.c_str());
  std::string om_dir_var =  R"("
current_dir=`pwd`
om_dir=")";
  pack_cmd.append(om_dir_var);
  pack_cmd.append(om_dir);

  std::string name_var = R"("
normalize_name=")";
  pack_cmd.append(name_var);
  pack_cmd.append(normalize_name);
  if (with_hash) {
  pack_cmd.append(R"("
release_dir_tmp=$(dirname "$release_dir")
release_dir_tmp="${release_dir_tmp}_tmp"
rm -rf "$release_dir_tmp/udf_resource"
mkdir -p "$release_dir_tmp/udf_resource"
cp -fr "$release_dir"* "$release_dir_tmp/udf_resource/"
cd "$release_dir_tmp/udf_resource"
mkdir -p "$om_dir"
item_list=$(ls ./ | grep -v tar.gz | grep -v '\.om_dir' | grep -v '\.om' | xargs)
for item in $item_list; do
  mv "$item" "$om_dir"/
done
cd "$om_dir"
so_list=$(find ./ -type f -name "*.so*")
for so_full_name in $so_list; do
  so_md5_val=$(md5sum "$so_full_name"|awk '{print $1}')
  so_new_name="${so_full_name}.${so_md5_val}"
  relative_path=$(realpath --relative-to="$so_new_name" ./ | awk '{print $1}')
  mv -n "$so_full_name" "$so_new_name"
  so_name_with_md5=`basename "$so_new_name"`
  mv "$so_new_name" ../
  ln -sf "$relative_path"/"$so_name_with_md5" "$so_full_name"
done
cd ../..
tar --exclude=*release.tar.gz -zcf release.tar.gz *
cd $current_dir > /dev/null
mv -f "$release_dir_tmp/release.tar.gz" "$release_dir"/"$normalize_name".tar.gz
rm -rf "$release_dir_tmp"
rm -rf "$release_dir"/"$normalize_name".om
)");
  } else {
  pack_cmd.append(R"("
release_dir_tmp=$(dirname "$release_dir")
release_dir_tmp="${release_dir_tmp}_tmp"
rm -rf "$release_dir_tmp/udf_resource"
mkdir -p "$release_dir_tmp/udf_resource"
cp -fr "$release_dir"* "$release_dir_tmp/udf_resource/"
cd "$release_dir_tmp/udf_resource"
mkdir -p "$om_dir"
item_list=$(ls ./ | grep -v tar.gz | grep -v '\.om_dir' | grep -v '\.om' | xargs)
for item in $item_list; do
  mv "$item" "$om_dir"/
done
cd ..
tar --exclude=*release.tar.gz -zcf release.tar.gz *
cd $current_dir > /dev/null
mv -f "$release_dir_tmp/release.tar.gz" "$release_dir"/"$normalize_name".tar.gz
rm -rf "$release_dir_tmp"
rm -rf "$release_dir"/"$normalize_name".om
)");
  }
  GELOGD("Generate pack cmd is[%s].", pack_cmd.c_str());
}

Status UdfModelBuilder::PackReleaseWithHash(const std::string &release_pkg_path, const std::string &normalize_name) {
  std::string pack_cmd;
  GenerateTarCmd(release_pkg_path, normalize_name, true, pack_cmd);
  GE_CHK_STATUS_RET(ProcessUtils::System(pack_cmd), "Failed to execute cmd[%s].", pack_cmd.c_str());
  // release.tar.gz include a dir udf_resource
  GELOGI("pack release with hash successfully, release_pkg_path=%s", release_pkg_path.c_str());
  return SUCCESS;
}

Status UdfModelBuilder::PackReleaseWithoutHash(const std::string &release_pkg_path, const std::string &normalize_name) {
  std::string pack_cmd;
  GenerateTarCmd(release_pkg_path, normalize_name, false, pack_cmd);
  GE_CHK_STATUS_RET(ProcessUtils::System(pack_cmd), "Failed to execute cmd[%s].", pack_cmd.c_str());
  GELOGI("pack release without hash successfully, release_pkg_path=%s", release_pkg_path.c_str());
  return SUCCESS;
}

std::string UdfModelBuilder::GenNormalizeModelName(const std::string &model_name) const {
  std::stringstream ss;
  for (const char &element : model_name) {
    if ((!isalpha(element)) && (!isdigit(element)) && (element != '_')) {
      ss << std::hex << static_cast<uint32_t>(element);
    } else {
      ss << element;
    }
  }
  std::string result = ss.str() + "_release";
  if (result.size() + strlen(".tar.gz") > NAME_MAX) {
    uint8_t sha256[SHA256_DIGEST_LENGTH];
    (void)SHA256(reinterpret_cast<const uint8_t *>(result.c_str()), static_cast<std::size_t>(result.size()), sha256);
    std::stringstream shass;
    for (const auto byte : sha256) { // 使用范围for循环
        shass << std::hex << static_cast<int32_t>(byte);
    }
    result = shass.str() + "_release";
  }
  return result;
}
}  // namespace ge
