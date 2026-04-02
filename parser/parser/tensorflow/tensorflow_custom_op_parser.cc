/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow_custom_op_parser.h"

#include "parser/tensorflow/tensorflow_parser.h"
#include <algorithm>
#include <iostream>
#include <regex>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include "parser/common/convert/pb2json.h"
#include "parser/common/acl_graph_parser_util.h"
#include "base/err_msg.h"
#include "parser/tensorflow_parser.h"
#include "custom_op_factory_impl.h"
#include "register/scope/scope_fusion_pass_register.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "iterator_fusion_pass.h"
#include "omg/parser/parser_factory.h"
#include "parser/common/model_saver.h"
#include "parser/common/parser_fp16_t.h"
#include "parser/common/thread_pool.h"
#include "parser/common/util.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "register/auto_mapping_util.h"
#include "register/op_registry.h"
#include "common/checker.h"
#include "common/op_registration_tbe.h"
#include "graph/custom_op_factory.h"
#include "graph/opsproto_manager.h"
#include "common/types_map.h"
#include "utils/file_utils.h"

namespace ge {
const std::regex kSafePathRegex(R"(^(?!.*\.{2})[A-Za-z0-9./+\-_]+$)");
std::string TfDTypeToGeSymbol(domi::tensorflow::DataType dt) {
  auto it = TF_DATATYPE_TO_GE_SYMBOL_MAP.find(dt);
  return (it != TF_DATATYPE_TO_GE_SYMBOL_MAP.end()) ? it->second : "";
}

template <typename T>
std::string   FormatListValue(const google::protobuf::RepeatedField<T>& list_vals, const std::string& type) {
  std::ostringstream oss;
  oss << "{";
  for (int i = 0; i < list_vals.size(); ++i) {
    if (i > 0) oss << ", ";
    if (type == "int") {
      oss << list_vals.Get(i);
    }
    if (type == "float") {
      oss << list_vals.Get(i);
    }
    if (type == "string") {
      oss << "\"" << list_vals.Get(i) << "\"";
    }
    if (type == "type") {
      auto dt = static_cast<domi::tensorflow::DataType>(list_vals.Get(i));
      auto it = TF_DATATYPE_TO_GE_SYMBOL_MAP.find(dt);
      oss << (it != TF_DATATYPE_TO_GE_SYMBOL_MAP.end() ? it->second : "DT_FLOAT");
    }
  }
  oss << "}";
  return oss.str();
}

std::string FormatListValue(const ascend_private::protobuf::RepeatedPtrField<std::string>& list_vals, const std::string& type) {
  std::ostringstream oss;
  oss << "{";
  int i = 0;
  for (const auto& val : list_vals) {
    if (i > 0) oss << ", ";
    if (type == "string") {
      oss << "\"" << val << "\"";
    }
    i++;
  }
  oss << "}";
  return oss.str();
}

std::string FormatShapeValue(const domi::tensorflow::TensorShapeProto& shape) {
  std::ostringstream oss;
  oss << "\"[";
  for (int i = 0; i < shape.dim_size(); ++i) {
    if (i > 0) oss << ", ";
    oss << shape.dim(i).size();
  }
  oss << "]\"";
  return oss.str();
}

std::string GetAttrDefaultValue(const domi::tensorflow::AttrValue& default_val, const std::string& attr_type) {
  if (attr_type == "int") {
    return std::to_string(default_val.i());
  }
  if (attr_type == "float") {
    return std::to_string(default_val.f());
  }
  if (attr_type == "bool") {
    return default_val.b() ? "true" : "false";
  }
  if (attr_type == "string") {
    return "\"" + default_val.s() + "\"";
  }
  if (attr_type == "type") {
    auto dt = static_cast<domi::tensorflow::DataType>(default_val.type());
    auto it = TF_DATATYPE_TO_GE_SYMBOL_MAP.find(dt);
    return (it != TF_DATATYPE_TO_GE_SYMBOL_MAP.end()) ? it->second : "DT_FLOAT";
  }
  if (attr_type == "shape") {
    return FormatShapeValue(default_val.shape());
  }
  const auto& list_val = default_val.list();
  if (attr_type == "list(int)") {
    return FormatListValue(list_val.i(), "int");
  }
  if (attr_type == "list(float)") {
    return FormatListValue(list_val.f(), "float");
  }
  if (attr_type == "list(bool)") {
    return FormatListValue(list_val.b(), "bool");
  }
  if (attr_type == "list(string)") {
    return FormatListValue(list_val.s(), "string");
  }
  if (attr_type == "list(type)") {
    return FormatListValue(list_val.type(), "type");
  }
  GELOGW("Unsupported attr type: %s, return empty string", attr_type.c_str());
  return "\"\"";
}

bool ArgHasFixedType(const domi::tensorflow::OpDef::ArgDef &arg) {
  if (!arg.type_attr().empty() || !arg.type_list_attr().empty()) {
    return false;
  }
  return true;
}

void AppendLine(std::string &out, const std::string &indent, const std::string &line) {
  out.append(indent).append(line).append("\n");
}

bool IsListArg(const domi::tensorflow::OpDef::ArgDef &arg) {
  // 动态输入/输出的判定：number_attr 或 type_list_attr
  return !arg.number_attr().empty() || !arg.type_list_attr().empty();
}

std::string FormatTensorTypeExpr(const std::vector<std::string> &syms) {
  if (syms.empty()) {
    return "TensorType::ALL()";
  }
  std::ostringstream oss;
  oss << "TensorType({";
  for (size_t i = 0; i < syms.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << syms[i];
  }
  oss << "})";
  return oss.str();
}

std::vector<std::string> CollectAllowedTypeSyms(
    const domi::tensorflow::OpDef::ArgDef &arg,
    const std::unordered_map<std::string, const domi::tensorflow::OpDef::AttrDef*> &attr_map) {
  std::vector<std::string> out;
  // Handling fixed type parameters
  if (ArgHasFixedType(arg)) {
    const std::string sym = TfDTypeToGeSymbol(static_cast<domi::tensorflow::DataType>(arg.type()));
    if (!sym.empty()) {
      out.push_back(sym);
    }
    return out;
  }
  // Handling dynamically typed parameters (associated properties)
  const std::string &ref = !arg.type_attr().empty() ? arg.type_attr() : arg.type_list_attr();
  if (ref.empty()) {
    GELOGW("Arg %s has no type_attr/type_list_attr", arg.name().c_str());
    return out;
  }
  auto it = attr_map.find(ref);
  if (it == attr_map.end() || it->second == nullptr) {
    GELOGW("Attr %s not found in attr_map for arg %s", ref.c_str(), arg.name().c_str());
    return out;
  }
  const auto &attr_def = *(it->second);
  if (!attr_def.has_allowed_values()) {
    GELOGW("Attr %s has no allowed_values for arg %s", ref.c_str(), arg.name().c_str());
    return out;
  }

  const auto &lv = attr_def.allowed_values().list();
  for (int i = 0; i < lv.type_size(); ++i) {
    const std::string sym = TfDTypeToGeSymbol(static_cast<domi::tensorflow::DataType>(lv.type(i)));
    if (!sym.empty()) {
      out.push_back(sym);
    }
  }

  return out;
}

std::unordered_map<std::string, const domi::tensorflow::OpDef::AttrDef*>
BuildAttrDefMap(const domi::tensorflow::OpDef &opdef) {
  std::unordered_map<std::string, const domi::tensorflow::OpDef::AttrDef*> m;
  m.reserve(static_cast<size_t>(opdef.attr_size()));
  for (const auto &a : opdef.attr()) {
    m.emplace(a.name(), &a);
  }
  return m;
}
bool HasArgDefaultValue(const domi::tensorflow::OpDef::ArgDef &arg,
    const std::unordered_map<std::string, const domi::tensorflow::OpDef::AttrDef*> &attr_map) {
  const std::vector<std::string> possible_attr_names = {
    arg.name() + "_default",
    "default_" + arg.name(),
    arg.name() + "_def_val"
  };
  for (const auto& attr_name : possible_attr_names) {
    auto it = attr_map.find(attr_name);
    if ((it != attr_map.end()) && (it->second != nullptr) && (it->second->has_default_value())) {
        return true;
    }
  }
  return false;
}
void ProcessArg(std::string &reg_op, const std::string &indent,
                const domi::tensorflow::OpDef::ArgDef &arg,
                const std::unordered_map<std::string, const domi::tensorflow::OpDef::AttrDef*> &attr_map,
                bool is_input) {
  if (arg.name().empty()) {
    GELOGW("Empty arg name found, skip processing");
    return;
  }
  const auto type_syms = CollectAllowedTypeSyms(arg, attr_map);
  const auto type_expr = FormatTensorTypeExpr(type_syms);
  std::string arg_type;
  if (is_input) {
    bool has_default = HasArgDefaultValue(arg, attr_map);
    if (has_default) {
      arg_type = "OPTIONAL_INPUT";
    } else {
      arg_type = IsListArg(arg) ? "DYNAMIC_INPUT" : "INPUT";
    }
  } else {
     arg_type = IsListArg(arg) ? "DYNAMIC_OUTPUT" : "OUTPUT";
  }
  std::ostringstream oss;
  oss << "." << arg_type << "(" << arg.name() << ", " << type_expr << ")";
  AppendLine(reg_op, indent, oss.str());
}

std::unordered_set<std::string> CollectTypeAttrNames(const domi::tensorflow::OpDef &opdef) {
  std::unordered_set<std::string> type_attr_names;
  std::unordered_set<std::string> all_exist_attr_names;
  for (const auto &attr: opdef.attr()) {
    all_exist_attr_names.insert(attr.name());
  }
  for (const auto &input_arg: opdef.input_arg()) {
    if (!input_arg.type_attr().empty()) {
      type_attr_names.insert(input_arg.type_attr());
    }
    const std::string arg_name = input_arg.name();
    const std::vector<std::string> default_attr_names = {
      arg_name + "_default",
      "default_" + arg_name,
      arg_name + "_def_val"
    };
    for (const auto &attr_name: default_attr_names) {
      if (all_exist_attr_names.count(attr_name) > 0) {
        type_attr_names.insert(attr_name);
        break;
      }
    }
  }
  for (const auto &output_arg: opdef.output_arg()) {
    if (!output_arg.type_attr().empty()) {
      type_attr_names.insert(output_arg.type_attr());
    }
  }
  return type_attr_names;
}

int32_t RemoveDirectories(const std::string &path) {
  DIR *dir = opendir(path.c_str());
  if (!dir) {
    GELOGE(FAILED, "%s is not a directory", path.c_str());
    return -1;
  }
  struct dirent *entry;
  struct stat st;
  while ((entry = readdir(dir)) != nullptr) {
    if ((strcmp(entry->d_name, ".") == 0) || (strcmp(entry->d_name, "..") == 0)) {
      continue;
    }

    std::string full_name = path + "/" + entry->d_name;
    if (lstat(full_name.c_str(), &st) == 0) {
      if (S_ISDIR(st.st_mode)) {
        RemoveDirectories(full_name);
      } else {
        remove(full_name.c_str());
      }
    }
  }
  closedir(dir);
  return rmdir(path.c_str());
}

Status TensorFlowCustomOpParser::ConstructRegOpString(const domi::tensorflow::OpDef &opdef, std::string &reg_op) {
  const std::string &op_type = opdef.name();
  reg_op.reserve(reg_op.size() + 2048);
  std::ostringstream oss_header;
  oss_header << "REG_OP(" << opdef.name() << ")\n";
  reg_op.append(oss_header.str());
  const std::string indent = "    ";
  const auto attr_map = BuildAttrDefMap(opdef);
  const auto type_attr_names = CollectTypeAttrNames(opdef);
  for (const auto &input_arg : opdef.input_arg()) {
    ProcessArg(reg_op, indent, input_arg, attr_map, true);
  }
  for (const auto &output_arg : opdef.output_arg()) {
    ProcessArg(reg_op, indent, output_arg, attr_map, false);
  }
  for (const auto &attr : opdef.attr()) {
    const std::string &attr_name = attr.name();
    const std::string &attr_type = attr.type();
    if (attr_name.empty() || attr_type.empty() || type_attr_names.count(attr_name)) {
      continue;
    }
    std::string ge_attr_type = "String";
    std::string default_val = "\"\"";
    auto it = attr_type_map.find(attr_type);
    if (it != attr_type_map.end()) {
      ge_attr_type = it->second.first;
      default_val  = it->second.second;
    }
    default_val = GetAttrDefaultValue(attr.default_value(), attr_type);
    AppendLine(reg_op, indent, ".ATTR(" + attr_name + ", " + ge_attr_type + ", " + default_val + ")");
  }
  AppendLine(reg_op, indent, ".OP_END_FACTORY_REG(" + op_type + ")");
  return SUCCESS;
}

Status TensorFlowCustomOpParser::ConstructRegCustomOpString(const domi::tensorflow::OpDef &opdef, const domi::tensorflow::NodeDef &node_def, std::string &reg_op_custom_string) {
  bool has_dynamic = false;
  for (const auto &input_arg : opdef.input_arg()) {
    if (IsListArg(input_arg)) { has_dynamic = true; break; }
  }
  std::string parse_fn = "AutoMappingByOpFn";
  if (has_dynamic) {
    std::string fn = opdef.name();
    parse_fn = "AutoMappingFnCustomDynamic_" + fn;
    reg_op_custom_string.append("Status ").append(parse_fn).append("(const google::protobuf::Message* op_src, ge::Operator& op) {\n");
    AppendLine(reg_op_custom_string, "  ", "map<string, pair<string, string>> value;");
    for (const auto &arg : opdef.input_arg()) {
      const bool is_list = (!arg.number_attr().empty() || !arg.type_list_attr().empty());
      if (!is_list) continue;
      const std::string &tensor_name = arg.name();
      if (tensor_name.empty()) continue;
      const std::string list_attr = !arg.number_attr().empty() ? arg.number_attr() : arg.type_list_attr();
      if (list_attr.empty()) continue;
      std::string line;
      line.reserve(128);
      line.append("value[\"in").append("\"] = pair<string, string>(\"").append(tensor_name).append("\", \"").append(list_attr).append("\");");
      AppendLine(reg_op_custom_string, "  ", line);
    }
    for (const auto &arg : opdef.output_arg()) {
      const bool is_list = (!arg.number_attr().empty() || !arg.type_list_attr().empty());
      if (!is_list) continue;
      const std::string &tensor_name = arg.name();
      if (tensor_name.empty()) continue;
      const std::string list_attr = !arg.number_attr().empty() ? arg.number_attr() : arg.type_list_attr();
      if (list_attr.empty()) continue;
      std::string line;
      line.reserve(128);
      line.append("value[\"out").append("\"] = pair<string, string>(\"").append(tensor_name).append("\", \"").append(list_attr).append("\");");
      AppendLine(reg_op_custom_string, "  ", line);
    }
    AppendLine(reg_op_custom_string, "  ", "AutoMappingFnDynamic(op_src, op, value);");
    AppendLine(reg_op_custom_string, "  ", "return SUCCESS;");
    reg_op_custom_string.append("}\n");
  }
  reg_op_custom_string.append("REGISTER_CUSTOM_OP(\"").append(node_def.op()).append("\")\n");
  AppendLine(reg_op_custom_string, "    ", ".FrameworkType(TENSORFLOW)");
  reg_op_custom_string.append("    .OriginOpType(\"").append(node_def.op()).append("\")\n");
  if (has_dynamic) {
    reg_op_custom_string.append("    .ParseParamsFn(").append(parse_fn).append(")\n");
  } else {
    reg_op_custom_string.append("    .ParseParamsByOperatorFn(").append(parse_fn).append(")\n");
  }
  AppendLine(reg_op_custom_string, "    ", ".ImplyType(ImplyType::CUSTOM);");
  return SUCCESS;
}

Status GetCompilePath(std::string &include_path, std::string &register_path) {
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, path_env);
  GE_ASSERT_TRUE(path_env != nullptr, "[Call][MM_SYS_GET_ENV] Failed to get env ASCEND_HOME_PATH.");
  std::string home_path = path_env;
  if (home_path.back() != '/') {
    home_path += '/';
  }
  if (home_path.find("include/") == std::string::npos) {
    include_path = home_path + "include/";
  }
  if (home_path.find("lib64/") == std::string::npos) {
    register_path = home_path + "lib64/";
  }
  return SUCCESS;
}

bool CheckPathInCmdIsValid(const std::string &so_path, const std::string &file_path,
                           const std::string &ascend_include_path) {
  return std::regex_match(so_path, kSafePathRegex) && std::regex_match(file_path, kSafePathRegex) &&
         std::regex_match(ascend_include_path, kSafePathRegex);
}

Status TensorFlowCustomOpParser::CompileCustomOpFiles(const std::string &custom_op_cc_path, const std::string &output_so_path) {
  std::string incloud_path;
  std::string register_path;
  GetCompilePath(incloud_path,register_path);
  GELOGI("Header file search directory: %s", register_path.c_str());
  std::string command = "g++ -O2 -fstack-protector-all -shared -fPIC -Wl,-z,now -Wl,-z,noexecstack -s -o " + output_so_path + " -D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=ascend_private -I " +
    incloud_path + " -L c_sec -L " + register_path + " -lregister -lgraph -lruntime -x c++ " + custom_op_cc_path;
  GE_ASSERT_TRUE(CheckPathInCmdIsValid(output_so_path, incloud_path, register_path),
               "CheckPathInCmdIsValid failed, output_so_path = %s, incloud_path = %s, register_path = %s.", incloud_path.c_str(), incloud_path.c_str(), register_path.c_str());
  int rc = system(command.c_str());
  if ((rc == -1) || (WEXITSTATUS(rc) != 0)) {
    int real_exit_code = (rc == -1) ? -1 : WEXITSTATUS(rc);
    GELOGE(FAILED, "Failed to compile custom ops .so file, real exit code: %d, command: %s", real_exit_code, command.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TensorFlowCustomOpParser::RegisteredTfaOps() {
  std::vector<AscendString> all_registered_ops;
  ge::CustomOpFactory::GetAllRegisteredOps(all_registered_ops);
  std::unordered_set<std::string> registered_set;
  registered_set.reserve(all_registered_ops.size());
  for (const auto &op : all_registered_ops) {
    const char *p = op.GetString();
    if (p != nullptr) {
      registered_set.emplace(p);
    }
  }
  const auto &registration_datas = domi::OpRegistry::Instance()->registrationDatas;
  for (const auto &reg_data : registration_datas) {
    const auto fmk = reg_data.GetFrameworkType();
    if (fmk != domi::TENSORFLOW) {
      continue;
    }
    ge::AscendString om_op_type;
    (void)reg_data.GetOmOptype(om_op_type);
    const char *om = om_op_type.GetString();
    if (registered_set.find(om) == registered_set.end()) {
      GELOGD("Skip: om_op_type not in CustomOpFactory registered set: %s", om);
      continue;
    }
    (void)OpRegistrationTbe::Instance()->Finalize(reg_data,true, true);
    (void)domi::OpRegistry::Instance()->Register(reg_data, true);
  }
  return SUCCESS;
}

Status TensorFlowCustomOpParser::LoadCustomOpsLibrary(const std::string &so_path) {
  ge::OpsProtoManager* const protoManager = ge::OpsProtoManager::Instance();
  protoManager->LoadOpsProtoPluginSo(so_path);
  GE_ASSERT_SUCCESS(RegisteredTfaOps());
  return SUCCESS;
}

Status TensorFlowCustomOpParser::BuildCustomOpStrings(const std::unordered_map<std::string, const domi::tensorflow::NodeDef *> &custom_nodes_map, std::string &all_reg_op_strings) {
  std::string op_ss = R"(#ifndef OP_REG_CUSTOM_H
#define OP_REG_CUSTOM_H

#include "graph/operator_reg.h"
#include "register/register.h"
#include "graph/operator.h"
namespace ge {
)";
  std::string custom_ss;
  custom_ss.append("namespace domi {\n");
  for (const auto &kv : custom_nodes_map) {
    const std::string &node_name = kv.first;
    const domi::tensorflow::NodeDef *node_def = kv.second;
    const std::string node_op = node_def->op();
    domi::tensorflow::AttrValue attr_v;
    if (!ge::TensorFlowUtil::FindAttrValue(node_def, ge::ATTR_NAME_FRAMEWORK_OP_DEF, attr_v)) {
      GELOGE(FAILED, "[ERROR] Custom op %s missing necessary attr: %s", node_name.c_str(), ge::ATTR_NAME_FRAMEWORK_OP_DEF.c_str());
      return FAILED;
    }
    const std::string &opdef_blob = attr_v.s();
    domi::tensorflow::OpDef opdef;
    GE_ASSERT_TRUE(opdef.ParseFromString(opdef_blob));
    std::string reg_op_string;
    ConstructRegOpString(opdef, reg_op_string);
    op_ss += reg_op_string;
    std::string reg_op_custom_string;
    ConstructRegCustomOpString(opdef, *node_def, reg_op_custom_string);
    GELOGI("Generated REGISTER_CUSTOM_OP:\n%s", reg_op_custom_string.c_str());
    custom_ss += reg_op_custom_string;
  }
  op_ss += "}\n";
  custom_ss += "}\n#endif  // OP_REG_CUSTOM_H\n";
  all_reg_op_strings = op_ss + custom_ss;
  return SUCCESS;
}

Status TensorFlowCustomOpParser::WriteTextFile(const std::string &file_path, const std::string &content) {
  std::ofstream reg_op_file(file_path, std::ios::out | std::ios::trunc);
  if (!reg_op_file.is_open()) {
    GELOGE(FAILED, "Failed to open file %s for writing", file_path.c_str());
    return FAILED;
  }
  reg_op_file << content;
  reg_op_file.close();
  std::ifstream read_file(file_path);
  if (read_file.is_open()) {
    std::string file_content((std::istreambuf_iterator<char>(read_file)), std::istreambuf_iterator<char>());
    read_file.close();
  } else {
    GELOGE(FAILED, "Failed to read file %s for check", file_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TensorFlowCustomOpParser::WriteWrapperCc(const std::string &cc_path) {
  std::ofstream custom_op_cc_file(cc_path, std::ios::out | std::ios::trunc);
  GE_ASSERT_TRUE(custom_op_cc_file.is_open());
  std::string wrapper_code = R"(
#include <iostream>
#include "op_reg_custom.h"

extern "C" {
  void InitCustomOps() {
    std::cout << "Custom operators initialized successfully!" << std::endl;
  }
}
)";
  custom_op_cc_file << wrapper_code;
  custom_op_cc_file.close();
  return SUCCESS;
}

Status TensorFlowCustomOpParser::DeleteTmpDirectoryContents(const std::string &out_dir) {
  if ((mmAccess(out_dir.c_str()) == EN_OK) && (mmIsDir(out_dir.c_str()) == EN_OK)) {
    (void)RemoveDirectories(out_dir);
  } else {
    GELOGW("Directory %s does not exist.", out_dir.c_str());
  }
  return SUCCESS;
}

Status TensorFlowCustomOpParser::ParseCustomOp(const std::unordered_map<std::string, const domi::tensorflow::NodeDef *> &custom_nodes_map) {
  if (custom_nodes_map.empty()) {
    GELOGI("No custom operators found, custom_nodes_map is empty");
    return SUCCESS;
  }
  // generate register code
  std::string all_reg_op_strings;
  GE_ASSERT_SUCCESS(BuildCustomOpStrings(custom_nodes_map, all_reg_op_strings));
  const std::string out_dir("./custom_op_tmp");
  const std::string header_path = out_dir + "/op_reg_custom.h";
  const std::string cc_path = out_dir + "/op_custom.cc";
  const std::string so_path = out_dir + "/lib_op_custom.so";
  GE_ASSERT_TRUE((ge::CreateDir(out_dir) == EOK), "Create direct failed, path: %s.", out_dir.c_str());
  GE_ASSERT_SUCCESS(WriteTextFile(header_path, all_reg_op_strings));
  GE_ASSERT_SUCCESS(WriteWrapperCc(cc_path));
  GE_ASSERT_SUCCESS(CompileCustomOpFiles(cc_path, so_path));
  GE_ASSERT_SUCCESS(LoadCustomOpsLibrary(so_path));
  GE_ASSERT_SUCCESS(DeleteTmpDirectoryContents(out_dir));
  return SUCCESS;
}
}  // namespace ge