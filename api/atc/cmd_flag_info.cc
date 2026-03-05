/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmd_flag_info.h"

#include "base/err_msg.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <regex>
#include <set>
#include <unordered_set>
#include <string>

#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"

namespace ge {
namespace {
const static std::unordered_set<std::string> kDeprecatedFlags = {"shape_generalized_build_mode"};
const static std::map<std::string, std::set<std::string>> kStrValueRange = {
    {"sparsity", {"0", "1"}},
    {"dump_mode", {"0", "1"}},
    {"virtual_type", {"0", "1"}},
    {"status_check", {"0", "1"}},
    {"deterministic", {"0", "1"}},
    {"deterministic_level", {"0", "1", "2"}},
    {"external_weight", {"0", "1", "2"}},
    {"display_model_info", {"0", "1"}},
    {"atomic_clean_policy", {"0", "1"}},
    {"disable_reuse_memory", {"0", "1"}},
    {"framework", {"0", "1", "3", "5"}},
    {"op_debug_level", {"0", "1", "2", "3", "4"}},
    {"mode", {"0", "1", "3", "5", "6", "7","30"}},
    {"core_type", {"AiCore", "VectorCore"}},
    {"log", {"debug", "info", "warning", "error", "null"}},
    {"op_compiler_cache_mode", {"enable", "disable", "force"}},
    {"buffer_optimize", {"l1_optimize", "l2_optimize", "off_optimize"}},
    {"shape_generalized_build_mode", {"shape_generalized", "shape_precise"}},
    {"op_select_implmode",
     {"high_precision", "high_performance", "high_precision_for_all", "high_performance_for_all"}},
    {"precision_mode",
     {"force_fp16", "force_fp32", "cube_fp16in_fp32out", "allow_mix_precision", "allow_fp32_to_fp16",
      "must_keep_origin_dtype", "allow_mix_precision_fp16", "allow_mix_precision_bf16", "allow_fp32_to_bf16"}},
    {"precision_mode_v2",
     {"fp16", "origin", "cube_fp16in_fp32out", "mixed_float16", "mixed_bfloat16", "cube_hif8", "mixed_hif8"}},
};
}  // namespace
namespace flgs {
#define GFPRINTF(fmt, ...)                              \
  do {                                                  \
    fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); \
  } while (false)

namespace {
namespace FlagName {
  const char HELP[] = "help";
}

namespace FlagIndex {
  const int32_t COLON = ':';
  const int32_t QUESTION_MARK = '?';
  const int32_t HELP = 'h';
  const int32_t CUSTOM_START_INDEX = 'z';
}

const char CMD_SHORT_OPTS[] = ":h";

class CmdFlagInfo;
using CmdFlagInfoMap = std::map<int32_t, std::shared_ptr<CmdFlagInfo>>;
static GfStatus StringToBool(const std::string &value, bool &out);
static GfStatus StringToInt32(const std::string &value, int32_t &out);

class CmdFlagInfo {
public:
  CmdFlagInfo(int32_t has_arg, int32_t index, const std::string &flag_name,
              const std::string &default_val, const std::string &msg);
  CmdFlagInfo(int32_t has_arg, int32_t index, const std::string &flag_name,
              const bool default_val, const std::string &msg);
  CmdFlagInfo(int32_t has_arg, int32_t index, const std::string &flag_name,
              const int32_t default_val, const std::string &msg);
  GfStatus SetFlagValue(const std::string &value);
  int32_t GetFlagHasArg() const { return has_arg_; }
  int32_t GetFlagIndex() const { return index_; }
  const std::string& GetFlagName() const { return flag_name_; }
  const std::string& GetFlagMsg() const { return msg_; }
  std::string& GetFlagValueString() { return value_string_; }
  bool& GetFlagValueBool() { return value_bool_; }
  int32_t& GetFlagValueInt32() { return value_int32_; }
  void PrintTypeError(const char *type);
  void PrintValueError();
  std::string PrintValueRange();

private:
  enum class DataType {
    Unknown,
    String,
    Bool,
    Int32
  };
  int32_t has_arg_;
  int32_t index_;
  std::string flag_name_;
  std::string default_val_;
  std::string msg_;
  DataType data_type_;
  std::string value_string_;
  bool value_bool_;
  int32_t value_int32_;
};

CmdFlagInfo::CmdFlagInfo(int32_t has_arg, int32_t index, const std::string &flag_name, const std::string &default_val,
                         const std::string &msg)
    : has_arg_(has_arg),
      index_(index),
      flag_name_(flag_name),
      default_val_(default_val),
      msg_(msg),
      data_type_(CmdFlagInfo::DataType::String),
      value_string_(default_val) {}

CmdFlagInfo::CmdFlagInfo(int32_t has_arg, int32_t index, const std::string &flag_name, const bool default_val,
                         const std::string &msg)
    : has_arg_(has_arg),
      index_(index),
      flag_name_(flag_name),
      default_val_(std::to_string(default_val)),
      msg_(msg),
      data_type_(CmdFlagInfo::DataType::Bool),
      value_bool_(default_val) {}

CmdFlagInfo::CmdFlagInfo(int32_t has_arg, int32_t index, const std::string &flag_name, const int32_t default_val,
                         const std::string &msg)
    : has_arg_(has_arg),
      index_(index),
      flag_name_(flag_name),
      default_val_(std::to_string(default_val)),
      msg_(msg),
      data_type_(CmdFlagInfo::DataType::Int32),
      value_int32_(default_val) {}

void CmdFlagInfo::PrintTypeError(const char *type) {
  std::string reason = "The value type must be [" + std::string(type) + "].";
  REPORT_PREDEFINED_ERR_MSG("E10003", std::vector<const char *>({"value", "parameter", "reason"}),
                            std::vector<const char *>({value_string_.c_str(), flag_name_.c_str(), reason.c_str()}));
}

std::string CmdFlagInfo::PrintValueRange() {
  const auto it = kStrValueRange.find(flag_name_);
  if (it != kStrValueRange.cend()) {
    std::string result;
    const auto &ranges = it->second;
    for (const auto &range : ranges) {
      result += "[" + range + "] ";
    }
    return result;
  }
  return "";
}

void CmdFlagInfo::PrintValueError() {
  if ((flag_name_ == "status_check") || (flag_name_ == "deterministic") ||
      (flag_name_ == "external_weight") || (flag_name_ == "display_model_info") ||
      (flag_name_ == "atomic_clean_policy") || (flag_name_ == "dump_mode") || (flag_name_ == "disable_reuse_memory") ||
      flag_name_ == "sparsity") {
    REPORT_PREDEFINED_ERR_MSG("E10006", std::vector<const char *>({"value", "parameter"}),
                              std::vector<const char *>({value_string_.c_str(), flag_name_.c_str()}));
  } else if (flag_name_ == "log") {
    REPORT_PREDEFINED_ERR_MSG("E10010", std::vector<const char *>({"loglevel"}),
                              std::vector<const char *>({value_string_.c_str()}));
  } else if (flag_name_ == "framework") {
    const std::string support = "0(Caffe) or 1(MindSpore) or 3(TensorFlow) or 5(Onnx)";
    REPORT_PREDEFINED_ERR_MSG("E10007", std::vector<const char *>({"parameter", "support"}),
                              std::vector<const char *>({"framework", support.c_str()}));
  } else {
    std::string parameter = "--" + flag_name_;
    const std::string reason =
        "The value is not within the range of values. The valid range is " + PrintValueRange() + ".";
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"value", "parameter", "reason"}),
                              std::vector<const char *>({value_string_.c_str(), parameter.c_str(), reason.c_str()}));
  }
}

GfStatus CmdFlagInfo::SetFlagValue(const std::string &value)
{
  value_string_ = value;
  const auto it = kStrValueRange.find(flag_name_);
  if ((it != kStrValueRange.cend()) && (it->second.count(value_string_) == 0U)) {
    PrintValueError();
    return GF_FAILED;
  }
  if ((data_type_ == DataType::Bool) && (StringToBool(value_string_, value_bool_) != GF_SUCCESS)) {
    PrintTypeError("bool");
    return GF_FAILED;
  }
  if ((data_type_ == DataType::Int32) && (StringToInt32(value_string_, value_int32_) != GF_SUCCESS)) {
    PrintTypeError("int32");
    return GF_FAILED;
  }
  if (kDeprecatedFlags.count(flag_name_) > 0U) {
    GELOG_DEPRECATED(flag_name_);
  }
  return GF_SUCCESS;
}

GfStatus StringToBool(const std::string &value, bool &out)
{
  const std::set<std::string> true_values = { "1", "t", "true", "y", "yes" };
  const std::set<std::string> false_values = { "0", "f", "false", "n", "no" };
  std::string val = value;
  std::transform(value.begin(), value.end(), val.begin(), ::tolower);
  if (val.empty() || (true_values.count(val) > 0U)) {
    out = true;
    return GF_SUCCESS;
  }
  if (false_values.count(val) > 0U) {
    out = false;
    return GF_SUCCESS;
  }
  return GF_FAILED;
}

GfStatus StringToInt32(const std::string &value, int32_t &out)
{
  if (std::any_of(value.begin(), value.end(), [](const char &v) { return !std::isdigit(v); })) {
    return GF_FAILED;
  }
  try {
    out = std::stoi(value);
  } catch (std::exception &e) {
    return GF_FAILED;
  }
  return GF_SUCCESS;
}

static CmdFlagInfoMap& GetCmdFlagInfoMap()
{
  static CmdFlagInfoMap cmd_flag_info_map;
  return cmd_flag_info_map;
}

static int32_t GetNextFlagIndex(const std::string &name)
{
  static int32_t current_index = FlagIndex::CUSTOM_START_INDEX;
  if (name == FlagName::HELP) {
    return FlagIndex::HELP;
  }
  return ++current_index;
}

template <class T>
static std::shared_ptr<CmdFlagInfo> RegisterCmdFlagInfo(int32_t has_arg, const std::string &name,
                                                        const T &default_val, const std::string &msg)
{
  std::shared_ptr<CmdFlagInfo> cmd_flag_info = nullptr;
  try {
    cmd_flag_info = std::make_shared<CmdFlagInfo>(has_arg, GetNextFlagIndex(name), name, default_val, msg);
  } catch (...) {
    return nullptr;
  }
  GetCmdFlagInfoMap()[cmd_flag_info->GetFlagIndex()] = cmd_flag_info;
  return cmd_flag_info;
}

static GfStatus UpdateCmdFlagInfo(const int32_t index, const std::string &value)
{
  const auto info = GetCmdFlagInfoMap().find(index);
  if (info != GetCmdFlagInfoMap().cend()) {
    GE_ASSERT_NOTNULL(info->second);
    auto ret = info->second->SetFlagValue(value);
    if (ret == GF_SUCCESS) {
      GetUserOptions().emplace(info->second->GetFlagName(), value);
    }
    return ret;
  }
  return GF_FAILED;
}

static std::vector<mmStructOption>& GetOptionsVec()
{
  static std::vector<mmStructOption> options_vec;
  options_vec.clear();
  const CmdFlagInfoMap &info_map = GetCmdFlagInfoMap();
  for (const auto &info : info_map) {
    mmStructOption opt;
    opt.name = info.second->GetFlagName().c_str();
    opt.has_arg = info.second->GetFlagHasArg();
    opt.flag = nullptr;
    opt.val = info.second->GetFlagIndex();
    options_vec.push_back(opt);
    if (info.second->GetFlagName() == "help") {
      opt.name = "h";
      options_vec.push_back(opt);
    }
  }
  // last line fill zero
  mmStructOption tmp_opt{};
  options_vec.push_back(tmp_opt);
  return options_vec;
}

static std::string GetFlagMsg(const std::string &flag_name)
{
  const std::regex r("^-+");
  const std::string &name = std::regex_replace(flag_name, r, "");
  const CmdFlagInfoMap &info_map = GetCmdFlagInfoMap();
  for (const auto &info : info_map) {
    if (info.second->GetFlagName() == name) {
      return info.second->GetFlagMsg();
    }
  }
  return "";
}

static void ReplaceFlagPrefixSingleMinus(std::string &str)
{
  const std::regex r("^-(?=[[:alpha:]]{2,})");
  str = std::regex_replace(str, r, "--");
}

static void ReplaceFlagNameMinus(std::string &str)
{
  const std::regex r("^--(?=[[:alpha:]]{2,})");
  const int prefix_len = 4;
  if (!std::regex_search(str, r)) {
    return;
  }
  auto it = str.begin() + prefix_len;
  while (it != str.end()) {
    if (*it == '=') {
      break;
    }
    if (*it == '-') {
      *it = '_';
    }
    ++it;
  }
}

static char** FormatArgv(int32_t argc, char *argv[])
{
  static std::vector<std::shared_ptr<std::string>> vec_str;
  static std::vector<char *> vec_ptr;
  vec_str.clear();
  vec_ptr.clear();
  for (int i = 0; i < argc; ++i) {
    std::string str(argv[i]);
    ReplaceFlagPrefixSingleMinus(str);
    ReplaceFlagNameMinus(str);
    vec_str.push_back(std::make_shared<std::string>(str));
    vec_ptr.push_back(const_cast<char *>(vec_str.back()->c_str()));
  }
  return vec_ptr.data();
}

static std::string GetFlagName(const std::string &flag_name)
{
  const std::regex r("^-+|=.*");
  return std::regex_replace(flag_name, r, "");
}
} // namespace

std::string& RegisterParamString(const std::string &name, const std::string &default_val, const std::string &msg)
{
  return RegisterCmdFlagInfo(MMPA_REQUIRED_ARGUMENT, name, default_val, msg)->GetFlagValueString();
}

bool& RegisterParamBool(const std::string &name, bool default_val, const std::string &msg)
{
  return RegisterCmdFlagInfo(MMPA_OPTIONAL_ARGUMENT, name, default_val, msg)->GetFlagValueBool();
}

int32_t& RegisterParamInt32(const std::string &name, int32_t default_val, const std::string &msg)
{
  return RegisterCmdFlagInfo(MMPA_REQUIRED_ARGUMENT, name, default_val, msg)->GetFlagValueInt32();
}

static std::string& GetUsageMessage()
{
  static std::string usage;
  return usage;
}

static void PrintUsageMessage()
{
  if (!GetUsageMessage().empty()) {
    std::cout << GetUsageMessage() << std::endl;
  }
}

static GfStatus CheckNonOptionParameters(const int32_t argc, char** const argv)
{
  int32_t idx = mmGetOptInd();
  if (idx >= argc) {
    return GF_SUCCESS;
  }
  if (idx < 0) {
    return GF_FAILED;
  }
  while (idx < argc) {
    GFPRINTF("Non-option parameter: %s", argv[idx]);
    idx++;
  }
  return GF_FAILED;
}

void SetUsageMessage(const std::string &usage)
{
  GetUsageMessage() = usage;
}

std::string& GetArgv()
{
  static std::string argv;
  return argv;
}

std::unordered_map<std::string, std::string> &GetUserOptions()
{
  static std::unordered_map<std::string, std::string> user_options;
  return user_options;
}

GfStatus ParseCommandLine(int32_t argc, char* argv[])
{
  int32_t index = 0;
  GfStatus ret = GF_FAILED;
  GetArgv().clear();
  for (int32_t i = 0; i < argc; ++i) {
    GetArgv() += std::string(argv[i]) + ((i == argc - 1) ? "" : " ");
  }
  mmSetOptErr(0);
  char **buff = FormatArgv(argc, argv);
  optind = 1;
  while ((index = mmGetOptLong(argc, buff, CMD_SHORT_OPTS, GetOptionsVec().data(), nullptr)) != -1) {
    int32_t idx = mmGetOptInd() - 1;
    if ((idx < 0) || (idx >= argc)) {
      GFPRINTF("input argument value is empty or command line format is invalid");
      return GF_FAILED;
    }
    if (index == FlagIndex::COLON) {
      const char *flag_name = buff[idx];
      const std::string &flag_msg = GetFlagMsg(flag_name);
      if (flag_msg.empty()) {
        GFPRINTF("flag '%s' is missing its argument", flag_name);
      } else {
        GFPRINTF("flag '%s' is missing its argument; flag description: %s", flag_name, flag_msg.c_str());
      }
      return GF_FAILED;
    }
    if (index == FlagIndex::QUESTION_MARK) {
      GFPRINTF("unknown command line flag '%s'", GetFlagName(buff[idx]).c_str());
      return GF_FAILED;
    }
    std::string value = (mmGetOptArg() == nullptr) ? "" : mmGetOptArg();
    ret = UpdateCmdFlagInfo(index, value);
    if (ret != GF_SUCCESS) {
      return GF_FAILED;
    }
    if (index == FlagIndex::HELP) {  // parameter is --help
      const CmdFlagInfoMap &info_map = GetCmdFlagInfoMap();
      const auto it = info_map.find(FlagIndex::HELP);
      if ((it != info_map.end()) && it->second->GetFlagValueBool()) {
        PrintUsageMessage();
        return GF_HELP;
      }
    }
  }

  return CheckNonOptionParameters(argc, buff);
}
} // namespace flgs
} // namespace ge
