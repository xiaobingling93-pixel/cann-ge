/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen.h"

#include <sstream>
#include <fstream>

#include "codegen_kernel.h"
#include "codegen_tiling_data.h"
#include "ascir_utils.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "codegen_infershape.h"

using namespace codegen;

namespace {
constexpr uint32_t ELEMENTS_PER_LINE = 20;
constexpr size_t kMaxUnfoldedIoNum = 64U;
constexpr size_t kKernelMaxIoNum = 190U;

const std::string kKernelTaskTypeAIVOnly = "KERNEL_TYPE_AIV_ONLY";
const std::string kKernelTaskTypeMixAIVOneZero = "KERNEL_TYPE_MIX_AIV_1_0";
const std::string kKernelTaskTypeAICOnly = "KERNEL_TYPE_AIC_ONLY";
const std::string kKernelTaskTypeMixAICOneTwo = "KERNEL_TYPE_MIX_AIC_1_2";

// Include path prefixes to be removed
const std::string kBasicApiInclude = "#include \"basic_api/";
const std::string kAdvApiInclude = "#include \"adv_api/";
const std::string kMicroApiInclude = "#include \"micro_api/";
const std::string kSimtApiInclude = "#include \"simt_api/";
const std::string kUtilsStdInclude = "#include \"utils/std/";

std::string GetKernelTaskType(const ascir::FusedScheduledResult &schedule_results) {
  if (ascgen_utils::IsJustCubeFixpip(schedule_results)) {
    return kKernelTaskTypeAICOnly;
  } else if (ascgen_utils::IsCubeFusedScheduled(schedule_results)) {
    return kKernelTaskTypeMixAICOneTwo;
  }
  return schedule_results.workspace_nodes.size() != 0 ? kKernelTaskTypeMixAIVOneZero : kKernelTaskTypeAIVOnly;
}

std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(kTilingHeadGuard) == std::string::npos) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

Status CombineTilings(const std::map<std::string, std::string> &tiling_file_name_to_content, std::string &result) {
  GE_CHK_BOOL_RET_STATUS(tiling_file_name_to_content.find(kTilingHeadIdentify) != tiling_file_name_to_content.end(), ge::FAILED,
                         "tiling_file_name_to_content has no tiling head");
  result += RemoveAutoFuseTilingHeadGuards(tiling_file_name_to_content.at(kTilingHeadIdentify));  // 删除头文件的宏保护，cpp文件不需要

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tiling_file_name_to_content) {
    if (key == kTilingHeadIdentify || key.find(kTilingDataIdentify) != std::string::npos) {
      continue;
    }

    // 查找并跳过第一行头文件行
    size_t include_pos = value.find(kTilingHeadInclude);
    if (include_pos != std::string::npos) {
      // 找到 include 行，跳过它，并去掉后面的换行符
      size_t content_start = include_pos + kTilingHeadInclude.length();
      while (content_start < value.size() && (value[content_start] == '\n' || value[content_start] == '\r')) {
        content_start++;
      }
      result += value.substr(content_start);
    } else {
      // 如果没有 include 行，直接拼接整个内容
      result += value;
    }

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }

  return ge::SUCCESS;
}

}  // namespace

std::string RemoveSubDirInclude(const std::string& kernel_str) {
  std::string result = R"(
#include "kernel_operator.h"
)";
  std::stringstream ss(kernel_str);
  std::string line;
  while (std::getline(ss, line)) {
    auto shouldRemove = [&line]() {
      return line.compare(0, kBasicApiInclude.size(), kBasicApiInclude) == 0 ||
             line.compare(0, kAdvApiInclude.size(), kAdvApiInclude) == 0 ||
             line.compare(0, kMicroApiInclude.size(), kMicroApiInclude) == 0 ||
             line.compare(0, kSimtApiInclude.size(), kSimtApiInclude) == 0 ||
             line.compare(0, kUtilsStdInclude.size(), kUtilsStdInclude) == 0;
    };
    if (!shouldRemove()) {
      result += line + "\n";
    }
  }
  return result;
}

Codegen::Codegen(const CodegenOptions &options)
    : tiling_lib_(options.tiling_lib_path, options.tiling_lib_codegen_symbol),
      using_att_calc_qbt_size_(options.using_att_calc_qbt_size) {}

Status Codegen::Generate(const ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result) const {
  // fot UT/ST shape info stub
  std::map<std::string, std::string> shape_info;
  return this->Generate(shape_info, fused_schedule_result, result);
}

// inductor路径不做tiling func文件拆分,把拆分的文件合一
Status Codegen::GenerateForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                    CodegenResult &result) const {
  GE_CHK_STATUS_RET(GenerateKernel(fused_schedule_result, result.kernel, true), "Codegen generate kernel failed");
  result.tiling_data = GenerateTilingData(fused_schedule_result);
  auto tiling_file_name_to_content = GenerateTilingForInductor(fused_schedule_result);
  for (const auto &[key, value] : tiling_file_name_to_content) {
    (void)key;
    GE_CHK_BOOL_RET_STATUS(value != ascgen_utils::INVALID_TILING, ge::FAILED, "tilings(%s) is invalid",
                           value.c_str());
  }
  GE_CHK_STATUS_RET(CombineTilings(tiling_file_name_to_content, result.tiling));
  return ge::SUCCESS;
}

Status Codegen::Generate(const std::map<std::string, std::string> &shape_info,
                         const ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result) const {
  GE_CHK_STATUS_RET(GenerateKernel(fused_schedule_result, result.kernel, false), "Codegen generate kernel failed");
  result.tiling_data = GenerateTilingData(fused_schedule_result);
  auto tiling_file_name_to_content = GenerateTiling(fused_schedule_result, shape_info, "", "0");
  for (const auto &[key, value] : tiling_file_name_to_content) {
    (void)key;
    GE_CHK_BOOL_RET_STATUS(value != ascgen_utils::INVALID_TILING, ge::FAILED, "tilings(%s) is invalid",
                           value.c_str());
  }
  GE_CHK_STATUS_RET(CombineTilings(tiling_file_name_to_content, result.tiling));

  return ge::SUCCESS;
}

std::string Codegen::GenerateTilingData(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  ss << TilingData("Autofuse").Generate(fused_schedule_result);
  return ss.str();
}

std::map<std::string, std::string> Codegen::GenerateTilingForInductor(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  return this->tiling_lib_.GenerateForInductor(fused_schedule_result);
}

std::map<std::string, std::string> Codegen::GenerateTiling(
    const ascir::FusedScheduledResult &fused_schedule_result,
    const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
    const std::string &core_num) const {
  return this->tiling_lib_.Generate(fused_schedule_result, shape_info, pgo_dir, core_num);
}

std::string Codegen::GenerateInferShape(const std::vector<std::vector<std::string>> &symbol_shape_str,
                                        const std::map<std::string, string> &shape_info) const {
  InfershapeGen gen;
  return gen.GenInferShapeFunc(symbol_shape_str, shape_info);
}

std::string Codegen::GeneratorPgo(const ascir::FusedScheduledResult &fused_schedule_result, const std::string &pgo_dir) const {
  return this->tiling_lib_.GenerateForPgo(fused_schedule_result, pgo_dir);
}

Status Codegen::GenerateKernel(const ascir::FusedScheduledResult &fused_schedule_result, std::string &result,
                               bool is_inductor) const {
  const auto io_num = fused_schedule_result.input_nodes.size() + fused_schedule_result.output_nodes.size();
  bool use_list_tensor = io_num >= kMaxUnfoldedIoNum;
  CodegenConfig config = {is_inductor, using_att_calc_qbt_size_};
  if (is_inductor) {
    use_list_tensor = false;
    GE_ASSERT_TRUE(io_num < kKernelMaxIoNum, "Too many io, io num is %zu", io_num);
  }
  std::string graph_name = ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString());
  GELOGI("kernel_name = %s, num_inputs = %zu, num_outputs = %zu, use list tensor desc = %d", graph_name.c_str(),
         fused_schedule_result.input_nodes.size(), fused_schedule_result.output_nodes.size(),
         static_cast<int32_t>(use_list_tensor));
  std::stringstream ss;
  std::string kernel_task_type = GetKernelTaskType(fused_schedule_result);
  ss << Kernel::IncludeAndDefines(fused_schedule_result, kernel_task_type, use_list_tensor, is_inductor);
  GE_CHK_STATUS_RET(Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss, use_list_tensor, config,
                    kernel_task_type), "Generate kernel func by tiling_key failed.");
  if (is_inductor) {
    ss << Kernel::GenKernelFuncCallForInductor(fused_schedule_result);
  }
  result = ss.str();
  return ge::SUCCESS;
}

std::string Codegen::GenGetKernelAndJson(const std::string &kernel_path, const std::string &json_path) const {
  // 当前只获取kernel，后续计划json也会打包起来
  (void)json_path;
  std::stringstream ss;
  std::string real_kernel_path;
  if (!ascgen_utils::GetRealPath(kernel_path, real_kernel_path)) {
    GELOGE(ge::FAILED, "kernel_path::%s realpath failed", kernel_path.c_str());
    return "";
  }
  std::ifstream kernel_file(real_kernel_path, std::ios::binary | std::ios::ate);
  if (!kernel_file.is_open()) {
    GELOGE(ge::FAILED, "kernel_path::%s open failed", kernel_path.c_str());
    return "";
  }

  std::streamsize kernel_file_size = kernel_file.tellg();
  kernel_file.seekg(0, std::ios::beg);
  std::vector<uint8_t> kernel_data(kernel_file_size);
  kernel_file.read(reinterpret_cast<char *>(kernel_data.data()), kernel_file_size);
  kernel_file.close();

  ss << "#include <cstdint>" << std::endl;
  ss << "#include <cstring>" << std::endl;
  ss << "#include <vector>" << std::endl;
  ss << "extern \"C\" void GetKernelBin(std::vector<char> &kernel_bin) {" << std::endl;
  if (kernel_file_size == 0) {
    ss << "  std::vector<uint8_t> temp_kernel = {};" << std::endl;
    ss << "  return;" << std::endl;
  } else {
    ss << "  std::vector<uint8_t> temp_kernel = {";
    for (uint32_t i = 0; i < kernel_file_size; i++) {
      if (i % ELEMENTS_PER_LINE == 0) {
        ss << std::endl << "    ";
      }
      ss << std::to_string(kernel_data[i]) << ", ";
    }
    ss << "};" << std::endl;

    ss << "  kernel_bin.resize(temp_kernel.size());" << std::endl;
    ss << "  std::memcpy(kernel_bin.data(), temp_kernel.data(), temp_kernel.size() * sizeof(uint8_t));" << std::endl;
  }
  ss << "}";
  return ss.str();
}
