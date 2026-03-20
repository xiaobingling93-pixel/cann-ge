/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "om2_code_printer.h"
#include "common/checker.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {

void Om2CodePrinter::AddLine(GeneratedFileIndex generated_file_index, const std::string &input_string) {
  output_[static_cast<size_t>(generated_file_index)].content << input_string << std::endl;
}

void Om2CodePrinter::SetFileInfo(GeneratedFileIndex generated_file_index, const std::string &file_name,
                                 const std::string &base_dir) {
  output_[static_cast<size_t>(generated_file_index)].file_name = file_name;
  output_[static_cast<size_t>(generated_file_index)].file_path = base_dir + file_name;
}

void Om2CodePrinter::InitDefaultFileInfo(const std::string &model_name, const std::string &base_dir) {
  SetFileInfo(GeneratedFileIndex::kInterfaceHeaderFile, model_name + "_interface.h", base_dir);
  SetFileInfo(GeneratedFileIndex::kResourcesFile, model_name + "_resources.cpp", base_dir);
  SetFileInfo(GeneratedFileIndex::kArgsManagerFile, model_name + "_args_manager.cpp", base_dir);
  SetFileInfo(GeneratedFileIndex::kKernelRegistryFile, model_name + "_kernel_reg.cpp", base_dir);
  SetFileInfo(GeneratedFileIndex::kLoadingAndRunningFile, model_name + "_load_and_run.cpp", base_dir);
  SetFileInfo(GeneratedFileIndex::kCMakeListsFile, "Makefile", base_dir);
}

Status Om2CodePrinter::WriteFiles(const std::string &target_path) {
  const std::string real_target_path = RealPath(target_path.c_str());
  GE_ASSERT_TRUE(!real_target_path.empty(), "[OM2] Failed to get real path for output directory: %s",
                 target_path.c_str());
  const std::string normalized_target_path =
      (real_target_path.back() == '/') ? real_target_path : real_target_path + "/";
  for (const auto &generated_file_info : output_) {
    if (generated_file_info.file_name.empty()) {
      continue;
    }
    const auto &file_path = normalized_target_path + generated_file_info.file_name;
    const auto &file_content = generated_file_info.content.str();
    std::ofstream om2_file(file_path);
    GE_ASSERT_TRUE(om2_file.good(), "Failed to open file: %s", file_path.c_str());
    om2_file << file_content << std::endl;
    GELOGI("[OM2] File %s is successfully written.", file_path.c_str());
  }
  return SUCCESS;
}

void Om2CodePrinter::GetOutputFilePaths(std::vector<std::string> &file_paths) {
  file_paths.clear();
  for (auto &generated_file_info : output_) {
    file_paths.push_back(generated_file_info.file_path);
  }
}
}  // namespace ge
