/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_CODE_PRINTER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_CODE_PRINTER_H_
#include <string>
#include <sstream>
#include "ge_common/ge_api_error_codes.h"

namespace ge {
enum class GeneratedFileIndex {
  kKernelRegistryFile = 0U,
  kResourcesFile,
  kArgsManagerFile,
  kLoadingAndRunningFile,
  kInterfaceHeaderFile,
  kCMakeListsFile,
  kEnd
};
struct GeneratedFileInfo {
  std::string file_name;
  std::string file_path;
  std::stringstream content;
};
class Om2CodePrinter {
 public:
  Om2CodePrinter(const std::string &model_name, const std::string &base_dir) {
    output_.resize(static_cast<size_t>(GeneratedFileIndex::kEnd));
    InitDefaultFileInfo(model_name, base_dir);
  }
  ~Om2CodePrinter() = default;
  void AddLine(GeneratedFileIndex generated_file_index, const std::string &input_string);
  Status WriteFiles(const std::string &target_path);
  void GetOutputFilePaths(std::vector<std::string> &file_paths);

 private:
  void SetFileInfo(GeneratedFileIndex generated_file_index, const std::string &file_name, const std::string &base_dir);
  void InitDefaultFileInfo(const std::string &model_name, const std::string &base_dir);

 private:
  std::vector<GeneratedFileInfo> output_;
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_CODE_PRINTER_H_