/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_RESULT_CHECKER_UTILS_H
#define ATT_RESULT_CHECKER_UTILS_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
namespace att {
class ResultCheckerUtils {
 public:
  static std::string DefineCheckerFunction();
  static bool IsFileContainsString(const std::string &filename, const std::string &search_sub_string);
  static bool ReplaceLogMacros(const std::string& filename = "autofuse_tiling_func_common.h");
  static bool ReplaceLogMacrosGeneric(const std::string& filename = "autofuse_tiling_func_common.h");
  static std::string GetDependAscendIncPath();

 private:
  static bool ReadFileLines(const std::string &filename, std::vector<std::string> &lines);
  static bool CreateBackupAndWrite(const std::string &filename, const std::vector<std::string> &lines);
  static bool FinalizeLogFileReplacement(const std::string &filename, const std::vector<std::string> &lines, const std::string &log_type = "log macros");
};
}  // namespace att
#endif  // ATT_RESULT_CHECKER_UTILS_H