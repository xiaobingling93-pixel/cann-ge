/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_FE_ENV_UTILS_H_
#define AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_FE_ENV_UTILS_H_
#include "common/fe_type_utils.h"
#include "google/protobuf/text_format.h"
#include <iostream>
#include <fstream>

namespace fe_env {
class FeEnvUtils {
  public:
  static std::string GetAscendPath() {
    const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
    string ascend_path = "/mnt/d/Ascend";
    if (ascend_custom_path_ptr != nullptr) {
        ascend_path = fe::GetRealPath(string(ascend_custom_path_ptr));
    } else {
        const char *ascend_home_path_ptr = std::getenv("ASCEND_HOME");
        if (ascend_home_path_ptr != nullptr) {
        ascend_path = fe::GetRealPath(string(ascend_home_path_ptr));
        } else {
        ascend_path = "/mnt/d/Ascend";
        }
    }
    return ascend_path;
  }

  static std::string GetFFTSLogFile() {
    std::string ascend_path = GetAscendPath();
    std::string file_name = ascend_path + "/ffts_st";
    return file_name;
  }

  static void SaveTaskDefContextToFile(std::vector<domi::TaskDef> &task_defs) {
    string task_def_str = "";
    for (auto task_iter : task_defs) {
      string task_def_str_temp;
      google::protobuf::TextFormat::PrintToString(task_iter, &task_def_str_temp);
      task_def_str.append(task_def_str_temp);
    }
    
    std::string ascend_path = GetAscendPath();
    std::string file_name = ascend_path + "/task_def_context.txt";
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
      std::cout << "open to write task_context failed!" << std::endl;
    }
    ofs << task_def_str;
    ofs.close();
  }

  static std::string GetTaskDefStr(std::string task_def_path) {
    std::ifstream ifs;
    ifs.open(task_def_path, std::ios::in);
    if (!ifs.is_open()) {
      std::cout << "open task_context failed!" << std::endl;
    }
    std::string task_def_str;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    task_def_str = oss.str();
    ifs.close();
    return task_def_str;
  }
  static void CopyFile(const std::string &src_file, const std::string &dst_file) {
    std::ifstream in;
    std::ofstream out;
    try {
      in.open(src_file, std::ios::binary);
      if (in.fail()) {
        std::cout << "Failed to open src file " << src_file << std::endl;
        in.close();
        out.close();
        return;
      }
      out.open(dst_file, std::ios::binary);
      if (out.fail()) {
        std::cout << "Failed to open dst file " << dst_file << std::endl;
        in.close();
        out.close();
        return;
      }
      out << in.rdbuf();
      out.close();
      in.close();
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
};
}
#endif
