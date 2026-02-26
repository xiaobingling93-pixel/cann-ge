/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_
#define TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_

#include <cstdlib>
#include <string>

namespace autofuse {
namespace test {

// 清理测试用例生成的临时文件和stub拷贝
inline void CleanupTestArtifacts() {
  // 删除stub相关目录和文件
  system("rm -rf ./stub ./tiling ./register ./graph ./lib ./kernel_tiling");
  // 删除公共文件
  system("rm -f ./*.h");
  // 删除日志文件
  system("rm -f *.log");
  // 删除生成的二进制文件
  system("rm -f ./tiling_func_main ./tiling_func_main_concat ./tiling_func_main_transpose ./tiling_func_main_softmax");
  // 删除生成的tiling data和func文件
  system("rm -f ./*_tiling_func.cpp ./tiling_func_main_*.cpp");

  // 清理build根目录下可能残留的tiling和register目录
  // 测试在build/tests/autofuse/st/att下运行，build根目录在../../
  system("rm -rf ../../tiling ../../register 2>/dev/null");
}

// 拷贝stub文件到tiling和register目录
// base_dir: 基础目录（如 ST_DIR, UT_DIR, TOP_DIR）
// stub_path_prefix: stub文件的相对路径前缀
//   - ST测试: "testcase/stub/"
//   - UT测试: "testcase/stub/"
//   - TOP_DIR: "tests/autofuse/st/att/testcase/stub/"
inline int CopyStubFiles(const std::string& base_dir, const std::string& stub_path_prefix) {
  // 创建目录
  (void)std::system("mkdir -p ./tiling ./register");

  // 拷贝4个stub文件
  int ret = 0;
  ret = std::system(std::string("cp ").append(base_dir).append("/").append(stub_path_prefix).append("platform_ascendc.h ./tiling/ -f").c_str());
  ret = std::system(std::string("cp ").append(base_dir).append("/").append(stub_path_prefix).append("tiling_api.h ./tiling/ -f").c_str());
  ret = std::system(std::string("cp ").append(base_dir).append("/").append(stub_path_prefix).append("tiling_context.h ./tiling/ -f").c_str());
  ret = std::system(std::string("cp ").append(base_dir).append("/").append(stub_path_prefix).append("tilingdata_base.h ./register/ -f").c_str());

  return ret;
}

}  // namespace test
}  // namespace autofuse

#endif  // TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_
