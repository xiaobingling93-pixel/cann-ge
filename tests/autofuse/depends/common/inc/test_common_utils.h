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
#include <map>

// 前置声明
namespace ge {
namespace ascir {
class AscGraph;
class ScheduleGroup;
class FusedScheduledResult;
}
}

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

// 辅助函数：构建单图到ScheduleGroup
// 需要前置声明相关类型，使用时需包含对应头文件
class AscGraph;
class ScheduleGroup;

inline void BuildSingleGraphToScheduleGroup(
    AscGraph& graph,
    ScheduleGroup& schedule_group,
    uint32_t tiling_key);

// 辅助函数：生成tiling函数并写文件
// 需要前置声明相关类型，使用时需包含对应头文件
class FusedScheduledResult;

inline void GenerateTilingFunctionAndWriteToFile(
    const std::string& op_name,
    const FusedScheduledResult& fused_scheduled_result,
    std::map<std::string, std::string>& options);

// 辅助函数：生成tiling数据和头文件
inline void GenerateTilingDataAndHeader(
    const std::string& op_name,
    const std::string& graph_name,
    const FusedScheduledResult& fused_scheduled_result,
    std::map<std::string, std::string>& options);

// 辅助函数：准备测试环境文件
inline void PrepareTestEnvironmentFiles(const std::string& test_header_content = "");

// 辅助函数：编译生成的tiling代码
inline void CompileGeneratedTilingCode();

}  // namespace test
}  // namespace autofuse

#endif  // TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_
