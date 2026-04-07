/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include "te_llt_utils.h"

#define private public
#define protected public
#include "common/te_config_info.h"
#include "python_adapter/python_adapter_manager.h"
#undef protected public
#undef private public

using namespace testing;
namespace te {
namespace fusion {
class PythonAdapterManagerUT : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "PythonAdapterManagerUT SetUpTestSuite" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "PythonAdapterManagerUT TearDownTestSuite" << std::endl;
    }

protected:
    void TearDown() override {
        std::cout << "PythonAdapterManagerUT TearDown" << std::endl;
        InitPyHandleStub();
    }
};

TEST(PythonAdapterManagerUT, init_and_finalize_case1) {
    PythonAdapterManager pythonAdapterManager;
    EXPECT_EQ(pythonAdapterManager.Initialize(), true);
    EXPECT_EQ(pythonAdapterManager.Initialize(), true);
    EXPECT_EQ(pythonAdapterManager.IsInitParallelCompilation(), true);
    EXPECT_EQ(pythonAdapterManager.Finalize(), true);
    EXPECT_EQ(pythonAdapterManager.Finalize(), true);
    InitPyHandleStub();
}

TEST(PythonAdapterManagerUT, init_and_finalize_case2) {
    TeConfigInfo::Instance().isOpCompileDisabled_ = true;
    TeConfigInfo::Instance().isUbFusionDisabled_ = true;
    PythonAdapterManager pythonAdapterManager;
    EXPECT_EQ(pythonAdapterManager.Initialize(), true);
    EXPECT_EQ(pythonAdapterManager.IsInitParallelCompilation(), false);
    EXPECT_EQ(pythonAdapterManager.Finalize(), true);
    TeConfigInfo::Instance().isOpCompileDisabled_ = false;
    TeConfigInfo::Instance().isUbFusionDisabled_ = false;
    InitPyHandleStub();
}

TEST(PythonAdapterManagerUT, process_num_check_case) {
    PythonAdapterManager pythonAdapterManager;
    TeConfigInfo::Instance().env_item_vec_[static_cast<size_t>(TeConfigInfo::EnvItem::TeParallelCompiler)] = "7";
    pythonAdapterManager.ParallelCompilerProcessesNumCheck();
    TeConfigInfo::Instance().env_item_vec_[static_cast<size_t>(TeConfigInfo::EnvItem::TeParallelCompiler)] = "9";
    pythonAdapterManager.ParallelCompilerProcessesNumCheck();
    TeConfigInfo::Instance().env_item_vec_[static_cast<size_t>(TeConfigInfo::EnvItem::TeParallelCompiler)] = "abc";
    pythonAdapterManager.ParallelCompilerProcessesNumCheck();
}

TEST(PythonAdapterManagerUT, get_finished_compilation_task) {
    PythonAdapterManager pythonAdapterManager;
    EXPECT_EQ(pythonAdapterManager.Initialize(), true);
    std::vector<OpBuildTaskResultPtr> taskRetVec;
    bool ret = pythonAdapterManager.GetFinishedCompilationTask(100, taskRetVec);
    EXPECT_EQ(ret , true);
    EXPECT_EQ(pythonAdapterManager.Finalize(), true);
    InitPyHandleStub();
}

TEST(PythonAdapterManagerUT, parse_op_task_result_case0) {
    PyObject *pyRes = nullptr;
    OpBuildTaskResultPtr opBuildTaskResultPtr = PythonAdapterManager::ParseOpTaskResult(pyRes);
    EXPECT_EQ(opBuildTaskResultPtr, nullptr);
    PyObject *pySocParamDict = HandleManager::Instance().TE_PyDict_New();
    opBuildTaskResultPtr = PythonAdapterManager::ParseOpTaskResult(pySocParamDict);
    EXPECT_NE(opBuildTaskResultPtr, nullptr);
}

TEST(PythonAdapterManagerUT, parse_result_case0) {
    std::string jsonFilePath;
    std::string pattern;
    std::string coreType;
    std::string compileInfo;
    std::string compileInfoKey;
    int fusionCheckResCode = 0;
    bool ret = PythonAdapterManager::ParseResult(nullptr, 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, false);
    std::string opResCompile = "{\"pattern\":\"Opaque\",\"core_type\":\"AiCore\"}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(pattern, "Opaque");
    EXPECT_EQ(coreType, "AiCore");

    opResCompile = "{\"core_type\":\"AiCore\"}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, false);
    opResCompile = "{\"pattern\":\"Opaque\"}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, true);
}

TEST(PythonAdapterManagerUT, parse_result_case1) {
    std::string jsonFilePath;
    std::string pattern;
    std::string coreType;
    std::string compileInfo;
    std::string compileInfoKey;
    int fusionCheckResCode = 0;
    bool ret = PythonAdapterManager::ParseResult(nullptr, 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, false);
    std::string opResCompile = "{\"pattern\":\"Opaque\",\"fusion_check_result\":1}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, true);

    opResCompile = "{\"json_file_path\":\"\",\"fusion_check_result\":1}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, false);

    opResCompile = "{\"json_file_path\":\"xxx_path\",\"fusion_check_result\":1}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(jsonFilePath, "xxx_path");
    EXPECT_EQ(fusionCheckResCode, 1);

    opResCompile = "{\"json_file_path\":\"xxx_path\",\"compile_info\":\"compile_info_xxx\"}";
    ret = PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckResCode);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(jsonFilePath, "xxx_path");
    EXPECT_EQ(compileInfo, "\"compile_info_xxx\"");
}
}
}