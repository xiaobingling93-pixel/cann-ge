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

#define private public
#define protected public
#include "common/te_config_info.h"
#include "graph/ge_local_context.h"
#undef protected public
#undef private public
#include "ge_common/ge_api_types.h"
using namespace std;
using namespace testing;
namespace te {
namespace fusion {
class TeConfigInfoST : public testing::Test {
protected:
    static void SetUpTestSuite() {
        std::cout << "TeConfigInfoST SetUpTestSuite" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "TeConfigInfoST TearDownTestSuite" << std::endl;
    }
};

TEST(TeConfigInfoST, init_lib_path)
{
    TeConfigInfo teConfigInfo;
    map<string, string> options;
    EXPECT_EQ(teConfigInfo.Initialize(options), true);
    EXPECT_EQ(teConfigInfo.GetLibPath().empty(), false);
    std::cout << "Current lib path = " << teConfigInfo.GetLibPath() << std::endl;
    //EXPECT_EQ(teConfigInfo.IsBinaryInstalled(), false);
    EXPECT_EQ(teConfigInfo.GetUniqueKernelMetaHash().empty(), false);
    std::cout << "UniqueKernelMetaHash = " << teConfigInfo.GetUniqueKernelMetaHash() << std::endl;
    EXPECT_EQ(teConfigInfo.GetKernelMetaParentDir().empty(), false);
    std::cout << "KernelMetaParentDir = " << teConfigInfo.GetKernelMetaParentDir() << std::endl;
    EXPECT_EQ(teConfigInfo.GetKernelMetaTempDir().empty(), false);
    std::cout << "KernelMetaTempDir = " << teConfigInfo.GetKernelMetaTempDir() << std::endl;
}

TEST(TeConfigInfoST, init_env_param)
{
    TeConfigInfo teConfigInfo;
    map<string, string> options;
    EXPECT_EQ(teConfigInfo.Initialize(options), true);
    EXPECT_EQ(teConfigInfo.GetEnvHome().empty(), false);
    EXPECT_EQ(teConfigInfo.GetEnvPath().empty(), false);

    EXPECT_EQ(teConfigInfo.GetEnvNpuCollectPath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvParaDebugPath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvTeParallelCompiler().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvTeNewDfxinfo().empty(), true);
    // EXPECT_EQ(teConfigInfo.GetEnvMinCompileResourceUsageCtrl().empty(), true);
    //EXPECT_EQ(teConfigInfo.GetEnvOppPath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvWorkPath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvCachePath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvMaxOpCacheSize().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvRemainCacheSizeRatio().empty(), true);
    EXPECT_EQ(teConfigInfo.GetEnvOpCompilerWorkPathInKernelMeta().empty(), true);

    setenv("NPU_COLLECT_PATH", "npu_collect_path", 1);
    setenv("PARA_DEBUG_PATH", "para_debug_path", 1);
    setenv("TE_PARALLEL_COMPILER", "8", 1);
    setenv("TEFUSION_NEW_DFXINFO", "tefusion_new_dfxinfo", 1);
    setenv("MIN_COMPILE_RESOURCE_USAGE_CTRL", "ub_fusion,op_compile", 1);
    setenv("ASCEND_OPP_PATH", teConfigInfo.GetLibPath().c_str(), 1);
    setenv("ASCEND_WORK_PATH", "ascend_work_path", 1);
    setenv("ASCEND_CACHE_PATH", "ascend_cache_path", 1);
    setenv("ASCEND_MAX_OP_CACHE_SIZE", "1024", 1);
    setenv("ASCEND_REMAIN_CACHE_SIZE_RATIO", "85", 1);
    setenv("ASCEND_OP_COMPILER_WORK_PATH_IN_KERNEL_META", "op_compiler_work_path", 1);
    teConfigInfo.InitEnvItem();
    EXPECT_EQ(teConfigInfo.GetEnvNpuCollectPath(), "npu_collect_path");
    EXPECT_EQ(teConfigInfo.GetEnvParaDebugPath(), "para_debug_path");
    EXPECT_EQ(teConfigInfo.GetEnvTeParallelCompiler(), "8");
    EXPECT_EQ(teConfigInfo.GetEnvTeNewDfxinfo(), "tefusion_new_dfxinfo");
    EXPECT_EQ(teConfigInfo.GetEnvMinCompileResourceUsageCtrl(), "ub_fusion,op_compile");
    EXPECT_EQ(teConfigInfo.IsDisableUbFusion(), true);
    EXPECT_EQ(teConfigInfo.IsDisableOpCompile(), true);
    EXPECT_EQ(teConfigInfo.GetEnvOppPath(), teConfigInfo.GetLibPath());
    EXPECT_EQ(teConfigInfo.GetOppRealPath().empty(), false);
    EXPECT_EQ(teConfigInfo.GetEnvWorkPath(), "ascend_work_path");
    EXPECT_EQ(teConfigInfo.GetEnvCachePath(), "ascend_cache_path");
    EXPECT_EQ(teConfigInfo.GetEnvMaxOpCacheSize(), "1024");
    EXPECT_EQ(teConfigInfo.GetEnvRemainCacheSizeRatio(), "85");
    EXPECT_EQ(teConfigInfo.GetEnvOpCompilerWorkPathInKernelMeta(), "op_compiler_work_path");
    unsetenv("NPU_COLLECT_PATH");
    unsetenv("PARA_DEBUG_PATH");
    unsetenv("TE_PARALLEL_COMPILER");
    unsetenv("TEFUSION_NEW_DFXINFO");
    unsetenv("MIN_COMPILE_RESOURCE_USAGE_CTRL");
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("ASCEND_WORK_PATH");
    unsetenv("ASCEND_CACHE_PATH");
    unsetenv("ASCEND_MAX_OP_CACHE_SIZE");
    unsetenv("ASCEND_REMAIN_CACHE_SIZE_RATIO");
    unsetenv("ASCEND_OP_COMPILER_WORK_PATH_IN_KERNEL_META");
}

TEST(TeConfigInfoST, init_param_case1)
{
    TeConfigInfo teConfigInfo;
    map<string, string> options;
    EXPECT_EQ(teConfigInfo.Initialize(options), true);
    EXPECT_EQ(teConfigInfo.GetCompileCacheMode(), CompileCacheMode::Enable);
    EXPECT_EQ(teConfigInfo.GetSocVersion().empty(), true);
    EXPECT_EQ(teConfigInfo.GetShortSocVersion().empty(), true);
    EXPECT_EQ(teConfigInfo.GetAiCoreNum().empty(), true);
    EXPECT_EQ(teConfigInfo.GetCoreType().empty(), true);
    EXPECT_EQ(teConfigInfo.GetDeviceId().empty(), true);
    EXPECT_EQ(teConfigInfo.GetFpCeilingMode().empty(), true);
    EXPECT_EQ(teConfigInfo.GetL1Fusion(), "false");
    EXPECT_EQ(teConfigInfo.GetL2Fusion(), "false");
    EXPECT_EQ(teConfigInfo.GetL2Mode().empty(), true);
    EXPECT_EQ(teConfigInfo.GetMdlBankPath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetOpBankPath().empty(), true);
    EXPECT_EQ(teConfigInfo.GetOpDebugLevel(), OpDebugLevel::Disable);
    EXPECT_EQ(teConfigInfo.GetOpDebugLevelStr(), "0");
    EXPECT_EQ(teConfigInfo.GetOpDebugConfig().empty(), true);
    EXPECT_EQ(teConfigInfo.GetCompileCacheDir().empty(), true);

    options[ge::OP_COMPILER_CACHE_MODE] = "force";
    options[ge::SOC_VERSION] = "Ascend910B1";
    options["short_soc_version"] = "Ascend910B";
    options[ge::AICORE_NUM] = "8";
    options[ge::BUFFER_OPTIMIZE] = "l2_optimize";
    options[ge::CORE_TYPE] = "AiCore";
    options[ge::OPTION_EXEC_DEVICE_ID] = "0";
    options["l2_mode"] = "1";
    options["ge.fpCeilingMode"] = "1";
    options[ge::MDL_BANK_PATH_FLAG] = "mdl_bank_path";
    options[ge::OP_BANK_PATH_FLAG] = "op_bank_path";
    options[ge::OP_DEBUG_LEVEL] = "3";
    options["op_debug_config_te"] = "dump_cce";
    options[ge::OP_COMPILER_CACHE_DIR] = teConfigInfo.GetLibPath();
    ge::GetThreadLocalContext().SetGraphOption(options);
    EXPECT_EQ(teConfigInfo.RefreshConfigItems(), true);
    EXPECT_EQ(teConfigInfo.GetCompileCacheMode(), CompileCacheMode::Force);
    EXPECT_EQ(teConfigInfo.GetSocVersion(), "Ascend910B1");
    EXPECT_EQ(teConfigInfo.GetShortSocVersion(), "Ascend910B");
    EXPECT_EQ(teConfigInfo.GetAiCoreNum(), "8");
    EXPECT_EQ(teConfigInfo.GetCoreType(), "AiCore");
    EXPECT_EQ(teConfigInfo.GetDeviceId(), "0");
    EXPECT_EQ(teConfigInfo.GetFpCeilingMode(), "1");
    EXPECT_EQ(teConfigInfo.GetL1Fusion(), "false");
    EXPECT_EQ(teConfigInfo.GetL2Fusion(), "true");
    EXPECT_EQ(teConfigInfo.GetL2Mode(), "1");
    EXPECT_EQ(teConfigInfo.GetMdlBankPath(), "mdl_bank_path");
    EXPECT_EQ(teConfigInfo.GetOpBankPath(), "op_bank_path");
    EXPECT_EQ(teConfigInfo.GetOpDebugLevel(), OpDebugLevel::Level3);
    EXPECT_EQ(teConfigInfo.GetOpDebugConfig(), "dump_cce");
    EXPECT_EQ(teConfigInfo.GetCompileCacheDir(), teConfigInfo.GetLibPath());
    options.clear();
    ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST(TeConfigInfoST, init_param_case2)
{
    TeConfigInfo teConfigInfo;
    map<string, string> options;
    options[ge::OP_COMPILER_CACHE_MODE] = "force";
    options[ge::SOC_VERSION] = "Ascend910B1";
    options["short_soc_version"] = "Ascend910B";
    options[ge::AICORE_NUM] = "8";
    options[ge::BUFFER_OPTIMIZE] = "l2_optimize";
    options[ge::CORE_TYPE] = "AiCore";
    options[ge::OPTION_EXEC_DEVICE_ID] = "0";
    options["l2_mode"] = "1";
    options["ge.fpCeilingMode"] = "1";
    options[ge::MDL_BANK_PATH_FLAG] = "mdl_bank_path";
    options[ge::OP_BANK_PATH_FLAG] = "op_bank_path";
    options[ge::OP_DEBUG_LEVEL] = "3";
    options["op_debug_config_te"] = "dump_cce";
    options[ge::OP_COMPILER_CACHE_DIR] = "./cache_dir";
    EXPECT_EQ(teConfigInfo.Initialize(options), true);
    EXPECT_EQ(teConfigInfo.GetCompileCacheMode(), CompileCacheMode::Force);
    EXPECT_EQ(teConfigInfo.GetSocVersion(), "Ascend910B1");
    EXPECT_EQ(teConfigInfo.GetShortSocVersion(), "Ascend910B");
    EXPECT_EQ(teConfigInfo.GetAiCoreNum(), "8");
    EXPECT_EQ(teConfigInfo.GetCoreType(), "AiCore");
    EXPECT_EQ(teConfigInfo.GetDeviceId(), "0");
    EXPECT_EQ(teConfigInfo.GetFpCeilingMode(), "1");
    EXPECT_EQ(teConfigInfo.GetL1Fusion(), "false");
    EXPECT_EQ(teConfigInfo.GetL2Fusion(), "true");
    EXPECT_EQ(teConfigInfo.GetL2Mode(), "1");
    EXPECT_EQ(teConfigInfo.GetMdlBankPath(), "mdl_bank_path");
    EXPECT_EQ(teConfigInfo.GetOpBankPath(), "op_bank_path");
    EXPECT_EQ(teConfigInfo.GetOpDebugLevel(), OpDebugLevel::Level3);
    EXPECT_EQ(teConfigInfo.GetOpDebugConfig(), "dump_cce");
    EXPECT_EQ(teConfigInfo.GetCompileCacheDir(), "./cache_dir");

    EXPECT_EQ(teConfigInfo.RefreshConfigItems(), true);
    EXPECT_EQ(teConfigInfo.GetCompileCacheMode(), CompileCacheMode::Force);
    EXPECT_EQ(teConfigInfo.GetSocVersion(), "Ascend910B1");
    EXPECT_EQ(teConfigInfo.GetOpDebugLevel(), OpDebugLevel::Level3);
    EXPECT_EQ(teConfigInfo.GetOpDebugConfig(), "dump_cce");
    EXPECT_EQ(teConfigInfo.GetCompileCacheDir(), "./cache_dir");

    options[ge::OP_COMPILER_CACHE_MODE] = "disable";
    options[ge::SOC_VERSION] = "Ascend910B2";
    options["short_soc_version"] = "Ascend910B";
    options[ge::AICORE_NUM] = "10";
    options[ge::BUFFER_OPTIMIZE] = "l2_optimize";
    options[ge::CORE_TYPE] = "VectorCore";
    options[ge::OPTION_EXEC_DEVICE_ID] = "5";
    options["l2_mode"] = "1";
    options["ge.fpCeilingMode"] = "0";
    options[ge::MDL_BANK_PATH_FLAG] = "mdl_bank_path";
    options[ge::OP_BANK_PATH_FLAG] = "op_bank_path";
    options[ge::OP_DEBUG_LEVEL] = "4";
    options["op_debug_config_te"] = "dump_cce,dump_bin";
    options[ge::OP_COMPILER_CACHE_DIR] = teConfigInfo.GetLibPath();
    ge::GetThreadLocalContext().SetGraphOption(options);
    EXPECT_EQ(teConfigInfo.InitConfigItemsFromContext(), true);
    EXPECT_EQ(teConfigInfo.GetCompileCacheMode(), CompileCacheMode::Disable);
    EXPECT_EQ(teConfigInfo.GetSocVersion(), "Ascend910B2");
    EXPECT_EQ(teConfigInfo.GetShortSocVersion(), "Ascend910B");
    EXPECT_EQ(teConfigInfo.GetAiCoreNum(), "10");
    EXPECT_EQ(teConfigInfo.GetCoreType(), "VectorCore");
    EXPECT_EQ(teConfigInfo.GetDeviceId(), "5");
    EXPECT_EQ(teConfigInfo.GetFpCeilingMode(), "0");
    EXPECT_EQ(teConfigInfo.GetL1Fusion(), "false");
    EXPECT_EQ(teConfigInfo.GetL2Fusion(), "true");
    EXPECT_EQ(teConfigInfo.GetL2Mode(), "1");
    EXPECT_EQ(teConfigInfo.GetMdlBankPath(), "mdl_bank_path");
    EXPECT_EQ(teConfigInfo.GetOpBankPath(), "op_bank_path");
    EXPECT_EQ(teConfigInfo.GetOpDebugLevel(), OpDebugLevel::Level4);
    EXPECT_EQ(teConfigInfo.GetOpDebugConfig(), "dump_cce,dump_bin");
    EXPECT_EQ(teConfigInfo.GetCompileCacheDir(), teConfigInfo.GetLibPath());
    options.clear();
    ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST(TeConfigInfoST, init_param_case3)
{
    TeConfigInfo teConfigInfo;
    map<string, string> options;
    options[ge::OP_COMPILER_CACHE_MODE] = "enabl";
    EXPECT_EQ(teConfigInfo.Initialize(options), false);
    EXPECT_EQ(teConfigInfo.InitConfigItemsFromOptions(options), false);

    options[ge::OP_COMPILER_CACHE_MODE] = "disabl";
    ge::GetThreadLocalContext().SetGraphOption(options);
    EXPECT_EQ(teConfigInfo.RefreshConfigItems(), false);
    options.clear();
    ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST(TeConfigInfoST, init_param_case4)
{
    TeConfigInfo teConfigInfo;
    map<string, string> options;
    options[ge::OP_DEBUG_LEVEL] = "5";
    EXPECT_EQ(teConfigInfo.Initialize(options), false);
    EXPECT_EQ(teConfigInfo.InitConfigItemsFromOptions(options), false);

    options[ge::OP_DEBUG_LEVEL] = "-1";
    ge::GetThreadLocalContext().SetGraphOption(options);
    EXPECT_EQ(teConfigInfo.RefreshConfigItems(), false);
    options.clear();
    ge::GetThreadLocalContext().SetGraphOption(options);
}
}
}