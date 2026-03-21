/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <filesystem>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/ChainingMockHelper.h>
#include "ge_common/ge_api_types.h"
#include "common/common_utils.h"
#include "python_adapter/python_api_call.h"
#include "common/compile_result_utils.h"

#define private public
#define protected public
#include "common/te_config_info.h"
#include "cache/te_cache_manager.h"
#include "../te_fusion_base.h"
#undef protected public
#undef private public

using namespace std;
using namespace testing;
namespace fs = std::filesystem;
namespace te {
namespace fusion {
class TeCacheManagerSTest : public testing::Test
{
protected:
    static void SetUpTestSuite() {
        std::cout << "TeCacheManagerSTest SetUpTestSuite" << std::endl;
        TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
        TeCacheManager::Instance().cache_dir_path_ = RealPath(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/atc_data/kernel_cache");
        if (TeCacheManager::Instance().cache_dir_path_.empty()) {
            TeCacheManager::Instance().cache_dir_path_ = RealPath("../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache");
        }
        if (TeCacheManager::Instance().cache_dir_path_.empty()) {
            TeCacheManager::Instance().cache_dir_path_ = RealPath("../../../../../../../../../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache");
        }
        if (TeCacheManager::Instance().cache_dir_path_.empty()) {
            TeCacheManager::Instance().cache_dir_path_ = RealPath("../../../../../../../../../../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache");
        }
        std::cout << "cache dir:" << TeCacheManager::Instance().cache_dir_path_ << std::endl;
        TeCacheManager::Instance().precompile_ret_cache_map_.clear();
        TeCacheManager::Instance().compile_ret_cache_map_.clear();
    }

    static void TearDownTestSuite() {
        std::cout << "TeCacheManagerSTest TearDownTestSuite" << std::endl;
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_01)
{
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "enable");
    options.emplace(ge::OP_DEBUG_LEVEL, "1");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);

    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), true);
    EXPECT_EQ(te_cache_manager.cache_mode_, CompileCacheMode::Disable);
    TeConfigInfo::Instance().config_enum_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigEnumItem::OpDebugLevel)] = 0;
}

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_02)
{
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "enable");
    options.emplace("op_debug_config_te", "1");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);

    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), true);
    EXPECT_EQ(te_cache_manager.cache_mode_, CompileCacheMode::Disable);
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::OpDebugConfig)].clear();
}

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_03)
{
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "enable");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);

    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), true);
    EXPECT_EQ(te_cache_manager.cache_mode_, CompileCacheMode::Enable);
    EXPECT_EQ(te_cache_manager.cache_parent_dir_path_.empty(), true);
    EXPECT_NE(te_cache_manager.cache_dir_path_.find("/atc_data/kernel_cache/Ascend910B1"), std::string::npos);
}

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_04)
{
    TeConfigInfo::Instance().env_item_vec_[static_cast<size_t>(TeConfigInfo::EnvItem::CachePath)] = "./te_data";
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "enable");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);

    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), true);
    EXPECT_EQ(te_cache_manager.cache_mode_, CompileCacheMode::Enable);
    EXPECT_EQ(te_cache_manager.cache_parent_dir_path_.empty(), true);
    EXPECT_NE(te_cache_manager.cache_dir_path_.find("/te_data/kernel_cache/Ascend910B1"), std::string::npos);
}

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_05)
{
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "enable");
    options.emplace(ge::OP_COMPILER_CACHE_DIR, "./te_cache");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);

    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), true);
    EXPECT_EQ(te_cache_manager.cache_mode_, CompileCacheMode::Enable);
    EXPECT_EQ(te_cache_manager.cache_parent_dir_path_.empty(), false);
    EXPECT_NE(te_cache_manager.cache_dir_path_.find("/te_cache/kernel_cache/Ascend910B1"), std::string::npos);
}

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_06)
{
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "enable");
    options.emplace(ge::OP_COMPILER_CACHE_DIR, "./te_cache$");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);

    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), false);
}

TEST(TeCacheManagerSTest, init_cache_mode_and_dir_07)
{
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    options.emplace(ge::OP_COMPILER_CACHE_MODE, "disable");
    options.emplace(ge::OP_DEBUG_LEVEL, "0");
    EXPECT_EQ(TeConfigInfo::Instance().InitConfigItemsFromOptions(options), true);
 
    TeCacheManager te_cache_manager;
    EXPECT_EQ(te_cache_manager.Initialize(), true);
    EXPECT_EQ(te_cache_manager.cache_mode_, CompileCacheMode::Disable);
    TeConfigInfo::Instance().config_enum_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigEnumItem::OpDebugLevel)] = 0;
}

TEST(TeCacheManagerSTest, pre_compile_cache_01)
{
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = RealPath(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/atc_data/kernel_cache");
    if (TeCacheManager::Instance().cache_dir_path_.empty()) {
        TeCacheManager::Instance().cache_dir_path_ = RealPath("../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache");
    }
    if (TeCacheManager::Instance().cache_dir_path_.empty()) {
        TeCacheManager::Instance().cache_dir_path_ = RealPath("../../../../../../../../../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache");
    }
    if (TeCacheManager::Instance().cache_dir_path_.empty()) {
        TeCacheManager::Instance().cache_dir_path_ = RealPath("../../../../../../../../../../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache");
    }
    std::cout << "cache dir:" << TeCacheManager::Instance().cache_dir_path_ << std::endl;
    TeCacheManager::Instance().precompile_ret_cache_map_.clear();
    TeCacheManager::Instance().compile_ret_cache_map_.clear();

    PreCompileResultPtr pre_ret1 = TeCacheManager::Instance().MatchPreCompileCache("te_Mul_cd58ce363a48125986d4fd8ef5c8df0e4f0d554e9fc856072d28d6799222c408_pre");
    ASSERT_NE(pre_ret1, nullptr);
    EXPECT_EQ(pre_ret1->coreType, "VectorCore");
    EXPECT_EQ(pre_ret1->opPattern, "Broadcast");
    EXPECT_EQ(pre_ret1->prebuiltOptions, "sth_just_like_this");

    std::string cache_kernel_name = "te_Mul_zzzzce363a48125986d4fd8ef5c8df0e4f0d554e9fc856072d28d6799222c408_pre";
    std::string invalid_file = TeCacheManager::Instance().cache_dir_path_ + "/" + cache_kernel_name + ".json";
    if (fs::exists(invalid_file)) {
        system(("rm -rf " + invalid_file).c_str());
    }
    PreCompileResultPtr pre_ret2 = TeCacheManager::Instance().MatchPreCompileCache(cache_kernel_name);
    EXPECT_EQ(pre_ret2, nullptr);

    PreCompileResultPtr pre_ret3 = std::make_shared<PreCompileResult>("Elemwise");
    pre_ret3->coreType = "SomeCore";
    pre_ret3->prebuiltOptions = "SomeOptions";
    EXPECT_EQ(TeCacheManager::Instance().SetPreCompileResult(cache_kernel_name, pre_ret3), true);
    pre_ret2 = TeCacheManager::Instance().MatchPreCompileCache(cache_kernel_name);
    ASSERT_NE(pre_ret2, nullptr);
    EXPECT_EQ(pre_ret2->coreType, "SomeCore");
    EXPECT_EQ(pre_ret2->opPattern, "Elemwise");
    EXPECT_EQ(pre_ret2->prebuiltOptions, "SomeOptions");

    TeCacheManager::Instance().precompile_ret_cache_map_.clear();
    PreCompileResultPtr pre_ret4 = TeCacheManager::Instance().MatchPreCompileCache(cache_kernel_name);
    ASSERT_NE(pre_ret4, nullptr);
    EXPECT_EQ(pre_ret4->coreType, "SomeCore");
    EXPECT_EQ(pre_ret4->opPattern, "Elemwise");
    EXPECT_EQ(pre_ret4->prebuiltOptions, "SomeOptions");
}

bool GetBinFileSha256Value_Stub1(te::fusion::PythonApiCall *This,
                                 const char *binData, const size_t binSize, std::string &sha256Val)
{
    sha256Val = "916d252d364e662b9c9a8ff06d3736fbc5842ae9fa481ebbc91f27d5a190f7d1";
    return true;
}
TEST(TeCacheManagerSTest, compile_cache_01)
{
    MOCKER_CPP(&te::fusion::PythonApiCall::GetBinFileSha256Value,
               bool(te::fusion::PythonApiCall::*)(const char *binData, const size_t, std::string &res) const)
        .stubs()
        .will(invoke(GetBinFileSha256Value_Stub1));
    std::string cache_kernel_name1 = "te_Mul_d77a6f72c3a58aecb67c9c35cb75aee7371d2c724a75539f9500b8e97f7dab20";
    CompileResultPtr com_ret1 = TeCacheManager::Instance().MatchCompileCache(cache_kernel_name1, false);
    ASSERT_NE(com_ret1, nullptr);
    EXPECT_NE(com_ret1->jsonPath.find(cache_kernel_name1), std::string::npos);
    EXPECT_NE(com_ret1->binPath.find(cache_kernel_name1), std::string::npos);
    EXPECT_EQ(com_ret1->headerPath.empty(), true);
    EXPECT_NE(com_ret1->jsonInfo, nullptr);
    EXPECT_NE(com_ret1->kernelBin, nullptr);
    GlobalMockObject::reset();
}

bool GetBinFileSha256Value_Stub2(te::fusion::PythonApiCall *This,
                                 const char *binData, const size_t binSize, std::string &sha256Val)
{
    sha256Val = "b97559990204d88759446ca413be0189aa572e941ff87908a658d7044ead8854";
    return true;
}
TEST(TeCacheManagerSTest, compile_cache_02)
{
    MOCKER_CPP(&te::fusion::PythonApiCall::GetBinFileSha256Value,
               bool(te::fusion::PythonApiCall::*)(const char *binData, const size_t, std::string &res) const)
        .stubs()
        .will(invoke(GetBinFileSha256Value_Stub2));
    std::string cache_kernel_name2 = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    CompileResultPtr com_ret2 = TeCacheManager::Instance().MatchCompileCache(cache_kernel_name2, false);
    EXPECT_EQ(com_ret2, nullptr);

    std::string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    CompileResultPtr com_ret4 = CompileResultUtils::ParseCompileResult(json_file_path);
    // EXPECT_EQ(TeCacheManager::Instance().SetCompileResult(com_ret4), true);

    com_ret2 = TeCacheManager::Instance().MatchCompileCache(cache_kernel_name2, false);
    // ASSERT_NE(com_ret2, nullptr);
    // EXPECT_NE(com_ret2->jsonPath.find(cache_kernel_name2), std::string::npos);
    // EXPECT_NE(com_ret2->binPath.find(cache_kernel_name2), std::string::npos);
    // EXPECT_EQ(com_ret2->headerPath.empty(), true);
    // EXPECT_NE(com_ret2->jsonInfo, nullptr);
    // EXPECT_NE(com_ret2->kernelBin, nullptr);

    TeCacheManager::Instance().compile_ret_cache_map_.clear();
    CompileResultPtr com_ret3 = TeCacheManager::Instance().MatchCompileCache(cache_kernel_name2, false);
    // ASSERT_NE(com_ret3, nullptr);
    // EXPECT_NE(com_ret3->jsonPath.find(cache_kernel_name2), std::string::npos);
    // EXPECT_NE(com_ret3->binPath.find(cache_kernel_name2), std::string::npos);
    // EXPECT_EQ(com_ret3->headerPath.empty(), true);
    // EXPECT_NE(com_ret3->jsonInfo, nullptr);
    // EXPECT_NE(com_ret3->kernelBin, nullptr);
    GlobalMockObject::reset();
}

TEST(TeCacheManagerSTest, compile_cache_autofuse_02)
{
    system("g++ -fPIC -DENABLE_T -shared -o ../../disk_cache/kernel_meta/libte_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.so "
           "../../disk_cache/kernel_meta/example.cc");
    CompileResultPtr com_ret4 = CompileResultUtils::ParseCompileResult(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/kernel_meta/te_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.json");
    system("rm ../../disk_cache/kernel_meta/libte_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.so");
    EXPECT_EQ(TeCacheManager::Instance().SetCompileResult(com_ret4), false);
}

TEST(TeCacheManagerSTest, compile_cache_autofuse_fail_02)
{
    system("g++ -fPIC -shared -o ../../disk_cache/kernel_meta/libte_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.so "
           "../../disk_cache/kernel_meta/example.cc");
    CompileResultPtr com_ret4 = CompileResultUtils::ParseCompileResult(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/kernel_meta/te_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.json");
    system("rm ../../disk_cache/kernel_meta/libte_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.so");
    EXPECT_EQ(TeCacheManager::Instance().SetCompileResult(com_ret4), false);
}

TEST(TeCacheManagerSTest, compile_cache_autofuse_empty_file_02)
{
    system("touch ../../disk_cache/kernel_meta/libte_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.so");
    CompileResultPtr com_ret4 = CompileResultUtils::ParseCompileResult(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/kernel_meta/te_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.json");
    system("rm ../../disk_cache/kernel_meta/libte_ascbackend_9865e0fbe56139e85cc42efcf59ac9ab77ee34c3f14e3272f0ea1e2bbedf8abe.so");
    EXPECT_EQ(TeCacheManager::Instance().SetCompileResult(com_ret4), false);
}

bool GetBinFileSha256Value_Stub3(te::fusion::PythonApiCall *This,
                                 const char *binData, const size_t binSize, std::string &sha256Val)
{
    sha256Val = "a411ce71496907a94d5e2df4ab3b85a307d0a5efa4b934b83cf19d1e26a75343";
    return true;
}
TEST(TeCacheManagerSTest, compile_cache_03)
{
    MOCKER_CPP(&te::fusion::PythonApiCall::GetBinFileSha256Value,
               bool(te::fusion::PythonApiCall::*)(const char *binData, const size_t, std::string &res) const)
        .stubs()
        .will(invoke(GetBinFileSha256Value_Stub3));
    TeCacheManager::Instance().compile_ret_cache_map_.clear();
    std::string cache_kernel_name1 = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    CompileResultPtr com_ret1 = TeCacheManager::Instance().MatchCompileCache(cache_kernel_name1, false);
    EXPECT_EQ(com_ret1 , nullptr);
    GlobalMockObject::reset();
}

TEST(TeCacheManagerSTest, compile_cache_04)
{
    const char *binData = "test_bin_data";
    size_t binSize =  sizeof(binData);
    std::string sha256Val;
    bool hash_ret = PythonApiCall::Instance().GetBinFileSha256Value(binData, binSize, sha256Val);
    EXPECT_EQ(hash_ret, true);
}
}
}
