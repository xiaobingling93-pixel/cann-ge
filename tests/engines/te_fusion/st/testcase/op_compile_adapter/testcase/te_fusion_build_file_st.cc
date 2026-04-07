/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/ChainingMockHelper.h>
#include <gtest/gtest_pred_impl.h>

#define private public
#define protected public

#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"
#include "cache/te_cache_space_manager.h"
#include "file_handle/te_file_handle.h"
#include "common/common_utils.h"
#include "common/te_config_info.h"
#include "common/te_file_utils.h"
#include "te_fusion_base.h"
#include "cache/te_cache_manager.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace te::fusion;

const std::string PID_FILE_NAME = "buildPidInfo.json";
const std::string PID_DELETED = "pidDeleted";

class TeFusionApiSTest : public testing::Test
{
    public:
        TeFusionApiSTest(){}
    protected:
        virtual void SetUp()
        {
        }
        virtual void TearDown()
        {
            GlobalMockObject::verify();
            GlobalMockObject::reset();
        }
    protected:

};

TEST(TeFusionApiSTest, GetCurrProcessesSt) {
    uint32_t localPid = static_cast<uint32_t>(getpid());
    std::vector<uint32_t> pids = {};
    bool getLocalPid = false;
    GetCurrProcesses(pids);

    for (auto &pid : pids) {
        if (pid == localPid) {
            getLocalPid = true;
        }
    }

    EXPECT_EQ(getLocalPid, true);
}

TEST(TeFusionApiSTest, RecordPidTimeIdInfoSt) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = "/aaaa/ddd";
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);

    RecordPidTimeIdInfo();

    std::string filePath = currentFilePath + "/kernel_meta/" + PID_FILE_NAME;
    std::string path = te::fusion::RealPath(filePath);
    std::cout << "filePath: " << path << std::endl;
    EXPECT_EQ(path.empty(), false);

    std::vector<std::pair<uint32_t, std::string>> pidUniqueId;
    bool res = ReadPidTimeIdFromJson(path, pidUniqueId);

    EXPECT_EQ(res, true);
    EXPECT_EQ(pidUniqueId.empty(), false);

    uint32_t localPid = static_cast<uint32_t>(getpid());
    std::string unqId = "/aaaa/ddd";
    bool writeSucc = false;

    for (auto iter = pidUniqueId.end() - 1; iter >= pidUniqueId.begin(); --iter) {
        if (iter->first == localPid && iter->second == unqId) {
            writeSucc = true;
            break;
        }
        if (iter == pidUniqueId.begin()) {
            break;
        }
    }

    EXPECT_EQ(writeSucc, true);
    te::fusion::TeFileUtils::DeleteFile(filePath);
}

TEST(TeFusionApiSTest, PushPidUniqueIdSt) {
    // construct pidUniqueId vector
    std::vector<std::pair<uint32_t, std::string>> pidUniqueId = {
        {2, "aaaa"},
        {3, "cccc"}
    };

    // check emplace same
    std::string bbbb = "bbbb";
    PushPidTimeId(pidUniqueId, 2, bbbb);

    // check emplace ok  and increase order
    PushPidTimeId(pidUniqueId, 1, bbbb);
    PushPidTimeId(pidUniqueId, 4, bbbb);

    EXPECT_EQ(pidUniqueId.size(), 4);

    EXPECT_EQ(pidUniqueId[0].first, 1);
    EXPECT_EQ(pidUniqueId[1].first, 2);
    EXPECT_EQ(pidUniqueId[2].first, 3);
    EXPECT_EQ(pidUniqueId[3].first, 4);

    EXPECT_EQ(pidUniqueId[1].second, "aaaa");
}

TEST(TeFusionApiSTest, RemovePidTimeIdInfoSt) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }

    RemovePidTimeIdInfo();

    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath + "/kernel_meta";

    RecordPidTimeIdInfo();

    // call remove func
    RemovePidTimeIdInfo();

    // check PID_FILE_NAME file delete
    std::string filePath = currentFilePath + "/kernel_meta/" + PID_FILE_NAME;
    std::string path = te::fusion::RealPath(filePath);
    EXPECT_EQ(path.empty(), true);
}

TEST(TeFusionApiSTest, TestnotexistfileSt_01) {
    std::string currentFilePath = "/test/notexistfile";

    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath + "/kernel_meta";
    RecordPidTimeIdInfo();
}

TEST(TeFusionApiSTest, RemovePidTimeIdInfoSt_01) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }

    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath + "/kernel_meta";
    CreateDir(te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_);
    RecordPidTimeIdInfo();

    std::string filePath = currentFilePath + "/kernel_meta/" + PID_FILE_NAME;
    std::string path = te::fusion::RealPath(filePath);
    EXPECT_EQ(path.empty(), false);

    // create local pid .json and .o file
    std::string uqId =  te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_;
    std::string localPidJFilePath = currentFilePath + "/kernel_meta/uqIdxxx.json";
    std::string localPidOFilePath = currentFilePath + "/kernel_meta/uqIdxxx.o";
    CreateFile(localPidJFilePath);
    CreateFile(localPidOFilePath);

    // record exception pid
    std::vector<uint32_t> pids = {};
    GetCurrProcesses(pids);
    uint32_t excpPid = 11;
    while (true) {
        size_t i;
        for (i = 0; i < pids.size(); i++) {
            if (pids[i] == excpPid) {
                excpPid++;
                break;
            }
        }
        if (i == pids.size()) {
            break;
        }
    }
    std::string excpUqId = currentFilePath + "/kernel_exp";
    WritePidTimeIdToJson(path, excpPid, excpUqId);

    // create exception pid .json and .o file
    std::string excpPidJFilePath = currentFilePath + "/kernel_exp/uqIdxxx.json";
    std::string excpPidOFilePath = currentFilePath + "/kernel_exp/uqIdxxx.o";
    CreateFile(excpPidJFilePath);
    CreateFile(excpPidOFilePath);

    // create using pid and json file
    uint32_t localPid = static_cast<uint32_t>(getpid());
    uint32_t usingPid;
    for (size_t i = 0; i < pids.size(); i++) {
        if (pids[i] != localPid) {
            usingPid = pids[i];
            break;
        }
    }
    std::string usingUqId = currentFilePath + "/kernel_test";
    WritePidTimeIdToJson(path, usingPid, usingUqId);
    std::string usingPidJFilePath = currentFilePath + "/kernel_test/uqIdxxx.json";
    std::string usingPidOFilePath = currentFilePath + "/kernel_test/uqIdxxx.o";
    CreateFile(usingPidJFilePath);
    CreateFile(usingPidOFilePath);

    // call remove func
    RemovePidTimeIdInfo();

    // check local pid .json and .o file exist, del file manually
    std::string localPidJPath = te::fusion::RealPath(localPidJFilePath);
    std::string localPidOPath = te::fusion::RealPath(localPidOFilePath);
    EXPECT_EQ(localPidJPath.empty(), false);
    EXPECT_EQ(localPidOPath.empty(), false);
    te::fusion::TeFileUtils::DeleteFile(localPidJPath);
    te::fusion::TeFileUtils::DeleteFile(localPidOPath);

    // check exception pid .json and .o file delete
    std::string excpPidJPath = te::fusion::RealPath(excpPidJFilePath);
    std::string excpPidOPath = te::fusion::RealPath(excpPidOFilePath);
    EXPECT_EQ(excpPidJPath.empty(), true);
    EXPECT_EQ(excpPidOPath.empty(), true);

    // check using pid json file exist, del file manually
    std::string usingPidJPath = te::fusion::RealPath(usingPidJFilePath);
    std::string usingPidOPath = te::fusion::RealPath(usingPidOFilePath);
    //EXPECT_EQ(usingPidJPath.empty(), false);
    //EXPECT_EQ(usingPidOPath.empty(), false);
    te::fusion::TeFileUtils::DeleteFile(usingPidJPath);
    te::fusion::TeFileUtils::DeleteFile(usingPidOPath);

    // check PID_FILE_NAME file exist, and delete file manually
    std::string fileRealPath = te::fusion::RealPath(filePath);
    EXPECT_EQ(fileRealPath.empty(), false);
    te::fusion::TeFileUtils::DeleteFile(fileRealPath);
}

TEST(TeFusionApiSTest, MgrPidUniqueIdInfoJsonSt) {
    // construct pidUniqueId vector
    std::vector<std::pair<uint32_t, std::string>> delPidUniqueId = {
        {2, "aaaa"},
        {3, PID_DELETED}
    };

    std::vector<std::pair<uint32_t, std::string>> curPidUniqueId = {
        {1, "aaaa"},
        {2, "aaaa"},
        {3, "cccc"},
        {4, "cccc"}
    };

    // check emplace same
    std::vector<std::pair<uint32_t, std::string>> mgrPidUniqueId;
    MgrPidTimeIdInfo(delPidUniqueId, curPidUniqueId, mgrPidUniqueId);

    EXPECT_EQ(mgrPidUniqueId.size(), 3);

    EXPECT_EQ(mgrPidUniqueId[0].first, 1);
    EXPECT_EQ(mgrPidUniqueId[1].first, 2);
    EXPECT_EQ(mgrPidUniqueId[2].first, 4);
}

TEST(TeFusionApiSTest, RemoveBuildLockFileSt) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }

    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;

    std::string aFilePath = currentFilePath + "/" + "aaa.lock";
    std::string bFilePath = currentFilePath + "/" + "bbb.lock";

    CreateFile(aFilePath);
    CreateFile(bFilePath);
    std::string afileRealPath = te::fusion::RealPath(aFilePath);
    EXPECT_EQ(afileRealPath.empty(), false);
    std::string bfileRealPath = te::fusion::RealPath(bFilePath);
    EXPECT_EQ(bfileRealPath.empty(), false);

    RemoveBuildLockFile(currentFilePath);

    // check file delete
    std::string aPath = te::fusion::RealPath(aFilePath);
    EXPECT_EQ(aPath.empty(), true);
    std::string bPath = te::fusion::RealPath(bFilePath);
    EXPECT_EQ(bPath.empty(), true);
}

TEST(TeFusionApiSTest, RemoveKernelTempDir) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    CreateDir(currentFilePath + "/kernel_meta_temp_test_id");

    RemoveBuildLockFile(currentFilePath + "/kernel_meta_temp_test_id");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    CreateDir(currentFilePath + "/kernel_meta_temp_test_id");

    std::string aFilePath = currentFilePath + "/kernel_meta_temp_test_id/" + "aaa.lock";
    CreateFile(aFilePath);
    std::string afileRealPath = te::fusion::RealPath(aFilePath);
    EXPECT_EQ(afileRealPath.empty(), false);

    RemoveKernelTempDir(currentFilePath + "/kernel_meta_temp_test_id/");
    TeFileUtils::DeleteFile(afileRealPath);

    RemoveKernelTempDir(currentFilePath + "/kernel_meta_temp_test_id/");
    RemoveKernelMetaDir(currentFilePath + "/kernel_meta_temp_test_id");

    // check file delete
    std::string aPath = te::fusion::RealPath(aFilePath);
    EXPECT_EQ(aPath.empty(), true);
}

TEST(TeFusionApiSTest, RemoveKernelTempDir1) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    te::fusion::TeConfigInfo::Instance().kernelMetaUniqueHash_ = "test_id";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    CreateDir(currentFilePath + "/kernel_meta/kernel_meta_temp_test_id");

    std::string aFilePath = currentFilePath + "/kernel_meta/kernel_meta_temp_test_id/" + "aaa.lock";
    CreateFile(aFilePath);
    std::string afileRealPath = te::fusion::RealPath(aFilePath);

    RemoveKernelTempDir(currentFilePath + "/kernel_meta/kernel_meta_temp_test_id/");
    TeFileUtils::DeleteFile(afileRealPath);

    RemoveKernelTempDir(currentFilePath + "/kernel_meta/kernel_meta_temp_test_id/");
    RemoveKernelMetaDir(currentFilePath + "/kernel_meta/kernel_meta_temp_test_id");

    // check file delete
    std::string aPath = te::fusion::RealPath(aFilePath);
}

TEST(TeFusionApiSTest, RemoveKernelMetaDir) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta_pid_timid");
    CreateDir(currentFilePath + "/kernel_meta_pid_timid");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath);
    CreateDir(currentFilePath + "/kernel_meta_pid_timid/kernel_meta");
    std::string bFilePath = currentFilePath + "/kernel_meta_pid_timid/kernel_meta/" + "aaa.o";
    CreateFile(bFilePath);
    std::string bfileRealPath = te::fusion::RealPath(bFilePath);
    EXPECT_EQ(bfileRealPath.empty(), false);

    RemoveKernelMetaDir(currentFilePath + "/kernel_meta_pid_timid");
    TeFileUtils::DeleteFile(bfileRealPath);

    RemoveKernelMetaDir(currentFilePath + "/kernel_meta_pid_timid");

    // check file delete
    std::string bPath = te::fusion::RealPath(bFilePath);
    EXPECT_EQ(bPath.empty(), true);
}

TEST(TeFusionApiSTest, RemoveParentKernelMetaDir) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    RemoveParentKernelMetaDir(te::fusion::TeConfigInfo::Instance().GetDebugDir() + KERNEL_META);
    CreateDir(currentFilePath + "/kernel_meta/kernel_meta");
    std::string filepath = currentFilePath + "/kernel_meta/kernel_meta/kernel_meta123.json";
    CreateFile(filepath);
    RemoveParentKernelMetaDir(te::fusion::TeConfigInfo::Instance().GetDebugDir() + KERNEL_META);
    std::string parentPath = currentFilePath + "/kernel_meta/kernel_meta";
    std::string pPath = te::fusion::RealPath(parentPath);
    EXPECT_EQ(pPath.empty(), false);
    TeFileUtils::DeleteFile(filepath);
    RemoveParentKernelMetaDir(te::fusion::TeConfigInfo::Instance().GetDebugDir() + KERNEL_META);
    pPath = te::fusion::RealPath(parentPath);
    EXPECT_EQ(pPath.empty(), true);
}

TEST(TeFusionApiSTest, TestDelCacheFileByAccessTime) {
    MOCKER(te::fusion::TeFileUtils::DeleteFile).stubs();
    MOCKER_CPP(&te::fusion::TeCacheSpaceManager::GetCacheSpaceMaxSizeCfg).stubs().will(returnValue((uint64_t)1024000));
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compile_adapter/disk_cache/atc_data/kernel_cache";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st/disk_cache/atc_data/kernel_cache";
    }
    te::fusion::TeCacheSpaceManager &fusionSerial = te::fusion::TeCacheSpaceManager::Instance();
    bool res = fusionSerial.CacheSpaceInitialize(currentFilePath);
    EXPECT_EQ(res, true);
}