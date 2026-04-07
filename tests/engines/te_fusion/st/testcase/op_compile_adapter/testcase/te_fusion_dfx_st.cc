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
#include <iostream>
#include "dfxinfo_manager/trace_utils.h"
#include "atrace_pub.h"
#define private public
#define protected public
#include "dfxinfo_manager/dfxinfo_manager.h"
#undef protected public
#undef private public

using namespace std;
using namespace te;
using namespace te::fusion;
class TeFusionDfxInfoSTest : public testing::Test
{
    public:
        TeFusionDfxInfoSTest() {}
    protected:
        virtual void SetUp() {}
        virtual void TearDown() {}
};

TEST(TeFusionDfxInfoSTest, manager_init_stest) {
    DfxInfoManager dfxInfoManager;
    dfxInfoManager.Initialize();
    dfxInfoManager.Initialize();
    std::vector<std::string> statisticsMsg;
    dfxInfoManager.GetStatisticsMsgs(statisticsMsg);
    EXPECT_EQ(statisticsMsg.empty(), true);
    dfxInfoManager.Finalize();
    dfxInfoManager.Finalize();
}

TEST(TeFusionDfxInfoSTest, recore_stat_msg) {
    DfxInfoManager dfxInfoManager;
    dfxInfoManager.Initialize();

    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::MATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::REUSE_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::REUSE_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::COPY);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::COPY_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::COPY_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::JSON_INVALID);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::UPDATE_ACCESS_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::FILE_LOCK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::CACHE_NOT_EXIST);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::SHA256_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::SIMKEY_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::VERSION_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::TASK_SUBMIT);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::TASK_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::TASK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::BOTTOM);

    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::MATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::REUSE_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::REUSE_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::COPY);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::COPY_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::COPY_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::JSON_INVALID);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::UPDATE_ACCESS_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::FILE_LOCK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::CACHE_NOT_EXIST);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::SHA256_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::SIMKEY_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::VERSION_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::TASK_SUBMIT);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::TASK_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::TASK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::BOTTOM);

    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::MATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::REUSE_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::REUSE_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::COPY);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::COPY_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::COPY_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::JSON_INVALID);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::UPDATE_ACCESS_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::FILE_LOCK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::CACHE_NOT_EXIST);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::SHA256_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::SIMKEY_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::VERSION_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::TASK_SUBMIT);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::TASK_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::TASK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::BOTTOM);

    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::MATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::REUSE_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::REUSE_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::COPY);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::COPY_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::COPY_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::JSON_INVALID);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::UPDATE_ACCESS_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::FILE_LOCK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::CACHE_NOT_EXIST);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::SHA256_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::SIMKEY_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::VERSION_MISMATCH);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::TASK_SUBMIT);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::TASK_SUCC);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::TASK_FAIL);
    dfxInfoManager.RecordStatistics(StatisticsType::BOTTOM, RecordEventType::BOTTOM);

    std::vector<std::string> statisticsMsg;
    dfxInfoManager.GetStatisticsMsgs(statisticsMsg);
    EXPECT_EQ(statisticsMsg.size(), 15);
    for (const std::string &msg : statisticsMsg) {
        std::cout << msg << std::endl;
    }
    dfxInfoManager.Finalize();
}

TEST(TeFusionDfxInfoSTest, submit_trace_info) {
    TraceUtils::SubmitGlobalTrace("Hello world!");
    TraHandle global_handle = AtraceCreate(TracerType::TRACER_TYPE_SCHEDULE, "FE_Global_Trace");
    TraHandle compile_handle = AtraceCreate(TracerType::TRACER_TYPE_SCHEDULE, "FE_CompileTd_0");
    TraceUtils::SubmitGlobalTrace("Hello world!");
    TraceUtils::SubmitGlobalTrace("Init");
    TraceUtils::SubmitGlobalTrace("Deinit");
    TraceUtils::SubmitGlobalTrace("");
    TraceUtils::SubmitCompileDetailTrace(10, 123, "Conv2D", "do sth");
    TraceUtils::SubmitCompileDetailTrace(20, 124, "Relu", "do sth");
    TraceUtils::SubmitCompileDetailTrace(15, 125, "Compress", "do sth");
    TraceUtils::SubmitCompileDetailTrace(16, 125, "XXX", "");
    AtraceDestroy(global_handle);
    AtraceDestroy(compile_handle);
}