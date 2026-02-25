/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#define private public
#include "execute/memory_statistic_manager.h"
#undef private

#include "common/scope_guard.h"

namespace FlowFunc {
class MemoryStatisticManagerSTest : public testing::Test {
 protected:
  virtual void SetUp() {
    ResetStatistic();
  }

  virtual void TearDown() {
    ResetStatistic();
    GlobalMockObject::verify();
  }

  void ResetStatistic() {
    auto &instance = MemoryStatisticManager::Instance();
    instance.Finalize();
    instance.rss_avg_ = 0;
    instance.rss_hwm_ = 0;
    instance.rss_statistic_count_ = 0;

    instance.xsmem_avg_ = 0;
    instance.xsmem_hwm_ = 0;
    instance.xsmem_statistic_count_ = 0;
  }
};

TEST_F(MemoryStatisticManagerSTest, StatisticRss) {
  MemoryStatisticManager::Instance().process_status_file_name_ =
      MemoryStatisticManager::Instance().GetProcessStatusFile();
  MemoryStatisticManager::Instance().StatisticRss();
  EXPECT_EQ(MemoryStatisticManager::Instance().rss_statistic_count_, 1);
  EXPECT_GE(MemoryStatisticManager::Instance().rss_hwm_,
            static_cast<int64_t>(MemoryStatisticManager::Instance().rss_avg_));
  MemoryStatisticManager::Instance().StatisticRss();
  EXPECT_EQ(MemoryStatisticManager::Instance().rss_statistic_count_, 2);
  EXPECT_GE(MemoryStatisticManager::Instance().rss_hwm_,
            static_cast<int64_t>(MemoryStatisticManager::Instance().rss_avg_));
}

TEST_F(MemoryStatisticManagerSTest, StatisticRss_file_not_exist) {
  MemoryStatisticManager::Instance().process_status_file_name_ = "/test/xxxx/not_exits_statistic_file_name";
  MemoryStatisticManager::Instance().StatisticRss();
  EXPECT_EQ(MemoryStatisticManager::Instance().rss_statistic_count_, 0);
}

TEST_F(MemoryStatisticManagerSTest, StatisticRss_avoid_division_by_zero) {
  MemoryStatisticManager::Instance().process_status_file_name_ =
      MemoryStatisticManager::Instance().GetProcessStatusFile();
  MemoryStatisticManager::Instance().rss_statistic_count_ = UINT32_MAX;
  MemoryStatisticManager::Instance().StatisticRss();
  EXPECT_EQ(MemoryStatisticManager::Instance().rss_statistic_count_, 1);
  EXPECT_GE(MemoryStatisticManager::Instance().rss_hwm_,
            static_cast<int64_t>(MemoryStatisticManager::Instance().rss_avg_));
}

TEST_F(MemoryStatisticManagerSTest, testReadXsmem) {
  std::string mock_summary_file = std::string("./test_summary_") + std::to_string(getpid());
  {
    std::ofstream out_file(mock_summary_file);
    out_file << "task pid 555010 pool cnt 1\n"
             << "pool id(name): alloc_size real_size\n"
             << "2(DM_QS_GROUP_555010_18446604357349956096) 73276925 73276928 732769280\n"
             << "summary: 73276925 73276928";
  }
  ScopeGuard gd([&mock_summary_file]() { remove(mock_summary_file.c_str()); });
  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file;
  uint64_t xsmem = 0;
  uint64_t xsmem_peak_size = 0;
  EXPECT_TRUE(MemoryStatisticManager::Instance().ReadXsmemValue(xsmem, xsmem_peak_size));
  EXPECT_EQ(xsmem, 73276928);
  EXPECT_EQ(xsmem_peak_size, 732769280);
}

TEST_F(MemoryStatisticManagerSTest, testReadXsmem_not_exist) {
  std::string mock_summary_file = "./test_summary_not_exist";
  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file;
  uint64_t xsmem = 0;
  uint64_t xsmem_peak_size = 0;
  EXPECT_FALSE(MemoryStatisticManager::Instance().ReadXsmemValue(xsmem, xsmem_peak_size));
}

TEST_F(MemoryStatisticManagerSTest, testReadXsmem_format_error_no_summary) {
  std::string mock_summary_file = std::string("./test_summary_") + std::to_string(getpid());
  {
    std::ofstream out_file(mock_summary_file);
    out_file << "task pid 555010 pool cnt 1\n"
             << "pool id(name): alloc_size real_size\n"
             << "2(DM_QS_GROUP_11111_18446604357349956096) 73276925 73276928\n";
  }
  ScopeGuard gd([&mock_summary_file]() { remove(mock_summary_file.c_str()); });
  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file;
  uint64_t xsmem = 0;
  uint64_t xsmem_peak_size = 0;
  EXPECT_FALSE(MemoryStatisticManager::Instance().ReadXsmemValue(xsmem, xsmem_peak_size));
}

TEST_F(MemoryStatisticManagerSTest, testReadXsmem_format_error_summary) {
  std::string mock_summary_file = std::string("./test_summary_") + std::to_string(getpid());
  {
    std::ofstream out_file(mock_summary_file);
    out_file << "task pid 555010 pool cnt 1\n"
             << "pool id(name): alloc_size real_size\n"
             << "2(DM_QS_GROUP_555010_18446604357349956096) 73276925 \n"
             << "summary: 73276925 ";
  }
  ScopeGuard gd([&mock_summary_file]() { remove(mock_summary_file.c_str()); });
  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file;
  uint64_t xsmem = 0;
  uint64_t xsmem_peak_size = 0;
  EXPECT_FALSE(MemoryStatisticManager::Instance().ReadXsmemValue(xsmem, xsmem_peak_size));
}

TEST_F(MemoryStatisticManagerSTest, testReadXsmem_ignore_zero) {
  std::string mock_summary_file = std::string("./test_summary_") + std::to_string(getpid());
  {
    std::ofstream out_file(mock_summary_file);
    out_file << "task pid 555010 pool cnt 1\n"
             << "pool id(name): alloc_size real_size\n"
             << "2(DM_QS_GROUP_555010_18446604357349956096) 73276925 0\n"
             << "summary: 0 0";
  }
  ScopeGuard gd([&mock_summary_file]() { remove(mock_summary_file.c_str()); });
  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file;
  uint64_t xsmem = 0;
  uint64_t xsmem_peak_size = 0;
  EXPECT_FALSE(MemoryStatisticManager::Instance().ReadXsmemValue(xsmem, xsmem_peak_size));
}

TEST_F(MemoryStatisticManagerSTest, StatisticXsmem) {
  std::string mock_summary_file0 = std::string("./test_summary0_") + std::to_string(getpid());
  std::string mock_summary_file1 = std::string("./test_summary1_") + std::to_string(getpid());
  std::string mock_summary_file2 = std::string("./test_summary2_") + std::to_string(getpid());
  std::string mock_summary_file3 = std::string("./test_summary3_") + std::to_string(getpid());
  {
    std::ofstream out_file0(mock_summary_file0);
    out_file0 << "task pid 555010 pool cnt 1\n"
              << "pool id(name): alloc_size real_size\n"
              << "2(DM_QS_GROUP_555010_18446604357349956096) 0 0\n"
              << "summary: 0 0";
    std::ofstream out_file1(mock_summary_file1);
    out_file1 << "task pid 555010 pool cnt 1\n"
              << "pool id(name): alloc_size real_size\n"
              << "2(DM_QS_GROUP_555010_18446604357349956096) 100 150\n"
              << "summary: 100 150";
    std::ofstream out_file2(mock_summary_file2);
    out_file2 << "task pid 555010 pool cnt 1\n"
              << "pool id(name): alloc_size real_size\n"
              << "2(DM_QS_GROUP_555010_18446604357349956096) 200 300\n"
              << "summary: 200 300";
    std::ofstream out_file3(mock_summary_file3);
    out_file3 << "task pid 555010 pool cnt 1\n"
              << "pool id(name): alloc_size real_size\n"
              << "2(DM_QS_GROUP_555010_18446604357349956096) 100 150\n"
              << "summary: 100 150";
  }
  ScopeGuard gd([&mock_summary_file0, &mock_summary_file1, &mock_summary_file2, &mock_summary_file3]() {
    remove(mock_summary_file0.c_str());
    remove(mock_summary_file1.c_str());
    remove(mock_summary_file2.c_str());
    remove(mock_summary_file3.c_str());
  });

  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file0;
  MemoryStatisticManager::Instance().StatisticXsmem();
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file1;
  MemoryStatisticManager::Instance().StatisticXsmem();
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file2;
  MemoryStatisticManager::Instance().StatisticXsmem();
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file3;
  MemoryStatisticManager::Instance().StatisticXsmem();
  // value 0 is ignore
  EXPECT_EQ(MemoryStatisticManager::Instance().xsmem_statistic_count_, 3);
  EXPECT_EQ(MemoryStatisticManager::Instance().xsmem_hwm_, 300);
  EXPECT_EQ(static_cast<int64_t>(MemoryStatisticManager::Instance().xsmem_avg_), (150 + 300 + 150) / 3);
}

TEST_F(MemoryStatisticManagerSTest, StatisticXsmem_avoid_division_by_zero) {
  std::string mock_summary_file = std::string("./test_summary0_") + std::to_string(getpid());
  {
    std::ofstream out_file(mock_summary_file);
    out_file << "task pid 555010 pool cnt 1\n"
             << "pool id(name): alloc_size real_size\n"
             << "2(DM_QS_GROUP_555010_18446604357349956096) 100 200\n"
             << "summary: 100 200";
  }
  ScopeGuard gd([&mock_summary_file]() { remove(mock_summary_file.c_str()); });
  MemoryStatisticManager::Instance().xsmem_statistic_count_ = UINT32_MAX;
  MemoryStatisticManager::Instance().xsmem_group_name_ = "DM_QS_GROUP_555010";
  MemoryStatisticManager::Instance().xsmem_summary_file_name_ = mock_summary_file;
  MemoryStatisticManager::Instance().StatisticXsmem();
  EXPECT_EQ(MemoryStatisticManager::Instance().xsmem_statistic_count_, 1);
  EXPECT_EQ(MemoryStatisticManager::Instance().xsmem_hwm_, 200);
  EXPECT_EQ(static_cast<int64_t>(MemoryStatisticManager::Instance().xsmem_avg_), 200);
}
}  // namespace FlowFunc