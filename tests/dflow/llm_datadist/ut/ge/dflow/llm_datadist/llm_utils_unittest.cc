/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include <gtest/gtest.h>

#include "nlohmann/json.hpp"
#include "common/llm_utils.h"
#include "dlog_pub.h"
#include "common/llm_inner_types.h"
#include "flow_graph/flow_graph.h"
#include "llm_test_helper.h"
#include "common/llm_file_saver.h"
#include "common/llm_ge_api.h"
#include "common/cache_manager.h"
#include "common/llm_checker.h"

using namespace std;
using namespace ::testing;

namespace ge {
class LLMUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
  }

  // 在测试类中进行清理工作，如果需要的话
  void TearDown() override {
  }

  static std::map<ge::AscendString, ge::AscendString> GetOptions(const std::string &server_type) {
    // 构建选项参数
    const uint32_t kv_size = 2;
    std::string ref_shapes;
    std::string ref_dtype;
    for (uint32_t i = 0U; i < kv_size; i++) {
      if (i < kv_size - 1U) {
        ref_shapes.append("8192,8192;");
        ref_dtype.append("1;");
      } else {
        ref_shapes.append("8192,8192");
        ref_dtype.append("1");
      }
    }
    char_t deploy_cluster_path[4096]{};
    (void)realpath("../tests/dflow/llm_datadist/ut/ge/dflow/llm_datadist/json_file/deploy_cluster_info.json", deploy_cluster_path);
    std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", "0"},
        {"ge.socVersion", "TestSocType1"},
        {"ge.graphRunMode", "0"},
        {llm::LLM_OPTION_ROLE, server_type.c_str()},
        {llm::LLM_OPTION_CLUSTER_INFO, deploy_cluster_path},
    };
    return options;
  }
};

class TestLLMFileSaver : public llm::LLMFileSaver {
public:
  static ge::Status CheckPathValid(const std::string &file_path) {
    return  llm::LLMFileSaver::CheckPathValid(file_path);
  }

  static ge::Status OpenFile(int32_t &fd, const std::string &file_path, const bool append = false) {
    return  llm::LLMFileSaver::OpenFile(fd, file_path, append);
  }

  static ge::Status WriteData(const void * const data, uint64_t size, const int32_t fd) {
    return  llm::LLMFileSaver::WriteData(data, size, fd);
  }
};

TEST_F(LLMUtilsTest, TestFileSaver) {
  int32_t fd = -1;
  const std::string err_file_path(MMPA_MAX_PATH + 1, 'x');
  const std::string buf(MMPA_MAX_PATH + 1, 'c');

  ge::Status ret;
  ret = TestLLMFileSaver::CheckPathValid(err_file_path);
  EXPECT_NE(ret, ge::SUCCESS);

  ret = llm::CreateDirectory(err_file_path);
  EXPECT_NE(ret, 0);

  ret = llm::LLMFileSaver::SaveToFile(err_file_path, buf.c_str(), MMPA_MAX_PATH + 1);
  EXPECT_NE(ret, ge::SUCCESS);

  ret = TestLLMFileSaver::WriteData(buf.c_str(), MMPA_MAX_PATH + 1, fd);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(LLMUtilsTest, TestLlmUitls) {
  // test FindContiguousBlockIndexPair
  const std::vector<uint64_t> src_blocks = {0};
  const std::vector<uint64_t> dst_blocks = {1};
  std::vector<std::vector<std::pair<int64_t, int64_t>>> result;
  ge::Status ret;
  ret = llm::LLMUtils::FindContiguousBlockIndexPair(src_blocks, dst_blocks, result);
  EXPECT_EQ(ret, ge::SUCCESS);

  // test CheckMultiplyOverflowInt64
  int64_t a = 0;
  int64_t b = 0;
  b = 2;
  a = (std::numeric_limits<int64_t>::max() / b) + 1;
  EXPECT_TRUE(llm::LLMUtils::CheckMultiplyOverflowInt64(a, b));

  a = 2;
  b = (std::numeric_limits<int64_t>::min() / a) - 1;
  EXPECT_TRUE(llm::LLMUtils::CheckMultiplyOverflowInt64(a, b));

  b = 2;
  a = (std::numeric_limits<int64_t>::min() / b) - 1;
  EXPECT_TRUE(llm::LLMUtils::CheckMultiplyOverflowInt64(a, b));

  a = -2;
  b = (std::numeric_limits<int64_t>::max() / a) - 1;
  EXPECT_TRUE(llm::LLMUtils::CheckMultiplyOverflowInt64(a, b));

  // test CeilDiv
  int32_t c = 1;
  int32_t d = 1;
  int32_t e = llm::LLMUtils::CeilDiv(c, d);
  EXPECT_TRUE((e > 0));

  // test GetDataTypeLength
  uint32_t len;
  EXPECT_TRUE(llm::LLMUtils::GetDataTypeLength(ge::DT_STRING, len));
  EXPECT_FALSE(llm::LLMUtils::GetDataTypeLength(ge::DT_MAX, len));

  // test GetSizeInBytes
  int64_t mem_size;
  ret = llm::LLMUtils::GetSizeInBytes(1, ge::DT_FLOAT6_E3M2, mem_size);
  EXPECT_EQ(ret, ge::SUCCESS);

  // test ConvertToInt64
  const std::string str = "a";
  int64_t  val;
  
  ret = llm::ConvertToInt64(str, val);
  EXPECT_NE(ret, ge::SUCCESS);
  const std::string str2 = std::to_string(std::numeric_limits<int64_t>::max()) + '1';
  ret = llm::ConvertToInt64(str2, val);
  EXPECT_NE(ret, ge::SUCCESS);

  const std::string str3 = "1";
  ret = llm::ConvertToInt64(str3, val);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(LLMUtilsTest, TestParserOptions) {
    // 构建选项参数
  const std::map<ge::AscendString, ge::AscendString> llm_options = {
      {"ge.socVersion", "TestSocType1"},
      {"ge.graphRunMode", "0"},
      {llm::LLM_OPTION_ROLE, "decoder"},
      {"ge.distributed_cluster_build", "1"},
      {"RESOURCE_CONFIG_PATH", "/tmp/numa_config_path"},
      {llm::LLM_OPTION_SYNC_KV_CACHE_WAIT_TIME, "20"},
  };

  llm::DecoderWaitTimeInfo wait_time_info;
  ge::Status ret;
  ret = llm::LLMUtils::ParserWaitTimeInfo(llm_options, wait_time_info);
  EXPECT_EQ(ret, ge::SUCCESS);

  std::vector<int32_t> device_ids;
  ret = llm::LLMUtils::ParseDeviceId(llm_options, device_ids, ge::OPTION_EXEC_DEVICE_ID);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(LLMUtilsTest, TestCacheManager) {
  llm::CacheManager cache_manager;
  llm::CacheEntry src_cache_entry;
  llm::CacheEntry dst_cache_entry;
  llm::CopyCacheParam copy_cache_param;
  src_cache_entry.num_blocks = 1;
  src_cache_entry.batch_size = 1;
  src_cache_entry.stride = 2;
  dst_cache_entry.num_blocks = 1;
  dst_cache_entry.batch_size = 1;
  dst_cache_entry.stride = 1;
  copy_cache_param.offset = 1;
  ge::Status ret;
  ret = cache_manager.CopyCacheForContinuous(src_cache_entry, dst_cache_entry, copy_cache_param, 1, 1);
  EXPECT_NE(ret, ge::SUCCESS);
}

static ge::Status TestLogTooLong() {
  std::string testlog(MSG_LENGTH * 2, 'c');
  LLM_ASSERT_TRUE(false, "TestLogTooLong:%s", testlog.c_str());
  return ge::SUCCESS;
}

TEST_F(LLMUtilsTest, TestLogTooLong) {
  ge::Status ret = TestLogTooLong();
  EXPECT_NE(ret, ge::SUCCESS);
}
}  // namespace ge
