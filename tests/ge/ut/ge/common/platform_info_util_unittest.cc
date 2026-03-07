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
#include <gmock/gmock.h>
#include <vector>
#include "common/platform_info_util/platform_info_util.h"
#include "depends/runtime/src/runtime_stub.h"

#include <ge_common/ge_api_error_codes.h>

namespace ge {
namespace {
class MockRuntime : public RuntimeStub {
 public:
  rtError_t rtGetSocSpec(const char *label, const char *key, char *value, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)strncpy_s(value, maxLen, "2102", maxLen);
    return 0;
  }
};
class MockRuntimeFail : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    return 1;
  }
  rtError_t rtGetSocSpec(const char *label, const char *key, char *value, const uint32_t maxLen) override {
    return 1;
  }
};
}

class UtestPlatformInfoUtil : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestPlatformInfoUtil, GetJitCompileDefaultValue_Ok_EnableByDefault) {
  ge::RuntimeStub stub;
  auto mock_runtime = std::make_shared<MockRuntime>();
  stub.SetInstance(mock_runtime);

  auto jit_compile = ge::PlatformInfoUtil::GetJitCompileDefaultValue();
  ASSERT_STREQ(jit_compile.c_str(), "1");
  stub.Reset();
}

TEST_F(UtestPlatformInfoUtil, GetSocSpecFromPlatform) {
  ge::RuntimeStub stub;
  auto mock_runtime = std::make_shared<MockRuntimeFail>();
  stub.SetInstance(mock_runtime);
  std::string value;
  auto ret = ge::PlatformInfoUtil::GetSocSpec("", "", value);
  EXPECT_NE(ret, SUCCESS);
  ASSERT_STREQ(value.c_str(), "");
  stub.Reset();

  ret = ge::PlatformInfoUtil::GetSocSpec("version", "NpuArch", value);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_STREQ(value.c_str(), "2201");
  
}

TEST_F(UtestPlatformInfoUtil, GetJitCompileDefaultValueGetSocVersionFailed) {
  ge::RuntimeStub stub;
  auto mock_runtime = std::shared_ptr<MockRuntimeFail>(new MockRuntimeFail());
  stub.SetInstance(mock_runtime);

  auto jit_compile = ge::PlatformInfoUtil::GetJitCompileDefaultValue();
  ASSERT_STREQ(jit_compile.c_str(), "2");
  stub.Reset();
}
}
