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
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "tensor_engine/fusion_api.h"
#include "te_llt_utils.h"

#define private public
#define protected public
#include "common/te_config_info.h"
#undef protected public
#undef private public

#include "ge_common/ge_api_types.h"

using namespace std;
using namespace testing;
namespace te {
namespace fusion {
class FusionApiUTest : public testing::Test
{
protected:
    static void SetUpTestCase() {
        std::cout << "FusionApiUTest SetUpTestCase" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "FusionApiUTest TearDownTestCase" << std::endl;
    }
};

TEST(FusionApiUTest, init_and_finalize)
{
    EXPECT_EQ(TbeFinalize(), true);
    InitPyHandleStub();
    TeConfigInfo::Instance().env_item_vec_[static_cast<size_t>(TeConfigInfo::EnvItem::TeParallelCompiler)] = "7";
    std::map<std::string, std::string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    bool isSupportParallel = false;
    bool* isSupportParallelCompilation = &isSupportParallel;
    EXPECT_EQ(TbeInitialize(options, isSupportParallelCompilation), true);
    EXPECT_EQ(isSupportParallel, true);
}
}
}