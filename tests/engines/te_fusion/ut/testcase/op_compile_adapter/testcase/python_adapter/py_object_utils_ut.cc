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
#include "python_adapter/py_wrapper.h"
#include "python_adapter/py_object_utils.h"

using namespace std;
using namespace testing;
namespace te {
namespace fusion {
class PyObjectUtilsUt : public testing::Test {
public:
    static void SetUpTestSuite() {
        std::cout << "PyObjectUtilsUt SetUpTestSuite" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "PyObjectUtilsUt TearDownTestSuite" << std::endl;
    }
};

TEST(PyObjectUtilsUt, TestGenPySocInfo)
{
    PyObject *pySocInfo = PyObjectUtils::GenPySocInfo();
    AUTO_PY_DECREF(pySocInfo);
    EXPECT_NE(pySocInfo, nullptr);
}

TEST(PyObjectUtilsUt, TestGenPyOptionDict)
{
    PyObject *pyOptionDict = PyObjectUtils::GenPyOptionDict();
    AUTO_PY_DECREF(pyOptionDict);
    EXPECT_NE(pyOptionDict, nullptr);
}

TEST(PyObjectUtilsUt, TestGenPyOptionsInfo_1)
{
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>("relu_0", "impl.relu", "Relu", "AiCore");
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {tbeOpInfo};
    PyObject *pyOptionInfo = PyObjectUtils::GenPyOptionsInfo(tbeOpInfoVec);
    AUTO_PY_DECREF(pyOptionInfo);
    EXPECT_NE(pyOptionInfo, nullptr);
}

TEST(PyObjectUtilsUt, TestGenPyOptionsInfo_2)
{
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>("relu_0", "impl.relu", "Relu", "VectorCore");
    tbeOpInfo->SetOpCoreType("AiCore");
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {tbeOpInfo};
    PyObject *pyOptionInfo = PyObjectUtils::GenPyOptionsInfo(tbeOpInfoVec);
    AUTO_PY_DECREF(pyOptionInfo);
    EXPECT_NE(pyOptionInfo, nullptr);
}
}
}