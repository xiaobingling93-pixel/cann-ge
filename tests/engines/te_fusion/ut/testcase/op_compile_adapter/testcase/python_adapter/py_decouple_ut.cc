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
#include "python_adapter/py_decouple.h"
#undef protected public
#undef private public

#include "te_llt_utils.h"

using namespace testing;
namespace te {
namespace fusion {
class PyDecoupleUT : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "PyDecoupleUT SetUpTestSuite" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "PyDecoupleUT TearDownTestSuite" << std::endl;
    }

protected:
    void TearDown() override {
        std::cout << "PyDecoupleUT TearDown" << std::endl;
        InitPyHandleStub();
    }
};

TEST(PyDecoupleUT, init_and_finalize_case1) {
    HandleManager handleManager;
    EXPECT_EQ(handleManager.Initialize(), true);
    EXPECT_EQ(handleManager.IsPyEnvInitBeforeTbe(), false);
    EXPECT_NE(handleManager.get_py_false(), nullptr);
    EXPECT_NE(handleManager.get_py_none(), nullptr);
    EXPECT_NE(handleManager.get_py_true(), nullptr);

    EXPECT_NE(handleManager.TE_PyObject_GetAttrString, nullptr);
    EXPECT_NE(handleManager.TE_PyObject_HasAttrString, nullptr);
    EXPECT_NE(handleManager.TE_PyObject_IsTrue, nullptr);
    EXPECT_NE(handleManager.TE_PyTuple_GetItem, nullptr);
    EXPECT_NE(handleManager.TE_PyTuple_New, nullptr);
    EXPECT_NE(handleManager.TE_PyTuple_SetItem, nullptr);
    EXPECT_NE(handleManager.TE_PyTuple_Size, nullptr);
    EXPECT_NE(handleManager.TE_PyUnicode_AsUTF8, nullptr);
    EXPECT_NE(handleManager.TE_PyUnicode_FromString, nullptr);
    EXPECT_NE(handleManager.TE_PyGILState_Release, nullptr);
    EXPECT_NE(handleManager.TE_PyDict_SetItemString, nullptr);
    EXPECT_NE(handleManager.TE_PyRun_SimpleString, nullptr);
    EXPECT_NE(handleManager.TE_PyObject_CallObject, nullptr);
    EXPECT_NE(handleManager.TE_PyDict_New, nullptr);
    EXPECT_NE(handleManager.TE_PyObject_Str, nullptr);
    EXPECT_NE(handleManager._PyArg_Parse, nullptr);
    EXPECT_NE(handleManager._PyArg_ParseTuple, nullptr);
    EXPECT_NE(handleManager._Py_BuildValue, nullptr);
    EXPECT_NE(handleManager._PyObject_CallFunction, nullptr);
    EXPECT_NE(handleManager.TE_PyObject_CallMethod_SizeT, nullptr);
    EXPECT_NE(handleManager.TE_PyDict_GetItem, nullptr);
    EXPECT_NE(handleManager.TE_PyDict_GetItemString, nullptr);
    EXPECT_NE(handleManager.TE_PyDict_Keys, nullptr);
    EXPECT_NE(handleManager.TE_PyErr_Fetch, nullptr);
    EXPECT_NE(handleManager.TE_PyErr_NormalizeException, nullptr);
    EXPECT_NE(handleManager.TE_PyErr_Print, nullptr);
    EXPECT_NE(handleManager.TE_PyEval_InitThreads, nullptr);
    EXPECT_NE(handleManager.TE_PyEval_RestoreThread, nullptr);
    EXPECT_NE(handleManager.TE_PyEval_SaveThread, nullptr);
    EXPECT_NE(handleManager.TE_PyEval_ThreadsInitialized, nullptr);
    EXPECT_NE(handleManager.TE_Py_Finalize, nullptr);
    EXPECT_NE(handleManager.TE_PyFloat_FromDouble, nullptr);
    EXPECT_NE(handleManager.TE_PyGILState_Check, nullptr);
    EXPECT_NE(handleManager.TE_PyGILState_Ensure, nullptr);
    EXPECT_NE(handleManager.TE_PyImport_ImportModule, nullptr);
    EXPECT_NE(handleManager.TE_Py_Initialize, nullptr);
    EXPECT_NE(handleManager.TE_PyList_GetItem, nullptr);
    EXPECT_NE(handleManager.TE_Py_IsInitialized, nullptr);
    EXPECT_NE(handleManager.TE_PyList_New, nullptr);
    EXPECT_NE(handleManager.TE_PyList_SetItem, nullptr);
    EXPECT_NE(handleManager.TE_PyList_Size, nullptr);
    EXPECT_NE(handleManager.TE_PyLong_FromLong, nullptr);
    EXPECT_NE(handleManager.TE_PyObject_Call, nullptr);
    EXPECT_NE(handleManager.TE_Py_Dealloc, nullptr);

    EXPECT_EQ(handleManager.Finalize(), true);
}

TEST(PyDecoupleUT, init_and_finalize_case2) {
    HandleManager handleManager1;
    EXPECT_EQ(handleManager1.Initialize(), true);
    EXPECT_EQ(handleManager1.IsPyEnvInitBeforeTbe(), false);
    HandleManager handleManager2;
    EXPECT_EQ(handleManager2.Initialize(), true);
    EXPECT_EQ(handleManager2.IsPyEnvInitBeforeTbe(), true);
    EXPECT_EQ(handleManager1.Finalize(), true);
    EXPECT_EQ(handleManager2.Finalize(), true);
}
}
}