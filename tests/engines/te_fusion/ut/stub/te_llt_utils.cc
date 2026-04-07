/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "te_llt_utils.h"

#define private public
#define protected public
#include "python_adapter/py_decouple.h"
#undef protected public
#undef private public

#include "ge_common/ge_api_types.h"
#include "tensor_engine/fusion_api.h"
#include "Python_stub.h"

namespace te {
namespace fusion {
void InitPyHandleStub() {
    HandleManager::Instance().TE_Py_IsInitialized = Py_IsInitialized_Stub;
    HandleManager::Instance().TE_Py_Initialize = Py_Initialize_Stub;
    HandleManager::Instance().TE_PyEval_ThreadsInitialized = PyEval_ThreadsInitialized_Stub;
    HandleManager::Instance().TE_PyEval_InitThreads = PyEval_InitThreads_Stub;
    HandleManager::Instance().TE_PyGILState_Check = PyGILState_Check_Stub;
    HandleManager::Instance().TE_PyEval_SaveThread = PyEval_SaveThread_Stub;
    HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
    HandleManager::Instance().TE_PyImport_ImportModule = PyImport_ImportModule_Stub;
    HandleManager::Instance()._PyArg_Parse = PyArg_Parse_Stub;
    HandleManager::Instance().TE_Py_Dealloc = Py_Dealloc_Stub;
    HandleManager::Instance().TE_PyErr_Fetch = PyErr_Fetch_Stub;
    HandleManager::Instance().TE_PyErr_NormalizeException = PyErr_NormalizeException_Stub;
    HandleManager::Instance().TE_py_true = Get_py_true_stub();
    HandleManager::Instance().TE_py_false = Get_py_false_stub();
    HandleManager::Instance().TE_py_none = Py_None_Stub;
    HandleManager::Instance().TE_PyErr_Print = PyErr_Print_Stub;
    HandleManager::Instance().TE_PyEval_RestoreThread = PyEval_RestoreThread_Stub;
    HandleManager::Instance().TE_Py_Finalize = Py_Finalize_Stub;
    HandleManager::Instance().TE_PyTuple_New = PyTuple_New_Stub;
    HandleManager::Instance().TE_PyTuple_SetItem = PyTuple_SetItem_Stub;
    HandleManager::Instance().TE_PyDict_New = PyDict_New_Stub;
    HandleManager::Instance().TE_PyDict_SetItemString = PyDict_SetItemString_Stub;
    HandleManager::Instance()._Py_BuildValue = Py_BuildValue_Stub;
    HandleManager::Instance().TE_PyFloat_FromDouble = PyFloat_FromDouble_Stub;
    HandleManager::Instance().TE_PyLong_FromLong = PyLong_FromLong_stub;
    HandleManager::Instance().TE_PyUnicode_FromString = PyUnicode_FromString_Stub;
    HandleManager::Instance().TE_PyList_New = PyList_New_stub;
    HandleManager::Instance().TE_PyObject_GetAttrString = PyObject_GetAttrString_Stub;
    HandleManager::Instance().TE_PyObject_HasAttrString = PyObject_HasAttrString_Stub;
    HandleManager::Instance().TE_PyObject_IsTrue = PyObject_IsTrue_Stub;
    HandleManager::Instance().TE_PyTuple_GetItem = PyTuple_GetItem_stub;
    HandleManager::Instance().TE_PyTuple_Size = PyTuple_Size_Stub;
    HandleManager::Instance().TE_PyUnicode_AsUTF8 = PyUnicode_AsUTF8_Stub;
    HandleManager::Instance().TE_PyGILState_Release = PyGILState_Release_Stub;
    HandleManager::Instance().TE_PyRun_SimpleString = TE_PyRun_SimpleString_Stub;
    HandleManager::Instance().TE_PyObject_CallObject = PyObject_CallObject_Stub;
    HandleManager::Instance().TE_PyObject_Str = PyObject_Str_Stub;
    HandleManager::Instance()._PyArg_ParseTuple = PyArg_ParseTuple_Stub;
    HandleManager::Instance()._PyObject_CallFunction = PyObject_CallFunction_Stub;
    HandleManager::Instance().TE_PyDict_GetItem = PyDict_GetItem_Stub;
    HandleManager::Instance().TE_PyDict_GetItemString = PyDict_GetItemString_Stub;
    HandleManager::Instance().TE_PyDict_Keys = PyDict_Keys_Stub;
    HandleManager::Instance().TE_PyGILState_Ensure = PyGILState_Ensure_Stub;
    HandleManager::Instance().TE_PyList_GetItem = PyList_GetItem_Stub;
    HandleManager::Instance().TE_PyList_SetItem = PyList_SetItem_Stub;
    HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;
    HandleManager::Instance().TE_PyObject_Call = PyObject_Call_Stub;
    HandleManager::Instance().isInit_ = true;
    std::cout << "InitPyHandleStub end" << std::endl;
}

void InitTbe() {
    InitPyHandleStub();
    std::map<std::string, std::string> te_options;
    te_options.emplace(ge::SOC_VERSION, "Ascend910B1");
    bool isSupportParallel = false;
    bool* isSupportParallelCompilation = &isSupportParallel;
    te::TbeInitialize(te_options, isSupportParallelCompilation);
}
}
}