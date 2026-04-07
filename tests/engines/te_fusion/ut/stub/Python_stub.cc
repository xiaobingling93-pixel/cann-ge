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
#include "Python_stub.h"
#include <string>
#include <iostream>
#include <cstring>
#include <stdarg.h>
#include <nlohmann/json.hpp>
#include "common/util/error_manager/error_manager.h"
#include "ge_attr_value.h"
#include "anchor.h"
#include "op_desc.h"
#include "common_stub.h"
#include "compile/fusion_manager.h"
#include "common/common_utils.h"

namespace {
std::vector<PyObjectTestPtr> gPyObjectPtrVec;

std::string GetCodeDir() {
  static std::string gCachedCodeDir;
  if (gCachedCodeDir.empty()) {
    const char *code_path_ptr = std::getenv("AIR_CODE_DIR");
    if (code_path_ptr != nullptr) {
      gCachedCodeDir = string(code_path_ptr);
    }
  }
  return gCachedCodeDir;
}
}
static PyObject truePyObj;
static PyObject falsePyObj;
static PyObject nonePyObj;
static PyObject tmpPyObj;

nlohmann::json jsonObj;
using namespace te::fusion;
std::string nullJson = jsonObj.dump();

char succCharStub[] = "success";
char *succStub = succCharStub;
char nullResChar[] = "";
char *nullRes = nullResChar;

char hashCharStub[] = "c775e7b757ede630cd0aa1113bd102661ab38829ca52a6422ab782862f268646";
char *hashStub = hashCharStub;

char hashfileStub[] = "b97559990204d88759446ca413be0189aa572e941ff87908a658d7044ead8854";
char *hashfile = hashfileStub;

char g_JsonPath_stub[] = "json_file_path";

int g_PyThreadsInitialized_stub = 0;
int g_PyInitialized_stub = 0;
int g_taskId_stub = 1;

PyThreadState g_PyThrdState_stub;

void Py_Initialize_Stub() {
    g_PyInitialized_stub = 1;
}

int Py_IsInitialized_Stub() {
    return g_PyInitialized_stub;
}

PyGILState_STATE PyGILState_Ensure_Stub(void) {
    return PyGILState_LOCKED;
}

void PyGILState_Release_Stub(PyGILState_STATE) {
    return;
}

bool PyRun_SimpleString_Stub(const std::string &s) {
    return true;
}

PyObject* TE_PyRun_SimpleString_Stub(const char* s) {
    return nullptr;
}

PyObject *PyImport_ImportModule_Stub(const char* s) {
    if (std::strstr(s, ".dynamic.fill") != NULL) {
        truePyObj.func = "get_op_support_info";
        truePyObj.res = "dynamic.fill";
    }
    if (std::strstr(s, "impl.fill") != NULL) {
        truePyObj.res = "static.fill";
    }
    return &truePyObj;
}

PyObject *PyImport_ImportModule_Stub_Return_None(const char* s) {
    return nullptr;
}

PyObject *Py_BuildValue_Stub(const char*, ...) {
    return &truePyObj;
}

PyObject *PyObject_Str_Stub(PyObject *v) {
    return v;
}

PyObject *Py_VaBuildValue_Stub(const char *format, ...) {
    return &truePyObj;
}

PyObject *PyTuple_Pack_Stub(uint32_t size, ...) {
    return static_cast<PyObject *>(nullptr);
}

PyObject *PyUnicode_FromString_Stub(const char *v) {
    return &truePyObj;
}

char* PyString_AsString_Stub(PyObject * v)
{
    return "";
}

PyObject *PyFloat_FromDouble_Stub(double) {
    return &truePyObj;
}

int PyTuple_SetItem_Stub(PyObject* PyTotal, Py_ssize_t idx, PyObject* PySub)
{
    return 0;
}

int TE_PyTuple_SetItem_Stub(PyObject* PyTotal, long int idx, PyObject* PySub)
{
    return 1;
}

int PyList_SetItem_Stub(PyObject* PyTotal, Py_ssize_t idx, PyObject* PySub)
{
    return 0;
}

int PyDict_SetItemString_Stub(PyObject* PyTotal, const char* name, PyObject* PySub)
{
    return 0;
}

PyObject *PyTuple_New_Stub(Py_ssize_t idx)
{
    return &truePyObj;
}

PyObject *TE_PyTuple_New_Stub(long int idx)
{
    return &truePyObj;
}

PyObject *PyDict_New_Stub()
{
    return &truePyObj;
}

PyObject *PyObject_GetAttrString_Stub(PyObject *pModule, const char *s) {
    return &truePyObj;
}

PyObject *PyObject_CallObject_Stub(PyObject *pFunc, PyObject *pArg) {
    return &truePyObj;
}

void PyErr_Print_Stub() {
    return;
}

void Py_Finalize_Stub() {
    return;
}

PyObject *PyEval_CallObject_Stub(PyObject *pFunc, PyObject *pArg) {
    return &truePyObj;
}

int PyObject_IsTrue_Stub(PyObject *pRes) {
    return 1;
}

int PyObject_IsFalse_Stub(PyObject *pRes) {
    return 0;
}

void PyObject_InitTruePyObj_Stub() {
    std::cout << "Into PyObject_InitTruePyObj_Stub /r/n" << std::endl;
    truePyObj.func = "";
    truePyObj.res = "";
}

int PyArg_Parse_Stub(PyObject *args, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::string testPath = te::fusion::RealPath(currentFilePath);
    std::cout << "PyArg_Parse args->func:" << args->func << ",args->res:" << args->res  << ",testPath:" << testPath << std::endl;
    if (args->func == "is_generalize_func_register_from_c" ||
        args->func == "cann_kb_init" || args->func == "cann_kb_finalize") {
        int *parsedResult = va_arg(ap, int*);
        if (args->res == "0") {
            *parsedResult = 0;
        } else {
            *parsedResult = 1;
        }
    }

    if (args->func == "init_multi_process_env") {
        int *count = va_arg(ap, int*);
        *count = 8;
    }

    if (args->func == "generalize_shape_and_range_from_c") {
        char **generalizeResult = va_arg(ap, char **);
        if (args->res == "0") {
            *generalizeResult = "null";
        } else if (args->res == "Add") {
            GetAddGeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "Mul") {
            GetMulGeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "Fill") {
            GetMulGeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "Div") {
            GetDivGeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "Conv2D") {
            GetMul1GeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "Mul1") {
            GetMul1GeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "Minimum") {
            GetMul1GeneralizeFuncReturn(generalizeResult);
        } else if (args->res == "AscendQuant") {
            GetAscendQuantGeneralizeFuncReturn(generalizeResult);
        } else {
            *generalizeResult = succStub;
        }
    }

    if (args->func == "get_op_args_json") {
        char **generalizeResult = va_arg(ap, char **);
        *generalizeResult = succStub;
    }

    if (args->func == "get_file_sha256_hash_from_c" ||
        args->func == "get_str_sha256_hash_from_c") {
        char **generalizeResult = va_arg(ap, char **);
        if (args->res == "1234567890" || args->res == "\"1234567890\"") {
            *generalizeResult = hashStub;
        } else if (args->res.find("te_matmul_cache") != string::npos) {
            *generalizeResult = hashfile;
        } else if (args->res == testPath + "/st/disk_cache/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.o") {
            *generalizeResult = hashfile;
        } else if (args->res == testPath + "/ut/disk_cache/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.o") {
            *generalizeResult = hashfile;
        }else {
            *generalizeResult = succStub;
        }
    }

    if (args->func == "get_binfile_sha256_hash_from_c") {
        char **generalizeResult = va_arg(ap, char **);
        *generalizeResult = hashfileStub;
    }

    if (args->func == "check_supported") {
        char **result = va_arg(ap, char **);
        if (args->res == "Fill") {
            *result = "True";
        } else if (args->res == "Add") {
            *result = "False";
        } else if (args->res == "isTuple") {
            *result = "Unknown";
        }
    }

    if (args->func == "op_select_format") {
        char **result = va_arg(ap, char **);
        if (args->res.find("add") != string::npos) {
            *result = "FLOAT : NCHW";
        }
    }

    if (args->func == "get_op_specific_info") {
        char **result = va_arg(ap, char **);
        if (args->res.find("add") != string::npos) {
            *result = "SpecificInfo : Unsupported";
        }
        if (args->res.find("lamb_next_mv_with_decay") != string::npos) {
            GetOpSpecificInfoReturn(result);
        }
    }

    if (args->func == "get_op_support_info") {
        char **result = va_arg(ap, char **);
        if (args->res.find("add") != string::npos) {
            *result = "support lxfusion";
        }
        if (args->res.find("fill") != string::npos) {
            *result = "not support lxfusion";
        }
    }

    if (args->func == "get_dict_string") {
        string formatStr(format);
        std::cout << "get_dict_string, format:" << formatStr << std::endl;
        if (formatStr == "i") {
            int *intValue = va_arg(ap, int*);
            *intValue = 0;
        }
        if (formatStr == "k") {
            uint64_t *uint64Value = va_arg(ap, uint64_t*);
            *uint64Value = 8;
        }
        if (formatStr == "s") {
            char **result = va_arg(ap, char **);
            *result = "{\"pattern\":\"Opaque\",\"core_type\":\"AiCore\"}";
        }
        if (formatStr == "O") {
            PyObject **obj = va_arg(ap, PyObject **);
            PyObjectTestPtr pyObj = std::make_shared<PyObject>();
            gPyObjectPtrVec.push_back(pyObj);
            *obj = pyObj.get();
        }
    }

    if (args->func == "") {
        char **result = va_arg(ap, char **);
        *result = "op_unique_key";
    }
    va_end(ap);
    return 1;
}

PyObject* PyObject_Repr_Stub(PyObject *o)
{
    return &truePyObj;
}

Py_ssize_t PyList_Size_Stub(PyObject *list)
{
    return 0;
}

Py_ssize_t PyList_Size_1_Stub(PyObject *list)
{
    return 1;
}

Py_ssize_t PyTuple_Size_Stub(PyObject *list)
{
    return 0;
}

Py_ssize_t PyTuple_Size_2_Stub(PyObject *list)
{
    return 2;
}

PyObject* PyList_GetItem_Stub(PyObject *list, Py_ssize_t index)
{
    std::cout << "PyList_GetItem_Stub" << std::endl;
    tmpPyObj.func = "";
    tmpPyObj.res = "";
    return &tmpPyObj;
}

int PyArg_ParseTuple_Stub(PyObject *args, const char *format, ...)
{
    if (std::strcmp(format, "kkiss") == 0) {
        va_list ap;
        va_start(ap, format);
        uint64_t *graphId = va_arg(ap, uint64_t*);
        uint64_t *taskId = va_arg(ap, uint64_t*);
        int *status = va_arg(ap, int*);
        char **errmsg = va_arg(ap, char**);
        char **json = va_arg(ap, char**);

        *graphId = 1;
        *taskId = g_taskId_stub++;
        *status = 0;
        *errmsg = succStub;
        *json = g_JsonPath_stub;

        va_end(ap);
        return 1;
    }

    if (std::strcmp(format, "iOO") == 0) {
        va_list ap;
        va_start(ap, format);
        int *cnt = va_arg(ap, int*);
        *cnt = 1;
        va_end(ap);
        return 1;
    }

    if (args->func == "check_supported") {
        va_list ap;
        va_start(ap, format);
        PyObject **parsedResult = va_arg(ap, PyObject **);
        char **parsedReason = va_arg(ap, char **);
        if (args->res.find("fill") != string::npos) {
            tmpPyObj.func = args->func;
            tmpPyObj.res = "Fill";
            *parsedResult = &tmpPyObj;
        } else if (args->res.find("add") != string::npos) {
            tmpPyObj.func = args->func;
            tmpPyObj.res = "Add";
            *parsedResult = &tmpPyObj;
            *parsedReason = "Null";
        } else if (args->res == "parse_failed") {
            return 0;
        } else if (args->res == "isTuple") {
            tmpPyObj.func = args->func;
            tmpPyObj.res = args->res;
            *parsedResult = &tmpPyObj;
            *parsedReason = "Null";
        }
        va_end(ap);
        return 1;
    }

    return 1;
}

int PyArg_ParseTuple_Stub_Return_Zero(PyObject *args, const char *format, ...)
{
    return 0;
}

void Py_XDECREF_Stub(PyObject *o)
{
    return;
}

void Py_XINCREF_Stub(PyObject *o)
{
    return;
}

void PyErr_Clear_Stub()
{
    return;
}

int PyObject_HasAttrString_Stub(PyObject *args, const char *attr_name)
{
    if (args->func == "get_op_support_info") {
        if (args->res == "dynamic.fill") {
            return 0;
        }
    }
    return 1;
}

PyObject* PyObject_CallFunction_Stub(PyObject *callable, const char *format, ...)
{
    return &truePyObj;
}

PyObject* PyObject_CallMethod_Stub(PyObject *obj, const char *name, const char *format, ...)
{
    std::cout << "common stub CallMethod_SizeT: " << name << std::endl;

    if (std::string("build_single_const_op") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("get_single_compile_op_result") == name) {
        tmpPyObj.func = name;
        tmpPyObj.msg = name;
        return &tmpPyObj;
    }

    if (std::string("init_multi_process_env") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = 8;
        return &tmpPyObj;
    }

    if (std::string("is_generalize_func_register_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "Fill") {
            tmpPyObj.res = "0";
        } else if (opType == "Crop") {
            return nullptr;
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("generalize_shape_and_range_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "LambNextMVWithDecay") {
            tmpPyObj.res = "0";
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("get_op_register_pattern") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("get_op_args_json") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("get_file_sha256_hash_from_c") == name ||
        std::string("get_str_sha256_hash_from_c") == name ||
        std::string("get_binfile_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        va_list ap;
        va_start(ap, format);
        tmpPyObj.res = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("call_op_func") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        tmpPyObj.res = va_arg(ap, char *);
        tmpPyObj.func = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("multi_process_check") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("generate_op_unique_key") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("check_op_impl_mode") == name) {
        va_list ap;
        va_start(ap, format);
        std::string opModule = va_arg(ap, char *);
        va_end(ap);
        std::cout << "===check_op_impl_mode==" << opModule << std::endl;
        tmpPyObj.func = name;
        if (opModule == "mul.mul") {
            return nullptr;
        } else {
            tmpPyObj.res = "1";
        }
        return &tmpPyObj;
    }

    if (std::string("cann_kb_init") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = "0";
        return &tmpPyObj;
    }

    if (std::string("cann_kb_finalize") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = "0";
        return &tmpPyObj;
    }

    return &truePyObj;
}

PyObject* PyObject_CallMethod_Stub_Return_None(PyObject *obj, const char *name, const char *format, ...)
{
    return nullptr;
}

PyObject* PyObject_CallMethod_Stub_ForBinary(PyObject *obj, const char *name, const char *format, ...)
{
    std::cout << "CallMethod: " << name << std::endl;

    if (std::string("dispatch_single_tune_task") == name) {
        return nullptr;
    }

    if (std::string("init_multi_process_env") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = 8;
        return &tmpPyObj;
    }

    if (std::string("is_generalize_func_register_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        tmpPyObj.res = "0";

        return &tmpPyObj;
    }

    if (std::string("generalize_shape_and_range_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "LambNextMVWithDecay") {
            tmpPyObj.res = "0";
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("get_file_sha256_hash_from_c") == name ||
        std::string("get_op_args_json") == name ||
        std::string("get_binfile_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("get_str_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        va_list ap;
        va_start(ap, format);
        tmpPyObj.res = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("call_op_func") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        tmpPyObj.res = va_arg(ap, char *);
        tmpPyObj.func = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("multi_process_check") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("cann_kb_init") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = "0";
        return &tmpPyObj;
    }

    if (std::string("cann_kb_finalize") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = "0";
        return &tmpPyObj;
    }
    return &truePyObj;
}

PyObject* PyObject_CallMethod_Stub_ForRL(PyObject *obj, const char *name, const char *format, ...)
{
    std::cout << "CallMethod: " << name << std::endl;

    if (std::string("dispatch_single_tune_task") == name) {
        return nullptr;
    }

    if (std::string("init_multi_process_env") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = 8;
        return &tmpPyObj;
    }

    if (std::string("is_generalize_func_register_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "Fill") {
            tmpPyObj.res = "0";
        } else if (opType == "Crop") {
            return nullptr;
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("generalize_shape_and_range_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "LambNextMVWithDecay") {
            tmpPyObj.res = "0";
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("get_file_sha256_hash_from_c") == name ||
        std::string("get_op_args_json") == name ||
        std::string("get_binfile_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("get_str_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        va_list ap;
        va_start(ap, format);
        tmpPyObj.res = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("call_op_func") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        tmpPyObj.res = va_arg(ap, char *);
        tmpPyObj.func = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("multi_process_check") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("cann_kb_init") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = "0";
        return &tmpPyObj;
    }

    if (std::string("cann_kb_finalize") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = "0";
        return &tmpPyObj;
    }
    return &truePyObj;
}

PyObject* PyObject_CallMethod_Stub_ForRL_Fail(PyObject *obj, const char *name, const char *format, ...)
{
    std::cout << "RL CallMethod: " << name << std::endl;
    if (std::string("init_multi_process_env") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = 8;
        return &tmpPyObj;
    }

    if (std::string("rl_tune_init") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = false;
        return &tmpPyObj;
    }

    return &truePyObj;
}

PyObject* PyObject_CallMethod_PreBuild_Fail_Stub(PyObject *obj, const char *name, const char *format, ...)
{
    std::cout << "CallMethod: " << name << std::endl;

    if (std::string("dispatch_prebuild_task") == name) {
        return nullptr;
    }

    if (std::string("build_single_op_from_c") == name) {
        return nullptr;
    }

    if (std::string("dispatch_single_op_compile_task") == name) {
        return nullptr;
    }

    if (std::string("init_multi_process_env") == name) {
        tmpPyObj.func = name;
        tmpPyObj.res = 8;
        return &tmpPyObj;
    }

    if (std::string("is_generalize_func_register_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "Fill") {
            tmpPyObj.res = "0";
        } else if (opType == "Crop") {
            return nullptr;
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("generalize_shape_and_range_from_c") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        std::string opType = va_arg(ap, char *);
        va_end(ap);
        tmpPyObj.func = name;
        if (opType == "LambNextMVWithDecay") {
            tmpPyObj.res = "0";
        } else {
            tmpPyObj.res = opType;
        }
        return &tmpPyObj;
    }

    if (std::string("get_file_sha256_hash_from_c") == name ||
        std::string("get_op_args_json") == name ||
        std::string("get_binfile_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        return &tmpPyObj;
    }

    if (std::string("get_str_sha256_hash_from_c") == name) {
        tmpPyObj.func = name;
        va_list ap;
        va_start(ap, format);
        tmpPyObj.res = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("call_op_func") == name) {
        va_list ap;
        va_start(ap, format);
        PyObject* input = va_arg(ap, PyObject *);
        PyObject* output = va_arg(ap, PyObject *);
        PyObject* attr = va_arg(ap, PyObject *);
        tmpPyObj.res = va_arg(ap, char *);
        tmpPyObj.func = va_arg(ap, char *);
        va_end(ap);
        return &tmpPyObj;
    }

    if (std::string("multi_process_check") == name) {
        return nullptr;
    }
    return &truePyObj;
}

PyObject* PyTuple_GetItem_Stub(PyObject *p, Py_ssize_t pos)
{
    return &truePyObj;
}

int PyEval_ThreadsInitialized_Stub()
{
    return g_PyThreadsInitialized_stub;
}

void PyEval_InitThreads_Stub()
{
    g_PyThreadsInitialized_stub = 1;
    return;
}


PyThreadState* PyEval_SaveThread_Stub()
{
    return &g_PyThrdState_stub;
}

void PyEval_RestoreThread_Stub(PyThreadState *tstate)
{
    return;
}


void PyErr_Fetch_Stub(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback)
{
    return;
}

void PyErr_NormalizeException_Stub(PyObject**exc, PyObject**val, PyObject**tb)
{
    return;
}

void PyErr_Restore_Stub(PyObject *type, PyObject *value, PyObject *traceback)
{
    return;
}

void Py_DECREF_Stub(PyObject *o)
{
    return;
}

int PyGILState_Check_Stub()
{
    return 1;
}

int PyList_Check_Stub(PyObject *p)
{
    return 1;
}

int PyDict_Check_Stub(PyObject *p)
{
    return 1;
}

int PyTuple_Check_Stub(PyObject *args)
{
    if (args->func == "check_supported") {
        if (args->res == "isNotTuple") {
            return 0;
        }
    }
    return 1;
}

int PyUnicode_Check_Stub(PyObject *p)
{
    return 1;
}

PyObject* PyDict_GetItemString_Stub(PyObject *p, const char *key)
{
    PyObjectTestPtr objPtr = std::make_shared<PyObject>();
    objPtr->func = "get_dict_string";
    gPyObjectPtrVec.push_back(objPtr);
    return objPtr.get();
}

PyObject* PyDict_GetItem_Stub(PyObject *p, PyObject *key)
{
    return &truePyObj;
}

PyObject* PyDict_Keys_Stub(PyObject *p)
{
    return &truePyObj;
}

PyObject* PyObject_Call_Stub(PyObject *pyFunc, PyObject *tmpArgs, PyObject *kwargs)
{
    return &truePyObj;
}

const char* PyUnicode_AsUTF8_Stub(PyObject *p)
{
    return p->msg.c_str();
}

void Py_Dealloc_Stub(PyObject *p)
{
    return;
}

PyObject* Get_py_true_stub()
{
    truePyObj.res = "1";
    return &truePyObj;
}

PyObject* Get_py_false_stub()
{
    truePyObj.res = "0";
    return &truePyObj;
}

PyObject* PyLong_FromLong_stub(long idx)
{
    return &truePyObj;
}

PyObject* PyList_New_stub(Py_ssize_t idx)
{
    return &truePyObj;
}

PyObject* PyList_New_stub_null(Py_ssize_t idx)
{
    return nullptr;
}

PyObject* PyTuple_GetItem_stub(PyObject* obj, Py_ssize_t idx)
{
    return &truePyObj;
}

PyObject *Py_None_Stub = &nonePyObj;

///
/// @brief Obtain ErrorManager instance
/// @return ErrorManager instance
///
ErrorManager &ErrorManager::GetInstance() {
  static ErrorManager instance;
  return instance;
}

const std::string &ErrorManager::GetLogHeader() {
    return "log";
}

///
/// @brief init
/// @param [in] path current so path
/// @return int 0(success) -1(fail)
///
int ErrorManager::Init(std::string path) { return 0; }

///
/// @brief report error message
/// @param [in] errCode  error code
/// @param [in] mapArgs  parameter map
/// @return int 0(success) -1(fail)
///
int ErrorManager::ReportErrMessage(std::string error_code, const std::map<std::string, std::string> &args_map) {
  return 0;
}

///
/// @brief report error message
/// @param [in] errCode  error code
/// @param [in] key
/// @param [in] value
/// @return int 0(success) -1(fail)
///
void ErrorManager::ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key,
                                      const std::vector<std::string> &value) {
}

///
/// @brief output error message
/// @param [in] handle  print handle
/// @return int 0(success) -1(fail)
///
int ErrorManager::OutputErrMessage(int handle) { return 0; }

///
/// @brief parse json file
/// @param [in] path json path
/// @return int 0(success) -1(fail)
///
int ErrorManager::ParseJsonFile(std::string path) { return 0; }

///
/// @brief read json file
/// @param [in] file_path json path
/// @param [in] handle  print handle
/// @return int 0(success) -1(fail)
///
int ErrorManager::ReadJsonFile(const std::string &file_path, void *handle) { return 0; }

// using namespace ge;

// const std::uint8_t *TensorData::GetData() const
// {
//     return nullptr;
// }

// GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const ProtoMsgOwner& protoMsgOwner, ge::proto::OpDef* opDef) {
//     int64_t id = 0;
//     (void)AttrUtils::GetInt(this, "ATTR_NAME_ID", id);
//     opDef->set_id(id);

//     int64_t streamId = 0;
//     (void)AttrUtils::GetInt(this, "ATTR_NAME_STREAM_ID", streamId);
//     opDef->set_stream_id(streamId);
// }

// OutDataAnchor::Vistor<InDataAnchorPtr> OutDataAnchor::GetPeerInDataAnchors() const
// {
//     vector<InDataAnchorPtr> ret;
//     for (auto anchor: peer_anchors_) {
//         auto inDataAnchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
//         if (inDataAnchor != nullptr) {
//             ret.push_back(inDataAnchor);
//         }
//     }
//     return OutDataAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
// }

// bool AttrUtils::HasAttr(ConstAttrHolderAdapter&& obj, const string& name)
// {
//     if(!obj) {
//         return false;
//     }
//     return obj->HasAttr(name);
// }

