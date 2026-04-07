/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "Python.h"
#include <string>
#include <iostream>
#include <cstring>
#include <stdarg.h>
#include "common/util/error_manager/error_manager.h"
#include "ge_attr_value.h"
#include "anchor.h"
#include "op_desc.h"

namespace {
PyObject truePyObj;

PyObject tmpPyObj;

char succChar[] = "success";
char *succ = succChar;
char g_JsonPath[] = "json_file_path";

int g_PyThreadsInitialized = 0;
int g_PyInitialized = 0;
int g_taskId = 1;

PyThreadState g_PyThrdState;
}

void Py_Initialize() {
    g_PyInitialized = 1;
}

int Py_IsInitialized() {
    return g_PyInitialized;
}

PyGILState_STATE PyGILState_Ensure(void) {
    return PyGILState_LOCKED;
}

void PyGILState_Release(PyGILState_STATE) {
    return;
}

bool PyRun_SimpleString(const std::string &s) {
    return true;
}

PyObject* TE_PyRun_SimpleString(const char* s) {
    return nullptr;
}

PyObject *PyImport_ImportModule(const char* s) {
    return &truePyObj;
}

PyObject *Py_BuildValue(...) {
    return &truePyObj;
}

PyObject *PyObject_Str(PyObject *v) {
    return &truePyObj;
}

PyObject *Py_VaBuildValue(const char *format, ...) {
    return &truePyObj;
}

PyObject *PyTuple_Pack(uint32_t size, ...) {
    return static_cast<PyObject *>(nullptr);
}

PyObject *PyUnicode_FromString(const char *v) {
    return static_cast<PyObject *>(nullptr);
}

PyObject *PyLong_FromLong(int64_t) {
    return static_cast<PyObject *>(nullptr);
}

PyObject *PyFloat_FromDouble(double) {
    return static_cast<PyObject *>(nullptr);
}

int PyTuple_SetItem(PyObject* PyTotal, int idx, PyObject* PySub)
{
    return 0;
}

int TE_PyTuple_SetItem(PyObject* PyTotal, long int idx, PyObject* PySub)
{
    return 0;
}

int PyList_SetItem(PyObject* PyTotal, int idx, PyObject* PySub)
{
    return 0;
}

int PyDict_SetItem(PyObject* PyTotal, PyObject* key, PyObject* val)
{
    return 0;
}

int PyDict_SetItemString(PyObject* PyTotal, const char* name, PyObject* PySub)
{
    return 0;
}

PyObject *PyTuple_New(int idx)
{
    return &truePyObj;
}

PyObject *TE_PyTuple_New(long int idx)
{
    return &truePyObj;
}

PyObject *PyList_New(int idx)
{
    return &truePyObj;
}

PyObject *PyDict_New()
{
    return &truePyObj;
}

PyObject *PyObject_GetAttrString(PyObject *pModule, const char *s) {
    return &truePyObj;;
}

PyObject *PyObject_CallObject(PyObject *pFunc, PyObject *pArg) {
    return &truePyObj;
}

bool PyErr_Print() {
    return true;
}

bool Py_Finalize() {
    g_PyInitialized = 0;
    return true;
}

PyObject *PyEval_CallObject(PyObject *pFunc, PyObject *pArg) {
    return &truePyObj;
}

int PyObject_IsTrue(PyObject *pRes) {
    return 1;
}

int PyArg_Parse(PyObject *args, const char *format, void *res)
{
    return 1;
}

int PyArg_Parse(PyObject *args, const char *format, char **res)
{
    *res = succ;
    return 1;
}

int PyArg_Parse(PyObject *args, const char *format, int *res)
{
    *res = 0;
    if (args->res.size() > 0) {
        *res = 8;
        return 1;
    }
    if (std::strcmp("p", format) == 0) {
        *res = 1;
        return 1;
    }

    if (args->func == "is_generalize_func_register_from_c") {
        if (args->res == "0") {
            return 0;
        } else {
            return 1;
        }
    }
    return 1;
}

int PyArg_Parse(PyObject *args, const char *format, PyObject **res)
{
    *res = &truePyObj;
    return 1;
}

PyObject* PyObject_Repr(PyObject *o)
{
    return &truePyObj;
}

Py_ssize_t PyList_Size(PyObject *list)
{
    return 1;
}

Py_ssize_t PyTuple_Size(PyObject *list)
{
    return 1;
}

PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index)
{
    return &truePyObj;
}

int PyArg_ParseTuple(PyObject *args, const char *format, ...)
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
        *taskId = g_taskId++;
        *status = 0;
        *errmsg = succ;
        *json = g_JsonPath;

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

    return 1;
}

void Py_XDECREF(PyObject *o)
{
    return;
}

void Py_XINCREF(PyObject *o)
{
    return;
}

void PyErr_Clear()
{
    return;
}

int PyObject_HasAttrString(PyObject *o, const char *attr_name)
{
    return 1;
}

PyObject* PyObject_CallFunction(PyObject *callable, const char *format, ...)
{
    return &truePyObj;
}

PyObject* PyObject_CallMethod(PyObject *obj, const char *name, const char *format, ...)
{
    std::cout << "common stub CallMethod: " << name << std::endl;

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
            tmpPyObj.res = "1";
        }
        return &tmpPyObj;
    }

    return &truePyObj;
}

PyObject* PyTuple_GetItem(PyObject *p, Py_ssize_t pos)
{
    return &truePyObj;
}

int PyEval_ThreadsInitialized()
{
    return g_PyThreadsInitialized;
}

void PyEval_InitThreads()
{
    g_PyThreadsInitialized = 1;
    return;
}


PyThreadState* PyEval_SaveThread()
{
    return &g_PyThrdState;
}

void PyEval_RestoreThread(PyThreadState *tstate)
{
    return;
}


void PyErr_Fetch(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback)
{
    return;
}

void PyErr_NormalizeException(PyObject**exc, PyObject**val, PyObject**tb)
{
    return;
}

void PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback)
{
    return;
}

void Py_DECREF(PyObject *o)
{
    return;
}

int PyGILState_Check()
{
    return 1;
}

int PyList_Check(PyObject *p)
{
    return 1;
}

int PyDict_Check(PyObject *p)
{
    return 1;
}

int PyTuple_Check(PyObject *p)
{
    return 1;
}

int PyUnicode_Check(PyObject *p)
{
    return 1;
}

PyObject* PyDict_GetItemString(PyObject *p, const char *key)
{
    return &truePyObj;
}

PyObject* PyDict_GetItem(PyObject *p, PyObject *key)
{
    return &truePyObj;
}

PyObject* PyDict_Keys(PyObject *p)
{
    return &truePyObj;
}

PyObject* PyObject_Call(PyObject *pyFunc, PyObject *tmpArgs, PyObject *kwargs)
{
    return &truePyObj;
}

const char* PyUnicode_AsUTF8(PyObject *p)
{
    return "";
}

PyObject *Py_None = &truePyObj;

///
/// @brief Obtain ErrorManager instance
/// @return ErrorManager instance
///
// ErrorManager &ErrorManager::GetInstance() {
//   static ErrorManager instance;
//   return instance;
// }

// const std::string &ErrorManager::GetLogHeader() {
//     return "log";
// }

///
/// @brief init
/// @param [in] path current so path
/// @return int 0(success) -1(fail)
///
// int ErrorManager::Init(std::string path) { return 0; }

///
/// @brief report error message
/// @param [in] errCode  error code
/// @param [in] mapArgs  parameter map
/// @return int 0(success) -1(fail)
///
// int ErrorManager::ReportErrMessage(std::string error_code, const std::map<std::string, std::string> &args_map) {
//   return 0;
// }

///
/// @brief report error message
/// @param [in] errCode  error code
/// @param [in] key
/// @param [in] value
/// @return int 0(success) -1(fail)
///
// void ErrorManager::ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key,
 //                                      const std::vector<std::string> &value) {
// }

///
/// @brief output error message
/// @param [in] handle  print handle
/// @return int 0(success) -1(fail)
///
// int ErrorManager::OutputErrMessage(int handle) { return 0; }

///
/// @brief parse json file
/// @param [in] path json path
/// @return int 0(success) -1(fail)
///
// int ErrorManager::ParseJsonFile(std::string path) { return 0; }

///
/// @brief read json file
/// @param [in] file_path json path
/// @param [in] handle  print handle
/// @return int 0(success) -1(fail)
///
// int ErrorManager::ReadJsonFile(const std::string &file_path, void *handle) { return 0; }

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

