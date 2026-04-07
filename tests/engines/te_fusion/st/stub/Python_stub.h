/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _PYTHON_STUB_H
#define _PYTHON_STUB_H

#include <string>
#include <memory>

typedef ssize_t Py_ssize_t;

class PyObject {
public:
    PyObject() {
    }

    std::string func;
    std::string res;
    int ob_refcnt = 1;
    std::string msg;
};

using PyObjectTestPtr = std::shared_ptr<PyObject>;

struct  PyThreadState {
    int dummy;
};

enum PyGILState_STATE {PyGILState_LOCKED, PyGILState_UNLOCKED};

PyGILState_STATE PyGILState_Ensure_Stub(void);
void PyGILState_Release_Stub(PyGILState_STATE);

void Py_Initialize_Stub();

int Py_IsInitialized_Stub();

bool PyRun_SimpleString_Stub(const std::string &s);

PyObject *PyImport_ImportModule_Stub(const char*);

PyObject *PyImport_ImportModule_Stub_Return_None(const char*);

PyObject *Py_BuildValue_Stub(const char*, ...);

PyObject *PyObject_Str_Stub(PyObject *v);

PyObject *Py_VaBuildValue_Stub(const char *format, ...);

PyObject *PyTuple_Pack_Stub(uint32_t size, ...);

PyObject *PyUnicode_FromString_Stub(const char *v);

char* PyString_AsString_Stub(PyObject *);

PyObject *PyFloat_FromDouble_Stub(double v);

int PyTuple_SetItem_Stub(PyObject*, Py_ssize_t, PyObject*);

PyObject *PyTuple_New_Stub(Py_ssize_t);

int PyList_SetItem_Stub(PyObject*, Py_ssize_t, PyObject*);

PyObject *PyDict_New_Stub();

int PyDict_SetItemString_Stub(PyObject*, const char*, PyObject*);

PyObject *PyObject_GetAttrString_Stub(PyObject *pModule, const char *s);

PyObject *PyObject_CallObject_Stub(PyObject *pFunc, PyObject *pArg);

void PyErr_Print_Stub();

void Py_Finalize_Stub();

PyObject *PyEval_CallObject_Stub(PyObject *pFunc, PyObject *pArg);

int PyObject_IsTrue_Stub(PyObject *pRes);

int PyObject_IsFalse_Stub(PyObject *pRes);

void PyObject_InitTruePyObj_Stub();

int PyArg_Parse_Stub(PyObject *args, const char *format, ...);

PyObject* PyObject_Repr_Stub(PyObject *o);

Py_ssize_t PyList_Size_Stub(PyObject *list);

Py_ssize_t PyList_Size_1_Stub(PyObject *list);

Py_ssize_t PyTuple_Size_Stub(PyObject *list);

Py_ssize_t PyTuple_Size_2_Stub(PyObject *list);

PyObject* PyList_GetItem_Stub(PyObject *list, Py_ssize_t index);

int PyArg_ParseTuple_Stub(PyObject *args, const char *format, ...);

int PyArg_ParseTuple_Stub_Return_Zero(PyObject *args, const char *format, ...);

void Py_XDECREF_Stub(PyObject *o);

void Py_XINCREF_Stub(PyObject *o);

void PyErr_Clear_Stub();

int PyObject_HasAttrString_Stub(PyObject *args, const char *attr_name);

PyObject* PyObject_CallFunction_Stub(PyObject *callable, const char *format, ...);

PyObject* PyObject_CallMethod_Stub(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyObject_CallMethod_Stub_Return_None(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyObject_CallMethod_Stub_ForRL(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyObject_CallMethod_Stub_ForBinary(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyObject_CallMethod_Stub_ForRL_Fail(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyObject_CallMethod_PreBuild_Fail_Stub(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyTuple_GetItem_Stub(PyObject *p, Py_ssize_t pos);

int PyEval_ThreadsInitialized_Stub();

void PyEval_InitThreads_Stub();

PyThreadState* PyEval_SaveThread_Stub();

void PyEval_RestoreThread_Stub(PyThreadState *tstate);

void PyErr_Fetch_Stub(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback);

void PyErr_NormalizeException_Stub(PyObject**exc, PyObject**val, PyObject**tb);

void PyErr_Restore_Stub(PyObject *type, PyObject *value, PyObject *traceback);

void Py_DECREF_Stub(PyObject *o);

int PyGILState_Check_Stub();

int PyList_Check_Stub(PyObject *p);

int PyDict_Check_Stub(PyObject *p);

int PyTuple_Check_Stub(PyObject *args);

int PyUnicode_Check_Stub(PyObject *p);

PyObject* PyDict_GetItemString_Stub(PyObject *p, const char *key);

PyObject* PyDict_GetItem_Stub(PyObject *p, PyObject *key);

PyObject* PyDict_Keys_Stub(PyObject *p);

PyObject* PyObject_Call_Stub(PyObject *pyFunc, PyObject *tmpArgs, PyObject *kwargs);

const char* PyUnicode_AsUTF8_Stub(PyObject *p);

int TE_PyTuple_SetItem_Stub(PyObject* PyTotal, long int idx, PyObject* PySub);

PyObject *TE_PyTuple_New_Stub(long int idx);

PyObject* TE_PyRun_SimpleString_Stub(const char* s);

void Py_Dealloc_Stub(PyObject *);

PyObject* Get_py_true_stub();

PyObject* Get_py_false_stub();

PyObject* PyLong_FromLong_stub(long);

PyObject* PyList_New_stub(Py_ssize_t);

PyObject* PyList_New_stub_null(Py_ssize_t);

PyObject* PyTuple_GetItem_stub(PyObject*, Py_ssize_t);


extern PyObject *Py_None_Stub;

#endif
