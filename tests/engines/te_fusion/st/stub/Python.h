/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef STUB_PYTHON_H
#define STUB_PYTHON_H

#include <string>
#include "Python_stub.h"

PyGILState_STATE PyGILState_Ensure(void);
void PyGILState_Release(PyGILState_STATE);

void Py_Initialize();

int Py_IsInitialized();

bool PyRun_SimpleString(const std::string &s);

PyObject *PyImport_ImportModule(const char* s);

PyObject *Py_BuildValue(...);

PyObject *PyObject_Str(PyObject *v);

PyObject *Py_VaBuildValue(const char *format, ...);

PyObject *PyTuple_Pack(uint32_t size, ...);

PyObject *PyUnicode_FromString(const char *v);

PyObject *PyFloat_FromDouble(double v);

PyObject *PyLong_FromLong(int64_t);

int PyTuple_SetItem(PyObject*, int, PyObject*);

PyObject *PyTuple_New(int);

PyObject *PyList_New(int);

int PyList_SetItem(PyObject*, int, PyObject*);

PyObject *PyDict_New();

int PyDict_SetItemString(PyObject*, const char*, PyObject*);

int PyDict_SetItem(PyObject* PyTotal, PyObject* key, PyObject* val);

PyObject *PyObject_GetAttrString(PyObject *pModule, const char *s);

PyObject *PyObject_CallObject(PyObject *pFunc, PyObject *pArg);

bool PyErr_Print();

bool Py_Finalize();

PyObject *PyEval_CallObject(PyObject *pFunc, PyObject *pArg);

int PyObject_IsTrue(PyObject *pRes);

int PyArg_Parse(PyObject *args, const char *format, char **res);
int PyArg_Parse(PyObject *args, const char *format, int *res);
int PyArg_Parse(PyObject *args, const char *format, void *res);
int PyArg_Parse(PyObject *args, const char *format, PyObject **res);

PyObject* PyObject_Repr(PyObject *o);

Py_ssize_t PyList_Size(PyObject *list);

Py_ssize_t PyTuple_Size(PyObject *list);

PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index);

int PyArg_ParseTuple(PyObject *args, const char *format, ...);

void Py_XDECREF(PyObject *o);

void Py_XINCREF(PyObject *o);

void PyErr_Clear();

int PyObject_HasAttrString(PyObject *o, const char *attr_name);

PyObject* PyObject_CallFunction(PyObject *callable, const char *format, ...);

PyObject* PyObject_CallMethod(PyObject *obj, const char *name, const char *format, ...);

PyObject* PyTuple_GetItem(PyObject *p, Py_ssize_t pos);

int PyEval_ThreadsInitialized();

void PyEval_InitThreads();

PyThreadState* PyEval_SaveThread();

void PyEval_RestoreThread(PyThreadState *tstate);

void PyErr_Fetch(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback);

void PyErr_NormalizeException(PyObject**exc, PyObject**val, PyObject**tb);

void PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback);

void Py_DECREF(PyObject *o);

int PyGILState_Check();

int PyList_Check(PyObject *p);

int PyDict_Check(PyObject *p);

int PyTuple_Check(PyObject *p);

int PyUnicode_Check(PyObject *p);

PyObject* PyDict_GetItemString(PyObject *p, const char *key);

PyObject* PyDict_GetItem(PyObject *p, PyObject *key);

PyObject* PyDict_Keys(PyObject *p);

PyObject* PyObject_Call(PyObject *pyFunc, PyObject *tmpArgs, PyObject *kwargs);

const char* PyUnicode_AsUTF8(PyObject *p);

int TE_PyTuple_SetItem(PyObject* PyTotal, long int idx, PyObject* PySub);

PyObject *TE_PyTuple_New(long int idx);

PyObject* TE_PyRun_SimpleString(const char* s);

extern PyObject *Py_None;

#endif
