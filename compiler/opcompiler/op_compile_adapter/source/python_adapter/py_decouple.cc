/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "python_adapter/py_decouple.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <map>
#include <cstring>
#include <functional>
#include <Python.h>

#include "inc/te_fusion_check.h"
#include "inc/te_fusion_log.h"
#include "common/te_config_info.h"
#include "common/common_utils.h"
#include "mmpa/mmpa_api.h"

namespace te {
namespace fusion {

static const std::string PY = "Python";
static const std::string NotFound = "not found ";
static const uint32_t CMD_MAX_SIZE = 1024;

HandleManager::HandleManager() {}
HandleManager::~HandleManager() {}

HandleManager& HandleManager::Instance()
{
    static HandleManager handle_manager;
    return handle_manager;
}

bool HandleManager::Initialize()
{
    if (isInit_) {
        return true;
    }
    std::lock_guard<std::mutex> lock_guard(pyhandle_mutex_);
    TE_DBGLOG("Begin to initialize decouple handle manager.");
    void *py_init_func = dlsym(RTLD_DEFAULT, "Py_Initialize");
    if (py_init_func == nullptr) {
        TE_DBGLOG("Begin to load dynamic library.");
        if (!LaunchDynamicLib()) {
            TE_ERRLOG("Launch dynamic-handle failed.");
            return false;
        }
        if (!LoadFuncs(true)) {
            TE_ERRLOG("Failed to load functions from dynamic library.");
            return false;
        }
    } else {
        TE_DBGLOG("Begin to load static library.");
        if (!LoadFuncs(false)) {
            TE_ERRLOG("Failed to load functions from static library.");
            return false;
        }
    }

    // initialize python env
    if (TE_Py_IsInitialized() == 0) {
        pyEnvStatusBeforeTbe_ = false;
        TE_Py_Initialize();
    } else {
        pyEnvStatusBeforeTbe_ = true;
    }

    if (TE_PyEval_ThreadsInitialized() == 0) {
        TE_PyEval_InitThreads();
    }

    if (TE_PyGILState_Check() != 0) {
        pyThreadState_ = TE_PyEval_SaveThread();
    }
    isInit_ = true;
    TE_DBGLOG("Handle manager has been initialized.");
    return true;
}

bool HandleManager::CheckPythonVersion(FILE *fp, char* line, bool flag) const
{
    TE_DBGLOG("Begin to CheckPythonVersion, with flag is [%d].", flag);
    if (fp == nullptr) {
        return false;
    }
    while (fgets(line, CMD_MAX_SIZE, fp)) {
        std::string line_tmp(line);
        if (flag) {
            auto index = line_tmp.find(PY);
            if (index != string::npos) {
                return true;
            }
        } else {
            line_tmp.pop_back();
            if (!RealPath(line_tmp).empty()) {
                return true;
            }
        }
    }
    return false;
}

bool HandleManager::CheckCommandValid(string &command, char *line) const
{
    std::string tmp_command = command + " 2>&1";
    FILE *fptr = popen(tmp_command.data(), "r");
    if (fptr == nullptr) {
        TE_WARNLOG("Failed to execute command [%s] while attempting to load the library.", command.c_str());
        return false;
    }
    while (fgets(line, CMD_MAX_SIZE, fptr))
    {
        std::string line_tmp(line);
        if (line_tmp.find("not found") != std::string::npos) {
            TE_WARNLOG("The current environment couldn't find the command [%s], caused by [%s].", command.c_str(), line);
            pclose(fptr);
            return false;
        } else {
            line_tmp.pop_back();
            if (!RealPath(line_tmp).empty()) {
                TE_DBGLOG("Valid python lib path [%s]", line_tmp.c_str());
                pclose(fptr);
                return true;
            }
        }
    }
    pclose(fptr);
    return false;
}

bool HandleManager::LoadDynLibFromPyCfg(std::string &pythonLib, void *handle) const
{
    TE_DBGLOG("Beginning to load dynamic lib from python config");
    std::string cmd_path = "python3-config --prefix";
    char line_path[CMD_MAX_SIZE] = {0};
    std::string pythonLibUpdate;
    if (CheckCommandValid(cmd_path, line_path)) {
        if (strlen(line_path) - 1 >= CMD_MAX_SIZE) {
            return false;
        }
        line_path[strlen(line_path) - 1] = 0;
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "Get lib path[%s].", line_path);
        std::string line_path_temp = line_path;
        pythonLibUpdate = line_path_temp + "/lib/" + pythonLib;
        handle = mmDlopen(pythonLibUpdate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            TE_WARNLOG("Python path[%s] is invalid, please confirm ld_library_path of the python library.",
                       pythonLibUpdate.c_str());
            return false;
        }
        pythonLib = pythonLibUpdate;
    } else {
        TE_WARNLOG("Cannot fetch the return value from cmd[%s].", cmd_path.c_str());
        return false;
    }
    return true;
}

bool HandleManager::LaunchDynamicLib()
{
    TE_DBGLOG("Begin to Launch python dynamic lib");
    char line[CMD_MAX_SIZE] = {0};
    int x, y, z;
    const int X_STD = 3;
    const int Y_STD = 7;
    std::string cmd = "python3 -V";
    std::string cmd_backup = "python -V";
    const char* sysCammand = cmd.data();
    FILE *fp = popen(sysCammand, "r");
    bool res = CheckPythonVersion(fp, line, true);
    if (!res) {
        TE_WARNLOG("Failed to run cmd[%s], try to run cmd[%s]", cmd.c_str(), cmd_backup.c_str());
        fp = popen(cmd_backup.data(), "r");
        res = CheckPythonVersion(fp, line, true);
        if (!res) {
            std::map<std::string, std::string> pythonPathMap = {{"value", TeConfigInfo::Instance().GetEnvPath()},
                {"env", "PATH"}, {"situation", "executing the cmd python3 -V and python -V"},
                {"reason", "invalid Python version"}};
            TeErrMessageReport(EM_ENVIRONMENT_INVALID_ERROR, pythonPathMap);
            TE_ERRLOG("Failed to run cmd[%s]", cmd_backup.c_str());
            return false;
        }
    }
    std::string line_str = line;
    line_str.erase(std::remove_if(line_str.begin(), line_str.end(),
                                  std::bind(std::isspace<char>, std::placeholders::_1, std::locale::classic())),
                   line_str.end());
    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "line:[%s].", line_str.c_str());
    char buffer[CMD_MAX_SIZE] = {0};
    (void)sscanf_s(line, "%s %d.%d.%d", buffer, sizeof(buffer), &x, &y, &z);
    if (strcmp(PY.c_str(), buffer) != 0 || x != X_STD || y < Y_STD) {
        std::map<std::string, std::string> pythonVersionMap = {{"currentVersion", line},
                                                                {"supportVersion", "Python 3.7"}};
        TeErrMessageReport(EM_PYTHON_VERSION_INVALID_ERROR, pythonVersionMap);
        TE_ERRLOG("The current Python version is [%d.%d.%d], but only Python 3 is supported and version should no less than 3.7.", x, y, z);
        pclose(fp);
        return false;
    }
    std::string temp_X = std::to_string(x);
    std::string temp_Y = std::to_string(y);
    std::string pythonLib = "libpython" + temp_X + "." + temp_Y + std::string(y > Y_STD ? "" : "m") + ".so.1.0" ;
    pclose(fp);

    void *handle = nullptr;
    std::string pythonLibPath = pythonLib;
    if (!LoadDynLibFromPyCfg(pythonLibPath, handle) || handle == nullptr) {
        void *libHandle = mmDlopen(pythonLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (libHandle == nullptr) {
            std::map<std::string, std::string> pythonPathMap = {{"value", TeConfigInfo::Instance().GetEnvPath()},
                {"env", "PATH"},
                {"situation", "executing the cmd python3-config --prefix, Unable to load a valid Python SO "},
                {"reason", "invalid Python version"}};
            TeErrMessageReport(EM_ENVIRONMENT_INVALID_ERROR, pythonPathMap);
            return false;
        }
        TE_DBGLOG("Start to load dynamic python library [%s] from ld_library_path.", pythonLib.c_str());
    } else {
        TE_DBGLOG("Start to load dynamic python library from path[%s].", pythonLibPath.c_str());
    }

    pyHandle = handle;
    return true;
}

bool HandleManager::HandleClose() const
{
    if (pyHandle != nullptr) {
        if (dlclose(pyHandle) != 0) {
            REPORT_TE_INNER_ERROR("Failed to close current python handle.");
            return false;
        }
        TE_DBGLOG("This handle has not been closed.");
        return true;
    }
    TE_DBGLOG("This handle has not been loaded.");
    return true;
}

bool HandleManager::IsPyEnvInitBeforeTbe() const
{
    return pyEnvStatusBeforeTbe_;
}

#define LOAD_FUNC(funcHandle, funcType, pyFunc) \
    do { \
        funcHandle = (funcType)dlsym(handle, pyFunc); \
        if (funcHandle == nullptr) { \
            TE_ERRLOG("Error message:[%s]. Failed to dlsym function %s.", dlerror(), pyFunc); \
            return false; \
        } \
    } while (0)

bool HandleManager::LoadFuncs(bool isDynamic)
{
    void *handle;
    if (isDynamic) {
        handle = pyHandle;
    } else {
        handle = RTLD_DEFAULT;
    }

    LOAD_FUNC(TE_PyDict_SetItemString, TEPyDictSetItemString, "PyDict_SetItemString");
    LOAD_FUNC(TE_PyObject_Str, TEPyObjectStr, "PyObject_Str");
    LOAD_FUNC(TE_PyDict_New, TEPyDictNew, "PyDict_New");
    LOAD_FUNC(_PyArg_Parse, TEPyArgParse, "PyArg_Parse");
    LOAD_FUNC(_PyArg_ParseTuple, TEPyArgParseTuple, "PyArg_ParseTuple");
    LOAD_FUNC(_Py_BuildValue, TEPyBuildValue, "Py_BuildValue");
    LOAD_FUNC(_PyObject_CallFunction, TEPyObjectCallFunction, "PyObject_CallFunction");
    LOAD_FUNC(TE_PyObject_CallMethod_SizeT, TEPyObjectCallMethodSizeT, "_PyObject_CallMethod_SizeT");
    LOAD_FUNC(TE_PyDict_GetItem, TEPyDictGetItem, "PyDict_GetItem");
    LOAD_FUNC(TE_PyDict_GetItemString, TEPyDictGetItemString, "PyDict_GetItemString");
    LOAD_FUNC(TE_PyDict_Keys, TEPyDictKeys, "PyDict_Keys");
    LOAD_FUNC(TE_PyErr_Fetch, TEPyErrFetch, "PyErr_Fetch");
    LOAD_FUNC(TE_PyErr_NormalizeException, TEPyErrNormalizeException, "PyErr_NormalizeException");
    LOAD_FUNC(TE_PyErr_Print, TEPyErrPrint, "PyErr_Print");
    LOAD_FUNC(TE_PyEval_InitThreads, TEPyEvalInitThreads, "PyEval_InitThreads");
    LOAD_FUNC(TE_PyEval_RestoreThread, TEPyEvalRestoreThread, "PyEval_RestoreThread");
    LOAD_FUNC(TE_PyEval_SaveThread, TEPyEvalSaveThread, "PyEval_SaveThread");
    LOAD_FUNC(TE_PyEval_ThreadsInitialized, TEPyEvalThreadsInitialized, "PyEval_ThreadsInitialized");
    LOAD_FUNC(TE_Py_Finalize, TEPyFinalize, "Py_Finalize");
    LOAD_FUNC(TE_PyFloat_FromDouble, TEPyFloatFromDouble, "PyFloat_FromDouble");
    LOAD_FUNC(TE_PyGILState_Check, TEPyGILStateCheck, "PyGILState_Check");
    LOAD_FUNC(TE_PyGILState_Ensure, TEPyGILStateEnsure, "PyGILState_Ensure");
    LOAD_FUNC(TE_PyGILState_Release, TEPyGILStateRelease, "PyGILState_Release");
    LOAD_FUNC(TE_PyImport_ImportModule, TEPyImportImportModule, "PyImport_ImportModule");
    LOAD_FUNC(TE_Py_Initialize, TEPyInitialize, "Py_Initialize");
    LOAD_FUNC(TE_PyList_GetItem, TEPyListGetItem, "PyList_GetItem");
    LOAD_FUNC(TE_Py_IsInitialized, TEPyIsInitialized, "Py_IsInitialized");
    LOAD_FUNC(TE_PyList_New, TEPyListNew, "PyList_New");
    LOAD_FUNC(TE_PyList_SetItem, TEPyListSetItem, "PyList_SetItem");
    LOAD_FUNC(TE_PyList_Size, TEPyListSize, "PyList_Size");
    LOAD_FUNC(TE_PyLong_FromLong, TEPyLongFromLong, "PyLong_FromLong");
    LOAD_FUNC(TE_PyObject_Call, TEPyObjectCall, "PyObject_Call");
    LOAD_FUNC(TE_PyObject_GetAttrString, TEPyObjectGetAttrString, "PyObject_GetAttrString");
    LOAD_FUNC(TE_PyObject_HasAttrString, TEPyObjectHasAttrString, "PyObject_HasAttrString");
    LOAD_FUNC(TE_PyObject_IsTrue, TEPyObjectIsTrue, "PyObject_IsTrue");
    LOAD_FUNC(TE_PyTuple_GetItem, TEPyTupleGetItem, "PyTuple_GetItem");
    LOAD_FUNC(TE_PyTuple_New, TEPyTupleNew, "PyTuple_New");
    LOAD_FUNC(TE_PyTuple_SetItem, TEPyTupleSetItem, "PyTuple_SetItem");
    LOAD_FUNC(TE_PyTuple_Size, TEPyTupleSize, "PyTuple_Size");
    LOAD_FUNC(TE_PyUnicode_FromString, TEPyUnicodeFromString, "PyUnicode_FromString");
    LOAD_FUNC(TE_PyUnicode_AsUTF8, TEPyUnicodeAsUTF8, "PyUnicode_AsUTF8");
    LOAD_FUNC(TE_PyRun_SimpleString, TEPyRunSimpleString, "PyRun_SimpleString");
    LOAD_FUNC(TE_PyObject_CallObject, TEPyObjectCallObject, "PyObject_CallObject");
    LOAD_FUNC(TE_py_true, TEPyObjPtr, "_Py_TrueStruct");
    LOAD_FUNC(TE_py_false, TEPyObjPtr, "_Py_FalseStruct");
    LOAD_FUNC(TE_py_none, TEPyObjPtr, "_Py_NoneStruct");
    LOAD_FUNC(TE_Py_Dealloc, TEPyDealloc, "_Py_Dealloc");

    return true;
}

bool HandleManager::Finalize()
{
    if (!isInit_) {
        TE_INFOLOG("HandleManager has not been initialized.");
        return true;
    }
    TE_DBGLOG("Start to finalize handle manager.");
    // finalize python env
    if (!pyEnvStatusBeforeTbe_) {
        if (TE_Py_IsInitialized() != 0) {
            if (pyThreadState_) {
                TE_PyEval_RestoreThread(pyThreadState_);
            }
            TE_Py_Finalize();
        }
    }
    pyEnvStatusBeforeTbe_ = false;
    bool res = HandleClose();
    if (!res) {
        TE_ERRLOG("Failed to close handle.");
    }
    isInit_ = false;
    TE_DBGLOG("Handle manager has been finalized successfully.");
    return res;
}
}
}
