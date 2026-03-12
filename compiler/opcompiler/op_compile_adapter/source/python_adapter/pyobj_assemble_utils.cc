/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "python_adapter/pyobj_assemble_utils.h"
#include "inc/te_fusion_check.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "python_adapter/py_decouple.h"
#include "python_adapter/py_wrapper.h"

namespace te {
namespace fusion {
using GetPyTensorByTbeOpTensor = std::function<bool(const TbeOpTensor &, PyObject *&, bool &, const TbeOpInfo &)>;
using GetComplexAttrPy = std::function<bool(const TbeAttrValue &, PyObject *&)>;
using GetSimpleAttrPy = std::function<bool(const TbeAttrValue &, PyObject *&)>;

/*
 * @brief: transform PyObject to string
 * @param [in] pyObj: PyObject value
 * @return std::string: string result for pyObj
 */
std::string PyObjectToStr(PyObject *&pyObj)
{
    std::string res{"(NULL}"};

    TE_FUSION_CHECK((pyObj == nullptr), {
        REPORT_TE_INNER_ERROR("Input parameter 'pyobj' is a nullptr.");
        return res;
    });
    PyObject *strArgs = HandleManager::Instance().TE_PyObject_Str(pyObj);
    AUTO_PY_DECREF(strArgs);

    char *pChar = nullptr;
    if (strArgs != nullptr) {
        (void)HandleManager::Instance()._PyArg_Parse(strArgs, "s", &pChar);
    }
    if (pChar != nullptr) {
        res = pChar;
    }

    return res;
}

namespace {
bool GetPyShapeRange(const std::vector<std::pair<int64_t, int64_t>> &shapeRange, PyObject *&pyRange)
{
    if (shapeRange.size() == 0) {
        pyRange = HandleManager::Instance()._Py_BuildValue("()");
        return true;
    }

    PyObject *pyObj = nullptr;
    std::vector<PyObject *> pyList;
    for (auto &range : shapeRange) {
        if (range.second > 0) {
            pyObj = HandleManager::Instance()._Py_BuildValue("(LL)", range.first, range.second);
        } else {
            pyObj = HandleManager::Instance()._Py_BuildValue("(Ls)", range.first, nullptr);
        }
        pyList.emplace_back(pyObj);
    }

    PyObject *pytmp = HandleManager::Instance().TE_PyTuple_New(pyList.size());
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pytmp == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to TE_PyTuple_New, size: %zu", pyList.size());
        return false;
    });

    int idx = 0;
    for (auto &pyIter : pyList) {
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pytmp, idx++, pyIter);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add shape range to python tuple.");
            return false;
        });
    }

    pyRange = pytmp;
    return true;
}
} // namespace

bool SetRangeToPyObj(const bool &hasSet, std::vector<std::pair<int64_t, int64_t>> &shapeRange,
                     const std::vector<int64_t> &shapes, const std::string &rangeType, PyObject *pyTensor)
{
    PyObject *pyRange = nullptr;
    bool bres = true;
    if (hasSet) {
        bres = GetPyShapeRange(shapeRange, pyRange);
    } else {
        bool needRange = false;
        for (const int64_t &shape : shapes) {
            if (shape > 0) {
                shapeRange.emplace_back(shape, shape);
            } else {
                shapeRange.emplace_back(0, -1);
                needRange = true;
            }
        }
        if (needRange) {
            bres = GetPyShapeRange(shapeRange, pyRange);
        }
    }

    TE_FUSION_CHECK((!bres), {
        REPORT_TE_INNER_ERROR("Failed to set shapeRange to Python object for %s.", rangeType.c_str());
        return false;
    });

    if (pyRange != nullptr) {
        bres = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, rangeType.c_str(), pyRange);
        TE_FUSION_CHECK((bres), {
            TE_ERRLOG("Failed to add %s.", rangeType.c_str());
            return false;
        });
    }
    TE_PY_DECREF(pyRange);
    return true;
}

/*
 * @brief: get op python shape parameter
 * @param [in] stype: op tensor shape list or tuple info
 * @param [in] shape: op tensor shape info
 * @param [out] pyShape: op python tensor shape result
 * @return bool: get python tensor shape ok or not
 */
template<typename T>
bool GetPyShape(ATTR_SHAPETYPE stype, const std::vector<T>& shape, PyObject *&pyTensor, const std::string &pyObjKey)
{
    int ires;
    PyObject *pyShape = nullptr;

    int32_t shapeSize = shape.size();
    TE_FUSION_CHECK((shapeSize == 0), {
        pyShape = HandleManager::Instance()._Py_BuildValue("()");
        TE_FUSION_CHECK_WITH_DUMP_PYERR((pyShape == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create current python shape.");
            return false;
        });
        AUTO_PY_DECREF(pyShape);
        ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, pyObjKey.c_str(), pyShape);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add shape to dictionary.");
            return false;
        });
        return true;
    });

    if (stype == ATTR_SHAPE_TUPLE) {
        pyShape = HandleManager::Instance().TE_PyTuple_New(shapeSize);
    } else {
        pyShape = HandleManager::Instance().TE_PyList_New(shapeSize);
    }
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyShape == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create current python shape.");
        return false;
    });
    AUTO_PY_DECREF(pyShape);
    int idx = 0;
    for (auto &dim : shape) {
        PyObject *pIntShape = HandleManager::Instance().TE_PyLong_FromLong(dim);
        if (stype == ATTR_SHAPE_TUPLE) {
            ires = HandleManager::Instance().TE_PyTuple_SetItem(pyShape, idx++, pIntShape);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add dim to python tuple shape.");
                return false;
            });
        } else {
            ires = HandleManager::Instance().TE_PyList_SetItem(pyShape, idx++, pIntShape);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add dim to python list shape.");
                return false;
            });
        }
    }

    ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, pyObjKey.c_str(), pyShape);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add shape to dictionary.");
        return false;
    });

    return true;
}

bool GetPyConstValue(const TbeOpTensor &paraTensor, PyObject *&pyTensor)
{
    bool res = false;
    int ires;
    PyObject *pyAttr = nullptr;
    if (paraTensor.IsConstValueNone()) {
        PyObject *pyNone = HandleManager::Instance().get_py_none();
        res = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, "const_value", pyNone);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((!res), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attr const to dictionary.");
            return false;
        });
        if (paraTensor.IsConstValueRange()) {
            res = GetPyAttr(paraTensor.GetConstValueRange(), pyAttr);
            TE_FUSION_CHECK((!res), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to convert attribute to py value.");
                return false;
            });
            AUTO_PY_DECREF(pyAttr);
            ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, "const_value_range", pyAttr);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add const valueRange to dictionary.");
                return false;
            });
        }
        return true;
    }

    res = GetPyAttr(paraTensor.GetConstValue(), pyAttr);
    TE_FUSION_CHECK((!res), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to convert attribute to py value.");
        return false;
    });
    AUTO_PY_DECREF(pyAttr);
    ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, "const_value", pyAttr);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add const value to dictionary.");
        return false;
    });
    return true;
}

bool GetPyObjOfShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    ATTR_SHAPETYPE shapeType = paraTensor.GetShapeType();
    const std::vector<int64_t> &curShape = paraTensor.GetShape();

    bool bres = GetPyShape(shapeType, curShape, pyTensor, "shape");
    TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert current shape to PyObject, shape size=[%zu].", curShape.size());
        return false;
    });

    return true;
}

bool GetPyObjOfOriShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    ATTR_SHAPETYPE shapeType = paraTensor.GetShapeType();
    const std::vector<int64_t> &oriShape = paraTensor.GetOriginShape();

    bool bres = GetPyShape(shapeType, oriShape, pyTensor, "ori_shape");
    TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert original shape to PyObject, shape size=[%zu].", oriShape.size());
        return false;
    });

    return true;
}

bool SetStringInfoToPyObj(const std::string &info, const std::string &key, PyObject *&pyTensor)
{
    PyObject *pyString = HandleManager::Instance()._Py_BuildValue("s", info.c_str());
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyString == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert parameter from current %s[%s] to PyObject.", key.c_str(), info.c_str());
        return false;
    });

    AUTO_PY_DECREF(pyString);
    int ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, key.c_str(), pyString);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add %s info to dictionary.", key.c_str());
        return false;
    });

    return true;
}

template<typename T>
bool SetIntInfoToPyObj(const T &info, const std::string &typeStr, const std::string &key, PyObject *&pyTensor)
{
    PyObject *pyInt = HandleManager::Instance()._Py_BuildValue(typeStr.c_str(), info);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyInt == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert parameter from %s to PyObject.", key.c_str());
        return false;
    });
    AUTO_PY_DECREF(pyInt);
    int ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, key.c_str(), pyInt);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add %s info to dictionary.", key.c_str());
        return false;
    });

    return true;
}

bool GetPyObjOfFormat(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    TE_FUSION_CHECK(!SetStringInfoToPyObj(paraTensor.GetFormat(), "format", pyTensor), {
        TE_ERRLOG("Failed to SetStringInfoToPyObj.");
        return false;
    });

    return true;
}

bool GetPyObjOfCurSubFormat(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    int32_t currSubFormat = paraTensor.GetSubFormat();
    TE_FUSION_CHECK(!SetIntInfoToPyObj(currSubFormat, "i", "sub_format", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });

    return true;
}

bool GetPyObjOfOriFormat(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    TE_FUSION_CHECK(!SetStringInfoToPyObj(paraTensor.GetOriginFormat(), "ori_format", pyTensor), {
        TE_ERRLOG("Failed to SetStringInfoToPyObj.");
        return false;
    });

    return true;
}

bool GetPyObjOfDtype(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    TE_FUSION_CHECK(!SetStringInfoToPyObj(paraTensor.GetType(), "dtype", pyTensor), {
        TE_ERRLOG("Failed to SetStringInfoToPyObj.");
        return false;
    });

    return true;
}

bool GetPyObjOfAddrType(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    size_t addrType = paraTensor.GetAddrType();
    DdrBaseType ddrBaseProp = paraTensor.GetDdrBaseProp();

    TE_FUSION_CHECK(!SetIntInfoToPyObj(addrType, "k", "addr_type", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });
    TE_FUSION_CHECK(!SetIntInfoToPyObj(ddrBaseProp, "l", "ddr_base_prop", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });
    return true;
}

bool GetPyObjOfCAxisValue(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    int64_t cAxisValue = paraTensor.GetCAxisValue();
    TE_FUSION_CHECK(!SetIntInfoToPyObj(cAxisValue, "l", "input_c_values", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });
    return true;
}

bool GetPyObjOfValidShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, const bool &isInput, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    ATTR_SHAPETYPE shapeType = paraTensor.GetShapeType();
    const std::vector<int64_t> &validShape = paraTensor.GetValidShape();

    bool bres = GetPyShape(shapeType, validShape, pyTensor, "valid_shape");
    TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert valid shape to PyObject, shape size=[%zu].", validShape.size());
        return false;
    });

    if (NotZero(validShape)) {
        if (isInput) {
            // refresh_shape
            bres = GetPyShape(shapeType, validShape, pyTensor, "shape");
            TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                   "Failed to convert valid shape to PyObject, shape size=[%zu].", validShape.size());
                return false;
            });
        } else {
            // refresh_total_shape
            bres = GetPyShape(ATTR_SHAPE_LIST, validShape, pyTensor, "total_shape");
            TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                   "Failed to convert valid shape to PyObject, shape size=[%zu].", validShape.size());
                return false;
            });
        }
    }

    return true;
}

bool GetPyObjOfSliceOffset(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    ATTR_SHAPETYPE shapeType = paraTensor.GetShapeType();
    const std::vector<int64_t> &sliceOffset = paraTensor.GetSliceOffset();

    bool bres = GetPyShape(shapeType, sliceOffset, pyTensor, "slice_offset");
    TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert slice offset to PyObject, shape size=[%zu].", sliceOffset.size());
        return false;
    });

    return true;
}

bool GetPyObjOfL1Info(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    int64_t l1AddrFlag = paraTensor.GetL1AddrFlag();
    int64_t l1AddrOffset = paraTensor.GetAddrOffset();
    int64_t l1ValidSize = paraTensor.GetL1ValidSize();
    int32_t l1FusionType = paraTensor.GetL1FusionType();
    int64_t l1WorkspaceSize = paraTensor.GetL1WorkspaceSize();

    TE_FUSION_CHECK(!SetIntInfoToPyObj(l1AddrOffset, "l", "L1_addr_offset", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });

    if (l1AddrFlag != -1) {
        TE_FUSION_CHECK(!SetIntInfoToPyObj(l1AddrFlag, "l", "L1_addr_flag", pyTensor), {
            TE_ERRLOG("Failed to SetIntInfoToPyObj.");
            return false;
        });
    }

    TE_FUSION_CHECK(!SetIntInfoToPyObj(l1FusionType, "i", "L1_fusion_type", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });

    TE_FUSION_CHECK(!SetIntInfoToPyObj(l1WorkspaceSize, "l", "L1_workspace_size", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });

    if (l1AddrFlag == 1) {
        TE_FUSION_CHECK(!SetIntInfoToPyObj(l1ValidSize, "l", "L1_valid_size", pyTensor), {
            TE_ERRLOG("Failed to SetIntInfoToPyObj.");
            return false;
        });
    }

    return true;
}

bool GetPyObjOfTotalShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    const std::vector<int64_t> &curShape = paraTensor.GetShape();

    bool bres = GetPyShape(ATTR_SHAPE_LIST, curShape, pyTensor, "total_shape");
    TE_FUSION_CHECK_WITH_DUMP_PYERR((!bres), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to convert current shape to PyObject, shape size=[%zu].", curShape.size());
        return false;
    });

    return true;
}

bool GetPyObjSplitIndex(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    uint32_t splitIndex = paraTensor.GetSplitIndex();
    TE_FUSION_CHECK(!SetIntInfoToPyObj(splitIndex, "I", "split_index", pyTensor), {
        TE_ERRLOG("Failed to SetIntInfoToPyObj.");
        return false;
    });

    return true;
}

bool GetPyObjOfIsFirstLayer(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    bool isFirstLayer = false;
    bool hasSet = paraTensor.GetFirstLayer(isFirstLayer);

    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    bool isPyTrueFalseNullptr = pyTrue == nullptr || pyFalse == nullptr;
    if (isPyTrueFalseNullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse from class HandleManager.");
        return false;
    }

    if (hasSet) {
        int ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, "is_first_layer",
                                                                     isFirstLayer ? pyTrue : pyFalse);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add is_first_layer to dictionary.");
            return false;
        });
    }

    return true;
}

bool GetPyObjOfShapeRange(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    const std::vector<int64_t> &curShape = paraTensor.GetShape();
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    bool hasSet = paraTensor.GetShapeRange(shapeRange);
    TE_FUSION_CHECK(!(SetRangeToPyObj(hasSet, shapeRange, curShape, "range", pyTensor)), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to SetRangeToPyObj for range.");
        return false;
    });

    return true;
}

bool GetPyObjOfOriginalShapeRange(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    const std::vector<int64_t> &oriShape = paraTensor.GetOriginShape();
    std::vector<std::pair<int64_t, int64_t>> oriShapeRange;
    bool hasSet = paraTensor.GetOriginShapeRange(oriShapeRange);
    TE_FUSION_CHECK(!(SetRangeToPyObj(hasSet, oriShapeRange, oriShape, "ori_range", pyTensor)), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to SetRangeToPyObj for ori_range.");
        return false;
    });

    return true;
}

bool GetPyObjOfValueRange(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    const std::vector<int64_t> &curShape = paraTensor.GetShape();
    std::vector<std::pair<int64_t, int64_t>> valueRange;
    bool hasSet = paraTensor.GetValueRange(valueRange);
    if (hasSet) {
        TE_FUSION_CHECK(!(SetRangeToPyObj(hasSet, valueRange, curShape, "value_range", pyTensor)), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to SetRangeToPyObj for value_range.");
            return false;
        })
    }

    return true;
}

bool GetPyObjOfConstValue(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    bool isConst = false;
    (void)paraTensor.GetConstFlag(isConst);

    if (isConst) {
        TE_FUSION_CHECK(!(GetPyConstValue(paraTensor, pyTensor)), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get PyConstValue.");
            return false;
        });
    }

    return true;
}

bool GetPyObjAtomicAtrr(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    std::string atomicType;
    std::string dtype;
    paraTensor.GetAtomicType(atomicType);
    paraTensor.GetType(dtype);
    if (!atomicType.empty()) {
        atomicType = atomicType + "." + dtype;
    }

    TE_FUSION_CHECK(!SetStringInfoToPyObj(atomicType, "atomic_type", pyTensor), {
        TE_ERRLOG("Failed to Set atomicType to pyObj.");
        return false;
    });

    return true;
}

bool GetPyObjectOfInputConst(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
    (void)opinfo;
    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    bool isPyTrueFalseNullptr = pyTrue == nullptr || pyFalse == nullptr;
    if (isPyTrueFalseNullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse from class HandleManager.");
        return false;
    }
    if (paraTensor.GetInputConst() == -1) {
        TE_DBGLOG("Not setting input const for aoe_tuning scenario.");
        return true;
    }
    bool isInputConst = paraTensor.GetInputConst();
    int ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, "is_input_const",
                                                                 isInputConst ? pyTrue : pyFalse);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add is_input_const to pyObj.");
        return false;
    });

    return true;
}

bool GetPyObjectOfNullableOutputExist(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo)
{
 	(void)opinfo;
 	PyObject *pyTrue = HandleManager::Instance().get_py_true();
 	PyObject *pyFalse = HandleManager::Instance().get_py_false();
 	bool isPyTrueFalseNullptr = pyTrue == nullptr || pyFalse == nullptr;
 	if (isPyTrueFalseNullptr) {
 	    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse from class HandleManager.");
 	    return false;
 	}
 	bool is_null_output = false;
 	paraTensor.GetIsNullOutput(is_null_output);
 	if(is_null_output) {
 	    TE_DBGLOG("Get the is_null_output as True.");
 	    int ires = HandleManager::Instance().TE_PyDict_SetItemString(pyTensor, "is_null_output",
 	                                                                is_null_output ? pyTrue : pyFalse);
 	    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
 	        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add is_null_output to pyObj.");
 	        return false;
 	    });
 	}

    return true;
}

const std::vector<GetPyTensorByTbeOpTensor> PY_TENSOR_GET_FUNCS = {
    GetPyObjOfShape,          GetPyObjOfOriShape,         GetPyObjOfFormat,               GetPyObjOfCurSubFormat,
    GetPyObjOfOriFormat,      GetPyObjOfDtype,            GetPyObjOfAddrType,             GetPyObjOfTotalShape,
    GetPyObjOfSliceOffset,    GetPyObjOfL1Info,           GetPyObjOfValidShape,           GetPyObjSplitIndex,
    GetPyObjOfIsFirstLayer,   GetPyObjOfShapeRange,       GetPyObjOfOriginalShapeRange,   GetPyObjOfValueRange,
    GetPyObjOfConstValue,     GetPyObjAtomicAtrr,         GetPyObjOfCAxisValue,           GetPyObjectOfInputConst,
    GetPyObjectOfNullableOutputExist
};

/*
 * @brief: get op python tensor parameter
 * @param [in] paraTensor: op tensor info
 * @param [out] pyTensor: op python tensor result
 * @return bool: get python tensor ok or not
 */
bool GetPyTensor(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &isInput, const TbeOpInfo &opinfo)
{
    pyTensor = HandleManager::Instance().TE_PyDict_New();
    TE_FUSION_CHECK(pyTensor == nullptr, return false);

    for (auto iter = PY_TENSOR_GET_FUNCS.begin(); iter != PY_TENSOR_GET_FUNCS.end(); ++iter) {
        TE_FUSION_CHECK(!((*iter)(paraTensor, pyTensor, isInput, opinfo)), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get py tensor.");
            return false;
        });
    }
    return true;
}

/*
 * @brief: convert options op tensor parameter from class to PyObject
 * @param [in] tensors: options op tensor parameter set
 * @param [out] pArgs: op PyObject parameter set
 * @param [in&out] argsIdx: current PyObject parameter index
 * @return bool: convert op parameter ok or not
 */
bool AddOptTensorArgs(const std::vector<TbeOpTensor> &tensors, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                      const TbeOpInfo &opinfo)
{
    PyObject *pTensinfo = nullptr;
    int ires = 0;

    TE_FUSION_CHECK((tensors.size() > 1), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Tensor of option type has wrong size[%zu].", tensors.size());
        return false;
    });
    TE_FUSION_CHECK((tensors.size() == 0), {
        // tensor size is 0, add null to para
        PyObject *pyNull = HandleManager::Instance()._Py_BuildValue("");
        TE_FUSION_CHECK_WITH_DUMP_PYERR((pyNull == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to convert parameter from null to PyObject.");
            return false;
        });
        ires = HandleManager::Instance().TE_PyTuple_SetItem(pArgs, argsIdx++, pyNull);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add shape PyObject to total args.");
            return false;
        });
    });
    TE_FUSION_CHECK((tensors.size() == 1), {
        bool res = GetPyTensor(tensors[0], pTensinfo, isInput, opinfo);
        TE_FUSION_CHECK((!res || pTensinfo == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get option tensor PyObject.");
            return false;
        });
        ires = HandleManager::Instance().TE_PyTuple_SetItem(pArgs, argsIdx++, pTensinfo);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add tensor PyObject to total args.");
            return false;
        });
    });

    return true;
}

/*
 * @brief: convert dynamic op tensor parameter from class to PyObject
 * @param [in] tensors: dynamic op tensor parameter set
 * @param [out] pArgs: op PyObject parameter set
 * @param [in&out] argsIdx: current PyObject parameter index
 * @return bool: convert op parameter ok or not
 */
bool AddDynTensorArgs(const std::vector<TbeOpTensor> &tensors, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                      const TbeOpInfo &opinfo)
{
    PyObject *pTensinfo = nullptr;
    PyObject *pTensArg = nullptr;
    int ires;

    TE_FUSION_CHECK((tensors.size() == 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Dynamic tensor is null, it's error.");
        return false;
    });
    // tensors.size must be > 0 here
    pTensArg = HandleManager::Instance().TE_PyTuple_New(tensors.size());
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pTensArg == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create dyn tensor PyObject value.");
        return false;
    });

    int tensorIdx = 0;
    for (auto &inTensor : tensors) {
        bool res = GetPyTensor(inTensor, pTensinfo, isInput, opinfo);
        TE_FUSION_CHECK((!res || pTensinfo == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get dynamic tensor PyObject.");
            return false;
        });

        ires = HandleManager::Instance().TE_PyTuple_SetItem(pTensArg, tensorIdx++, pTensinfo);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add one tensor to dynamic args.");
            return false;
        });
    }
    ires = HandleManager::Instance().TE_PyTuple_SetItem(pArgs, argsIdx++, pTensArg);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add all dynamic tensors to total args.");
        return false;
    });

    return true;
}

/*
 * @brief: convert request op tensor parameter from class to PyObject
 * @param [in] tensors: request op tensor parameter set
 * @param [out] pArgs: op PyObject parameter set
 * @param [in&out] argsIdx: current PyObject parameter index
 * @return bool: convert op parameter ok or not
 */
bool AddReqTensorArgs(const std::vector<TbeOpTensor> &tensors, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                      const TbeOpInfo &opinfo)
{
    PyObject *pTensinfo = nullptr;

    TE_FUSION_CHECK((tensors.size() != 1), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Tensor of request type has wrong size[%lu].", tensors.size());
        return false;
    });

    bool res = GetPyTensor(tensors[0], pTensinfo, isInput, opinfo);
    TE_FUSION_CHECK((!res || pTensinfo == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get request tensor PyObject.");
        return false;
    });

    int ires = HandleManager::Instance().TE_PyTuple_SetItem(pArgs, argsIdx++, pTensinfo);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to add request tensor PyObject to total args.");
        return false;
    });

    return true;
}

template<typename T> void PyAttrIntGetter(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    T value = {};
    attr.GetValue(value);
    pyAttr = HandleManager::Instance().TE_PyLong_FromLong(value);
}

std::map<ATTR_DTYPE, std::function<void(const TbeAttrValue&, PyObject *&)>> g_pyAttrIntGetFuncs = {
    {ATTR_INT8,         PyAttrIntGetter<int8_t>},
    {ATTR_UINT8,        PyAttrIntGetter<uint8_t>},
    {ATTR_INT16,        PyAttrIntGetter<int16_t>},
    {ATTR_UINT16,       PyAttrIntGetter<uint16_t>},
    {ATTR_INT32,        PyAttrIntGetter<int32_t>},
    {ATTR_UINT32,       PyAttrIntGetter<uint32_t>},
    {ATTR_INT64,        PyAttrIntGetter<int64_t>},
    {ATTR_UINT64,       PyAttrIntGetter<uint64_t>}
};

/*
 * @brief: get op python attribute parameter
 * @param [in] attr: op attribute info
 * @param [in] listInt: attri list value
 * @param [out] pyAttr: op python attribute result
 * @return bool: get python attribute ok or not
 */
template<typename T> bool GetPyAttrListInt(const TbeAttrValue &attr, T &listInt, PyObject *&pyAttr)
{
    attr.GetValue(listInt);
    pyAttr = HandleManager::Instance().TE_PyTuple_New(listInt.size());
    TE_FUSION_CHECK((pyAttr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create int attribute PyObject.");
        return false;
    });
    int idx = 0;
    for (const auto &dim : listInt) {
        PyObject *pDimShape = HandleManager::Instance().TE_PyLong_FromLong(dim);
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttr, idx++, pDimShape);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0),
                                        {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                                            "Failed to add int attribute to tuple PyObject, index=[%d], list size=[%zu].",
                                                            idx, listInt.size());
                                            return false;}
        );
    }
    return true;
}

bool SimpleAttrForFp32(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    float singleFloat = 0.0;
    attr.GetValue(singleFloat);
    pyAttr = HandleManager::Instance().TE_PyFloat_FromDouble(singleFloat);

    return true;
}

bool SimpleAttrForDouble(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    double singleDouble = 0.0;
    attr.GetValue(singleDouble);
    pyAttr = HandleManager::Instance().TE_PyFloat_FromDouble(singleDouble);

    return true;
}

bool SimpleAttrForBool(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    bool singleBool = true;
    attr.GetValue(singleBool);

    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    TE_FUSION_CHECK((pyTrue == nullptr || pyFalse == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse from class HandleManager.");
        return false;
    });
    pyAttr = singleBool ? pyTrue : pyFalse;
    Py_XINCREF(pyAttr);

    return true;
}

bool SimpleAttrForString(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    string singleStr = "";
    attr.GetValue(singleStr);
    pyAttr = HandleManager::Instance().TE_PyUnicode_FromString(singleStr.c_str());

    return true;
}

bool SimpleAttrForInt8(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<int8_t> listInt8;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listInt8, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get int8 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForUint8(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<uint8_t> listUint8;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listUint8, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get uint8 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForInt16(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<int16_t> listInt16;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listInt16, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get int16 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForUint16(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<uint16_t> listUint16;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listUint16, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get uint16 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForInt32(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<int32_t> listInt32;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listInt32, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get int32 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForUint32(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<uint32_t> listUint32;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listUint32, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get uint32 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForInt64(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<int64_t> listInt64;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listInt64, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get int64 PyObject attribute value.");
        return false;
    });

    return true;
}

bool SimpleAttrForUint64(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<uint64_t> listUint64;
    TE_FUSION_CHECK(!GetPyAttrListInt(attr, listUint64, pyAttr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get uint64 PyObject attribute value.");
        return false;
    });

    return true;
}

const std::map<ATTR_DTYPE, GetSimpleAttrPy> PY_SIMPLE_ATTR_GET_FUNCS = {
    {ATTR_FLOAT32, SimpleAttrForFp32},
    {ATTR_DOUBLE, SimpleAttrForDouble},
    {ATTR_BOOL, SimpleAttrForBool},
    {ATTR_STR, SimpleAttrForString},
    {ATTR_LIST_INT8, SimpleAttrForInt8},
    {ATTR_LIST_UINT8, SimpleAttrForUint8},
    {ATTR_LIST_INT16, SimpleAttrForInt16},
    {ATTR_LIST_UINT16, SimpleAttrForUint16},
    {ATTR_LIST_INT32, SimpleAttrForInt32},
    {ATTR_LIST_UINT32, SimpleAttrForUint32},
    {ATTR_LIST_INT64, SimpleAttrForInt64},
    {ATTR_LIST_UINT64, SimpleAttrForUint64},
};

/*
 * @brief: get op python attribute parameter
 * @param [in] attr: op attribute info
 * @param [out] pyAttr: op python attribute result
 * @return bool: get python attribute ok or not
 */
bool GetPyAttrSimple(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    ATTR_DTYPE attrDtype = attr.GetType();
    auto func = PY_SIMPLE_ATTR_GET_FUNCS.find(attrDtype);
    if (func == PY_SIMPLE_ATTR_GET_FUNCS.end()) {
        REPORT_TE_INNER_ERROR("Attr dtype[%d] is invalid.", attrDtype);
        return false;
    } else {
        if (!func->second(attr, pyAttr)) {
            REPORT_TE_INNER_ERROR("Failed to get pyAttrSimple obj for %d.", attrDtype);
            return false;
        }
    }

    return true;
}

bool ComplexAttrForListFp32(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<float> listFloat;
    attr.GetValue(listFloat);
    pyAttr = HandleManager::Instance().TE_PyTuple_New(listFloat.size());
    TE_FUSION_CHECK((pyAttr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create float tuple PyObject value.");
        return false;
    });
    int idx = 0;
    for (const auto &dim : listFloat) {
        PyObject *pDimShape = HandleManager::Instance().TE_PyFloat_FromDouble(dim);
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttr, idx++, pDimShape);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to add float attribute tuple PyObject value, index=[%d], tuple size=[%zu].",
                               idx, listFloat.size());
            return false;
        });
    }

    return true;
}

bool ComplexAttrForListDouble(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<double> listDouble;
    attr.GetValue(listDouble);
    pyAttr = HandleManager::Instance().TE_PyTuple_New(listDouble.size());
    TE_FUSION_CHECK((pyAttr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create python double tuple PyObject value.");
        return false;
    });
    int idx = 0;
    for (const auto &dim : listDouble) {
        PyObject *pDimShape = HandleManager::Instance().TE_PyFloat_FromDouble(dim);
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttr, idx++, pDimShape);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to add double attribute tuple PyObject value, index=[%d], tuple size=[%zu].",
                               idx, listDouble.size());
            return false;
        });
    }

    return true;
}

bool ComplexAttrForListBool(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<bool> listBool;
    attr.GetValue(listBool);
    pyAttr = HandleManager::Instance().TE_PyTuple_New(listBool.size());
    TE_FUSION_CHECK((pyAttr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create bool tuple PyObject value.");
        return false;
    });

    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    TE_FUSION_CHECK((pyTrue == nullptr || pyFalse == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse from class HandleManager.");
        return false;
    });

    int idx = 0;
    for (const auto &dim : listBool) {
        PyObject *pyBool = dim ? pyTrue : pyFalse;
        Py_XINCREF(pyBool);
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttr, idx++, pyBool);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to add bool attribute tuple PyObject value, index=[%d], tuple size=[%zu].",
                               idx, listBool.size());
            return false;
        });
    }

    return true;
}

bool ComplexAttrForListString(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<string> listStr;
    attr.GetValue(listStr);
    pyAttr = HandleManager::Instance().TE_PyTuple_New(listStr.size());
    TE_FUSION_CHECK((pyAttr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create string tuple PyObject value.");
        return false;
    });
    int idx = 0;
    for (const auto &dim : listStr) {
        PyObject *pDimShape = HandleManager::Instance().TE_PyUnicode_FromString(dim.c_str());
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttr, idx++, pDimShape);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to add string attribute to python tuple, index=[%d], tuple size=[%zu].",
                               idx, listStr.size());
            return false;
        });
    }

    return true;
}

bool ComplexAttrForListInt64(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    std::vector<std::vector<int64_t>> listListInt64;
    attr.GetValue(listListInt64);
    pyAttr = HandleManager::Instance().TE_PyTuple_New(listListInt64.size());
    TE_FUSION_CHECK((pyAttr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create python tuple.");
        return false;
    });

    int idx = 0;
    int ires;
    for (const auto &vec : listListInt64) {
        PyObject *pyTuple = HandleManager::Instance().TE_PyTuple_New(vec.size());
        TE_FUSION_CHECK((pyTuple == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create python tuple.");
            return false;
        });

        int idx2 = 0;
        for (const auto &dim : vec) {
            PyObject *pDimShape = HandleManager::Instance().TE_PyLong_FromLong(dim);
            ires = HandleManager::Instance().TE_PyTuple_SetItem(pyTuple, idx2++, pDimShape);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0),
                                            {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                                                "Failed to add int attribute to python tuple, index=[%d], list size=[%zu].",
                                                                idx2, vec.size());
                                                return false;
                                            });
        }

        ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttr, idx++, pyTuple);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to add string attribute to python tuple, index=[%d], tuple size=[%zu].",
                               idx, listListInt64.size());
            return false;
        });
    }

    return true;
}

const std::map<ATTR_DTYPE, GetComplexAttrPy> PY_COMPLEX_ATTR_GET_FUNCS = {
    {ATTR_LIST_FLOAT32, ComplexAttrForListFp32},
    {ATTR_LIST_DOUBLE, ComplexAttrForListDouble},
    {ATTR_LIST_BOOL, ComplexAttrForListBool},
    {ATTR_LIST_STR, ComplexAttrForListString},
    {ATTR_LIST_LIST_INT64, ComplexAttrForListInt64},
};
/*
 * @brief: get op python attribute parameter
 * @param [in] attr: op attribute info
 * @param [out] pyAttr: op python attribute result
 * @return bool: get python attribute ok or not
 */
bool GetPyAttrComplex(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    ATTR_DTYPE attrDtype = attr.GetType();
    auto func = PY_COMPLEX_ATTR_GET_FUNCS.find(attrDtype);
    if (func == PY_COMPLEX_ATTR_GET_FUNCS.end()) {
        REPORT_TE_INNER_ERROR("Attr dtype[%d] is invalid.", attrDtype);
        return false;
    }
    if (!func->second(attr, pyAttr)) {
        REPORT_TE_INNER_ERROR("Failed to get pyAttrComplex obj for %d.", attrDtype);
        return false;
    }
    return true;
}

/*
 * @brief: get op python attribute parameter
 * @param [in] attr: op attribute info
 * @param [out] pyAttr: op python attribute result
 * @return bool: get python attribute ok or not
 */
bool GetPyAttr(const TbeAttrValue &attr, PyObject *&pyAttr)
{
    bool bres = false;
    ATTR_DTYPE attrDtype = attr.GetType();
    const std::map<ATTR_DTYPE, std::function<void(const TbeAttrValue&, PyObject *&)>>::const_iterator func =
        g_pyAttrIntGetFuncs.find(attrDtype);
    if (func != g_pyAttrIntGetFuncs.end()) {
        (func->second)(attr, pyAttr);
        return true;
    }

    bool isNum = (attrDtype == ATTR_FLOAT32 ||
                  attrDtype == ATTR_DOUBLE  ||
                  attrDtype == ATTR_BOOL    ||
                  attrDtype == ATTR_STR     ||
                  attrDtype == ATTR_LIST_INT8   ||
                  attrDtype == ATTR_LIST_UINT8  ||
                  attrDtype == ATTR_LIST_INT16  ||
                  attrDtype == ATTR_LIST_UINT16 ||
                  attrDtype == ATTR_LIST_INT32  ||
                  attrDtype == ATTR_LIST_UINT32 ||
                  attrDtype == ATTR_LIST_INT64  ||
                  attrDtype == ATTR_LIST_UINT64);

    bool isList = (attrDtype == ATTR_LIST_FLOAT32 ||
                   attrDtype == ATTR_LIST_DOUBLE  ||
                   attrDtype == ATTR_LIST_BOOL  ||
                   attrDtype == ATTR_LIST_STR  ||
                   attrDtype == ATTR_LIST_LIST_INT64);

    if (isNum) {
        bres = GetPyAttrSimple(attr, pyAttr);
        TE_FUSION_CHECK(!bres, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to get pyoject simple attr, attrDtype=[%d].", attrDtype);
            return false;
        });
    } else if (isList) {
        bres = GetPyAttrComplex(attr, pyAttr);
        TE_FUSION_CHECK(!bres, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to get pyoject complex attr, attrDtype=[%d].", attrDtype);
            return false;
        });
    } else {
    }
    return true;
}

bool PyObjectToPyFullAttr(const TbeAttrValue &attr, PyObject *&pyAttr, PyObject *&pyFullAttr,
                          std::vector<std::string> &variableAttrs)
{
    if (JudgeAttrIsVariableAttr(attr, variableAttrs)) {
        TE_DBGLOG("Attr'name is in variableAttr value, set the attr value to None.");
        pyAttr = HandleManager::Instance().get_py_none();
        Py_XINCREF(pyAttr);
    } else {
        if (!GetPyAttr(attr, pyAttr)) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to convert attribute to PyObject.");
            return false;
        }
    }

    PyObject *pyName = HandleManager::Instance()._Py_BuildValue("s", attr.GetName().c_str());
    TE_FUSION_CHECK((pyName == nullptr), {
        TE_ERRLOG("Build of optionsValue[pyName] failed with [%s]", attr.GetName().c_str());
        return false;
    });
    AUTO_PY_DECREF(pyName);

    std::string attrTypeStr;
    (void)TbeAttrDtypeToString(attr.GetType(), attrTypeStr);
    PyObject *pyDtype = HandleManager::Instance()._Py_BuildValue("s", attrTypeStr.c_str());
    TE_FUSION_CHECK((pyDtype == nullptr), {
        TE_ERRLOG("build optionsValue[pyDtype] failed with [%s]", attrTypeStr.c_str());
        return false;
    });
    AUTO_PY_DECREF(pyDtype);
    TE_DBGLOG("Build options for pyName[%s] and pyDtype[%s].", attr.GetName().c_str(), attrTypeStr.c_str());

    int ires = HandleManager::Instance().TE_PyDict_SetItemString(pyFullAttr, "name", pyName);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attribute pyName to total args.");
        return false;
    });

    ires = HandleManager::Instance().TE_PyDict_SetItemString(pyFullAttr, "dtype", pyDtype);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attribute pyDtype to total args.");
        return false;
    });

    ires = HandleManager::Instance().TE_PyDict_SetItemString(pyFullAttr, "value", pyAttr);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attribute pyAttr to total args.");
        return false;
    });

    TE_DBGLOG("Get isDefaultValue is [%s].", attr.GetIsDefaultValue() ? "true" : "false");
    if (attr.GetIsDefaultValue()) {
        PyObject *pyTrue = HandleManager::Instance().get_py_true();
        ires = HandleManager::Instance().TE_PyDict_SetItemString(pyFullAttr, "is_default_value", pyTrue);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attribute is_default_value to total args.");
            return false;
        });
    }
    return true;
}

/*
 * @brief: convert op attribute parameter from class to PyObject
 * @param [in] opinfo: op total parameter set
 * @param [out] pArgs: op PyObject parameter set
 * @param [in&out] argsIdx: current PyObject parameter index
 * @return bool: convert op parameter ok or not
 */
bool AddAttrArgs(const std::vector<TbeAttrValue> &attrs, PyObject *&pArgs, int32_t &argsIdx, bool isSingleOpBuild,
                 std::vector<std::string> &variableAttrs)
{
    PyObject *pyAttr = nullptr;
    if (isSingleOpBuild) {
        for (auto &attr : attrs) {
            PyObject *pyFullAttr = HandleManager::Instance().TE_PyDict_New();
            TE_FUSION_CHECK((pyFullAttr == nullptr), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to create python dict.");
                return false;
            });
            bool setFullAttrFlag = PyObjectToPyFullAttr(attr, pyAttr, pyFullAttr, variableAttrs);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((!setFullAttrFlag), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to set attribute PyObject to total args.");
                return false;
            });
            int ires = HandleManager::Instance().TE_PyTuple_SetItem(pArgs, argsIdx++, pyFullAttr);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attribute PyObject to total args.");
                return false;
            });
            TE_DBGLOGF("pyFullAttr: %s.", PyObjectToStr(pyFullAttr).c_str());
            TE_PY_DECREF(pyAttr);
        }
    } else {
        for (auto &attr : attrs) {
            bool flag = JudgeAttrIsVariableAttr(attr, variableAttrs);
            if (flag) {
                TE_DBGLOGF("Attr'name is in variableAttr value, set the attr value to None.");
                pyAttr = HandleManager::Instance().get_py_none();
            } else {
                bool res = GetPyAttr(attr, pyAttr);
                TE_FUSION_CHECK(!res, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to convert attribute to PyObject.");
                    return false;
                });
            }
            int ires = HandleManager::Instance().TE_PyTuple_SetItem(pArgs, argsIdx++, pyAttr);
            TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add attribute PyObject to total args.");
                return false;
            });
        }
    }
    return true;
}

/*
 * @brief: convert op tensor parameter from class to PyObject
 * @param [in] puts: op tensor parameter set
 * @param [out] pArgs: op PyObject parameter set
 * @param [in&out] argsIdx: current PyObject parameter index
 * @return bool: convert op parameter ok or not
 */
bool AddTensorArgs(const std::vector<TbeOpParam> &puts, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                   const TbeOpInfo &opinfo)
{
    bool bres = true;

    for (auto &put : puts) {
        const std::vector<TbeOpTensor> &tensors = put.GetTensors();
        TensorType tsType = put.GetType();
        switch (tsType) {
            // one tensor in one input
            case TT_REQ: {
                bres = AddReqTensorArgs(tensors, pArgs, argsIdx, isInput, opinfo);
                TE_FUSION_CHECK(!bres, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add request Tensor to PyObject args.");
                    return false;
                });
                break;
            }
                // one or zero tensor in one input
            case TT_OPT: {
                bres = AddOptTensorArgs(tensors, pArgs, argsIdx, isInput, opinfo);
                TE_FUSION_CHECK(!bres, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add request Tensor to PyObject args.");
                    return false;
                });
                break;
            }
                // one or more tensor in one input
            case TT_DYN: {
                bres = AddDynTensorArgs(tensors, pArgs, argsIdx, isInput, opinfo);
                TE_FUSION_CHECK(!bres, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add dynamic Tensor to PyObject args.");
                    return false;
                });
                break;
            }
            default: {
                break;
            }
        }
    }

    return true;
}

bool AssembleInputsAndOutputs(const TbeOpInfo &opinfo, PyObject *&pyInputs, PyObject *&pyOutputs)
{
    bool isInput = true;
    const std::vector<TbeOpParam> &inputs = opinfo.GetInputs();
    pyInputs = HandleManager::Instance().TE_PyTuple_New(inputs.size());
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyInputs == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to PyTuple_New %zu.", inputs.size());
        return false;
    });
    int32_t input_idx = 0;
    if (!AddTensorArgs(inputs, pyInputs, input_idx, isInput, opinfo)) {
        TE_ERRLOG("Failed to add input tensor for node[name:%s, type:%s], idx[%d].",
                  opinfo.GetName().c_str(), opinfo.GetOpType().c_str(), input_idx);
        return false;
    }

    isInput = false;
    const std::vector<TbeOpParam> &outputs = opinfo.GetOutputs();
    pyOutputs = HandleManager::Instance().TE_PyTuple_New(outputs.size());
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyOutputs == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to PyTuple_New %zu.", outputs.size());
        return false;
    });
    int32_t output_idx = 0;
    if (!AddTensorArgs(outputs, pyOutputs, output_idx, isInput, opinfo)) {
        TE_ERRLOG("Failed to add output tensor for node[name:%s, type:%s], idx[%d].",
                  opinfo.GetName().c_str(), opinfo.GetOpType().c_str(), output_idx);
        return false;
    }
    return true;
}

bool AssembleAttrs(const TbeOpInfo &opInfo, const std::string &kernelName, PyObject *&pyAttrs, bool isSingleOpBuild)
{
    bool result = false;
    int32_t index = 0;
    int32_t size = 0;
    const std::vector<TbeAttrValue> &attrs = opInfo.GetAttrValues();
    if (kernelName != "") {
        size = attrs.size() + 1;
    } else {
        size = attrs.size();
    }

    PyObject *pyAttrTemp = HandleManager::Instance().TE_PyTuple_New(size);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyAttrTemp == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to TE_PyTuple_New %zu.", size);
        return false;
    });

    std::vector<std::string> variableAttrs;
    GetVariableAttrValue(opInfo, variableAttrs);
    result = AddAttrArgs(attrs, pyAttrTemp, index, isSingleOpBuild, variableAttrs);
    TE_FUSION_CHECK((!result), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add op attrs.");
        return false;
    });

    if (kernelName != "") {
        PyObject *pyName = HandleManager::Instance().TE_PyUnicode_FromString(kernelName.c_str());
        TE_FUSION_CHECK_WITH_DUMP_PYERR((pyName == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to TE_PyUnicode_FromString %s.", kernelName.c_str());
            return false;
        });
        int ires = HandleManager::Instance().TE_PyTuple_SetItem(pyAttrTemp, index++, pyName);
        TE_FUSION_CHECK((ires != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add kernel name to total args.");
            return false;
        });
    }

    pyAttrs = pyAttrTemp;
    return true;
}

bool AssembleOpPrivateAttrs(const TbeOpInfo &opInfo, PyObject *&pyPrivateAttrs, bool isSingleOpBuild)
{
    const std::vector<TbeAttrValue> &privateAttrs = opInfo.GetPrivateAttrValues();
    pyPrivateAttrs = HandleManager::Instance().TE_PyTuple_New(privateAttrs.size());
    std::vector<std::string> variableAttrs;
    GetVariableAttrValue(opInfo, variableAttrs);
    int32_t index = 0;
    bool result = AddAttrArgs(privateAttrs, pyPrivateAttrs, index, isSingleOpBuild, variableAttrs);
    TE_FUSION_CHECK((!result), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add op pirvate attrs.");
        return false;
    });
    return true;
}

/*
 * @brief: convert op total parameters from class to PyObject
 * @param [in] opinfo: op total parameter set
 * @param [out] pArgs: op all PyObject parameter set
 * @return bool: convert op parameter ok or not
 */
bool AssembleOpArgs(const TbeOpInfo &opinfo, const std::string &kernelName,
                    PyObject *&pyInputs, PyObject *&pyOutputs, PyObject *&pyAttrs, bool isSingleOpBuild)
{
    bool res = false;
    PyLockGIL pyLockGIL;
    if (!AssembleInputsAndOutputs(opinfo, pyInputs, pyOutputs)) {
        return false;
    }

    res = AssembleAttrs(opinfo, kernelName, pyAttrs, isSingleOpBuild);
    TE_FUSION_CHECK((!res), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to add op attrs.");
        return false;
    });
    return true;
}

bool AssemleStringDict(const std::map<std::string, std::string> &infoMap, PyObject *&pyInfoDict)
{
    for (auto iter = infoMap.begin(); iter != infoMap.end(); iter++) {
        PyObject *pyValue = HandleManager::Instance()._Py_BuildValue("s", iter->second.c_str());
        TE_FUSION_CHECK((pyValue == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "build pyValue PyObject failed with args: %s",
                               iter->second.c_str());
            return false;
        });
        AUTO_PY_DECREF(pyValue);
        int res = HandleManager::Instance().TE_PyDict_SetItemString(pyInfoDict, iter->first.c_str(), pyValue);
        TE_FUSION_CHECK((res != 0), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "build TE_PyDict_SetItemString failed: [%s, %s]",
                               iter->first.c_str(), iter->second.c_str());
            return false;
        });
    }
    return true;
}
} // namespace fusion
} // namespace te
