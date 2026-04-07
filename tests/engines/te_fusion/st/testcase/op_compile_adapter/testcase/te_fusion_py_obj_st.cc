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

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/ChainingMockHelper.h>
#include <gtest/gtest_pred_impl.h>

#define private public
#define protected public
#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"
#include "file_handle/te_file_handle.h"
#include "te_fusion_base.h"
#include "python_adapter/pyobj_assemble_utils.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace te::fusion;


class TeFusionPyObjSTest : public testing::Test
{
    public:
        TeFusionPyObjSTest(){}
    protected:
        virtual void SetUp()
        {
        }
        virtual void TearDown()
        {
            GlobalMockObject::verify();
            GlobalMockObject::reset();
        }
    protected:

};

TEST(TeFusionPyObjSTest, GetPyConstValue) {
    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor op_tensor("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    PyObject *py_obj = PyDict_New();
    bool res = GetPyConstValue(op_tensor, py_obj);

    EXPECT_EQ(res, true);
}

TEST(TeFusionPyObjSTest, AddDynTensorArgs) {
    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "bool", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_DYN, tensors);
    std::vector<TbeOpParam> paras;
    paras.emplace_back(opinput1);
    PyObject *pyPara = PyTuple_New(paras.size());
    int32_t idx = 0;
    bool is_input = true;
    TbeOpInfo opInfo("test1", "", "Test", "AIcoreEngine");
    bool res = AddTensorArgs(paras, pyPara, idx, is_input, opInfo);
    EXPECT_EQ(res, true);
}

TEST(TeFusionPyObjSTest, AddAtomicAtrr) {
    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "bool", "ND", ATTR_SHAPE_LIST);
    tensorop.SetAtomicType("add");
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam op_output(TT_REQ, tensors);
    std::vector<TbeOpParam> paras;
    paras.emplace_back(op_output);
    PyObject *pyPara = PyTuple_New(paras.size());
    int32_t idx = 0;
    bool is_input = false;
    TbeOpInfo opInfo("test1", "", "Test", "AIcoreEngine");
    bool res = AddTensorArgs(paras, pyPara, idx, is_input, opInfo);
    EXPECT_EQ(res, true);
}

TEST(TeFusionPyObjSTest, GetPyAttr) {
    TbeAttrValue attr_v_int32("axis", (int32_t)1);
    PyObject *PyAttr = nullptr;
    bool res = GetPyAttr(attr_v_int32, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_float("axis", (float)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_float, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_double("axis", (double)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_double, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_bool("axis", true);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_bool, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_string("axis", "1");
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_string, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_int8("axis", (int8_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_int8, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_uint8("axis", (uint8_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_uint8, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_int16("axis", (int16_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_int16, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_uint16("axis", (uint16_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_uint16, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_uint32("axis", (uint32_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_uint32, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_int64("axis", (int64_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_int64, PyAttr);
    EXPECT_EQ(res, true);

    TbeAttrValue attr_v_uint64("axis", (uint64_t)1);
    PyAttr = nullptr;
    res = GetPyAttr(attr_v_uint64, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<int8_t>  list_int8_v;
    list_int8_v.push_back(1);
    list_int8_v.push_back(2);
    TbeAttrValue attr_v_list_int8("axis", list_int8_v);
    res = GetPyAttr(attr_v_list_int8, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<uint8_t>  list_uint8_v;
    list_uint8_v.push_back(1);
    list_uint8_v.push_back(2);
    TbeAttrValue attr_v_list_uint8("axis", list_uint8_v);
    res = GetPyAttr(attr_v_list_uint8, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<int16_t>  list_int16_v;
    list_int16_v.push_back(1);
    list_int16_v.push_back(2);
    TbeAttrValue attr_v_list_int16("axis", list_int16_v);
    res = GetPyAttr(attr_v_list_int16, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<uint16_t>  list_uint16_v;
    list_uint16_v.push_back(1);
    list_uint16_v.push_back(2);
    TbeAttrValue attr_v_list_uint16("axis", list_uint16_v);
    res = GetPyAttr(attr_v_list_uint16, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<int32_t>  list_int32_v;
    list_int32_v.push_back(1);
    list_int32_v.push_back(2);
    TbeAttrValue attr_v_list_int32("axis", list_int32_v);
    res = GetPyAttr(attr_v_list_int32, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<uint32_t>  list_uint32_v;
    list_uint32_v.push_back(1);
    list_uint32_v.push_back(2);
    TbeAttrValue attr_v_list_uint32("axis", list_uint32_v);
    res = GetPyAttr(attr_v_list_uint32, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<int64_t>  list_int64_v;
    list_int64_v.push_back(1);
    list_int64_v.push_back(2);
    TbeAttrValue attr_v_list_int64("axis", list_int64_v);
    res = GetPyAttr(attr_v_list_int64, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<uint64_t>  list_uint64_v;
    list_uint64_v.push_back(1);
    list_uint64_v.push_back(2);
    TbeAttrValue attr_v_list_uint64("axis", list_uint64_v);
    res = GetPyAttr(attr_v_list_uint64, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<float>  list_float_v;
    list_float_v.push_back(1);
    list_float_v.push_back(2);
    TbeAttrValue attr_v_list_float("axis", list_float_v);
    res = GetPyAttr(attr_v_list_float, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<double>  list_double_v;
    list_double_v.push_back(1);
    list_double_v.push_back(2);
    TbeAttrValue attr_v_list_double("axis", list_double_v);
    res = GetPyAttr(attr_v_list_double, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<bool>  list_bool_v;
    list_bool_v.push_back(1);
    list_bool_v.push_back(2);
    TbeAttrValue attr_v_list_bool("axis", list_bool_v);
    res = GetPyAttr(attr_v_list_bool, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<std::string>  list_str_v;
    list_str_v.push_back("test1");
    list_str_v.push_back("test2");
    TbeAttrValue attr_v_list_str("axis", list_str_v);
    res = GetPyAttr(attr_v_list_str, PyAttr);
    EXPECT_EQ(res, true);

    std::vector<std::vector<int64_t>>  list_list_int64_v;
    list_list_int64_v.push_back(std::vector<int64_t>({1,2,3}));
    list_list_int64_v.push_back(std::vector<int64_t>({4,5,6}));
    TbeAttrValue attr_v_list_list_int64("axis", list_list_int64_v);
    res = GetPyAttr(attr_v_list_list_int64, PyAttr);
    EXPECT_EQ(res, true);
}