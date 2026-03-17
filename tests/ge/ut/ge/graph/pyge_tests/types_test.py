#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Types 功能测试 - 使用 pytest 框架
测试 types.py 中的类型定义
"""

import pytest
import sys
import os

# 添加 ge 到 Python 路径
try:
    from ge.graph.types import DataType, AttrValueType
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestDataType:
    """DataType 功能测试类"""

    def test_data_type_enum_values(self):
        """测试 DataType 枚举值"""
        # 测试基本数据类型
        assert DataType.DT_FLOAT == 0
        assert DataType.DT_FLOAT16 == 1
        assert DataType.DT_INT8 == 2
        assert DataType.DT_INT32 == 3
        assert DataType.DT_UINT8 == 4
        assert DataType.DT_INT16 == 6
        assert DataType.DT_UINT16 == 7
        assert DataType.DT_UINT32 == 8
        assert DataType.DT_INT64 == 9
        assert DataType.DT_UINT64 == 10
        assert DataType.DT_DOUBLE == 11
        assert DataType.DT_BOOL == 12
        assert DataType.DT_STRING == 13

    def test_data_type_enum_values_advanced(self):
        """测试 DataType 高级枚举值"""
        # 测试高级数据类型
        assert DataType.DT_DUAL_SUB_INT8 == 14
        assert DataType.DT_DUAL_SUB_UINT8 == 15
        assert DataType.DT_COMPLEX64 == 16
        assert DataType.DT_COMPLEX128 == 17
        assert DataType.DT_QINT8 == 18
        assert DataType.DT_QINT16 == 19
        assert DataType.DT_QINT32 == 20
        assert DataType.DT_QUINT8 == 21
        assert DataType.DT_QUINT16 == 22
        assert DataType.DT_RESOURCE == 23
        assert DataType.DT_STRING_REF == 24
        assert DataType.DT_DUAL == 25
        assert DataType.DT_VARIANT == 26
        assert DataType.DT_BF16 == 27
        assert DataType.DT_UNDEFINED == 28

    def test_data_type_enum_values_special(self):
        """测试 DataType 特殊枚举值"""
        # 测试特殊数据类型
        assert DataType.DT_INT4 == 29
        assert DataType.DT_UINT1 == 30
        assert DataType.DT_INT2 == 31
        assert DataType.DT_UINT2 == 32
        assert DataType.DT_COMPLEX32 == 33
        assert DataType.DT_HIFLOAT8 == 34
        assert DataType.DT_FLOAT8_E5M2 == 35
        assert DataType.DT_FLOAT8_E4M3FN == 36
        assert DataType.DT_FLOAT8_E8M0 == 37
        assert DataType.DT_FLOAT6_E3M2 == 38
        assert DataType.DT_FLOAT6_E2M3 == 39
        assert DataType.DT_FLOAT4_E2M1 == 40
        assert DataType.DT_FLOAT4_E1M2 == 41
        assert DataType.DT_MAX == 42

    def test_data_type_enum_inheritance(self):
        """测试 DataType 枚举继承"""
        # 测试 DataType 继承自 IntEnum
        assert isinstance(DataType.DT_FLOAT, int)
        assert isinstance(DataType.DT_INT32, int)
        assert isinstance(DataType.DT_STRING, int)

    def test_data_type_enum_comparison(self):
        """测试 DataType 枚举比较"""
        # 测试枚举值比较
        assert DataType.DT_FLOAT < DataType.DT_FLOAT16
        assert DataType.DT_INT8 < DataType.DT_INT32
        assert DataType.DT_STRING > DataType.DT_BOOL

    def test_data_type_enum_iteration(self):
        """测试 DataType 枚举迭代"""
        # 测试枚举迭代
        data_types = list(DataType)
        assert len(data_types) > 0
        assert DataType.DT_FLOAT in data_types
        assert DataType.DT_INT32 in data_types
        assert DataType.DT_STRING in data_types


    def test_data_type_enum_repr(self):
        """测试 DataType 枚举 repr"""
        # 测试枚举 repr
        assert repr(DataType.DT_FLOAT) == "<DataType.DT_FLOAT: 0>"
        assert repr(DataType.DT_INT32) == "<DataType.DT_INT32: 3>"
        assert repr(DataType.DT_STRING) == "<DataType.DT_STRING: 13>"


class TestAttrValueType:
    """AttrValueType 功能测试类"""

    def test_attr_data_type_enum_values(self):
        """测试 AttrValueType 枚举值"""
        # 测试基本属性数据类型
        assert AttrValueType.VT_STRING == 1
        assert AttrValueType.VT_FLOAT == 2
        assert AttrValueType.VT_BOOL == 3
        assert AttrValueType.VT_INT == 4
        assert AttrValueType.VT_DATA_TYPE == 11

    def test_attr_data_type_enum_values_lists(self):
        """测试 AttrValueType 列表枚举值"""
        # 测试列表属性数据类型
        assert AttrValueType.VT_LIST_FLOAT == 1002
        assert AttrValueType.VT_LIST_BOOL == 1003
        assert AttrValueType.VT_LIST_INT == 1004
        assert AttrValueType.VT_LIST_DATA_TYPE == 1011
        assert AttrValueType.VT_LIST_STRING == 1001

    def test_attr_data_type_enum_inheritance(self):
        """测试 AttrValueType 枚举继承"""
        # 测试 AttrValueType 继承自 IntEnum
        assert isinstance(AttrValueType.VT_STRING, int)
        assert isinstance(AttrValueType.VT_FLOAT, int)
        assert isinstance(AttrValueType.VT_LIST_STRING, int)

    def test_attr_data_type_enum_comparison(self):
        """测试 AttrValueType 枚举比较"""
        # 测试枚举值比较
        assert AttrValueType.VT_STRING < AttrValueType.VT_FLOAT
        assert AttrValueType.VT_FLOAT < AttrValueType.VT_BOOL
        assert AttrValueType.VT_LIST_STRING > AttrValueType.VT_STRING

    def test_attr_data_type_enum_iteration(self):
        """测试 AttrValueType 枚举迭代"""
        # 测试枚举迭代
        attr_data_types = list(AttrValueType)
        assert len(attr_data_types) > 0
        assert AttrValueType.VT_STRING in attr_data_types
        assert AttrValueType.VT_FLOAT in attr_data_types
        assert AttrValueType.VT_LIST_STRING in attr_data_types


    def test_attr_data_type_enum_repr(self):
        """测试 AttrValueType 枚举 repr"""
        # 测试枚举 repr
        assert repr(AttrValueType.VT_STRING) == "<AttrValueType.VT_STRING: 1>"
        assert repr(AttrValueType.VT_FLOAT) == "<AttrValueType.VT_FLOAT: 2>"
        assert repr(AttrValueType.VT_LIST_STRING) == "<AttrValueType.VT_LIST_STRING: 1001>"

    def test_attr_data_type_enum_usage_in_attr_value(self):
        """测试 AttrValueType 在 AttrValue 中的使用"""
        # 这个测试验证 AttrValueType 可以用于类型检查
        from ge.graph._attr import _AttrValue as AttrValue
        
        attr = AttrValue()
        
        # 测试字符串类型
        attr.set_string("test")
        assert attr.get_value_type() == AttrValueType.VT_STRING
        
        # 测试浮点数类型
        attr.set_float(3.14)
        assert attr.get_value_type() == AttrValueType.VT_FLOAT
        
        # 测试整数类型
        attr.set_int(42)
        assert attr.get_value_type() == AttrValueType.VT_INT
        
        # 测试布尔类型
        attr.set_bool(True)
        assert attr.get_value_type() == AttrValueType.VT_BOOL

    def test_attr_data_type_enum_usage_in_lists(self):
        """测试 AttrValueType 在列表中的使用"""
        from ge.graph._attr import _AttrValue as AttrValue
        
        attr = AttrValue()
        
        # 测试浮点数列表类型
        attr.set_list_float([1.0, 2.0, 3.0])
        assert attr.get_value_type() == AttrValueType.VT_LIST_FLOAT
        
        # 测试整数列表类型
        attr.set_list_int([1, 2, 3])
        assert attr.get_value_type() == AttrValueType.VT_LIST_INT
        
        # 测试布尔列表类型
        attr.set_list_bool([True, False, True])
        assert attr.get_value_type() == AttrValueType.VT_LIST_BOOL
        
        # 测试字符串列表类型
        attr.set_list_string(["hello", "world"])
        assert attr.get_value_type() == AttrValueType.VT_LIST_STRING

    @pytest.mark.parametrize("data_type", [
        DataType.DT_FLOAT,
        DataType.DT_INT32,
        DataType.DT_INT64,
        DataType.DT_BOOL,
        DataType.DT_STRING
    ])
    def test_data_type_enum_parametrized(self, data_type):
        """测试 DataType 枚举参数化"""
        assert isinstance(data_type, DataType)
        assert isinstance(data_type, int)
        assert data_type.value >= 0

    @pytest.mark.parametrize("attr_data_type", [
        AttrValueType.VT_STRING,
        AttrValueType.VT_FLOAT,
        AttrValueType.VT_BOOL,
        AttrValueType.VT_INT,
        AttrValueType.VT_LIST_STRING,
        AttrValueType.VT_LIST_FLOAT
    ])
    def test_attr_data_type_enum_parametrized(self, attr_data_type):
        """测试 AttrValueType 枚举参数化"""
        assert isinstance(attr_data_type, AttrValueType)
        assert isinstance(attr_data_type, int)
        assert attr_data_type.value >= 0
