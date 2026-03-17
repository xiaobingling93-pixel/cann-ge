#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
from enum import IntEnum


class DataType(IntEnum):
    # The enum value of DataType must be the same with the value in c++.
    # c++ file path: metadef/inc/external/graph/types.h
    DT_FLOAT = 0  # float type
    DT_FLOAT16 = 1  # fp16 type
    DT_INT8 = 2  # int8 type
    DT_INT32 = 3  # int32 type
    DT_UINT8 = 4  # uint8 type
    # reserved
    DT_INT16 = 6  # int16 type
    DT_UINT16 = 7  # uint16 type
    DT_UINT32 = 8  # unsigned int32
    DT_INT64 = 9  # int64 type
    DT_UINT64 = 10  # unsigned int64
    DT_DOUBLE = 11  # double type
    DT_BOOL = 12  # bool type
    DT_STRING = 13  # string type
    DT_DUAL_SUB_INT8 = 14  # dual output int8 type
    DT_DUAL_SUB_UINT8 = 15  # dual output uint8 type
    DT_COMPLEX64 = 16  # complex64 type
    DT_COMPLEX128 = 17  # complex128 type
    DT_QINT8 = 18  # qint8 type
    DT_QINT16 = 19  # qint16 type
    DT_QINT32 = 20  # qint32 type
    DT_QUINT8 = 21  # quint8 type
    DT_QUINT16 = 22  # quint16 type
    DT_RESOURCE = 23  # resource type
    DT_STRING_REF = 24  # string ref type
    DT_DUAL = 25  # dual output type
    DT_VARIANT = 26  # dt_variant type
    DT_BF16 = 27  # bf16 type
    DT_UNDEFINED = 28  # Used to indicate a DataType field has not been set.
    DT_INT4 = 29  # int4 type
    DT_UINT1 = 30  # uint1 type
    DT_INT2 = 31  # int2 type
    DT_UINT2 = 32  # uint2 type
    DT_COMPLEX32 = 33  # complex32 type
    DT_HIFLOAT8 = 34  # hifloat8 type
    DT_FLOAT8_E5M2 = 35  # float8_e5m2 type
    DT_FLOAT8_E4M3FN = 36  # float8_e4m3fn type
    DT_FLOAT8_E8M0 = 37  # float8_e8m0 type
    DT_FLOAT6_E3M2 = 38  # float6_e3m2 type
    DT_FLOAT6_E2M3 = 39  # float6_e2m3 type
    DT_FLOAT4_E2M1 = 40  # float4_e2m1 type
    DT_FLOAT4_E1M2 = 41  # float4_e1m2 type
    DT_MAX = 42  # Mark the boundaries of data types


class Format(IntEnum):
    # The enum value of Format must be the same with the value in c++.
    # c++ file path: metadef/inc/external/graph/types.h

    FORMAT_NCHW = 0  # NCHW
    FORMAT_NHWC = 1  # NHWC
    FORMAT_ND = 2  # Nd Tensor
    FORMAT_NC1HWC0 = 3  # NC1HWC0
    FORMAT_FRACTAL_Z = 4  # FRACTAL_Z
    FORMAT_NC1C0HWPAD = 5
    FORMAT_NHWC1C0 = 6
    FORMAT_FSR_NCHW = 7
    FORMAT_FRACTAL_DECONV = 8
    FORMAT_C1HWNC0 = 9
    FORMAT_FRACTAL_DECONV_TRANSPOSE = 10
    FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11
    FORMAT_NC1HWC0_C04 = 12  # NC1HWC0, C0 is 4
    FORMAT_FRACTAL_Z_C04 = 13  # FRACZ, C0 is 4
    FORMAT_CHWN = 14
    FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15
    FORMAT_HWCN = 16
    FORMAT_NC1KHKWHWC0 = 17  # KH,KW kernel h& kernel w maxpooling max output format
    FORMAT_BN_WEIGHT = 18
    FORMAT_FILTER_HWCK = 19  # filter input tensor format
    FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20
    FORMAT_HASHTABLE_LOOKUP_KEYS = 21
    FORMAT_HASHTABLE_LOOKUP_VALUE = 22
    FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23
    FORMAT_HASHTABLE_LOOKUP_HITS = 24
    FORMAT_C1HWNCoC0 = 25
    FORMAT_MD = 26
    FORMAT_NDHWC = 27
    FORMAT_FRACTAL_ZZ = 28
    FORMAT_FRACTAL_NZ = 29
    FORMAT_NCDHW = 30
    FORMAT_DHWCN = 31  # 3D filter input tensor format
    FORMAT_NDC1HWC0 = 32
    FORMAT_FRACTAL_Z_3D = 33
    FORMAT_CN = 34
    FORMAT_NC = 35
    FORMAT_DHWNC = 36
    FORMAT_FRACTAL_Z_3D_TRANSPOSE = 37  # 3D filter(transpose) input tensor format
    FORMAT_FRACTAL_ZN_LSTM = 38
    FORMAT_FRACTAL_Z_G = 39
    FORMAT_RESERVED = 40
    FORMAT_ALL = 41
    FORMAT_NULL = 42
    FORMAT_ND_RNN_BIAS = 43
    FORMAT_FRACTAL_ZN_RNN = 44
    FORMAT_NYUV = 45
    FORMAT_NYUV_A = 46
    FORMAT_NCL = 47
    FORMAT_FRACTAL_Z_WINO = 48
    FORMAT_C1HWC0 = 49
    FORMAT_FRACTAL_NZ_C0_16 = 50
    FORMAT_FRACTAL_NZ_C0_32 = 51
    FORMAT_END = 52
    FORMAT_MAX = 53


class AttrValueType(IntEnum):
    # The enum value of AttrDataType must be the same with the value in c++(enum ValueType.
    # c++ file path: metadef/inc/graph/any_value.h
    VT_NONE = 0
    VT_STRING = 1
    VT_FLOAT = 2
    VT_BOOL = 3
    VT_INT = 4
    VT_TENSOR_DESC = 5
    VT_TENSOR = 6
    VT_BYTES = 7
    VT_GRAPH = 8
    VT_NAMED_ATTRS = 9
    VT_LIST_LIST_INT = 10
    VT_DATA_TYPE = 11
    VT_LIST_LIST_FLOAT = 12

    VT_LIST_BASE = 1000
    VT_LIST_STRING = VT_LIST_BASE + VT_STRING
    VT_LIST_FLOAT = VT_LIST_BASE + VT_FLOAT
    VT_LIST_BOOL = VT_LIST_BASE + VT_BOOL
    VT_LIST_INT = VT_LIST_BASE + VT_INT
    VT_LIST_TENSOR_DESC = VT_LIST_BASE + VT_TENSOR_DESC
    VT_LIST_TENSOR = VT_LIST_BASE + VT_TENSOR
    VT_LIST_BYTES = VT_LIST_BASE + VT_BYTES
    VT_LIST_GRAPH = VT_LIST_BASE + VT_GRAPH
    VT_LIST_NAMED_ATTRS = VT_LIST_BASE + VT_NAMED_ATTRS
    VT_LIST_DATA_TYPE = VT_LIST_BASE + VT_DATA_TYPE
