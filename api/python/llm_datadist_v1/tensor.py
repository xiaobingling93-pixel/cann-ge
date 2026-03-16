#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Union, List, Tuple
import numpy as np
import ctypes

from llm_datadist_v1.utils import utils
from llm_datadist_v1.status import handle_llm_status
from llm_datadist_v1.data_type import _dwrapper_dtype_to_python_dtype
from llm_datadist_v1 import data_type

from llm_datadist_v1 import llm_wrapper

class TensorDesc(object):
    def __init__(self, dtype: data_type.DataType, shape: Union[List[int], Tuple[int]]):
        """
        初始化
        Args:
            dtype: 数据类型
            shape: 数据维度信息
        """
        utils.check_isinstance("dtype", dtype, data_type.DataType)
        utils.check_isinstance("shape", shape, [list, tuple], int)
        self._dtype = dtype
        self._shape = list(shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def __str__(self):
        return f"TensorDesc(dtype={str(self.dtype)}, shape={str(self.shape)})"


class Tensor(object):
    def __init__(self, data, tensor_desc: TensorDesc = None):
        """
        初始化
        Args:
            data: 数据
            tensor_desc: 描述信息
        """
        utils.check_isinstance("data", data, [np.ndarray, Tensor, int])
        utils.check_isinstance("tensor_desc", tensor_desc, TensorDesc)
        self._tensor_id = 0
        if isinstance(data, Tensor):
            self._tensor_desc = data._tensor_desc
            self._tensor_id = llm_wrapper.clone_tensor(data._tensor_id)
        elif isinstance(data, int):
            self._tensor_desc = tensor_desc
            self._tensor_id = data
        else:
            self._init_by_ndarray(data, tensor_desc)

    def __del__(self):
        # 保底释放kv cache, 但更推荐主动通过调用kv_cache_manager.deallocate_cache来释放kv cache，而不应该遗留到此处自动释放
        if self._tensor_id != 0:
            llm_wrapper.destroy_tensor(self._tensor_id)

    def __str__(self):
        return f"Tensor({self.numpy(True if self._is_inner_dtype_str() else False)},tensor_desc={self._tensor_desc})"

    @staticmethod
    def from_tensor_tuple(tensor_tuple: Tuple[int, int, List[int]]):
        tensor_desc = TensorDesc(_dwrapper_dtype_to_python_dtype[tensor_tuple[1]], tensor_tuple[2])
        return Tensor(tensor_tuple[0], tensor_desc)

    def _init_by_ndarray(self, data: np.ndarray, tensor_desc: TensorDesc = None):
        if tensor_desc:
            if list(data.shape) != tensor_desc.shape:
                raise RuntimeError(
                    f"The shape of data:{data.shape} is not same as tensor_desc shape:{tensor_desc.shape}")
            desc_np_dtype = data_type.dtype_to_np_dtype.get(tensor_desc.dtype)
            if data.dtype != desc_np_dtype:
                raise RuntimeError(
                    f"The dtype of data:{data.dtype} is not same as tensor_desc dtype:{tensor_desc.dtype}")
        else:
            if data.dtype not in data_type.valid_np_dtypes and not self._is_origin_dtype_str(data.dtype):
                raise RuntimeError(
                    f"The dtype of data:{data.dtype} is not valid, only support {data_type.valid_np_dtypes}")
        if tensor_desc:
            self._tensor_desc = tensor_desc
        elif self._is_origin_dtype_str(data.dtype):
            self._tensor_desc = TensorDesc(data_type.DataType.DT_STRING, list(data.shape))
        else:
            self._tensor_desc = TensorDesc(data_type.np_dtype_to_dtype[data.dtype], list(data.shape))
        if self._is_origin_dtype_str(data.dtype):
            data = self._convert_raw_str_data(data)
        if not data.flags.c_contiguous:
            raise RuntimeError("The data is not c_contiguous")
        data_ptr = data.ctypes.data_as(ctypes.c_void_p).value
        size = data.nbytes
        self._tensor_id = llm_wrapper.build_tensor(
            data_ptr,
            size,
            data_type.python_dtype_2_dwrapper_dtype.get(self._tensor_desc.dtype),
            list(self._tensor_desc.shape))

    def _is_origin_dtype_str(self, dtype):
        return np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_)

    def _is_inner_dtype_str(self):
        return self._tensor_desc is not None and self._tensor_desc.dtype == data_type.DataType.DT_STRING

    def _convert_raw_str_data(self, data):
        format_data = data.astype(np.bytes_)
        end_point = '\0'.encode('ascii', errors='ignore')
        new_data = np.char.add(format_data, end_point)
        return new_data

    def numpy(self, copy=False):
        """
        获取数据的numpy表示
        Args:
            copy: 是否复制

        Returns:
            数据的numpy表示
        """
        utils.check_isinstance("copy", copy, bool)
        if self._is_inner_dtype_str():
            if not copy:
                raise RuntimeError("String tensor only support when param copy is True.")
            return np.array(llm_wrapper.get_string_tensor(self._tensor_id)).reshape(self._tensor_desc.shape)
        ret, tensor = llm_wrapper.tensor_get_buffer(self._tensor_id)
        handle_llm_status(ret, 'Tensor.numpy', 'Failed to get tensor buffer')
        if self._tensor_desc.dtype == data_type.DataType.DT_BF16:
            np_array = np.frombuffer(tensor, dtype=np.uint16)
            return (np_array.astype(np.uint32) << 16).view(np.float32)
        elif self._tensor_desc.dtype == data_type.DataType.DT_FLOAT16:
            np_array = np.frombuffer(tensor, dtype=np.uint16)
            return np_array.view(np.float16)
        if copy:
            ret = np.array(tensor, copy=True)
        else:
            ret = np.asarray(tensor)
        return ret
