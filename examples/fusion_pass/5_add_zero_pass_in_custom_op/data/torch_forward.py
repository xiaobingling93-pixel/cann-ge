# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# -*- coding:utf-8 -*-

from typing import Any
import torch
import torch.nn as nn
import torch_npu
import torchair
from torch.library import impl
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
from torch_npu.op_plugin.meta._meta_registrations import m


# 实现Meta推导函数
@impl(m, "npu_add_custom")
def npu_add_custom_meta(x, y):
    return torch.empty_like(x + y)


# 实现Converter
@register_fx_node_ge_converter(torch.ops.npu.npu_add_custom.default)
def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "AddCustom",
        inputs={
            "x": x,
            "y": y,
        },
        outputs=['z']
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        zero_tensor = torch.tensor(0.0, dtype=torch.float16, device=x.device)
        add = torch_npu.npu_add_custom(x, zero_tensor)
        return torch_npu.npu_add_custom(add, y)


if __name__ == "__main__":
    model = Model()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    length = [8, 2048]
    x = torch.rand(length, device='npu', dtype=torch.float16)
    y = torch.rand(length, device='npu', dtype=torch.float16)
    model = torch.compile(model, backend=npu_backend)
    res = model(x, y)
    print(res)
