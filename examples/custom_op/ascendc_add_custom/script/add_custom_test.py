# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchair
import torch._dynamo
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor, TensorSpec, DataType

# 环境校验
assert torch.npu.is_available(), "NPU is not available!"

# 加载自定义算子库（兼容不同路径）
lib_path = os.path.join(os.path.dirname(__file__), "../build/libcust_opapi.so")
if not os.path.exists(lib_path):
    lib_path = "libcust_opapi.so"
torch.ops.load_library(lib_path)


@torchair.register_fx_node_ge_converter(torch.ops.ascendc_ops.ascendc_add.default)
# 实现converter
def convert_ascendc_ascendc_add(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: TensorSpec = None):
    return torchair.ge.custom_op(
        "AddCustom",
        inputs={
            "x": x,
            "y": y,
        },
        outputs=['z']
    )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.ops.ascendc_ops.ascendc_add(x, y)


class TestCustomAdd(TestCase):
    def setUp(self):
        # 固定随机种子，保证结果可复现
        torch.manual_seed(42)
        self.shape = [8, 2048]
        self.dtype = torch.float16
        # 相对误差容忍度
        self.rtol = 1e-3
        # 绝对误差容忍度
        self.atol = 1e-3

    def test_add_custom_ops(self):
        # 生成CPU输入张量
        x_cpu = torch.rand(self.shape, device='cpu', dtype=torch.float16)
        y_cpu = torch.rand(self.shape, device='cpu', dtype=torch.float16)
        # 拷贝到NPU
        x_npu = x_cpu.npu()
        y_npu = y_cpu.npu()
        # 调用自定义算子
        z_npu = torch.ops.ascendc_ops.ascendc_add(x_npu, y_npu)
        z_cpu = z_npu.cpu()
        # CPU标准加法作为基准
        z_ref = torch.add(x_cpu, y_cpu)
        # 精度校验（float16允许1e-3的误差）
        torch.testing.assert_close(z_cpu, z_ref, rtol=self.rtol, atol=self.atol,
                                   msg="Numerical values do not match within tolerance")

        model = Model().npu()
        # 配置图模式config
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        # 基于NPU backend编译模型
        opt_model = torch.compile(model, backend=npu_backend)

        x_compile = torch.rand(self.shape, device='cpu', dtype=torch.float16)
        y_compile = torch.rand(self.shape, device='cpu', dtype=torch.float16)
        z_compile_npu = opt_model(x_compile.npu(), y_compile.npu())
        z_compile_cpu = z_compile_npu.cpu()
        # 重新计算基准值（与编译测试数据匹配）
        z_compile_ref = torch.add(x_compile.cpu(), y_compile.cpu())
        # 精度校验
        torch.testing.assert_close(z_compile_cpu, z_compile_ref, rtol=self.rtol, atol=self.atol,
                                   msg="Numerical values do not match within tolerance")

if __name__ == "__main__":
    # 运行测试
    run_tests()