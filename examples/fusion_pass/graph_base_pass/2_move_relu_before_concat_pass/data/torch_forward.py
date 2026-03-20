#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch_npu
import torchair


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y, z):
        x_1 = torch.add(x, 1)
        y_1 = torch.add(y, 1)
        z_1 = torch.add(z, 1)
        out = torch.cat([x_1, y_1, z_1], dim=0)
        return torch.add(self.relu(out), 1)


if __name__ == "__main__":
    model = Model().npu()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    x = torch.randn(1, 6, 9, 9).npu()
    y = torch.randn(1, 6, 9, 9).npu()
    z = torch.randn(1, 6, 9, 9).npu()
    model = torch.compile(model, backend=npu_backend)
    res = model(x, y, z)
    print(res)
