# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# -*- coding:utf-8 -*-

import numpy as np

from ge.es import GraphBuilder
from ge.graph import Tensor, DumpFormat
from ge.graph.types import DataType, Format
from ge.ge_global import GeApi
from ge.session import Session
from ge.es.all import *


def build_matmul_add_graph():
    builder = GraphBuilder("MakeMatmulAddGraph")
    input1, input2 = builder.create_inputs(2)
    input3_data = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    input3 = Tensor(
        input3_data.flatten().tolist(),
        None,
        DataType.DT_FLOAT,
        Format.FORMAT_ND,
        [2, 2]
    )
    matmul_tensor_holder = MatMul(
        input1,
        input2,
        None,
        transpose_x1=False,
        transpose_x2=False
    )
    add_tensor_holder = Add(matmul_tensor_holder, Const(builder, value=input3))
    builder.set_graph_output(add_tensor_holder, 0)
    return builder.build_and_reset()


def save_graph_to_air(graph):
    print("save to air")
    graph.save_to_air(file_path="graph.air")


graph = build_matmul_add_graph()
save_graph_to_air(graph)
