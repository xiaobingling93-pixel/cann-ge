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


def build_concat_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakeConcatV2FedtoReluGraph")
    # 2、创建图输入节点
    tensor_holder1 = builder.create_input(
        index=0,
        name="tensor1",
        data_type=DataType.DT_FLOAT,
        shape=[8, 64, 128]
    )
    tensor_holder2 = builder.create_input(
        index=1,
        name="tensor2",
        data_type=DataType.DT_FLOAT,
        shape=[2, 64, 128]
    )
    tensor_holder3 = builder.create_input(
        index=2,
        name="tensor3",
        data_type=DataType.DT_FLOAT,
        shape=[6, 64, 128]
    )
    tensor_holder_list = [tensor_holder1, tensor_holder2, tensor_holder3]
    concat_tensor_holder = ConcatV2(tensor_holder_list, 0, N=3)
    relu_tensor_holder = Relu(concat_tensor_holder)
    # 3、设置图输出节点
    builder.set_graph_output(relu_tensor_holder, 0)
    # 4、构建图
    return builder.build_and_reset()


def save_graph_to_air(graph):
    print("save to air")
    graph.save_to_air(file_path="graph.air")


graph = build_concat_graph()
save_graph_to_air(graph)
