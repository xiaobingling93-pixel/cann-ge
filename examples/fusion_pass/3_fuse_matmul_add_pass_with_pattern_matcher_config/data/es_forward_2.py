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
    # 1、创建图构建器
    builder = GraphBuilder("MakeMatmulAddGraph")
    # 2、创建图输入节点
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
        transpose_x1=True,  # 与pattern中定义的不同
        transpose_x2=False
    )
    add_tensor_holder = Add(matmul_tensor_holder, Const(builder, value=input3))
    # 3、设置图输出节点
    builder.set_graph_output(add_tensor_holder, 0)
    # 4、构建图
    return builder.build_and_reset()


def dump_matmul_add_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx,
                       suffix="make_matmul_add_graph"
                       )


def run_matmul_add_graph(graph) -> int:
    """
    编译并运行图

    Args:
        graph: Graph对象

    Returns:
        int: 0表示成功，非0表示失败
    """
    config = {
        "ge.exec.deviceId": "0",
        "ge.graphRunMode": "0"  # 0: 图模式, 1: 单算子模式
    }
    ge_api = GeApi()
    ret = ge_api.ge_initialize(config)
    if ret != 0:
        print(f"GE初始化失败，返回码: {ret}")
        return ret
    print("GE环境初始化成功 (Device ID: 0)")

    try:
        # 2. 创建Session
        session = Session()

        # 3. 添加图到Session
        graph_id = 1
        ret = session.add_graph(graph_id, graph)
        if ret != 0:
            print(f"添加图失败，返回码: {ret}")
            return ret
        print(f"图已添加到Session (Graph ID: {graph_id})")

        # 4. 准备输入数据
        input1_data = np.random.randn(3, 2).astype(np.float32)
        input2_data = np.random.randn(3, 2).astype(np.float32)

        # 创建Tensor对象
        tensor1 = Tensor(
            input1_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [3, 2]
        )
        tensor2 = Tensor(
            input2_data.flatten().tolist(),
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [3, 2]
        )

        inputs = [tensor1, tensor2]
        # 5. 运行图
        ret = session.run_graph(graph_id, inputs)
        if not isinstance(ret, list):
            print(f"运行图失败，返回码: {ret}")
            return ret
        print("图运行成功！")
        return 0

    except Exception as e:
        print(f"[Error] 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return -1

    finally:
        # 6. 清理GE环境
        print("[Info] 清理GE环境...")
        ge_api.ge_finalize()
        print("[Success] GE环境已清理")


graph = build_matmul_add_graph()
dump_matmul_add_graph(graph)
run_matmul_add_graph(graph)
