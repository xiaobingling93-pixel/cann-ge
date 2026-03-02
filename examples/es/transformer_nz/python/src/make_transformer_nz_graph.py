# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

from ge.es import GraphBuilder
from ge.graph import Tensor, DumpFormat
from ge.graph.types import DataType, Format
from ge.ge_global import GeApi
from ge.session import Session
from ge.es.all import *


def create_nz_tensor(data: np.ndarray, dtype: DataType, shape: list) -> Tensor:
    """
    创建NZ格式的张量
    Args:
        data: 原始数据
        dtype: 数据类型
        shape: 张量形状
    Returns:
        Tensor: NZ格式的张量
    """
    tensor = Tensor(
        data.flatten().tolist(),
        None,
        dtype,
        Format.FORMAT_FRACTAL_NZ,  # 核心修改：将tensor设置为NZ格式
        shape
    )
    return tensor


def build_transformer_nz_graph():

    # 1、创建图构建器
    builder = GraphBuilder("MakeTransformerSubGraph")

    # 2、创建图输入节点
    input1, input2, input3 = builder.create_inputs(3)
    input1.set_format(Format.FORMAT_FRACTAL_NZ)
    input2.set_format(Format.FORMAT_FRACTAL_NZ)
    input3.set_format(Format.FORMAT_FRACTAL_NZ)

    matmul_result1 = MatMul(
        Cast(input1, dst_type=DataType.DT_FLOAT),
        Transpose(Cast(input2, dst_type=DataType.DT_FLOAT), builder.create_vector_int64([1, 0])),
        transpose_x1=False,
        transpose_x2=False
    )
    sigmoid_result1 = Sigmoid(matmul_result1)
    add_result1 = Reshape(sigmoid_result1, [-1, 256]) + Cast(input3, dst_type=DataType.DT_FLOAT)

    topkv2_result1 = TopKV2(add_result1, 2)
    reducesum_result1 = ReduceSum(topkv2_result1.values, [-1])
    topkv2_result2 = TopKV2(reducesum_result1, 4, sorted=False)

    cast_result2 = Cast(topkv2_result2.indices, dst_type=DataType.DT_INT64)
    scatterelements_result1 = ScatterElements(
        ZerosLike(reducesum_result1),
        cast_result2,
        Fill(Shape(cast_result2), Cast(builder.create_scalar_float(1.0), dst_type=DataType.DT_FLOAT))
    )
    identity_result1 = Identity(BroadcastTo(Unsqueeze(scatterelements_result1, axes=[-1]), [256, 256]))
    maskedfill_result1 = MaskedFill(
        add_result1,
        LogicalNot(
            Cast(Reshape(identity_result1, builder.create_vector_int64([256, 256])),
                 dst_type=DataType.DT_BOOL)),
        builder.create_scalar_float(0.0)
    )
    cast_result3 = Cast(
        TopKV2(maskedfill_result1, 4, sorted=False).indices,
        dst_type=DataType.DT_INT64
    )
    # 3、设置图输出节点
    gatherelements_result1 = GatherElements(sigmoid_result1, cast_result3, dim=1)
    realdiv_result1 = RealDiv(gatherelements_result1, 1e-20)
    output = Cast(realdiv_result1 * builder.create_scalar_float(2.5), dst_type=DataType.DT_FLOAT)
    # 4、构建图
    return builder.build_and_reset([cast_result3, output])


def build_transformer_nz_graph_and_dump(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_transformer_nz_graph")


def run_graph(graph) -> int:
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
        input1_data = np.random.randn(256, 7168).astype(np.float32)
        input2_data = np.random.randn(256, 7168).astype(np.float32)
        input3_data = np.random.randn(1, 256).astype(np.float32)

        # 创建Tensor对象
        tensor1 = create_nz_tensor(input1_data, DataType.DT_FLOAT, [256, 7168])
        tensor2 = create_nz_tensor(input2_data, DataType.DT_FLOAT, [256, 7168])
        tensor3 = create_nz_tensor(input3_data, DataType.DT_FLOAT, [1, 256])

        inputs = [tensor1, tensor2, tensor3]
        # 5. 运行图
        ret = session.run_graph(graph_id, inputs)
        print("[Info] 图运行成功！")
        for idx, tensor in enumerate(ret, start=1):
            print(f"Tensor{idx}详情：", {tensor})
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


graph = build_transformer_nz_graph()
build_transformer_nz_graph_and_dump(graph)
run_graph(graph)
