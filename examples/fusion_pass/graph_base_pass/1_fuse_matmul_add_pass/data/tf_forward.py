# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tensorflow as tf
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import numpy as np


def generate_tf_graph():
    a = tf.compat.v1.placeholder(tf.float32, shape=[2, 3], name='a')
    b = tf.compat.v1.placeholder(tf.float32, shape=[3, 2], name='b')
    matmul = tf.linalg.matmul(a, b, name="matmul")
    c = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name='c')
    add = tf.add(matmul, c, name="add")
    return tf.compat.v1.get_default_graph()


def NetworkRun():
    graph = generate_tf_graph()
    input_a = graph.get_tensor_by_name('a:0')
    input_b = graph.get_tensor_by_name('b:0')
    input_c = graph.get_tensor_by_name('c:0')
    output_nodes = graph.get_tensor_by_name('add:0')
    a = np.array([[1.0, 2, 3], [4, 5, 6]])
    b = np.array([[1.0, 2], [3, 4], [5, 6]])
    c = np.array([[1.0, 1], [1, 1]])
    
    # 适配npu
    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()

    # 配置1：选择在昇腾AI处理器上执行推理
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

    # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
    custom_op.parameter_map["graph_run_mode"].i = 0

    # 配置4：关闭remapping和MemoryOptimizer
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    with tf.compat.v1.Session(config=config, graph=graph) as sess:
        out = sess.run(output_nodes, feed_dict={input_a: a, input_b: b, input_c: c})
        print('---out---\n', out)

if __name__ == "__main__":
    NetworkRun()
