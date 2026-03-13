# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import logging
import tensorflow as tf
import numpy as np
tf.enable_resource_variables()
logging.basicConfig(level=logging.INFO)

# np.allclose比较函数的相对公差参数
ATOL = 0.001
# np.allclose比较函数的绝对公差参数
RTOL = 0.001


def main(unused_argv):
    custom_op_lib = tf.load_op_library(os.path.join("./outputs/libcustom_ops.so")) # 加载自定义算子库
    # 定义输入数据
    shape_params = (8, 2048)
    dtype_params = np.float32

    x_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)
    y_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)

    x = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    y = tf.compat.v1.placeholder(dtype_params, shape=shape_params)

    tf_z = tf.math.add(x, y)
    ac_z = custom_op_lib.add_custom(x, y)    # 调用Ascend C AddCustom自定义算子

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    # custom_op.parameter_map["compile_dynamic_mode"].b = True  # 动态或静态shape配置（默认走静态，添加配置走动态）

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf_golden = sess.run(tf_z, feed_dict={x: x_data, y: y_data})

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ac_golden = sess.run(ac_z, feed_dict={x: x_data, y: y_data})

    # 通过np.allclose函数比较TensorFlow和Ascend C的输出是否一致
    np.array(tf_golden).astype(dtype_params)
    np.array(ac_golden).astype(dtype_params)

    cmp_result = np.allclose(tf_golden, ac_golden, atol=ATOL, rtol=RTOL)
    if cmp_result:
        logging.info("The result of tf and ac is the same.")
    else:
        logging.info("The result of tf and ac is different.")


if __name__ == '__main__':
    tf.app.run()