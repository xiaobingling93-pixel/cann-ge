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

import tensorflow as tf
import numpy as np
## TF1.X environment
from npu_bridge.npu_init import *

if __main__ == '__main__':
    data1 = tf.placeholder(tf.float32, shape=[128])
    input_data1 = np.random.rand(128).astype(np.float32)
    data2 = tf.placeholder(tf.float32, shape=[192, 128])
    input_data2 = np.random.rand(192, 128).astype(np.float32)
    ## 构造模型结构
    abs_0 = tf.abs(data1)
    add_0 = tf.add(abs_0, data2)
    relu_0 = tf.nn.relu(add_0)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line”].b = True
    # 设置为推理模式
    custom_op.parameter_map["graph_run_mode"].i = 0

    feed_dict = {data1: input_data1, data2: input_data2}
    step = 100
    ## 执行模型
    with tf.compat.v1.Session(config=sess_config) as sess:
        for _ in range(step):
            sess.run(relu_0, feed_dict=feed_dict)`
