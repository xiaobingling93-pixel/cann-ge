# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tensorflow as tf


def generate_tf_model():
    a = tf.compat.v1.placeholder(tf.float32, shape=[2, 3], name='a')
    b = tf.compat.v1.placeholder(tf.float32, shape=[3, 2], name='b')
    matmul = tf.linalg.matmul(a, b, name="matmul")
    c = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name='c')
    add = tf.add(matmul, c, name="add")
    with tf.compat.v1.Session() as sess:
        tf.io.write_graph(tf.compat.v1.get_default_graph(), '.', './matmul_add.pb', as_text=False)
        print('Create Model Successful.')

if __name__ == "__main__":
    generate_tf_model()