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

__version__ = "0.0.1"

import os
from setuptools import setup
from setuptools import find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

workspace_base_dir = os.getenv("WORKSPACE_BASE_DIR")
ascend_install_path = os.getenv("ASCEND_INSTALL_PATH")

ext_modules = [
    Pybind11Extension(
        "dflow_wrapper",
        [workspace_base_dir + "/wrapper/dflow_wrapper.cpp", "./dataflow_stub.cpp"],
        include_dirs=[
            "./",
            workspace_base_dir + "/wrapper",
            workspace_base_dir + "/../..",
            workspace_base_dir + "/../../inc/external",
            workspace_base_dir + "/../../inc/graph_metadef/external",
            workspace_base_dir + "/../../inc/parser/external",
            ascend_install_path + "/include",
            ascend_install_path + "/include/external",
            ascend_install_path + "/pkg_inc/base"
        ],
    ),
    Pybind11Extension(
        "data_wrapper",
        [workspace_base_dir + "/wrapper/data_wrapper.cpp"],
        include_dirs=[
            "./",
            workspace_base_dir + "/wrapper",
            workspace_base_dir + "/../../inc/graph_metadef/external",
            ascend_install_path + "/include"
        ],
    ),
    Pybind11Extension(
        "flow_func_wrapper",
        [
            workspace_base_dir + "/wrapper/flow_func_wrapper/flow_func_wrapper.cpp",
            "./flow_func/ascend_string.cpp",
            "./flow_func/flow_func_stub.cpp",
        ],
        include_dirs=[
            "./",
            workspace_base_dir + "/wrapper",
            workspace_base_dir + "/../../inc/graph_metadef/external",
            ascend_install_path + "/pkg_inc/base",
            ascend_install_path + "/include"
        ],
    ),
]

setup(
    name="dataflow",
    version=__version__,
    description="A test project using pybind11",
    ext_modules=ext_modules,
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    python_requires=">=3.7",
)
