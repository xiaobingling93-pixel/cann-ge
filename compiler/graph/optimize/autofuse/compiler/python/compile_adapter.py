#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import tempfile
import argparse
import subprocess
from typing import List
from autofuse import ascendc_compile
from autofuse.ascendc_compile import str2bool
import re

def camel_to_snake(camel_str):
    # 使用正则表达式匹配大写字母
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    # 使用正则表达式匹配小写字母后跟大写字母的情况
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def gen_valid_name(t_name):
    result = []
    last_was_underscore = False

    for c in t_name:
        if c.isalnum():
            result.append(c)
            last_was_underscore = False
        else:
            if not last_was_underscore:
                result.append('_')
                last_was_underscore = True

    ret_name = ''.join(result)

    # 删除开头的下划线
    if ret_name and ret_name[0] == '_':
        ret_name = ret_name[1:]

    # 如果以数字开头，添加前缀
    if ret_name and ret_name[0].isdigit():
        ret_name = "t_" + ret_name

    return ret_name


def parse_compile_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', default='autofuse', type=str, help='Graph name.')
    parser.add_argument('--output_file', required=True, type=str, help='Destination directory.')
    parser.add_argument('--output_path', default='', type=str, help='Output directory.')
    parser.add_argument('--force_unknown', default=False, type=str2bool, help='force unknown shape.')
    parser.add_argument('--config_file', default='', type=str, help='PGO tiling config file after turning.')
    parser.add_argument('--soc_version', default='Ascend910B', type=str, help='chip soc version.')
    return parser.parse_args(argv)


def generate_file(dst_dir, file_name, text):
    os.makedirs(dst_dir, exist_ok=True)
    file_path = os.path.join(dst_dir, file_name)
    with open(file_path, "w") as file:
        file.write(text)


def get_dfx_env_result():
    result = {}
    # 获取环境变量AUTOFUSE_DFX_FLAGS的值
    dfx_flags = os.getenv('AUTOFUSE_DFX_FLAGS')
    if not dfx_flags:
        return result

    params = dfx_flags.split(';')
    for param in params:
        # 检查包含等号
        if '=' in param:
            # 分割键值对 只分割一次 避免值存在等号场景
            key_part, value_part = param.split('=', 1)
            # 去除key前面的--
            key = key_part.lstrip('-')
            result[key] = value_part

    return result

def get_debug_flag():
    dfx_dict = get_dfx_env_result()
    debug_flag = dfx_dict.get('codegen_compile_debug', "false")
    flg = debug_flag.lower()
    if flg == 'true':
        return True
    else:
        return False


def get_pgo_topn():
    default_topn = 5
    dfx_dict = get_dfx_env_result()
    topn_str = dfx_dict.get('autofuse_pgo_topn', str(default_topn))
    try:
        topn = int(topn_str)
        if topn < 0:
            return default_topn
        return topn
    except ValueError:
        return default_topn


def get_pgo_env_flag():
    result = {}
    # 获取环境变量AUTOFUSE_DFX_FLAGS的值
    dfx_flags = os.getenv('AUTOFUSE_FLAGS')
    if not dfx_flags:
        return result

    params = dfx_flags.split(';')
    for param in params:
        # 检查包含等号
        if '=' in param:
            # 分割键值对 只分割一次 避免值存在等号场景
            key_part, value_part = param.split('=', 1)
            # 去除key前面的--
            key = key_part.lstrip('-')
            result[key] = value_part

    debug_flag = result.get('autofuse_enable_pgo', "false")
    flg = debug_flag.lower()
    if flg == 'true':
        return True
    else:
        return False


def compile_inner(tiling_def, host_tiling, op_kernel, temp_dir, argv: List[str]):
    print("创建临时目录路径：", temp_dir)
    args = parse_compile_args(argv)

    args.graph_name = camel_to_snake(gen_valid_name(args.graph_name))

    generate_file(os.path.join(temp_dir, "host"), "autofuse_tiling_data.h", tiling_def)
    generate_file(os.path.join(temp_dir, "host"), args.graph_name + "_tiling_func.cpp", host_tiling)
    generate_file(os.path.join(temp_dir, "device"), "autofuse_tiling_data.h", tiling_def)
    generate_file(os.path.join(temp_dir, "device"), args.graph_name + "_op_kernel.cpp", op_kernel)

    argv.extend(["--host_files", os.path.join(temp_dir, "host") + "/" + args.graph_name + "_tiling_func.cpp"])
    argv.extend(["--device_files", os.path.join(temp_dir, "device") + "/" + args.graph_name + "_op_kernel.cpp"])

    ascendc_compile.main(argv, temp_dir)


def jit_compile(tiling_def, host_tiling, op_kernel, argv: List[str]):
    args = parse_compile_args(argv)
    if args.output_path != '':
        compile_inner(tiling_def, host_tiling, op_kernel, args.output_path, argv)
        return

    # 使用上下文管理器确保临时目录在结束时被删除
    compile_debug = get_debug_flag()
    if not compile_debug:
        with tempfile.TemporaryDirectory() as temp_dir:
            compile_inner(tiling_def, host_tiling, op_kernel, temp_dir, argv)
    else:
        temp_dir = tempfile.mkdtemp()
        compile_inner(tiling_def, host_tiling, op_kernel, temp_dir, argv)


def extract_time(line):
    try:
        time_str = line.split('#')[-1].strip()
        if time_str == '1.79769e+308': # 采样失败的返回值
            return float('inf')
        return float(time_str)
    except (ValueError, IndexError):
        return float('inf')


def pgo_get_top_result(search_path, top_n=5):
    with open(search_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        return None, None, None

    origin_line = lines[-1]
    solution_set_line = lines[:-1]

    sorted_lines = sorted(solution_set_line, key=extract_time)
    if top_n == 0 or top_n > len(sorted_lines):
        top_lines = sorted_lines
    else:
        top_lines = sorted_lines[:top_n]

    return top_lines, origin_line, top_n


def pgo_write_config(config_path, tiling_data, is_last_result=False):
    # 写入配置文件
    # 只有调优结束后，才写1标记内存复用，否则写0强制每次读文件
    with open(config_path, 'w') as file:
        if is_last_result:
            file.write(f'1\n')
        else:
            file.write(f'0\n')
        file.write(f"{tiling_data}\n")
        file.flush()


def pgo_generate_config(search_path, config_path, topn=5):
    with open(search_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    target_lines = lines[-(topn + 1):]

    result = min(target_lines, key=extract_time)
    if extract_time(result) == float('inf'):
        result = lines[-1]
    pgo_write_config(config_path, result, is_last_result=True)