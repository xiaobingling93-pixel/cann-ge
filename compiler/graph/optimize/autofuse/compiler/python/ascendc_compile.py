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

import os
import sys
import shutil
import tempfile
import argparse
import subprocess
import platform
from typing import List
from asc_op_compile_base.common.platform.platform_info import get_soc_spec
PYF_PATH = os.path.dirname(os.path.realpath(__file__))
ASCEND_PATH = os.path.join(PYF_PATH, "..", "..", "..")
machine = platform.machine()
if not os.path.exists(ASCEND_PATH):
    ASCEND_PATH = os.getenv("ASCEND_HOME_PATH", ASCEND_PATH)

class ExtractError(Exception):
    """Extract host stub base exception."""

class ArgumentError(ExtractError):
    """Argument error."""


def get_soc_type(args):
    """根据 soc_version 返回对应的类型"""
    if args.soc_version.startswith("Ascend910B"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend910_93"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend950"):
        return "dav-3510"
    else:
        raise ValueError(f"Unsupported soc_version: {args.soc_version}")


def generate_host_compile_cmd(args: argparse.Namespace, temp_dir, base_host_file, soc_version):
    compile_command = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-D", "kernel_EXPORTS",
        "-I", f"{temp_dir}/host",
        "-I", f"{ASCEND_PATH}/include",
        "-I", f"{ASCEND_PATH}/include/base",
        "-I", f"{ASCEND_PATH}/include/experiment",
        "-I", f"{ASCEND_PATH}/{machine}-linux/include",
        "-I", f"{ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform",
        "-fPIC", f"--npu-arch={soc_version}", "-O2", "-fno-common", "-Wextra", "-Wfloat-equal", "-fvisibility=default",
        f"{args.compile_options}",
        "-D", "LOG_CPP", "-o",
        f"{temp_dir}/host/{base_host_file}.o", "-c", "-x", "asc",
        f"{temp_dir}/host/{base_host_file}"]
    return compile_command


def generate_device_compile_cmd(args: argparse.Namespace, temp_dir, base_kernel_file, soc_version):
    compile_command = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-I", f"{temp_dir}/device",
        "-fPIC", "-D", "HAVE_TILING", "-D", "AUTO_FUSE_DEVICE=1", f"--npu-arch={soc_version}",
        "-o", f"{temp_dir}/device/{base_kernel_file}.o",
        "-c", "-x", "asc", f"{temp_dir}/device/{base_kernel_file}"]
    return compile_command


def generate_target_link_cmd(args: argparse.Namespace, temp_dir, base_kernel_file, base_host_file, target_file):
    link_command = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        f"{temp_dir}/host/{base_host_file}.o",
        f"{temp_dir}/device/{base_kernel_file}.o",
        "-fPIC", "--shared",
        "-o", f"{target_file}"]
    return link_command


def compile_autofuse_target(args: argparse.Namespace, temp_dir):
    target_file = os.path.basename(args.output_file)
    base_host_file = os.path.basename(args.host_files)
    base_device_file = os.path.basename(args.device_files)
    soc_version = get_soc_type(args)
    host_compile_cmd = generate_host_compile_cmd(args, temp_dir, base_host_file, soc_version)
    device_compile_cmd = generate_device_compile_cmd(args, temp_dir, base_device_file, soc_version)
    link_cmd = generate_target_link_cmd(args, temp_dir, base_device_file, base_host_file, target_file)
    subprocess.run(host_compile_cmd)
    subprocess.run(device_compile_cmd)
    subprocess.run(link_cmd)


def str2bool(v):
    v_lower = v.lower()
    if v_lower in ['true', '1', 'yes', 'y']:
        return True
    elif v_lower in ['false', '0', 'no', 'n']:
        return False
    else:
        raise ValueError("Invalid boolean value: '{}'".format(v))


def parse_compile_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib_type', default='SHARED', type=str, help='Generate lib type.')
    parser.add_argument('--host_files', type=str, required=True, help='Host file name.')
    parser.add_argument('--device_files', type=str, required=True, help='Device file name.')
    parser.add_argument('--compile_options', default='', type=str, help='Compile options.')
    parser.add_argument('--job', type=str, help='Jobs num.')
    parser.add_argument('--output_file', required=True, type=str, help='Destination directory.')
    parser.add_argument('--graph_name', default='autofuse', type=str, help='Graph name.')
    parser.add_argument('--output_path', default='', type=str, help='Output directory.')
    parser.add_argument('--force_unknown', default=False, type=str2bool, help='force unknown shape.')
    parser.add_argument('--config_file', default='', type=str, help='PGO tiling config file after turning.')
    parser.add_argument('--soc_version', default='Ascend910B', type=str, help='chip soc version.')

    return parser.parse_args(argv)


def ascendc_clean(temp_dir):
    src_directory = os.getcwd()
    keep_dirs = {'host', 'device'}
    # 遍历目录中的所有条目
    for entry in os.listdir(temp_dir):
        entry_path = os.path.join(temp_dir, entry)
        # 检查是否是文件或目录
        if os.path.isfile(entry_path):
            # 如果是文件，直接删除
            os.remove(entry_path)
            print(f"delete file: {entry_path}")
        else:
            # 如果是目录，检查是否需要保留
            if entry not in keep_dirs:
                shutil.rmtree(entry_path)
                print(f"delete dir: {entry_path}")
    os.chdir(src_directory)


def static_shape_kernel_proc(args: argparse.Namespace, temp_dir):
    import re
    base_device_files = os.path.basename(args.device_files)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)
    # 定义正则表达式模式
    pattern = re.compile(r'^extern\s+"C"\s+__global__\s+__aicore__\s+void\s+(\w+)\s*\(([^)]*)\)\s*{')

    with open(kernel_file, 'r') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        match = pattern.match(line)
        if match:
            func_name = match.group(1)
            params_str = match.group(2).strip()
            if not params_str:
                # 无参数的情况，直接添加原行
                result.append(line)
                continue

            params = [p.strip() for p in params_str.split(',')]
            if params and params[-1] == 'AutofuseTilingData t':
                # 修改最后一个参数
                params[-1] = 'AutofuseTilingData param'
                new_params = ', '.join(params)
                new_line = f'extern "C" __global__ __aicore__ void {func_name}({new_params}) '
                new_line += '{\n'
                result.append(new_line)
                # 插入新声明
                result.append('  const AutofuseTilingData t;\n')
            else:
                # 不处理，直接添加原行
                result.append(line)
        else:
            result.append(line)

        # 写入处理后的内容到新文件
        with open(kernel_file, 'w') as f:
            f.writelines(result)


def static_shape_compile(args: argparse.Namespace, temp_dir, so_path):
    if not args.force_unknown:
        import ctypes
        lib = ctypes.CDLL(so_path)
        lib.AutofuseIsStaticShape.argtypes = []
        lib.AutofuseIsStaticShape.restype = ctypes.c_bool
        is_static = lib.AutofuseIsStaticShape()
        if bool(is_static):
            print("static shape recompile")
            ascendc_clean(temp_dir)
            static_shape_kernel_proc(args, temp_dir)
            lib.GenConstTilingData.argtypes = [ctypes.c_char_p]
            lib.GenConstTilingData.restype = ctypes.c_char_p
            # 传入PGO Tiling调优结果文件
            config_file = ctypes.c_char_p(args.config_file.encode('utf-8'))
            result = lib.GenConstTilingData(config_file)
            const_tiling_data = result.decode('utf-8')
            tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
            tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")
            # 备份原tilingdata文件
            shutil.copy(tiling_data, tiling_data_bak)
            with open(tiling_data, "w") as file:
                file.write(const_tiling_data)
            # 重新编译生成so
            compile_autofuse_target(args, temp_dir)


def main(argv: List[str], temp_dir):
    """Main process."""
    # 解析编译的参数
    print(argv)
    args = parse_compile_args(argv)
    args.lib_type = str.upper(args.lib_type)

    src_directory = os.getcwd()
    print("current work dir: ", src_directory)
    print("temp dir: ", temp_dir)

    # 切换到临时工作目录
    os.chdir(temp_dir)
    current_directory = os.getcwd()
    print("change work dir:", current_directory)

    compile_autofuse_target(args, temp_dir)

    so_file_name = os.path.basename(args.output_file)
    src_file = os.path.join(current_directory, so_file_name)

    # 静态shape编译流程
    static_shape_compile(args, temp_dir, src_file)

    # 拷贝so到目标目录
    dst_file = os.path.realpath(args.output_file)
    dst_dir_path = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    print("src file path: ", src_file)
    print("dst file path: ", dst_file)

    shutil.copy(src_file, dst_file)
    print(f'copy file {so_file_name} to {dst_dir_path}')
    os.chdir(src_directory)

    return True

def main_with_except(argv: List[str]):
    """Main process with except exceptions."""
    try:
        print("Enter main func")
        return main(argv)
    except ArgumentError as ex:
        print(f'error: check arguments error, {ex}')
        return False

if __name__ == "__main__":
    if not main_with_except(sys.argv[1:]):
        sys.exit(1)
