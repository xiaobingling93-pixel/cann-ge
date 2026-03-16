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


def generate_cmake_base_config(args: argparse.Namespace) -> str:
    """生成CMake的基础配置部分"""
    # 包含SOC版本设置、路径配置等基础设置
    source = "cmake_minimum_required(VERSION 3.16.0)\n"
    source += f"find_package(ASC REQUIRED HINTS {ASCEND_PATH}/{machine}-linux/lib64/cmake)\n"
    source += "project(Ascend_C LANGUAGES ASC CXX)\n"
    source += "set(CMAKE_CXX_STANDARD 17)\n"
    source += "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n"

    # 获取特定环境变量的值
    home_dir = os.environ.get('ASCEND_OPP_PATH')
    home_dir = os.path.realpath(home_dir + "/..")
    if not os.path.exists(home_dir):
        print('Error: Please set environment variable ASCEND_HOME_PATH')
        return ''

    source += f"set(ASCEND_CANN_PACKAGE_PATH \"{home_dir}\" "
    source += "CACHE PATH \"ASCEND CANN package installation directory\")\n"
    source += "set(RUN_MODE \"npu\" CACHE STRING \"run mode: npu\")\n"

    source += "set(CMAKE_BUILD_TYPE \"Release\" CACHE STRING \"Build type Release/Debug (default Debug)\" FORCE)\n"
    source += "set(CMAKE_INSTALL_PREFIX \"${CMAKE_CURRENT_LIST_DIR}/out\" CACHE STRING \"path for install()\" FORCE"
    source += ")\n\n"

    source += "if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)\n"
    source += "    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)\n"
    source += "elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)\n"
    source += "    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)\n"
    source += "elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)\n"
    source += "    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)\n"
    source += "else()\n"
    source += "    message(FATAL_ERROR \"ascendc_kernel_cmake does not exist, "
    source += "please check whether the cann package is installed.\")\n"
    source += "endif()\n\n"

    return source


def generate_target_configurations(args: argparse.Namespace) -> str:
    """生成CMake的目标配置部分(库和链接设置)"""
    base_file = os.path.basename(args.output_file)
    base_host_files = os.path.basename(args.host_files)
    base_device_files = os.path.basename(args.device_files)
    soc_version = get_soc_type(args)

    print("base_file:", base_file)
    if base_file.endswith('.so'):
        base_file = base_file.rstrip('.so')
        print("base_file:", base_file)
    print("base_host_files:", base_host_files)
    print("base_device_files:", base_device_files)

    source = f"ascendc_library({base_file}_kernel STATIC\n"
    source += f"    build/device/{base_device_files}\n"
    source += ")\n\n"

    source += f"ascendc_include_directories({base_file}_kernel PRIVATE\n"
    source += "  ${CMAKE_CURRENT_SOURCE_DIR}/build/device\n"
    source += ")\n\n"

    source += "set_source_files_properties(\n"
    source += f"    build/host/{base_host_files}\n"
    source += "     PROPERTIES LANGUAGE ASC\n"
    source += ")\n\n"

    source += f"add_library({base_file} {args.lib_type}\n"
    source += f"    build/host/{base_host_files}\n"
    source += ")\n\n"

    source += f"set_target_properties({base_file} PROPERTIES\n"
    source += f"    OUTPUT_NAME {base_file}.so\n"
    source += f"    PREFIX \"\"\n"
    source += f"    SUFFIX \"\")\n\n"

    source += f"target_link_libraries({base_file} PRIVATE -Wl,--whole-archive {base_file}_kernel"
    source += f"    -Wl,--no-whole-archive c_sec ascendalog platform tiling_api)\n"
    source += f"target_include_directories({base_file} PRIVATE\n"
    source += "     ${CMAKE_CURRENT_SOURCE_DIR}/build/host\n"
    source += f"    {ASCEND_PATH}/include\n"
    source += f"    {ASCEND_PATH}/include/base\n"
    source += f"    {ASCEND_PATH}/include/experiment\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/include\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform\n"
    source += "    )\n\n"
    source += f"target_compile_options({base_file} PRIVATE\n"
    source += f"    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch={soc_version}> -O2 -fno-common -Wextra -Wfloat-equal"
    source += " -fvisibility=default -DLOG_CPP -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=\n"
    source += ")\n\n"
    source += f"target_compile_options({base_file}_kernel PRIVATE\n"
    source += "    -DHAVE_TILING\n"
    source += "    -DAUTO_FUSE_DEVICE=1\n"
    source += f"    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch={soc_version}>\n"

    if args.compile_options is not None:
        source += f"  {args.compile_options}\n"

    source += ")\n\n"

    return source


def generate_cmake_lists(args: argparse.Namespace):
    """生成完整的CMakeLists.txt内容"""
    source = generate_cmake_base_config(args)
    source += generate_target_configurations(args)
    return source


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
    parser.add_argument('--compile_options',  help='Compile options.')
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
    build_dir = os.path.join(temp_dir, 'build')
    os.chdir(build_dir)
    keep_dirs = {'host', 'device'}
    # 遍历目录中的所有条目
    for entry in os.listdir(build_dir):
        entry_path = os.path.join(build_dir, entry)
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
    kernel_file = os.path.join(temp_dir, "build", "device", base_device_files)
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
    cmake_command = ["cmake", "-S", ".", "-B", "./build", "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++"]
    make_command = ["make", "-C", "./build", '-j']
    if args.job:
        make_command += [str(args.job)]

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
            tiling_data = os.path.join(temp_dir, "build", "device", "autofuse_tiling_data.h")
            tiling_data_bak = os.path.join(temp_dir, "build", "device", "autofuse_tiling_data_bak.h")
            # 备份原tilingdata文件
            shutil.copy(tiling_data, tiling_data_bak)
            with open(tiling_data, "w") as file:
                file.write(const_tiling_data)
            # 重新编译生成so
            subprocess.run(cmake_command)
            subprocess.run(make_command)


def main(argv: List[str], temp_dir):
    """Main process."""
    # 解析编译的参数
    print(argv)
    args = parse_compile_args(argv)
    args.lib_type = str.upper(args.lib_type)

    # 生成CmakeList.txt
    source = generate_cmake_lists(args)
    if source == '' :
        print("Generate CMakeLists.txt failed")
        return False

    dst_dir = os.path.realpath(args.output_file)
    src_directory = os.getcwd()
    print("current work dir: ", src_directory)
    print("temp dir: ", temp_dir)

    # 切换到临时工作目录
    os.chdir(temp_dir)
    current_directory = os.getcwd()
    print("change work dir:", current_directory)

    with open(current_directory + "/CMakeLists.txt", "w") as cmake_file:
        cmake_file.write(source)

    cmake_command = ["cmake", "-S", ".", "-B", "./build", "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++"]

    # 运行CMake
    subprocess.run(cmake_command)

    # 定义 CMake 编译命令
    make_command = ["make", "-C", "./build", '-j']
    if args.job:
        make_command += [str(args.job)]
    print(make_command)
    # 编译生成so
    subprocess.run(make_command)
    source_dir = os.path.join(current_directory, 'build')
    so_file_name = os.path.basename(args.output_file)
    src_file = os.path.join(source_dir, so_file_name)

    # 静态shape编译流程
    static_shape_compile(args, temp_dir, src_file)

    # 拷贝so到目标目录
    dst_file = os.path.realpath(args.output_file)
    dst_dir_path = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    print("src file path: ", src_file)
    print("dst file path:", dst_file)

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
