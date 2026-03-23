#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
date +"build begin: %Y-%m-%d %H:%M:%S"

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
BUILD_RELATIVE_PATH="build"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
PYTHON_PATH=python3

# MDC build para
ENABLE_BUILD_DEVICE=ON
USE_CXX11_ABI=1
CMAKE_TOOLCHAIN_FILE=""
MDC_BUILD_COMPONENT=""
CMAKE_TOOLCHAIN_PREFIX="${BASEPATH}/../cmake/toolchain"

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | --help] [-v | --verbose] [-j<N>]"
  echo "              [--ge_compiler] [--ge_executor]  [--dflow]"
  echo "              [--build_type=<TYPE> | --build-type=<TYPE>]"
  echo "              [--output_path=<PATH>] [--cann_3rd_lib_path=<PATH>]"
  echo "              [--python_path=<PATH>]"
  echo "              [--enable-sign] [--sign-script=<PATH>]"
  echo "              [--asan] [--cov]"
  echo ""
  echo "Options:"
  echo "    -h, --help        Print usage"
  echo "    -v, --verbose     Show detailed build commands during the build process"
  echo "    -j<N>             Set the number of threads used for building AIR, default is 8"
  echo "    --ge_compiler   Build ge-compiler run package with kernel bin"
  echo "    --ge_executor   Build ge-executor run package with kernel bin"
  echo "    --dflow         Build dflow-executor run package with kernel bin"
  echo "    --asan            Enable AddressSanitizer"
  echo "    --cov             Enable Coverage"
  echo "    --build_type=<TYPE>, --build-type=<TYPE>"
  echo "                      Specify build type (TYPE option: Release/Debug), Default: Release"
  echo "    --output_path=<PATH>"
  echo "                      Set output path, default ./output"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                      Set third_party package install path, default ./output/third_party"
  echo "    --python_path=<PATH>"
  echo "                      Set python path, for example:/usr/local/bin/python3.9, default python3"
  echo "    --enable-sign"
  echo "                      Enable sign device package"
  echo "    --sign-script=<PATH>"
  echo "                      Set custom sign script path to <PATH>"
  echo "    --version=<VERSION>"
  echo "                      Set sign version to <VERSION>"
  echo ""
}

# check value of build_type option
# usage: check_build_type build_type
check_build_type() {
  arg_value="$1"
  if [ "X$arg_value" != "XRelease" ] && [ "X$arg_value" != "XDebug" ]; then
    echo "Invalid value $arg_value for option --$2"
    usage
    exit 1
  fi
}

parse_cmake_extra_args() {
    echo "Parse cmake extra args."
    # para check
    local args_str="$1"
    if [[ -z "$args_str" ]]; then
        echo "The parsed parameter string is empty."
        return 0
    fi

    IFS=',' read -ra kv_pairs <<< "$args_str"

    for kv_pair in "${kv_pairs[@]}"; do
        if [[ -z "$kv_pair" ]]; then
            continue
        fi

        local key="${kv_pair%%=*}"
        local value="${kv_pair#*=}"

        case "$key" in
            "ENABLE_BUILD_DEVICE")
                ENABLE_BUILD_DEVICE="$value"
                echo "[MDC compile] Set ENABLE_BUILD_DEVICE to ${ENABLE_BUILD_DEVICE}."
                ;;
            "USE_CXX11_ABI")
                USE_CXX11_ABI="$value"
                echo "[MDC compile] Set USE_CXX11_ABI to ${USE_CXX11_ABI}."
                ;;
            "CMAKE_TOOLCHAIN_FILE")
                CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_PREFIX}/${value}"
                export LLVM_PATH="${BASEPATH}/../build/bin/os/aos_llvm_libs/aos_llvm_x86_ubuntu_20_04_adk/llvm/bin"
                echo "[MDC compile] Set CMAKE_TOOLCHAIN_FILE to ${CMAKE_TOOLCHAIN_FILE}."
                ;;
            *)
                echo "invalid parameter key: $key"
                ;;
        esac
    done

    lower_abi=$(echo "$USE_CXX11_ABI" | tr '[:upper:]' '[:lower:]')
    if [[ "$lower_abi" == "on" || "$USE_CXX11_ABI" == "1" ]]; then
        USE_CXX11_ABI=1
    elif [[ "$lower_abi" == "off" || "$USE_CXX11_ABI" == "0" ]]; then
        USE_CXX11_ABI=0
    fi
}

# parse and set options
checkopts() {
  VERBOSE=""
  THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)

  ENABLE_SIGN="off"
  CUSTOM_SIGN_SCRIPT=""
  VERSION_INFO="9.0.0"

  OUTPUT_PATH="${BASEPATH}/output"
  CANN_3RD_LIB_PATH="${BASEPATH}/output/third_party"
  BUILD_METADEF="on"
  CMAKE_BUILD_TYPE="Release"

  BUILD_COMPONENT_COMPILER="ge-compiler"
  BUILD_COMPONENT_EXECUTOR="ge-executor"
  BUILD_COMPONENT_DFLOW="dflow-executor"
  THIRD_PARTY_DL="${BASEPATH}/build_third_party.sh"
  BUILD_OUT_PATH="${BASEPATH}/build_out"
  
  # Process the options
  parsed_args=$(getopt -a -o j:hvf: -l help,verbose,ge_compiler,ge_executor,dflow,asan,cov,cann_3rd_lib_path:,extra-cmake-args:,output_path:,build_type:,build-type:,python_path:,enable-sign,sign-script:,version: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="VERBOSE=1"
        shift
        ;;
      --ge_compiler)
        ENABLE_GE_COMPILER_PKG="on"
        MDC_BUILD_COMPONENT=${BUILD_COMPONENT_COMPILER}
        shift
        ;;
      --ge_executor)
        ENABLE_GE_EXECUTOR_PKG="on"
        MDC_BUILD_COMPONENT=${BUILD_COMPONENT_EXECUTOR}
        shift
        ;;
      --dflow)
        ENABLE_DFLOW_EXECUTOR_PKG="on"
        shift
        ;;
      --asan)
        ENABLE_ASAN="on"
        shift
        ;;
      --cov)
        ENABLE_GCOV="on"
        shift
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath $2)"
        shift 2
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --build-type)
        check_build_type "$2" build-type
        CMAKE_BUILD_TYPE="$2"
        shift 2
        ;;
      --extra-cmake-args)
        parse_cmake_extra_args "$2"
        shift 2
        ;;
      --python_path)
        PYTHON_PATH="$2"
        shift 2
        ;;
      --enable-sign)
        ENABLE_SIGN="on"
        shift
        ;;
      --sign-script)
        CUSTOM_SIGN_SCRIPT="$(realpath $2)"
        shift 2
        ;;
      --version)
        VERSION_INFO=$2
        shift 2
        ;;
      -f)
        CHANGED_FILES_FILE="$2"
        if [ ! -f "$CHANGED_FILES_FILE" ]; then
          echo "Error: File $CHANGED_FILES_FILE not found"
          exit 1
        fi
        CHANGED_FILES=$(cat "$CHANGED_FILES_FILE")
        shift 2
        ;;
      --)
        shift
        if [ $# -ne 0 ]; then
          echo "ERROR: Undefined parameter detected: $*"
          usage
          exit 1
        fi
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done

  if [ -n "$ASCEND_HOME_PATH" ]; then
    ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
  else
    echo "Error: No environment variable 'ASCEND_HOME_PATH' was found, please check the cann environment configuration."
    exit 1
  fi

  # dflow-executor子包依赖ge-executor包，所以不能同时编译
  if [ "X$ENABLE_GE_COMPILER_PKG" != "Xon" ] && [ "X$ENABLE_GE_EXECUTOR_PKG" != "Xon" ] && [ "X$ENABLE_DFLOW_EXECUTOR_PKG" != "Xon" ]; then
    ENABLE_GE_COMPILER_PKG="on"
    ENABLE_GE_EXECUTOR_PKG="on"
    ENABLE_DFLOW_EXECUTOR_PKG="on"
  fi

  set +e
  python_full_path=$(which ${PYTHON_PATH})
  set -e
  if [ -z "${python_full_path}" ]; then
    echo "Error: python_path=${PYTHON_PATH} is not exist"
    exit 1
  else
    PYTHON_PATH=${python_full_path}
    echo "use python: ${PYTHON_PATH}"
  fi
}

# check if changed files only include docs/, examples/ or README.md
# usage: check_changed_files "file1 file2 file3"
check_changed_files() {
  local changed_files="$1"
  local skip_build=true

  # if no changed files provided, return false (don't skip build)
  if [ -z "$changed_files" ]; then
    return 1
  fi

  # check each changed file
  for file in $changed_files; do
    # remove leading/trailing spaces and quotes
    file=$(echo "$file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')

    # check if file is README.md (case insensitive)
    if echo "$file" | grep -qi "^README\.md$"; then
      continue
    fi

    # check if file is CONTRIBUTING.md (case insensitive)
    if echo "$file" | grep -qi "^CONTRIBUTING\.md$"; then
      continue
    fi

    # check if file is in docs/ directory
    if echo "$file" | grep -q "^docs/"; then
      continue
    fi

    # check if file is in examples/ directory
    if echo "$file" | grep -q "^examples/"; then
      continue
    fi

    # check if file is in .claude/ directory
    if echo "$file" | grep -q "^\.claude/"; then
      continue
    fi

    # check if file is in .opencode/ directory
    if echo "$file" | grep -q "^\.opencode/"; then
      continue
    fi

    # check if file is AGENTS.md (case insensitive)
    if echo "$file" | grep -qi "^AGENTS\.md$"; then
      continue
    fi

    # if any file doesn't match the above patterns, don't skip build
    skip_build=false
    break
  done

  if [ "$skip_build" = true ]; then
    echo "[INFO] Changed files only contain docs/, examples/, .claude/, .opencode/, README.md, CONTRIBUTING.md or AGENTS.md, skipping build."
    echo "[INFO] Changed files: $changed_files"
    return 0
  fi

  return 1
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

copy_pkg() {
  if [ "${ENABLE_BUILD_DEVICE}" = "ON" ]; then
    mv ${BUILD_PATH}_CPack_Packages/makeself_staging/cann-*.run ${BUILD_OUT_PATH}/
  elif [ -z "${CMAKE_TOOLCHAIN_FILE}" ]; then
    if [ -f "/etc/lsb-release" ]; then
      ubuntu_version=$(grep -E '^DISTRIB_RELEASE=' /etc/lsb-release | cut -d'=' -f2 | xargs)
      ubuntu_version="ubuntu${ubuntu_version}"
      mv ${BUILD_PATH}_CPack_Packages/makeself_staging/cann-*.run ${BUILD_OUT_PATH}/cann-${MDC_BUILD_COMPONENT}-${VERSION_INFO}-${ubuntu_version}.x86_64.run
    else
      echo "Error: operate enviroment is not ubuntu."
      exit 1
    fi
  else
    mv ${BUILD_PATH}_CPack_Packages/makeself_staging/cann-*.run ${BUILD_OUT_PATH}/cann-${MDC_BUILD_COMPONENT}-${VERSION_INFO}-aoskernel.aarch64.run
  fi
}

make_package() {
  echo "---------------- Build AIR package:  $1 ----------------"
  rm -rf ${BUILD_PATH}_CPack_Packages/makeself_staging/
  cmake -D BUILD_OPEN_PROJECT=True \
        -D ENABLE_OPEN_SRC=True \
        -D ENABLE_ASAN=${ENABLE_ASAN} \
        -D ENABLE_GCOV=${ENABLE_GCOV} \
        -D BUILD_METADEF=${BUILD_METADEF} \
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
        -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
        -D ASCEND_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} \
        -D HI_PYTHON=${PYTHON_PATH} \
        -D FORCE_REBUILD_CANN_3RD=False \
        -D BUILD_COMPONENT=$1 \
        -D CMAKE_FIND_DEBUG_MODE=OFF \
        -D ENABLE_SIGN=${ENABLE_SIGN} \
        -D CUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT} \
        -D VERSION_INFO=${VERSION_INFO} \
        -D ENABLE_BUILD_DEVICE=${ENABLE_BUILD_DEVICE} \
        -D USE_CXX11_ABI=${USE_CXX11_ABI} \
        -D LLVM_PATH=${LLVM_PATH} \
        -D CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        ..
  make ${VERBOSE} $1 -j${THREAD_NUM} && cpack
  copy_pkg
}

build_pkg() {
  echo "Create build directory and build AIR";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  if [ "X$ENABLE_GE_COMPILER_PKG" == "Xon" ]; then
    make_package "${BUILD_COMPONENT_COMPILER}" || { echo "Build Build ge-compiler run package failed."; exit 1; }
  fi
  if [ "X$ENABLE_GE_EXECUTOR_PKG" == "Xon" ]; then
    make_package "${BUILD_COMPONENT_EXECUTOR}" || { echo "Build Build ge-executor run package failed."; exit 1; }
  fi
  if [ "X$ENABLE_DFLOW_EXECUTOR_PKG" == "Xon" ]; then
    TOOLCHAIN_DIR=${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc \
    make_package "${BUILD_COMPONENT_DFLOW}" || { echo "Build Build dflow-executor run package failed."; exit 1; }
  fi

  ls -l ${BUILD_OUT_PATH}/cann-*.run && echo "AIR package success!"
}

main() {
  cd "${BASEPATH}"
  checkopts "$@"

  # check if changed files only contain docs/, examples/ or README.md
  if [ -n "$CHANGED_FILES" ]; then
    if check_changed_files "$CHANGED_FILES"; then
      exit 200
    fi
  fi

  env
  g++ -v

  # 编译三方库
  if [ "X$ENABLE_DFLOW_EXECUTOR_PKG" == "Xon" ]; then
    bash ${THIRD_PARTY_DL} ${CANN_3RD_LIB_PATH} ${THREAD_NUM} ${BUILD_COMPONENT_DFLOW} ${ENABLE_BUILD_DEVICE} ${USE_CXX11_ABI} ${CMAKE_TOOLCHAIN_FILE}
  fi
  if [ "X$ENABLE_GE_EXECUTOR_PKG" == "Xon" ]; then
    bash ${THIRD_PARTY_DL} ${CANN_3RD_LIB_PATH} ${THREAD_NUM} ${BUILD_COMPONENT_EXECUTOR} ${ENABLE_BUILD_DEVICE} ${USE_CXX11_ABI} ${CMAKE_TOOLCHAIN_FILE}
  fi
  if [ "X$ENABLE_GE_COMPILER_PKG" == "Xon" ]; then
    bash ${THIRD_PARTY_DL} ${CANN_3RD_LIB_PATH} ${THREAD_NUM} ${BUILD_COMPONENT_COMPILER} ${ENABLE_BUILD_DEVICE} ${USE_CXX11_ABI} ${CMAKE_TOOLCHAIN_FILE}
  fi

  echo "---------------- Build AIR package ----------------"
  mk_dir ${OUTPUT_PATH}
  mk_dir ${BUILD_OUT_PATH}
  mk_dir ${OUTPUT_PATH}/package
  build_pkg || { echo "AIR build failed."; exit 1; }
  echo "---------------- AIR build finished ----------------"
  date +"build end: %Y-%m-%d %H:%M:%S"
}

main "$@"
