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

if [ -n "${PYDFLOW_SRC_PATH}" ]; then
  BASEPATH="${PYDFLOW_SRC_PATH}/../.."
  PYDFLOW_SRC_PATH="${PYDFLOW_SRC_PATH}"
else
  BASEPATH=$(cd "$(dirname $0)/../../.."; pwd)
  PYDFLOW_SRC_PATH="${BASEPATH}/dflow/pydflow"
fi

if [ -n "${PYDFLOW_TEST_PATH}" ]; then
  PYDFLOW_TEST_PATH="${PYDFLOW_TEST_PATH}"
else
  PYDFLOW_TEST_PATH="${BASEPATH}/tests/dflow/pydflow"
fi

# print usage message
usage() {
  echo "Usage:"
  echo "sh run_test.sh [-u | --ut] [-s | --st]"
  echo "               [-h | --help]"
  echo "               [--ascend_install_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -u, --ut       Build ut"
  echo "    -s, --st       Build st"
  echo "    -h, --help     Print usage"
  echo "        --ascend_install_path=<PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/ascend-toolkit/latest"
  echo ""
}

# parse and set options
checkopts() {
  ENABLE_DFLOW_UT="off"
  ENABLE_DFLOW_ST="off"

  if [ -n "$ASCEND_INSTALL_PATH" ]; then
    ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
  else
    ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
  fi

  parsed_args=$(getopt -a -o ush -l ut,st,help,ascend_install_path: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -u | --ut)
        ENABLE_DFLOW_UT="on"
        shift
        ;;
      -s | --st)
        ENABLE_DFLOW_ST="on"
        shift
        ;;
      -h | --help)
        usage
        exit 1
        ;;
      --ascend_install_path)
        ASCEND_INSTALL_PATH="$(realpath $2)"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

prepare_test_env() {
  rm -rf ${PYDFLOW_TEST_PATH}/build_stub
  rm -rf ${PYDFLOW_SRC_PATH}/python/dataflow/dflow_wrapper.so
  rm -rf ${PYDFLOW_SRC_PATH}/python/dataflow/data_wrapper.so
  rm -rf ${PYDFLOW_SRC_PATH}/python/dataflow/flow_func/flowfunc_wrapper.so

  export WORKSPACE_BASE_DIR=${BASEPATH}/dflow/pydflow
  export ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}

  if command -v ccache &> /dev/null; then
    export CC="ccache gcc"
    export CXX="ccache g++"
    echo "ccache detected and enabled"
  fi

  cd ${PYDFLOW_TEST_PATH}/stub
  python3 setup.py build_ext --build-temp=${PYDFLOW_TEST_PATH}/build_stub --build-lib=${PYDFLOW_TEST_PATH}/build_stub
  find ${PYDFLOW_TEST_PATH}/build_stub -name dflow_wrapper*.so | xargs -i cp -rf {} ${PYDFLOW_SRC_PATH}/python/dataflow/dflow_wrapper.so
  find ${PYDFLOW_TEST_PATH}/build_stub -name data_wrapper*.so | xargs -i cp -rf {} ${PYDFLOW_SRC_PATH}/python/dataflow/data_wrapper.so
  find ${PYDFLOW_TEST_PATH}/build_stub -name flow_func_wrapper*.so | xargs -i cp -rf {} ${PYDFLOW_SRC_PATH}/python/dataflow/flow_func/flowfunc_wrapper.so
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

main() {
  echo "BASEPATH=${BASEPATH}"
  echo "PYDFLOW_TEST_PATH=${PYDFLOW_TEST_PATH}"
  echo "PYDFLOW_SRC_PATH=${PYDFLOW_SRC_PATH}"
  cd "${PYDFLOW_TEST_PATH}"
  checkopts "$@"

  if [ "X$ENABLE_DFLOW_UT" != "Xon" ] && [ "X$ENABLE_DFLOW_ST" != "Xon" ]; then
    ENABLE_DFLOW_UT="on"
    ENABLE_DFLOW_ST="on"
  fi
  prepare_test_env
  # ut
  if [ "X$ENABLE_DFLOW_UT" == "Xon" ]; then
      echo "run ut start"
      if [ -n "${PYDFLOW_BUILD_PATH}" ]; then
        PYDFLOW_BUILD_PATH="${PYDFLOW_BUILD_PATH}"
      else
        PYDFLOW_BUILD_PATH="${PYDFLOW_TEST_PATH}/build_ut"
      fi
      mk_dir "${PYDFLOW_BUILD_PATH}"
      cd "${PYDFLOW_BUILD_PATH}"

      export PYTHONPATH=${BASEPATH}/dflow/pydflow/python:$PYTHONPATH
      echo "PYTHONPATH=${PYTHONPATH}"
      ASAN_OPTIONS=detect_leaks=0 COVERAGE_FILE=coverage_dflow coverage run -m unittest discover -v ${PYDFLOW_TEST_PATH}/ut
      echo "run ut end"
      unset PYDFLOW_BUILD_PATH
  fi

  # st
  if [ "X$ENABLE_DFLOW_ST" = "Xon" ]; then
      echo "run st start"
      if [ -n "${PYDFLOW_BUILD_PATH}" ]; then
        PYDFLOW_BUILD_PATH="${PYDFLOW_BUILD_PATH}"
      else
        PYDFLOW_BUILD_PATH="${PYDFLOW_TEST_PATH}/build_st"
      fi
      mk_dir "${PYDFLOW_BUILD_PATH}"
      cd "${PYDFLOW_BUILD_PATH}"

      export PYTHONPATH=${BASEPATH}/dflow/pydflow/python/:$PYTHONPATH
      echo "PYTHONPATH=${PYTHONPATH}"
      ASAN_OPTIONS=detect_leaks=0 COVERAGE_FILE=coverage_dflow coverage run -m unittest discover -v ${PYDFLOW_TEST_PATH}/st
      echo "run st end"
      unset PYDFLOW_BUILD_PATH
  fi
  rm  ${PYDFLOW_SRC_PATH}/python/dataflow/dflow_wrapper.so
  rm  ${PYDFLOW_SRC_PATH}/python/dataflow/data_wrapper.so
  rm  ${PYDFLOW_SRC_PATH}/python/dataflow/flow_func/flowfunc_wrapper.so
}

main "$@"
