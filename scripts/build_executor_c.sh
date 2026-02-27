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
BASEPATH=$(cd "$(dirname $0)/.."; pwd)
echo "ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}"
echo "ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH}"
echo "BUILD_METADEF=${BUILD_METADEF}"
if [ -z "${BUILD_METADEF}" ] ; then
  BUILD_METADEF=ON
fi
if [ -z "${OUTPUT_PATH}" ] ; then
  OUTPUT_PATH="${BASEPATH}/output"
fi

BUILD_RELATIVE_PATH="build"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build_executor_c.sh [-j[n]] [-h] [-v] [-s] [-u] [-c]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Build and execute ut"
  echo "    -s Build and execute st"
  echo "    -j[n] Set the number of threads used for building GE-executor-c, default is 8"
  echo "    -c Build llt with coverage tag"
  echo "    -v Display build command"
  echo "to be continued ..."
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  ENABLE_GE_C_UT="off"
  ENABLE_GE_C_ST="off"
  ENABLE_GE_C_LLT="off"
  ENABLE_GE_COV="off"
  PLATFORM="executor_c"
  PRODUCT="normal"
  MINDSPORE_MODE="off"
  CMAKE_BUILD_TYPE="Release"
  # Process the options
  while getopts 'uschj:p:g:v' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        ENABLE_GE_C_UT="on"
        ENABLE_GE_C_LLT="on"
        ENABLE_TEST_C="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      s)
        ENABLE_GE_C_ST="on"
        ENABLE_GE_C_LLT="on"
        ENABLE_TEST_C="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      h)
        usage
        exit 0
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      v)
        VERBOSE="VERBOSE=1"
        ;;
      c)
        ENABLE_LLT_COV="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

# GE-executor-c build start
echo "---------------- GE-executor-c build start ----------------"

# create build path
build_executor_c()
{
  echo "create build directory and build GE-executor-c";
  BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  cmake -D ENABLE_OPEN_SRC=True \
        -D ENABLE_TEST_C=${ENABLE_TEST_C} \
        -D ENABLE_GE_C_LLT=${ENABLE_GE_C_LLT} \
        -D MINDSPORE_MODE=${MINDSPORE_MODE} \
        -D PLATFORM=${PLATFORM} \
        -D PRODUCT=${PRODUCT} \
        -D BUILD_METADEF=${BUILD_METADEF} \
        -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
        -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
        ..
  if [ $? -ne 0 ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi
  make ${VERBOSE} select_targets -j${THREAD_NUM}
  if [ $? -ne 0 ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  make install
  echo "GE-executor-c build success!"
}

generate_inc_coverage() {
  echo "Generating inc coverage, please wait..."
  rm -rf ${BASEPATH}/diff
  mk_dir ${BASEPATH}/cov/diff

  git diff --src-prefix=${BASEPATH}/ --dst-prefix=${BASEPATH}/ HEAD^ > ${BASEPATH}/cov/diff/inc_change_diff.txt
  addlcov --diff ${BASEPATH}/cov/coverage.info ${BASEPATH}/cov/diff/inc_change_diff.txt -o ${BASEPATH}/cov/diff/inc_coverage.info
  genhtml --prefix ${BASEPATH} -o ${BASEPATH}/cov/diff/html ${BASEPATH}/cov/diff/inc_coverage.info --legend -t CHG --no-branch-coverage --no-function-coverage
}

g++ -v
mk_dir ${OUTPUT_PATH}
build_executor_c || { echo "GE-executor-c build failed."; return; }
echo "---------------- GE-executor-c build finished ----------------"
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
OUTPUT_SO_LIST=$(find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0)
if [ -n "${OUTPUT_SO_LIST}" ]; then
    chmod 500 ${OUTPUT_SO_LIST}
fi

echo "---------------- GE-executor-c output generated ----------------"
if [[ "X$ENABLE_GE_C_LLT" = "Xon" ]]; then
    cp -rf ${BUILD_PATH}tests/test_c/ut/testcase/executor/ut_ge_executor_c_utest ${OUTPUT_PATH}
    #execute ut testcase
    echo "Begin to run tests WITH leaks check"
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_ge_executor_c_utest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi

    cp -rf ${BUILD_PATH}tests/test_c/ut/testcase/executor/ut_ge_executor_c_liteos_utest ${OUTPUT_PATH}
    #execute ut testcase
    echo "Begin to run tests WITH leaks check"
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_ge_executor_c_liteos_utest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi

    cp -rf ${BUILD_PATH}tests/test_c/ut/testcase/executor/ut_dbg_liteos_static_utest ${OUTPUT_PATH}
    #execute ut testcase
    echo "Begin to run tests WITH leaks check"
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_dbg_liteos_static_utest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi

    cp -rf ${BUILD_PATH}tests/test_c/ut/testcase/acl/ut_ascendcl_c_utest ${OUTPUT_PATH}
    #execute ut testcase
    echo "Begin to run tests WITH leaks check"
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_ascendcl_c_utest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi

    if [[ "X$ENABLE_LLT_COV" = "Xon" ]]; then
      echo "Generating coverage statistics, please wait..."
      cd ${BASEPATH}
      if [[ "X$ENABLE_GE_C_UT" = "Xon" ]]; then
        rm -rf ${BASEPATH}/cov_executor_c_ut
        mk_dir ${BASEPATH}/cov_executor_c_ut
        lcov -c -d ${BUILD_PATH}/tests/test_c/ut/testcase/executor -o cov_executor_c_ut/tmp.info
        lcov -r cov_executor_c_ut/tmp.info '*/output/*' '*/${BUILD_RELATIVE_PATH}/opensrc/*' '*/gtest_shared/*' '*/third_party/*' '*/tests/test_c/*' '/usr/local/*' '/usr/include/*' '*/metadef/*' '*/parser/*' '*/c_base/*' -o cov_executor_c_ut/coverage.info
        cd ${BASEPATH}/cov_executor_c_ut
        genhtml coverage.info
      fi

      if [[ "X$ENABLE_GE_C_ST" = "Xon" ]]; then
        rm -rf ${BASEPATH}/cov_executor_c_st
        mk_dir ${BASEPATH}/cov_executor_c_st
        lcov -c -d ${BUILD_PATH}/tests/test_c/ut/testcase/executor -o cov_executor_c_st/tmp.info
        lcov -r cov_executor_c_st/tmp.info '*/output/*' '*/${BUILD_RELATIVE_PATH}/opensrc/*' '*/gtest_shared/*' '*/third_party/*' '*/tests/test_c/*' '/usr/local/*' '/usr/include/*' '*/metadef/*' '*/parser/*' '*/c_base/*' -o cov_executor_c_st/coverage.info
        cd ${BASEPATH}/cov_executor_c_st
        genhtml coverage.info
      fi
    fi
fi

# generate output package in tar form for runtime/compiler, including ut/st libraries/executables
generate_package()
{
  cd "${BASEPATH}"

  LIB_PATH="lib/c"
  RUNTIME_PATH="runtime/lib"
  RUNTIME_LIB=(libge_executor.a)

  rm -rf ${OUTPUT_PATH:?}/${RUNTIME_PATH}/

  mk_dir "${OUTPUT_PATH}/${RUNTIME_PATH}"

  cd "${OUTPUT_PATH}"

  for lib in "${RUNTIME_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${RUNTIME_PATH} \;
  done

  tar -zcf ge_executor_lib.tar runtime
}

if [[ "X$ENABLE_GE_C_UT" = "Xoff" ]] && [[ "X$ENABLE_GE_C_ST" = "Xoff" ]]; then
  generate_package
else
  cd "${OUTPUT_PATH}"
  find ./ -name ge_executor_lib.tar -exec rm {} \;
  tar -zcf ge_executor_lib.tar lib
fi
echo "---------------- GE-executor-c package archive generated ----------------"
