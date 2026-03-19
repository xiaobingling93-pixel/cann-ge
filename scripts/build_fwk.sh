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
USE_ASAN=$(gcc -print-file-name=libasan.so)

source ${BASEPATH}/scripts/support_multiple_versions_of_lcov.sh

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build_fwk.sh [-a] [-j[n]] [-h] [-v] [-s] [-b] [-t] [-u] [-c] [-M]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build and execute st"
  echo "    -j[n] Set the number of threads used for building GraphEngine, default is 8"
  echo "    -t Build and execute ut"
  echo "    -c Build ut with coverage tag"
  echo "    -b Build Benchmark test"
  echo "    -p Build inference or train"
  echo "    -v Display build command"
  echo "    -M build MindSpore mode"
  echo "    -d Build ut with dump graph"
  echo "to be continued ..."
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  ENABLE_GE_UT="off"
  ENABLE_RT2_UT="off"
  ENABLE_GE_DT="off"
  ENABLE_GE_ST="off"
  ENABLE_GE_COV="off" # Full coverage
  ENABLE_ICOV="off" # Incremental coverage
  ENABLE_RT2_ST="off"
  ENABLE_RT3_ST="off"
  ENABLE_PYTHON_ST="off"
  ENABLE_PYTHON_UT="off"
  ENABLE_PARSER_ST="off"
  ENABLE_PARSER_UT="off"
  ENABLE_DFLOW_ST="off"
  ENABLE_DFLOW_UT="off"
  ENABLE_GE_BENCHMARK="off"
  PLATFORM=""
  PRODUCT="normal"
  MINDSPORE_MODE="off"
  CMAKE_BUILD_TYPE="Release"
  ENABLE_ASAN="false"
  ENABLE_GCOV="false"

  # Process the options
  while getopts 'CDstTcbhPlmnovj:p:g:MROudKL' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        ENABLE_GE_UT="on"
        ENABLE_RT2_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      s)
        ENABLE_GE_ST="on"
        ENABLE_RT2_ST="on"
        ENABLE_RT3_ST="on"
        ENABLE_PYTHON_ST="on"
        ENABLE_PARSER_ST="on"
        ENABLE_DFLOW_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      R)
        ENABLE_RT2_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      K)
        ENABLE_RT3_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      P)
        ENABLE_PYTHON_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      l)
        ENABLE_PYTHON_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      m)
        ENABLE_PARSER_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      n)
        ENABLE_PARSER_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      o)
        ENABLE_DFLOW_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      D)
        ENABLE_DFLOW_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      L)
        ENABLE_RT2_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;      
      O)
        ENABLE_GE_ST="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_st
        ;;
      t)
        ENABLE_GE_UT="on"
        ENABLE_RT2_UT="on"
        ENABLE_PYTHON_UT="on"
        ENABLE_PARSER_UT="on"
        ENABLE_DFLOW_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      T)
        ENABLE_GE_UT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        BUILD_RELATIVE_PATH=build_ut
        ;;
      c)
        echo "Full coverage"
        ENABLE_GE_COV="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      C)
        echo "Incremental coverage"
        ENABLE_ICOV="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      b)
        ENABLE_GE_BENCHMARK="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="GCOV"
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

      p)
        PLATFORM=$OPTARG
        ;;
      g)
        PRODUCT=$OPTARG
        ;;
      M)
        MINDSPORE_MODE="on"
        ENABLE_D="ON"
        ;;
      d)
        ENABLE_ICOV="off"
        ENABLE_GE_DT="on"
        ENABLE_TEST="True"
        CMAKE_BUILD_TYPE="DT"
        BUILD_RELATIVE_PATH=build_ut
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

# run pyge python tests (shared by UT/ST)
run_pyge_pytests() {
  echo "----------pyge tests start----------"
  PYGE_INSTALL_PATH=${BUILD_PATH}/tests/ge/ut/ge/graph/pyge_tests/ge_py_install
  PYGE_SRC_PATH=${BASEPATH}/api/python/ge
  ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=${PYGE_INSTALL_PATH}/ge/_capi/:${ORIGINAL_LD_LIBRARY_PATH}
  # 优先使用源码路径，因为使用安装路径存在覆盖率统计不到问题
  export PYTHONPATH=${PYGE_SRC_PATH}:${PYGE_INSTALL_PATH}:$PYTHON_ORIGINAL_PATH
  ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 coverage run --data-file=coverage_pyge --source=${PYGE_SRC_PATH}/ge -m pytest ${BASEPATH}/tests/ge/ut/ge/graph/pyge_tests/*_test.py -vv -s
  export LD_LIBRARY_PATH=${ORIGINAL_LD_LIBRARY_PATH}
  export PYTHONPATH=$PYTHON_ORIGINAL_PATH
}

# GraphEngine build start
echo "---------------- GraphEngine build start ----------------"

# create build path
build_graphengine()
{
  echo "create build directory and build GraphEngine";
  BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  if [[ "X$CMAKE_BUILD_TYPE" = "XGCOV" ]]; then
    ENABLE_GCOV="true"
    if [[ "X$ENABLE_PARSER_UT" != "Xon" ]] && [[ "X$ENABLE_PARSER_ST" != "Xon" ]]; then
      ENABLE_ASAN="true"
    fi
  fi
  cmake -D ENABLE_OPEN_SRC=True \
        -D ENABLE_TEST=${ENABLE_TEST} \
        -D ENABLE_GE_BENCHMARK=$ENABLE_GE_BENCHMARK \
        -D ENABLE_GE_ST=${ENABLE_GE_ST} \
        -D ENABLE_RT2_ST=${ENABLE_RT2_ST} \
        -D ENABLE_RT3_ST=${ENABLE_RT3_ST} \
        -D ENABLE_PYTHON_ST=${ENABLE_PYTHON_ST} \
        -D ENABLE_GE_UT=${ENABLE_GE_UT} \
        -D ENABLE_RT2_UT=${ENABLE_RT2_UT} \
        -D ENABLE_PYTHON_UT=${ENABLE_PYTHON_UT} \
        -D ENABLE_PARSER_UT=${ENABLE_PARSER_UT} \
        -D ENABLE_PARSER_ST=${ENABLE_PARSER_ST} \
        -D ENABLE_DFLOW_UT=${ENABLE_DFLOW_UT} \
        -D ENABLE_DFLOW_ST=${ENABLE_DFLOW_ST} \
        -D ENABLE_GE_DT=${ENABLE_GE_DT} \
        -D ENABLE_D=${ENABLE_D} \
        -D MINDSPORE_MODE=${MINDSPORE_MODE} \
        -D PLATFORM=${PLATFORM} \
        -D PRODUCT=${PRODUCT} \
        -D BUILD_METADEF=${BUILD_METADEF} \
        -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
        -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
        -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -D ENABLE_PKG=${ENABLE_PKG} \
        -D ENABLE_ASAN=${ENABLE_ASAN} \
        -D ENABLE_GCOV=${ENABLE_GCOV} \
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
  echo "GraphEngine build success!"
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
report_dir="${OUTPUT_PATH}/report"
mk_dir ${report_dir}/ut
mk_dir ${report_dir}/st
mk_dir ${report_dir}/benchmark

build_graphengine || { echo "GraphEngine build failed."; return; }
echo "---------------- GraphEngine build finished ----------------"
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 -r chmod 500

echo "---------------- GraphEngine output generated ----------------"
if [ ! -z "$ONLY_BUILD" ];then
  echo "Note: Only the build mode has been enabled, so no tests have been run."
  exit 0
fi
export LD_LIBRARY_PATH=${BUILD_PATH}/tests/depends/aoe/:$LD_LIBRARY_PATH
if [[ "X$ENABLE_GE_UT" = "Xon" ]] || [[ "X$ENABLE_RT2_UT" = "Xon" ]] || [[ "X$ENABLE_PYTHON_UT" = "Xon" ]] || [[ "X$ENABLE_PARSER_UT" = "Xon" ]] || [[ "X$ENABLE_DFLOW_UT" = "Xon" ]]; then
    COV_DIRS=()   
    COV_DIRS+=("${BUILD_PATH}/api")
    COV_DIRS+=("${BUILD_PATH}/base")
    #execute ut testcase with mem leaks by default
    if [[ "X$ENABLE_GE_UT" = "Xon" ]]; then
      echo "[TEST GE COMMON] Begin to run tests with leaks check"
      export LD_PRELOAD=${USE_ASAN}
      ASAN_OPTIONS=detect_container_overflow=0 \
      ctest --output-on-failure -j ${THREAD_NUM} -L ut -L ge_common --test-dir ${BUILD_PATH} --no-tests=error \
            -O ${BUILD_PATH}/ctest_ut_ge_common.log
      unset LD_PRELOAD
      unset ASAN_OPTIONS
      COV_DIRS+=("${BUILD_PATH}/graph_metadef")
      COV_DIRS+=("${BUILD_PATH}/compiler")
    fi
    if [[ "X$ENABLE_RT2_UT" = "Xon" ]]; then
      echo "[TEST GE RT] Begin to run tests with leaks check"
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_container_overflow=0
      ctest --output-on-failure -j ${THREAD_NUM} -L ut -L ge_rt --test-dir ${BUILD_PATH} --no-tests=error \
            -O ${BUILD_PATH}/ctest_ut_rt.log
      unset ASAN_OPTIONS
      unset LD_PRELOAD
      COV_DIRS+=("${BUILD_PATH}/runtime/v1")
      COV_DIRS+=("${BUILD_PATH}/runtime/v2")
    fi
    if [[ "X$ENABLE_PYTHON_UT" = "Xon" ]]; then
      unset LD_PRELOAD
      cp ${BUILD_PATH}/tests/depends/python/llm_wrapper.so ${BASEPATH}/api/python/llm_datadist_v1
      cp -r ${BASEPATH}/tests/python_tests ./
      PYTHON_ORIGINAL_PATH=$PYTHONPATH
      export PYTHONPATH=$PYTHON_ORIGINAL_PATH:${BASEPATH}/api/python/llm_datadist/:${BASEPATH}/api/python/
      export LD_PRELOAD=${USE_ASAN}
      echo "----------v1 ut start----------"

      ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 coverage run --data-file=coverage_python -m unittest discover python_tests/v1/ut
      run_pyge_pytests
      export PYTHONPATH=$PYTHON_ORIGINAL_PATH:${BASEPATH}/api/python/llm_datadist/:${BASEPATH}/api/python/
      unset ASAN_OPTIONS
      unset LD_PRELOAD
    fi

    if [[ "X$ENABLE_PARSER_UT" = "Xon" ]]; then
      echo "---------------- Parser UT Run Start ----------------"
      cp ${BUILD_PATH}/tests/parser/ut/parser/ut_parser ${OUTPUT_PATH}
      cp -rf ${BUILD_PATH}/tests/graph_metadef/ut/graph/ut_graph ${OUTPUT_PATH}
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0
      RUN_TEST_CASE="${OUTPUT_PATH}/ut_parser --gtest_output=xml:${report_dir}/ut/ut_parser.xml" && ${RUN_TEST_CASE} &&
      RUN_TEST_CASE="${OUTPUT_PATH}/ut_graph --gtest_output=xml:${report_dir}/ut/ut_graph.xml" && ${RUN_TEST_CASE}
      if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1
      fi
      unset ASAN_OPTIONS
      unset LD_PRELOAD
      COV_DIRS+=("${BUILD_PATH}/graph_metadef")
    fi

    if [[ "X$ENABLE_DFLOW_UT" = "Xon" ]]; then
      echo "---------------- Dflow Python UT Run Start ----------------"
      export PYDFLOW_SRC_PATH=${BASEPATH}/dflow/pydflow
      export PYDFLOW_TEST_PATH=${BUILD_PATH}/tests/dflow/pydflow
      export PYDFLOW_BUILD_PATH=${BUILD_PATH}
      rm -fr ${PYDFLOW_TEST_PATH}
      cp -r ${BASEPATH}/tests/dflow/pydflow ${PYDFLOW_TEST_PATH}
      bash tests/dflow/pydflow/run_test.sh -u --ascend_install_path=${ASCEND_INSTALL_PATH}
      unset PYDFLOW_SRC_PATH
      unset PYDFLOW_TEST_PATH
      unset PYDFLOW_BUILD_PATH

      echo "---------------- Dflow Udf UT Run Start ----------------"
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_container_overflow=0:detect_odr_violation=0
      ctest --output-on-failure -j ${THREAD_NUM} -L ut -L ut_dflow --test-dir ${BUILD_PATH} --no-tests=error \
                    -O ${BUILD_PATH}/ctest_ut_dflow.log
      if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1
      fi
      unset LD_PRELOAD
      unset ASAN_OPTIONS
      COV_DIRS+=("${BUILD_PATH}/dflow")
    fi

    if [[ "X$ENABLE_GE_COV" = "Xon" ]]; then
      echo "Generating coverage statistics, please wait..."
      cd ${BASEPATH}
      rm -rf ${BASEPATH}/cov
      mk_dir ${BASEPATH}/cov
      if [[ "X$ENABLE_PYTHON_UT" = "Xon" ]] || [[ "X$ENABLE_DFLOW_UT" = "Xon" ]]; then
        echo "generate python coverage" ${BUILD_PATH}/coverage_*
        coverage combine ${BUILD_PATH}/coverage_*
        mv .coverage ${BASEPATH}/cov/
      fi

      # 去重
      IFS=$'\n' COV_DIRS_UNIQUE=($(sort -u <<<"${COV_DIRS[*]}"))
      unset IFS

      # 转换为 lcov 参数格式
      COV_DIRS_PARAMS=()
      for dir in "${COV_DIRS_UNIQUE[@]}"; do
        COV_DIRS_PARAMS+=(-d "$dir")
      done

      lcov -c "${COV_DIRS_PARAMS[@]}" -o cov/tmp.info $(add_lcov_ops_by_major_version 2 "--ignore-errors empty,mismatch,negative")
      if [ ! -s "cov/tmp.info" ] || ! grep -q "SF:" "cov/tmp.info"; then
        echo "No valid cpp coverage data found; skip filtering."
        touch cov/coverage.info  # 生成空文件占位，避免后续流程报错
      else
        lcov -r cov/tmp.info '*/output/*' "*/${BUILD_RELATIVE_PATH}/opensrc/*" "*/${BUILD_RELATIVE_PATH}/proto/*" \
                             '*/op_impl/*' "*/${BUILD_RELATIVE_PATH}/grpc_*" '*/third_party/*' '*/op_impl/*' '*/tests/*' \
                             '/usr/local/*' '/usr/include/*' '*/metadef/*' \
                             "${ASCEND_INSTALL_PATH}/*" "${ASCEND_3RD_LIB_PATH}/*" \
                             -o cov/coverage.info $(add_lcov_ops_by_major_version 2 "--ignore-errors unused")
        genhtml cov/coverage.info -o cov/html

        if [[ "X$ENABLE_ICOV" = "Xon" ]]; then
          generate_inc_coverage
        fi
      fi
    fi
fi


echo "---------------- Parser llt finished ----------------"

if [[ "X$ENABLE_GE_DT" = "Xon" ]] || [[ "X$ENABLE_GE_UT" = "Xon" ]]; then
    cp -rf ${BUILD_PATH}/tests/ge/ut/ge/ge_manual_test ${OUTPUT_PATH}
    if [[ ! -d ${OUTPUT_PATH}/plugin/nnengine/ge_config ]]; then
      mk_dir ${OUTPUT_PATH}/plugin/nnengine/ge_config
    fi
    cp -f ${BASEPATH}/compiler/engines/manager/engine_manager/engine_conf.json ${OUTPUT_PATH}/plugin/nnengine/ge_config

    #execute ut testcase
    export ASAN_OPTIONS=detect_leaks=0
    export LD_PRELOAD=${USE_ASAN}
    echo "Begin to run tests WITHOUT leaks check"
    RUN_TEST_CASE="${OUTPUT_PATH}/ge_manual_test --gtest_output=xml:${report_dir}/ut/ge_manual_test.xml" && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
fi

if [[ "X$ENABLE_GE_BENCHMARK" = "Xon" ]]; then
    RUN_TEST_CASE="${BUILD_PATH}/tests/benchmark/ge_runtime_benchmark --gtest_output=xml:${report_dir}/benchmark/ge_runtime_benchmark.xml" && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! runtime benchmark failed  please check!!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
fi
if [[ "X$ENABLE_GE_ST" = "Xon" ]] || [[ "X$ENABLE_RT2_ST" = "Xon" ]] || [[ "X$ENABLE_RT3_ST" = "Xon" ]] || [[ "X$ENABLE_PYTHON_ST" = "Xon" ]] || [[ "X$ENABLE_PARSER_ST" = "Xon" ]] || [[ "X$ENABLE_DFLOW_ST" = "Xon" ]]; then
    COV_DIRS=()
    COV_DIRS+=("${BUILD_PATH}/api")
    COV_DIRS+=("${BUILD_PATH}/base")
    cp -rf ${BUILD_PATH}/tests/ge/st/testcase/st_run_data ${BUILD_PATH}/
    cp -rf ${BUILD_PATH}/tests/depends/graph_tuner/libgraphtuner_executor.so ${BUILD_PATH}/tests/ge/st/testcase/
    if [ -d ${BUILD_PATH}/compiler/plugin/nnengine ]; then
      rm -rf ${BUILD_PATH}/compiler/plugin/nnengine/*engine*.so
    else
      mk_dir ${BUILD_PATH}/compiler/plugin/nnengine
    fi
    find ${BUILD_PATH}/compiler/engines -type f -name "*engine*.so" -print0 | xargs -0 -I {} cp -rf {} ${BUILD_PATH}/compiler/plugin/nnengine

    if [ -d ${BUILD_PATH}/compiler/plugin/opskernel/ ]; then
      rm -rf ${BUILD_PATH}/compiler/plugin/opskernel/*engine*.so
    else
      mk_dir ${BUILD_PATH}/compiler/plugin/opskernel/
    fi

    #execute st testcase with memory leak detection by default
    if [[ "X$ENABLE_GE_ST" = "Xon" ]];then
      cp ${BUILD_PATH}/compiler/plugin/nnengine/*engine*.so ${BUILD_PATH}/compiler/plugin/opskernel/
      echo "Run tests with leaks check"
      RUN_TEST_CASE="${BUILD_PATH}/tests/ge/st/testcase/graph_engine_test --gtest_output=xml:${report_dir}/st/graph_engine_test.xml" && ${RUN_TEST_CASE}
      if [[ "$?" -ne 0 ]]; then
        echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
      fi
      COV_DIRS+=("${BUILD_PATH}/graph_metadef")
      COV_DIRS+=("${BUILD_PATH}/compiler")
    fi

    if [[ "X$ENABLE_RT3_ST" = "Xon" ]]; then
      echo "Run tests with leaks check"
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_container_overflow=0
      ctest --output-on-failure -j ${THREAD_NUM} -L st -L st_hetero --test-dir ${BUILD_PATH} --no-tests=error \
            -O ${BUILD_PATH}/ctest_st_hetero.log
      unset LD_PRELOAD
      unset ASAN_OPTIONS
      COV_DIRS+=("${BUILD_PATH}/runtime/v1")
      COV_DIRS+=("${BUILD_PATH}/runtime/v2")
    fi

    if [[ "X$ENABLE_RT2_ST" = "Xon" ]]; then
      echo "Run tests with leaks check"
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_container_overflow=0
      RUN_TEST_CASE="${BUILD_PATH}/tests/ge/st/hybrid_model_exec/hybrid_model_async_exec_test --gtest_output=xml:${report_dir}/st/hybrid_model_async_exec_test.xml" && ${RUN_TEST_CASE} &&
      RUN_TEST_CASE="${BUILD_PATH}/tests/ge/st/testcase/fast_runtime_v2/st_fast_runtime2_test --gtest_output=xml:${report_dir}/st/st_fast_runtime2_test.xml" && ${RUN_TEST_CASE} &&
      RUN_TEST_CASE="${BUILD_PATH}/tests/ge/st/testcase/fast_runtime_v2/dvpp/dvpp_rtkernel/st_dvpp_runtime2_test --gtest_output=xml:${report_dir}/st/st_dvpp_runtime2_test.xml" && ${RUN_TEST_CASE}
      if [[ "$?" -ne 0 ]]; then
          echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
          echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
          exit 1;
      fi
      unset LD_PRELOAD
      unset ASAN_OPTIONS
      COV_DIRS+=("${BUILD_PATH}/runtime/v1")
      COV_DIRS+=("${BUILD_PATH}/runtime/v2")
    fi

    if [[ "X$ENABLE_PYTHON_ST" = "Xon" ]]; then
      unset LD_PRELOAD
      cp ${BUILD_PATH}/tests/depends/python/llm_wrapper.so ${BASEPATH}/api/python/llm_datadist_v1
      cp -r ${BASEPATH}/tests/python_tests ./
      PYTHON_ORIGINAL_PATH=$PYTHONPATH
      export PYTHONPATH=$PYTHON_ORIGINAL_PATH:${BASEPATH}/api/python/llm_datadist/:${BASEPATH}/api/python/
      export LD_PRELOAD=${USE_ASAN}
      echo "----------v1 st start----------"
      ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 coverage run --data-file=coverage_python -m unittest discover python_tests/v1/st
      run_pyge_pytests
      export PYTHONPATH=$PYTHON_ORIGINAL_PATH:${BASEPATH}/api/python/llm_datadist/:${BASEPATH}/api/python/
      unset LD_PRELOAD
    fi

    if [[ "X$ENABLE_PARSER_ST" = "Xon" ]]; then
      echo "---------------- Parser ST Run Start ----------------"
      cp ${BUILD_PATH}/tests/parser/st/st_parser ${OUTPUT_PATH}
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0
      RUN_TEST_CASE="${OUTPUT_PATH}/st_parser --gtest_output=xml:${report_dir}/st/st_parser.xml" && ${RUN_TEST_CASE} &&
      RUN_TEST_CASE="${BUILD_PATH}/tests/ge/st/testcase/graph_engine_compile_test --gtest_output=xml:${report_dir}/st/graph_engine_compile_test.xml" && ${RUN_TEST_CASE}
      if [[ "$?" -ne 0 ]]; then
        echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1
      fi
      unset LD_PRELOAD
      unset ASAN_OPTIONS
      COV_DIRS+=("${BUILD_PATH}/graph_metadef")
    fi

    if [[ "X$ENABLE_DFLOW_ST" = "Xon" ]]; then
      echo "---------------- Dflow Python ST Run Start ----------------"
      export PYDFLOW_SRC_PATH=${BASEPATH}/dflow/pydflow
      export PYDFLOW_TEST_PATH=${BUILD_PATH}/tests/dflow/pydflow
      export PYDFLOW_BUILD_PATH=${BUILD_PATH}
      rm -fr ${PYDFLOW_TEST_PATH}
      cp -r ${BASEPATH}/tests/dflow/pydflow ${PYDFLOW_TEST_PATH}
      bash tests/dflow/pydflow/run_test.sh -s --ascend_install_path=${ASCEND_INSTALL_PATH}
      unset PYDFLOW_SRC_PATH
      unset PYDFLOW_TEST_PATH
      unset PYDFLOW_BUILD_PATH

      echo "---------------- Dflow Udf ST Run Start ----------------"
      export LD_PRELOAD=${USE_ASAN}
      export ASAN_OPTIONS=detect_container_overflow=0
      ctest --output-on-failure -j ${THREAD_NUM} -L st -L st_dflow --test-dir ${BUILD_PATH} --no-tests=error \
            -O ${BUILD_PATH}/ctest_st_dflow.log
      if [[ "$?" -ne 0 ]]; then
        echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1
      fi
      unset LD_PRELOAD;
      unset ASAN_OPTIONS;
      COV_DIRS+=("${BUILD_PATH}/dflow")
    fi

    # remove plugin
    rm -rf ${OUTPUT_PATH}/plugin
    if [[ "X$ENABLE_GE_COV" = "Xon" ]]; then
      echo "Generating coverage statistics, please wait..."
      cd ${BASEPATH}
      rm -rf ${BASEPATH}/cov
      mk_dir ${BASEPATH}/cov
      if [[ "X$ENABLE_PYTHON_ST" = "Xon" ]] || [[ "X$ENABLE_DFLOW_ST" = "Xon" ]]; then
        echo "generate python coverage" ${BUILD_PATH}/coverage_*
        coverage combine ${BUILD_PATH}/coverage_*
        mv .coverage ${BASEPATH}/cov/
      fi

      # 去重
      IFS=$'\n' COV_DIRS_UNIQUE=($(sort -u <<<"${COV_DIRS[*]}"))
      unset IFS

      # 转换为 lcov 参数格式
      COV_DIRS_PARAMS=()
      for dir in "${COV_DIRS_UNIQUE[@]}"; do
        COV_DIRS_PARAMS+=(-d "$dir")
      done

      lcov -c "${COV_DIRS_PARAMS[@]}" -o cov/tmp.info $(add_lcov_ops_by_major_version 2 "--ignore-errors empty,mismatch,negative")
      if [ ! -s "cov/tmp.info" ] || ! grep -q "SF:" "cov/tmp.info"; then
        echo "No valid cpp coverage data found; skip filtering."
        touch cov/coverage.info  # 生成空文件占位，避免后续流程报错
      else
        lcov -r cov/tmp.info '*/deployer/proto/*' '*/output/*' '*/inc/*' '*/op_impl/*' \
                             "*/${BUILD_RELATIVE_PATH}/opensrc/*" "*/${BUILD_RELATIVE_PATH}/proto/*" \
                             "*/${BUILD_RELATIVE_PATH}/grpc_*" '*/third_party/*' '*/tests/*' '/usr/local/*' \
                             '/usr/include/*' '*/metadef/*' \
                             "${ASCEND_INSTALL_PATH}/*" "${ASCEND_3RD_LIB_PATH}/*" \
                             -o cov/coverage.info $(add_lcov_ops_by_major_version 2 "--ignore-errors unused")
        genhtml cov/coverage.info -o cov/html

        if [[ "X$ENABLE_ICOV" = "Xon" ]]; then
          generate_inc_coverage
        fi
      fi
    fi
fi

# generate output package in tar form for runtime/compiler, including ut/st libraries/executables
generate_package()
{
  cd "${BASEPATH}"

  GRAPHENGINE_LIB_PATH="lib"
  RUNTIME_PATH="runtime/lib64"
  COMPILER_PATH="compiler/lib64"
  COMPILER_BIN_PATH="compiler/bin"
  COMPILER_INCLUDE_PATH="compiler/include"
  NNENGINE_PATH="plugin/nnengine/ge_config"
  OPSKERNEL_PATH="plugin/opskernel"
  CPUCOMPILER_PATH="plugin/pnecompiler"

  RUNTIME_LIB=(libdflow_runner.so libge_common.so libge_common_base.so libgraph.so libgraph_base.so libregister.so liberror_manager.so libge_executor.so libdavinci_executor.so libhybrid_executor.so libmodel_deployer.so deployer_daemon libgert.so libexe_graph.so)
  COMPILER_LIB=(libdflow_runner.so libc_sec.so liberror_manager.so libge_common.so libge_common_base.so libslice.so libge_compiler.so libge_executor.so libdavinci_executor.so libhybrid_executor.so libge_runner.so libge_runner_v2.so libgraph.so libgraph_base.so libregister.so libgert.so libexe_graph.so)
  PLUGIN_OPSKERNEL=(libge_local_engine.so libge_local_opskernel_builder.so optimizer_priority.pbtxt)
  PARSER_LIB=(lib_caffe_parser.so libfmk_onnx_parser.so libfmk_parser.so libparser_common.so)

  rm -rf ${OUTPUT_PATH:?}/${RUNTIME_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${COMPILER_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${COMPILER_BIN_PATH}/

  mk_dir "${OUTPUT_PATH}/${RUNTIME_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}/${NNENGINE_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}/${CPUCOMPILER_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_BIN_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_INCLUDE_PATH}"

  cd "${OUTPUT_PATH}"

  find ./ -name graphengine_lib.tar -exec rm {} \;

  cp ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH}/engine_conf.json ${OUTPUT_PATH}/${COMPILER_PATH}/${NNENGINE_PATH}

  find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name libengine.so -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${NNENGINE_PATH}/../ \;

  find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name libcpu_compiler.so -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${CPUCOMPILER_PATH}/../ \;

  MAX_DEPTH=1
  for lib in "${PLUGIN_OPSKERNEL[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_PATH} \;
  done

  for lib in "${PARSER_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  done

  for lib in "${RUNTIME_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${RUNTIME_PATH} \;
  done

  for lib in "${COMPILER_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  done

  find ./lib/fwkacl -name atc.bin -exec cp {} "${OUTPUT_PATH}/${COMPILER_BIN_PATH}" \;

  cp -rf ${BASEPATH}/base/metadef/inc/external/* ${COMPILER_INCLUDE_PATH}
  cp -rf ${BASEPATH}/base/parser/inc/external/* ${COMPILER_INCLUDE_PATH}
  cp -rf ${BASEPATH}/inc/external/* ${COMPILER_INCLUDE_PATH}

  tar -zcf graphengine_lib.tar runtime compiler
}

if [[ "X$ENABLE_GE_UT" = "Xoff" && "X$ENABLE_GE_ST" = "Xoff" && "X$MINDSPORE_MODE" = "Xoff" &&
      "X$ENABLE_GE_BENCHMARK" = "Xoff" && "X$ENABLE_RT2_ST" = "Xoff" && "X$ENABLE_GE_DT" = "Xoff" &&
      "X$ENABLE_RT3_ST" = "Xoff" && "X$ENABLE_RT2_UT" = "Xoff" && "X$ENABLE_PYTHON_ST" = "Xoff" &&
      "X$ENABLE_PYTHON_UT" = "Xoff" && "X$ENABLE_PARSER_UT" = "Xoff" && "X$ENABLE_PARSER_ST" = "Xoff" &&
      "X$ENABLE_DFLOW_UT" = "Xoff" && "X$ENABLE_DFLOW_ST" = "Xoff" ]] ; then
  generate_package
elif [ "X$MINDSPORE_MODE" = "Xon" ]
then
  cd "${OUTPUT_PATH}"
  find ./ -name graphengine_lib.tar -exec rm {} \;
  tar -zcf graphengine_lib.tar lib
fi
echo "---------------- GraphEngine package archive generated ----------------"
