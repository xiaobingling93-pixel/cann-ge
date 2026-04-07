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
date +"test begin: %Y-%m-%d %H:%M:%S"

BASEPATH=$(cd "$(dirname $0)/.."; pwd)
OUTPUT_PATH="${BASEPATH}/output"
BUILD_RELATIVE_PATH="build_ut"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"
echo "PYTHONPATH:${PYTHONPATH}"
echo "LD_LIBRARY_PATH:${LD_LIBRARY_PATH}"
echo "LD_PRELOAD:${LD_PRELOAD}"
echo "ASCEND_OPP_PATH:${ASCEND_OPP_PATH}"
unset LD_LIBRARY_PATH
unset PYTHONPATH
# delete ascend dir in LD_LIBRARY_PATH for test
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's/:*[^:]*Ascend[^:]*:*//g' -e 's/^://' -e 's/:$//')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's/:*[^:]*cann[^:]*:*//g' -e 's/^://' -e 's/:$//')
unset LD_PRELOAD
unset ASCEND_OPP_PATH
echo "PYTHONPATH:${PYTHONPATH}"
echo "LD_LIBRARY_PATH:${LD_LIBRARY_PATH}"
echo "LD_PRELOAD:${LD_PRELOAD}"
echo "ASCEND_OPP_PATH:${ASCEND_OPP_PATH}"

# print usage message
usage() {
  echo "Usage:"
  echo "sh run_test.sh [-u | --ut] [-s | --st]"
  echo "               [-c | --cov] [-j<N>] [-h | --help] [-v | --verbose]"
  echo "               [--cann_3rd_lib_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -u, --ut       Build all ut"
  echo "        =ge               Build all ge ut"
  echo "            =ge_common    Build ge common ut"
  echo "            =rt           Build ge runtime ut"
  echo "            =python       Build ge python ut"
  echo "            =parser       Build ge parser ut"
  echo "            =dflow        Build ge dflow ut"
  echo "        =engines          Build all engines ut"
  echo "            =fe           Build fusion engine ut"
  echo "            =tefusion      Build tefusion engine ut"
  echo "            =dvpp         Build dvpp engine ut"
  echo "            =aicpu        Build aicpu engine ut"
  echo "            =ffts         Build ffts engine ut"
  echo "            =rts          Build rts engine ut"
  echo "            =hcce         Build hcce engine ut"
  echo "        =executor_c       Build executor_c ut"
  echo "        =autofuse_framework         Build autofuse_framework ut"
  echo "        =autofuse_ascendc_api       Build autofuse_ascendc_api ut"
  echo "    -s, --st       Build all st"
  echo "        =ge               Build all ge st"
  echo "            =ge_common    Build ge common st"
  echo "            =rt           Build ge runtime st"
  echo "            =hetero       Build ge hetero st"
  echo "            =python       Build ge python st"
  echo "            =parser       Build ge parser st"
  echo "            =dflow        Build ge dflow st"
  echo "        =engines          Build all engines st"
  echo "            =fe           Build fusion engine st"
  echo "            =tefusion      Build tefusion engine st"
  echo "            =dvpp         Build dvpp engine st"
  echo "            =aicpu        Build aicpu engine st"
  echo "            =ffts         Build ffts engine st"
  echo "            =hcce         Build hcce engine st"
  echo "        =executor_c       Build executor_c st"
  echo "        =autofuse_framework         Build autofuse_framework st"
  echo "        =autofuse_ascendc_api       Build autofuse_ascendc_api st"
  echo "        =autofuse_e2e       Build autofuse_e2e st"
  echo "    -h, --help     Print usage"
  echo "    -c, --cov      Build ut/st with coverage tag"
  echo "                   Please ensure that the environment has correctly installed lcov, gcov, and genhtml."
  echo "                   and the version matched gcc/g++."
  echo "    -v, --verbose  Show detailed build commands during the build process"
  echo "    -j<N>          Set the number of threads used for building Parser, default 8"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                   Set third_party package install path, default ./output/third_party"
  echo ""
}


# $1: ENABLE_UT/ENABLE_ST
# $2: ENABLE_ST/ENABLE_UT
# $3: ENABLE_GE
# $4: ENABLE_ENGINES
# $5: input value of ut or st
check_on() {
  if [ "X$1" = "Xon" ]; then
    usage;
    exit 1
  elif [ "X$2" = "Xon" ]; then
    if [ "X$3" != "Xon" ] || [ "X$4" != "Xon" ] || [ -n "$5" ]; then
      usage
      exit 1
    fi
  fi
}


# parse and set options
checkopts() {
  VERBOSE=""
  THREAD_NUM=16
  COVERAGE=""

  ENABLE_UT="off"
  ENABLE_ST="off"

  ENABLE_GE="off"
  ENABLE_GE_COMMON="off"
  ENABLE_RT="off"
  ENABLE_HETERO="off"
  ENABLE_PYTHON="off"
  ENABLE_PARSER="off"
  ENABLE_DFLOW="off"
  ENABLE_ENGINES="off"
  ENABLE_FE="off"
  ENABLE_TEFUSION="off"
  ENABLE_FFTS="off"
  ENABLE_RTS="off"
  ENABLE_HCCE="off"
  ENABLE_AICPU="off"
  ENABLE_DVPP="off"
  ENABLE_ST_WHOLE_PROCESS="off"
  ENABLE_GE_C="off"
  ENABLE_GE_AUTOFUSE="off"
  ENABLE_GE_AUTOFUSE_FRAMEWORK="off"
  ENABLE_GE_AUTOFUSE_ASCENDC_API="off"
  ENABLE_GE_AUTOFUSE_E2E="off"
  ENABLE_ACL_UT="off"

  ENABLE_GE_BENCHMARK="off"

  ASCEND_3RD_LIB_PATH="$BASEPATH/output/third_party"
  BUILD_METADEF="off"

  parsed_args=$(getopt -a -o u::s::cj:hvf: -l ut::,st::,process_st::,cov,help,verbose,cann_3rd_lib_path: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -u | --ut)
        check_on "$ENABLE_UT" "$ENABLE_ST" "$ENABLE_GE" "$ENABLE_ENGINES" "$2"
        ENABLE_UT="on"
        case "$2" in
          "acl")
            ENABLE_ACL_UT="on"
            shift 2
            ;;
          "")
            ENABLE_GE="on"
            ENABLE_GE_COMMON="on"
            ENABLE_RT="on"
            ENABLE_PYTHON="on"
            ENABLE_PARSER="on"
            ENABLE_DFLOW="on"
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            ENABLE_TEFUSION="on"
            ENABLE_DVPP="on"
            ENABLE_AICPU="on"
            ENABLE_FFTS="on"
            ENABLE_RTS="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "ge")
            ENABLE_GE_COMMON="on"
            ENABLE_RT="on"
            ENABLE_PYTHON="on"
            ENABLE_GE="on"
            ENABLE_PARSER="on"
            ENABLE_DFLOW="on"
            shift 2
            ;;
          "ge_common")
            ENABLE_GE_COMMON="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "rt")
            ENABLE_RT="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "python")
            ENABLE_PYTHON="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "parser")
            ENABLE_PARSER="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "dflow")
            ENABLE_DFLOW="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "engines")
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            ENABLE_TEFUSION="on"
            ENABLE_DVPP="on"
            ENABLE_AICPU="on"
            ENABLE_FFTS="on"
            ENABLE_RTS="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "fe")
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "tefusion")
            ENABLE_ENGINES="on"
            ENABLE_TEFUSION="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "")
            ENABLE_GE="on"
            ENABLE_GE_COMMON="on"
            ENABLE_RT="on"
            ENABLE_PYTHON="on"
            ENABLE_PARSER="on"
            ENABLE_DFLOW="on"
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            ENABLE_DVPP="on"
            ENABLE_AICPU="on"
            ENABLE_FFTS="on"
            ENABLE_RTS="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "ge")
            ENABLE_GE_COMMON="on"
            ENABLE_RT="on"
            ENABLE_PYTHON="on"
            ENABLE_GE="on"
            ENABLE_PARSER="on"
            ENABLE_DFLOW="on"
            shift 2
            ;;
          "ge_common")
            ENABLE_GE_COMMON="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "rt")
            ENABLE_RT="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "python")
            ENABLE_PYTHON="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "parser")
            ENABLE_PARSER="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "dflow")
            ENABLE_DFLOW="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "engines")
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            ENABLE_DVPP="on"
            ENABLE_AICPU="on"
            ENABLE_FFTS="on"
            ENABLE_RTS="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "fe")
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "dvpp")
            ENABLE_ENGINES="on"
            ENABLE_DVPP="on"
            shift 2
            ;;
          "aicpu")
            ENABLE_ENGINES="on"
            ENABLE_AICPU="on"
            shift 2
            ;;
          "ffts")
            ENABLE_ENGINES="on"
            ENABLE_FFTS="on"
            shift 2
            ;;
          "hcce")
            ENABLE_ENGINES="on"
            ENABLE_HCCE="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "executor_c")
            ENABLE_GE_C="on"
            shift 2
            ;;
          "rts")
            ENABLE_ENGINES="on"
            ENABLE_RTS="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "autofuse")
            ENABLE_GE_AUTOFUSE="on"
            shift 2
            ;;
          "autofuse_framework")
            ENABLE_GE_AUTOFUSE_FRAMEWORK="on"
            shift 2
            ;;
          "autofuse_ascendc_api")
            ENABLE_GE_AUTOFUSE_ASCENDC_API="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      -s | --st)
        check_on "$ENABLE_ST" "$ENABLE_UT" "$ENABLE_GE" "$ENABLE_ENGINES" "$2"
        ENABLE_ST="on"
        case "$2" in
          "")
            ENABLE_GE="on"
            ENABLE_GE_COMMON="on"
            ENABLE_RT="on"
            ENABLE_HETERO="on"
            ENABLE_PYTHON="on"
            ENABLE_PARSER="on"
            ENABLE_DFLOW="on"
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            ENABLE_TEFUSION="on"
            ENABLE_DVPP="on"
            ENABLE_AICPU="on"
            ENABLE_FFTS="on"
            ENABLE_RTS="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "ge")
            ENABLE_GE_COMMON="on"
            ENABLE_RT="on"
            ENABLE_HETERO="on"
            ENABLE_PYTHON="on"
            ENABLE_PARSER="on"
            ENABLE_DFLOW="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "ge_common")
            ENABLE_GE_COMMON="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "rt")
            ENABLE_RT="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "hetero")
            ENABLE_HETERO="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "python")
            ENABLE_PYTHON="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "parser")
            ENABLE_PARSER="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "dflow")
            ENABLE_DFLOW="on"
            ENABLE_GE="on"
            shift 2
            ;;
          "engines")
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            ENABLE_TEFUSION="on"
            ENABLE_DVPP="on"
            ENABLE_AICPU="on"
            ENABLE_FFTS="on"
            ENABLE_RTS="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "fe")
            ENABLE_ENGINES="on"
            ENABLE_FE="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "tefusion")
            ENABLE_ENGINES="on"
            ENABLE_TEFUSION="on"
            BUILD_METADEF="on"
            shift 2
            ;;
          "dvpp")
            ENABLE_ENGINES="on"
            ENABLE_DVPP="on"
            shift 2
            ;;
          "aicpu")
            ENABLE_ENGINES="on"
            ENABLE_AICPU="on"
            shift 2
            ;;
          "ffts")
            ENABLE_ENGINES="on"
            ENABLE_FFTS="on"
            shift 2
            ;;
          "hcce")
            ENABLE_ENGINES="on"
            ENABLE_HCCE="on"
            shift 2
            ;;
          "executor_c")
            ENABLE_GE_C="on"
            shift 2
            ;;
          "rts")
            ENABLE_ENGINES="on"
            ENABLE_RTS="on"
            shift 2
            ;;
          "autofuse")
            ENABLE_GE_AUTOFUSE="on"
            shift 2
            ;;
          "autofuse_framework")
            ENABLE_GE_AUTOFUSE_FRAMEWORK="on"
            shift 2
            ;;
          "autofuse_ascendc_api")
            ENABLE_GE_AUTOFUSE_ASCENDC_API="on"
            shift 2
            ;;
          "autofuse_e2e")
            ENABLE_GE_AUTOFUSE_E2E="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      --process_st)
        ENABLE_ST="on"
        ENABLE_ENGINES="on"
        ENABLE_FE="on"
        ENABLE_ST_WHOLE_PROCESS="on"
        BUILD_METADEF="on"
        shift 2
        ;;
      -c | --cov)
        COVERAGE="-c"
        shift
        ;;
      -h | --help)
        usage
        exit 1
        ;;
      -j)
        THREAD_NUM=$2
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="-v"
        shift
        ;;
      --cann_3rd_lib_path)
        ASCEND_3RD_LIB_PATH="$(realpath $2)"
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
    echo "[INFO] Changed files only contain docs/, examples/, .claude/, .opencode/, README.md, CONTRIBUTING.md or AGENTS.md, skipping test."
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

build_acl() {
  echo "create build directory and build acl ut";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"

  ENABLE_ACL_COV="$(echo ${ENABLE_ACL_COV} | tr 'a-z' 'A-Z')"
  ENABLE_ACL_UT="$(echo ${ENABLE_ACL_UT} | tr 'a-z' 'A-Z')"
  ENABLE_C_COV="$(echo ${ENABLE_C_COV} | tr 'a-z' 'A-Z')"
  ENABLE_C_UT="$(echo ${ENABLE_C_UT} | tr 'a-z' 'A-Z')"
  CMAKE_ARGS="-DBUILD_OPEN_PROJECT=True \
              -DENABLE_OPEN_SRC=True \
              -DASCENDCL_C=${ASCENDCL_C} \
              -DBUILD_METADEF=${BUILD_METADEF} \
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
              -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -DASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -DASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -DENABLE_ACL_COV=${ENABLE_ACL_COV} \
              -DENABLE_ACL_UT=${ENABLE_ACL_UT} \
              -DENABLE_C_COV=${ENABLE_C_COV} \
              -DENABLE_C_UT=${ENABLE_C_UT} \
              -DPLATFORM=${PLATFORM} \
              -DPRODUCT=${PRODUCT}"

  echo "CMAKE_ARGS=${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ..
  if [ $? -ne 0 ]; then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi

  make ${VERBOSE} acl_utest -j${THREAD_NUM}
  if [ $? -ne 0 ]; then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} failed."
    return 1
  fi
  echo "acl ut build success!"
}


run_ut_acl() {
  cp ${BUILD_PATH}/tests/acl_ut/ut/acl/acl_utest ${OUTPUT_PATH}

  local report_dir="${OUTPUT_PATH}/report/ut" && mk_dir "${report_dir}"
  export LD_PRELOAD=${USE_ASAN}
  export ASAN_OPTIONS=detect_odr_violation=0
  RUN_TEST_CASE="${OUTPUT_PATH}/acl_utest --gtest_output=xml:${report_dir}/acl_utest.xml" && ${RUN_TEST_CASE}
  if [[ "$?" -ne 0 ]]; then
    echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
    echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
    exit 1;
  fi
  unset LD_PRELOAD
  unset ASAN_OPTIONS
  echo "Generated coverage statistics, please wait..."
  cd ${BASEPATH}
  rm -rf ${BASEPATH}/cov
  mkdir ${BASEPATH}/cov
  source ${BASEPATH}/scripts/support_multiple_versions_of_lcov.sh
  lcov -c -d ${BUILD_RELATIVE_PATH}/tests/acl_ut/ut/acl -o cov/tmp.info
  lcov -r cov/tmp.info '*/output/*' "*/${BUILD_RELATIVE_PATH}/opensrc/*" "*/${BUILD_RELATIVE_PATH}/proto/*" \
      '*/third_party/*' '*/tests/*' '/usr/local/*' '/usr/include/*' \
      "${ASCEND_INSTALL_PATH}/*" "${ASCEND_3RD_LIB_PATH}/*" \
      -o cov/coverage.info $(add_lcov_ops_by_major_version 2 "--ignore-errors unused")
  cd ${BASEPATH}/cov
  genhtml coverage.info -o cov/html
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

  # 为当前 run_test.sh 进程创建唯一的汇总文件名
  TEST_SUMMARY_FILE="${OUTPUT_PATH}/.test_summary_$$.tmp"
  export TEST_SUMMARY_FILE

  # build cann 3rd lib
  bash ${BASEPATH}/build_third_party.sh ${ASCEND_3RD_LIB_PATH} ${THREAD_NUM} "LLT"

  # build acl ut
  if [ "X$ENABLE_ACL_UT" = "Xon" ]; then
    echo "---------------- acl ut build start ----------------"
    g++ -v
    mk_dir ${OUTPUT_PATH}
    build_acl
    if [[ "$?" -ne 0 ]]; then
      echo "acl ut build failed.";
      exit 1;
    fi
    echo "---------------- acl ut build finished ----------------"

    rm -f ${OUTPUT_PATH}/libgmock*.so
    rm -f ${OUTPUT_PATH}/libgtest*.so
    rm -f ${OUTPUT_PATH}/lib*_stub.so

    chmod -R 750 ${OUTPUT_PATH}

    echo "---------------- acl ut output generated ----------------"
    run_ut_acl
    exit 0
  fi

  if [ "X$ENABLE_UT" != "Xon" ] && [ "X$ENABLE_ST" != "Xon" ]; then
    ENABLE_UT="on"
    ENABLE_ST="on"
    ENABLE_GE="on"
    ENABLE_GE_COMMON="on"
    ENABLE_RT="on"
    ENABLE_HETERO="on"
    ENABLE_PYTHON="on"
    ENABLE_PARSER="on"
    ENABLE_DFLOW="on"
    ENABLE_ENGINES="on"
    ENABLE_FE="on"
    ENABLE_TEFUSION="on"
    ENABLE_DVPP="on"
    ENABLE_AICPU="on"
    ENABLE_FFTS="on"
    ENABLE_RTS="on"
    ENABLE_HCCE="on"
  fi

  export BUILD_METADEF=${BUILD_METADEF}
  export ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}
  export ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH}

  if [ ! -f ${ASCEND_INSTALL_PATH}/fwkacllib/lib64/switch_by_index.o ]; then
    mkdir -p ${ASCEND_INSTALL_PATH}/fwkacllib/lib64/
    touch ${ASCEND_INSTALL_PATH}/fwkacllib/lib64/switch_by_index.o
  fi

  # module ge
  if [ "X$ENABLE_GE" = "Xon" ]; then
    # ge ut
    if [ "X$ENABLE_UT" == "Xon" ]; then
      if [ "X$ENABLE_GE_COMMON" = "Xon" ] && [ "X$ENABLE_RT" = "Xon" ] && [ "X$ENABLE_PYTHON" = "Xon" ] && [ "X$ENABLE_PARSER" = "Xon" ] && [ "X$ENABLE_DFLOW" = "Xon" ]; then
        bash scripts/build_fwk.sh -t -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_GE_COMMON" = "Xon" ]; then
        bash scripts/build_fwk.sh -T -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_RT" = "Xon" ]; then
        bash scripts/build_fwk.sh -L -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_PYTHON" = "Xon" ]; then
        bash scripts/build_fwk.sh -l -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_PARSER" = "Xon" ]; then
        bash scripts/build_fwk.sh -m -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_DFLOW" = "Xon" ]; then
        bash scripts/build_fwk.sh -o -j $THREAD_NUM $VERBOSE $COVERAGE
      else
        echo "unknown ut type."
      fi
    fi

    # ge st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      if [ "X$ENABLE_GE_COMMON" = "Xon" ] && [ "X$ENABLE_RT" = "Xon" ] && [ "X$ENABLE_HETERO" = "Xon" ] && [ "X$ENABLE_PYTHON" == "Xon" ] && [ "X$ENABLE_PARSER" == "Xon" ] && [ "X$ENABLE_DFLOW" == "Xon" ]; then
        bash scripts/build_fwk.sh -s -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_GE_COMMON" = "Xon" ]; then
        bash scripts/build_fwk.sh -O -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_RT" = "Xon" ]; then
        bash scripts/build_fwk.sh -R -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_HETERO" = "Xon" ]; then
        bash scripts/build_fwk.sh -K -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_PYTHON" = "Xon" ]; then
        bash scripts/build_fwk.sh -P -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_PARSER" = "Xon" ]; then
        bash scripts/build_fwk.sh -n -j $THREAD_NUM $VERBOSE $COVERAGE
      elif [ "X$ENABLE_DFLOW" = "Xon" ]; then
        bash scripts/build_fwk.sh -D -j $THREAD_NUM $VERBOSE $COVERAGE
      else
        echo "unknown type st."
      fi
    fi
  fi

  # module fe
  if [ "X$ENABLE_ENGINES" = "Xon" ]; then
    # engines ut
    if [ "X$ENABLE_UT" == "Xon" ]; then
      if [ "X$ENABLE_FE" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -u -n -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_TEFUSION" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -u -t -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_DVPP" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -k -u -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_AICPU" = "Xon"  ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -a -u -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_FFTS" = "Xon" ]; then
        bash scripts/build.sh -f -u -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_RTS" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -u -r -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_HCCE" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -u -e -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
    fi

    # engines st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      if [ "X$ENABLE_FE" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -s -n -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_DVPP" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -k -s -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_AICPU" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -a -s -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_FFTS" = "Xon" ]; then
        bash scripts/build.sh -f -s -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_TEFUSION" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -s -t -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
      if [ "X$ENABLE_HCCE" = "Xon" ]; then
        export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${ASCEND_INSTALL_PATH}/devlib:$LD_LIBRARY_PATH
        bash scripts/build.sh -s -e -j $THREAD_NUM $VERBOSE $COVERAGE
      fi

      # fe process st
      if [ "X$ENABLE_ST_WHOLE_PROCESS" = "Xon" ]; then
        bash scripts/build.sh -w -n -j $THREAD_NUM $VERBOSE $COVERAGE
      fi
    fi
  fi

  # module executor_c
  if [ "X$ENABLE_GE_C" == "Xon" ]; then
    # executor_c ut
    if [ "X$ENABLE_UT" = "Xon" ]; then
      bash scripts/build_executor_c.sh -u -j $THREAD_NUM $VERBOSE $COVERAGE
    fi

    # executor_c st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      bash scripts/build_executor_c.sh -s -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
  fi

  # module autofuse
  if [ "X$ENABLE_GE_AUTOFUSE" == "Xon" ]; then
    # executor_c ut
    if [ "X$ENABLE_UT" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -u -j $THREAD_NUM $VERBOSE $COVERAGE
    fi

    # executor_c st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -s -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
  fi

  # module autofuse_framework
  if [ "X$ENABLE_GE_AUTOFUSE_FRAMEWORK" == "Xon" ]; then
    # autofuse_framework ut
    if [ "X$ENABLE_UT" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -u -m framework -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
    # autofuse_framework st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -s -m framework -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
  fi

  # module autofuse_ascendc_api
  if [ "X$ENABLE_GE_AUTOFUSE_ASCENDC_API" == "Xon" ]; then
    # autofuse_ascendc_api ut
    if [ "X$ENABLE_UT" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -u -m ascendc_api -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
    # autofuse_ascendc_api st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -s -m ascendc_api -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
  fi

  # module autofuse_e2e
  if [ "X$ENABLE_GE_AUTOFUSE_E2E" == "Xon" ]; then
    # autofuse_e2e st
    if [ "X$ENABLE_ST" = "Xon" ]; then
      bash scripts/test/run_autofuse_test.sh -s -m e2e -j $THREAD_NUM $VERBOSE $COVERAGE
    fi
  fi

  date +"test end: %Y-%m-%d %H:%M:%S"
}

main "$@"
