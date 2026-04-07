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
AIRDIR="$(basename $BASEPATH)"
echo "ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}"
echo "ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH}"
echo "FE_ST_PATH=${FE_ST_PATH}"
echo "BUILD_METADEF=${BUILD_METADEF}"
if [ -z "${BUILD_METADEF}" ] ; then
  BUILD_METADEF=ON
fi
if [ -z "${OUTPUT_PATH}" ] ; then
  OUTPUT_PATH="${BASEPATH}/output"
fi

export BUILD_PATH="${BASEPATH}/build/"
export AIR_CODE_DIR=${AIRDIR}
echo "AIR_CODE_DIR=${AIR_CODE_DIR}"

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build.sh [-h] [-n] [-a] [-k] [-e] [-f] [-c] [-p] [-v] [-g] [r] [-o] [-j[n]]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -n Build fe ut/st"
  echo "    -f Build ffts ut/st"
  echo "    -r Build rts ut/st"
  echo "    -a Build aicpu ut/st"
  echo "    -k Build dvpp ut/st"
  echo "    -e Build hcce ut/st"
  echo "    -c Build ut/st with coverage tag"
  echo "    -u Build and run ut"
  echo "    -s Build and run st"
  echo "    -p Build inference or train"
  echo "    -v Display build command"
  echo "    -o Only compile ut and st, not execute it , default on"
  echo "    -j[n] Set the number of threads used for building AIR, default is 8"
  echo "to be continued ..."
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  ENABLE_FE_LLT="off"
  ENABLE_TEFUSION_LLT="off"
  ENABLE_FFTS_LLT="off"
  ENABLE_AICPU_LLT="off"
  ENABLE_DVPP_LLT="off"
  NABLE_RTS_LLT="off"
  ENABLE_HCCE_LLT="off"
  ENABLE_LLT_COV="off"
  PLATFORM="all"
  PRODUCT="normal"
  ENABLE_UT="off"
  ENABLE_ST="off"
  ENABLE_ST_WHOLE_PROCESS="off"
  ENABLE_ASAN="false"
  ENABLE_RUN_LLT="on"
  CMAKE_BUILD_TYPE="Release"

  # Process the options
  while getopts 'dnftakerchj:vp:guswo' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      n)
        ENABLE_FE_LLT="on"
        ENABLE_ASAN="true"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      t)
        ENGINE_DT="on"
        ENABLE_TEFUSION_LLT="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      f)
        ENGINE_DT="on"
        ENABLE_FFTS_LLT="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      a)
        ENGINE_DT="on"
        ENABLE_AICPU_LLT="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      k)
        ENGINE_DT="on"
        ENABLE_DVPP_LLT="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      e)
        ENGINE_DT="on"
        ENABLE_HCCE_LLT="on"
        ENABLE_PKG="on"
        ENABLE_LLT_PKG="on"
        CMAKE_BUILD_TYPE="Release"
        ;;
      r)
        ENGINE_DT="on"
        ENABLE_RTS_LLT="on"
        ENABLE_PKG="on"
        ENABLE_LLT_PKG="on"
        CMAKE_BUILD_TYPE="Release"
        ;;
      c)
        ENABLE_LLT_COV="on"
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
      u)
        ENABLE_UT="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      s)
        ENABLE_ST="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      w)
        ENABLE_ST_WHOLE_PROCESS="on"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      o)
        # only compile llt, not run
        ENABLE_RUN_LLT="off"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

TOPI_INIT=""
PYTHON_SITE_PKG_PATH=$(python3 -m site --user-site)
CANN_KB_INIT="$PYTHON_SITE_PKG_PATH/tbe/common/repository_manager/interface.py"
INIT_MULTI_PROCESS_ENV_PATH="$PYTHON_SITE_PKG_PATH/te_fusion/parallel_compilation.py"
insert_str="\ \ \ \ return 1, None, None, None\ #warning:\ fe_st\ substitute,\ need\ to\ restore"
install_python_stub_file()
{
  pip3 install sympy -i https://pypi.tuna.tsinghua.edu.cn/simple
  # 1. install dependencies
  echo "CANN_KB_INIT = ${CANN_KB_INIT}"
  echo "INIT_MULTI_PROCESS_ENV_PATH = ${INIT_MULTI_PROCESS_ENV_PATH}"
#  cann_kb_exist=$()
#  init_multi_process_env_exist=$(! -f "${INIT_MULTI_PROCESS_ENV_PATH}")
  if [ ! -f "${CANN_KB_INIT}" ] || [ ! -f "${INIT_MULTI_PROCESS_ENV_PATH}" ];
  then
    pip3 install --user "$ASCEND_INSTALL_PATH/compiler/lib64/te-0.4.0-py3-none-any.whl" --force-reinstall
  else
    echo "Both files exist."
  fi

  # 2. substitue keywords
#  real_parallel_tbe=$TE_PARALLEL_COMPILER
#  if [ "$real_parallel_tbe" = "-1" ]
#  then
#  insert_str_finder=$(grep "${insert_str}" ${INIT_MULTI_PROCESS_ENV_PATH} -n |awk -F ":" '{print $1}')
#  if [ "$insert_str_finder" = "" ]
#  then
#    line_num=$(grep "def init_multi_process_env(" ${INIT_MULTI_PROCESS_ENV_PATH} -n -A6\
#      |grep "):" |awk -F "-" '{print $1}')
#    echo "line_num = ${line_num}"
#   sed -i "${line_num} a ${insert_str}" ${INIT_MULTI_PROCESS_ENV_PATH}
#  fi
#  fi
}

restore_python_stub_file()
{
  echo "restore_python_stub_file"
  delete_line_nums=$(grep "${insert_str}" ${INIT_MULTI_PROCESS_ENV_PATH} -n |awk -F ":" '{print $1}')
  arr=(${delete_line_nums//,/})
  delete_size=${#arr[@]}
  echo "delete size ${delete_size}"
  for ((i=$delete_size-1; i>=0; i--))
  do
    sed -i "${arr[i]}d" ${INIT_MULTI_PROCESS_ENV_PATH}
  done
}

download_mockcpp() 
{
    if [ -d "${MOCKCPP_DIR}/mockcpp" ];then
        rm -rf ${MOCKCPP_DIR}/mockcpp
        echo "Info: delete ${MOCKCPP_DIR}/mockcpp"
    fi
 
    if [ -d "${MOCKCPP_BUILD_DIR}" ]; then
        echo "Info: mockcpp already built, skipping download and compilation."
        return
    fi
 
    # 下载mockcpp
    echo "Info: Downloading mockcpp..."
 
    cd ${MOCKCPP_DIR}
    git clone https://gitcode.com/cann-src-third-party/mockcpp.git || {
        echo "ERROR: Failed to download mockcpp."
        echo "ERROR: Please execute separately [git clone https://gitcode.com/cann-src-third-party/mockcpp.git]"
        exit 1
    }
 
    cd ${MOCKCPP_DIR}/mockcpp
    tar -zxvf mockcpp-2.7.tar.gz
    cd ${MOCKCPP_DIR}/mockcpp/mockcpp-2.7
    patch -p1 < ../mockcpp-2.7.patch
}
 
build_mockcpp() 
{   
    cd "${MOCKCPP_BUILD_DIR}"
    echo "Info compiler mockcpp"

    sed -i 's/^	print sys\.argv\[0\], getUsageString(longOpts)$/	print(sys.argv[0], getUsageString(longOpts))/g' "${MOCKCPP_DIR}/mockcpp/mockcpp-2.7/src/get_long_opt.py"
    sed -i 's/^	except getopt.GetoptError, err:$/	except getopt.GetoptError as err:/g' "${MOCKCPP_DIR}/mockcpp/mockcpp-2.7/src/get_long_opt.py"
    sed -i 's/^	print >> sys.stderr, str(err)$/	print(str(err), file=sys.stderr)/g' "${MOCKCPP_DIR}/mockcpp/mockcpp-2.7/src/get_long_opt.py"

    cmake "${MOCKCPP_DIR}/mockcpp/mockcpp-2.7" \
          || {
        echo "ERROR: CMake configure failed"
        exit 1
    }

    cmake --build . --target mockcpp ${JOB_NUM} || {
        echo "ERROR: Build failed"
        exit 1
    }
 
    chmod 775 libmockcpp.a
}

build_air()
{
  echo "create build directory and build AIR";
  if [ "X$ENABLE_HCCE_LLT" = "Xon" ];then
    MOCKCPP_DIR="${BASEPATH}/tests/engines/hccl_engine/third_party"
    MOCKCPP_BUILD_DIR="${MOCKCPP_DIR}/mockcpp/mockcpp-2.7/build"
    # mockcpp编译过了之后就不编译了
    if [ -f "${MOCKCPP_BUILD_DIR}/libmockcpp.a" ];then
        echo "Info: mockcpp is compiled"
    else
        echo "Info: begin compiler mockcpp"
        mkdir -p ${MOCKCPP_DIR}
        download_mockcpp

        if [ -d "${MOCKCPP_DIR}/mockcpp/mockcpp-2.7" ]; then
            mkdir -p ${MOCKCPP_BUILD_DIR}
            build_mockcpp
        else
            echo "ERROR: The compilation directory does not exist."
        fi
    fi
  fi
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  cmake -D BUILD_OPEN_PROJECT=True \
        -D ENABLE_OPEN_SRC=True \
        -D ENABLE_LLT_COV=${ENABLE_LLT_COV} \
        -D ENGINE_DT=${ENGINE_DT} \
        -D ENABLE_FE_LLT=${ENABLE_FE_LLT} \
        -D ENABLE_TEFUSION_LLT=${ENABLE_TEFUSION_LLT} \
        -D ENABLE_FFTS_LLT=${ENABLE_FFTS_LLT} \
        -D ENABLE_RTS_LLT=${ENABLE_RTS_LLT} \
        -D ENABLE_AICPU_LLT=${ENABLE_AICPU_LLT} \
        -D ENABLE_DVPP_LLT=${ENABLE_DVPP_LLT} \
        -D ENABLE_HCCE_LLT=${ENABLE_HCCE_LLT} \
        -D ENABLE_UT=${ENABLE_UT} \
        -D ENABLE_ST=${ENABLE_ST} \
        -D ENABLE_ST_WHOLE_PROCESS=${ENABLE_ST_WHOLE_PROCESS} \
        -D PLATFORM=${PLATFORM} \
        -D PRODUCT=${PRODUCT} \
        -D BUILD_METADEF=${BUILD_METADEF} \
        -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
        -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
        -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
        -D ENABLE_ASAN=${ENABLE_ASAN} \
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D ENABLE_PKG=${ENABLE_PKG} \
        -D ENABLE_LLT_PKG=${ENABLE_LLT_PKG} \
        ..
  if [ "X$ENABLE_AICPU_LLT" = "Xon" ] || [ "X$ENABLE_FE_LLT" = "Xon" ] || [ "X$ENABLE_TEFUSION_LLT" = "Xon" ] || [ "X$ENABLE_FFTS_LLT" = "Xon" ] || [ "X$ENABLE_DVPP_LLT" = "Xon" ] || [ "X$ENABLE_RTS_LLT" = "Xon" ] || [ "X$ENABLE_HCCE_LLT" = "Xon" ];then
    make ${VERBOSE} select_targets -j${THREAD_NUM}
  else
    make ${VERBOSE} select_targets -j${THREAD_NUM} && make install
  fi

  if [ $? -ne 0 ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "AIR build success!"
}

run_llt_with_cov()
{
  if [[ "X$ENABLE_RUN_LLT" != "Xon" ]]
  then
    return 0
  fi
  export ASCEND_OPP_PATH=$ASCEND_INSTALL_PATH/opp
  export PATH=$ASCEND_INSTALL_PATH/compiler/ccec_compiler/bin:$PATH
  export LD_LIBRARY_PATH=$ASCEND_INSTALL_PATH/compiler/lib64:$ASCEND_INSTALL_PATH/runtime/lib64:$ASCEND_INSTALL_PATH/runtime/lib64/stub:$LD_LIBRARY_PATH
  cd "${BASEPATH}"
  if [ "X$ENABLE_FE_LLT" = "Xon" ]
  then
    cd "${BASEPATH}/build/tests/engines/nn_engine/depends/te_fusion/"
    ln -sf libte_fusion_stub.so libte_fusion.so
    cd "${BASEPATH}"
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run fe ut ----------------"
      cd "${BASEPATH}/build/tests/engines/nn_engine/"
      mk_dir ./ut/plugin/opskernel/fe_config/
      ln -sf ../depends/te_fusion/libte_fusion_stub.so ./ut/libte_fusion.so
      ln -sf ../../../depends/graph_tuner/libgraph_tuner_stub.so ./ut/plugin/opskernel/libgraph_tuner.so
      cp "${BASEPATH}/compiler/engines/nn_engine/optimizer/fe_config/fe.ini" "${BASEPATH}/build/tests/engines/nn_engine/ut/plugin/opskernel/fe_config"
      cd "${BASEPATH}/../"
      if [ "X$ENABLE_ASAN" = "Xtrue" ];then
        USE_ASAN=$(gcc -print-file-name=libasan.so)
        export LD_PRELOAD=${USE_ASAN}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
        export ASAN_OPTIONS=detect_leaks=0
      fi
      ./$AIRDIR/build/tests/engines/nn_engine/ut/fe_ut
      if [ "X$ENABLE_ASAN" = "Xtrue" ];then
        unset LD_PRELOAD
        unset ASAN_OPTIONS
      fi
      echo "---------------- Finish the fe ut ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of fe ut ----------------"
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_fe_ut/"
        lcov -c -d build/tests/engines/nn_engine/ut -o cov_fe_ut/tmp.info
        lcov -r cov_fe_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_fe_ut/coverage.info
        cd "${BASEPATH}/cov_fe_ut/"
        genhtml coverage.info
        echo "---------------- Finish the generating coverage of fe ut ----------------"
      fi
    fi
    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run fe st ----------------"
      cd "${BASEPATH}/build/tests/engines/nn_engine/"
      mk_dir ./st/plugin/opskernel/fe_config/
      ln -sf ../depends/te_fusion/libte_fusion_stub.so ./st/libte_fusion.so
      ln -sf ../../../depends/graph_tuner/libgraph_tuner_stub.so ./st/plugin/opskernel/libgraph_tuner.so
      cp "${BASEPATH}/compiler/engines/nn_engine/optimizer/fe_config/fe.ini" "${BASEPATH}/build/tests/engines/nn_engine/st/plugin/opskernel/fe_config"
      cd "${BASEPATH}/../"
      if [ "X$ENABLE_ASAN" = "Xtrue" ];then
        USE_ASAN=$(gcc -print-file-name=libasan.so)
        export LD_PRELOAD=${USE_ASAN}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
        export ASAN_OPTIONS=detect_leaks=0
      fi
      ./$AIRDIR/build/tests/engines/nn_engine/st/fe_st
      if [ "X$ENABLE_ASAN" = "Xtrue" ];then
        unset LD_PRELOAD
        unset ASAN_OPTIONS
      fi
      echo "---------------- Finish the fe st ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_fe_st/"
        lcov -c -d build/tests/engines/nn_engine/st -o cov_fe_st/tmp.info
        lcov -r cov_fe_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_fe_st/coverage.info
        cd "${BASEPATH}/cov_fe_st/"
        genhtml coverage.info
      fi
    fi
    if [ "X$ENABLE_ST_WHOLE_PROCESS" = "Xon" ]
    then
      install_python_stub_file
      echo "---------------- Begin to run fe whole process st ----------------"
      cd "${BASEPATH}/../"
      date
      ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess310P3.*
      ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess910A.*
      ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess910B.*
      ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeOmConsistencyCheck.*
      ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess310B.*
      ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcessNano.*
      date
      echo "---------------- Finish the fe whole process st ----------------"
      restore_python_stub_file
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_fe_st_whole_process/"
        lcov -c -d build/tests/engines/nn_engine/st -o cov_fe_st_whole_process/tmp.info
        lcov -r cov_fe_st_whole_process/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_fe_st_whole_process/coverage.info
        cd "${BASEPATH}/cov_fe_st_whole_process/"
        genhtml coverage.info
      fi
    fi
  fi
  if [ "X$ENABLE_TEFUSION_LLT" = "Xon" ]
  then
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run tefusion ut ----------------"
      cd "${BASEPATH}/../"
      ./$AIRDIR/build/tests/engines/te_fusion/ut/tefusion_ut
      echo "---------------- Finish tefusion ut ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of tefusion ut ----------------"
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_tefusion_ut/"
        lcov -c -d build/tests/engines/te_fusion/ut -o cov_tefusion_ut/tmp.info
        lcov -r cov_tefusion_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_tefusion_ut/coverage.info
        cd "${BASEPATH}/cov_tefusion_ut/"
        genhtml coverage.info
        echo "---------------- Finish generating coverage of tefusion ut ----------------"
      fi
    fi
    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run tefusion st ----------------"
      cd "${BASEPATH}/../"
      ./$AIRDIR/build/tests/engines/te_fusion/st/tefusion_st
      echo "---------------- Finish tefusion st ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_tefusion_st/"
        lcov -c -d build/tests/engines/te_fusion/st -o cov_tefusion_st/tmp.info
        lcov -r cov_tefusion_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_tefusion_st/coverage.info
        cd "${BASEPATH}/cov_tefusion_st/"
        genhtml coverage.info
      fi
    fi
  fi
  if [ "X$ENABLE_FFTS_LLT" = "Xon" ]
  then
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run ffts ut ----------------"
      cd "${BASEPATH}/../"
      ./$AIRDIR/build/tests/engines/ffts_engine/ut/ffts_ut
      echo "---------------- Finish the ffts ut ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of ffts ut ----------------"
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_ffts_ut/"
        lcov -c -d build/tests/engines/ffts_engine/ut -o cov_ffts_ut/tmp.info
        lcov -r cov_ffts_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_ffts_ut/coverage.info
        cd "${BASEPATH}/cov_ffts_ut/"
        genhtml coverage.info
        echo "---------------- Finish the generating coverage of ffts ut ----------------"
      fi
    fi
    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run ffts st ----------------"
      cd "${BASEPATH}/../"
      ./$AIRDIR/build/tests/engines/ffts_engine/st/ffts_st
      echo "---------------- Finish the ffts st ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        mk_dir "${BASEPATH}/cov_ffts_st/"
        lcov -c -d build/tests/engines/ffts_engine/st -o cov_ffts_st/tmp.info
        lcov -r cov_ffts_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_ffts_st/coverage.info
        cd "${BASEPATH}/cov_ffts_st/"
        genhtml coverage.info
      fi
    fi
  fi

  echo "tt===> ready do rts ut : $ENABLE_RTS_LLT : $ENABLE_UT : $ENABLE_LLT_COV, path=${BASEPATH}"
  if [ "X$ENABLE_RTS_LLT" = "Xon" ]
  then
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run rts ut ----------------"
      unset ASCEND_OPP_PATH
      echo "${BASEPATH}/build/tests/engines/rts_engine/ut/"
      cd "${BASEPATH}/build/tests/engines/rts_engine/ut/"
      ./rts_engine_utest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run rts ut ------------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of rts ut ----------------"
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_rts_ut/" ];then
          rm -rf "${BASEPATH}/cov_rts_ut/"
        fi
        mk_dir "${BASEPATH}/cov_rts_ut/"
        lcov -c -d ${BASEPATH}/build/tests/engines/rts_engine/ut -o ${BASEPATH}/cov_rts_ut/tmp.info
        lcov -r ${BASEPATH}/cov_rts_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*' -o ${BASEPATH}/cov_rts_ut/coverage.info
        cd "${BASEPATH}/cov_rts_ut/"
        genhtml coverage.info
        echo "---------------- Finish the generating coverage of rts ut ----------------"
      fi
    fi

    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run rts st ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/rts_engine/ut/"
      ./rts_engine_utest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run rts st ------------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_rts_st/" ];then
          rm -rf "${BASEPATH}/cov_rts_st/"
        fi
        mk_dir "${BASEPATH}/cov_rts_st/"
        lcov -c -d ${BASEPATH}/build/tests/engines/rts_engine/ut -o ${BASEPATH}/cov_rts_st/tmp.info
        lcov -r ${BASEPATH}/cov_rts_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*' -o ${BASEPATH}/cov_rts_st/coverage.info
        cd "${BASEPATH}/cov_rts_st/"
        genhtml coverage.info
      fi
    fi
  fi

  if [ "X$ENABLE_AICPU_LLT" = "Xon" ]
  then
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run aicpu ut ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/cpueng/ut/"
      ./host_engine_utest
      ./cpu_engine_utest
      ./tf_engine_utest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run aicpu ut ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of aicpu ut ----------------"
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_aicpu_ut/" ];then
          rm -rf "${BASEPATH}/cov_aicpu_ut/"
        fi
        mk_dir "${BASEPATH}/cov_aicpu_ut/"
        lcov -c -d build/tests/engines/cpueng/ut -o cov_aicpu_ut/tmp.info
        lcov -r cov_aicpu_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_aicpu_ut/coverage.info
        cd "${BASEPATH}/cov_aicpu_ut/"
        genhtml coverage.info
        echo "---------------- Finish the generating coverage of aicpu ut ----------------"
      fi
    fi
    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run aicpu st ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/cpueng/st/"
      ./host_engine_stest
      ./cpu_engine_stest
      ./tf_engine_stest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run aicpu st ----------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_aicpu_st/" ];then
          rm -rf "${BASEPATH}/cov_aicpu_st/"
        fi
        mk_dir "${BASEPATH}/cov_aicpu_st/"
        lcov -c -d build/tests/engines/cpueng/st -o cov_aicpu_st/tmp.info
        lcov -r cov_aicpu_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_aicpu_st/coverage.info
        cd "${BASEPATH}/cov_aicpu_st/"
        genhtml coverage.info
      fi
    fi
  fi

  echo "tt===> ready do dvpp ut : $ENABLE_DVPP_LLT : $ENABLE_UT : $ENABLE_LLT_COV, path=${BASEPATH}"
  if [ "X$ENABLE_DVPP_LLT" = "Xon" ]
  then
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run dvpp ut ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/dvppeng/ut/"
      ./dvpp_engine_utest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run dvpp ut ------------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of dvpp ut ----------------"
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_dvpp_ut/" ];then
          rm -rf "${BASEPATH}/cov_dvpp_ut/"
        fi
        mk_dir "${BASEPATH}/cov_dvpp_ut/"
        lcov -c -d ${BASEPATH}/build/tests/engines/dvppeng/ut -o ${BASEPATH}/cov_dvpp_ut/tmp.info
        lcov -r ${BASEPATH}/cov_dvpp_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*' -o ${BASEPATH}/cov_dvpp_ut/coverage.info
        cd "${BASEPATH}/cov_dvpp_ut/"
        genhtml coverage.info
        echo "---------------- Finish the generating coverage of dvpp ut ----------------"
      fi
    fi

    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run dvpp st ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/dvppeng/ut/"
      ./dvpp_engine_utest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run dvpp st ------------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_dvpp_st/" ];then
          rm -rf "${BASEPATH}/cov_dvpp_st/"
        fi
        mk_dir "${BASEPATH}/cov_dvpp_st/"
        lcov -c -d ${BASEPATH}/build/tests/engines/dvppeng/ut -o ${BASEPATH}/cov_dvpp_st/tmp.info
        lcov -r ${BASEPATH}/cov_dvpp_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*' -o ${BASEPATH}/cov_dvpp_st/coverage.info
        cd "${BASEPATH}/cov_dvpp_st/"
        genhtml coverage.info
      fi
    fi
  fi

  echo "tt===> ready do hcce ut : $ENABLE_HCCE_LLT : $ENABLE_UT : $ENABLE_LLT_COV, path=${BASEPATH}"
  if [ "X$ENABLE_HCCE_LLT" = "Xon" ]
  then
    if [ "X$ENABLE_UT" = "Xon" ]
    then
      echo "---------------- Begin to run hcce ut ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/hccl_engine/ut/"
      ./hccl_engine_utest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run hcce ut ------------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        echo "---------------- Begin to generate coverage of hcce ut ----------------"
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_hcce_ut/" ];then
          rm -rf "${BASEPATH}/cov_hcce_ut/"
        fi
        mk_dir "${BASEPATH}/cov_hcce_ut/"
        lcov -c -d ${BASEPATH}/build/tests/engines/hccl_engine/ut -o ${BASEPATH}/cov_hcce_ut/tmp.info
        lcov -r ${BASEPATH}/cov_hcce_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/local/*' '/usr/include/*' '*/opensdk/*' '*/inc/*' -o ${BASEPATH}/cov_hcce_ut/coverage.info
        cd "${BASEPATH}/cov_hcce_ut/"
        genhtml coverage.info
        echo "---------------- Finish the generating coverage of hcce ut ----------------"
      fi
    fi
 
    if [ "X$ENABLE_ST" = "Xon" ]
    then
      echo "---------------- Begin to run hcce st ----------------"
      unset ASCEND_OPP_PATH
      cd "${BASEPATH}/build/tests/engines/hccl_engine/st/"
      ./hccl_engine_stest
      export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp
      echo "---------------- Finish run hcce st ------------------"
      if [ "X$ENABLE_LLT_COV" = "Xon" ]
      then
        cd "${BASEPATH}"
        if [ -d "${BASEPATH}/cov_hcce_st/" ];then
          rm -rf "${BASEPATH}/cov_hcce_st/"
        fi
        mk_dir "${BASEPATH}/cov_hcce_st/"
        lcov -c -d ${BASEPATH}/build/tests/engines/hccl_engine/st -o ${BASEPATH}/cov_hcce_st/tmp.info
        lcov -r ${BASEPATH}/cov_hcce_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/local/*' '/usr/include/*' '*/opensdk/*' '*/inc/*' -o ${BASEPATH}/cov_hcce_st/coverage.info
        cd "${BASEPATH}/cov_hcce_st/"
        genhtml coverage.info
      fi
    fi
  fi

  cd "${BASEPATH}"
}

generate_package()
{
  cd "${BASEPATH}"

  AIR_LIB_PATH="lib"
  COMPILER_PATH="compiler/lib64"

  OPSKERNEL_PATH="plugin/opskernel"
  OPSKERNEL_FE_CONFIG_PATH="plugin/opskernel/fe_config"
  OPSKERNEL_FFTS_CONFIG_PATH="plugin/opskernel/ffts_config"
  OPSKERNEL_CPU_CONFIG_PATH="plugin/opskernel/config"

  COMMON_LIB=("libaicore_utils.so" "libopskernel.so" "libaicpu_engine_common.so" "libfusion_pass.so" "libop_compile_adapter.so")
  OPSKERNEL_LIB=("libfe.so" "libffts.so" "libaicpu_ascend_engine.so"
                 "libaicpu_tf_engine.so" "libdvpp_engine.so" "librts_engine.so" "libhost_cpu_opskernel_builder.so" "libhost_cpu_engine.so" 
                 "libhcom_graph_adaptor.so" "libhcom_opskernel_builder.so" "libhcom_gradtune_opskernel_builder.so" "libhcom_gradient_split_tune.so" "libhcom_executor.so")
  PLUGIN_OPSKERNEL_FE=("fe.ini")
  PLUGIN_OPSKERNEL_FFTS=("ffts.ini")
  PLUGIN_OPSKERNEL_CPU=("init.conf" "ir2tf_op_mapping_lib.json" "aicpu_ops_parallel_rule.json")

  rm -rf ${OUTPUT_PATH:?}/${COMPILER_PATH}/
  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_FE_CONFIG_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_FFTS_CONFIG_PATH}"
  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_CPU_CONFIG_PATH}"

  cd "${OUTPUT_PATH}"

  find ./ -name air.tar -exec rm {} \;

  MAX_DEPTH=1

  for lib in "${COMMON_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${AIR_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  done

  for lib in "${OPSKERNEL_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${AIR_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_PATH} \;
  done

  for lib in "${PLUGIN_OPSKERNEL_CPU[@]}";
  do
    find ${OUTPUT_PATH}/${AIR_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_CPU_CONFIG_PATH} \;
  done

  for lib in "${PLUGIN_OPSKERNEL_FE[@]}";
  do
    find ${OUTPUT_PATH}/${AIR_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_FE_CONFIG_PATH} \;
  done

  for lib in "${PLUGIN_OPSKERNEL_FFTS[@]}";
  do
    find ${OUTPUT_PATH}/${AIR_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH}/${OPSKERNEL_FFTS_CONFIG_PATH} \;
  done
  # generate tar package
  tar -zcf air.tar compiler
}

main() {
  # AIR build start
  echo "---------------- AIR build start ----------------"
  checkopts "$@"

  env
  g++ -v
  ###
  export LD_LIBRARY_PATH=$ASCEND_INSTALL_PATH/compiler/lib64:$ASCEND_INSTALL_PATH/runtime/lib64:$ASCEND_INSTALL_PATH/runtime/lib64/stub:$LD_LIBRARY_PATH
  mk_dir ${OUTPUT_PATH}
  build_air || { echo "AIR build failed."; exit 1; }
  echo "---------------- AIR build finished ----------------"

  if [[ "X$ENABLE_FE_LLT" = "Xoff" ]] && [[ "X$ENABLE_TEFUSION_LLT" = "Xoff" ]] && [[ "X$ENABLE_AICPU_LLT" = "Xoff" ]] && [[ "X$ENABLE_DVPP_LLT" = "Xoff" ]]  && [[ "X$ENABLE_RTS_LLT" = "Xoff" ]] && [[ "X$ENABLE_FFTS_LLT" = "Xoff" ]] && [[ "X$ENABLE_HCCE_LLT" = "Xoff" ]]; then
    generate_package
  else
    echo "....---> beforerun_llt_with_cov"
    run_llt_with_cov
  fi
  echo "---------------- AIR package archive generated ----------------"
}

main "$@"
