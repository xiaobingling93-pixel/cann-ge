#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

function build()
{
  UserKernel=`arch`
  if [[ ${TargetKernel} = "x86" ]] || [[ ${TargetKernel} = "X86" ]];then
    TargetCompiler="g++"
    TargetKernel="x86"
  else
    if [[ ${UserKernel} == "x86_64" ]];then
      TargetCompiler="aarch64-linux-gnu-g++"
      TargetKernel="arm"
    else
      TargetCompiler="g++"
      TargetKernel="arm"
    fi
  fi
  if [ -d ${ScriptPath}/../build/intermediates/host ];then
    rm -rf ${ScriptPath}/../build/intermediates/host
  fi
    
  mkdir -p ${ScriptPath}/../build/intermediates/host
  cd ${ScriptPath}/../build/intermediates/host

  # Start compiling
  cmake ../../../src -DCMAKE_CXX_COMPILER=${TargetCompiler} -DCMAKE_SKIP_RPATH=TRUE
  if [ $? -ne 0 ];then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi
  make
  if [ $? -ne 0 ];then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi
  cd - > /dev/null
}

function target_kernel()
{
  declare -i CHOICE_TIMES=0
  while [[ ${TargetKernel}"X" = "X" ]]
  do
    # three times choice 
    [[ ${CHOICE_TIMES} -ge 3 ]] && break || ((CHOICE_TIMES++))
    read -p "please input TargetKernel? [arm/x86]:" TargetKernel
    if [ ${TargetKernel}"z" = "armz" ] || [ ${TargetKernel}"z" = "Armz" ] || [ ${TargetKernel}"z" = "x86z" ] || [ ${TargetKernel}"z" = "X86z" ]; then
      echo "[INFO] input is normal, start preparation."
    else
      echo "[WARNING] The ${CHOICE_TIMES}th parameter input error!"
      TargetKernel=""
    fi
  done
  if [ ${TargetKernel}"z" = "z" ];then
    echo "[ERROR] TargetKernel entered incorrectly three times, please input arm/x86!"
    return 1
  else
    return 0
  fi
}
function main()
{
  echo "[INFO] Sample preparation"

  target_kernel
  if [ $? -ne 0 ];then
    return 1
  fi
    
  build
  if [ $? -ne 0 ];then
    return 1
  fi
    
  echo "[INFO] Sample preparation is complete"
}
main

