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

# 开发环境构建代码适配
set -e

TOP_DIR="$1"
FARM_LAND=`echo $2 | awk -F\@ '{print "vendor/"$1"/"$2" project:"$3}'`
METADEF_DIR="${TOP_DIR}/metadef"
GE_METADEF_DIR="${TOP_DIR}/air/graph_metadef"

if [ -d ${METADEF_DIR} ]; then
  if [ ! -d ${METADEF_DIR}/proto ];then
    if [ ! -d ${GE_METADEF_DIR} ]; then
      echo "ERROR: Yellow zone need add air/graph_metadef to ${FARM_LAND}"
    fi

    if [ ! -d ${TOP_DIR}/air/inc/graph_metadef ]; then
      echo "ERROR: Yellow zone need add air/inc/graph_metadef to ${FARM_LAND}"
    else
      if [ ! -d ${METADEF_DIR}/inc/include/register ]; then
        mkdir -p ${METADEF_DIR}/inc/include/register
      fi
      cp -rf ${METADEF_DIR}/inc/external/register/register.h ${METADEF_DIR}/inc/include/register/register.h
    fi
  fi
fi
