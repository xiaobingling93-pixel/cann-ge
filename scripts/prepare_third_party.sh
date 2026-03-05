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

THIRD_PARTY_DIR=$1

if [ ! -d "${THIRD_PARTY_DIR}" ]; then
    echo "[Third part prepare] info: third part path ${THIRD_PARTY_DIR} not found."
    exit 1
fi

cd "${THIRD_PARTY_DIR}"

# process_file $file_name $dir_name
process_file() {
    local tar_file=$1
    local dir_name=$2

    if [ ! -f "${tar_file}" ]; then
        echo "[Third part prepare] info: ${tar_file} not exist, skip."
        return
    fi

    mkdir -p "${dir_name}"
    mv "${tar_file}" "${dir_name}/"
    echo "[Third part prepare] info: ${tar_file} mv to ${dir_name}, success."
}

process_file "protobuf-25.1.tar.gz" "protobuf"
process_file "boost_1_87_0.tar.gz" "boost"
process_file "abseil-cpp-20230802.1.tar.gz" "abseil-cpp"
process_file "c-ares-1.19.1.tar.gz" "c-ares"
process_file "benchmark-1.8.3.tar.gz" "benchmark"
process_file "grpc-1.60.0.tar.gz" "grpc"
process_file "googletest-1.14.0.tar.gz" "gtest_shared"
process_file "json-3.11.3.tar.gz" "json"
process_file "openssl-openssl-3.0.9.tar.gz" "openssl"
process_file "re2-2024-02-01.tar.gz" "re2"
process_file "symengine-0.12.0.tar.gz" "symengine"
process_file "zlib-1.2.13.tar.gz" "zlib"
process_file "makeself-release-2.5.0-patch1.tar.gz" "makeself"
process_file "mockcpp-2.7.tar.gz" "mockcpp-2.7"
process_file "libseccomp-2.5.4.tar.gz" "libseccomp-2.5.4"

echo "[Third part prepare] finish."
exit 0