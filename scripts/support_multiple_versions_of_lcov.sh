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

get_lcov_major_version() {
    local major_version
    if ! major_version=$(set -o pipefail; lcov --version | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)*' | head -1 | cut -d. -f1); then
        echo "Error: Failed to parse LCOV major version number, please check 'lcov --version'." >&2
        exit 1
    fi
    echo "$major_version"
}

add_lcov_ops_by_major_version() {
    local expected_major_version="$1"
    local ops_to_be_added="$2"
    if [ "$(get_lcov_major_version)" -ge $expected_major_version ]; then
        echo "$ops_to_be_added"
    fi   
}
