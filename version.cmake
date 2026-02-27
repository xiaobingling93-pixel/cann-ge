# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_package(ge-executor VERSION "9.0.0")

set_build_dependencies(runtime ">=9.0")
set_build_dependencies(metadef ">=9.0")
set_build_dependencies(hcomm ">=9.0")
set_build_dependencies(asc-devkit ">=9.0")

set_run_dependencies(runtime "CUR_MAJOR_MINOR_VER")
set_run_dependencies(metadef "CUR_MAJOR_MINOR_VER")
set_run_dependencies(hcomm "CUR_MAJOR_MINOR_VER")
set_run_dependencies(asc-devkit "CUR_MAJOR_MINOR_VER")

set_package(ge-compiler VERSION "9.0.0")

set_build_dependencies(runtime ">=9.0")
set_build_dependencies(metadef ">=9.0")
set_build_dependencies(hcomm ">=9.0")
set_build_dependencies(asc-devkit ">=9.0")
set_build_dependencies(ge-executor ">=9.0")

set_run_dependencies(runtime "CUR_MAJOR_MINOR_VER")
set_run_dependencies(metadef "CUR_MAJOR_MINOR_VER")
set_run_dependencies(hcomm "CUR_MAJOR_MINOR_VER")
set_run_dependencies(asc-devkit "CUR_MAJOR_MINOR_VER")
set_run_dependencies(ge-executor "CUR_MAJOR_MINOR_VER")

set_package(dflow-executor VERSION "9.0.0")

set_build_dependencies(runtime ">=9.0")
set_build_dependencies(metadef ">=9.0")
set_build_dependencies(hcomm ">=9.0")
set_build_dependencies(bisheng-compiler ">=9.0")
set_build_dependencies(ge-executor ">=9.0")

set_run_dependencies(runtime "CUR_MAJOR_MINOR_VER")
set_run_dependencies(metadef "CUR_MAJOR_MINOR_VER")
set_run_dependencies(hcomm "CUR_MAJOR_MINOR_VER")
set_run_dependencies(bisheng-compiler "CUR_MAJOR_MINOR_VER")
set_run_dependencies(ge-executor "CUR_MAJOR_MINOR_VER")
