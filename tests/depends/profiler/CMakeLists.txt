# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(profiler_stub SHARED src/profiler_stub.cc src/dump_stub.cc)
target_link_libraries(profiler_stub
    PRIVATE
        intf_pub
        runtime_headers
        c_sec
    PUBLIC
        msprof_headers
        adump_headers
)
target_include_directories(profiler_stub SYSTEM PRIVATE
        ${AIR_CODE_DIR}/inc/framework
        ${METADEF_DIR}/inc/external
        )
set_target_properties(profiler_stub PROPERTIES LIBRARY_OUTPUT_NAME profapi)

stub_module(profapi_share profiler_stub)
stub_module(msprofiler_fwk_share profiler_stub)
stub_module(ascend_dump profiler_stub)
stub_module(adump_server profiler_stub)
stub_module(adcore profiler_stub)
