# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(LX_FUSION_SRC_CPP
        src/lx_fusion_stub.cc)

# compile for fe_st
add_library(graph_tuner_stub SHARED
        ${LX_FUSION_SRC_CPP})

# add it for slog compile at 12.16
target_compile_definitions(graph_tuner_stub PRIVATE
        LOG_CPP
        )

target_compile_options(graph_tuner_stub PRIVATE
        -Dgoogle=ascend_private)

target_link_libraries(graph_tuner_stub
    PRIVATE
        intf_llt_pub
        msprof_headers
        metadef_headers
        air_headers
        runtime_headers
        c_sec
        graph)

install(TARGETS graph_tuner_stub OPTIONAL
        EXPORT graph_tuner_stub-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})
