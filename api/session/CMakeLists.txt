# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_subdirectory(jit_execution)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
    message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")

set(RUNNER_SRC_LIST_V2
    "client/ge_api_v2.cc"
    "session/ge_session_impl.cc"
    "session/inner_session.cc"
    "session/user_hybrid_graph_manager.cc"
    "common/plugin/tbe_plugin_manager.cc"
)

############ libge_runner_v2.so ############
add_library(ge_runner_v2 SHARED
    ${RUNNER_SRC_LIST_V2}
)

target_compile_definitions(ge_runner_v2 PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    DAVINCI_CLOUD
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_runner_v2 PRIVATE
    -O2
    -fno-common
    -Werror
    -Wextra
    -Wfloat-equal
    -fvisibility=default
)

target_include_directories(ge_runner_v2 SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/runtime/v1
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl/error_codes
    )

target_link_options(ge_runner_v2 PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_runner_v2
    PRIVATE
        intf_pub
        cce_headers
        ge_compiler
        ge_executor_shared
        davinci_executor
        hybrid_executor
        gert
        -Wl,--as-needed
        graph
        graph_base
        ge_common
        ge_common_base
        register
        opp_registry
        runtime
        error_manager
        ascend_hal_stub
        msprofiler_fwk_share
        profapi_share
        adump_server
        $<$<BOOL:${ENABLE_OPEN_SRC}>:adcore>
        static_mmpa
        unified_dlog
        c_sec
        ascend_protobuf
        json
        jit_exe
        -Wl,--no-as-needed
        -lrt
        -ldl
    PUBLIC
        air_headers
)

set(RUNNER_SRC_LIST
    "client/ge_api.cc"
    "session/inner_session.cc"
    "session/session_manager.cc"
    "session/user_hybrid_graph_manager.cc"
    "common/plugin/tbe_plugin_manager.cc"
)

############ libge_runner.so ############
add_library(ge_runner SHARED
    ${RUNNER_SRC_LIST}
)

target_compile_definitions(ge_runner PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    DAVINCI_CLOUD
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_runner PRIVATE
    -O2
    -fno-common
    -Werror
    -Wextra
    -Wfloat-equal
    -fvisibility=default
)

target_include_directories(ge_runner SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/runtime/v1
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl/error_codes
    )

target_link_options(ge_runner PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_runner
    PRIVATE
        intf_pub
        cce_headers
        ge_compiler
        ge_executor_shared
        davinci_executor
        hybrid_executor
        gert
        -Wl,--as-needed
        graph
        graph_base
        ge_common
        ge_common_base
        register
        opp_registry
        runtime
        dflow_runner
        error_manager
        ascend_hal_stub
        msprofiler_fwk_share
        profapi_share
        adump_server
        $<$<BOOL:${ENABLE_OPEN_SRC}>:adcore>
        static_mmpa
        unified_dlog
        c_sec
        ascend_protobuf
        json
        jit_exe
	ge_runner_v2
        -Wl,--no-as-needed
        -lrt
        -ldl
    PUBLIC
        air_headers
)

############ install ############
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)
endif()

ADD_SUBDIRECTORY(stub)
