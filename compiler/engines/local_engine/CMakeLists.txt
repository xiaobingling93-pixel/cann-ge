# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SRC_LIST
    "engine/ge_local_engine.cc"
    "engine/ge_local_graph_optimizer.cc"
    "ops_kernel_store/ge_local_ops_kernel_info_store.cc"
    "ops_kernel_store/op/op_factory.cc"
    "ops_kernel_store/op/op.cc"
    "ops_kernel_store/op/ge_deleted_op.cc"
    "ops_kernel_store/op/no_op.cc"
)

set(OPS_KERNEL_SRC_LIST
    "ops_kernel_store/ge_local_ops_kernel_builder.cc"
    "ops_kernel_store/ge_local_ops_kernel_calc_op_param.cc"
    "ops_kernel_store/op/op_factory.cc"
    "ops_kernel_store/op/op.cc"
    "ops_kernel_store/op/ge_deleted_op.cc"
    "ops_kernel_store/op/no_op.cc"
)

############ libge_local_engine.so ############
add_library(ge_local_engine SHARED ${SRC_LIST})

add_dependencies(ge_local_engine
    graphengine_protos
)

target_compile_options(ge_local_engine PRIVATE
    -Werror
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=hidden
)

target_compile_definitions(ge_local_engine PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(ge_local_engine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_local_engine PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_local_engine PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    -Wl,--no-as-needed
    graph
    graph_base
    ascend_protobuf
    c_sec
    unified_dlog
    -Wl,--as-needed
)

######### atclib/libge_local_engine.so #############
add_library(atc_ge_local_engine SHARED ${SRC_LIST})

add_dependencies(atc_ge_local_engine
    graphengine_protos
)

target_compile_options(atc_ge_local_engine PRIVATE
    -Werror
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=hidden
)

target_compile_definitions(atc_ge_local_engine PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(atc_ge_local_engine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(atc_ge_local_engine PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(atc_ge_local_engine PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    -Wl,--no-as-needed
    graph
    graph_base
    ascend_protobuf
    c_sec
    unified_dlog
    -Wl,--as-needed
)

set_target_properties(atc_ge_local_engine PROPERTIES
    OUTPUT_NAME ge_local_engine
    LIBRARY_OUTPUT_DIRECTORY atclib
)

############ libge_local_opskernel_builder.so ############
add_library(ge_local_opskernel_builder SHARED ${OPS_KERNEL_SRC_LIST})

add_dependencies(ge_local_opskernel_builder
    graphengine_protos
)

target_compile_options(ge_local_opskernel_builder PRIVATE
    -Werror
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=hidden
)

target_compile_definitions(ge_local_opskernel_builder PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(ge_local_opskernel_builder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(ge_local_opskernel_builder PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_local_opskernel_builder PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    unified_dlog
    register
    graph
    graph_base
    ge_common_base
    -Wl,--as-needed
)

############ atclib/libge_local_opskernel_builder.so ############
add_library(atc_ge_local_opskernel_builder SHARED ${OPS_KERNEL_SRC_LIST})

add_dependencies(atc_ge_local_opskernel_builder
    graphengine_protos
)

target_compile_options(atc_ge_local_opskernel_builder PRIVATE
    -Werror
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=hidden
)

target_compile_definitions(atc_ge_local_opskernel_builder PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(atc_ge_local_opskernel_builder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_options(atc_ge_local_opskernel_builder PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(atc_ge_local_opskernel_builder PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    unified_dlog
    register
    graph
    graph_base
    -Wl,--as-needed
)

set_target_properties(atc_ge_local_opskernel_builder PROPERTIES
    OUTPUT_NAME ge_local_opskernel_builder
    LIBRARY_OUTPUT_DIRECTORY atclib
)

############ libge_local_opskernel_builder.a ############
add_library(ge_local_opskernel_builder_static STATIC ${OPS_KERNEL_SRC_LIST})

add_dependencies(ge_local_opskernel_builder_static
    graphengine_protos
)

target_compile_options(ge_local_opskernel_builder_static PRIVATE
    -Werror
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=hidden
)

target_compile_definitions(ge_local_opskernel_builder_static PRIVATE
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
)

target_include_directories(ge_local_opskernel_builder_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_link_libraries(ge_local_opskernel_builder_static PRIVATE
    intf_pub
    metadef_headers
    ascend_protobuf
    c_sec
)

install(TARGETS ge_local_engine ge_local_opskernel_builder OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS atc_ge_local_engine atc_ge_local_opskernel_builder OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/atclib
)
