# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE OPIMPL_SRCS CONFIGURE_DEPENDS "op_impl/*.cc")
add_library(gert_op_impl SHARED ${OPIMPL_SRCS})
target_include_directories(gert_op_impl PRIVATE
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${CMAKE_BINARY_DIR}/proto/gert_proto
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${ASCEND_INSTALL_PATH}/include/toolchain
        )
target_compile_definitions(gert_op_impl PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        )
target_link_libraries(gert_op_impl
        PRIVATE
            intf_pub
            metadef_headers
            slog_headers
            ${AIR_COMMON_LINK_OPTION}
         #   metadef_graph
            c_sec_headers
            c_sec
            json
            ascend_protobuf
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            gert_op_impl2  #本不需要该库文件，这里是为了触发gert_op_impl2动态库的编译
            autofuse_stub
            slice_autofuse_stub
        )
target_compile_options(gert_op_impl PRIVATE ${AIR_COMMON_COMPILE_OPTION})

file(GLOB_RECURSE OPIMPL_SRCS2 CONFIGURE_DEPENDS "op_impl/*.cc")
list(APPEND OPIMPL_SRCS "op_impl2/add_op_impl2.cc")
add_library(gert_op_impl2 SHARED ${OPIMPL_SRCS})
target_include_directories(gert_op_impl2 PRIVATE
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${CMAKE_BINARY_DIR}/proto/gert_proto
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${ASCEND_INSTALL_PATH}/include/toolchain
        )
target_compile_definitions(gert_op_impl2 PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        )
target_link_libraries(gert_op_impl2
        PRIVATE
        intf_pub
        metadef_headers
        slog_headers
        ${AIR_COMMON_LINK_OPTION}
        c_sec_headers
        c_sec
        json
        ascend_protobuf
        gert
        -Wl,--whole-archive
        rt2_registry_static
        -Wl,--no-whole-archive
        ascendcl_stub
        )
target_compile_options(gert_op_impl2 PRIVATE ${AIR_COMMON_COMPILE_OPTION})

file(GLOB_RECURSE AUTOFUSE_SRC CONFIGURE_DEPENDS "autofuse_impl/*.cc")
add_library(autofuse_stub SHARED ${AUTOFUSE_SRC})
target_include_directories(autofuse_stub PRIVATE
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
        ${ASCEND_INSTALL_PATH}/include
)
target_compile_definitions(autofuse_stub PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)
target_link_libraries(autofuse_stub
        PRIVATE
        intf_pub
        metadef_headers
        ${AIR_COMMON_LINK_OPTION}
        #   metadef_graph
        c_sec_headers
        c_sec
        json
)
target_compile_options(autofuse_stub PRIVATE ${AIR_COMMON_COMPILE_OPTION})

file(GLOB_RECURSE SLICE_AUTOFUSE_SRC CONFIGURE_DEPENDS "slice_autofuse_impl/*.cc")
add_library(slice_autofuse_stub SHARED ${SLICE_AUTOFUSE_SRC})
target_include_directories(slice_autofuse_stub PRIVATE
        ${AIR_CODE_DIR}/inc/external
)
target_compile_definitions(slice_autofuse_stub PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)
target_link_libraries(slice_autofuse_stub
        PRIVATE
        intf_pub
        metadef_headers
        ${AIR_COMMON_LINK_OPTION}
        #   metadef_graph
        c_sec_headers
        c_sec
        json
)
target_compile_options(slice_autofuse_stub PRIVATE ${AIR_COMMON_COMPILE_OPTION})

file(GLOB_RECURSE STATIC_OPIMPL_SRCS CONFIGURE_DEPENDS "less_important_op_impl.cc" "data_flow_op_impl.cc" "autofuse_impl/*.cc")
add_library(gert_op_impl_static STATIC ${STATIC_OPIMPL_SRCS})
target_include_directories(gert_op_impl_static PRIVATE
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${CMAKE_BINARY_DIR}/proto/gert_proto
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${ASCEND_INSTALL_PATH}/include/toolchain
        )
target_compile_definitions(gert_op_impl_static PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        )
target_link_libraries(gert_op_impl_static
        PRIVATE
        intf_pub
        metadef_headers
        slog_headers
        ${AIR_COMMON_LINK_OPTION}
        c_sec_headers
        c_sec
        json
        ascend_protobuf
        -Wl,--whole-archive
        rt2_registry_static
        -Wl,--no-whole-archive
        )
target_compile_options(gert_op_impl_static PRIVATE ${AIR_COMMON_COMPILE_OPTION})

add_subdirectory(so_stub/lib64/device/lib64)

#本不需要该库文件，这里是为了触发gert_op_impl2动态库的编译
add_dependencies(gert_op_impl_static gert_op_impl2 autofuse_stub slice_autofuse_stub)