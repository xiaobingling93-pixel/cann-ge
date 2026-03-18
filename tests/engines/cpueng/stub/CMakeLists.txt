# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(proto_src_files
    ${METADEF_PROTO_DIR}/task.proto
)

protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${proto_src_files})

set(engine_common_stub_files
    ${CMAKE_CURRENT_SOURCE_DIR}/platform_info_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/util_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu/aicpu_stub.cpp
    ${PROTO_HDRS}
)

set(engine_common_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
    ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
    ${CMAKE_BINARY_DIR}/proto/aicpu
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
)
add_subdirectory(slog)
add_subdirectory(runtime)

add_library(stub_for_aicpu SHARED
    ${engine_common_stub_files}
)
add_dependencies(stub_for_aicpu GTest::gtest)
target_include_directories(stub_for_aicpu PRIVATE
    ${engine_common_includes}
)

target_compile_definitions(stub_for_aicpu PRIVATE
    RUN_TEST
)

target_compile_options(stub_for_aicpu PRIVATE
    -fPIC
)

target_link_libraries(stub_for_aicpu PRIVATE
    intf_llt_pub
    metadef_headers
    air_headers
    msprof_headers
    c_sec_headers
    slog_aicpu_stub
    runtime_aicpu_stub
    ascend_protobuf
    json
    aicpu_headers
    cce_headers
)

set(COPY_TO_PATH ${AIR_CODE_DIR}/out/onetrack/llt/ut/obj/lib/)
add_custom_command(TARGET stub_for_aicpu POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E make_directory ${COPY_TO_PATH}
                  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:stub_for_aicpu> ${COPY_TO_PATH}
                  COMMENT "[AICPU] copying stub_for_aicpu library to '${COPY_TO_PATH}'"
)

install(TARGETS stub_for_aicpu OPTIONAL
    EXPORT stub_for_aicpu-targets
    LIBRARY DESTINATION ${COPY_TO_PATH}
)
