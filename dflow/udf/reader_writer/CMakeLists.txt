# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
###for rader_writer so
set(reader_writer_files
    queue_wrapper.cpp
    proxy_queue_wrapper.cpp
    mbuf_reader.cpp
    mbuf_writer.cpp
    data_aligner.cpp
)
##for so inc path
set(reader_writer_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/external
)

add_library(reader_writer SHARED
    ${reader_writer_files}
)
target_include_directories(reader_writer PRIVATE
    ${reader_writer_inc_path}
)

target_compile_options(reader_writer PRIVATE
    -ftrapv
    -O2
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_link_libraries(reader_writer PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    PUBLIC c_sec
    ascend_hal_stub
    $<IF:$<STREQUAL:${PRODUCT_SIDE},device>,alog,unified_dlog>
)

set(INSTALL_LIBRARY_DIR lib)

install(TARGETS reader_writer OPTIONAL
    EXPORT reader_writer-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}
)
