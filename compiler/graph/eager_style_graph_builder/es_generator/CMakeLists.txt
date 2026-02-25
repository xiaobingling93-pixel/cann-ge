# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_library(es_generator_inner_headers INTERFACE)
target_include_directories(es_generator_inner_headers INTERFACE
        ${AIR_CODE_DIR}/compiler/graph/eager_style_graph_builder/es_generator
        )

add_library(geir_collector STATIC
        ge_ir_collector.cc
        )

target_compile_options(geir_collector PRIVATE ${AIR_COMMON_COMPILE_OPTION})

target_link_libraries(geir_collector PRIVATE
        intf_pub
        graph
        graph_base
        metadef
        error_manager
        c_sec
        slog_headers
        mmpa_headers
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        )

add_library(gen_esb_impl STATIC
        gen_esb.cc
        utils.cc
        generator_manager.cc
        history/history_registry_interface.cc
        history/history_registry_reader.cc
        history/history_registry_writer.cc
        history/history_registry_utils.cc
        history/ir_proto_codec.cc
        )

target_compile_options(gen_esb_impl PRIVATE ${AIR_COMMON_COMPILE_OPTION})
target_link_libraries(gen_esb_impl PRIVATE
        intf_pub
        graph
        graph_base
        error_manager
        geir_collector
        json
        c_sec
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        )
add_executable(gen_esb main.cc)
target_compile_options(gen_esb PRIVATE ${AIR_COMMON_COMPILE_OPTION})
target_link_libraries(gen_esb PRIVATE c_sec metadef  intf_pub cmd_flag_info_static gen_esb_impl json)
install(TARGETS gen_esb OPTIONAL
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR})
