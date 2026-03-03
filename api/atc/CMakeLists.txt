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
        "omg.cc"
        "main_impl.cc"
        "cmd_flag_info.cc"
        )

set(COMPILE_OPTIONS
        -Werror
        -Wextra
        -Wfloat-equal
        -O2
        -fno-common
        -fvisibility=hidden
        )

set(COMPILE_DEFINITIONS
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        LOG_CPP
        FUNC_VISIBILITY
        )

set(CMD_FLAG_SRC_LIST "cmd_flag_info.cc"
		)
add_library(cmd_flag_info_static ${CMD_FLAG_SRC_LIST})

target_compile_options(cmd_flag_info_static PRIVATE ${COMPILE_OPTIONS})
target_compile_definitions(cmd_flag_info_static PRIVATE ${COMPILE_DEFINITIONS})


target_include_directories(cmd_flag_info_static PUBLIC
		${CMAKE_CURRENT_LIST_DIR}
		${AIR_CODE_DIR}
		${AIR_CODE_DIR}/base
		${AIR_CODE_DIR}/compiler
		${AIR_CODE_DIR}/inc
		${AIR_CODE_DIR}/inc/external
		${AIR_CODE_DIR}/inc/framework
		${AIR_CODE_DIR}/parser
		${AIR_CODE_DIR}/inc/parser
		${AIR_CODE_DIR}/inc/parser/external/
		${CMAKE_BINARY_DIR}
		)

target_link_options(cmd_flag_info_static PRIVATE
		-Wl,-Bsymbolic
		-Wl,-rpath-link,${ASCEND_COMPILER_DIR}/stub
		-Wl,-rpath-link,${ASCEND_RUNTIME_DIR}/stub
		)

target_link_libraries(cmd_flag_info_static PRIVATE
		intf_pub
		c_sec
		error_manager
		slog
		mmpa_headers
		static_mmpa
		metadef_headers
		-lrt
		-ldl
		)
############ atc_static ############
add_library(atc_static ${SRC_LIST})

add_dependencies(atc_static
        graphengine_protos
        )

target_compile_options(atc_static PRIVATE ${COMPILE_OPTIONS})
target_compile_definitions(atc_static PRIVATE ${COMPILE_DEFINITIONS})


target_include_directories(atc_static PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/parser
        ${AIR_CODE_DIR}/inc/parser
        ${AIR_CODE_DIR}/inc/parser/external/
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${ASCGEN_DIR}/inc
		${METADEF_DIR}/pkg_inc
        )

target_link_options(atc_static PRIVATE
        -Wl,-Bsymbolic
        -Wl,-rpath-link,${ASCEND_COMPILER_DIR}/stub
        -Wl,-rpath-link,${ASCEND_RUNTIME_DIR}/stub
        )

target_link_libraries(atc_static PRIVATE
        intf_pub
        cce_headers
        msprof_headers
        metadef_headers
        ascend_protobuf
        ge_common
        ge_common_base
        davinci_executor
        gert
        exe_graph
        lowering
        register
        c_sec
        graph
        graph_base
        metadef
        error_manager
        ge_compiler
        ascend_hal_stub
        parser_common
        json
        runtime
        slog
	mmpa
        -lrt
        -ldl
        )

############ atc_atc.bin ############
add_executable(atc_atc.bin main.cc)

target_compile_options(atc_atc.bin PRIVATE ${COMPILE_OPTIONS})

target_compile_definitions(atc_atc.bin PRIVATE ${COMPILE_DEFINITIONS})

target_link_options(atc_atc.bin PRIVATE
        -Wl,-Bsymbolic
        -Wl,-rpath-link,${ASCEND_COMPILER_DIR}/stub
        -Wl,-rpath-link,${ASCEND_RUNTIME_DIR}/stub
        )

target_link_libraries(atc_atc.bin PRIVATE
        intf_pub
        atc_static
        )

set_target_properties(atc_atc.bin PROPERTIES
        OUTPUT_NAME atc.bin
        RUNTIME_OUTPUT_DIRECTORY atclib
        )

############ fwk_atc.bin ############
add_executable(fwk_atc.bin main.cc)

target_compile_options(fwk_atc.bin PRIVATE ${COMPILE_OPTIONS})

target_compile_definitions(fwk_atc.bin PRIVATE ${COMPILE_DEFINITIONS})

target_link_options(fwk_atc.bin PRIVATE
        -Wl,-Bsymbolic
        -Wl,-rpath-link,${ASCEND_COMPILER_DIR}/stub
        -Wl,-rpath-link,${ASCEND_RUNTIME_DIR}/stub
        )

target_link_libraries(fwk_atc.bin PRIVATE
        intf_pub
        atc_static
        )

set_target_properties(fwk_atc.bin PROPERTIES
        OUTPUT_NAME atc.bin
        RUNTIME_OUTPUT_DIRECTORY fwkacl
        )

install(TARGETS atc_atc.bin OPTIONAL
        RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}/atclib
        )

install(TARGETS fwk_atc.bin OPTIONAL
        RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}/fwkacl
        )
