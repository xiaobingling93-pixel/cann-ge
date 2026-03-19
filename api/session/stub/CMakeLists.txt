# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
##################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_ir_build.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api_v2.cc
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/gen_stubapi.py ${AIR_CODE_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && mv ge_ir_build.cc stub_ge_ir_build.cc
            && mv ge_api.cc stub_ge_api.cc
            && mv ge_api_v2.cc stub_ge_api_v2.cc
            &&  echo "Generating stub files end."
)

add_custom_target(ge_stub
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_ir_build.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api.cc
)

##################################################################
############ stub/libge_compiler.so ############
add_library(atc_stub_ge_compiler SHARED
    stub_ge_ir_build.cc
)

add_dependencies(atc_stub_ge_compiler ge_stub)

target_compile_options(atc_stub_ge_compiler PRIVATE
    -fno-common
    -Werror=return-type
    -Wno-unused-parameter
)

target_link_libraries(atc_stub_ge_compiler PRIVATE
    intf_pub
    metadef_headers
    eager_style_graph_builder_base_headers
)

set_target_properties(atc_stub_ge_compiler PROPERTIES
    OUTPUT_NAME ge_compiler
    LIBRARY_OUTPUT_DIRECTORY atc_stub
    RUNTIME_OUTPUT_DIRECTORY atc_stub
)

target_include_directories(atc_stub_ge_compiler PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/inc/external
)

############ stub/libge_runner_v2.so ############
add_library(fwk_stub_ge_runner_v2 SHARED
    stub_ge_api_v2.cc
    stub_ge_ir_build.cc
)

add_dependencies(fwk_stub_ge_runner_v2 ge_stub)

target_compile_options(fwk_stub_ge_runner_v2 PRIVATE
    -fno-common
    -Werror=return-type
    -Wno-unused-parameter
)


target_link_libraries(fwk_stub_ge_runner_v2
    PRIVATE
        intf_pub
        metadef_headers
        eager_style_graph_builder_base_headers
    PUBLIC
        air_headers
)

set_target_properties(fwk_stub_ge_runner_v2 PROPERTIES
    OUTPUT_NAME ge_runner_v2
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
    RUNTIME_OUTPUT_DIRECTORY fwk_stub
)

target_include_directories(fwk_stub_ge_runner_v2 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/inc/external
)

############ stub/libge_runner.so ############
add_library(fwk_stub_ge_runner SHARED
    stub_ge_api.cc
    stub_ge_ir_build.cc
)

add_dependencies(fwk_stub_ge_runner ge_stub)

target_compile_options(fwk_stub_ge_runner PRIVATE
    -fno-common
    -Werror=return-type
    -Wno-unused-parameter
)


target_link_libraries(fwk_stub_ge_runner
    PRIVATE
        intf_pub
        metadef_headers
        eager_style_graph_builder_base_headers
    PUBLIC
        air_headers
)

set_target_properties(fwk_stub_ge_runner PROPERTIES
    OUTPUT_NAME ge_runner
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
    RUNTIME_OUTPUT_DIRECTORY fwk_stub
)

target_include_directories(fwk_stub_ge_runner PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/inc/external
)

###############################################################
############ install ############
install(TARGETS atc_stub_ge_compiler
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub OPTIONAL
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}/stub OPTIONAL
)

endif()
