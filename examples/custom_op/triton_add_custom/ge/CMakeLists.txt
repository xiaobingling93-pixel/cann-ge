cmake_minimum_required(VERSION 3.5.1)
project(custom_op_sample)

set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})

set(INCLUDE_DIR ${ASCEND_PATH}/include)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
        -std=c++17
        -Wall
)

set(common_compile_definitions
        _GLIBCXX_USE_CXX11_ABI=0
)
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

############ libcust_opapi.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

# 固定.so名字
set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME cust_opapi
)
target_compile_options(${PROJECT_NAME} PRIVATE
        ${common_compile_options}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
        ${common_compile_definitions}
)


target_include_directories(${PROJECT_NAME} PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/graph
        ${INCLUDE_DIR}/register
        ${INCLUDE_DIR}/external
)

target_link_options(${PROJECT_NAME} PRIVATE
)

target_link_directories(${PROJECT_NAME} PRIVATE
        ${ASCEND_PATH}/lib64/
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        -Wl,--no-as-needed
        ascendcl
        lowering
        -Wl,--as-needed
)


############ 生成npu_supported_ops.json ############
# 输入
# TARGET:
# OPS_INFO_DIR: 源代码所在目录
# OUT_DIR: 输出目录, 当前为 ${CMAKE_SOURCE_DIR}/build_out

function(add_npu_support_target)
    cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR" "" ${ARGN})

    set(IN_DIR ${NPUSUP_OPS_INFO_DIR})
    set(OUT_DIR ${NPUSUP_OUT_DIR}/framework/tensorflow)

    file(GLOB_RECURSE CODE_FILES "${IN_DIR}/*")

    add_custom_command(
            OUTPUT ${OUT_DIR}/npu_supported_ops.json
            COMMAND mkdir -p ${OUT_DIR}
            COMMAND bash ${CMAKE_SOURCE_DIR}/gen_npu_supported_ops_json.sh ${IN_DIR} ${OUT_DIR}
            DEPENDS ${CODE_FILES} ${CMAKE_SOURCE_DIR}/gen_npu_supported_ops_json.sh
            COMMENT "Generating NPU supported ops JSON from ${IN_DIR}"
    )

    add_custom_target(${NPUSUP_TARGET} ALL
            DEPENDS ${OUT_DIR}/npu_supported_ops.json
    )
endfunction()

set(CUSTOM_INSTALL_PATH ${CMAKE_SOURCE_DIR}/build_out)
set(OPS_INFO_PATH ${CMAKE_SOURCE_DIR}/src)


message(STATUS "CUSTOM_INSTALL_PATH: ${CUSTOM_INSTALL_PATH}")
message(STATUS "OPS_INFO_PATH: ${OPS_INFO_PATH}")
add_npu_support_target(
        TARGET npu_supported_ops
        OPS_INFO_DIR ${OPS_INFO_PATH}
        OUT_DIR ${CUSTOM_INSTALL_PATH}
)

install(
        TARGETS ${PROJECT_NAME}
        DESTINATION ${CUSTOM_INSTALL_PATH}
)

set(MY_PROJECT_NAME ${PROJECT_NAME})
set(MY_CUSTOM_INSTALL_PATH ${CUSTOM_INSTALL_PATH})

install(CODE "
    message(STATUS \"\")
    message(STATUS \"===================================================\")
    message(STATUS \"Installation of ${MY_PROJECT_NAME} completed!\")
    message(STATUS \"Generated files are:\")
    message(STATUS \"  ${MY_CUSTOM_INSTALL_PATH}/framework/tensorflow/npu_supported_ops.json\")
    message(STATUS \"  ${MY_CUSTOM_INSTALL_PATH}/libcust_opapi.so\")
    message(STATUS \"Please set environment variables:\")
    message(STATUS \"  export ASCEND_CUSTOM_OPP_PATH=${MY_CUSTOM_INSTALL_PATH}:\$ASCEND_CUSTOM_OPP_PATH\")
")