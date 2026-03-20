message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

###########  generate eager style api using generate_es_package.cmake  ######################
set(CMAKE_CURRENT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)

message(STATUS "Binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# 设置输出路径
set(ES_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/es_output)
message(STATUS "ES_OUTPUT_PATH: ${ES_OUTPUT_PATH}")

# 尝试使用 find_package 方式（适用于 run 包环境）
# 如果 run 包中有 generate_es_package.cmake，优先使用
if (EXISTS "${ASCEND_PATH}/include/ge/cmake/generate_es_package.cmake")
    list(APPEND CMAKE_MODULE_PATH "${ASCEND_PATH}/include/ge/cmake")
    find_package(GenerateEsPackage REQUIRED)
    message(STATUS "Using GenerateEsPackage from run package")
else ()
    message(FATAL_ERROR "Cannot find generate_es_package.cmake. Please ensure:\n"
            "  1. Run package is installed and ASCEND_PATH is set correctly, or\n"
            "  2. Source code contains generate_es_package.cmake")
endif ()

include_directories(
        ${ASCEND_PATH}/include
        ${ASCEND_PATH}/include/graph/
)

#生成算子 ES API
add_library(opgraph_all INTERFACE)
set_target_properties(opgraph_all PROPERTIES
        INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${ASCEND_PATH}/opp/built-in/op_proto"
)
add_es_library_and_whl(
        ES_LINKABLE_AND_ALL_TARGET es_all
        OPP_PROTO_TARGET opgraph_all
        OUTPUT_PATH ${ES_OUTPUT_PATH}
)