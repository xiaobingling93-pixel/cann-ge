set(CMAKE_CURRENT_BINARY_DIR ${PROJECT_SOURCE_DIR}/tests/framework/eager_style_graph_builder/build)
set(ES_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/es_output)
include(${AIR_CODE_DIR}/cmake/generate_es_package.cmake)

message(STATUS "[ES Test Ops] Generating ES API for test framework")

# 生成测试算子API
add_library(test_ops_proto SHARED
    ${CMAKE_CURRENT_LIST_DIR}/proto/test_ops.cc
)

target_compile_definitions(test_ops_proto PRIVATE OP_PROTO_LIB)
target_compile_options(test_ops_proto PRIVATE
    -fvisibility=hidden
)

target_include_directories(test_ops_proto PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/proto
    ${AIR_CODE_DIR}/inc/graph_metadef/external
)

target_link_libraries(test_ops_proto PRIVATE
    metadef_headers
    $<BUILD_INTERFACE:intf_llt_pub>
)

if(ENABLE_CROSS_COMPILE)
    target_link_directories(test_ops_proto PRIVATE
        ${CMAKE_COMPILE_COMPILER_LIBRARY}
        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()

# 调用 add_es_library 生成测试算子
add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_ge_test
    OPP_PROTO_TARGET test_ops_proto
    OUTPUT_PATH ${ES_OUTPUT_PATH}
)

# GCOV 模式下，动态库也需要链接 gcov 库，避免 hidden symbol 错误
if (CMAKE_BUILD_TYPE STREQUAL "GCOV")
    target_link_options(es_ge_test_so PRIVATE --coverage)
endif ()

