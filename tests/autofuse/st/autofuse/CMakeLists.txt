set(AUTOFUSION_ROOT_DIR ${CODE_ROOT_DIR}/autofuse)

file(GLOB_RECURSE AUTOFUSION_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../ut/autofuse/*.cpp")
list(REMOVE_ITEM AUTOFUSION_TEST_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../ut/autofuse/loop_lowering_manager_asc_backend_unittest.cpp")

add_executable(autofusion_st
#        test.cpp
        ${AUTOFUSION_TEST_FILES}
        fusion_strategy_solver_eagle_graph_test.cpp
        loop_lowering_manager_asc_backend_unittest.cpp
        )

set(ASCGEN_DT_COMPILE_OPTION
        -O0
        -g
        --coverage -fprofile-arcs -ftest-coverage
#        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
        )

set(ASCGEN_COMMON_LINK_OPTION
#        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
        -lrt -ldl -lgcov
        )

add_dependencies(autofusion_st metadef_graph_protos_obj ascir_ops_headers)

target_compile_definitions(autofusion_st PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        )

target_compile_options(autofusion_st PRIVATE ${ASCGEN_DT_COMPILE_OPTION}
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        -Wno-subobject-linkage
        )

target_include_directories(autofusion_st PUBLIC
        ${AUTOFUSION_ROOT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ut/autofuse
        ${CODE_ROOT_DIR}/../../../../
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/easy_graph/include
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/ge_graph_dsl/include
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/eager_style_graph_builder
        )

target_link_libraries(autofusion_st ${ASCGEN_COMMON_LINK_OPTION}
        ascir_ops_headers
        metadef_headers
        easy_graph_b
        ge_graph_dsl_b
        autofuse_eager_graph_builder_lib
        atrace
        graph_base
        graph
        metadef
        aihac_symbolizer
        autofuse
        error_manager
        mmpa
        ascgen_common
        ascend_protobuf
        aihac_symbolizer
        GTest::gtest
        GTest::gtest_main
        asc_slog_stub
        autofuse_runtime_stub
        json
        -Wl,--no-as-needed
        aihac_ir
        aihac_ir_register
)
