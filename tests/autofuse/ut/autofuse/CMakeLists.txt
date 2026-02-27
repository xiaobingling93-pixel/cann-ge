set(AUTOFUSE_ROOT_DIR ${CODE_ROOT_DIR}/autofuse)

file(GLOB_RECURSE AUTOFUSE_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(autofusion_ut
        test.cpp
        ${AUTOFUSE_TEST_FILES}
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

add_dependencies(autofusion_ut metadef_graph_protos_obj ascir_ops_headers)

target_compile_definitions(autofusion_ut PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        )

target_compile_options(autofusion_ut PRIVATE ${ASCGEN_DT_COMPILE_OPTION}
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        #        -Werror
        -Wno-subobject-linkage
        )

target_include_directories(autofusion_ut PUBLIC
        ${AUTOFUSE_ROOT_DIR}
        ${CODE_ROOT_DIR}/../../../../
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/easy_graph/include
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/ge_graph_dsl/include
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/eager_style_graph_builder
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/ut/autofuse
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/depends/runtime/src
        )

target_link_libraries(autofusion_ut PRIVATE ${ASCGEN_COMMON_LINK_OPTION}
        -Wl,--no-as-needed
        easy_graph_b
        ge_graph_dsl_b
        autofuse_eager_graph_builder_lib
        ascir_ops_headers
        aihac_symbolizer
        atrace
        autofuse
        error_manager
        mmpa
        ascgen_common
        ascend_protobuf
        aihac_ir
        aihac_ir_register
        graph_base
        graph
        metadef
        metadef_headers
        atrace_headers
        asc_slog_stub
        autofuse_runtime_stub
        GTest::gtest
        json
        )