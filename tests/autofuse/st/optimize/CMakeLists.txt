set(OPTIMIZE_ROOT_DIR ${CODE_ROOT_DIR}/src/optimize)

file(GLOB_RECURSE OPTIMIZE_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(optimize_st
        test.cpp
        ${OPTIMIZE_TEST_FILES}
        ${CMAKE_SOURCE_DIR}/tests/autofuse/framework/easy_asc_graph/asc_graph_builder.cpp
)

set(ASCGEN_DT_COMPILE_OPTION
        -O0
        -g
        --coverage -fprofile-arcs -ftest-coverage
        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
)

set(ASCGEN_COMMON_LINK_OPTION
        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
        -lrt -ldl -lgcov
)

target_include_directories(optimize_st PRIVATE
        ${CODE_ROOT_DIR}/../../../../
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/st/optimize
        ${CODE_ROOT_DIR}
)

target_compile_options(optimize_st PRIVATE ${ASCGEN_DT_COMPILE_OPTION}
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        #        -Werror
        -Wno-subobject-linkage
)

target_link_libraries(optimize_st ${ASCGEN_COMMON_LINK_OPTION}
        -Wl,--no-as-needed
        atrace
        optimize
        ascir
        aihac_ir
        aihac_ir_register
        ascgen_common
        asc_slog_stub
        autofuse_runtime_stub
        error_manager
        graph
        metadef
        e2e
        GTest::gtest
)
