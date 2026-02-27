set (CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set (CMAKE_C_FLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
set(ATT_DIR ${TOP_DIR}/compiler/graph/optimize/autofuse/att)

cmake_minimum_required(VERSION 3.14)
project (Att[CXX])
add_compile_options(-std=c++17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_compile_definitions(TOP_DIR=\"${TOP_DIR}\")
add_compile_definitions(UT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

file(GLOB SOURCES
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/test_expr/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass_gen/axes_reorder_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass_gen/general_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass_gen/l2_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass_gen/l0_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass/l0_solver/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass/l2_solver/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass/general_solver/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/solver_pass_manager/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/util/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/generator/att_generator_unittest.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info/api_perf_register/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info/expr_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info/parser/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info/pass/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info/equivalent_graph_recongnizer/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/ascir/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/check_input/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/matmul/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/concat/*.cpp
        ${TOP_DIR}/tests/autofuse/ut/att/utils/*.cpp
    )

add_executable(att_ut
        ${SOURCES}
        )

target_link_directories(att_ut PRIVATE ${ASCEND_INSTALL_LIB_PATH})

target_include_directories(att_ut PRIVATE
        ${TOP_DIR}/tests/autofuse/ut/att/testcase
        ${TOP_DIR}/tests/autofuse/ut/att/utils
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/gen_model_info
        ${TOP_DIR}/
        ${TOP_DIR}/compiler/graph/optimize/autofuse
        ${TOP_DIR}/common
        ${TOP_DIR}/tests/autofuse/depends/common/inc
        ${ATT_DIR}/util
        ${ATT_DIR}/dtree
        ${ATT_DIR}/common_utils
        ${ATT_DIR}/generator
        ${ATT_DIR}/generator/decision_tree
        ${ATT_DIR}/generator/preprocess
        ${ATT_DIR}/generator/solver_pass_gen
        ${ATT_DIR}/generator/decision_tree
        ${ATT_DIR}/gen_model_info
        ${ASCEND_INSTALL_PATH}/include
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops/ascir_ops.h
        ${TOP_DIR}/tests/autofuse/ut/att/testcase/ascir
        ${TOP_DIR}/tests/autofuse/ut/att/utils
        )

target_compile_options(att_ut PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -fpermissive
        )

target_link_options(att_ut PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(att_ut PRIVATE
        GTest::gtest GTest::gmock GTest::gtest_main -lgcov
        -Wl,--no-as-needed
        att
        asc_slog_stub
        autofuse_runtime_stub
        aihac_ir
        aihac_symbolizer
        ascir_builtin_ops
        aihac_ir_register
        ascir_ops_headers
        symengine
        Boost::boost
        graph_base
        error_manager
        metadef
        graph
        c_sec
        json
        ascend_protobuf
        )
add_dependencies(att_ut ascir_builtin_ops)