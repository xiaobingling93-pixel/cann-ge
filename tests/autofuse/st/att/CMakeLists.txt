set (CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set (CMAKE_C_FLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address")

message(STATUS "Using environment variable LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")
message(STATUS "Using environment variable ASCEND_INSTALL_LIB_PATH: ${ASCEND_INSTALL_LIB_PATH}")
set(CMAKE_BUILD_TYPE Debug)

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
set(ATT_DIR ${TOP_DIR}/compiler/graph/optimize/autofuse/att)

add_compile_definitions(TOP_DIR=\"${TOP_DIR}\")
add_compile_definitions(ST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

file(GLOB SOURCES
        ${TOP_DIR}/tests/autofuse/st/att/testcase/matmul/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/ffn/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/att_st_test/att_test.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/test_expr/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass_gen/axes_reorder_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass_gen/general_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass_gen/l2_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass_gen/l0_solver_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass/l0_solver/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass/l2_solver/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass/general_solver/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/solver_pass_manager/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/select_model/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/stub/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/utils_test/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/gen_model_info/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/gen_model_info/api_tiling_gen/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/gen_model_info/api_tiling_gen/api/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/test_log/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/add_layer_norm/test_add_layer_norm.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/add_layer_norm/add_layer_norm.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/concat/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/cache/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/reduce_split_penalty/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/e2e_st_test/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/profiler_e2e/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/e2e_brc_buf/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/utils/*.cpp
)

file(GLOB DEPENDS
        ${ATT_PATH}/sample/depends/fake/*.cpp
        )

add_executable(att_st
        ${DEPENDS}
        ${SOURCES}
        ${TOP_DIR}/tests/autofuse/st/att/testcase/kernel_tiling/kernel_tiling.h
)
target_link_directories(att_st PRIVATE ${ASCEND_INSTALL_LIB_PATH})
target_include_directories(att_st PRIVATE
        ${TOP_DIR}/tests/autofuse/st/att/testcase
        ${TOP_DIR}/tests/autofuse/st/att/utils
        ${TOP_DIR}
        ${TOP_DIR}/compiler/graph/optimize/autofuse
        ${TOP_DIR}/common
        ${TOP_DIR}/base
        ${TOP_DIR}/tests/autofuse/st/att/utils
        ${TOP_DIR}/tests/autofuse/depends/common/inc
        ${ATT_DIR}/util
        ${ATT_DIR}/common_utils
        ${ATT_DIR}/generator
        ${ATT_DIR}/generator/preprocess
        ${ATT_DIR}/generator/solver_pass
        ${ATT_DIR}/generator/solver_pass_gen
        ${ATT_DIR}/generator/cache
        ${ATT_DIR}/gen_model_info
        ${ATT_DIR}/sample/depends/fake
        ${ASCEND_INSTALL_PATH}/include
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops/ascir_ops.h
)

target_compile_options(att_st PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -fpermissive
        )
target_compile_definitions(att_st PRIVATE ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH}")

target_link_options(att_st PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(att_st
        GTest::gtest GTest::gtest_main -lgcov
        symengine
        Boost::boost
        atrace
        att
        asc_slog_stub
        autofuse_runtime_stub
        aihac_ir
        ascir_builtin_ops
        aihac_symbolizer
        ascir_ops_headers
        aihac_ir_register
        graph_base
        graph
        error_manager
        c_sec
        json
        ascend_protobuf
        )
add_dependencies(att_st ascir_builtin_ops)
