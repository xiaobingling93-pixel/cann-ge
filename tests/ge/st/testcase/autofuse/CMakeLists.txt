
file(GLOB_RECURSE AUTO_FUSE_TEST_SRCS CONFIGURE_DEPENDS "*.cc")
file(GLOB_RECURSE SYMBOLIC_SRCS CONFIGURE_DEPENDS  "${AIR_CODE_DIR}/compiler/graph/optimize/symbolic/*.cc")
add_executable(autofuse_test
        ${AUTO_FUSE_TEST_SRCS}
        "${AIR_CODE_DIR}/tests/ge/st/testcase/autofuse/test_symbolic_shape_inference.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/autofuse/test_symbolic_shape_compute.cc"
        ${SYMBOLIC_SRCS}
)

target_compile_options(autofuse_test PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
)

target_compile_definitions(autofuse_test PRIVATE
        google=ascend_private
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(autofuse_test PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base/
        ${AIR_CODE_DIR}/compiler/
        ${AIR_CODE_DIR}/tests/depends/
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/framework/
        ${AIR_CODE_DIR}/tests/depends/op_stub
        ${AIR_CODE_DIR}/tests/depends/aihacb_autofusion/src
        ${AIR_CODE_DIR}/tests/framework/ge_graph_dsl/include
        ${AIR_CODE_DIR}/tests/framework/easy_graph/include
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${AIR_CODE_DIR}/tests/ge/ut/ge/graph/optimize/symbolic
        ${AIR_CODE_DIR}/compiler/graph/optimize/symbolic/infer_symbolic_shape
        ${AIR_CODE_DIR}/compiler/graph/optimize/symbolic/infer_symbolic_shape/infer
        ${AIR_CODE_DIR}/tests/framework/ge_running_env/include
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/att/
        ${ASCGEN_DIR}
        ${ASCGEN_DIR}/autofuse
        ${CMAKE_BINARY_DIR}/tests/depends/aihacb_autofusion/
        ${CMAKE_BINARY_DIR}/proto/data_flow_protos
        ${METADEF_DIR}/base/registry
        ${METADEF_DIR}/register
)

target_link_libraries(autofuse_test PRIVATE
        st_stubs
        ge_common
        ge_common_base
        ge_graph_dsl
        ge_runtime_stub
        ge_running_env
        graphengine
        flow_graph
        metadef_headers
        ascendcl_stub
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        es_ge_test
        metadef_graph
        intf_pub
        -Wl,--whole-archive
        rt2_registry_static
        -Wl,--no-whole-archive
        json
        ge_running_env ge_runtime_stub GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lgcov
        -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

target_link_options(autofuse_test PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

add_dependencies(autofuse_test gert_op_impl)

add_test(NAME autofuse_test COMMAND autofuse_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/autofuse_test.xml)
set_tests_properties(autofuse_test PROPERTIES LABELS "st;st_hetero;autofuse_test")

include(CTest)
enable_testing()
add_test(NAME test COMMAND autofuse_test)