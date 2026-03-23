add_executable(codegen_st
        test.cpp
        test_codegen_infershape.cpp
)
target_compile_definitions(codegen_st PRIVATE
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        ASCEND_INSTALL_PATH=\"${ASCEND_INSTALL_PATH}\"
)
target_link_libraries(codegen_st PRIVATE
    -Wl,--no-as-needed
    atrace
    optimize
    codegen
    aihac_ir
    aihac_ir_register
    graph_base
    graph
    metadef
    ascir
    ascgen_common
    aihac_symbolizer
    json
    ascendc_api_extend
    GTest::gtest)

add_test(NAME codegen_st COMMAND codegen_st --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/codegen_st.xml)
set_tests_properties(codegen_st PROPERTIES LABELS "st;codegen_st")