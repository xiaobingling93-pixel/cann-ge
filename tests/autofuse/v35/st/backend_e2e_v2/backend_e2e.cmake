function(do_backend_e2e_st_test)
    set(one_value_arg
        WORKDIR # Workdir
        )
    set(mul_value_arg
        CODEGEN # Codegen library source file
        KERNEL_SRC # Kernel source file that codegen will generate
        TEST_SRC # Test case source file
        TILING_KEY
        )

    set(TEST_NAME ${ARGV0})
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${one_value_arg}" "${mul_value_arg}")

    if(NOT ARG_TILING_KEY)
        set(ARG_TILING_KEY 0)
    endif()

    message(STATUS "ARG_WORKDIR=${ARG_WORKDIR}")

    foreach(file ${ARG_KERNEL_SRC})
        list(APPEND KERNEL_SRC "${ARG_WORKDIR}/${file}")
    endforeach()

    set(E2E_ST1_GENERATOR_EXE_NAME ${TEST_NAME}_codegen_v2)
    set(E2E_ST2_EXE_KERNEL_EXE_NAME ${TEST_NAME}_e2e_v2)

    add_executable(${E2E_ST1_GENERATOR_EXE_NAME} ${ARG_CODEGEN})
    target_include_directories(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
        ${CODE_ROOT_DIR}/tests/ge/st/common)
    target_link_options(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
            -Wl,--no-as-needed
    )
    target_include_directories(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
            ${CODE_ROOT_DIR}/../../../../
            ${CODE_ROOT_DIR}/../../../../tests/autofuse/st/common
    )
    target_link_libraries(${E2E_ST1_GENERATOR_EXE_NAME}
                          easy_asc_graph
                          atrace
                          asc_slog_stub
                          optimize
                          att
                          codegen
                          common_stub
                          share_graph
                          ascgen_common
                          aihac_symbolizer
                          error_manager
                          metadef
                          json
                          GTest::gtest
                          GTest::gtest_main
                          autofuse_runtime_stub)


    list (JOIN ARG_KERNEL_SRC ":" KERNEL_SRC_LIST)
    message(STATUS "KERNEL_SRC_LIST = ${KERNEL_SRC_LIST}")

    target_compile_definitions(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
       ATT_SO_NAME=\"./libatt.so\"
       KERNEL_SRC_LIST=\"${KERNEL_SRC_LIST}\"
    )

    # 一个完整ST用例的两个ST用例源2文件放在一个目录下
    execute_process(COMMAND touch ${ARG_KERNEL_SRC}   # 创建第一类用例生成的空文件
                    WORKING_DIRECTORY ${ARG_WORKDIR})

    add_custom_target(call_${E2E_ST1_GENERATOR_EXE_NAME} ALL
                      COMMAND ${E2E_ST1_GENERATOR_EXE_NAME}
                      DEPENDS ${E2E_ST1_GENERATOR_EXE_NAME}
                      WORKING_DIRECTORY ${ARG_WORKDIR})

    add_executable(${E2E_ST2_EXE_KERNEL_EXE_NAME} ${KERNEL_SRC} ${ARG_TEST_SRC})
    add_dependencies(${E2E_ST2_EXE_KERNEL_EXE_NAME} call_${E2E_ST1_GENERATOR_EXE_NAME})
    target_include_directories(${E2E_ST2_EXE_KERNEL_EXE_NAME} PRIVATE ${ARG_WORKDIR})
    target_link_libraries(${E2E_ST2_EXE_KERNEL_EXE_NAME} tikicpulib_ascend950pr_9599 GTest::gtest GTest::gtest_main)
    target_link_libraries(${E2E_ST2_EXE_KERNEL_EXE_NAME} unified_dlog)
    target_compile_options(${E2E_ST2_EXE_KERNEL_EXE_NAME} PRIVATE -DAUTO_FUSE_DEVICE=1  -DTILING_KEY_VAR=${ARG_TILING_KEY})
    #gtest_discover_tests(${E2E_ST2_EXE_KERNEL_EXE_NAME})
endfunction()

macro(backend_e2e_st_test)
    do_backend_e2e_st_test(${ARGV}  WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro()
