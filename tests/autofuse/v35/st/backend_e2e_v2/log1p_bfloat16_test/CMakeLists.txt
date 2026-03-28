backend_e2e_st_test(log1p_bfloat16_test
    CODEGEN log1p_bfloat16_backend_generator.cpp
    KERNEL_SRC
        log1p_bfloat16_test_kernel.cpp
        log1p_bfloat16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_log1p_bfloat16_expect_kernel.cpp)