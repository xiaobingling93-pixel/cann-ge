backend_e2e_st_test(log10_float_test
    CODEGEN log10_float_backend_generate.cpp
    KERNEL_SRC
        log10_float_test_kernel.cpp
        log10_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_log10_float_expect_kernel.cpp)
