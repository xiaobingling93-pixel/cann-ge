backend_e2e_st_test(logicalxor_float_test
    CODEGEN logicalxor_float_backend_generate.cpp
    KERNEL_SRC
        logicalxor_float_test_kernel.cpp
        logicalxor_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_logicalxor_float_expect_kernel.cpp)
