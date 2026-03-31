backend_e2e_st_test(lgamma_float_test
    CODEGEN lgamma_float_backend_generate.cpp
    KERNEL_SRC
        lgamma_float_test_kernel.cpp
        lgamma_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_lgamma_float_expect_kernel.cpp)