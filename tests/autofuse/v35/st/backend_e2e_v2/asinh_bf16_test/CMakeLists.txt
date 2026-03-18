backend_e2e_st_test(asinh_bf16_test
    CODEGEN asinh_bf16_backend_generate.cpp
    KERNEL_SRC
        asinh_bf16_test_kernel.cpp
        asinh_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_asinh_bf16_expect_kernel.cpp)
