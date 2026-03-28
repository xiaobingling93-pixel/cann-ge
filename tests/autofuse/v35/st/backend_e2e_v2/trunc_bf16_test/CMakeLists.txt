backend_e2e_st_test(trunc_bf16_test
    CODEGEN trunc_bf16_backend_generate.cpp
    KERNEL_SRC
        trunc_bf16_test_kernel.cpp
        trunc_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_trunc_bf16_expect_kernel.cpp)
