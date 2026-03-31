backend_e2e_st_test(truncdiv_bf16_test
    CODEGEN truncdiv_bf16_backend_generate.cpp
    KERNEL_SRC
        truncdiv_bf16_test_kernel.cpp
        truncdiv_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_truncdiv_bf16_expect_kernel.cpp)
