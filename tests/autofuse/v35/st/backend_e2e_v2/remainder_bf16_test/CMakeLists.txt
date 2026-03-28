backend_e2e_st_test(remainder_bf16_test
    CODEGEN remainder_bf16_backend_generate.cpp
    KERNEL_SRC
        remainder_bf16_test_kernel.cpp
        remainder_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_remainder_bf16_expect_kernel.cpp)
