backend_e2e_st_test(pow_bf16_test
    CODEGEN pow_bf16_backend_generate.cpp
    KERNEL_SRC
        pow_bf16_test_kernel.cpp
        pow_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_pow_bf16_expect_kernel.cpp)
