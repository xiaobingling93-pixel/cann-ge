backend_e2e_st_test(ceil2int_bf16_test
    CODEGEN ceil2int_bf16_backend_generator.cpp
    KERNEL_SRC
        ceil2int_bf16_test_kernel.cpp
        ceil2int_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_ceil2int_bf16_expect_kernel.cpp)
