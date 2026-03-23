backend_e2e_st_test(copysign_bf16_test
    CODEGEN copysign_bf16_backend_generate.cpp
    KERNEL_SRC
        copysign_bf16_test_kernel.cpp
        copysign_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_copysign_bf16_expect_kernel.cpp)
