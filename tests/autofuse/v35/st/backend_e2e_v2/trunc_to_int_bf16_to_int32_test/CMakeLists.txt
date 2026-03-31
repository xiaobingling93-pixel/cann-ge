backend_e2e_st_test(trunc_to_int_bf16_to_int32_test
    CODEGEN trunc_to_int_bf16_to_int32_backend_generate.cpp
    KERNEL_SRC
        trunc_to_int_bf16_to_int32_test_kernel.cpp
        trunc_to_int_bf16_to_int32_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_trunc_to_int_bf16_to_int32_expect_kernel.cpp)
