backend_e2e_st_test(round_bf16_test
    CODEGEN round_bf16_backend_generate.cpp
    KERNEL_SRC
        round_bf16_test_kernel.cpp
        round_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_round_bf16_expect_kernel.cpp)
