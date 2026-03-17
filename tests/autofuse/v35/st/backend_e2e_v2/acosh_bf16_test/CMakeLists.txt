backend_e2e_st_test(acosh_bf16_test
    CODEGEN acosh_bf16_backend_generator.cpp
    KERNEL_SRC
        acosh_bf16_test_kernel.cpp
        acosh_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_acosh_bf16_expect_kernel.cpp)
