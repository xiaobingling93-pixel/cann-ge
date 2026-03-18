backend_e2e_st_test(digamma_bf16_test
    CODEGEN digamma_bf16_backend_generator.cpp
    KERNEL_SRC
        digamma_bf16_test_kernel.cpp
        digamma_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_digamma_bf16_expect_kernel.cpp)
