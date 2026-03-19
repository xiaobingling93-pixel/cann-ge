backend_e2e_st_test(truediv_bf16_test
    CODEGEN truediv_bf16_backend_generate.cpp
    KERNEL_SRC
        truediv_bf16_test_kernel.cpp
        truediv_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_truediv_bf16_expect_kernel.cpp)
