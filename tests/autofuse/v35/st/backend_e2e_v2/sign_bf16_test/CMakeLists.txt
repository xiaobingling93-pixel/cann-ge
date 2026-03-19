backend_e2e_st_test(sign_bf16_test
    CODEGEN sign_bf16_backend_generate.cpp
    KERNEL_SRC
        sign_bf16_test_kernel.cpp
        sign_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_sign_bf16_expect_kernel.cpp)
