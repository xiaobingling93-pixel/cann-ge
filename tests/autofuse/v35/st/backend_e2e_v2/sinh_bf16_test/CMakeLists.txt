backend_e2e_st_test(sinh_bf16_test
    CODEGEN sinh_bf16_backend_generate.cpp
    KERNEL_SRC
        sinh_bf16_test_kernel.cpp
        sinh_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_sinh_bf16_expect_kernel.cpp)