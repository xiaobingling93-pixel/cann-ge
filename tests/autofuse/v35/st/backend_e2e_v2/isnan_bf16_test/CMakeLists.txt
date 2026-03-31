backend_e2e_st_test(isnan_bf16_test
    CODEGEN isnan_bf16_backend_generate.cpp 
    KERNEL_SRC
        isnan_bf16_test_kernel.cpp
        isnan_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_isnan_bf16_expect_kernel.cpp)
