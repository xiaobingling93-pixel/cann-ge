backend_e2e_st_test(isfinite_bf16_test
    CODEGEN isfinite_bf16_backend_generate.cpp 
    KERNEL_SRC
        isfinite_bf16_test_kernel.cpp
        isfinite_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_isfinite_bf16_expect_kernel.cpp)
