backend_e2e_st_test(asin_bf16_test
    CODEGEN asin_bf16_backend_generate.cpp
    KERNEL_SRC
        asin_bf16_test_kernel.cpp
        asin_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_asin_bf16_expect_kernel.cpp)
