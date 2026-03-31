backend_e2e_st_test(tan_bf16_test
    CODEGEN tan_bf16_backend_generate.cpp
    KERNEL_SRC
        tan_bf16_test_kernel.cpp
        tan_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_tan_bf16_expect_kernel.cpp)