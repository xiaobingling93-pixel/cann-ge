backend_e2e_st_test(atanh_bf16_test
    CODEGEN atanh_bf16_backend_generator.cpp
    KERNEL_SRC
        atanh_bf16_test_kernel.cpp
        atanh_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_atanh_bf16_expect_kernel.cpp)
