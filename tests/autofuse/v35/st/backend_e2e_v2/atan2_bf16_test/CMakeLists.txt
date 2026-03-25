backend_e2e_st_test(atan2_bf16_test
    CODEGEN atan2_bf16_backend_generator.cpp
    KERNEL_SRC
        atan2_bf16_test_kernel.cpp
        atan2_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_atan2_bf16_expect_kernel.cpp)
