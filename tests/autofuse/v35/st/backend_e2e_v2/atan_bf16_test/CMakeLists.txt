backend_e2e_st_test(atan_bf16_test
    CODEGEN atan_bf16_backend_generator.cpp
    KERNEL_SRC
        atan_bf16_test_kernel.cpp
        atan_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_atan_bf16_expect_kernel.cpp)
