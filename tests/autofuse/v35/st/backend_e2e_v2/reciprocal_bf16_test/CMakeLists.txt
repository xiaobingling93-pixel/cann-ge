backend_e2e_st_test(reciprocal_bf16_test
    CODEGEN reciprocal_bf16_backend_generator.cpp
    KERNEL_SRC
        reciprocal_bf16_test_kernel.cpp
        reciprocal_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_reciprocal_bf16_expect_kernel.cpp)
