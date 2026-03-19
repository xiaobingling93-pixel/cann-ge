backend_e2e_st_test(tanh_bf16_test
    CODEGEN tanh_bf16_backend_generate.cpp
    KERNEL_SRC
        tanh_bf16_test_kernel.cpp
        tanh_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_tanh_bf16_expect_kernel.cpp)
