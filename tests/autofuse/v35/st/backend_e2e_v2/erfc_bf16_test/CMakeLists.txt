backend_e2e_st_test(erfc_bf16_test
    CODEGEN erfc_bf16_backend_generator.cpp
    KERNEL_SRC
        erfc_bf16_test_kernel.cpp
        erfc_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_erfc_bf16_expect_kernel.cpp)
