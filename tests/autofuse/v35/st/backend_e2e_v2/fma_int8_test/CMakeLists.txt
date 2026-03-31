backend_e2e_st_test(fma_int8_test
    CODEGEN fma_int8_backend_generate.cpp 
    KERNEL_SRC
        fma_int8_test_kernel.cpp
        fma_int8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_fma_int8_expect_kernel.cpp)
