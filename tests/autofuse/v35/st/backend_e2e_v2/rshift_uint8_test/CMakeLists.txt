backend_e2e_st_test(rshift_uint8_test
    CODEGEN rshift_uint8_backend_generate.cpp
    KERNEL_SRC
        rshift_uint8_test_kernel.cpp
        rshift_uint8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_rshift_uint8_expect_kernel.cpp)
