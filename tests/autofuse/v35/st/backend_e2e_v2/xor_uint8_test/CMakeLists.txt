backend_e2e_st_test(xor_uint8_test
    CODEGEN xor_uint8_backend_generate.cpp
    KERNEL_SRC
        xor_uint8_test_kernel.cpp
        xor_uint8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_xor_uint8_expect_kernel.cpp)
