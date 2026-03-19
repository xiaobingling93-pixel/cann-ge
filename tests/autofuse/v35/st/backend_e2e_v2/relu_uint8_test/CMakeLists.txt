backend_e2e_st_test(relu_uint8_test
    CODEGEN relu_uint8_backend_generate.cpp
    KERNEL_SRC
        relu_uint8_test_kernel.cpp
        relu_uint8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_relu_uint8_expect_kernel.cpp)
