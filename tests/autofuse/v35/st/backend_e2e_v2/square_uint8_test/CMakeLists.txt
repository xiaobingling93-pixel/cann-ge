backend_e2e_st_test(square_uint8_test
    CODEGEN square_uint8_backend_generate.cpp
    KERNEL_SRC
        square_uint8_test_kernel.cpp
        square_uint8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_square_uint8_expect_kernel.cpp)
