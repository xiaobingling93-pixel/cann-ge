backend_e2e_st_test(sign_uint8_test
    CODEGEN sign_uint8_backend_generate.cpp
    KERNEL_SRC
        sign_uint8_test_kernel.cpp
        sign_uint8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_sign_uint8_expect_kernel.cpp)
