backend_e2e_st_test(floortoint_float_test
    CODEGEN floortoint_float_backend_generate.cpp
    KERNEL_SRC
        floortoint_float_test_kernel.cpp
        floortoint_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_floortoint_float_expect_kernel.cpp)
