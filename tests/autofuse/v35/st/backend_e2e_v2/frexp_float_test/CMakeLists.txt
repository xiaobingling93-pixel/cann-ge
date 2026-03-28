backend_e2e_st_test(frexp_float_test
    CODEGEN frexp_float_backend_generate.cpp
    KERNEL_SRC
        frexp_float_test_kernel.cpp
        frexp_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_frexp_float_expect_kernel.cpp)
