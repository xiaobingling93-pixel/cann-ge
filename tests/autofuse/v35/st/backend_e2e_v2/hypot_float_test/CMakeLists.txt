backend_e2e_st_test(hypot_float_test
    CODEGEN hypot_float_backend_generate.cpp
    KERNEL_SRC
        hypot_float_test_kernel.cpp
        hypot_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_hypot_float_expect_kernel.cpp)
