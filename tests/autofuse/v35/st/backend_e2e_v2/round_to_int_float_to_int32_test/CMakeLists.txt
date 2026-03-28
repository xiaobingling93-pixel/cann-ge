backend_e2e_st_test(round_to_int_float_to_int32_test
    CODEGEN round_to_int_float_to_int32_backend_generate.cpp
    KERNEL_SRC
        round_to_int_float_to_int32_test_kernel.cpp
        round_to_int_float_to_int32_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_round_to_int_float_to_int32_expect_kernel.cpp)