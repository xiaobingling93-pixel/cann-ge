backend_e2e_st_test(scalar_cast_add_test
    CODEGEN scalar_cast_add_backend_generate.cpp 
    KERNEL_SRC
        scalar_cast_add_test_kernel.cpp
        scalar_cast_add_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_scalar_cast_add_expect_kernel.cpp)
