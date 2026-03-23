backend_e2e_st_test(erfcx_test
    CODEGEN erfcx_test_backend_generator.cpp
    KERNEL_SRC
        erfcx_test_test_kernel.cpp
        erfcx_test_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_erfcx_test_expect_kernel.cpp)
