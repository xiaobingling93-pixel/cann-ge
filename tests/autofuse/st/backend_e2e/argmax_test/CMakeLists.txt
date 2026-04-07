backend_e2e_st_test(argmax_test
    CODEGEN argmax_backend_generate.cpp
    KERNEL_SRC
        argmax_test_kernel.cpp
        argmax_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_argmax_expect_kernel.cpp)
