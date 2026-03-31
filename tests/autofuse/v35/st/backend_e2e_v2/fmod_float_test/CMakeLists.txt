backend_e2e_st_test(fmod_float_test
    CODEGEN fmod_float_backend_generate.cpp
    KERNEL_SRC
        fmod_float_test_kernel.cpp
        fmod_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_fmod_float_expect_kernel.cpp)
