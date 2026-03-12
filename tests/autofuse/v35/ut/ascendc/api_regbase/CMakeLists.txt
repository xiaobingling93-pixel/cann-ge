add_library(test_regbase_ascendc OBJECT
    ./test_sub.cpp
    ./test_floor_div.cpp
    ./test_cast.cpp
    ./test_compare.cpp
    ./test_concat.cpp
    ./test_logical_and.cpp
    ./test_logical_not.cpp
    ./test_logical_or.cpp
    ./test_max.cpp
    ./test_min.cpp
    ./test_where.cpp
    ./test_reduce_init.cpp
    ./test_add.cpp
    ./test_bitwise_and.cpp
    ./test_rshift.cpp
    ./test_sin.cpp
    )
target_link_libraries(test_regbase_ascendc
    ascendc_api_regbase_extend
    tikicpulib_ascend950pr_9599
    depends
    GTest::gtest)

target_include_directories(test_regbase_ascendc PUBLIC
    ${CODE_ROOT_DIR}/ascendc
    ${CODE_ROOT_DIR}/v35/ascendc
)
