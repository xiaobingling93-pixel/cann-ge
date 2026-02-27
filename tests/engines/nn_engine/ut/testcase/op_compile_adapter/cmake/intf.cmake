add_library(intf_llt_pub INTERFACE)

target_include_directories(intf_llt_pub INTERFACE
)

target_compile_definitions(intf_llt_pub INTERFACE
        CFG_BUILD_DEBUG
        _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(intf_llt_pub INTERFACE
        -g
        -w
        $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
        $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
        -fPIC
        -pipe
)

target_link_options(intf_llt_pub INTERFACE
        $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
        $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_directories(intf_llt_pub INTERFACE
)

target_link_libraries(intf_llt_pub INTERFACE
        GTest::gtest
        -lpthread
        mockcpp_static
        $<$<BOOL:${ENABLE_GCOV}>:-lgcov>
)

