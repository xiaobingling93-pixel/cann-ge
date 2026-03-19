set(TRACE_SRC_CPP
        atrace_api_stub.cc)

add_library(atrace_stub SHARED
        ${TRACE_SRC_CPP})

target_compile_options(atrace_stub PRIVATE
        -Dgoogle=ascend_private)

target_link_libraries(atrace_stub
    PRIVATE
        intf_llt_pub
        c_sec
    PUBLIC
        atrace_headers
)

install(TARGETS atrace_stub OPTIONAL
        EXPORT atrace_stub-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})