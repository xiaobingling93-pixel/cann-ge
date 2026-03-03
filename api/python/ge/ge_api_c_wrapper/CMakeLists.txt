file(GLOB_RECURSE GE_API_C_WRAPPER CONFIGURE_DEPENDS "${AIR_CODE_DIR}/api/python/ge/ge_api_c_wrapper/*.cc")

add_library(graph_wrapper SHARED ${GE_API_C_WRAPPER})
target_compile_options(graph_wrapper PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})
target_link_libraries(graph_wrapper PRIVATE
        intf_pub
        slog_headers
        mmpa_headers
        metadef_headers
        c_sec
        graph
        graph_base
        -lrt
        -ldl
        air_headers
        ge_runner
        $<$<CONFIG:Release>:-s>
)