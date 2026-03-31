add_library(kernel_load_platform SHARED ${AIR_CODE_DIR}/tests/depends/op_stub/so_stub/lib64/device/lib64/platform_so_stub.cc)

target_compile_options(kernel_load_platform PRIVATE ${AIR_COMMON_COMPILE_OPTION})

target_link_libraries(kernel_load_platform PRIVATE $<BUILD_INTERFACE:intf_llt_pub>)

add_dependencies(davinci_executor kernel_load_platform)
