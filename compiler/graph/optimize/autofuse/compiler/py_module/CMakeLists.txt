Python3_add_library(pyautofuse MODULE
        pyascir.cpp pyascir_types.cpp pyascir_common_utils.cpp
        pyautofuse.cpp
)

target_compile_options(pyautofuse PRIVATE
        -Wno-error=variadic-macros -Wall
        -Wno-missing-field-initializers # TODO(For autofuse): Remove this option and fix the related compilation errors.
        -Wno-cast-function-type # TODO(For autofuse): Remove this option and fix the related compilation errors.
)

target_include_directories(pyautofuse PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
        ${CMAKE_CURRENT_SOURCE_DIR}/../../optimize
        ${CMAKE_CURRENT_SOURCE_DIR}/../../codegen
        ${CMAKE_CURRENT_SOURCE_DIR}/../../autofuse
)

target_compile_definitions(pyautofuse PRIVATE
        google=ascend_private
)

target_link_options(pyautofuse PRIVATE
        $<$<CONFIG:Release>:-s>
)

target_link_libraries(pyautofuse PRIVATE $<BUILD_INTERFACE:intf_pub>)

if (ENABLE_OPEN_SRC)
    target_link_libraries(pyautofuse PUBLIC
        -Wl,--no-as-needed
        atrace_share
        aihac_codegen
        aihac_autofusion
        aihac_ir
        ascend_protobuf
        json
        graph
        graph_base
        $<BUILD_INTERFACE:ascir_ops_headers>
        )
else ()
    target_link_libraries(pyautofuse PRIVATE
        $<BUILD_INTERFACE:atrace_headers>
        atrace_share
        aihac_codegen
        aihac_autofusion
        aihac_ir
        ascend_protobuf
        json
        graph
        graph_base
        unified_dlog
        $<BUILD_INTERFACE:ascir_ops_headers>
        $<BUILD_INTERFACE:c_sec_headers>
        )
endif ()
