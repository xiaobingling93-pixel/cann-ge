include(${GE_METADEF_DIR}/graph/ascendc_ir/generator/generator.cmake)

set(GENERATOR_DIR ${CODE_ROOT_DIR}/ascir/generator)
set(REG_FUNC_DIR ${CODE_ROOT_DIR}/ascir/reg_func)
file(GLOB_RECURSE GENERATOR_SOURCES "${GENERATOR_DIR}/*.cpp")
file(GLOB_RECURSE REG_FUNC_SOURCES "${REG_FUNC_DIR}/*.cpp")

add_library(ascir_builtin_ops SHARED
        ${GENERATOR_SOURCES}
        ${REG_FUNC_SOURCES}
        )

target_compile_options(ascir_builtin_ops PRIVATE
        -Wextra
        -Wfloat-equal
        -Wall
)

target_link_options(ascir_builtin_ops PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(ascir_builtin_ops PRIVATE
        aihac_ir_register
        ascend_protobuf
        $<BUILD_INTERFACE:intf_pub>
        )

target_include_directories(ascir_builtin_ops PRIVATE
        ${GE_METADEF_INC_DIR}
        ${GE_METADEF_INC_DIR}/graph
        ${GE_METADEF_INC_DIR}/graph/ascendc_ir
        ${GE_METADEF_INC_DIR}/external
        ${GE_METADEF_DIR}/graph/ascendc_ir/generator
        )

set(ops_header_dir ${CMAKE_BINARY_DIR}/ascir_builtin_ops)
set_target_properties(ascir_ops_header_generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ops_header_dir})
message("ops_header_dir is " ${ops_header_dir}/ascir_ops.h)
ascir_generate(ascir_builtin_ops ${ops_header_dir} ${CMAKE_CURRENT_BINARY_DIR}/libascir_builtin_ops.so ${ops_header_dir}/ascir_ops.h)

add_custom_target(ascir_builtin_ops_header ALL DEPENDS ${ops_header_dir}/ascir_ops.h)
add_library(ascir_ops_headers INTERFACE)
target_include_directories(ascir_ops_headers INTERFACE ${ops_header_dir})
add_dependencies(ascir_ops_headers ascir_builtin_ops_header)

################################################################################
target_sources(aihac_codegen PRIVATE
    ${GENERATOR_SOURCES}
    ${REG_FUNC_SOURCES}
)

target_include_directories(aihac_codegen PRIVATE
    ${GE_METADEF_INC_DIR}
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/external
    ${GE_METADEF_DIR}/graph/ascendc_ir/generator
    ${ops_header_dir}
)
