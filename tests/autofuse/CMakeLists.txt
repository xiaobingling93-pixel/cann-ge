cmake_minimum_required(VERSION 3.16.1)
project(autofuse)

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all -Wl,-z,now")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-all -Wl,-z,now")

if (EXISTS "${CMAKE_SOURCE_DIR}/.dev_env")
    # 存在`.dev_env`文件说明是蓝区开发环境，并且调用了`prepare_dev_env.sh`完成了开发环境准备
    # 除了设置好`.dev_env`中有的环境变量外，还会设置其他几个蓝区开发的必要变量
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.dev_env")
        set(ENABLE_OPEN_SRC True)
        # Read the file into a list of lines
        file(STRINGS "${CMAKE_SOURCE_DIR}/.dev_env" DEV_ENV_LINES)

        # Process each line
        foreach (line IN LISTS DEV_ENV_LINES)
            string(STRIP "${line}" line)
            if (line MATCHES "^([^=]+)=(.*)$")
                set(var_name "${CMAKE_MATCH_1}")
                set(var_value "${CMAKE_MATCH_2}")
                # Remove any leading/trailing whitespace
                string(STRIP "${var_name}" var_name)
                string(STRIP "${var_value}" var_value)
                # Set the variable
                set(${var_name} ${var_value})
            endif ()
        endforeach ()

        set(ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH}/latest)
    endif ()
endif ()
set(CODE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/graph/optimize/autofuse)
set(ASCEND_ROOT ${ASCEND_INSTALL_PATH}) # e.g. /usr/local/Ascend/ascend-toolkit/latest

set(ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH})

if (ENABLE_OPEN_SRC)
    if (NOT DEFINED ASCEND_3RD_LIB_PATH)
        set(ASCEND_3RD_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/output/third_party)
    endif ()

    if (NOT EXISTS "${ASCEND_3RD_LIB_PATH}")
        message(FATAL_ERROR "Error: Directory \"${ASCEND_3RD_LIB_PATH}\" does not exist!\n"
                "Please using build_third_party.sh compile first.")
    endif ()
    #    include(../../cmake/test_utils.cmake)
    set(GE_METADEF_DIR ${CODE_ROOT_DIR}/../../../../graph_metadef)
    set(GE_METADEF_INC_DIR ${CODE_ROOT_DIR}/../../../../inc/graph_metadef)
else ()
    set(GE_METADEF_DIR ${TOP_DIR}/air/graph_metadef)
    set(GE_METADEF_INC_DIR ${TOP_DIR}/air/inc/graph_metadef)
endif ()

if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH
            ${ASCEND_3RD_LIB_PATH}/cmake/modules
            ${GE_METADEF_DIR}/cmake/modules
            ${CODE_ROOT_DIR}/cmake/third_party/scripts/modules
    )
endif()

if (NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH
            ${ASCEND_3RD_LIB_PATH}/json
            ${ASCEND_3RD_LIB_PATH}/openssl
            ${ASCEND_3RD_LIB_PATH}/protoc
            ${ASCEND_3RD_LIB_PATH}/protoc_grpc
            ${ASCEND_3RD_LIB_PATH}/grpc
            ${ASCEND_3RD_LIB_PATH}/protobuf_static
            ${ASCEND_3RD_LIB_PATH}/ascend_protobuf
            ${ASCEND_3RD_LIB_PATH}/ascend_protobuf_static
            ${ASCEND_3RD_LIB_PATH}/gtest_shared/lib64/cmake/GTest
            ${ASCEND_3RD_LIB_PATH}/gtest
            ${ASCEND_3RD_LIB_PATH}/benchmark
            ${ASCEND_3RD_LIB_PATH}/pybind11
            ${ASCEND_3RD_LIB_PATH}/symengine/lib/cmake/symengine
            ${ASCEND_3RD_LIB_PATH}/boost/lib/cmake/Boost-1.87.0
    )
endif()


link_directories(${ASCEND_ROOT}/lib64)
include_directories(${GE_METADEF_INC_DIR}/external)
include_directories(${GE_METADEF_INC_DIR}/external/graph)
include_directories(${GE_METADEF_INC_DIR}/inc/graph)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/utils)


include_directories(${GE_METADEF_INC_DIR}/graph/debug)
include_directories(${GE_METADEF_INC_DIR}/graph/utils)
include_directories(${GE_METADEF_DIR}/graph)
include_directories(${GE_METADEF_DIR})
include_directories(${GE_METADEF_DIR}/../tests/graph_metadef/ut)
include_directories(${CODE_ROOT_DIR}/common)
include_directories(${CODE_ROOT_DIR}/ascendc/api)
include_directories(${CODE_ROOT_DIR}/compiler/py_module)
include_directories(${CODE_ROOT_DIR}/ascir/meta)
include_directories(${CODE_ROOT_DIR}/att)
include_directories(${CODE_ROOT_DIR}/inc)
include_directories(${CODE_ROOT_DIR}/inc/common)
include_directories(${ASCEND_ROOT}/include)
include_directories(${ASCEND_ROOT}/include/experiment)
include_directories(${ASCEND_ROOT}/include/experiment/runtime)
include_directories(${ASCEND_ROOT}/include/experiment/msprof)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc/base)



find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (ENABLE_OPEN_SRC)
    #    find_package(nlohmann_json REQUIRED)
    find_package(json REQUIRED)
    find_package(GTest CONFIG REQUIRED)
    find_package(Boost CONFIG REQUIRED) #
    find_package(SymEngine CONFIG REQUIRED) #
    find_package(ascend_protobuf_shared MODULE REQUIRED)
    #find_package(ascend_protobuf_static MODULE REQUIRED)
    #find_package(protoc MODULE REQUIRED)
    find_package(unified_dlog)
    find_package(mmpa)
    find_package(runtime)
    find_package(atrace)
endif ()

include(GoogleTest)
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

add_subdirectory(depends/slog)
add_subdirectory(depends/trace)
add_subdirectory(depends/runtime)
add_subdirectory(depends/common)
add_subdirectory(framework)
add_subdirectory(st)
add_subdirectory(ut)

include(CMakePrintHelpers)
message(STATUS "Variables in project:")
cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(INSTALL_BASE_DIR)
cmake_print_variables(ENABLE_OPEN_SRC)

if (ENABLE_OPEN_SRC)
    find_package(protoc MODULE REQUIRED)
    find_package(ascend_protobuf_static MODULE REQUIRED)
endif ()
####################################common############################################
set(COMMON_DIR ${CODE_ROOT_DIR}/common)
file(GLOB_RECURSE CODEGEN_COMMON_SRCS CONFIGURE_DEPENDS "${COMMON_DIR}/*.cpp" "${COMMON_DIR}/*.cc")
add_library(ascgen_common SHARED
        ${CODEGEN_COMMON_SRCS}
)

target_include_directories(ascgen_common PUBLIC
        ${CODE_ROOT_DIR}
        ${COMMON_DIR}
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops
)

if (ENABLE_OPEN_SRC)
    target_link_libraries(ascgen_common PUBLIC
            -lgcov
            graph
            ascir
            aihac_symbolizer
            mmpa
            c_sec
            unified_dlog
            autofuse_runtime_stub
            atrace
            ascend_protobuf
            aihac_ir_register
    )
else ()
    target_link_libraries(ascgen_common PUBLIC
            $<BUILD_INTERFACE:intf_pub_cxx17>
            $<BUILD_INTERFACE:mmpa_headers>
            $<BUILD_INTERFACE:msprof_headers>
            $<BUILD_INTERFACE:runtime_headers>
            $<BUILD_INTERFACE:atrace_headers>
            mmpa
            c_sec
            unified_dlog
            atrace
            graph
            json
            ascend_protobuf
            aihac_ir_register
    )
endif ()

#####################################att###########################################
SET(ENABLE_SYMENGINE "on")

set(ATT_DIR ${CODE_ROOT_DIR}/att)

file(GLOB SOURCES
    ${ATT_DIR}/generator/*.cpp
    ${ATT_DIR}/generator/cache/*.cpp
    ${ATT_DIR}/generator/preprocess/*.cpp
    ${ATT_DIR}/generator/solver_pass/*.cpp
    ${ATT_DIR}/generator/solver_pass/axes_reorder_solver_code/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/axes_reorder_solver/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/general_solver/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/golden_solver/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/l0_solver/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/l2_solver/*.cpp
    ${ATT_DIR}/generator/extra_info_gen/*.cpp
    ${ATT_DIR}/generator/generator_utils/*.cpp
    ${ATT_DIR}/generator/tiling_data_gen/*.cpp
    ${ATT_DIR}/generator/tiling_option_generator/*.cpp
    ${ATT_DIR}/util/*.cpp
    ${ATT_DIR}/gen_model_info/axes_priority_reg/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/v1/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/v2/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/utils/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/*.cpp
    ${ATT_DIR}/gen_model_info/expr_gen/*.cpp
    ${ATT_DIR}/gen_model_info/parser/*.cpp
    ${ATT_DIR}/gen_model_info/pass/*.cpp
    ${ATT_DIR}/gen_model_info/reuse_group_utils/*.cpp
    ${ATT_DIR}/gen_model_info/*.cpp
    ${ATT_DIR}/gen_model_info/axes_priority_reg/*.cpp
    ${ATT_DIR}/gen_model_info/api_tiling_gen/*.cpp
    ${ATT_DIR}/gen_model_info/api_tiling_gen/api/*.cpp
    ${ATT_DIR}/gen_model_info/sketch_gen/*.cpp
    ${ATT_DIR}/gen_tiling_impl.cpp
    ${CODEGEN_COMMON_SRCS}
    )


add_library(att SHARED
    ${SOURCES}
)

add_dependencies(att ascir_builtin_ops_header)
target_include_directories(att PRIVATE
        ${ATT_DIR}/../
        ${ATT_DIR}
        ${ATT_DIR}/../common
        ${ATT_DIR}/../inc/
        ${ATT_DIR}/base
        ${ATT_DIR}/util
        ${ATT_DIR}/common_utils
        ${ATT_DIR}/generator
        ${ATT_DIR}/generator/preprocess
        ${ATT_DIR}/generator/solver_pass
        ${ATT_DIR}/generator/solver_pass_gen
        ${ATT_DIR}/ascir
        ${ATT_DIR}/ascir/generator
        ${ATT_DIR}/gen_model_info
        ${ASCEND_INSTALL_PATH}/include
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops
        )

add_definitions(-DFORMULAS_PERF)

target_compile_options(att PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
)

target_link_options(att PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(att PRIVATE
        -lgcov
        symengine
        Boost::boost
        optimize
        json
        aihac_ir
        aihac_symbolizer
        graph_base
        graph
        error_manager
        ascend_protobuf
        ge_log_utils
        ascir_builtin_ops
        autofuse_runtime_stub
)

if (DEFINED ENABLE_OPEN_SRC)
    target_link_libraries(att PUBLIC
            $<BUILD_INTERFACE:mmpa_headers>
            $<BUILD_INTERFACE:runtime_headers>
            mmpa
            unified_dlog
            atrace
    )
endif ()

########################################codegen########################################
set(CODEGEN_DIR ${CODE_ROOT_DIR}/codegen)
file(GLOB_RECURSE CODEGEN_SOURCES "${CODEGEN_DIR}/*.cpp")
add_library(codegen SHARED
        ${CODEGEN_SOURCES}
)

set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH};${CODE_ROOT_DIR}/att/build")

target_compile_options(codegen PRIVATE 
    -Wfloat-equal -Wextra
)

target_include_directories(codegen PUBLIC
    ${CODEGEN_DIR}
    ${CODEGEN_DIR}/api_call
    ${CODE_ROOT_DIR}/common
    ${CODE_ROOT_DIR}/att
    ${CODE_ROOT_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/depends/runtime/src
)

if (ENABLE_OPEN_SRC)
    target_link_libraries(codegen PUBLIC
        -lgcov
        graph
        ascir
        ascgen_common
        aihac_symbolizer
        error_manager
        att
        json
        ascendc_api_extend
    )
else ()
    target_link_libraries(codegen PUBLIC
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:atrace_headers>
        mmpa
        c_sec
        unified_dlog
        atrace
        ascend_protobuf
        graph
        json
        ascendc_api_extend
    )
endif ()

######################################optimize##########################################
set(OPTIMIZE_DIR ${CODE_ROOT_DIR}/optimize)
file(GLOB_RECURSE SOURCES "${OPTIMIZE_DIR}/*.cpp")

add_library(optimize SHARED
        ${SOURCES}
)

target_include_directories(optimize PUBLIC
        ${OPTIMIZE_DIR}
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/autofuse
)

target_compile_options(optimize PRIVATE -fPIC -Werror -fno-common -Wextra -Wfloat-equal -O2 -Wall)

if (ENABLE_OPEN_SRC)
    target_link_libraries(optimize PUBLIC
            -lgcov
            ascir
            ascgen_common
            graph
            ascir
            unified_dlog
            autofuse_runtime_stub
            error_manager
            atrace
            graph_base
            ascir_builtin_ops
    )
else ()
    target_link_libraries(optimize PUBLIC
            graph
            unified_dlog
            atrace
            graph_base
            c_sec
    )
endif ()

######################################autofuse##########################################
set(AUTOFUSE_DIR ${CODE_ROOT_DIR}/autofuse)
file(GLOB_RECURSE AUTOFUSE_SOURCES "${AUTOFUSE_DIR}/*.cpp")

add_library(autofuse SHARED
   ${AUTOFUSE_SOURCES}
)
target_compile_definitions(autofuse PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
)

target_include_directories(autofuse PUBLIC
    ${AUTOFUSE_DIR}
    ${CODE_ROOT_DIR}
    ${CODE_ROOT_DIR}/autofuse
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_compile_options(autofuse PRIVATE -fPIC -Werror -fno-common -Wextra -Wfloat-equal -O0)

if (ENABLE_OPEN_SRC)
    target_link_libraries(autofuse PUBLIC
        -lgcov
        mmpa_headers
        metadef_headers
        ascir_ops_headers
        ascend_protobuf
        c_sec
        json
        atrace
        unified_dlog
        error_manager
        graph_base
        graph
        aihac_symbolizer
        ascgen_common
        codegen
        optimize
        autofuse_runtime_stub
    )
else ()
    target_link_libraries(autofuse PUBLIC
        graph
        unified_dlog
        atrace_share
        graph_base
        mmpa_headers
        metadef_headers
        ascir_ops_headers
        atrace_headers
        error_manager
        ascend_protobuf
        c_sec
        json
        aihac_symbolizer
        aihac_codegen
        runtime_headers
    )
endif ()

####################################pyautofuse############################################
set(PYAUTOFUSE_DIR ${CODE_ROOT_DIR}/compiler/py_module/)

Python3_add_library(pyautofuse MODULE
        ${PYAUTOFUSE_DIR}/pyascir.cpp
        ${PYAUTOFUSE_DIR}/pyascir_types.cpp
        ${PYAUTOFUSE_DIR}/pyascir_common_utils.cpp
        ${PYAUTOFUSE_DIR}/pyautofuse.cpp
)
target_compile_options(pyautofuse PRIVATE -Wno-error=variadic-macros)
target_include_directories(pyautofuse PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/optimize
        ${CODE_ROOT_DIR}/codegen
        ${CODE_ROOT_DIR}/autofuse
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
)
target_compile_definitions(pyautofuse PRIVATE
        google=ascend_private
)
if (ENABLE_OPEN_SRC)
    target_link_libraries(pyautofuse PUBLIC
            -Wl,--no-as-needed
            $<BUILD_INTERFACE:ascir_ops_headers>
            $<BUILD_INTERFACE:runtime_headers>
            atrace
            ascgen_common
            att
            codegen
            autofuse
            aihac_ir
            ascend_protobuf
            json
            graph
            graph_base
            autofuse_runtime_stub
    )
else ()
    target_link_libraries(pyautofuse PRIVATE
            $<BUILD_INTERFACE:atrace_headers>
            $<BUILD_INTERFACE:ascir_ops_headers>
            $<BUILD_INTERFACE:c_sec_headers>
            atrace_share
            ascgen_common
            att
            codegen
            aihac_autofusion
            aihac_ir
            ascend_protobuf
            json
            graph
            graph_base
            unified_dlog
    )
endif ()
if (ENABLE_OPEN_SRC)
    target_compile_options(att PRIVATE -g --coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(optimize PRIVATE -g --coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(ascgen_common PUBLIC -g --coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(codegen PUBLIC -g --coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(pyautofuse PUBLIC -g --coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif ()

if (IS_DIRECTORY "${CODE_ROOT_DIR}/v35")
    add_subdirectory(v35)
endif()

