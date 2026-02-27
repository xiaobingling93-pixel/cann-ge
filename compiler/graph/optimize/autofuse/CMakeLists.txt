set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all -Wl,-z,now")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-all -Wl,-z,now")

if(NOT MDC_COMPILE_RUNTIME)
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
else()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
endif()

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
set(ASCEND_ROOT ${ASCEND_INSTALL_PATH}) # e.g. /usr/local/Ascend/ascend-toolkit/latest
set(ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH})

set(CODE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_OPEN_SRC)
    set(GE_METADEF_DIR ${CODE_ROOT_DIR}/../../../../graph_metadef)
    set(GE_METADEF_INC_DIR ${CODE_ROOT_DIR}/../../../../inc/graph_metadef)

    if (NOT DEFINED ASCEND_3RD_LIB_PATH)
        set(ASCEND_3RD_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/output/third_party)
    endif ()

    if (NOT EXISTS "${ASCEND_3RD_LIB_PATH}")
        message(FATAL_ERROR "Error: Directory \"${ASCEND_3RD_LIB_PATH}\" does not exist!\n"
                "Please using build_third_party.sh compile first.")
    endif ()
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
include_directories(${GE_METADEF_INC_DIR}/graph)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/utils)
include_directories(${GE_METADEF_INC_DIR}/graph/utils)
include_directories(${GE_METADEF_DIR}/graph)
include_directories(${CODE_ROOT_DIR}/common)
include_directories(${CODE_ROOT_DIR}/ascendc/api)
include_directories(${CODE_ROOT_DIR}/compiler/py_module)
include_directories(${CODE_ROOT_DIR}/ascir/meta)
include_directories(${CODE_ROOT_DIR}/att)
include_directories(${CODE_ROOT_DIR}/inc)
include_directories(${ASCEND_ROOT}/include)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc/base)

if(DEFINED BUILD_COMPONENT)
    include_directories(${ASCEND_ROOT}/pkg_inc)
    include_directories(${ASCEND_ROOT}/pkg_inc/runtime)
    include_directories(${ASCEND_ROOT}/pkg_inc/profiling)
else()
    include_directories(${ASCEND_ROOT}/include/experiment)
    include_directories(${ASCEND_ROOT}/include/experiment/runtime)
    include_directories(${ASCEND_ROOT}/include/experiment/msprof)
endif()
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (ENABLE_OPEN_SRC)
#    find_package(nlohmann_json REQUIRED)
    find_package(json REQUIRED)
    if(NOT BUILD_PKG_COMPONENT)
        find_package(GTest CONFIG REQUIRED)
    endif()
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

# Common depends
add_library(depends INTERFACE)
target_link_directories(depends INTERFACE
    ${ASCEND_ROOT}/x86_64-linux/devlib)

if (DEFINED RUN_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif ()

add_library(aihac_codegen SHARED)

add_subdirectory(ascendc)
add_subdirectory(ascir)
add_subdirectory(optimize)
add_subdirectory(att)
add_subdirectory(codegen)
add_subdirectory(common)
if (IS_DIRECTORY "${CODE_ROOT_DIR}/v35")
    add_subdirectory(v35)
endif()
# 判断 RUN_TEST 未定义或值不为 1
if(NOT DEFINED RUN_TEST OR NOT RUN_TEST EQUAL 1)
add_subdirectory(compiler)
endif ()

# todo: att代码中部分通过宏控制, 根目录CMakelist需要增加定义, 后续整改
add_definitions(-DFORMULAS_PERF)

target_compile_definitions(aihac_codegen PRIVATE
    google=ascend_private
)

target_compile_options(aihac_codegen PRIVATE
    -O2
    -fPIC
    -fno-common
    -Wfloat-equal
    -Wextra
    -Wall
    -Werror
)

target_link_options(aihac_codegen PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,All
    $<$<CONFIG:Release>:-s>
)

target_link_libraries(aihac_codegen PUBLIC
    $<BUILD_INTERFACE:ascir_ops_headers>
    $<BUILD_INTERFACE:atrace_headers>
    graph
    graph_base
    aihac_ir
    aihac_ir_register
    aihac_symbolizer
    $<BUILD_INTERFACE:ge_log_utils>
    $<BUILD_INTERFACE:ascendc_api_extend>
    error_manager
    c_sec
    unified_dlog
    runtime
    atrace_share
    mmpa
    json
    ascend_protobuf
    symengine
    Boost::boost
    -lgcov
)

if (NOT ENABLE_OPEN_SRC)
    target_link_libraries(aihac_codegen PUBLIC
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:msprof_headers>
    )
endif ()

add_subdirectory(autofuse)
##########################以下为业务代码生成的so################################
if(NOT DEFINED RUN_TEST OR NOT RUN_TEST EQUAL 1)
install(TARGETS aihac_codegen
                pyautofuse
    LIBRARY DESTINATION lib OPTIONAL
    ARCHIVE DESTINATION lib OPTIONAL
    RUNTIME DESTINATION bin OPTIONAL
)

# #########################以下内容是生成一个autofuse的so################################
install(TARGETS aihac_autofusion
        LIBRARY DESTINATION lib OPTIONAL
        ARCHIVE DESTINATION lib OPTIONAL
        RUNTIME DESTINATION bin OPTIONAL
)

if (NOT ENABLE_OPEN_SRC)
    install_package(
        PACKAGE ascgen
        TARGETS aihac_codegen pyautofuse aihac_autofusion
        DESTINATION ${INSTALL_INCLUDE_DIR}/ascgen
    )
endif ()

if (ENABLE_OPEN_SRC)
    set(ARCH_LINX_PATH "${CMAKE_SYSTEM_PROCESSOR}-linux")

    install(TARGETS aihac_autofusion
                    aihac_codegen
        LIBRARY DESTINATION ${ARCH_LINX_PATH}/lib64 OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
        ARCHIVE DESTINATION ${ARCH_LINX_PATH}/lib64 OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
        RUNTIME DESTINATION ${ARCH_LINX_PATH}/bin OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
    )

    install(TARGETS pyautofuse
        LIBRARY DESTINATION python/site-packages/autofuse OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
    )

    install(FILES compiler/python/__init__.py
            compiler/python/asc_codegen_compile.py
            compiler/python/ascbc_kernel_compile.py
            compiler/python/compile_adapter.py
            compiler/python/ascendc_compile.py
            compiler/python/ascir_api.py
            DESTINATION python/site-packages/autofuse OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
    )

    set(version_info "${ASCEND_INSTALL_PATH}/runtime/version.info")
    set(install_script_dir "${CMAKE_CURRENT_BINARY_DIR}/install_scripts/")

    add_custom_target(generate_install_script_autofuse ALL
        COMMAND rm -rf ${install_script_dir}
        COMMAND cp -rf ${ASCEND_INSTALL_PATH}/toolkit/tools/ascend_project/open_install_scripts ${install_script_dir}
        COMMAND chmod -R u+w ${install_script_dir}
        COMMAND echo "base_package=compiler" > ${install_script_dir}/version.info
        COMMAND echo "backup_dir=autofuse" >> ${install_script_dir}/version.info
        COMMAND grep -iE "'^(version|version_dir)='" ${version_info} >> ${install_script_dir}/version.info
    )

    install(DIRECTORY ${install_script_dir}
        DESTINATION . OPTIONAL
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
        COMPONENT . EXCLUDE_FROM_ALL
    )

    set(CPACK_COMPONENTS_ALL ".;packages")
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack autofuse project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack autofuse project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/package)
    set(CPACK_PACKAGE_FILE_NAME "CANN-autofuse-linux.${CMAKE_SYSTEM_PROCESSOR}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_INSTALL_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif()
endif()
