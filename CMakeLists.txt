# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(CANN-GRAPH-ENGINE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BASE_DIR ${CMAKE_CURRENT_LIST_DIR})

# 存在`.dev_env`文件说明是蓝区开发环境，并且调用了`prepare_dev_env.sh`完成了开发环境准备
# 除了设置好`.dev_env`中有的环境变量外，还会设置其他几个蓝区开发的必要变量
if(EXISTS "${CMAKE_SOURCE_DIR}/.dev_env")
    set(ENABLE_OPEN_SRC True)
    set(ENABLE_GE_UT ON)
    set(ENABLE_GE_ST ON)
    set(ENABLE_TEST True)
    # Read the file into a list of lines
    file(STRINGS "${CMAKE_SOURCE_DIR}/.dev_env" DEV_ENV_LINES)

    # Process each line
    foreach(line IN LISTS DEV_ENV_LINES)
        string(STRIP "${line}" line)
        if(line MATCHES "^([^=]+)=(.*)$")
            set(var_name "${CMAKE_MATCH_1}")
            set(var_value "${CMAKE_MATCH_2}")
            # Remove any leading/trailing whitespace
            string(STRIP "${var_name}" var_name)
            string(STRIP "${var_value}" var_value)
            # Set the variable
            set(${var_name} ${var_value})
        endif()
    endforeach()
 
    set(ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH}/latest)
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif(CCACHE_PROGRAM)

set(OPEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../open_source")

if(DEFINED BUILD_COMPONENT)
    message(STATUS "Build pkgs in divided mode")
    option(BUILD_PKG_COMPONENT "Build pkgs in divided mode" TRUE)
else()
    message(STATUS "Build pkg in normal mode")
    option(BUILD_PKG_COMPONENT "Build pkg in normal mode" FALSE)
endif()

if (NOT DEFINED CMAKE_MODULE_PATH OR NOT DEFINED CMAKE_PREFIX_PATH)
    # In a developer environment, neither CMAKE_MODULE_PATH nor CMAKE_PREFIX_PATH is set.
    if (NOT DEFINED ASCEND_INSTALL_PATH OR NOT DEFINED ASCEND_3RD_LIB_PATH)
        message(FATAL_ERROR "Environment setup incomplete.\n"
                "To complete the setup, please refer to https://gitcode.com/cann/ge/blob/master/docs/build.md#2-%E5%AE%89%E8%A3%85%E8%BD%AF%E4%BB%B6%E5%8C%85.")
    endif ()
endif ()

if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake/modules
        ${CMAKE_CURRENT_LIST_DIR}/cmake/third_party/scripts/modules
        ${ASCEND_3RD_LIB_PATH}/cmake/modules
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
        ${ASCEND_3RD_LIB_PATH}/benchmark
        ${ASCEND_3RD_LIB_PATH}/pybind11
        ${ASCEND_3RD_LIB_PATH}/symengine/lib/cmake/symengine
        ${ASCEND_3RD_LIB_PATH}/boost/lib/cmake/Boost-1.87.0
        ${ASCEND_3RD_LIB_PATH}/zlib
        ${ASCEND_INSTALL_PATH}
        ${ASCEND_INSTALL_PATH}/pkg_inc
    )
endif()

include(CMakePrintHelpers)
message(STATUS "Variables in air project:")
cmake_print_variables(ASCEND_INSTALL_PATH)
cmake_print_variables(ASCEND_3RD_LIB_PATH)
cmake_print_variables(CMAKE_BUILD_TYPE)
cmake_print_variables(CMAKE_INSTALL_PREFIX)
cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(CMAKE_MODULE_PATH)
cmake_print_variables(BUILD_COMPONENT)
cmake_print_variables(ENABLE_BUILD_DEVICE USE_CXX11_ABI CMAKE_TOOLCHAIN_FILE)
cmake_print_variables(CMAKE_CXX_COMPILER_ID)

if(ENABLE_BUILD_DEVICE)
    # 非MDC编译流程
    set(USE_CXX11_ABI 0)
else()
    string(FIND "${CMAKE_TOOLCHAIN_FILE}" "ubuntu18.04-x86_64-llvm-toolchain.cmake" SUBSTRING_FOUND)
    if(SUBSTRING_FOUND GREATER_EQUAL 0)
        set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY false)
        # MDC编译运行态
        message("[MDC compile] MDC compile runtime state.")
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
        set(MDC_COMPILE_RUNTIME ON)
    else()
        # MDC编译开发态
        message("[MDC compile] MDC compile develop state.")
        set(USE_CXX11_ABI 0)
    endif()
endif()

if (ENABLE_ACL_UT)
    include(cmake/intf_pub_linux.cmake)
    find_package(GTest CONFIG REQUIRED)
    find_package(securec MODULE REQUIRED)
    find_package(json MODULE REQUIRED)
    add_subdirectory(tests/acl_ut)
    return()
endif()

if (ENABLE_OPEN_SRC AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/output CACHE PATH "cmake default install path" FORCE)
endif ()

# 新开关适配，完成后删除原开关，从外部传入
if (ENABLE_TEST)
    enable_testing()
    set(GE_DT TRUE)
endif ()
if (ENABLE_FE_LLT)
    set(ENGINE_DT TRUE)
endif ()
if (ENABLE_GE_C_LLT)
    set(GE_C_DT TRUE)
endif ()

cmake_print_variables(ENABLE_OPEN_SRC ENABLE_TEST ENABLE_GE_BENCHMARK ENABLE_GE_ST ENABLE_GE_UT MINDSPORE_MODE PLATFORM
        BUILD_METADEF ENABLE_GE_DT ENABLE_RT2_UT ENABLE_PYTHON_UT ENABLE_PARSER_UT ENABLE_DFLOW_UT)
cmake_print_variables(BUILD_OPEN_PROJECT ENABLE_LLT_COV ENABLE_FE_LLT ENABLE_FFTS_LLT ENABLE_AICPU_LLT ENABLE_DVPP_LLT
        ENABLE_HCCE_LLT ENABLE_RTS_LLT ENABLE_UT ENABLE_ST, HI_PYTHON ENABLE_ASAN ENABLE_GCOV)

# 针对 Clang 编译器，关闭特定警告（避免 -Werror 触发编译错误）
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Add compile option for Clang.")
    add_compile_options(-Wno-error=unknown-warning-option)
    add_compile_options(-Wno-unused)
    add_compile_options(-Wno-unused-private-field)
    add_compile_options(-Wno-inconsistent-missing-override)
    add_compile_options(-Wno-vla)
    add_compile_options(-Wno-error=inconsistent-missing-override)
    add_compile_options(-Wno-error=vla)
    add_compile_options(-Wno-error=unused)
    add_compile_options(-Wno-error=unused-private-field)
    add_compile_options(-Wno-error)
endif()

set(CMAKE_VERBOSE_MAKEFILE True)

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_RUNTIME_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CONFIG_DIR cmake)

include(CMakePackageConfigHelpers)
include(cmake/intf_pub_linux.cmake)
include(cmake/build_type.cmake)

if (ENABLE_DFLOW_UT OR ENABLE_DFLOW_ST)
    set(AIR_COMMON_COMPILE_OPTION ${AIR_COMMON_COMPILE_OPTION} -DUDF_TEST)
endif()

set(AIR_CODE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(DFLOW_CODE_DIR ${CMAKE_CURRENT_LIST_DIR}/dflow)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

# Release场景下清除so中的RPATH，避免安全风险
if (NOT (CMAKE_BUILD_TYPE MATCHES GCOV) AND NOT ENABLE_GE_DT AND NOT RUN_TEST)
    set(CMAKE_SKIP_BUILD_RPATH TRUE)
    set(CMAKE_SKIP_INSTALL_RPATH TRUE)
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/air_headers.cmake)

option(BUILD_OPEN_PROJECT "Enable air compile in opensource." FALSE)

include(cmake/find_3rd_party_packages.cmake)
if ((CMAKE_BUILD_TYPE MATCHES GCOV) OR ENABLE_GE_DT)
    find_package(GTest CONFIG REQUIRED)
endif ()

if(BUILD_PKG_COMPONENT)
    find_package(acl_rt MODULE REQUIRED)
    find_package(awatchdog MODULE REQUIRED)
endif()

# CANN软件包
include(cmake/find_common_cann_packages.cmake)
if ((CMAKE_BUILD_TYPE MATCHES GCOV) OR ENABLE_GE_DT)
    include(cmake/stub_test_mode_cann_packages.cmake)
else ()
    include(cmake/find_release_mode_cann_packages.cmake)
endif ()

include(version.cmake)
if(BUILD_COMPONENT)
    check_pkg_build_deps(${BUILD_COMPONENT})
endif()
add_version_info_targets()

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    if (NOT DEFINED HI_PYTHON)
        set(HI_PYTHON python3)
    endif()
    set(TARGET_SYSTEM_NAME "Linux")
    set(OP_PROTO_DIR ${AIR_CODE_DIR}/tests/depends/op_stub/op_proto/inc)
    include(cmake/test_utils.cmake)
else ()
    set(METADEF_DIR ${TOP_DIR}/metadef)
    set(OP_PROTO_DIR ${TOP_DIR}/asl/ops/cann/ops/built-in/op_proto/inc)
endif ()

include(cmake/python_depends.cmake)

set(PARSER_DIR ${AIR_CODE_DIR}/parser)
set(ASCGEN_DIR ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse)
set(GE_METADEF_DIR ${AIR_CODE_DIR}/graph_metadef)
set(GE_METADEF_INC_DIR ${AIR_CODE_DIR}/inc/graph_metadef)
set(METADEF_PROTO_DIR ${GE_METADEF_DIR}/proto)

message("metadef proto dir is ${METADEF_PROTO_DIR}")

cmake_print_variables(METADEF_DIR)
cmake_print_variables(GE_METADEF_DIR)
cmake_print_variables(PARSER_DIR)
cmake_print_variables(METADEF_PROTO_DIR)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    add_subdirectory(${AIR_CODE_DIR}/inc/graph_metadef)
endif()

if (BUILD_METADEF)
    add_subdirectory(${GE_METADEF_DIR})
endif()

add_subdirectory(base)
add_subdirectory(api)
add_subdirectory(runtime)
add_subdirectory(compiler)
add_subdirectory(parser)
add_subdirectory(dflow)

if(DEFINED BUILD_COMPONENT)
    add_subdirectory(api/acl/acl_model)
    add_subdirectory(api/acl/acl_op_executor)
    add_subdirectory(api/acl/acl_cblas)
    add_subdirectory(api/acl/acl_op_compiler)
    add_subdirectory(api/acl/stub)
endif()

if (GE_DT)
    if ((CMAKE_BUILD_TYPE MATCHES GCOV) OR ENABLE_GE_DT)
        add_subdirectory(${GE_METADEF_DIR})
    endif()
    add_subdirectory(tests)
endif ()
if (GE_C_DT)
    add_subdirectory(tests/test_c)
endif ()
if (ENGINE_DT)
    add_subdirectory(tests/engines)
endif ()
if (ENABLE_OPEN_SRC AND RUN_TEST)
    add_subdirectory(tests/autofuse)
endif ()

add_custom_target(select_targets)
if ((CMAKE_BUILD_TYPE MATCHES GCOV) OR ENABLE_GE_DT)
    set(TOP_DIR ${CMAKE_SOURCE_DIR})
    # build dt
    if (ENABLE_FE_LLT AND ENABLE_UT)
        add_dependencies(select_targets te_fusion_stub graph_tuner_stub fe_ut)
    endif()
    if (ENABLE_FE_LLT AND ENABLE_ST)
        add_dependencies(select_targets te_fusion_stub graph_tuner_stub fe_st)
    endif()
    if (ENABLE_FE_LLT AND ENABLE_ST_WHOLE_PROCESS)
        add_dependencies(select_targets fe_st_whole_process)
    endif()
    if (ENABLE_FFTS_LLT AND ENABLE_UT)
        add_dependencies(select_targets ffts_ut)
    endif()
    if (ENABLE_FFTS_LLT AND ENABLE_ST)
        add_dependencies(select_targets ffts_st)
    endif()
    if (ENABLE_AICPU_LLT AND ENABLE_UT)
        add_dependencies(select_targets aicpu_ut)
    endif()
    if (ENABLE_AICPU_LLT AND ENABLE_ST)
        add_dependencies(select_targets aicpu_st)
    endif()
    if (ENABLE_DVPP_LLT AND (ENABLE_UT OR ENABLE_ST))
        add_dependencies(select_targets dvpp_ut)
    endif()
    if (ENABLE_RTS_LLT AND (ENABLE_UT OR ENABLE_ST))
        add_dependencies(select_targets rts_ut)
    endif()
    if (ENABLE_HCCE_LLT AND ENABLE_UT)
        add_dependencies(select_targets hcce_ut)
    endif()
    if (ENABLE_HCCE_LLT AND ENABLE_ST)
        add_dependencies(select_targets hcce_st)
    endif()

    if (ENABLE_GE_BENCHMARK)
        add_dependencies(select_targets ge_runtime_benchmark)

    elseif (ENABLE_GE_ST OR ENABLE_RT2_ST OR ENABLE_RT3_ST OR ENABLE_PYTHON_ST OR ENABLE_PARSER_ST OR ENABLE_DFLOW_ST)
        if (ENABLE_RT2_ST)
            add_dependencies(select_targets hybrid_model_async_exec_test
                    graphtuner_executor
                    st_fast_runtime2_test
                    st_dvpp_runtime2_test)
        endif ()

        if (ENABLE_RT3_ST)
            add_dependencies(select_targets ge_graph_dsl_test
                    ge_running_env_test
                    llm_engine_test
                    autofuse_test
                    ut_libge_kernel_utest
                    ut_jit_execution
                    ut_eager_style_builder
                    ut_libge_common_utest
                    ut_graph_construction
                    graphtuner_executor)
        endif ()

        if (ENABLE_PYTHON_ST)
            add_dependencies(select_targets llm_wrapper
                    graphtuner_executor
                    install_ge_py_for_tests)
        endif ()

        if (ENABLE_GE_ST)
            add_dependencies(select_targets graph_engine_test
                    graphtuner_executor)
        endif ()

        if (ENABLE_PARSER_ST)
            add_dependencies(select_targets graphtuner_executor st_parser graph_engine_compile_test)
        endif ()

        if(ENABLE_DFLOW_ST)
            add_dependencies(select_targets graphtuner_executor st_built_in_flow_func st_flow_func_executor helper_runtime_test)
        endif ()

    elseif (ENABLE_GE_UT OR ENABLE_RT2_UT OR ENABLE_PYTHON_UT OR ENABLE_PARSER_UT OR ENABLE_DFLOW_UT)
        if (ENABLE_GE_UT)
            add_dependencies(select_targets ut_libgraph
                    ut_libge_multiparts_utest
                    ut_libge_symbol_infer_utest
                    ut_libge_others_utest
                    ut_libge_distinct_load_utest
                    ut_libge_kernel_utest
                    ut_libge_label_maker_utest
                    ut_libge_common_utest
                    ut_jit_execution
                    ut_eager_style_builder
                    graphtuner_executor
                    ut_sc_check
                    ge_manual_test)
        endif ()

        if (ENABLE_RT2_UT)
            add_dependencies(select_targets ut_register ut_exe_graph ut_ascendc_ir ut_expression
                    ut_libge_kernel_utest
                    ut_libgraph
                    ut_llm_engine_utest
                    ut_fast_runtime2_test
                    ut_dvpp_runtime2_test)
        endif ()

        if (ENABLE_PYTHON_UT)
            add_dependencies(select_targets ut_libgraph
                    llm_engine
                    llm_wrapper
                    install_ge_py_for_tests)
        endif ()

        if (ENABLE_PARSER_UT)
            add_dependencies(select_targets ut_libgraph ut_parser ut_graph)
        endif ()

        if (ENABLE_DFLOW_UT)
            add_dependencies(select_targets ut_libgraph
            ge_ut_common
            ut_built_in_flow_func ut_flow_func_executor
            ut_libge_helper_utest
            ut_flow_graph)
        endif ()
    elseif (ENABLE_GE_DT)
        add_dependencies(select_targets ge_manual_test)
    elseif(ENABLE_GE_C_LLT)
        add_dependencies(select_targets ut_ge_executor_c_utest ut_ge_executor_c_liteos_utest ut_dbg_liteos_static_utest ut_ascendcl_c_utest)
    endif ()
else ()
    # build air run package
    if (BUILD_METADEF)
        add_dependencies(select_targets graph graph_base register exe_graph lowering rt2_registry_static)
    endif()

    add_dependencies(select_targets fmk_parser parser_common _caffe_parser fmk_onnx_parser)

    if(MINDSPORE_MODE)
        add_dependencies(select_targets ge_common ge_common_base graph graph_base)
    endif ()
endif ()

if (NOT BUILD_OPEN_PROJECT)
    # blue zone for ge
    if (ENABLE_OPEN_SRC)
        install(TARGETS gert ge_common ge_common_base ge_executor_shared davinci_executor hybrid_executor ge_runner
                ge_runner_v2 dflow_runner data_flow_base
                ge_compiler slice air_headers _caffe_parser fmk_onnx_parser fmk_parser parser_common
                EXPORT air-targets
                LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
                ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
                RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
                )
    # yellow zone for ge/fe/others
    else ()
        install(TARGETS gert ge_common ge_common_base ge_executor_shared davinci_executor hybrid_executor ge_runner
                ge_runner_v2 dflow_runner data_flow_base
                ge_compiler aicore_utils slice air_headers
                om2_executor
                # 开发环境某些没有下载parser目录
                # _caffe_parser fmk_onnx_parser fmk_parser parser_common
                EXPORT air-targets
                LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
                ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
                RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
                )

        if ("${PRODUCT}" STREQUAL "ascend031")
            install(TARGETS ge_executor ge_common_base_static davinci_executor_static
                ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
            )

            install(TARGETS ge_common_static_stub hybrid_executor_static_stub gert_static_stub
                ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub OPTIONAL
            )
        endif ()
    endif ()

    install(TARGETS fwk_stub_ge_runner
        EXPORT air-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub OPTIONAL COMPONENT opensdk
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/stub OPTIONAL COMPONENT opensdk
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/stub OPTIONAL COMPONENT opensdk
    )
    install(TARGETS fwk_stub_ge_runner_v2
        EXPORT air-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub OPTIONAL COMPONENT opensdk
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/stub OPTIONAL COMPONENT opensdk
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}/stub OPTIONAL COMPONENT opensdk
    )
    install(FILES   ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/aicore_util_attr_define.h
            ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/aicore_util_types.h
            ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/fe_error_code.h
            ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/fe_log.h
            ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/lxfusion_json_util.h
            ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/base_config_parser.h
            ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc/common/configuration.h
    DESTINATION ${INSTALL_INCLUDE_DIR}/air/nn_engine/common COMPONENT opensdk EXCLUDE_FROM_ALL
    )

    install(DIRECTORY ${AIR_CODE_DIR}/inc/ DESTINATION ${INSTALL_INCLUDE_DIR}/air
        COMPONENT opensdk EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.h"
    )
endif()

if (PACKAGE STREQUAL "opensdk")
    install(EXPORT air-targets DESTINATION ${INSTALL_CONFIG_DIR}
        FILE air-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
    )
    set(PKG_NAME air)
    configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/air-config.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    )
    unset(PKG_NAME)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/air-config.cmake
        DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
    )
endif()

if ((CMAKE_BUILD_TYPE MATCHES GCOV) OR ENABLE_GE_DT)
    message(STATUS "In test mode, skip installing")
else ((BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC))
    set(ARCH_LINX_PATH "${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(version_info "${ASCEND_INSTALL_PATH}/runtime/version.info")
    set(install_script_dir "${CMAKE_CURRENT_BINARY_DIR}/install_scripts/")

    add_custom_target(generate_install_script_air ALL
        COMMAND rm -rf ${install_script_dir}
        COMMAND cp -rf ${ASCEND_INSTALL_PATH}/toolkit/tools/ascend_project/open_install_scripts ${install_script_dir}
        COMMAND chmod -R u+w ${install_script_dir}
        COMMAND cp -rf ${CMAKE_CURRENT_LIST_DIR}/scripts/install.sh ${install_script_dir}
        COMMAND echo "base_package=runtime\;compiler" > ${install_script_dir}/version.info
        COMMAND echo "backup_dir=air" >> ${install_script_dir}/version.info
        COMMAND grep -iE "'^(version|version_dir)='" ${version_info} >> ${install_script_dir}/version.info
    )

    install(DIRECTORY ${install_script_dir}
        DESTINATION . OPTIONAL
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
        COMPONENT . EXCLUDE_FROM_ALL
    )

    if(BUILD_PKG_COMPONENT)
        if (NOT MDC_COMPILE_RUNTIME)
            add_custom_target(ge-compiler)
            add_dependencies(ge-compiler parser_common aicore_utils fusion_pass op_compile_adapter aicpu_engine_common fmk_parser ge_compiler
                    fmk_onnx_parser opskernel ge_runner slice aicpu_const_folding llm_engine jit_exe _caffe_parser flow_graph aihac_autofusion
                    dflow_runner eager_style_graph_builder_base eager_style_graph_builder_base_static gen_esb aihac_ir aihac_ir_register
                    aihac_symbolizer pyautofuse ge_python llm_datadist_python_v1 dataflow_python)
            add_dependencies(ge-compiler hcom_gradient_split_tune hcom_graph_adaptor hcom_opskernel_builder hcom_gradtune_opskernel_builder)
            add_dependencies(ge-compiler aicpu_ascend_engine aicpu_tf_engine dvpp_engine fe ffts ge_local_engine ge_local_opskernel_builder
                    host_cpu_opskernel_builder host_cpu_engine engine)
            add_dependencies(ge-compiler cpu_compiler udf_compiler)
            add_dependencies(ge-compiler atc_atc.bin fwk_atc.bin)
            add_dependencies(ge-compiler compress compress_static compressweight compressweight_static rts_engine)
            add_dependencies(ge-compiler acl_op_compiler)
            add_dependencies(ge-compiler fmk_onnx_parser_stub fmk_parser_stub atc_stub_ge_compiler fwk_stub_ge_runner fwk_stub_ge_runner_v2
                    stub_acl_op_compiler)

            add_custom_target(ge-executor)
            add_dependencies(ge-executor ge_common ge_executor_shared ge_common_base davinci_executor hybrid_executor gert om2_executor register
                    graph npu_sched_model_loader lowering register_static graph_base model_deployer data_flow_base hcom_executor)
            add_dependencies(ge-executor acl_mdl acl_mdl_impl acl_op_executor acl_op_executor_impl acl_cblas)
            add_dependencies(ge-executor ge_common_stub stub_lowering atc_stub_graph gert_stub hybrid_executor_stub stub_register stub_acl_mdl
                    stub_acl_cblas stub_acl_op_executor)
            if(ENABLE_BUILD_DEVICE) # 交叉编译
                add_dependencies(ge-executor runtime_update_model_param_dav_2201)
                add_dependencies(ge-compiler switch_by_index)
            endif()
        else()
            # MDC 运行态编译
            message("[MDC compile] MDC compile target set.")
            add_custom_target(ge-compiler)
            add_dependencies(ge-compiler flow_graph)

            add_custom_target(ge-executor)
            add_dependencies(ge-executor ge_common ge_common_base davinci_executor hybrid_executor gert register graph graph_base acl_cblas)
            add_dependencies(ge-executor acl_mdl acl_mdl_impl acl_op_executor acl_op_executor_impl om2_executor ge_executor_shared lowering)
        endif ()

        add_dependencies(ge-compiler version_ge-compiler_info)
        add_dependencies(ge-executor version_ge-executor_info)

        add_custom_target(dflow-executor)	 
        add_dependencies(dflow-executor deployer_daemon npu_executor_main host_cpu_executor_main udf_dump	 
                reader_writer udf_profiling flow_func built_in_flowfunc udf_executor) 
        add_dependencies(dflow-executor version_dflow-executor_info)
        if (DEFINED ENV{TOOLCHAIN_DIR} AND ENABLE_BUILD_DEVICE)
            message(STATUS $ENV{TOOLCHAIN_DIR})
            set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
            set(CHILD_INSTALL_DIR ${CMAKE_BINARY_DIR}/device_install)
            cmake_print_variables(TOOLCHAIN_DIR ENABLE_SIGN CUSTOM_SIGN_SCRIPT)
            if (NOT DEFINED TOP_DIR)
                set(TOP_DIR ${CMAKE_SOURCE_DIR})
                message(STATUS "TOP_DIR is not exists, set it to ${CMAKE_SOURCE_DIR}")
            endif ()
            include(ExternalProject)
            ExternalProject_Add(udf_device
                    SOURCE_DIR ${CMAKE_SOURCE_DIR}/dflow/udf/cmake/device
                    BINARY_DIR ${CMAKE_BINARY_DIR}/device_build
                    CMAKE_ARGS
                    -DTOP_DIR=${TOP_DIR}
                    -DTOOLCHAIN_DIR=${TOOLCHAIN_DIR}
                    -DCMAKE_INSTALL_PREFIX=${CHILD_INSTALL_DIR}
                    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_SOURCE_DIR}/dflow/udf/cmake/aarch64-hcc-toolchain.cmake
                    -DOPEN_SOURCE_DIR=${ASCEND_3RD_LIB_PATH}
                    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                    -DINSTALL_LIBRARY_DIR=${INSTALL_LIBRARY_DIR}
                    -DASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DHI_PYTHON=${HI_PYTHON}
                    -DENABLE_SIGN=${ENABLE_SIGN}
                    -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT}
                    -DVERSION_INFO=${VERSION_INFO}
                    BUILD_ALWAYS TRUE)
            add_dependencies(dflow-executor udf_device)
        else ()
            message(WARNING "ENV TOOLCHAIN_DIR is not defined, can't build udf device pkg!!!")
        endif ()
        
        include(cmake/package.cmake)
    endif()
endif()
