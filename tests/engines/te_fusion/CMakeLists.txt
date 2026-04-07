# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ENABLE_LLT_COV)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${COVERAGE_COMPILER_FLAGS}")
endif()

if (NOT DEFINED ASCEND_INSTALL_PATH)
    message("Find ge_compiler in fix path /usr/local/Ascend/cann/lib64")
    find_module(ge_compiler_pkg libge_compiler.so /usr/local/Ascend/cann/lib64)
else()
    message("Find ge_compiler in ascend install path ${ASCEND_INSTALL_PATH}/lib64")
    find_module(ge_compiler_pkg libge_compiler.so ${ASCEND_INSTALL_PATH}/lib64)
endif()

if (NOT DEFINED ASCEND_INSTALL_PATH)
    message("Find ge_common in fix path /usr/local/Ascend/cann/lib64")
    find_module(ge_common_pkg libge_common.so /usr/local/Ascend/cann/lib64)
else()
    message("Find ge_common in ascend install path ${ASCEND_INSTALL_PATH}/lib64")
    find_module(ge_common_pkg libge_common.so ${ASCEND_INSTALL_PATH}/lib64)
endif()

set(OP_COMPILE_ADAPTER_ROOT_PATH ${AIR_CODE_DIR}/compiler/opcompiler/op_compile_adapter)
set(TEFUSION_TEST_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR})

set(TEFUSION_SRC_CC_LIST
        ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/python_adapter/py_decouple.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/python_adapter/py_object_utils.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/python_adapter/py_wrapper.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/python_adapter/pyobj_assemble_utils.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/python_adapter/python_adapter_manager.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/python_adapter/python_api_call.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/api/fusion_api.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/common_utils.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/compile_result_utils.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/fusion_common.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/signal_manager.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/tbe_op_info.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/tbe_op_info_cache.cc
 	  	${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/tbe_op_info_hash.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/te_config_info.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/te_context_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/common/te_file_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/dfxinfo_manager/trace_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/dfxinfo_manager/dfxinfo_manager.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/binary_manager.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/fusion_binary_info.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/fusion_build_binary.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/fusion_fuzzy_build.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/fusion_reuse_binary_check.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/fusion_reuse_single_op_binary.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/generate_simple_key.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/binary/shape_generalization.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/compile/fusion_manager.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/compile/prebuild_manager.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/compile/superkernel_task.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/compile/te_compile_task_cache.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/cache/te_cache_manager.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/cache/te_cache_space_manager.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/cache/te_cache_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/assemble_json/te_attr_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/assemble_json/te_json_assemble.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/assemble_json/te_json_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/assemble_json/te_op_custom_utils.cc
 	    ${OP_COMPILE_ADAPTER_ROOT_PATH}/source/file_handle/te_file_handle.cc)

include(ut/testcase/op_compile_adapter/cmake/third_party/boost.cmake)
include(ut/testcase/op_compile_adapter/cmake/third_party/mockcpp.cmake)

SET(FE_ROOT_PATH ${AIR_CODE_DIR}/compiler/engines/nn_engine)
set(FE_TEST_ROOT_PATH ${AIR_CODE_DIR}/tests/engines/nn_engine)
add_subdirectory(depends/slog)
add_subdirectory(depends/mmpa)
add_subdirectory(depends/runtime)
add_subdirectory(depends/cann_kb)
add_subdirectory(depends/op_tiling)
add_subdirectory(depends/trace)
add_subdirectory(depends/ascendcl)

add_subdirectory(ut)
add_subdirectory(st)
