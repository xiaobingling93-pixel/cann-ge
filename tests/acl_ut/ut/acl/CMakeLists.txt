# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(acl_utest)

set(CMAKE_CXX_STANDARD 17)

set(UT_FILES
    "testcase/test.cpp"
    "testcase/compatibility/enum_check.cpp"
    "testcase/compatibility/const_check.cpp"
    "testcase/compatibility/struct_check.cpp"
    "testcase/acl_common_unittest.cpp"
    "testcase/acl_utils_unittest.cpp"
    "testcase/acl_model_unittest.cpp"
    "testcase/acl_opCompile_unittest.cpp"
    "testcase/acl_opAttr_unittest.cpp"
    "testcase/acl_opExecutor_unittest.cpp"
    "testcase/acl_opModelCache_unittest.cpp"
    "testcase/acl_opModelParser_unittest.cpp"
    "testcase/acl_op_unittest.cpp"
    "testcase/acl_builtinOp_unittest.cpp"
    "testcase/acl_blas_unittest.cpp"
    "testcase/acl_resource_manager_unittest.cpp"
)

set(MODEL_SRC_FILES
    "${BASE_DIR}/api/acl/acl_model/model/model.cpp"
    "${BASE_DIR}/api/acl/acl_model/model/acl_model.cpp"
    "${BASE_DIR}/api/acl/acl_model/model/acl_aipp.cpp"
    "${BASE_DIR}/api/acl/acl_model/model/aipp_param_check.cpp"
    "${BASE_DIR}/api/acl/acl_model/model/model_config.cpp"
    "${BASE_DIR}/api/acl/acl_model/model/acl_resource_manager.cpp"
    "${BASE_DIR}/api/acl/acl_model/model/init_callback_register.cpp"
    "${BASE_DIR}/api/acl/acl_model/types/tensor_desc_internal.cpp"
    "${BASE_DIR}/api/acl/common/json_parser.cpp"
    "${BASE_DIR}/api/acl/common/log_inner.cpp"
    "${BASE_DIR}/api/acl/utils/string_utils.cpp"
    "${BASE_DIR}/api/acl/common/prof_api_reg.cpp"
)

set(CBLAS_SRC_FILES
    "${BASE_DIR}/api/acl/acl_cblas/single_op/blas/gemm_ops.cpp"
    "${BASE_DIR}/api/acl/acl_cblas/single_op/blas/gemv_ops.cpp"
    "${BASE_DIR}/api/acl/common/prof_api_reg.cpp"
    "${BASE_DIR}/api/acl/common/log_inner.cpp"
)

set(OP_COMPILE_SRC_FILES
    "${BASE_DIR}/api/acl/acl_op_compiler/single_op/op_compiler.cpp"
    "${BASE_DIR}/api/acl/acl_op_compiler/single_op/compile/local_compiler.cpp"
    "${BASE_DIR}/api/acl/acl_op_compiler/single_op/compile/op_compile_processor.cpp"
    "${BASE_DIR}/api/acl/acl_op_compiler/single_op/compile/init_callback_register.cpp"
    "${BASE_DIR}/api/acl/common/log_inner.cpp"
    "${BASE_DIR}/api/acl/common/json_parser.cpp"
    "${BASE_DIR}/api/acl/common/prof_api_reg.cpp"
    "${BASE_DIR}/api/acl/utils/attr_utils.cpp"
    "${BASE_DIR}/api/acl/utils/array_utils.cpp"
    "${BASE_DIR}/api/acl/utils/hash_utils.cpp"
)

set(OP_EXEC_SRC_FILES
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/acl_op_executor.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/op.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/op_executor.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/op_model_cache.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/ge_tensor_cache.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/acl_op_resource_manager.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/executor/init_callback_register.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/shape_range_utils.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/builtin/cast_op.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/op_model_parser.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/compile/op_kernel_selector.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/compile/op_kernel_registry.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/compile/op_compiler.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/compile/op_compile_service.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/executor/stream_executor.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/executor/resource_manager.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/single_op/executor/op_task.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/types/op_attr.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/types/op_model.cpp"
    "${BASE_DIR}/api/acl/acl_op_executor/types/acl_op.cpp"
    "${BASE_DIR}/api/acl/common/common_inner.cpp"
    "${BASE_DIR}/api/acl/common/log_inner.cpp"
    "${BASE_DIR}/api/acl/common/json_parser.cpp"
    "${BASE_DIR}/api/acl/utils/attr_utils.cpp"
    "${BASE_DIR}/api/acl/utils/array_utils.cpp"
    "${BASE_DIR}/api/acl/utils/hash_utils.cpp"
    "${BASE_DIR}/api/acl/utils/string_utils.cpp"
    "${BASE_DIR}/api/acl/utils/file_utils.cpp"
    "${BASE_DIR}/api/acl/common/prof_api_reg.cpp"
)

############ libut_acl_model_src.so ############
add_library(ut_acl_model_src SHARED
    ${MODEL_SRC_FILES}
)

target_include_directories(ut_acl_model_src PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/framework
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/graph_metadef/external
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/external/ge
    ${BASE_DIR}/inc/external/acl
    ${BASE_DIR}/api/acl
    ${BASE_DIR}/api/acl/acl_model
    ${BASE_DIR}/api/acl/common
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_definitions(ut_acl_model_src PRIVATE
    google=ascend_private
    ENABLE_DVPP_INTERFACE
    RUN_TEST
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_compile_options(ut_acl_model_src PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -Wall -Wextra -Wfloat-equal -Werror
)

target_link_libraries(ut_acl_model_src PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    c_sec
    json
    -lgcov
)

############ libut_acl_cblas_src.so ############
add_library(ut_acl_cblas_src SHARED
    ${CBLAS_SRC_FILES}
)

target_include_directories(ut_acl_cblas_src PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/external/acl
    ${BASE_DIR}/api/acl
    ${BASE_DIR}/api/acl/acl_cblas
    ${BASE_DIR}/api/acl/acl_op_executor
    ${BASE_DIR}/api/acl/acl_model
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_definitions(ut_acl_cblas_src PRIVATE
    google=ascend_private
    ENABLE_DVPP_INTERFACE
    RUN_TEST
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_compile_options(ut_acl_cblas_src PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -Wall -Wextra -Wfloat-equal -Werror
)

target_link_libraries(ut_acl_cblas_src PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    ut_acl_model_src
    c_sec
    json
    -lgcov
)

############ libut_acl_op_exec_src.so ############
add_library(ut_acl_op_exec_src SHARED
    ${OP_EXEC_SRC_FILES}
)

target_include_directories(ut_acl_op_exec_src PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/graph_metadef/external
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/external/ge
    ${BASE_DIR}/inc/framework
    ${BASE_DIR}/api/acl
    ${BASE_DIR}/api/acl/acl_op_executor
    ${BASE_DIR}/api/acl/common
    ${BASE_DIR}/api/acl/acl_model
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/watchdog
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_definitions(ut_acl_op_exec_src PRIVATE
    google=ascend_private
    ENABLE_DVPP_INTERFACE
    RUN_TEST
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_compile_options(ut_acl_op_exec_src PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -Wall -Wextra -Wfloat-equal -Werror
)

target_link_libraries(ut_acl_op_exec_src PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    ut_acl_model_src
    c_sec
    json
    -lgcov
)

############ libut_acl_op_compiler_src.so ############
add_library(ut_acl_op_compiler_src SHARED
    ${OP_COMPILE_SRC_FILES}
)

target_include_directories(ut_acl_op_compiler_src PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/graph_metadef/external
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/framework
    ${BASE_DIR}/api/acl
    ${BASE_DIR}/api/acl/acl_model
    ${BASE_DIR}/api/acl/acl_op_compiler
    ${BASE_DIR}/api/acl/acl_op_executor
    ${BASE_DIR}/api/acl/common
    ${BASE_DIR}/api/acl/acl_op_executor/single_op
    ${BASE_DIR}/api/acl/acl_op_executor/single_op/compile
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_definitions(ut_acl_op_compiler_src PRIVATE
    google=ascend_private
    ENABLE_DVPP_INTERFACE
    RUN_TEST
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_compile_options(ut_acl_op_compiler_src PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -Wall -Wextra -Wfloat-equal -Werror
)

target_link_libraries(ut_acl_op_compiler_src PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    ut_acl_op_exec_src
    c_sec
    json
    -lgcov
)

############ acl_utest ############
add_executable(acl_utest ${UT_FILES})

target_include_directories(acl_utest PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/graph_metadef/external
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/external/acl
    ${BASE_DIR}/inc/framework
    ${BASE_DIR}/tests/acl_ut/depends
    ${BASE_DIR}/api/acl
    ${BASE_DIR}/api/acl/acl_model
    ${BASE_DIR}/api/acl/acl_op_compiler
    ${BASE_DIR}/api/acl/acl_op_executor
    ${BASE_DIR}/api/acl/acl_op_executor/single_op
    ${BASE_DIR}/api/acl/acl_op_executor/single_op/compile
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/dump
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_options(acl_utest PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -O0
    -g
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror -Wextra
    -fno-access-control
)

target_compile_definitions(acl_utest PRIVATE
    google=ascend_private
    ENABLE_DVPP_INTERFACE
    RUN_TEST
    ACL_BASE_DIR=\"${BASE_DIR}\"
)

target_link_libraries(acl_utest PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -Wl,--whole-archive
    c_sec
    ut_acl_model_src
    ut_acl_op_exec_src
    ut_acl_cblas_src
    ut_acl_op_compiler_src
    -Wl,--no-whole-archive
    json
    acl_stub
    slog_stub
    mmpa_stub
    ge_stub
    toolchain_stub
    runtime_stub
    profiling_stub
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main
    -lgcov
)
