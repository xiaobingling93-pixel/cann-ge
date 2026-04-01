# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(proto_src_files
    ${METADEF_PROTO_DIR}/task.proto
)

set(proto_nodedef_src_files
    ${METADEF_PROTO_DIR}/aicpu/cpu_attr.proto
    ${METADEF_PROTO_DIR}/aicpu/cpu_node_def.proto
    ${METADEF_PROTO_DIR}/aicpu/cpu_tensor.proto
    ${METADEF_PROTO_DIR}/aicpu/cpu_tensor_shape.proto
)

protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${proto_src_files})
protobuf_generate(aicpu PROTO_NODEDEF_SRCS PROTO_NODEDEF_HDRS ${proto_nodedef_src_files})

set(local_lib_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/base_engine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_ops_kernel_info_store/kernel_info.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_graph_optimizer/aicpu_graph_optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_graph_optimizer/optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_graph_optimizer/graph_optimizer_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pass/concat_from_sequence_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/config/config_file.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_ops_kernel_builder/kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/config/ops_json_file.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/error_code/error_code.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_engine/common/kernel_builder/cpu_kernel_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_engine/common/optimizer/cpu_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_engine/common/util/cpu_engine_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/util/util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/config/ops_parallel_rule_json_file.cpp
    ${PROTO_HDRS}
    ${PROTO_NODEDEF_HDRS}
)

set(local_lib_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine
    ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/common/util
    ${CMAKE_BINARY_DIR}/proto/aicpu
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/pkg_inc
    ${METADEF_DIR}/inc/external
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

add_library(aicpu_engine_common SHARED
    ${local_lib_src_files}
)

##-----aicpu_engine_common
target_link_libraries(aicpu_engine_common PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    datagw_headers
    mmpa_headers
    c_sec
    slog_headers
    error_manager
    graph
    register
#    graph_base
    platform
    platform_headers
    ascend_protobuf
    aicore_utils
    mmpa
    json
    -lrt
    -ldl
)

##### target atclib/aicpu_engine_common
target_include_directories(aicpu_engine_common PRIVATE
    ${local_lib_inc_path}
    ${OP_PROTO_DIR}
)

target_compile_definitions(aicpu_engine_common PRIVATE
    AICPU_PLUGIN
    google=ascend_private
)

target_compile_options(aicpu_engine_common PRIVATE
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Werror=format
    -Wextra
    -Wfloat-equal
    -Wno-pmf-conversions
)

target_link_options(aicpu_engine_common PRIVATE
    -Wl,-Bsymbolic
)

install(TARGETS aicpu_engine_common OPTIONAL
    EXPORT aicpu_engine_common-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
)

set(init.conf_src_path  ${CMAKE_CURRENT_SOURCE_DIR}/config/init.conf)
install(FILES ${init.conf_src_path}
    DESTINATION ${INSTALL_LIBRARY_DIR}
    OPTIONAL
)

set(ops_parallel_json_src_path   ${CMAKE_CURRENT_SOURCE_DIR}/config/aicpu_ops_parallel_rule.json)
install(FILES ${ops_parallel_json_src_path}
    DESTINATION ${INSTALL_LIBRARY_DIR}
    OPTIONAL
)