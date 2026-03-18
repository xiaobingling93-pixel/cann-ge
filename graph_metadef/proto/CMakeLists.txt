# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(METADEF_PROTO_LIST
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/om.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/insert_op.proto"
        "${METADEF_PROTO_DIR}/dump_task.proto"
        "${METADEF_PROTO_DIR}/fwk_adapter.proto"
        "${METADEF_PROTO_DIR}/op_mapping.proto"
        "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
        "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
        "${METADEF_PROTO_DIR}/tensorflow/function.proto"
        "${METADEF_PROTO_DIR}/tensorflow/graph.proto"
        "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
        "${METADEF_PROTO_DIR}/tensorflow/op_def.proto"
        "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
        "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
        "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
        "${METADEF_PROTO_DIR}/tensorflow/types.proto"
        "${METADEF_PROTO_DIR}/tensorflow/versions.proto"
        "${METADEF_PROTO_DIR}/var_manager.proto"
        "${METADEF_PROTO_DIR}/flow_model.proto"
        "${METADEF_PROTO_DIR}/aicpu/cpu_attr.proto"
        "${METADEF_PROTO_DIR}/aicpu/cpu_node_def.proto"
        "${METADEF_PROTO_DIR}/aicpu/cpu_tensor_shape.proto"
        "${METADEF_PROTO_DIR}/aicpu/cpu_tensor.proto"
        "${METADEF_PROTO_DIR}/ascendc_ir.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto"
        )

set(METADEF_ATTR_GROUP_PROTO_LIST
        "${METADEF_PROTO_DIR}/attr_group_base.proto"
        )

protobuf_generate(metadef_protos METADEF_PROTO_SRCS METADEF_PROTO_HDRS ${METADEF_PROTO_LIST} ${METADEF_ATTR_GROUP_PROTO_LIST} TARGET)

set(METADEF_GRAPH_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/om.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/ge_ir.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/ascendc_ir.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/insert_op.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/task.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/fwk_adapter.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/onnx/ge_onnx.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/var_manager.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/flow_model.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/ge_ir_mobile.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/task_mobile.pb.cc"
        )

if(NOT (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC))
    list(APPEND METADEF_GRAPH_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/dump_task.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/op_mapping.pb.cc"
    )
endif()

set(METADEF_TENSORFLOW_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/graph.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/node_def.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/function.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/versions.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/attr_value.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/op_def.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/tensor.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/tensor_shape.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/types.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/resource_handle.pb.cc"
        )

set(METADEF_AICPU_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_attr.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_node_def.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_tensor_shape.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_tensor.pb.cc"
        )

set(METADEF_ATTR_GROUP_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/attr_group_base.pb.cc"
        )

add_library(metadef_graph_protos_obj OBJECT ${METADEF_GRAPH_PROTO_SRCS})
add_dependencies(metadef_graph_protos_obj metadef_protos)
target_compile_definitions(metadef_graph_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_graph_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_graph_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

add_library(metadef_tensorflow_protos_obj OBJECT ${METADEF_TENSORFLOW_PROTO_SRCS})
add_dependencies(metadef_tensorflow_protos_obj metadef_protos)
target_compile_definitions(metadef_tensorflow_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_tensorflow_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_tensorflow_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

add_library(metadef_aicpu_protos_obj OBJECT ${METADEF_AICPU_PROTO_SRCS})
add_dependencies(metadef_aicpu_protos_obj metadef_protos)
target_compile_definitions(metadef_aicpu_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_aicpu_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_aicpu_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

add_library(metadef_attr_group_protos_obj OBJECT ${METADEF_ATTR_GROUP_PROTO_SRCS})
add_dependencies(metadef_attr_group_protos_obj metadef_protos)
target_compile_definitions(metadef_attr_group_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_attr_group_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_attr_group_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

############ stub/libop_common.so ############
set(STUB_SRC_LIST
    ${CMAKE_CURRENT_BINARY_DIR}/stub_common_infershape_fns.cc
)

add_custom_command(
    OUTPUT ${STUB_SRC_LIST}
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${GE_METADEF_INC_DIR}/external ${CMAKE_CURRENT_BINARY_DIR}
            && echo "Generating stub files end."
)

add_custom_target(op_common_stub DEPENDS ${STUB_SRC_LIST})

add_library(stub_op_common SHARED ${STUB_SRC_LIST})

add_dependencies(stub_op_common op_common_stub)

target_include_directories(stub_op_common PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
)

target_compile_options(stub_op_common PRIVATE
    -Wfloat-equal
    -fno-common
    -Werror=return-type
    -Wno-unused-parameter
)

target_link_libraries(stub_op_common
    PRIVATE
        intf_pub
        c_sec
    PUBLIC
        metadef_headers
)

set_target_properties(stub_op_common PROPERTIES
    OUTPUT_NAME op_common
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ install ############
install(TARGETS stub_op_common OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)
