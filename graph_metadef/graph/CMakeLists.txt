# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(${GE_METADEF_DIR}/cmake/build_type.cmake)

######### for base #############
set(GRAPH_BASE_SOURCE_LIST
    "type/types.cc"
    "normal_graph/anchor.cc"
    "attr/ge_attr_value.cc"
    "utils/args_format_desc.cc"
    "normal_graph/operator.cc"
    "normal_graph/operator_impl.cc"
    "attr/attr_store.cc"
    "attr/attr_group_serialize.cc"
    "attr/attr_group_base.cc"
    "attr/attr_group_serializer_registry.cc"
    "buffer/buffer.cc"
    "normal_graph/compute_graph.cc"
    "normal_graph/model.cc"
    "serialization/model_serialize.cc"
    "normal_graph/node.cc"
    "normal_graph/op_desc.cc"
    "ir/ir_meta.cc"
    "ir/ir_data_type_symbol_store.cc"
    "type/sym_dtype.cc"
    "attr/ge_attr_define.cc"
    "normal_graph/ge_tensor.cc"
    "buffer/graph_buffer.cc"
    "hcom/hcom_topo_info.cc"
    "common/large_bm.cc"
    "common/hyper_status.cc"
    "detail/attributes_holder.cc"
    "utils/anchor_utils.cc"
    "utils/enum_attr_utils.cc"
    "utils/graph_utils.cc"
    "utils/dumper/ge_graph_dumper.cc"
    "utils/trace/trace_manager.cc"
    "utils/ge_ir_utils.cc"
    "utils/node_utils.cc"
    "utils/type_utils.cc"
    "utils/tensor_utils.cc"
    "utils/constant_utils.cc"
    "utils/connection_matrix.cc"
    "utils/cycle_detector.cc"
    "utils/op_type_utils.cc"
    "utils/fast_node_utils.cc"
    "utils/execute_graph_utils.cc"
    "utils/execute_graph_adapter.cc"
    "utils/args_format_desc_utils.cc"
    "utils/inference_rule.cc"
    "option/ge_context.cc"
    "option/ge_local_context.cc"
    "option/optimization_option.cc"
    "option/optimization_option_info.cc"
    "serialization/attr_serializer.cc"
    "serialization/string_serializer.cc"
    "serialization/data_type_serializer.cc"
    "serialization/named_attrs_serializer.cc"
    "serialization/bool_serializer.cc"
    "serialization/buffer_serializer.cc"
    "serialization/float_serializer.cc"
    "serialization/int_serializer.cc"
    "serialization/tensor_serializer.cc"
    "serialization/tensor_desc_serializer.cc"
    "serialization/graph_serializer.cc"
    "serialization/list_value_serializer.cc"
    "serialization/list_list_int_serializer.cc"
    "serialization/list_list_float_serializer.cc"
    "serialization/attr_serializer_registry.cc"
    "serialization/utils/serialization_util.cc"
    "cache_policy/cache_policy.cc"
    "cache_policy/compile_cache_desc.cc"
    "cache_policy/policy_register.cc"
    "cache_policy/cache_state.cc"
    "cache_policy/policy_management/match_policy/match_policy_exact_only.cc"
    "cache_policy/policy_management/match_policy/match_policy_for_exactly_the_same.cc"
    "cache_policy/policy_management/aging_policy/aging_policy_lru.cc"
    "cache_policy/policy_management/aging_policy/aging_policy_lru_k.cc"
    "utils/profiler.cc"
    "normal_graph/tensor.cc"
    "utils/readable_dump.cc"
    "utils/tensor_value_utils.cc"
    "${GE_METADEF_DIR}/base/utils/type_utils_inner.cc"
    "${GE_METADEF_DIR}/base/utils/file_utils.cc"
    "${GE_METADEF_DIR}/base/utils/aligned_ptr.cc"
    "${GE_METADEF_DIR}/base/common/plugin/plugin_manager.cc"
)

SET(GRAPH_SOURCE_LIST
    "type/ascend_string.cc"
    "attr/attr_value.cc"
    "type/axis_type_info.cc"
    "normal_graph/custom_op_factory.cc"
    "normal_graph/custom_op_factory_impl.cc"
    "normal_graph/operator_factory.cc"
    "normal_graph/operator_factory_impl.cc"
    "normal_graph/graph.cc"
    "normal_graph/gnode.cc"
    "args_format/args_format_serializer.cc"
    "args_format/arg_desc_info.cc"
    "args_format/arg_desc_info_impl.cc"
    "refiner/format_refiner.cc"
    "context/inference_context.cc"
    "refiner/ref_relation.cc"
    "context/resource_context_mgr.cc"
    "context/runtime_inference_context.cc"
    "refiner/shape_refiner.cc"
    "ir/ir_definitions_recover.cc"
    "opsproto/opsproto_manager.cc"
    "utils/op_desc_utils.cc"
    "utils/tuning_utils.cc"
    "utils/ffts_graph_utils.cc"
    "utils/transformer_utils.cc"
    "utils/graph_utils_ex.cc"
    "utils/node_utils_ex.cc"
    "utils/op_desc_utils_ex.cc"
    "utils/graph_thread_pool.cc"
    "utils/multi_thread_graph_builder.cc"
    "utils/type_utils_ex.cc"
    "utils/screen_printer.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_range_according_to_format.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc"
    "${GE_METADEF_DIR}/ops/op_imp.cpp"
)

SET(FAST_GRAPH_SOURCE_LIST
    "fast_graph/fast_node.cc"
    "fast_graph/execute_graph.cc"
)

######### libgraph_base.so #############
add_library(graph_base SHARED
    ${GRAPH_BASE_SOURCE_LIST}
    ${FAST_GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj>
)

target_compile_options(graph_base PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(graph_base PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(graph_base PRIVATE
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${ASCEND_INSTALL_PATH}/include
)

target_link_options(graph_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph_base
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        c_sec
        slog
        json
        metadef
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

target_link_libraries(graph_base PRIVATE ascend_protobuf error_manager)
target_compile_options(graph_base PRIVATE -O2 -Werror -DNO_METADEF_ABI_COMPATIABLE)

######### libgraph_base.a #############
if (NOT ENABLE_OPEN_SRC)
    target_clone(graph_base graph_base_static STATIC)

    target_compile_options(graph_base_static PRIVATE
        -Os
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
    )

    set_target_properties(graph_base_static PROPERTIES
        OUTPUT_NAME graph_base
    )
endif ()

######### libgraph.so #############
add_library(graph SHARED
    ${GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_attr_group_protos_obj>
)
target_compile_options(graph PRIVATE -DNO_METADEF_ABI_COMPATIABLE)

target_compile_options(graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/include
)

target_link_options(graph PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        c_sec
        slog
        json
        platform
        metadef
        -Wl,--as-needed
        ascend_protobuf_shared_headers
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

target_link_libraries(graph PRIVATE graph_base error_manager)
target_compile_options(graph PRIVATE -O2 -Werror)

if (${ENABLE_OPEN_SRC} STREQUAL "True")
else()
    ######### libgraph.a #############
    add_library(graph_share SHARED
            ${GRAPH_BASE_SOURCE_LIST}
            ${GRAPH_SOURCE_LIST}
            ${FAST_GRAPH_SOURCE_LIST}
            $<TARGET_OBJECTS:metadef_graph_protos_obj>
            $<TARGET_OBJECTS:metadef_attr_group_protos_obj>
            )

    target_compile_options(graph_share PRIVATE
            $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
            $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
            $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
            $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

    target_compile_definitions(graph_share PRIVATE
            $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
            $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
            google=ascend_private
            $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
            $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
            )

    target_include_directories(graph_share PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}
            ${CMAKE_BINARY_DIR}
            ${CMAKE_BINARY_DIR}/proto/metadef_protos
            ${GE_METADEF_DIR}
            ${AIR_CODE_DIR}/inc/graph_metadef
            )

    target_link_options(graph_share PRIVATE
            -Wl,-Bsymbolic
            )

    target_link_libraries(graph_share
            PRIVATE
            intf_pub
            static_mmpa
            -Wl,--no-as-needed
            c_sec
            slog
            json
            platform
            metadef
            -Wl,--as-needed
            $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
            -ldl
            PUBLIC
            metadef_headers
            )

    target_clone(graph_share graph_static STATIC)

    target_link_libraries(graph_share PRIVATE ascend_protobuf error_manager)
    target_link_libraries(graph_static PRIVATE ascend_protobuf_static)

    target_compile_options(graph_share PRIVATE -O2)
    target_compile_options(graph_static PRIVATE $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>)

    set_target_properties(graph_static PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
            OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libgraph_share,graph_share>
            )
endif()

##############################################################
set(STUB_HEADER_LIST
    ${GE_METADEF_INC_DIR}/external/graph/ascend_string.h
    ${GE_METADEF_INC_DIR}/external/graph/attr_value.h
    ${GE_METADEF_INC_DIR}/external/graph/gnode.h
    ${GE_METADEF_INC_DIR}/external/graph/graph.h
    ${GE_METADEF_INC_DIR}/external/graph/inference_context.h
    ${GE_METADEF_INC_DIR}/external/graph/operator_factory.h
    ${GE_METADEF_INC_DIR}/external/graph/types.h
    ${GE_METADEF_INC_DIR}/external/hcom/hcom_topo_info.h
    ${GE_METADEF_INC_DIR}/graph/utils/op_desc_utils.h
    ${GE_METADEF_INC_DIR}/graph/utils/op_desc_utils_ex.h
    ${GE_METADEF_INC_DIR}/graph/utils/node_utils_ex.h
    ${GE_METADEF_INC_DIR}/graph/utils/graph_utils_ex.h
    ${GE_METADEF_INC_DIR}/graph/utils/tensor_adapter.h
    ${GE_METADEF_INC_DIR}/graph/shape_refiner.h
    ${GE_METADEF_INC_DIR}/graph/opsproto_manager.h
    ${GE_METADEF_INC_DIR}/graph/runtime_inference_context.h
    ${GE_METADEF_INC_DIR}/graph/args_format_desc.h
    ${GE_METADEF_INC_DIR}/graph/ir_definitions_recover.h
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/transformer/src/axis_constants.h
    ${GE_METADEF_INC_DIR}/common/screen_printer.h
)

# 当前开放的sample样例中仅链接libgraph桩包，考虑兼容性，将libgraph_base里的内容也打到libgraph桩包内。
# ascend031下会同时使用libgraph桩包和libgraph_base编译和执行样例，而这些符号会影响对libgraph_base的链接，
# 故目前先做隔离，在ascend031下不将这些打到libgraph桩包内。
if (NOT "${PRODUCT}" STREQUAL "ascend031")
    list(APPEND STUB_HEADER_LIST
        ${GE_METADEF_INC_DIR}/external/graph/operator.h
        ${GE_METADEF_INC_DIR}/external/graph/tensor.h
        ${GE_METADEF_INC_DIR}/graph/ge_tensor.h
    )
endif ()

list(TRANSFORM STUB_HEADER_LIST
    REPLACE "^.*/([^/]+)\\.h$" "${CMAKE_CURRENT_BINARY_DIR}/stub_\\1.cc"
    OUTPUT_VARIABLE STUB_SRC_LIST
)

add_custom_command(
    OUTPUT ${STUB_SRC_LIST}
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${GE_METADEF_DIR}/../tests/graph_metadef/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
            && echo "Generating stub files end."
)

add_custom_target(graph_stub DEPENDS ${STUB_SRC_LIST})

#############################################################

############ stub/libgraph.so ############
add_library(atc_stub_graph SHARED ${STUB_SRC_LIST})

add_dependencies(atc_stub_graph graph_stub)

target_include_directories(atc_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${GE_METADEF_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${ASCEND_INSTALL_PATH}/include
)

target_compile_options(atc_stub_graph PRIVATE
    -Wfloat-equal
    -fno-common
    -Os
    -Werror=return-type
)

target_link_libraries(atc_stub_graph
    PRIVATE
        intf_pub
        mmpa_headers
        c_sec
        slog_headers
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

set_target_properties(atc_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY atc_stub
)

############ stub/libgraph.a ############
if (NOT ENABLE_OPEN_SRC)
    target_clone(atc_stub_graph atc_stub_graph_static STATIC)

    add_dependencies(atc_stub_graph_static graph_stub)

    target_compile_options(atc_stub_graph_static PRIVATE
        -ffunction-sections
        -fdata-sections
    )

    set_target_properties(atc_stub_graph_static PROPERTIES
        OUTPUT_NAME graph
        ARCHIVE_OUTPUT_DIRECTORY atc_stub
    )
endif ()


############ fwk_stub/libgraph.so ############
add_library(fwk_stub_graph SHARED ${STUB_SRC_LIST})

add_dependencies(fwk_stub_graph graph_stub)

target_compile_options(fwk_stub_graph PRIVATE
    -Wfloat-equal
    -fno-common
)

target_include_directories(fwk_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${GE_METADEF_DIR}
    ${AIR_CODE_DIR}/inc
)

target_link_libraries(fwk_stub_graph
    PRIVATE
        intf_pub
        static_mmpa
        c_sec
        -Wl,--no-as-needed
        slog
        json
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

set_target_properties(fwk_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
)
add_subdirectory(ascendc_ir)
add_subdirectory(expression)

############ install ############
install(TARGETS atc_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)

install(TARGETS fwk_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/fwk_stub
)
