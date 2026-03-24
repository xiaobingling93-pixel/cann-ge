# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(../cmake/function.cmake)
add_subdirectory(slice)
if (BUILD_METADEF AND (${METADEF_DIR} STREQUAL ${GE_METADEF_DIR}))
    add_subdirectory(metadef)
endif()

##################################################################
set(SRC_FORMAT_TRANSFERS
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwcn_fractal_z_3D.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwnc_fractal_z_3D_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z_tbe.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fz_c04.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/formats.cc"
    "${AIR_CODE_DIR}/base/formats/ge_format_util.cc"
    "${AIR_CODE_DIR}/base/formats/register_format_transfer.cc"
    "${AIR_CODE_DIR}/base/formats/utils/formats_trans_utils.cc"
)

set(SRC_LIST
    "common/b_cast/b_cast.cc"
    "common/plugin/runtime_plugin_loader.cc"
    "common/plugin/plugin_caller.cc"
    "common/omg_util/omg_util.cc"
    "common/context/properties_manager.cc"
    "common/op/transop_util.cc"
    "common/single_op_parser.cc"
    "common/platform_info_util/platform_info_util.cc"
    "common/op_tiling/op_tiling_rt2.cc"
    "common/op_tiling/tiling_memcheck.cc"
    "common/op_tiling/tiling_dfx.cc"
    "common/error_tracking/error_tracking.cc"
    "graph/bin_cache/node_bin_selector_factory.cc"
    "graph/bin_cache/node_compile_cache_module.cc"
    "graph/bin_cache/op_binary_cache.cc"
    "host_cpu_engine/host_cpu_engine.cc"
    "common/option_supportion_checker/option_supportion_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/fusion_pass_reg.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/pass_registry.cc"
)

file(GLOB_RECURSE OM2_SRCS CONFIGURE_DEPENDS
        "common/om2/*.cc"
        "common/helper/om2/*.cc"
        "common/helper/om2_package_helper.cc"
)

set(SRC_BASE_LIST
    "common/context/ctx.cc"
    "common/context/local_context.cc"
    "common/helper/file_saver.cc"
    "common/tbe_handle_store/cust_aicpu_kernel_store.cc"
    "common/debug/memory_dumper.cc"
    "common/dump/dump_manager.cc"
    "common/dump/dump_op.cc"
    "common/dump/exception_dumper.cc"
    "common/dump/dump_properties.cc"
    "common/fmk_error_codes.cc"
    "common/const_place_holder_utils/const_place_holder_utils.cc"
    "common/datatype_transfer/datatype_transfer.cc"
    "common/fp16_t/fp16_t.cc"
    "common/math/hif8_t.cc"
    "common/plugin/datatype_util.cc"
    "common/plugin/op_tiling_manager.cc"
    "common/ge_inner_error_codes.cc"
    "common/helper/model_helper.cc"
    "common/helper/pre_model_helper.cc"
    "common/helper/nano_model_save_helper.cc"
    "common/helper/om_file_helper.cc"
    "common/tbe_handle_store/kernel_store.cc"
    "common/model/ge_model.cc"
    "common/model/model_introduction.cc"
    "common/model/model_compress_manager.cc"
    "common/model/tlv/model_tensor_descs_value.cc"
    "common/model/tlv/vec_int_int_value.cc"
    "common/model/tlv/vec_int_value.cc"
    "common/model/tlv/vec_str_value.cc"
    "common/model/ge_root_model.cc"
    "common/model/external_allocator_manager.cc"
    "common/global_variables/diagnose_switch.cc"
    "exec_runtime/execution_runtime_utils.cc"
    "common/helper/model_parser_base.cc"
    "common/helper/model_saver.cc"
    "common/op/ge_op_utils.cc"
    "common/tbe_handle_store/tbe_kernel_store.cc"
    "common/kernel_handles_manager/aicore_kernel_handles_manager.cc"
    "common/kernel_handles_manager/aicpu_kernel_handles_manager.cc"
    "common/kernel_handles_manager/cust_aicpu_kernel_handles_manager.cc"
    "common/kernel_handles_manager/kernel_handles_manager.cc"
    "common/kernel_handles_manager/kernel_handle_utils.cc"
    "common/thread_pool/thread_pool.cc"
    "common/types.cc"
    "common/util.cc"
    "common/proto_util/proto_util.cc"
    "common/file_constant_utils/file_constant_utils.cc"
    "common/tbe_handle_store/tbe_handle_store.cc"
    "common/tbe_handle_store/bin_register_utils.cc"
    "graph/manager/graph_var_manager.cc"
    "graph/manager/session_id_manager.cc"
    "graph/manager/graph_external_weight_manager.cc"
    "graph/manager/graph_manager_utils.cc"
    "graph/build/memory/var_mem_assign_util.cc"
    "graph/unfold/graph_unfolder.cc"
    "common/profiling/profiling_properties.cc"
    "common/profiling/global_profiler.cc"
    "common/profiling/device_memory_recorder.cc"
    "common/dump/global_dumper.cc"
    "common/profiling/profiling_definitions.cc"
    "common/op_so_store/op_so_store.cc"
    "common/preload/partition/model_partition_base.cc"
    "common/preload/partition/model_desc_partition.cc"
    "common/preload/partition/model_desc_extend_partition.cc"
    "common/preload/partition/model_kernel_args_partition.cc"
    "common/preload/partition/model_kernel_bin_partition.cc"
    "common/preload/partition/model_task_desc_partition.cc"
    "common/preload/partition/nano_task_desc_partition.cc"
    "common/preload/task_info/pre_generate_task_registry.cc"
    "common/preload/task_info/pre_task_status.cc"
    "common/preload/task_info/aicore/aicore_task_info.cc"
    "common/preload/task_info/aicore/nano_aicore_task_info.cc"
    "common/preload/task_info/aicpu/nano_hostfunc_task_info.cc"
    "common/preload/task_info/aicpu/nano_hostfunc_param.cc"
    "common/preload/task_info/rts/nano_default_task_info.cc"
    "common/preload/dbg/nano_dbg_data.cc"
    "common/preload/model/pre_model_utils.cc"
    "common/preload/model/pre_davinci_model.cc"
    "common/preload/model/nano_davinci_model.cc"
    "common/preload/model/pre_model_partition_utils.cc"
    "common/preload/elf/elf_data.cc"
    "common/allocator/stream_allocator.cc"
    "common/allocator/notify_allocator.cc"
    "common/allocator/event_allocator.cc"
    "common/memory/external_weight_desc_impl.cc"
    "common/memory/mem_type_utils.cc"
    "common/memory/feature_memory_impl.cc"
    "common/memory/tensor_trans_utils.cc"
    "common/host_resource_center/host_resource_center.cc"
    "common/host_resource_center/host_resource_serializer.cc"
    "common/host_resource_center/tiling_resource_manager.cc"
    "common/host_resource_center/weight_manager.cc"
    "common/option_register.cc"
    "common/helper/mobile_model_helper.cc"
    "common/helper/mobile/compatible_info_v2.cc"
    "common/helper/mobile/compiled_model.cc"
    "common/helper/mobile/compiled_target_saver.cc"
    "common/helper/mobile/compiled_target.cc"
    "common/helper/mobile/kernelbin_info.cc"
    "common/helper/mobile/large_memory_ops.cc"
    "common/helper/mobile/model_buffer_helper.cc"
    "common/helper/mobile/model_file_saver.cc"
    "common/helper/mobile/model.cc"
    "common/error_registry/error_registry.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL)
############ libge_common.so ############
add_library(ge_common SHARED
    ${SRC_LIST}
    ${OM2_SRCS}
)

add_dependencies(ge_common
    graphengine_protos
)

target_compile_definitions(ge_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
)

target_compile_options(ge_common PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})

target_include_directories(ge_common PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/ge/fusion/pass
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCGEN_DIR}/inc
    ${ASCEND_INSTALL_PATH}/include/graph
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_options(ge_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        $<$<TARGET_EXISTS:cce_headers>:cce_headers>
        $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
        $<$<TARGET_EXISTS:datagw_headers>:datagw_headers>
        static_mmpa
        -Wl,--no-as-needed
        graph
        ge_common_base
        ascend_protobuf
        c_sec
        error_manager
        unified_dlog
        platform
        -Wl,--as-needed
        json
        minizip_static
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        ${AIR_COMMON_LINK_OPTION}
    PUBLIC
        $<BUILD_INTERFACE:air_headers>
        $<BUILD_INTERFACE:metadef_headers>
)

############ libge_common_base.so ############
add_library(ge_common_base SHARED
    ${SRC_BASE_LIST}
    ${SRC_FORMAT_TRANSFERS}
)

add_dependencies(ge_common_base
    graphengine_protos
)

target_compile_definitions(ge_common_base PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common_base PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})

target_include_directories(ge_common_base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCEND_INSTALL_PATH}/include/graph
    ${TOP_DIR}/runtime/pkg_inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_options(ge_common_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common_base
    PRIVATE
        intf_pub
        msprof_headers
        adump_headers
        runtime_headers
        $<$<TARGET_EXISTS:datagw_headers>:datagw_headers>
        $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
        static_mmpa
        -Wl,--no-as-needed
        runtime
        graph_base
        ascend_protobuf
        c_sec
        acl_rt
        error_manager
        unified_dlog
        platform
        ascend_dump
        exe_graph
        lowering
        -Wl,--as-needed
        json
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        ${AIR_COMMON_LINK_OPTION}
    PUBLIC
        air_headers
        metadef_headers
)

############ libge_common.a ############
add_library(ge_common_static STATIC
    ${SRC_LIST}
    ${SRC_BASE_LIST}
    ${SRC_FORMAT_TRANSFERS}
)

add_dependencies(ge_common_static
    graphengine_protos
)

target_compile_definitions(ge_common_static PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common_static PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:${AIR_COMMON_DYNAMIC_COMPILE_OPTION}>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_include_directories(ge_common_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCGEN_DIR}/inc
    ${METADEF_DIR}/pkg_inc
)

target_link_libraries(ge_common_static PRIVATE
    intf_pub
    slog_headers
    msprof_headers
    adump_headers
    runtime_headers
    mmpa_headers
    metadef_headers
    datagw_headers
    ascend_protobuf_static
    json
    ascend_dump
    c_sec
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

else ()
############ libge_common.so w/static protobuf ############
add_library(ge_common SHARED
    ${SRC_LIST}
)
add_dependencies(ge_common
    graphengine_protos
)

target_compile_definitions(ge_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})

target_include_directories(ge_common PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCGEN_DIR}/inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_options(ge_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        datagw_headers
        ascend_protobuf_static
        -Wl,--no-as-needed
        graph
        ge_common_base
        c_sec
        acl_rt
        error_manager
        unified_dlog
        platform
        static_mmpa
        -Wl,--as-needed
        json
        -lrt
        -ldl
        ${AIR_COMMON_LINK_OPTION}
    PUBLIC
        air_headers
)

############ libge_common_base.so w/static protobuf ############
add_library(ge_common_base SHARED
    ${SRC_BASE_LIST}
    ${SRC_FORMAT_TRANSFERS}
        common/memory/tensor_trans_utils.cc
        common/memory/tensor_trans_utils.h
)
add_dependencies(ge_common_base
    graphengine_protos
)

target_compile_definitions(ge_common_base PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_common_base PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})

target_include_directories(ge_common_base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCGEN_DIR}/inc
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/experiment/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/include/graph
    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_options(ge_common_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common_base
    PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        datagw_headers
        ascend_protobuf_static
        -Wl,--no-as-needed
        runtime
        graph_base
        c_sec
        error_manager
        unified_dlog
        platform
        static_mmpa
        -Wl,--as-needed
        json
        -lrt
        -ldl
        ${AIR_COMMON_LINK_OPTION}
    PUBLIC
        air_headers
        metadef_headers
)
endif ()


############ libge_common_base.a ############
if ("${PRODUCT}" STREQUAL "ascend031")
    target_clone(ge_common_base ge_common_base_static STATIC)

    add_dependencies(ge_common_base_static graphengine_protos)

    target_compile_options(ge_common_base_static PRIVATE
        -Os
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
    )

    set_target_properties(ge_common_base_static PROPERTIES
        OUTPUT_NAME ge_common_base
    )
endif ()


#############################################
set(STUB_HEADER_LIST
    ${AIR_CODE_DIR}/base/common/error_tracking/error_tracking.h
    ${AIR_CODE_DIR}/base/common/omg_util/omg_util.h
    ${AIR_CODE_DIR}/base/common/op_tiling/op_tiling_rt2.h
    ${AIR_CODE_DIR}/base/common/op_tiling/tiling_memcheck.h
    ${AIR_CODE_DIR}/base/common/op_tiling/tiling_dfx.h
    ${AIR_CODE_DIR}/base/common/plugin/runtime_plugin_loader.h
    ${AIR_CODE_DIR}/base/common/tbe_handle_store/bin_register_utils.h
    ${AIR_CODE_DIR}/base/host_cpu_engine/host_cpu_engine.h
    ${AIR_CODE_DIR}/base/common/platform_info_util/platform_info_util.h
)

list(TRANSFORM STUB_HEADER_LIST
    REPLACE "^.*/([^/]+)\\.h$" "${CMAKE_CURRENT_BINARY_DIR}/stub_\\1.cc"
    OUTPUT_VARIABLE STUB_SRC_LIST
)

if(NOT BUILD_PKG_COMPONENT)
    # TODO:待整包下线后删除
    add_custom_command(
        OUTPUT ${STUB_SRC_LIST}
        COMMAND echo "Generating ge_common_stub files."
                && ${HI_PYTHON} ${METADEF_DIR}/tests/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
                && echo "Generating stub files end."
    )
else()
    add_custom_command(
        OUTPUT ${STUB_SRC_LIST}
        COMMAND echo "Generating ge_common_stub files."
                && ${HI_PYTHON} ${GE_METADEF_DIR}/../tests/graph_metadef/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
                && echo "Generating stub files end."
    )
endif()

add_custom_target(stub_ge_common DEPENDS ${STUB_SRC_LIST})

############ libge_common_stub.so w/static protobuf ############
add_library(ge_common_stub SHARED
    ${STUB_SRC_LIST}
)

add_dependencies(ge_common_stub stub_ge_common)

target_compile_definitions(ge_common_stub PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
)

target_compile_options(ge_common_stub PRIVATE
    -fvisibility=default -O2 -fno-common -Wfloat-equal
    -Werror=return-type
    -Wno-unused-parameter
)

target_include_directories(ge_common_stub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCGEN_DIR}/inc
)

target_link_options(ge_common_stub PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common_stub
    PRIVATE
        intf_pub
        slog_headers
        metadef_headers
        runtime_headers
        msprof_headers
        mmpa_headers
        c_sec_headers
        -lrt
        -ldl
    PUBLIC
        air_headers
)

set_target_properties(ge_common_stub PROPERTIES
    OUTPUT_NAME ge_common
    LIBRARY_OUTPUT_DIRECTORY stub
)


############ libge_common_stub.a w/static protobuf ############
if ("${PRODUCT}" STREQUAL "ascend031")
    target_clone(ge_common_stub ge_common_static_stub STATIC)

    add_dependencies(ge_common_static_stub stub_ge_common)

    target_compile_options(ge_common_static_stub PRIVATE
        -ffunction-sections
        -fdata-sections
    )

    set_target_properties(ge_common_static_stub PROPERTIES
        OUTPUT_NAME ge_common
        ARCHIVE_OUTPUT_DIRECTORY stub
    )
endif ()


install(TARGETS ge_common_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)
