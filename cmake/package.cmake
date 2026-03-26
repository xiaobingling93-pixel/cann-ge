# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#### CPACK to package run #####
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Detected architecture: x86_64")
    set(ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    message(STATUS "Detected architecture: ARM64")
    set(ARCH aarch64)
else ()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif ()
# 打印路径
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
set(ARCH_LINUX_PATH "${ARCH}-linux")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/makeself-fetch.cmake)

# ============= 组件打包 =============
function(install_public_packages components)
    set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/${components}/scripts)
    install(DIRECTORY ${script_prefix}/
        DESTINATION share/info/${components}/script
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
        DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    )
    set(SCRIPTS_FILES
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.sh
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.csh
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.fish
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
    )

    install(FILES ${SCRIPTS_FILES}
        DESTINATION share/info/${components}/script
    )
    set(COMMON_FILES
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/install_common_parser.sh
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v2.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_installer.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/script_operator.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_cfg.inc
    )

    set(PACKAGE_FILES
        ${COMMON_FILES}
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/multi_version.inc
    )
    set(LATEST_MANGER_FILES
        ${COMMON_FILES}
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
    )
    set(CONF_FILES
        ${CMAKE_SOURCE_DIR}/scripts/package/common/cfg/path.cfg
    )
    install(FILES ${CMAKE_BINARY_DIR}/version.${components}.info
        DESTINATION share/info/${components}
        RENAME version.info
    )
    install(FILES ${CONF_FILES}
        DESTINATION ${components}/conf
    )
    install(FILES ${PACKAGE_FILES}
        DESTINATION share/info/${components}/script
    )
    install(FILES ${LATEST_MANGER_FILES}
        DESTINATION latest_manager
    )
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/package/latest_manager/scripts/
        DESTINATION latest_manager
    )
endfunction()

# ============= 为不同组件创建install =============
if (DEFINED BUILD_COMPONENT)
    install_public_packages(${BUILD_COMPONENT})
endif ()

if("${BUILD_COMPONENT}" STREQUAL "ge-compiler")
    message(STATUS "************Install ge-compiler packages***************")
    if(NOT MDC_COMPILE_RUNTIME)
        install(TARGETS parser_common aicore_utils fusion_pass op_compile_adapter aicpu_engine_common fmk_parser ge_compiler fmk_onnx_parser opskernel ge_runner
                        slice aicpu_const_folding llm_engine jit_exe _caffe_parser flow_graph aihac_autofusion dflow_runner eager_style_graph_builder_base
                        eager_style_graph_builder_base_static ge_runner_v2 aihac_codegen aihac_ir aihac_ir_register aihac_symbolizer compress compressweight
                        hcom_gradient_split_tune hcom_graph_adaptor acl_op_compiler
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64
        )
        install(TARGETS fmk_onnx_parser_stub fmk_parser_stub atc_stub_ge_compiler fwk_stub_ge_runner fwk_stub_ge_runner_v2 stub_acl_op_compiler
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
        )
        install(TARGETS engine
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/plugin/nnengine
        )
        install(TARGETS aicpu_ascend_engine aicpu_tf_engine dvpp_engine fe ffts ge_local_engine ge_local_opskernel_builder rts_engine
                        host_cpu_opskernel_builder host_cpu_engine hcom_opskernel_builder hcom_gradtune_opskernel_builder
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/plugin/opskernel
        )
        install(TARGETS cpu_compiler udf_compiler
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/plugin/pnecompiler
        )
        install(TARGETS gen_esb OPTIONAL
                RUNTIME DESTINATION ${BUILD_COMPONENT}/bin
        )
        install(TARGETS pyautofuse
                LIBRARY DESTINATION ${BUILD_COMPONENT}/python/site-packages/autofuse
        )
        install(TARGETS atc_atc.bin OPTIONAL
            RUNTIME DESTINATION ${BUILD_COMPONENT}/lib64/atclib
        )
        install(TARGETS fwk_atc.bin OPTIONAL
            RUNTIME DESTINATION ${BUILD_COMPONENT}/lib64/fwkacl
        )
        install(TARGETS fwk_atc.bin OPTIONAL
            RUNTIME DESTINATION ${BUILD_COMPONENT}/bin
        )
        install(FILES ${CMAKE_SOURCE_DIR}/api/atc/atc
            DESTINATION ${BUILD_COMPONENT}/bin
        )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/api/python/wheel2/dist/llm_datadist_v1-0.0.1-py3-none-any.whl
            DESTINATION ${BUILD_COMPONENT}/lib64
        )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/api/python/ge/wheel/dist/ge_py-0.0.1-py3-none-any.whl
            DESTINATION ${BUILD_COMPONENT}/lib64
        )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dflow/pydflow/dataflow-0.0.1-py3-none-any.whl
                DESTINATION ${BUILD_COMPONENT}/lib64
        )
        if(ENABLE_BUILD_DEVICE)
            install(FILES ${CMAKE_CURRENT_BINARY_DIR}/compiler/engines/rts_engine/switch_by_index.o
                DESTINATION ${BUILD_COMPONENT}/fwkacllib/lib64/
            )
        endif()
    else()
        # MDC 运行态编译
        install(TARGETS flow_graph
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64
        )
    endif()
    
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/inc/external/flow_graph
        DESTINATION ${BUILD_COMPONENT}/include
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/llm_datadist/llm_datadist.h
        DESTINATION ${BUILD_COMPONENT}/include/llm_datadist
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/hccl_engine/inc/hcom_gradient_split_tune.h
                  ${CMAKE_SOURCE_DIR}/compiler/engines/hccl_engine/inc/hcom_ops_stores.h
        DESTINATION ${BUILD_COMPONENT}/include/ge
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_ir_build.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_utils.h
        DESTINATION ${BUILD_COMPONENT}/include/ge
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/llm_datadist/llm_error_codes.h
                  ${CMAKE_SOURCE_DIR}/inc/external/llm_datadist/llm_engine_types.h
        DESTINATION ${BUILD_COMPONENT}/include/ge
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/graph_rewriter.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/match_result.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pattern.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pattern_matcher.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/subgraph_boundary.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pattern_matcher_config.h
        DESTINATION ${BUILD_COMPONENT}/include/ge/fusion
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pass/decompose_pass.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pass/fusion_base_pass.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pass/fusion_pass_reg.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/fusion/pass/pattern_fusion_pass.h
        DESTINATION ${BUILD_COMPONENT}/include/ge/fusion/pass
    )
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/FindGenerateEsPackage.cmake
                  ${CMAKE_SOURCE_DIR}/cmake/generate_es_package.cmake
        DESTINATION ${BUILD_COMPONENT}/include/ge/cmake
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/parser/external/parser/caffe_parser.h
                  ${CMAKE_SOURCE_DIR}/inc/parser/external/parser/onnx_parser.h
                  ${CMAKE_SOURCE_DIR}/inc/parser/external/parser/tensorflow_parser.h
        DESTINATION ${BUILD_COMPONENT}/include/parser
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_api.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_api_v2.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_feature_memory.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_data_flow_api.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_graph_compile_summary.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/c/esb_funcs.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/cpp/compliant_node_builder.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/cpp/es_c_graph_builder.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/cpp/es_c_tensor_holder.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/cpp/es_graph_builder.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/cpp/es_tensor_holder.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/eager_style_graph_builder/cpp/es_tensor_like.h
        DESTINATION ${BUILD_COMPONENT}/include/ge
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python/ascbc_kernel_compile.py
                  ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python/asc_codegen_compile.py
                  ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python/ascendc_compile.py
                  ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python/compile_adapter.py
                  ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python/ascir_api.py
                  ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python/__init__.py
        DESTINATION ${BUILD_COMPONENT}/python/site-packages/autofuse
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/manager/engine_manager/engine_conf.json
        DESTINATION ${BUILD_COMPONENT}/lib64/plugin/nnengine/ge_config
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/manager/opskernel_manager/optimizer_priority.pbtxt
        DESTINATION ${BUILD_COMPONENT}/lib64/plugin/opskernel
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/cpu_engine/common/config/init.conf
                  ${CMAKE_SOURCE_DIR}/compiler/engines/cpu_engine/tf_engine/config/ir2tf/ir2tf_op_mapping_lib.json
                  ${CMAKE_SOURCE_DIR}/compiler/engines/cpu_engine/common/config/aicpu_ops_parallel_rule.json
            DESTINATION ${BUILD_COMPONENT}/lib64/plugin/opskernel/config/
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/nn_engine/optimizer/fe_config/fe.ini
            DESTINATION ${BUILD_COMPONENT}/lib64/plugin/opskernel/fe_config
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/ffts_engine/common/ffts_config/ffts.ini
            DESTINATION ${BUILD_COMPONENT}/lib64/plugin/opskernel/ffts_config
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/acl/acl_op_compiler.h
            DESTINATION ${BUILD_COMPONENT}/include/acl
    )
    install(FILES ${CMAKE_SOURCE_DIR}/parser/parser/func_to_graph/func2graph.py
            DESTINATION ${BUILD_COMPONENT}/python/func2graph
    )
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/parser/parser/func_to_graph/proto_python_3_14/
            DESTINATION ${BUILD_COMPONENT}/python/func2graph/util_3_14/
    )
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/parser/parser/func_to_graph/proto_python_3_14/
            DESTINATION ${BUILD_COMPONENT}/python/func2graph/util
    )
elseif("${BUILD_COMPONENT}" STREQUAL "ge-executor")
    message(STATUS "************Install ge-executor packages***************")
    if(NOT MDC_COMPILE_RUNTIME)
        install(TARGETS ge_common ge_executor_shared ge_common_base davinci_executor hybrid_executor gert om2_executor register
                graph lowering register_static graph_base model_deployer npu_sched_model_loader data_flow_base hcom_executor
                acl_mdl acl_mdl_impl acl_op_executor acl_op_executor_impl acl_cblas
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64
        )
        install(TARGETS ge_common_stub stub_lowering atc_stub_graph gert_stub hybrid_executor_stub stub_register
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
        )
        install(TARGETS ge_common_stub stub_lowering atc_stub_graph gert_stub hybrid_executor_stub stub_register
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
        )
        install(TARGETS stub_acl_mdl stub_acl_cblas stub_acl_op_executor
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64/stub/linux/${ARCH}
        )
        install(TARGETS graph register lowering
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64/stub/minios/aarch64
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64/stub/minios/aarch64
        )
        if(ENABLE_BUILD_DEVICE)
            install(FILES ${CMAKE_SOURCE_DIR}/build/runtime/ops/update_model_param/dav_2201/UpdateModelParam_dav_2201.o
                    DESTINATION ${BUILD_COMPONENT}/lib64
            )
        endif()
    else()
       # MDC 运行态编译
        install(TARGETS ge_common ge_common_base davinci_executor hybrid_executor gert register graph graph_base acl_cblas
                        acl_mdl acl_mdl_impl acl_op_executor acl_op_executor_impl om2_executor ge_executor_shared lowering
                LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64
                ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64
        )
    endif()

    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_api_types.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_api_error_codes.h
                  ${CMAKE_SOURCE_DIR}/inc/external/ge/ge_external_weight_desc.h
        DESTINATION ${BUILD_COMPONENT}/include/ge
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/acl/acl_mdl.h
                  ${CMAKE_SOURCE_DIR}/inc/external/acl/acl_base_mdl.h
                  ${CMAKE_SOURCE_DIR}/inc/external/acl/acl_op.h
        DESTINATION ${BUILD_COMPONENT}/include/acl
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/external/acl/ops/acl_cblas.h
        DESTINATION ${BUILD_COMPONENT}/include/acl/ops
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/exe_graph/runtime/eager_op_execution_context.h
            DESTINATION ${BUILD_COMPONENT}/include/exe_graph/runtime
    )
    set(EXTERNAL_GRAPH_FILES
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/graph.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/ct_infer_shape_range_context.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/ct_infer_shape_context.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/operator_reg.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/gnode.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/graph_buffer.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/inference_context.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/attr_value.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/operator.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/operator_factory.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/resource_context.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/kernel_launch_info.h 
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/arg_desc_info.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/graph/custom_op.h
    )
    install(FILES ${EXTERNAL_GRAPH_FILES}
        DESTINATION ${BUILD_COMPONENT}/include/graph
    )
    install(FILES ${CMAKE_SOURCE_DIR}/compiler/engines/hccl_engine/inc/hcom_executor.h
        DESTINATION ${BUILD_COMPONENT}/include/graph
    )
    install(FILES ${CMAKE_SOURCE_DIR}/graph_metadef/proto/caffe/caffe.proto
                  ${CMAKE_SOURCE_DIR}/graph_metadef/proto/onnx/ge_onnx.proto
                  ${CMAKE_SOURCE_DIR}/graph_metadef/proto/ge_ir.proto
        DESTINATION ${BUILD_COMPONENT}/include/proto
    )

    set(EXTERNAL_REGISTER_FILES
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/register_base.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/op_lib_register.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/register_custom_pass.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/op_binary_resource_manager.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/op_config_registry.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/op_def.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/op_def_registry.h
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/op_def_factory.h
    )
    install(FILES ${EXTERNAL_REGISTER_FILES}
        DESTINATION ${BUILD_COMPONENT}/include/register
    )
    install(FILES ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external/register/scope/scope_fusion_pass_register.h
        DESTINATION ${BUILD_COMPONENT}/include/register/scope
    )
    install(FILES ${CMAKE_SOURCE_DIR}/graph_metadef/third_party/transformer/inc/transfer_def.h
                  ${CMAKE_SOURCE_DIR}/graph_metadef/third_party/transformer/inc/transfer_shape_according_to_format_ext.h
        DESTINATION ${BUILD_COMPONENT}/include/transformer
    )
elseif("${BUILD_COMPONENT}" STREQUAL "dflow-executor")
    message(STATUS "************Install dflow-executor packages***************")
    install(TARGETS udf_dump reader_writer udf_profiling flow_func built_in_flowfunc
            LIBRARY DESTINATION ${BUILD_COMPONENT}/lib64
            ARCHIVE DESTINATION ${BUILD_COMPONENT}/lib64
    )
    install(TARGETS deployer_daemon npu_executor_main host_cpu_executor_main udf_executor
            RUNTIME DESTINATION ${BUILD_COMPONENT}/bin
    )
    install(FILES
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/ascend_string.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/balance_config.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/flow_func_log.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/flow_msg_queue.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/meta_flow_func.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/meta_params.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/out_options.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/attr_value.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/flow_func_defines.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/flow_msg.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/meta_context.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/meta_multi_func.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/meta_run_context.h
            ${CMAKE_SOURCE_DIR}/dflow/udf/inc/external/flow_func/tensor_data_type.h
            DESTINATION ${BUILD_COMPONENT}/include/flow_func
    )
    install(FILES ${CMAKE_SOURCE_DIR}/dflow/deployer/monitor_cpu.sh
            DESTINATION ${BUILD_COMPONENT}/bin
            RENAME "monitor.sh")
    install(FILES ${CMAKE_BINARY_DIR}/device_install/cann-udf-compat.tar.gz OPTIONAL
            DESTINATION ${BUILD_COMPONENT}/lib64)
    install(FILES ${CMAKE_BINARY_DIR}/device_install/Ascend-runtime_device-minios.tar.gz OPTIONAL
            DESTINATION ${BUILD_COMPONENT}/lib64)
    install(FILES ${CMAKE_BINARY_DIR}/device_install/device/lib64/libflow_func.so OPTIONAL
            DESTINATION ${BUILD_COMPONENT}/devlib/device)
endif()


# ============= CPack =============
set(CPACK_BUILD_COMPONENT "${BUILD_COMPONENT}")

set(CPACK_COMPONENTS_ALL ".;packages")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(CPACK_MAKESELF_PATH "${MAKESELF_PATH}")
set(CPACK_ARCH "${ARCH}")
set(CPACK_GENERATOR External)
set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself.cmake")

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
include(CPack)
