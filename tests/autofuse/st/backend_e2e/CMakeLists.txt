set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_ROOT}/tools/tikicpulib/lib)
find_package(tikicpulib QUIET)

# 要修复的目标列表
set(TIKICPULIB_TARGETS
        tikicpulib_ascend910B1
        tikicpulib_ascend950pr_9599
        # 添加其他可能出错的 target
)

foreach(tgt IN LISTS TIKICPULIB_TARGETS)
    if(TARGET ${tgt})
        # 修复 INTERFACE_INCLUDE_DIRECTORIES
        get_target_property(include_dirs ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT include_dirs)
            set(include_dirs "")
        endif()

        set(valid_includes "")
        foreach(dir IN LISTS include_dirs)
            # 跳过明显错误的嵌套路径（可选）
            if(dir MATCHES "tools/tikicpulib/tools/tikicpulib")
                message(VERBOSE "Skipping malformed include dir: ${dir}")
                continue()
            endif()

            # 只保留真实存在的目录
            if(IS_DIRECTORY "${dir}")
                list(APPEND valid_includes "${dir}")
            else()
                message(VERBOSE "Skipping non-existent include dir: ${dir}")
            endif()
        endforeach()

        set_target_properties(${tgt}
                PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${valid_includes}"
        )

        # 同样修复 INTERFACE_LINK_DIRECTORIES（如有需要）
        get_target_property(link_dirs ${tgt} INTERFACE_LINK_DIRECTORIES)
        if(link_dirs)
            set(valid_links "")
            foreach(dir IN LISTS link_dirs)
                if(IS_DIRECTORY "${dir}")
                    list(APPEND valid_links "${dir}")
                endif()
            endforeach()
            set_target_properties(${tgt}
                    PROPERTIES INTERFACE_LINK_DIRECTORIES "${valid_links}"
            )
        endif()
    endif()
endforeach()

include_directories(${CODE_ROOT_DIR}/../../../../tests/autofuse/framework/share_graph/include)
include_directories(${CODE_ROOT_DIR}/../../../../tests/autofuse/st/backend_e2e)
include(backend_e2e.cmake)

add_subdirectory(add_abs_test)
add_subdirectory(sub_abs_test)
add_subdirectory(scalar_float_inf_test)
add_subdirectory(scalar_div_inf_test)
add_subdirectory(add_gelu_test)
add_subdirectory(compare_test)
add_subdirectory(compare_x2_tensor_test)
add_subdirectory(compare_x2_tensor_int32_test)
add_subdirectory(compare_x2_tensor_int64_eq_test)
add_subdirectory(compare_x2_tensor_int64_gt_test)
add_subdirectory(add_neg_test)
add_subdirectory(load_to_store_and_abs_test)
add_subdirectory(load_unalign_pad_test)
add_subdirectory(scalar_cast_add_test)
# add_subdirectory(concat_test)
add_subdirectory(concat_all_aligned_test)
add_subdirectory(concat_to_stores_test)
add_subdirectory(brc_inline_test)
add_subdirectory(load_where_store_test)
add_subdirectory(load_where_x2_x3_is_ubscalar_store_test)
add_subdirectory(load_where_x2_is_ubscalar_store_test)
add_subdirectory(load_where_x3_is_ubscalar_store_test)
add_subdirectory(load_logical_not_store_test)
add_subdirectory(add_rsqrt_test)
# add_subdirectory(sub_transpose_abs_test)
add_subdirectory(axpy_abs_test)
add_subdirectory(axpy_abs_half_test)
add_subdirectory(load_pow_all_input_is_scalar_store_test)
add_subdirectory(pgo_add_abs_test)
# add_subdirectory(pgo_sub_transpose_abs_test)
add_subdirectory(matmul_elemwise_test)
