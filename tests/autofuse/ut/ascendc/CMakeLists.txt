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
string(REGEX MATCH "NOTFOUND$" tikicpulib_not_found "${tikicpulib_DIR}")
if("${tikicpulib_not_found}" STREQUAL "NOTFOUND")
    message(WARNING "tikicpulib not found, skip e2e calculate testcase.")
else()
    add_subdirectory(api)
endif()

add_executable(test_ascendc_api ../test_main.cpp)
target_link_libraries(test_ascendc_api PRIVATE
        atrace
        error_manager
        test_ascendc
        GTest::gtest
)
gtest_discover_tests(test_ascendc_api)