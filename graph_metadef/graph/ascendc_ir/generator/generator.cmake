# 递归收集目标的所有动态库依赖路径
function(get_all_dynamic_dirs target result_var)
    set(dirs "")
    # 获取目标的直接依赖库
    get_target_property(libs ${target} LINK_LIBRARIES)
    foreach (lib IN LISTS libs)
        # 仅处理 CMake 目标（排除系统库和绝对路径）
        if (TARGET ${lib})
            get_target_property(type ${lib} TYPE)
            # 处理动态库（SHARED_LIBRARY）
            if (type STREQUAL "SHARED_LIBRARY")
                # 获取动态库的输出目录
                get_target_property(lib_output_dir ${lib} LIBRARY_OUTPUT_DIRECTORY)
                if (NOT lib_output_dir)
                    set(lib_output_dir $<TARGET_FILE_DIR:${lib}>)
                endif ()
                list(APPEND dirs ${lib_output_dir})
                # 递归处理依赖
                get_all_dynamic_dirs(${lib} child_dirs)
                list(APPEND dirs ${child_dirs})
            endif ()
        endif ()
    endforeach ()
    list(REMOVE_DUPLICATES dirs)
    set(${result_var} ${dirs} PARENT_SCOPE)
endfunction()

function(ascir_generate depend_so_target bin_dir so_var h_var)
    # 1. 收集所有动态库依赖的生成器表达式
    get_all_dynamic_dirs(ascir_ops_header_generator lib_dirs_genex)

    # 2. 添加自定义命令,ascend-toolkit的LD_PATH放在最后, 保证优先使用编译路径下的so
    add_custom_command(
            OUTPUT ${h_var}
            DEPENDS ${depend_so_target} ascir_ops_header_generator
            COMMAND ${CMAKE_COMMAND} -E echo "Raw Library Paths: $<JOIN:${lib_dirs_genex},:>"
            COMMAND bash -c " \
            IFS=: read -ra paths <<< '$<JOIN:${lib_dirs_genex},:>'; \
            non_ascend=(); \
            ascend=(); \
            seen=(); \
            for p in \"\${paths[@]}\"; do \
                duplicate=0; \
                for s in \"\${seen[@]}\"; do \
                    if [ \"\${p}\" = \"\${s}\" ]; then \
                        duplicate=1; \
                        break; \
                    fi; \
                done; \
                if [ \$duplicate -eq 1 ]; then \
                    continue; \
                fi; \
                seen+=(\"\${p}\"); \
                if [[ \"\${p}\" == */latest/* ]]; then \
                    ascend+=(\"\${p}\"); \
                else \
                    non_ascend+=(\"\${p}\"); \
                fi; \
            done; \
            final_paths=(\"\${non_ascend[@]}\" \"\${ascend[@]}\"); \
            lib_path_str=\$(IFS=:; echo \"\${final_paths[*]}\"); \
            echo \"Adjusted LD_LIBRARY_PATH: \${lib_path_str}:\$LD_LIBRARY_PATH\"; \
            export LD_LIBRARY_PATH=\"\${lib_path_str}:\$LD_LIBRARY_PATH\"; \
            '${bin_dir}/ascir_ops_header_generator' '${so_var}' '${h_var}' \
        "
            VERBATIM
            COMMENT "Generating header ${h_var} with fresh dependencies"
    )
endfunction()