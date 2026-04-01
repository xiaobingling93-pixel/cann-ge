# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 es_gen_esb_serial job pool
# 用于串行化 gen_esb 及其依赖的构建，避免 ar 和 ld 的文件竞态
set_property(GLOBAL PROPERTY JOB_POOLS es_gen_esb_serial=1)

# 在函数外部获取当前 cmake 文件的路径（函数内部 CMAKE_CURRENT_LIST_FILE 会指向调用者）
# 每次 include 时都重新设置，确保路径正确
get_filename_component(_ADD_ES_LIBRARY_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
message(STATUS "[add_es_library] Module loaded from: ${_ADD_ES_LIBRARY_CMAKE_DIR}")

# ======================================================================================================================
# 内部实现函数: _add_es_library_impl (不要直接调用此函数)
#
# 功能描述:
#   为指定的算子分包生成 Eager Style (ES) API 产物
#
# 参数说明:
#   ES_PACKAGE_TARGET  - [必需] 对外暴露的接口库 target 名称
#   OPP_PROTO_TARGET   - [必需] 算子原型库的 CMake target 名称
#   OUTPUT_PATH        - [必需] 产物输出的根目录
#   EXCLUDE_OPS        - [可选] 需要排除生成的OP算子
#   SKIP_WHL           - [可选] 是否跳过 Python wheel 包生成
#
# 依赖要求:
#   - CMake 版本: >= 3.16 (使用 CONFIGURE_DEPENDS、configure_file COPYONLY 等特性)
#   - gen_esb: 支持两种环境
#     1. 社区环境: 自动从 cmake 文件路径推导 run 包中的 gen_esb 位置
#     2. 开发环境: 使用源码编译的 gen_esb target
#   - OPP_PROTO_TARGET: 必须存在，且需要设置 LIBRARY_OUTPUT_DIRECTORY 属性
#   - eager_style_graph_builder_base: 支持两种来源
#     1. 源码编译的 target（优先）: 如果存在 target，直接使用
#     2. run 包中的库: 自动推导并查找
#   - Python3 可执行文件可用
#   - setuptools 和 wheel Python 包已安装
#
# 注意事项:
#   1. ES_LINKABLE_AND_ALL_TARGET 应使用小写字母和下划线，建议以 es_ 开头（如 es_math, es_nn）
#   2. OPP_PROTO_TARGET 必须存在且已设置 LIBRARY_OUTPUT_DIRECTORY 属性
#   3. 函数会自动从 OPP_PROTO_TARGET 的输出路径推导 ASCEND_OPP_PATH
#   4. 函数会自动检测 gen_esb 位置（run 包或源码编译）
#   5. 生成的 whl 包版本号默认为 1.0.0，可通过修改函数内 setup.py 模板调整
#   6. 使用每包独立文件锁确保同一包的代码生成串行，避免文件竞态
#   7. 多个 ES 包会自动添加依赖关系，确保共享依赖只构建一次
# ======================================================================================================================

# ======================================================================================================================
# add_custom_command 封装宏
#
# 统一封装代码生成命令，内部根据 COMBINED_COMMERCIAL_MODE 自动选择模式：
#   单次模式（COMBINED_COMMERCIAL_MODE=FALSE）：仅执行一次 gen_esb，生成 C++ API
#   双次模式（COMBINED_COMMERCIAL_MODE=TRUE）： 步骤1 代码生成 + 步骤2 历史原型库归档
#
# 三处调用点（外部 gen_esb 有依赖 / 外部 gen_esb 无依赖 / 源码 gen_esb）共用此宏，
# 仅 gen_esb 路径、LD_LIBRARY_PATH、EXCLUDE_OPS、DEPENDS、COMMENT 不同，由调用方通过参数传入。
#
# 参数：
#   _gen_esb_exe   gen_esb 可执行路径（字面量或生成器表达式 $<TARGET_FILE:gen_esb>）
#   _lib_dir       LD_LIBRARY_PATH（run 包环境传 ${ASCEND_LIB_DIR}，源码环境传空字符串）
#   _excl_ops      排除算子列表（INTERFACE 库传空字符串，其他传 ${ARG_EXCLUDE_OPS}）
#   _comment       COMMENT 说明文字
#
# 调用前需设置 _COMBINED_MODE_DEPENDS：
#   有依赖时：set(_COMBINED_MODE_DEPENDS "DEPENDS;dep1;dep2")
#   无依赖时：unset(_COMBINED_MODE_DEPENDS)
# ======================================================================================================================
macro(_es_add_gen_esb_cmd _gen_esb_exe _lib_dir _excl_ops _comment)
    if (COMBINED_COMMERCIAL_MODE)
        set(_HIST_STAGE_DIR "${ARG_OUTPUT_PATH}")
        if ("${_AUTO_HISTORY_REGISTRY}" STREQUAL "${ARG_OUTPUT_PATH}")
            # 首次构建（历史库尚不存在于 CANN 路径）：_AUTO_HISTORY_REGISTRY == ARG_OUTPUT_PATH，
            set(_PREPOPULATE_STAGING "")
        else ()
            # 非首次构建：将 CANN 只读历史库内容合并复制到 ARG_OUTPUT_PATH
            set(_PREPOPULATE_STAGING
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${_AUTO_HISTORY_REGISTRY}" "${ARG_OUTPUT_PATH}"
                COMMAND chmod u+w "${ARG_OUTPUT_PATH}/index.json"
            )
        endif ()

        add_custom_command(
                OUTPUT ${CODE_GEN_FLAG}
                COMMAND ${CMAKE_COMMAND} -E echo "Generating ES code (combined commercial mode) for package: ${ARG_ES_LINKABLE_AND_ALL_TARGET}"
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${GEN_CODE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_CODE_DIR}
                # 步骤1: 代码生成（消费历史原型库，gen_esb 自动选取窗口内历史版本对比）
                COMMAND bash ${ES_LOCK_SCRIPT}
                ${GEN_CODE_DIR}/.gen.lock
                ${_gen_esb_exe}
                ${OPP_PROTO_PATH}
                ${GEN_CODE_DIR}
                ${MODULE_NAME}
                "${_lib_dir}"
                "${_excl_ops}"
                ""
                ""
                ${HISTORY_REGISTRY_ARG}
                ""
                ""
                # 步骤2: 归档到 ARG_OUTPUT_PATH；若已有历史库则先复制并授权，gen_esb 追加新条目
                ${_PREPOPULATE_STAGING}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_HIST_STAGE_DIR}"
                COMMAND bash ${ES_LOCK_SCRIPT}
                ${_HIST_STAGE_DIR}/.extract.lock
                ${_gen_esb_exe}
                ${OPP_PROTO_PATH}
                ${_HIST_STAGE_DIR}
                ${MODULE_NAME}
                "${_lib_dir}"
                ""
                ${EXTRACT_HISTORY_FLAG}
                ${RELEASE_VERSION_ARG}
                ""
                ${RELEASE_DATE_ARG}
                ${BRANCH_NAME_ARG}
                COMMAND ${CMAKE_COMMAND} -E echo "[ES] Historical prototype library generation completed: ${ARG_OUTPUT_PATH}"
                # 步骤3: 动态生成 wrapper 文件的 include 内容
                COMMAND ${CMAKE_COMMAND} -P ${GENERATE_WRAPPER_SCRIPT}
                COMMAND ${CMAKE_COMMAND} -E touch ${CODE_GEN_FLAG}
                ${_COMBINED_MODE_DEPENDS}
                COMMENT "${_comment}"
                JOB_POOL es_gen_esb_serial
                VERBATIM
        )
    else ()
        # 历史库路径有效时：代码生成后将 CANN 历史库内容合并复制到 ARG_OUTPUT_PATH
        if (_AUTO_HISTORY_REGISTRY)
            set(_COPY_EXISTING_HISTORY
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${_AUTO_HISTORY_REGISTRY}" "${ARG_OUTPUT_PATH}"
                COMMAND ${CMAKE_COMMAND} -E echo "[ES] Existing history registry copied to output: ${ARG_OUTPUT_PATH}"
            )
        else ()
            set(_COPY_EXISTING_HISTORY "")
        endif ()
        add_custom_command(
                OUTPUT ${CODE_GEN_FLAG}
                COMMAND ${CMAKE_COMMAND} -E echo "Generating ES code for package: ${ARG_ES_LINKABLE_AND_ALL_TARGET}"
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${GEN_CODE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_CODE_DIR}
                # 单次代码生成
                COMMAND bash ${ES_LOCK_SCRIPT}
                ${GEN_CODE_DIR}/.gen.lock
                ${_gen_esb_exe}
                ${OPP_PROTO_PATH}
                ${GEN_CODE_DIR}
                ${MODULE_NAME}
                "${_lib_dir}"
                "${_excl_ops}"
                ${CODE_GEN_STEP_EXTRACT_FLAG}
                ${RELEASE_VERSION_ARG}
                ${HISTORY_REGISTRY_ARG}
                ${RELEASE_DATE_ARG}
                ${BRANCH_NAME_ARG}
                # 版本重复时将现有历史库复制到输出目录
                ${_COPY_EXISTING_HISTORY}
                COMMAND ${CMAKE_COMMAND} -P ${GENERATE_WRAPPER_SCRIPT}
                COMMAND ${CMAKE_COMMAND} -E touch ${CODE_GEN_FLAG}
                ${_COMBINED_MODE_DEPENDS}
                COMMENT "${_comment}"
                JOB_POOL es_gen_esb_serial
                VERBATIM
        )
    endif ()
endmacro()

function(_add_es_library_impl)
    # 0. 生成辅助 shell 脚本（自包含，无需外部文件）
    # 在首次调用时创建 run_gen_esb_with_lock.sh 到构建目录
    set(ES_LOCK_SCRIPT "${CMAKE_BINARY_DIR}/cmake/run_gen_esb_with_lock.sh")
    if (NOT EXISTS ${ES_LOCK_SCRIPT})
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/cmake")
        file(WRITE ${ES_LOCK_SCRIPT} "#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Auto-generated by generate_es_package.cmake
# ES code generation wrapper script with detailed logging and flock fallback
#
# Args: \$1=lock_file \$2=gen_esb_path \$3=ASCEND_OPP_PATH \$4=output_dir \$5=module_name \$6=LD_LIBRARY_PATH(optional) \$7=EXCLUDE_OPS(optional) \$8=MODE_ARG(optional, e.g. --es_mode=extract_history) \$9=RELEASE_VERSION(optional) \$10=HISTORY_REGISTRY_ARG(optional) \$11=RELEASE_DATE_ARG(optional) \$12=BRANCH_NAME_ARG(optional)

LOCK_FILE=\"\$1\"
GEN_ESB_EXE=\"\$2\"
OPP_PATH=\"\$3\"
OUTPUT_DIR=\"\$4\"
MODULE_NAME=\"\$5\"
LIB_PATH=\"\${6:-}\"
EXCLUDE_OPS=\"\${7:-}\"
EXTRACT_HISTORY_FLAG=\"\${8:-}\"
RELEASE_VERSION_FLAG=\"\${9:-}\"
HISTORY_REGISTRY_ARG=\"\${10:-}\"
RELEASE_DATE_ARG=\"\${11:-}\"
BRANCH_NAME_ARG=\"\${12:-}\"

# Enable pipefail to ensure exit code from gen_esb is captured through pipes
set -o pipefail

# Debug log file
DEBUG_LOG=\"\${OUTPUT_DIR}/.gen_esb_debug.log\"
mkdir -p \"\${OUTPUT_DIR}\"

log_debug() {
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] [ES-GEN] \$*\" | tee -a \"\${DEBUG_LOG}\" >&2
}

log_debug \"========== ES Code Generation Started ==========\"
log_debug \"Lock file: \${LOCK_FILE}\"
log_debug \"Gen ESB: \${GEN_ESB_EXE}\"
log_debug \"OPP path: \${OPP_PATH}\"
log_debug \"Output dir: \${OUTPUT_DIR}\"
log_debug \"Module: \${MODULE_NAME}\"
log_debug \"Lib path: \${LIB_PATH}\"
log_debug \"Exclude ops: \${EXCLUDE_OPS}\"

FLAG_FILE=\"\${OUTPUT_DIR}/generated_code.flag\"
INPROGRESS_FILE=\"\${OUTPUT_DIR}/.generating.lock\"
MAX_RETRIES=10
RETRY_COUNT=0

quick_check_if_done() {
    if [ -f \"\${INPROGRESS_FILE}\" ]; then
        log_debug \"[Quick Check] Another process is generating code, waiting...\"
        local wait_count=0
        while [ -f \"\${INPROGRESS_FILE}\" ] && [ \$wait_count -lt 20 ]; do
            sleep 0.5
            wait_count=\$((wait_count + 1))
        done
        log_debug \"[Quick Check] Wait completed (waited \${wait_count} times)\"
    fi
    if [ -f \"\${FLAG_FILE}\" ]; then
        local has_h=0
        local has_hc=0
        local has_py=0
        [ -f \"\${OUTPUT_DIR}/es_\${MODULE_NAME}_ops.h\" ] && has_h=1
        [ -f \"\${OUTPUT_DIR}/es_\${MODULE_NAME}_ops_c.h\" ] && has_hc=1
        [ -f \"\${OUTPUT_DIR}/es_\${MODULE_NAME}_ops.py\" ] && has_py=1
        if [ \${has_h} -eq 1 ] && [ \${has_hc} -eq 1 ] && [ \${has_py} -eq 1 ]; then
            log_debug \"[Fast Path] Code already generated, skipping\"
            return 0
        else
            log_debug \"[Stale FLAG] Removing stale flag (h=\${has_h} hc=\${has_hc} py=\${has_py})\"
            rm -f \"\${FLAG_FILE}\"
        fi
    fi
    return 1
}

execute_gen_esb() {
    if quick_check_if_done; then
        return 0
    fi

    touch \"\${INPROGRESS_FILE}\"
    trap \"rm -f '\${INPROGRESS_FILE}'\" EXIT INT TERM
    log_debug \"Waiting for filesystem sync...\"
    sleep 0.3
    sync

    while [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; do
        if [ ! -x \"\${GEN_ESB_EXE}\" ]; then
            log_debug \"[Retry \$((RETRY_COUNT + 1))/\$MAX_RETRIES] gen_esb not executable, waiting...\"
            sleep 0.3
            sync
            RETRY_COUNT=\$((RETRY_COUNT + 1))
            continue
        fi

        log_debug \"[Attempt \$((RETRY_COUNT + 1))/\$MAX_RETRIES] Executing gen_esb...\"
        if [ -n \"\$LIB_PATH\" ]; then
            ENV_PREFIX=\"LD_LIBRARY_PATH=\${LIB_PATH}:\\\$LD_LIBRARY_PATH ASCEND_OPP_PATH=\${OPP_PATH}\"
        else
            ENV_PREFIX=\"ASCEND_OPP_PATH=\${OPP_PATH}\"
        fi
        if [[ \"${ENABLE_ASAN}\" == \"true\" ]]; then
            USE_ASAN=\$(gcc -print-file-name=libasan.so)
            ENV_PREFIX=\"\${ENV_PREFIX} LD_PRELOAD=\${USE_ASAN}\"
        fi
        log_debug \"Command: \$ENV_PREFIX \\\"\${GEN_ESB_EXE}\\\" --output_dir=\\\"\${OUTPUT_DIR}\\\" --module_name=\\\"\${MODULE_NAME}\\\" --exclude_ops=\\\"\${EXCLUDE_OPS}\\\" \${EXTRACT_HISTORY_FLAG} \${RELEASE_VERSION_FLAG} \${HISTORY_REGISTRY_ARG} \${RELEASE_DATE_ARG} \${BRANCH_NAME_ARG}\"

        # Execute gen_esb directly (without env -i, with pipefail enabled)
        if eval \"\$ENV_PREFIX \\\"\${GEN_ESB_EXE}\\\" --output_dir=\\\"\${OUTPUT_DIR}\\\" --module_name=\\\"\${MODULE_NAME}\\\" --exclude_ops=\\\"\${EXCLUDE_OPS}\\\" \${EXTRACT_HISTORY_FLAG} \${RELEASE_VERSION_FLAG} \${HISTORY_REGISTRY_ARG} \${RELEASE_DATE_ARG} \${BRANCH_NAME_ARG} 2>&1 | tee -a \\\"\${DEBUG_LOG}\\\"\"; then
            log_debug \"[Success] gen_esb executed successfully\"
            return 0
        else
            EXIT_CODE=\$?
            # Exit code 139 = SIGSEGV (128 + 11), retry for ASan-induced failures
            if [ \$EXIT_CODE -eq 139 ]; then
                log_debug \"[Retriable] SIGSEGV (ASan shadow memory conflict), retrying...\"
                sleep 0.5
                sync
                RETRY_COUNT=\$((RETRY_COUNT + 1))
            elif [ \$EXIT_CODE -eq 126 ] || [ \$EXIT_CODE -eq 127 ]; then
                log_debug \"[Retriable] Exit code \$EXIT_CODE, retrying...\"
                sleep 0.5
                sync
                RETRY_COUNT=\$((RETRY_COUNT + 1))
            else
                log_debug \"[Non-retriable] Exit code \$EXIT_CODE\"
                return \$EXIT_CODE
            fi
        fi
    done
    log_debug \"[Final Failure] Reached maximum retry count \$MAX_RETRIES\"
    return 1
}

log_debug \"========== Main Logic Started ==========\"
if quick_check_if_done; then
    log_debug \"Quick check passed, exiting\"
    exit 0
fi

if command -v flock &> /dev/null; then
    log_debug \"Using flock file locking mechanism\"
    if flock -x -n 200 2>/dev/null; then
        log_debug \"[flock] Acquired lock immediately\"
        execute_gen_esb
        EXIT_CODE=\$?
        log_debug \"[flock] Execution completed, exit code: \$EXIT_CODE\"
        exit \$EXIT_CODE
    elif flock -x -w 180 200 2>/dev/null; then
        log_debug \"[flock] Acquired lock after waiting (max 3 minutes)\"
        if quick_check_if_done; then
            log_debug \"[flock] Another worker already completed, skipping\"
            exit 0
        fi
        execute_gen_esb
        EXIT_CODE=\$?
        log_debug \"[flock] Execution completed, exit code: \$EXIT_CODE\"
        exit \$EXIT_CODE
    else
        log_debug \"[flock] Failed to acquire lock after long wait, entering polling mode\"
        POLL_COUNT=0
        MAX_POLLS=360
        while [ \$POLL_COUNT -lt \$MAX_POLLS ]; do
            sleep 0.5
            if quick_check_if_done; then
                log_debug \"[Polling] Code generation completed, exiting\"
                exit 0
            fi
            POLL_COUNT=\$((POLL_COUNT + 1))
            if [ \$((\$POLL_COUNT % 10)) -eq 0 ]; then
                log_debug \"[Polling] Waiting... (\$POLL_COUNT/\$MAX_POLLS)\"
            fi
        done
        log_debug \"[Final Failure] Polling timeout and code not generated\"
        if quick_check_if_done; then
            log_debug \"[Final Check] Code generated, exiting\"
            exit 0
        fi
        log_debug \"[Final Check] Code still not generated, exiting\"
        exit 1
    fi 200>\"\${LOCK_FILE}\"
else
    log_debug \"Warning: flock not available, using mkdir fallback mechanism\"
    LOCK_DIR=\"\${OUTPUT_DIR}/.gen_esb.lock.d\"
    RETRY_LOCK_COUNT=0
    MAX_LOCK_RETRIES=30
    while [ \$RETRY_LOCK_COUNT -lt \$MAX_LOCK_RETRIES ]; do
        if mkdir \"\${LOCK_DIR}\" 2>/dev/null; then
            log_debug \"[mkdir lock] Successfully acquired lock\"
            trap \"rmdir '\${LOCK_DIR}' 2>/dev/null\" EXIT INT TERM
            execute_gen_esb
            EXIT_CODE=\$?
            rmdir \"\${LOCK_DIR}\" 2>/dev/null
            log_debug \"[mkdir lock] Released lock, exit code: \$EXIT_CODE\"
            exit \$EXIT_CODE
        else
            log_debug \"[mkdir lock] Lock occupied, waiting... (attempt \$((RETRY_LOCK_COUNT + 1))/\$MAX_LOCK_RETRIES)\"
            sleep 0.5
            if quick_check_if_done; then
                log_debug \"[mkdir lock] Another worker already completed, skipping\"
                exit 0
            fi
            RETRY_LOCK_COUNT=\$((RETRY_LOCK_COUNT + 1))
        fi
    done
    log_debug \"[mkdir lock] Failed to acquire lock (timeout)\"
    if quick_check_if_done; then
        log_debug \"[mkdir lock] Final check passed, skipping\"
        exit 0
    fi
    log_debug \"[mkdir lock] Final check failed, exiting\"
    exit 1
fi
log_debug \"========== Script End (should not reach here) ==========\"
exit 1
")
        # 设置脚本可执行权限
        execute_process(COMMAND chmod +x ${ES_LOCK_SCRIPT})
        message(STATUS "Generated helper script: ${ES_LOCK_SCRIPT}")
    endif ()

    # 1. 解析函数参数
    set(options SKIP_WHL)
    set(oneValueArgs ES_LINKABLE_AND_ALL_TARGET OPP_PROTO_TARGET OUTPUT_PATH EXCLUDE_OPS)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # 1.1. 只有 add_es_library_and_whl 调用时才需要检查 Python3
    if (NOT ARG_SKIP_WHL)
        if (NOT Python3_EXECUTABLE)
            find_package(Python3 COMPONENTS Interpreter)
            if (Python3_EXECUTABLE)
                message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
            else ()
                message(FATAL_ERROR "Python3 is required for wheel package generation, but not found.\n"
                        "Please choose one of the following options:\n"
                        "  1. Confirm if python3 is installed (run 'which python3' or 'python3 --version')\n"
                        "  2. If python3 exists but CMake cannot find it, you can define Python3_EXECUTABLE in your CMakeLists.txt:\n"
                        "     set(Python3_EXECUTABLE /path/to/python3)\n"
                        "  3. If you don't need the Python wheel package, you can use 'add_es_library'")
            endif ()
        else ()
            message(STATUS "Using existing Python3: ${Python3_EXECUTABLE}")
        endif ()
    endif ()

    # 2. 参数校验
    if (NOT ARG_ES_LINKABLE_AND_ALL_TARGET)
        message(FATAL_ERROR "_add_es_library_impl: ES_LINKABLE_AND_ALL_TARGET is required")
    endif ()
    if (NOT ARG_OPP_PROTO_TARGET)
        message(FATAL_ERROR "_add_es_library_impl: OPP_PROTO_TARGET is required")
    endif ()
    if (NOT ARG_OUTPUT_PATH)
        message(FATAL_ERROR "_add_es_library_impl: OUTPUT_PATH is required")
    endif ()

    if (NOT ARG_EXCLUDE_OPS)
        message(STATUS "_add_es_library_impl: EXCLUDE_OPS is not provided")
    else()
        message(STATUS "_add_es_library_impl: EXCLUDE_OPS is ${ARG_EXCLUDE_OPS}")
    endif ()

    # 历史原型库相关参数
    # 使用 cmake 变量（-D 传入）；若未定义则从同名环境变量捕获。
    # GE_ES_EXTRACT_HISTORY:   bool 开关，ON 时启用历史原型库归档模式（两次 gen_esb 调用）；
    #                           不设置或 OFF 时走纯代码生成模式（仅生成 C++ API，不归档）
    # GE_ES_RELEASE_VERSION:   当前新版本号（例如 "8.0.RC1"），用于历史原型库归档
    # GE_ES_RELEASE_DATE:      归档时的发布日期（可选，格式 YYYY-MM-DD，不指定则 gen_esb 使用当前日期）
    # GE_ES_BRANCH_NAME:       构建分支名（可选；master 分支自动屏蔽归档参数）
    #
    # 历史原型库路径由函数内部从 cmake 文件路径自动推导（${CANN_INSTALL_PATH}/cann/opp/history_registry/<module>），
    # 路径存在且非空时自动传 --history_registry，无需用户传参。
    #
    # 从环境变量兜底捕获（cmake 变量未定义时生效）
    foreach(_ES_VAR GE_ES_EXTRACT_HISTORY GE_ES_RELEASE_VERSION GE_ES_RELEASE_DATE GE_ES_BRANCH_NAME)
        if (NOT DEFINED ${_ES_VAR} AND DEFINED ENV{${_ES_VAR}})
            set(${_ES_VAR} "$ENV{${_ES_VAR}}")
            message(STATUS "[ES] Captured from environment variable: ${_ES_VAR}=${${_ES_VAR}}")
        endif ()
    endforeach ()

    # master 分支：屏蔽全部归档参数，走纯代码生成模式；历史原型库路径仍传递（用于生成带重载 C++ API）
    if (GE_ES_BRANCH_NAME STREQUAL "master")
        if (GE_ES_EXTRACT_HISTORY OR GE_ES_RELEASE_VERSION OR GE_ES_RELEASE_DATE)
            message(STATUS "[ES] Branch is master, ignoring GE_ES_EXTRACT_HISTORY/GE_ES_RELEASE_VERSION/GE_ES_RELEASE_DATE/GE_ES_BRANCH_NAME, using code-generation-only mode")
        endif ()
        set(GE_ES_EXTRACT_HISTORY OFF)
        set(GE_ES_RELEASE_VERSION "")
        set(GE_ES_RELEASE_DATE "")
        set(GE_ES_BRANCH_NAME "")
    endif ()

    set(EXTRACT_HISTORY_FLAG "")
    if (GE_ES_EXTRACT_HISTORY)
        set(EXTRACT_HISTORY_FLAG "--es_mode=extract_history")
        message(STATUS "[ES] Historical prototype library mode enabled (GE_ES_EXTRACT_HISTORY=ON), gen_esb will append --es_mode=extract_history")
    endif ()
    set(RELEASE_VERSION_ARG "")
    if (GE_ES_RELEASE_VERSION)
        set(RELEASE_VERSION_ARG "--release_version=${GE_ES_RELEASE_VERSION}")
    endif ()
    set(RELEASE_DATE_ARG "")
    if (GE_ES_RELEASE_DATE)
        set(RELEASE_DATE_ARG "--release_date=${GE_ES_RELEASE_DATE}")
    endif ()
    set(BRANCH_NAME_ARG "")
    if (GE_ES_BRANCH_NAME)
        set(BRANCH_NAME_ARG "--branch_name=${GE_ES_BRANCH_NAME}")
    endif ()

    # 2.1. 检查 OPP_PROTO_TARGET 是否存在
    if (NOT TARGET ${ARG_OPP_PROTO_TARGET})
        message(FATAL_ERROR "_add_es_library_impl: OPP_PROTO_TARGET '${ARG_OPP_PROTO_TARGET}' is not a valid CMake target")
    endif ()

    # 2.2. 从 OPP_PROTO_TARGET 获取原型库输出目录
    # 支持两种方式：
    #   1. 常规库（SHARED/STATIC）：从 LIBRARY_OUTPUT_DIRECTORY 获取
    #   2. INTERFACE 库（包装）：从 INTERFACE_LIBRARY_OUTPUT_DIRECTORY 获取（自定义属性）

    # 先检查 target 类型
    get_target_property(TARGET_TYPE ${ARG_OPP_PROTO_TARGET} TYPE)

    if (TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
        # INTERFACE 库，使用自定义属性
        get_target_property(OPP_PROTO_OUTPUT_DIR ${ARG_OPP_PROTO_TARGET} INTERFACE_LIBRARY_OUTPUT_DIRECTORY)
        if (NOT OPP_PROTO_OUTPUT_DIR OR OPP_PROTO_OUTPUT_DIR STREQUAL "OPP_PROTO_OUTPUT_DIR-NOTFOUND")
            message(FATAL_ERROR "add_es_package: OPP_PROTO_TARGET '${ARG_OPP_PROTO_TARGET}' is an INTERFACE library\n"
                    "Please set custom property: set_target_properties(${ARG_OPP_PROTO_TARGET} PROPERTIES INTERFACE_LIBRARY_OUTPUT_DIRECTORY <path>)")
        endif ()
        message(STATUS "add_es_package: Detected INTERFACE library, using INTERFACE_LIBRARY_OUTPUT_DIRECTORY property")
    else ()
        # 常规库，使用标准属性
        get_target_property(OPP_PROTO_OUTPUT_DIR ${ARG_OPP_PROTO_TARGET} LIBRARY_OUTPUT_DIRECTORY)
        if (NOT OPP_PROTO_OUTPUT_DIR OR OPP_PROTO_OUTPUT_DIR STREQUAL "OPP_PROTO_OUTPUT_DIR-NOTFOUND")
            # 如果未设置，使用默认输出路径
            if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
                set(OPP_PROTO_OUTPUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
                message(STATUS "add_es_package: Using CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${OPP_PROTO_OUTPUT_DIR}")
            else ()
                set(OPP_PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
                message(STATUS "add_es_package: Using default output directory: ${OPP_PROTO_OUTPUT_DIR}")
            endif ()
        endif ()
    endif ()

    message(STATUS "add_es_package: OPP_PROTO_TARGET output directory: ${OPP_PROTO_OUTPUT_DIR}")

    # 2.3. 从 LIBRARY_OUTPUT_DIRECTORY 推导 ASCEND_OPP_PATH
    # 需要去掉 /op_proto/custom 或 /built-in/op_proto 等后缀
    set(OPP_BASE_PATH "${OPP_PROTO_OUTPUT_DIR}")

    # 尝试移除常见的路径后缀（注意 built-in 是连字符）
    string(REGEX REPLACE "/op_proto/custom$" "" OPP_BASE_PATH "${OPP_BASE_PATH}")
    string(REGEX REPLACE "/built-in/op_proto$" "" OPP_BASE_PATH "${OPP_BASE_PATH}")
    string(REGEX REPLACE "/op_proto$" "" OPP_BASE_PATH "${OPP_BASE_PATH}")


    message(STATUS "add_es_package: Derived ASCEND_OPP_PATH: ${OPP_BASE_PATH}")

    # 2.3.1. 临时定义 BUILD_DIR（完整定义在后面）
    set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_ES_LINKABLE_AND_ALL_TARGET}_build")

    # 2.3.2. 如果不是标准路径，创建标准路径并拷贝原型库
    set(USE_STANDARD_PATH FALSE)
    set(OPP_COPY_FLAG "")
    # 如果路径没有变化，说明不是标准的 OPP 目录结构
    if (OPP_BASE_PATH STREQUAL OPP_PROTO_OUTPUT_DIR)
        # 非标准路径，需要创建标准路径
        set(USE_STANDARD_PATH TRUE)
        set(STANDARD_OPP_BASE "${BUILD_DIR}/opp_standard_path_${ARG_OPP_PROTO_TARGET}")
        set(STANDARD_OPP_PROTO_DIR "${STANDARD_OPP_BASE}/op_proto/custom")

        file(MAKE_DIRECTORY ${STANDARD_OPP_PROTO_DIR})

        # 只处理非 INTERFACE 类型的库
        if (NOT TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
            # 添加自定义命令拷贝原型库 .so 到标准路径
            # 使用 $<TARGET_FILE:target> 获取实际的 .so 文件位置（运行时确定）
            set(OPP_COPY_FLAG "${BUILD_DIR}/.opp_copied.flag")
            add_custom_command(
                    OUTPUT ${OPP_COPY_FLAG}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${STANDARD_OPP_PROTO_DIR}
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${ARG_OPP_PROTO_TARGET}> ${STANDARD_OPP_PROTO_DIR}/
                    COMMAND ${CMAKE_COMMAND} -E touch ${OPP_COPY_FLAG}
                    DEPENDS ${ARG_OPP_PROTO_TARGET}
                    COMMENT "Copying OPP proto library to standard path..."
            )

            # 更新 OPP_BASE_PATH 为标准路径
            set(OPP_BASE_PATH ${STANDARD_OPP_BASE})
            message(STATUS "  - Created standard OPP path: ${STANDARD_OPP_BASE}")
        else ()
            message(STATUS "  - Skipped standard path creation (INTERFACE library)")
        endif ()
    endif ()

    # 2.4. 自动检测 gen_esb 的位置
    # 优先级: TARGET gen_esb > 从 cmake 文件路径推导 run 包中的 gen_esb

    set(USE_EXTERNAL_GEN_ESB FALSE)
    set(GEN_ESB_EXE "")
    set(ASCEND_LIB_DIR "")

    if (TARGET gen_esb)
        # 开发环境/源码编译环境: 使用源码编译的 gen_esb target
        set(USE_EXTERNAL_GEN_ESB FALSE)
        message(STATUS "Yellow Zone Environment: Use the source-compiled gen_esb target")

        # 为 gen_esb 及其依赖添加 JOB_POOL 保护
        # 防止 ar 和 ld 的文件竞态（特别是在高负载下）
        if (TARGET geir_collector)
            set_target_properties(geir_collector PROPERTIES
                    JOB_POOL_COMPILE es_gen_esb_serial
                    JOB_POOL_LINK es_gen_esb_serial
                    )
            message(STATUS "  - Applied JOB_POOL to geir_collector")
        endif ()

        if (TARGET gen_esb_impl)
            set_target_properties(gen_esb_impl PROPERTIES
                    JOB_POOL_COMPILE es_gen_esb_serial
                    JOB_POOL_LINK es_gen_esb_serial
                    )
            message(STATUS "  - Applied JOB_POOL to gen_esb_impl")
        endif ()

        set_target_properties(gen_esb PROPERTIES
                JOB_POOL_COMPILE es_gen_esb_serial
                JOB_POOL_LINK es_gen_esb_serial
                )
        message(STATUS "  - Applied JOB_POOL to gen_esb")
    else ()
        # gen_esb target 不存在，尝试从环境中查找
        message(STATUS "_add_es_library_impl: gen_esb target not found, searching in environment...")

        # 方法 1: 检查 PATH 环境变量（用户执行了 source setenv.bash 的场景）
        find_program(GEN_ESB_IN_PATH gen_esb)
        if (GEN_ESB_IN_PATH)
            set(GEN_ESB_EXE "${GEN_ESB_IN_PATH}")
            set(USE_EXTERNAL_GEN_ESB TRUE)

            # 从 gen_esb 路径推导库路径
            get_filename_component(BIN_DIR "${GEN_ESB_EXE}" DIRECTORY)
            get_filename_component(INSTALL_BASE "${BIN_DIR}" DIRECTORY)

            # 检查是否有 lib64 目录
            if (EXISTS "${INSTALL_BASE}/lib64")
                set(ASCEND_LIB_DIR "${INSTALL_BASE}/lib64")
            endif ()

            message(STATUS "    Found gen_esb in PATH (environment variables configured)")
            message(STATUS "    - gen_esb path: ${GEN_ESB_EXE}")
            message(STATUS "    - library path: ${ASCEND_LIB_DIR}")
        else ()
            # 方法 2: 从 cmake 文件路径推导 run 包中的 gen_esb
            # 如果当前文件在 /usr/local/Ascend/cann/include/ge/cmake/，则 gen_esb 在相对路径 ../../../bin/gen_esb

            # 使用文件加载时缓存的路径（在函数外部已获取）
            set(CMAKE_SCRIPT_DIR "${_ADD_ES_LIBRARY_CMAKE_DIR}")
            message(STATUS "  - gen_esb not in PATH, trying to deduce from cmake file path...")
            message(STATUS "  - Cached module directory: ${_ADD_ES_LIBRARY_CMAKE_DIR}")
            message(STATUS "  - Using script directory: ${CMAKE_SCRIPT_DIR}")

            # 尝试推导 run 包的基础路径
            # 假设结构: <base>/include/ge/cmake/ -> <base>/bin/gen_esb
            get_filename_component(POTENTIAL_GE_DIR "${CMAKE_SCRIPT_DIR}" DIRECTORY)  # 去掉 /cmake
            get_filename_component(POTENTIAL_INCLUDE_DIR "${POTENTIAL_GE_DIR}" DIRECTORY)  # 去掉 /ge
            get_filename_component(POTENTIAL_BASE_DIR "${POTENTIAL_INCLUDE_DIR}" DIRECTORY)  # 去掉 /include

            # 自动检测系统架构
            execute_process(
                    COMMAND uname -m
                    OUTPUT_VARIABLE UNAME_MACHINE
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            # 根据 uname -m 结果映射到 ASCEND_ARCH
            if (UNAME_MACHINE MATCHES "x86_64")
                set(ASCEND_ARCH "x86_64-linux")
            elseif (UNAME_MACHINE MATCHES "aarch64")
                set(ASCEND_ARCH "aarch64-linux")
            else ()
                message(FATAL_ERROR "add_es_package: Unsupported architecture: ${UNAME_MACHINE}\n"
                        "Supported architectures: x86_64, aarch64")
            endif ()

            # 尝试多种可能的路径
            set(POTENTIAL_GEN_ESB_PATHS
                    "${POTENTIAL_BASE_DIR}/bin/gen_esb"                # 可能的路径1
                    "${POTENTIAL_BASE_DIR}/${ASCEND_ARCH}/bin/gen_esb" # 可能的路径2（带架构）
                    )

            foreach (POTENTIAL_PATH ${POTENTIAL_GEN_ESB_PATHS})
                if (EXISTS ${POTENTIAL_PATH})
                    set(GEN_ESB_EXE "${POTENTIAL_PATH}")
                    set(USE_EXTERNAL_GEN_ESB TRUE)
                    # 推导库路径
                    get_filename_component(BIN_DIR "${GEN_ESB_EXE}" DIRECTORY)
                    get_filename_component(ARCH_OR_BASE_DIR "${BIN_DIR}" DIRECTORY)
                    # 检查是否有 lib64 子目录
                    if (EXISTS "${ARCH_OR_BASE_DIR}/lib64")
                        set(ASCEND_LIB_DIR "${ARCH_OR_BASE_DIR}/lib64")
                    endif ()
                    message(STATUS "    Found gen_esb by path deduction (derived from cmake path)")
                    message(STATUS "    - gen_esb path: ${GEN_ESB_EXE}")
                    message(STATUS "    - library path: ${ASCEND_LIB_DIR}")
                    message(STATUS "    - detected architecture: ${ASCEND_ARCH} (uname -m: ${UNAME_MACHINE})")
                    break()
                endif ()
            endforeach ()
        endif ()

        # 如果所有方法都失败了
        if (NOT USE_EXTERNAL_GEN_ESB)
            message(FATAL_ERROR "add_es_package: gen_esb unavailable\n"
                    "Failed to locate a usable gen_esb. Tried the following:\n"
                    "  1. Looked for the source-built gen_esb target - not found\n"
                    "  2. Looked for gen_esb in the PATH environment variable - not found\n"
                    "  3. Derived gen_esb from the cmake file path - failed\n"
                    "\n"
                    "Please ensure:\n"
                    "  1. The run package is installed completely according to the installation guide\n"
                    "  2. 'source /usr/local/Ascend/cann/bin/setenv.bash' has been executed to configure environment variables\n"
                    "  3. This cmake file is included from the run package path (e.g. /usr/local/Ascend/cann/include/ge/cmake/)\n"
                    "\n"
                    "Debug info:\n"
                    "  - current cmake script path: ${CMAKE_SCRIPT_DIR}\n"
                    "  - derived base directory: ${POTENTIAL_BASE_DIR}\n"
                    "  - gen_esb paths attempted: ${POTENTIAL_GEN_ESB_PATHS}")
        endif ()
    endif ()

    # 2.5. 使用从 OPP_PROTO_TARGET 推导的原型库路径
    set(OPP_PROTO_PATH ${OPP_BASE_PATH})
    message(STATUS "Using proto library path (derived from OPP_PROTO_TARGET): ${OPP_PROTO_PATH}")

    # 2.6. 提取 gen_esb 需要的 module name
    # ES_LINKABLE_AND_ALL_TARGET 如果是 "es_math"，则 MODULE_NAME 是 "math"（用于 gen_esb --module_name）
    # 如果 ES_LINKABLE_AND_ALL_TARGET 不以 es_ 开头，直接使用
    set(EXPORTED_TARGET "${ARG_ES_LINKABLE_AND_ALL_TARGET}")
    string(REGEX REPLACE "^es_" "" MODULE_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}")

    if (MODULE_NAME STREQUAL ARG_ES_LINKABLE_AND_ALL_TARGET)
        # 没有 es_ 前缀，建议用户使用 es_ 前缀
        message(WARNING "add_es_package: ES_LINKABLE_AND_ALL_TARGET '${ARG_ES_LINKABLE_AND_ALL_TARGET}' does not start with 'es_'\n"
                "  It is recommended to use the es_ prefix to follow the naming convention, for example: es_math, es_nn")
    endif ()

    message(STATUS "add_es_package: Module name for gen_esb: ${MODULE_NAME}")

    # 2.7. 自动推导历史原型库路径
    # 从 cmake 文件路径推导安装根目录，查找 ${CANN_INSTALL_PATH}/cann/opp/history_registry/${MODULE_NAME}，
    # 路径存在且非空时自动传 --history_registry 给 gen_esb，无需用户显式设置。
    set(HISTORY_REGISTRY_ARG "")
    set(_AUTO_HISTORY_REGISTRY "")
    if (USE_EXTERNAL_GEN_ESB)
        get_filename_component(_HIST_GE_DIR "${_ADD_ES_LIBRARY_CMAKE_DIR}" DIRECTORY)   # 去掉 /cmake
        get_filename_component(_HIST_INCLUDE_DIR "${_HIST_GE_DIR}" DIRECTORY)           # 去掉 /ge
        get_filename_component(_HIST_CANN_DIR "${_HIST_INCLUDE_DIR}" DIRECTORY)         # 去掉 /include
        set(_CANDIDATE "${_HIST_CANN_DIR}/opp/history_registry/${MODULE_NAME}")
        if (IS_DIRECTORY "${_CANDIDATE}")
            file(GLOB _HIST_CONTENTS LIST_DIRECTORIES true "${_CANDIDATE}/*")
            if (_HIST_CONTENTS)
                set(_AUTO_HISTORY_REGISTRY "${_CANDIDATE}")
                set(HISTORY_REGISTRY_ARG "--history_registry=${_CANDIDATE}")
                message(STATUS "[add_es_library] Auto-detected history registry: ${_CANDIDATE}")
            else ()
                message(STATUS "[add_es_library] History registry path exists but is empty, skipping: ${_CANDIDATE}")
            endif ()
        else ()
            message(STATUS "[add_es_library] No history registry found at ${_CANDIDATE}, skipping")
        endif ()
    endif ()

    # 版本去重：若历史原型库中已存在相同版本号
    set(_DUPLICATE_VERSION_DETECTED FALSE)
    if (GE_ES_EXTRACT_HISTORY AND GE_ES_RELEASE_VERSION AND _AUTO_HISTORY_REGISTRY)
        set(_INDEX_JSON "${_AUTO_HISTORY_REGISTRY}/index.json")
        if (EXISTS "${_INDEX_JSON}")
            file(READ "${_INDEX_JSON}" _INDEX_CONTENT)
            string(FIND "${_INDEX_CONTENT}" "\"${GE_ES_RELEASE_VERSION}\"" _VER_POS)
            if (_VER_POS GREATER_EQUAL 0)
                message(STATUS "[ES] Version ${GE_ES_RELEASE_VERSION} already exists in historical prototype library, skipping archive, still using code generation mode")
                message(STATUS "[ES] Existing history registry will be copied to output: ${ARG_OUTPUT_PATH}")
                set(GE_ES_EXTRACT_HISTORY OFF)
                set(EXTRACT_HISTORY_FLAG "")
                set(_DUPLICATE_VERSION_DETECTED TRUE)
            endif ()
        endif ()
    endif ()

    # 只要 GE_ES_EXTRACT_HISTORY=ON 就走双次调用路径
    # 有已有历史原型库 → codegen（带重载）+ extract&merge
    # 无已有历史原型库（首次构建）→ codegen + extract 生成全新历史原型库，输出到 OUTPUT_PATH
    set(COMBINED_COMMERCIAL_MODE FALSE)
    if (GE_ES_EXTRACT_HISTORY)
        set(COMBINED_COMMERCIAL_MODE TRUE)
        if (_AUTO_HISTORY_REGISTRY)
            message(STATUS "  - [add_es_library] Combined commercial mode: "
                    "code gen with overload + extract & merge history registry (two gen_esb calls internally)")
        else ()
            set(_AUTO_HISTORY_REGISTRY "${ARG_OUTPUT_PATH}")
            message(STATUS "  - [add_es_library] Combined commercial mode (first build, no existing history registry): "
                    "code gen + fresh history registry → ${ARG_OUTPUT_PATH}")
        endif ()
    endif ()

    # 代码生成步骤的 --es_mode 参数：
    # 完整商发模式下，代码生成步骤不传 --es_mode=extract_history（历史原型库生成模式由第二次 gen_esb 调用完成）
    if (COMBINED_COMMERCIAL_MODE)
        set(CODE_GEN_STEP_EXTRACT_FLAG "")
    else ()
        set(CODE_GEN_STEP_EXTRACT_FLAG "${EXTRACT_HISTORY_FLAG}")
    endif ()

    # 2.9. 检测 eager_style_graph_builder_base 的来源
    set(HAS_ES_BASE_TARGET FALSE)
    set(ES_BASE_LIB "")

    if (TARGET eager_style_graph_builder_base)
        # 源码编译的 target
        set(HAS_ES_BASE_TARGET TRUE)
        set(ES_BASE_LIB eager_style_graph_builder_base)
        message(STATUS "Using source-built eager_style_graph_builder_base target")
    elseif (USE_EXTERNAL_GEN_ESB AND ASCEND_LIB_DIR)
        # 使用 run 包环境，尝试从 run 包中查找库
        message(STATUS "Trying to locate eager_style_graph_builder_base library from run package...")

        # 在 run 包路径中查找库
        find_library(ES_BASE_LIB_FOUND
                NAMES eager_style_graph_builder_base
                PATHS ${ASCEND_LIB_DIR}
                NO_DEFAULT_PATH
                )

        if (ES_BASE_LIB_FOUND)
            set(ES_BASE_LIB ${ES_BASE_LIB_FOUND})
            message(STATUS "  Found library in run package: ${ES_BASE_LIB}")
        else ()
            message(WARNING "  Failed to find eager_style_graph_builder_base library in run package\n"
                    "  Path: ${ASCEND_LIB_DIR}\n"
                    "  Will fall back to linking by library name")
            # 如果找不到，使用库名称，依赖运行时 LD_LIBRARY_PATH
            set(ES_BASE_LIB eager_style_graph_builder_base)
        endif ()
    else ()
        # 开发环境但 target 不存在
        message(WARNING "eager_style_graph_builder_base target not available, will try to link by library name\n"
                "  Please ensure the runtime LD_LIBRARY_PATH contains the directory of this library")
        set(ES_BASE_LIB eager_style_graph_builder_base)
    endif ()

    # 3. 定义目标名称和路径变量
    # 命名规则：使用 ES_LINKABLE_AND_ALL_TARGET 直接作为库名和目录名（如 es_math -> es_math_so, es_math_a, libes_math.so, libes_math.a）
    # 注意：BUILD_DIR 已在前面定义（标准路径处理需要）

    set(SO_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}_so")  # 内部 .so target 名称
    set(A_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}_a")   # 内部 .a target 名称
    set(OBJ_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}_obj")  # 内部 OBJECT target 名称（避免重复编译）
    set(PYTHON_PKG_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}")  # Python 包名（如 es_math）

    set(GEN_CODE_DIR "${BUILD_DIR}/generated_code")
    set(PYTHON_BUILD_DIR "${BUILD_DIR}/python_package")

    # 输出目录（直接使用 ES_LINKABLE_AND_ALL_TARGET 作为子目录名）
    set(INCLUDE_DIR "${ARG_OUTPUT_PATH}/include/${ARG_ES_LINKABLE_AND_ALL_TARGET}")
    set(LIB_DIR "${ARG_OUTPUT_PATH}/lib64")
    set(WHL_DIR "${ARG_OUTPUT_PATH}/whl")

    message(STATUS "Configuring ES package: ${ARG_ES_LINKABLE_AND_ALL_TARGET}")
    message(STATUS "  - Exported target: ${EXPORTED_TARGET} (public interface)")
    message(STATUS "  - Internal .obj target: ${OBJ_NAME} (compile once)")
    message(STATUS "  - Internal .so target: ${SO_NAME}")
    message(STATUS "  - Internal .a target: ${A_NAME}")
    message(STATUS "  - Module name (for gen_esb): ${MODULE_NAME}")
    message(STATUS "  - OPP proto target: ${ARG_OPP_PROTO_TARGET}")
    message(STATUS "  - OPP proto path: ${OPP_PROTO_PATH}")
    message(STATUS "  - Output path: ${ARG_OUTPUT_PATH}")
    message(STATUS "  - Library files: lib${ARG_ES_LINKABLE_AND_ALL_TARGET}.so, lib${ARG_ES_LINKABLE_AND_ALL_TARGET}.a")
    message(STATUS "  - Python package: ${PYTHON_PKG_NAME}")

    # 4. 创建必要的目录
    file(MAKE_DIRECTORY ${BUILD_DIR})
    file(MAKE_DIRECTORY ${GEN_CODE_DIR})
    file(MAKE_DIRECTORY ${PYTHON_BUILD_DIR})
    file(MAKE_DIRECTORY ${INCLUDE_DIR})
    file(MAKE_DIRECTORY ${LIB_DIR})
    file(MAKE_DIRECTORY ${WHL_DIR})

    # 5. 创建 wrapper 文件生成脚本（单文件方案）
    # 这个脚本会在代码生成完成后，动态扫描生成的 .cpp 文件并生成 include 列表
    set(GENERATE_WRAPPER_SCRIPT "${BUILD_DIR}/generate_wrapper.cmake")
    file(WRITE ${GENERATE_WRAPPER_SCRIPT} "# Auto-generated script for creating wrapper file
# This script scans generated .cpp files and creates a single wrapper with includes
# Generated by: generate_es_package.cmake

set(GEN_CODE_DIR \"${GEN_CODE_DIR}\")
set(WRAPPER_FILE \"${GEN_CODE_DIR}/es_${MODULE_NAME}_all_in_one.cpp\")
set(MODULE_NAME \"${MODULE_NAME}\")
set(TARGET_NAME \"${ARG_ES_LINKABLE_AND_ALL_TARGET}\")

# 扫描生成的 .cpp 文件（排除 wrapper 自身）
file(GLOB CPP_FILES
    \"\${GEN_CODE_DIR}/*.cpp\"
)

# 过滤掉 wrapper 文件自己
list(FILTER CPP_FILES EXCLUDE REGEX \"es_\${MODULE_NAME}_all_in_one\\\\.cpp\")

# 排序保证稳定性
list(SORT CPP_FILES)

# 统计数量
list(LENGTH CPP_FILES NUM_OPS)

# 生成 wrapper 文件内容
set(wrapper_content \"// Auto-generated wrapper for \${TARGET_NAME} ES operators\\n\")
set(wrapper_content \"\${wrapper_content}// This file dynamically includes all generated operator implementations\\n\")
set(wrapper_content \"\${wrapper_content}//\\n\")
set(wrapper_content \"\${wrapper_content}// Build workflow (single-file mode):\\n\")
set(wrapper_content \"\${wrapper_content}//   1. gen_esb generates individual operator .cpp files\\n\")
set(wrapper_content \"\${wrapper_content}//   2. This wrapper file is regenerated with #include statements\\n\")
set(wrapper_content \"\${wrapper_content}//   3. Compiler compiles this single file (includes all implementations)\\n\")
set(wrapper_content \"\${wrapper_content}//\\n\")
set(wrapper_content \"\${wrapper_content}// Generated by: generate_es_package.cmake\\n\")
set(wrapper_content \"\${wrapper_content}// DO NOT EDIT THIS FILE MANUALLY\\n\")
set(wrapper_content \"\${wrapper_content}//\\n\")
set(wrapper_content \"\${wrapper_content}// Total operators: \${NUM_OPS}\\n\")
set(wrapper_content \"\${wrapper_content}//\\n\")
set(wrapper_content \"\${wrapper_content}\\n\")

# 添加各个算子的 include
foreach(cpp_file \${CPP_FILES})
    get_filename_component(filename \${cpp_file} NAME)
    set(wrapper_content \"\${wrapper_content}#include \\\"\${filename}\\\"\\n\")
endforeach()

# 写入文件
file(WRITE \"\${WRAPPER_FILE}\" \"\${wrapper_content}\")

message(STATUS \"[ES Wrapper] Generated: \${WRAPPER_FILE}\")
message(STATUS \"[ES Wrapper] Total operators included: \${NUM_OPS}\")
")

    # 6. 创建初始 wrapper 文件（配置阶段占位，避免首次配置报错）
    set(ALL_IN_ONE_WRAPPER "${GEN_CODE_DIR}/es_${MODULE_NAME}_all_in_one.cpp")
    if (NOT EXISTS ${ALL_IN_ONE_WRAPPER})
        file(WRITE ${ALL_IN_ONE_WRAPPER}
"// Placeholder wrapper for ${ARG_ES_LINKABLE_AND_ALL_TARGET}
// This file will be regenerated after code generation
//
// Generated by: generate_es_package.cmake
// DO NOT EDIT THIS FILE MANUALLY
")
    endif ()

    # 7. 单文件方案：只使用 wrapper 文件作为源文件
    message(STATUS "ES package '${ARG_ES_LINKABLE_AND_ALL_TARGET}' using single-file compilation mode")
    message(STATUS "  - Wrapper file: ${ALL_IN_ONE_WRAPPER}")
    message(STATUS "  - Module name: ${MODULE_NAME}")

    # 8. 定义代码生成命令（单文件方案）
    # gen_esb 生成：
    #   - 聚合头文件: es_${MODULE_NAME}_ops.h, es_${MODULE_NAME}_ops_c.h, es_${MODULE_NAME}_ops.py
    #   - 每个算子独立的 .cpp 和 .h 文件（如 es_add.cpp, es_add.h）
    #   然后运行 generate_wrapper.cmake 生成包含所有 #include 的 wrapper 文件

    set(CODE_GEN_FLAG "${GEN_CODE_DIR}/generated_code.flag")
    set(GEN_ESB_OUTPUT_DIR "${GEN_CODE_DIR}")

    # 8.1. 准备依赖列表
    # INTERFACE 库不需要构建，不添加到 DEPENDS 中
    set(CODE_GEN_DEPENDS "")
    if (NOT TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
        set(CODE_GEN_DEPENDS "${ARG_OPP_PROTO_TARGET}")
        # 如果使用了标准路径，还需要依赖拷贝完成
        if (USE_STANDARD_PATH AND OPP_COPY_FLAG)
            list(APPEND CODE_GEN_DEPENDS ${OPP_COPY_FLAG})
        endif ()
    endif ()

    # INTERFACE 库不传 EXCLUDE_OPS（无算子需要排除）
    if (TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
        set(_EXCL_OPS_ARG "")
    else ()
        set(_EXCL_OPS_ARG "${ARG_EXCLUDE_OPS}")
    endif ()

    if (USE_EXTERNAL_GEN_ESB)
        # 使用 run 包的 gen_esb
        if (CODE_GEN_DEPENDS)
            set(_COMBINED_MODE_DEPENDS "DEPENDS;${CODE_GEN_DEPENDS}")
        else ()
            # INTERFACE 库，无需依赖
            unset(_COMBINED_MODE_DEPENDS)
        endif ()
        _es_add_gen_esb_cmd(
                "${GEN_ESB_EXE}" "${ASCEND_LIB_DIR}" "${_EXCL_OPS_ARG}"
                "Generating ES API code for '${ARG_ES_LINKABLE_AND_ALL_TARGET}' using run package gen_esb..."
        )
    else ()
        # 使用源码编译的 gen_esb target
        # 源码环境的 OPP_PROTO_TARGET 一定是实际的库 target，需要添加依赖
        # 构建完整依赖列表（包含 gen_esb 和标准路径拷贝）
        set(CODE_GEN_DEPENDS "$<TARGET_FILE:gen_esb>;${ARG_OPP_PROTO_TARGET}")
        if (USE_STANDARD_PATH AND OPP_COPY_FLAG)
            list(APPEND CODE_GEN_DEPENDS ${OPP_COPY_FLAG})
        endif ()
        set(_COMBINED_MODE_DEPENDS "DEPENDS;${CODE_GEN_DEPENDS}")
        _es_add_gen_esb_cmd(
                "$<TARGET_FILE:gen_esb>" "" "${_EXCL_OPS_ARG}"
                "Generating ES API code for '${ARG_ES_LINKABLE_AND_ALL_TARGET}' using source-built gen_esb..."
        )
    endif ()

    # 9. 创建自定义目标触发代码生成
    set(CODE_GEN_TARGET "generate_${ARG_ES_LINKABLE_AND_ALL_TARGET}_code")

    add_custom_target(${CODE_GEN_TARGET} ALL
            DEPENDS ${CODE_GEN_FLAG}
            )

    # 使用 JOB_POOL 和 USES_TERMINAL 确保串行执行
    set_target_properties(${CODE_GEN_TARGET} PROPERTIES
            JOB_POOL es_gen_esb_serial
            USES_TERMINAL_BUILD ON
            )

    # 10. 创建 OBJECT 库目标（避免重复编译）
    # OBJECT 库只编译源文件生成 .o 文件，不进行归档或链接
    # 共享库和静态库都链接同一个 .o 文件，避免重复编译
    add_library(${OBJ_NAME} OBJECT ${ALL_IN_ONE_WRAPPER})

    # 为 OBJECT 库设置 POSITION_INDEPENDENT_CODE（确保生成的 .o 文件是 PIC）
    # 这对于共享库链接是必需的，特别是使用 address sanitizer 时
    set_target_properties(${OBJ_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # 10.1 创建共享库目标（使用 OBJECT 库的 .o 文件）
    add_library(${SO_NAME} SHARED $<TARGET_OBJECTS:${OBJ_NAME}>)

    # 10.2 创建静态库目标（使用 OBJECT 库的 .o 文件）
    add_library(${A_NAME} STATIC $<TARGET_OBJECTS:${OBJ_NAME}>)

    # 11. 标记生成的文件属性（告诉 CMake/IDE 这是生成的文件）
    set_source_files_properties(
            ${ALL_IN_ONE_WRAPPER}
            PROPERTIES GENERATED TRUE
    )

    # 12. 确保代码生成完成后再编译
    # 这个依赖关系确保：先执行 CODE_GEN_TARGET 生成文件，再编译
    # OBJ_NAME 依赖 CODE_GEN_TARGET，SO_NAME 和 A_NAME 依赖 OBJ_NAME
    add_dependencies(${OBJ_NAME} ${CODE_GEN_TARGET})

    # 13. 配置编译选项（应用于 OBJECT 目标，编译选项只设置一次）
    if (DEFINED AIR_COMMON_DYNAMIC_COMPILE_OPTION)
        target_compile_options(${OBJ_NAME} PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})
        message(STATUS "add_es_library: Using CANN compile options: ${AIR_COMMON_DYNAMIC_COMPILE_OPTION}")
    else ()
        # 外部环境默认编译选项
        # 使用 C++17 标准 + ABI 兼容性处理
        set(ES_COMMON_COMPILE_OPTION
                -fPIC
                -Wall
                -fstack-protector-all
                -std=c++17
                -D_GLIBCXX_USE_CXX11_ABI=0  # 使用旧的 ABI，确保与依赖库兼容
                -O2
                -Wno-free-nonheap-object # 抑制高版本的gcc12+的误报
                )
        target_compile_options(${OBJ_NAME} PRIVATE ${ES_COMMON_COMPILE_OPTION})
        message(STATUS "add_es_library: Using default compile options: ${ES_COMMON_COMPILE_OPTION}")
    endif ()

    # 13.1 强制添加 ABI 兼容性定义（解决 std::string coredump 问题）
    # 确保所有库使用相同的 std::string ABI 版本
    if (NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${OBJ_NAME} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
        message(STATUS "add_es_library: Set _GLIBCXX_USE_CXX11_ABI=0 for ABI compatibility")
    endif ()

    # 13.1.1 动态库安全链接选项
    target_link_options(${SO_NAME} PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        # 在 Release 配置下添加 -s 选项去除符号表
        $<$<CONFIG:Release>:-s>)

    # 清除so中的RPATH，避免安全风险（仅对共享库）
    set_target_properties(${SO_NAME} PROPERTIES
        SKIP_BUILD_RPATH TRUE
        SKIP_INSTALL_RPATH TRUE
    )
    # 13.2 准备头文件搜索路径列表
    set(ES_INCLUDE_DIRS ${GEN_CODE_DIR} ${INCLUDE_DIR})

    # 如果使用外部 gen_esb（run 包环境），添加 run 包的头文件路径
    if (USE_EXTERNAL_GEN_ESB)
        # 从 cmake 模块文件路径推导基础路径
        get_filename_component(HEADER_GE_DIR "${_ADD_ES_LIBRARY_CMAKE_DIR}" DIRECTORY)  # 去掉 /cmake
        get_filename_component(HEADER_INCLUDE_DIR "${HEADER_GE_DIR}" DIRECTORY)  # 去掉 /ge
        get_filename_component(HEADER_BASE_PATH "${HEADER_INCLUDE_DIR}" DIRECTORY)  # 去掉 /include

        message(STATUS "add_es_library: Deduced header base path from cmake file: ${HEADER_BASE_PATH}")

        list(APPEND ES_INCLUDE_DIRS
                ${HEADER_BASE_PATH}/include
                ${HEADER_BASE_PATH}/include/ge/
                ${HEADER_BASE_PATH}/include/external/
                )
    else ()
        # 源码环境：添加 ES base 库的头文件路径
        if (DEFINED AIR_CODE_DIR)
            list(APPEND ES_INCLUDE_DIRS
                    ${AIR_CODE_DIR}/inc/external/ge/eager_style_graph_builder/c
                    ${AIR_CODE_DIR}/inc/external/ge/eager_style_graph_builder/cpp
                    )
        endif ()
    endif ()

    # 为 OBJECT 目标设置头文件路径（PRIVATE，仅用于编译）
    target_include_directories(${OBJ_NAME} PRIVATE ${ES_INCLUDE_DIRS})

    # OBJECT 库不会实际链接（只生成 .o 文件），但会继承头文件路径用于编译
    if (HAS_ES_BASE_TARGET)
        target_link_libraries(${OBJ_NAME} PRIVATE ${ES_BASE_LIB})
    endif ()

    # 如果使用 run 包的库，添加库搜索路径（仅对共享库有效）
    if (USE_EXTERNAL_GEN_ESB AND ASCEND_LIB_DIR AND NOT HAS_ES_BASE_TARGET)
        target_link_directories(${SO_NAME} PUBLIC ${ASCEND_LIB_DIR})
    endif ()

    # 14. 配置库链接（兼容性处理，同时应用于共享库和静态库）
    set(REQUIRED_LIBS "")

    # 检查 CANN 环境特有的库
    if (TARGET metadef_headers)
        list(APPEND REQUIRED_LIBS metadef_headers)
        message(STATUS "add_es_library: Found metadef_headers target")
    else ()
        message(STATUS "add_es_library: metadef_headers target not found, skipping")
    endif ()

    if (TARGET c_sec)
        list(APPEND REQUIRED_LIBS c_sec)
        message(STATUS "add_es_library: Found c_sec target")
    else ()
        message(STATUS "add_es_library: c_sec target not found, skipping")
    endif ()

    # 设置链接库（共享库和静态库使用相同的链接库）
    if (REQUIRED_LIBS)
        # OBJECT 库不会实际链接（只生成 .o 文件），但会继承头文件路径用于编译
        target_link_libraries(${OBJ_NAME} PRIVATE ${REQUIRED_LIBS})
        target_link_libraries(${SO_NAME} PUBLIC ${ES_BASE_LIB} ${REQUIRED_LIBS})
        target_link_libraries(${A_NAME} PUBLIC ${ES_BASE_LIB} ${REQUIRED_LIBS})
        message(STATUS "add_es_library: Linking with CANN libraries: ${REQUIRED_LIBS}")
    else ()
        target_link_libraries(${SO_NAME} PUBLIC ${ES_BASE_LIB})
        target_link_libraries(${A_NAME} PUBLIC ${ES_BASE_LIB})
        message(STATUS "add_es_library: Using minimal configuration (no CANN libraries found)")
    endif ()

    # 14.1 设置共享库输出名称
    # 新的命名规则：lib<ES_LINKABLE_AND_ALL_TARGET>.so (如 libes_math.so)
    set_target_properties(${SO_NAME} PROPERTIES
            OUTPUT_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}"
            PREFIX "lib"
            SUFFIX ".so"
            )

    # 14.2 设置静态库输出名称
    # 新的命名规则：lib<ES_LINKABLE_AND_ALL_TARGET>.a (如 libes_math.a)
    set_target_properties(${A_NAME} PROPERTIES
            OUTPUT_NAME "${ARG_ES_LINKABLE_AND_ALL_TARGET}"
            PREFIX "lib"
            SUFFIX ".a"
            )

    # 15. 创建生成 setup.py 的辅助脚本
    set(PYTHON_PKG_DIR "${PYTHON_BUILD_DIR}/${PYTHON_PKG_NAME}")
    set(SETUP_PY_FILE "${PYTHON_BUILD_DIR}/setup.py")
    set(CREATE_SETUP_SCRIPT "${BUILD_DIR}/create_setup.cmake")

    # 写入辅助脚本用于在构建时生成 setup.py
    file(WRITE ${CREATE_SETUP_SCRIPT} "# Auto-generated script for creating setup.py
file(WRITE \"${SETUP_PY_FILE}\" \"from setuptools import setup, find_packages

setup(
    name='${PYTHON_PKG_NAME}',
    version='1.0.0',
    description='ES Generated API for ${ARG_ES_LINKABLE_AND_ALL_TARGET} operators',
    author='Huawei Technologies Co., Ltd.',
    packages=find_packages(),
    python_requires='>=3.7',
    entry_points={
        'ge.es.plugins': [
            '${MODULE_NAME} = ${PYTHON_PKG_NAME}:get_module',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
\")
message(STATUS \"Created setup.py for package '${ARG_ES_LINKABLE_AND_ALL_TARGET}'\")
")

    # 18. 创建 Python 包结构的命令
    set(WHL_GEN_FLAG "${PYTHON_BUILD_DIR}/whl_generated.flag")

    # 创建 __init__.py 内容生成脚本
    set(CREATE_INIT_SCRIPT "${BUILD_DIR}/create_init.cmake")
    file(WRITE ${CREATE_INIT_SCRIPT} "# Auto-generated script for creating __init__.py
file(WRITE \"${PYTHON_PKG_DIR}/__init__.py\" \"# Auto-generated ES API for ${ARG_ES_LINKABLE_AND_ALL_TARGET}
from .es_${MODULE_NAME}_ops import *

__all__ = [name for name in dir() if not name.startswith('_')]

def get_module():
    import sys
    return sys.modules[__name__]
\")
")

    # 创建拷贝 Python 文件的脚本
    set(COPY_PY_SCRIPT "${BUILD_DIR}/copy_py_files.cmake")
    file(WRITE ${COPY_PY_SCRIPT} "# Auto-generated script for copying Python files
file(GLOB PY_FILES \"${GEN_CODE_DIR}/*.py\")
foreach(py_file \${PY_FILES})
    get_filename_component(filename \${py_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(\${py_file} \"${PYTHON_PKG_DIR}/\${filename}\" COPYONLY)
    message(STATUS \"Copied: \${filename}\")
endforeach()
")

    # 检查是否跳过 wheel 包生成
    if (NOT ARG_SKIP_WHL)
        add_custom_command(
                OUTPUT ${WHL_GEN_FLAG}
                COMMAND ${CMAKE_COMMAND} -E echo "Building Python wheel for package: ${PYTHON_PKG_NAME}"
                # 清理旧的构建产物
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_BUILD_DIR}/${PYTHON_PKG_NAME}
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_BUILD_DIR}/build
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_BUILD_DIR}/dist
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_BUILD_DIR}/*.egg-info
                # 创建 Python 包目录结构
                COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_PKG_DIR}
                # 拷贝生成的 Python 文件
                COMMAND ${CMAKE_COMMAND} -P ${COPY_PY_SCRIPT}
                # 创建 __init__.py
                COMMAND ${CMAKE_COMMAND} -P ${CREATE_INIT_SCRIPT}
                # 创建 setup.py
                COMMAND ${CMAKE_COMMAND} -P ${CREATE_SETUP_SCRIPT}
                # 构建 wheel 包 - 修复 chdir 语法
                COMMAND ${CMAKE_COMMAND} -E chdir ${PYTHON_BUILD_DIR} ${Python3_EXECUTABLE} -m pip wheel . --no-deps --wheel-dir=${PYTHON_BUILD_DIR}/dist
                # 标记完成
                COMMAND ${CMAKE_COMMAND} -E touch ${WHL_GEN_FLAG}
                DEPENDS ${CODE_GEN_TARGET}
                COMMENT "Building Python wheel package for '${PYTHON_PKG_NAME}'..."
                WORKING_DIRECTORY ${PYTHON_BUILD_DIR}
        )
    else ()
        # 如果跳过 wheel 生成，创建一个空的 flag 文件
        add_custom_command(
                OUTPUT ${WHL_GEN_FLAG}
                COMMAND ${CMAKE_COMMAND} -E echo "Skipping wheel generation"
                COMMAND ${CMAKE_COMMAND} -E touch ${WHL_GEN_FLAG}
                DEPENDS ${CODE_GEN_TARGET}
                COMMENT "Skipping Python wheel package generation for '${PYTHON_PKG_NAME}'"
        )
    endif ()

    # 19. 创建 wheel 生成目标
    set(WHL_GEN_TARGET "generate_${ARG_ES_LINKABLE_AND_ALL_TARGET}_whl")
    add_custom_target(${WHL_GEN_TARGET} ALL
            DEPENDS ${WHL_GEN_FLAG}
            )

    # 20. 创建拷贝头文件的脚本
    set(COPY_HEADERS_SCRIPT "${BUILD_DIR}/copy_headers.cmake")
    file(WRITE ${COPY_HEADERS_SCRIPT} "# Auto-generated script for copying headers
file(GLOB H_FILES \"${GEN_CODE_DIR}/*.h\")
foreach(h_file \${H_FILES})
    get_filename_component(filename \${h_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(\${h_file} \"${INCLUDE_DIR}/\${filename}\" COPYONLY)
    message(STATUS \"Copied header: \${filename}\")
endforeach()
")

    # 创建拷贝 whl 文件的脚本
    set(COPY_WHL_SCRIPT "${BUILD_DIR}/copy_whl.cmake")
    file(WRITE ${COPY_WHL_SCRIPT} "# Auto-generated script for copying whl files
file(GLOB WHL_FILES \"${PYTHON_BUILD_DIR}/dist/*.whl\")
foreach(whl_file \${WHL_FILES})
    get_filename_component(filename \${whl_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(\${whl_file} \"${WHL_DIR}/\${filename}\" COPYONLY)
    message(STATUS \"Copied wheel: \${filename}\")
endforeach()
")

    # 21. 安装头文件到 include 目录
    set(INSTALL_FLAG "${BUILD_DIR}/install.flag")
    add_custom_command(
            OUTPUT ${INSTALL_FLAG}
            COMMAND ${CMAKE_COMMAND} -E echo "Installing ES package: ${ARG_ES_LINKABLE_AND_ALL_TARGET}"
            # 拷贝头文件
            COMMAND ${CMAKE_COMMAND} -P ${COPY_HEADERS_SCRIPT}
            # 拷贝共享库
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${SO_NAME}> ${LIB_DIR}/
            # 拷贝静态库
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${A_NAME}> ${LIB_DIR}/
            # 拷贝 wheel 包
            COMMAND ${CMAKE_COMMAND} -P ${COPY_WHL_SCRIPT}
            # 标记完成
            COMMAND ${CMAKE_COMMAND} -E touch ${INSTALL_FLAG}
            DEPENDS ${SO_NAME} ${A_NAME} ${WHL_GEN_TARGET}
            COMMENT "Installing ES package '${ARG_ES_LINKABLE_AND_ALL_TARGET}' to ${ARG_OUTPUT_PATH}"
            VERBATIM
    )

    # 22. 创建安装目标（内部使用）
    set(INSTALL_TARGET "install_${ARG_ES_LINKABLE_AND_ALL_TARGET}")
    add_custom_target(${INSTALL_TARGET}
            DEPENDS ${INSTALL_FLAG}
            )

    # 22.5. 创建 SMART_BUILD_TARGET（单文件方案，直接依赖安装目标）
    # 单文件方案无需二次构建，外部使用此 target 即可触发完整构建流程
    set(SMART_BUILD_TARGET "build_${ARG_ES_LINKABLE_AND_ALL_TARGET}")
    add_custom_target(${SMART_BUILD_TARGET}
            DEPENDS ${INSTALL_TARGET}
            COMMENT "ES package '${ARG_ES_LINKABLE_AND_ALL_TARGET}' build completed (single-file mode)"
            )

    # 22.6. 依赖管理：防止多个 ES packages 并发构建共享依赖
    # 如果有多个 ES packages，让后续的包依赖第一个包，确保 gen_esb 等共享依赖只构建一次
    if (NOT DEFINED FIRST_ES_LINKABLE_AND_ALL_TARGET)
        set(FIRST_ES_LINKABLE_AND_ALL_TARGET ${SMART_BUILD_TARGET} CACHE INTERNAL "First ES package build target")
        message(STATUS "add_es_package: '${SMART_BUILD_TARGET}' is the first ES package (will be built first)")
    else ()
        add_dependencies(${SMART_BUILD_TARGET} ${FIRST_ES_LINKABLE_AND_ALL_TARGET})
        message(STATUS "add_es_package: '${SMART_BUILD_TARGET}' depends on '${FIRST_ES_LINKABLE_AND_ALL_TARGET}' to avoid concurrent builds of shared dependencies")
    endif ()

    # 23. 创建对外接口库（供使用方链接）
    # 单文件方案：直接链接内部 SO target（静态库用户按需直接链接）
    add_library(${EXPORTED_TARGET} INTERFACE)

    # INTERFACE 库依赖 SMART_BUILD_TARGET（确保外部链接时会触发构建）
    add_dependencies(${EXPORTED_TARGET} ${SMART_BUILD_TARGET})

    # 设置头文件搜索路径（传递给使用方）
    target_include_directories(${EXPORTED_TARGET} INTERFACE ${ES_INCLUDE_DIRS})

    # 如果使用 run 包的库，添加库搜索路径（传递给使用方）
    if (USE_EXTERNAL_GEN_ESB AND ASCEND_LIB_DIR AND NOT HAS_ES_BASE_TARGET)
        target_link_directories(${EXPORTED_TARGET} INTERFACE ${ASCEND_LIB_DIR})
    endif ()

    # 链接到内部 SO target 和 ES base 库（静态库用户按需直接链接）
    target_link_libraries(${EXPORTED_TARGET} INTERFACE
            ${SO_NAME}
            ${ES_BASE_LIB}
            )


    # 24. 输出总结信息
    message(STATUS "ES package '${ARG_ES_LINKABLE_AND_ALL_TARGET}' configured successfully (single-file mode):")
    message(STATUS "  - Exported target (for linking): ${EXPORTED_TARGET}")
    message(STATUS "  - Build target: ${SMART_BUILD_TARGET}")
    message(STATUS "  - Wrapper file: ${ALL_IN_ONE_WRAPPER}")
    message(STATUS "  - Internal targets:")
    message(STATUS "    * Code generation: ${CODE_GEN_TARGET}")
    message(STATUS "    * Object library: ${OBJ_NAME} (compiled once)")
    message(STATUS "    * Shared library: ${SO_NAME}")
    message(STATUS "    * Static library: ${A_NAME}")
    message(STATUS "    * Wheel generation: ${WHL_GEN_TARGET}")
    message(STATUS "    * Install: ${INSTALL_TARGET}")
    message(STATUS "  - Library files: lib${ARG_ES_LINKABLE_AND_ALL_TARGET}.so, lib${ARG_ES_LINKABLE_AND_ALL_TARGET}.a")
    message(STATUS "  - Python package: ${PYTHON_PKG_NAME}")
    if (HAS_ES_BASE_TARGET)
        message(STATUS "  - ES base library: eager_style_graph_builder_base (target)")
    else ()
        message(STATUS "  - ES base library: ${ES_BASE_LIB} (from library)")
    endif ()
    message(STATUS "")
    message(STATUS "Build mode: Single-file compilation (no reconfiguration needed)")
    message(STATUS "")
    message(STATUS "Quick start:")
    message(STATUS "  1. Link:    target_link_libraries(your_target PRIVATE ${EXPORTED_TARGET})")
    message(STATUS "  2. Build:   make your_target  (recommended, triggers automatically)")
    message(STATUS "")

endfunction()

# ======================================================================================================================
# 对外接口函数 1: add_es_library_and_whl
#
# 功能: 生成 ES API 的完整产物（C/C++ 动态库 + Python wheel 包）
#
# 参数:
#   ES_LINKABLE_AND_ALL_TARGET  - [必需] 对外暴露的接口库 target 名称（如 es_math）
#   OPP_PROTO_TARGET   - [必需] 算子原型库的 CMake target 名称
#   OUTPUT_PATH        - [必需] 产物输出的根目录
#
# 使用示例:
#   add_es_library_and_whl(
#       ES_LINKABLE_AND_ALL_TARGET es_math
#       OPP_PROTO_TARGET  opgraph_math
#       OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
#   )
#
# 生成产物:
#   - C/C++ 头文件: include/es_math/*.h
#   - 动态库: lib64/libes_math.so
#   - Python 包: whl/es_math-1.0.0-py3-none-any.whl
# ======================================================================================================================
function(add_es_library_and_whl)
    _add_es_library_impl(${ARGN})
endfunction()

# ======================================================================================================================
# 对外接口函数 2: add_es_library
#
# 功能: 只生成 ES API 的 C/C++ 动态库（不生成 Python wheel 包）
#
# 参数:
#   ES_LINKABLE_AND_ALL_TARGET  - [必需] 对外暴露的接口库 target 名称（如 es_math）
#   OPP_PROTO_TARGET   - [必需] 算子原型库的 CMake target 名称
#   OUTPUT_PATH        - [必需] 产物输出的根目录
#
# 使用示例:
#   add_es_library(
#       ES_LINKABLE_AND_ALL_TARGET es_math
#       OPP_PROTO_TARGET  opgraph_math
#       OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
#   )
#
# 生成产物:
#   - C/C++ 头文件: include/es_math/*.h
#   - 动态库: lib64/libes_math.so
#   - ⚠️  不生成 Python wheel 包
#
# 适用场景:
#   - 纯 C/C++ 项目，不需要 Python 接口
#   - 加快构建速度（跳过 wheel 打包步骤）
# ======================================================================================================================
function(add_es_library)
    _add_es_library_impl(${ARGN} SKIP_WHL)
endfunction()

