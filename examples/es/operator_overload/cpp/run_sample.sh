#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# 选项：
#   -t, --target [sample|sample_and_run]  指定要构建和运行的目标（默认: sample）
#   -h, --help                                    显示帮助信息


set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET   指定要构建和运行的目标 (sample 或 sample_and_run)
  -h, --help            显示此帮助信息

默认行为:
  当未指定目标时，默认构建并dump图
EOF
    exit 0
}

# 默认目标
TARGET="sample"
# 默认测试用例
CASE_NAME="add"

# ---------- 解析命令行参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# 验证目标有效性
VALID_TARGETS=("sample" "sample_and_run")
if [[ ! " ${VALID_TARGETS[@]} " =~ " ${TARGET} " ]]; then
    echo "错误: 无效目标 '${TARGET}'。有效目标: ${VALID_TARGETS[*]}" >&2
    exit 1
fi

echo "[Info] 目标设置为: ${TARGET}"
echo "[Info] 测试用例设置为: ${CASE_NAME}"

set +u
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo -e "ERROR 环境变量ASCEND_HOME_PATH 未配置" >&2
  echo -e "ERROR 请先执行: source /usr/local/Ascend/cann/set_env.sh   " >&2
  exit 1
fi


# ---------- 自动获取系统架构 ----------
ARCH=$(uname -m)
# 映射架构名称
case "${ARCH}" in
  x86_64|amd64)
    ASCEND_ARCH="x86_64-linux"
    ;;
  aarch64|arm64)
    ASCEND_ARCH="aarch64-linux"
    ;;
  *)
    echo "WARNING: 未识别的架构 ${ARCH}，使用默认值 x86_64-linux" >&2
    ASCEND_ARCH="x86_64-linux"
    ;;
esac

echo "[Info] 检测到系统架构: ${ARCH}"
echo "[Info] 使用 ASCEND 架构: ${ASCEND_ARCH}"

ASCEND_LIB_DIR="${ASCEND_HOME_PATH}/lib64"
echo "[Info] ASCEND_LIB_DIR = ${ASCEND_LIB_DIR}"

# 预先设置 LD_LIBRARY_PATH，保证 gen_esb 能加载
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] 预先设置 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} 以支持 gen_esb 运行"

# ---------- 3. 生成 build 目录 ----------
BUILD_DIR="build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "[Info] 创建构建目录 ${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
fi

# ---------- 5. 设置 LD_LIBRARY_PATH ----------
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] LD_LIBRARY_PATH 已设置为: ${LD_LIBRARY_PATH}"
# ---------- 6. 运行指定目标 ----------
case "${TARGET}" in
  sample)
    echo "[Info] 开始准备并编译目标: sample"
    echo "[Info] 清理旧的 ${BUILD_DIR}..."
    [ -n "${BUILD_DIR}" ] && rm -rf "${BUILD_DIR}" || true
    mkdir -p "${BUILD_DIR}"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target sample -j"$(nproc)"

    echo "[Info] 运行 ${BUILD_DIR}/sample dump ${CASE_NAME}"
    if [[ -x "${BUILD_DIR}/sample" ]]; then
      "${BUILD_DIR}/sample" dump "${CASE_NAME}"
      echo "[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示"
    else
      echo "ERROR: 找不到或不可执行 ${BUILD_DIR}/sample" >&2
      exit 1
    fi
    ;;
  sample_and_run)
    echo "[Info] 开始准备并编译目标: sample_and_run"
    bash "$0" -t sample
    echo "[Info] 设置NPU设备下的环境变量 ${ASCEND_HOME_PATH}/set_env.sh"
    echo "[Info] 检查环境变量和文件"
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
      echo "[Error] ASCEND_HOME_PATH 未设置"
      exit 1
    fi

    if [ -z "${ASCEND_ARCH:-}" ]; then
      echo "[Error] ASCEND_ARCH 未设置"
      exit 1
    fi

    SETENV_FILE="${ASCEND_HOME_PATH}/set_env.sh"
    if [ ! -f "$SETENV_FILE" ]; then
      echo "[Error] set_env.sh 不存在: $SETENV_FILE"
      exit 1
    fi
    # 临时禁用错误退出进行 source
    set +e
    source "$SETENV_FILE"
    set -e
      echo "[Info] 运行 ${BUILD_DIR}/sample run ${CASE_NAME}"
      "${BUILD_DIR}/sample" run "${CASE_NAME}" && echo "[Success] sample_and_run 执行成功，pbtxt和data输出dump 已生成在当前目录"
      ;;
  *)
    echo "错误: 未知目标 ${TARGET}" >&2
    exit 1
    ;;
esac