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

set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET   指定要构建和运行的目标 (sample_and_run_python)
  -h, --help            显示此帮助信息

默认行为:
  当未指定目标时，默认构建并dump图
EOF
    exit 0
}

# 默认目标
TARGET="sample"

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
VALID_TARGETS=("sample_and_run_python")
if [[ ! " ${VALID_TARGETS[@]} " =~ " ${TARGET} " ]]; then
    echo "错误: 无效目标 '${TARGET}'。有效目标: ${VALID_TARGETS[*]}" >&2
    exit 1
fi

echo "[Info] 目标设置为: ${TARGET}"
echo "[Info] 测试用例设置为: add"

set +u
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo -e "ERROR 环境变量ASCEND_HOME_PATH 未配置" >&2
  echo -e "ERROR 请先执行: source /usr/local/Ascend/cann/set_env.sh  " >&2
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

export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# ---------- 3. 生成 build 目录 ----------
BUILD_DIR="build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "[Info] 创建构建目录 ${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
fi

# ----------  设置 LD_LIBRARY_PATH ----------
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] LD_LIBRARY_PATH 已设置为: ${LD_LIBRARY_PATH}"
# ----------  运行 python 图构建代码 ----------

dump_and_run_python_graph(){
  echo "[Info] 开始运行 Python 图构建代码"
  SHOWCASE_DIR="src"
  INSTALL_LIB_DIR="build/whl_package"
  TMP_ES_PYTHON_PATH=""
  if [[ -d "${INSTALL_LIB_DIR}" ]]; then
      TMP_ES_PYTHON_PATH=$PWD/${INSTALL_LIB_DIR}
      echo "[Info] TMP_ES_PYTHON_PATH = ${TMP_ES_PYTHON_PATH}"
  fi
  if [[ -n "${TMP_ES_PYTHON_PATH}" ]]; then
      export PYTHONPATH="${TMP_ES_PYTHON_PATH}:${PYTHONPATH:-}"
      echo "[Info] PYTHONPATH = ${PYTHONPATH}"
  else
      echo "[Warning] 未找到有效的临时生成的Es Python模块路径"
  fi
  echo "[Info] LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
  if [[ ! -d "${SHOWCASE_DIR}" ]]; then
      echo "[Warning] 展示的目录不存在 ${SHOWCASE_DIR}"
      return 0
  fi

  python_files=()
  while IFS= read -r -d '' file; do
    python_files+=("$file")
  done < <(find  "${SHOWCASE_DIR}" -name "*.py" -print0)

  if [[ ${#python_files[@]} -eq 0 ]]; then
    echo "[Warning] 在 ${SHOWCASE_DIR} 未找到 Python 文件"
    return 0
  fi

  local has_error=0
  for py_file in "${python_files[@]}"; do
    echo "[Info] 运行：${py_file} "
    if python3 "${py_file}"; then
      echo "[Success] ${py_file} 执行成功"
    else
      echo "[Error] ${py_file} 执行失败" >&2
      has_error=1

    fi
  done

  if [[ ${has_error} -ne 0 ]]; then
    return 1
  fi
  return 0
}

case "${TARGET}" in
  sample_and_run_python)
    echo "[Info] 开始清理构建目录并准备重编译"
    rm -rf "${BUILD_DIR}"
    echo "[Info] 创建构建目录 ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    echo "[Info] 开始CMake构建"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    echo "[Info] 开始构建ES库"
    cmake --build "${BUILD_DIR}" --target build_es_all -j"$(nproc)"
    echo "[Info] 安装ES库"
    pip install --force-reinstall --upgrade --target ./${BUILD_DIR}/whl_package  "./${BUILD_DIR}/output/whl/es_all-1.0.0-py3-none-any.whl"
    export LD_LIBRARY_PATH="$PWD/${BUILD_DIR}/output/lib64:$LD_LIBRARY_PATH"
    if dump_and_run_python_graph; then
      echo "[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示"
    else
      echo "[Error] sample 执行失败，请检查上述错误信息" >&2
      exit 1
    fi
    ;;
  *)
    echo "错误: 未知目标 ${TARGET}" >&2
    exit 1
    ;;
esac