#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# 说明：
#   sample_and_run_python: 编译并在2个NPU设备上运行EP图（多卡并行）
#
# 注意：
#   本示例仅支持:
#     - A5(d806): 当前示例不支持此形态
#     - A2(d802): 使用 rank_table/a2/rank_table_2p.json (v1.0)
#   其他硬件平台暂不支持。

set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET      指定要构建和运行的目标 (sample_and_run_python)
  -h, --help               显示此帮助信息

默认行为:
  当未指定目标时，默认构建、dump图并在2个NPU设备上运行

注意:
  本示例仅支持:
    A2(d802) -> rank_table/a2/rank_table_2p.json (v1.0)
    A5(d806) -> 不支持此形态（脚本会报错退出）
  其他硬件平台暂不支持
EOF
    exit 0
}

select_rank_table_by_platform() {
  local curr_dir="$1"
  if ! command -v lspci >/dev/null 2>&1; then
    echo "[Error] 未找到 lspci 命令，无法识别硬件平台。仅支持 A2(d802)，A5(d806)不支持此形态" >&2
    return 1
  fi

  local platform=""
  local rank_table_file=""
  local pci_output=""
  pci_output="$(lspci 2>/dev/null || true)"
  if [[ -z "${pci_output}" ]]; then
    echo "[Error] 无法获取 lspci 输出，无法识别硬件平台。仅支持 A2(d802)，A5(d806)不支持此形态" >&2
    return 1
  fi

  if grep -qi "d806" <<<"${pci_output}"; then
    echo "[Error] 检测到 A5(d806)，当前示例不支持此形态。" >&2
    echo "[Hint] 请切换到 A2(d802) 环境运行。" >&2
    return 1
  elif grep -qi "d802" <<<"${pci_output}"; then
    platform="A2(d802)"
    rank_table_file="${curr_dir}/rank_table/a2/rank_table_2p.json"
  else
    echo "[Error] 当前硬件暂不支持。仅支持 A2(d802)。" >&2
    return 1
  fi

  if [[ ! -f "${rank_table_file}" ]]; then
    echo "[Error] rank table 文件不存在: ${rank_table_file}" >&2
    return 1
  fi
  export RANK_TABLE_FILE="${rank_table_file}"
  echo "[Info] 检测到平台: ${platform}"
  echo "[Info] 使用 rank table: ${RANK_TABLE_FILE}"
}

read_device_ids_from_rank_table() {
  if [[ -z "${RANK_TABLE_FILE:-}" ]]; then
    echo "[Error] RANK_TABLE_FILE 未设置，无法解析 device_id" >&2
    return 1
  fi
  if [[ ! -r "${RANK_TABLE_FILE}" ]]; then
    echo "[Error] rank table 文件不可读: ${RANK_TABLE_FILE}" >&2
    return 1
  fi

  local parsed=""
  parsed=$(awk '
  {
    if ($0 ~ /"device_id"[[:space:]]*:[[:space:]]*"?[0-9]+"?/) {
      value = $0
      sub(/.*"device_id"[[:space:]]*:[[:space:]]*"?/, "", value)
      sub(/[^0-9].*/, "", value)
      cur_device = value
    }
    if ($0 ~ /"rank_id"[[:space:]]*:[[:space:]]*"?[0-9]+"?/) {
      value = $0
      sub(/.*"rank_id"[[:space:]]*:[[:space:]]*"?/, "", value)
      sub(/[^0-9].*/, "", value)
      cur_rank = value
    }
    if (cur_device != "" && cur_rank != "") {
      if (cur_rank == "0" && dev0 == "") {
        dev0 = cur_device
      } else if (cur_rank == "1" && dev1 == "") {
        dev1 = cur_device
      }
      cur_device = ""
      cur_rank = ""
    }
  }
  END {
    if (dev0 == "" || dev1 == "") {
      exit 1
    }
    print dev0 " " dev1
  }' "${RANK_TABLE_FILE}")
  local awk_status=$?
  if [[ ${awk_status} -ne 0 || -z "${parsed}" ]]; then
    echo "[Error] 无法从 rank_table 解析 rank_id 0/1 对应 device_id: ${RANK_TABLE_FILE}" >&2
    echo "[Hint] 请检查 rank_table 是否同时包含 rank_id 与 device_id 字段" >&2
    echo "[Debug] rank_table 关键行预览:" >&2
    grep -nE '"rank_id"|"device_id"' "${RANK_TABLE_FILE}" >&2 || true
    return 1
  fi

  read -r DEVICE_ID_0 DEVICE_ID_1 <<<"${parsed}"
  if [[ -z "${DEVICE_ID_0}" || -z "${DEVICE_ID_1}" ]]; then
    echo "[Error] rank_table 解析结果异常: '${parsed}'" >&2
    echo "[Hint] 期望格式: '<device_id_0> <device_id_1>'" >&2
    return 1
  fi
  if [[ ! "${DEVICE_ID_0}" =~ ^[0-9]+$ || ! "${DEVICE_ID_1}" =~ ^[0-9]+$ ]]; then
    echo "[Error] rank_table 解析结果不是纯数字: DEVICE_ID_0='${DEVICE_ID_0}', DEVICE_ID_1='${DEVICE_ID_1}'" >&2
    return 1
  fi
  if [[ "${DEVICE_ID_0}" == "${DEVICE_ID_1}" ]]; then
    echo "[Warning] rank_id 0/1 映射到同一 device_id=${DEVICE_ID_0}，请确认配置是否符合预期" >&2
  fi
  echo "[Info] 从 rank_table 读取到设备ID: DEVICE_ID_0=${DEVICE_ID_0}, DEVICE_ID_1=${DEVICE_ID_1}"
}

# 默认目标
TARGET="sample_and_run_python"

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
echo "[Info] 测试用例设置为: ep"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
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

ASCEND_LIB_DIR="${ASCEND_HOME_PATH}/lib64"
BUILD_DIR="build"

# ----------  设置 LD_LIBRARY_PATH ----------
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# ----------  运行 python 图构建代码 ----------
dump_and_run_python_graph(){
  SHOWCASE_DIR="src"
  INSTALL_LIB_DIR="build/whl_package"

  # 设置PYTHONPATH
  if [[ -d "${INSTALL_LIB_DIR}" ]]; then
      export PYTHONPATH="$PWD/${INSTALL_LIB_DIR}:${PYTHONPATH:-}"
  else
      echo "[Warning] 未找到 ES Python 模块路径: ${INSTALL_LIB_DIR}"
  fi

  if [[ ! -d "${SHOWCASE_DIR}" ]]; then
      echo "[Warning] 展示的目录不存在: ${SHOWCASE_DIR}"
      return 0
  fi

  if [[ -z "${DEVICE_ID_0:-}" || -z "${DEVICE_ID_1:-}" ]]; then
    echo "[Error] 设备ID未准备完成，请先执行平台检测并解析 rank_table" >&2
    return 1
  fi
  echo "[Info] 开始运行多卡并行任务..."

  # 在第一个设备上运行 (RANK_ID=0)
  echo "[Info] 在设备${DEVICE_ID_0}上运行 Python sample (RANK_ID=0)"
  RANK_ID=0 python3 src/make_ep_graph.py "$DEVICE_ID_0" &
  PID_DEV0=$!

  # 在第二个设备上运行 (RANK_ID=1)
  echo "[Info] 在设备${DEVICE_ID_1}上运行 Python sample (RANK_ID=1)"
  RANK_ID=1 python3 src/make_ep_graph.py "$DEVICE_ID_1" &
  PID_DEV1=$!

  echo "[Info] 等待所有设备任务完成..."
  ALL_SUCCESS=true

  # 等待第一个设备进程
  wait "${PID_DEV0}"
  STATUS_DEV0=$?
  if [[ ${STATUS_DEV0} -eq 0 ]]; then
    echo "[Info] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 成功完成"
  else
    echo "[Warning] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 失败 (状态码: ${STATUS_DEV0})"
    ALL_SUCCESS=false
  fi

  # 等待第二个设备进程
  wait "${PID_DEV1}"
  STATUS_DEV1=$?
  if [[ ${STATUS_DEV1} -eq 0 ]]; then
    echo "[Info] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 成功完成"
  else
    echo "[Warning] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 失败 (状态码: ${STATUS_DEV1})"
    ALL_SUCCESS=false
  fi

  [ "$ALL_SUCCESS" = true ]
}

case "${TARGET}" in
  sample_and_run_python)
    echo "[Info] 提前进行平台检测，避免在不支持平台上耗时编译"
    CURR_DIR=$(pwd)
    if ! select_rank_table_by_platform "${CURR_DIR}"; then
      exit 1
    fi
    if ! read_device_ids_from_rank_table; then
      echo "[Error] 无法从 rank_table 中读取 rank_id 0/1 对应的 device_id" >&2
      exit 1
    fi

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
      echo "[Error] sample 执行失败" >&2
      exit 1
    fi
    ;;
  *)
    echo "错误: 未知目标 ${TARGET}" >&2
    exit 1
    ;;
esac
