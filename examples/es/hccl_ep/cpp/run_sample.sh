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
# 选项：
#   -t, --target [sample|sample_and_run]  指定要构建和运行的目标（默认: sample）
#   -h, --help                                    显示帮助信息
#
# 说明：
#   sample_and_run: 编译并在2个NPU设备上运行EP图（多卡并行）
#
# 注意：
#   本示例仅支持:
#     - A5(d806): 使用 rank_table/a5/rank_table_2p.json (v2.0)
#     - A2(d802): 使用 rank_table/a2/rank_table_2p.json (v1.0)
#   其他硬件平台暂不支持。


set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET      指定要构建和运行的目标 (sample 或 sample_and_run)
  -h, --help               显示此帮助信息

默认行为:
  当未指定目标时，默认构建并dump图

注意:
  本示例仅支持:
    A5(d806) -> rank_table/a5/rank_table_2p.json (v2.0)
    A2(d802) -> rank_table/a2/rank_table_2p.json (v1.0)
  其他硬件平台暂不支持
EOF
    exit 0
}

select_rank_table_by_platform() {
  local curr_dir="$1"
  if ! command -v lspci >/dev/null 2>&1; then
    echo "[Error] 未找到 lspci 命令，无法识别硬件平台。仅支持 A5(d806)/A2(d802)" >&2
    return 1
  fi

  local platform=""
  local rank_table_file=""
  local pci_output=""
  pci_output="$(lspci 2>/dev/null || true)"
  if [[ -z "${pci_output}" ]]; then
    echo "[Error] 无法获取 lspci 输出，无法识别硬件平台。仅支持 A5(d806)/A2(d802)" >&2
    return 1
  fi

  if grep -qi "d806" <<<"${pci_output}"; then
    platform="A5(d806)"
    rank_table_file="${curr_dir}/rank_table/a5/rank_table_2p.json"
  elif grep -qi "d802" <<<"${pci_output}"; then
    platform="A2(d802)"
    rank_table_file="${curr_dir}/rank_table/a2/rank_table_2p.json"
  else
    echo "[Error] 当前硬件暂不支持。仅支持 A5(d806) 和 A2(d802)" >&2
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
TARGET="sample"
# 默认测试用例
CASE_NAME="ep"

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
  echo -e "ERROR 请先执行: source /usr/local/Ascend/cann/set_env.sh  " >&2
  exit 1
fi

# ---------- 自动获取系统架构 ----------
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

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
      "${BUILD_DIR}/sample" dump ${CASE_NAME}
      echo "[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示"
    else
      echo "ERROR: 找不到或不可执行 ${BUILD_DIR}/sample" >&2
      exit 1
    fi
    ;;
  sample_and_run)
    echo "[Info] 开始准备并编译目标: sample_and_run (Multi-Device EP)"
    echo "[Info] 将根据硬件平台自动选择 rank_table"

    # 先编译 sample
    bash "$0" -t sample

    echo "[Info] 设置NPU设备下的环境变量 ${ASCEND_HOME_PATH}/set_env.sh"
    echo "[Info] 检查环境变量和文件"
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
      echo "[Error] ASCEND_HOME_PATH 未设置"
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

    # 根据硬件平台选择rank table
    CURR_DIR=$(pwd)
    if ! select_rank_table_by_platform "${CURR_DIR}"; then
      exit 1
    fi
    if ! read_device_ids_from_rank_table; then
      echo "[Error] 无法从 rank_table 中读取 rank_id 0/1 对应的 device_id" >&2
      exit 1
    fi
    echo "[Info] 开始运行 sample_and_run..."

    # 在第一个设备上运行 (RANK_ID=0)
    echo "[Info] 在设备${DEVICE_ID_0}上运行 sample (RANK_ID=0)"
    RANK_ID=0 DEVICE_ID="$DEVICE_ID_0" "${BUILD_DIR}/sample" run &
    PID_DEV0=$!

    # 在第二个设备上运行 (RANK_ID=1)
    echo "[Info] 在设备${DEVICE_ID_1}上运行 sample (RANK_ID=1)"
    RANK_ID=1 DEVICE_ID="$DEVICE_ID_1" "${BUILD_DIR}/sample" run &
    PID_DEV1=$!

    echo "[Info] 等待所有设备任务完成..."
    ALL_SUCCESS=true

    # 等待第一个设备进程
    echo "[Info] 等待设备${DEVICE_ID_0}进程 ${PID_DEV0} ..." >&2
    wait "${PID_DEV0}"
    STATUS_DEV0=$?
    if [[ ${STATUS_DEV0} -eq 0 ]]; then
      echo "[Info] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 成功完成" >&2
    else
      echo "[Warning] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 以非零状态退出: ${STATUS_DEV0}" >&2
      ALL_SUCCESS=false
    fi

    # 等待第二个设备进程
    echo "[Info] 等待设备${DEVICE_ID_1}进程 ${PID_DEV1} ..." >&2
    wait "${PID_DEV1}"
    STATUS_DEV1=$?
    if [[ ${STATUS_DEV1} -eq 0 ]]; then
      echo "[Info] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 成功完成" >&2
    else
      echo "[Warning] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 以非零状态退出: ${STATUS_DEV1}" >&2
      ALL_SUCCESS=false
    fi

    echo "========================================"
    echo "[Info] 所有设备任务处理完毕"
    echo "========================================"

    if [ "$ALL_SUCCESS" = true ]; then
      echo "[Success] sample_and_run 执行成功，pbtxt和data输出dump 已生成在当前目录"
    else
      echo "[Error] sample_and_run 执行失败！"
      exit 1
    fi
    ;;
  *)
    echo "错误: 未知目标 ${TARGET}" >&2
    exit 1
    ;;
esac
