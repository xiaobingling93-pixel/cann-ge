#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ==============================================================================
# ge (Graph Engine) 编译环境依赖检查脚本
#
# 仓库:   https://gitcode.com/cann/ge
# 依据:   https://gitcode.com/cann/ge/blob/master/docs/build.md
#         https://gitcode.com/cann/ge/blob/master/requirements.txt
# 用法:   bash scripts/check_env.sh
#
# 本脚本所有检查项和版本约束严格来源于 docs/build.md 和 requirements.txt，
# 如文档更新，请同步修改本脚本。
# ==============================================================================

# 不使用 set -e, 避免 grep 无匹配时静默退出

# ==================== 颜色定义 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==================== 计数器 ====================
ERROR_COUNT=0
WARNING_COUNT=0
PASS_COUNT=0

# ==============================================================================
# 版本要求 (严格来源于 docs/build.md 和 requirements.txt)
#
# build.md 原文:
#   - GCC >= 7.3.x
#   - Python3 >= 3.9.x
#   - CMake >= 3.16.0 （建议使用3.20.0版本）
#   - bash >= 5.1.16
#   - ccache/asan/autoconf/automake/libtool/gperf/lcov/libasan/patch/graph-easy
#     (其中graph-easy可选)
#
# requirements.txt:
#   numpy
#   pybind11>=2.13.6,<3.0.0
#   jinja2
#   setuptools>=59.0.1
#   wheel>=0.37.1
#   coverage
#   cloudpickle
# ==============================================================================
REQUIRED_GCC_MIN="7.3.0"
REQUIRED_PYTHON_MIN="3.9.0"
REQUIRED_CMAKE_MIN="3.16.0"
REQUIRED_CMAKE_RECOMMEND="3.20.0"
REQUIRED_BASH_MIN="5.1.16"
REQUIRED_PYBIND11_MIN="2.13.6"
REQUIRED_SETUPTOOLS_MIN="59.0.1"
REQUIRED_WHEEL_MIN="0.37.1"

# ==================== 工具函数 ====================
log_pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "  ${GREEN}[PASS]${NC}    $1"
}

log_warn() {
    WARNING_COUNT=$((WARNING_COUNT + 1))
    echo -e "  ${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    ERROR_COUNT=$((ERROR_COUNT + 1))
    echo -e "  ${RED}[ERROR]${NC}   $1"
}

log_info() {
    echo -e "  ${BLUE}[INFO]${NC}    $1"
}

version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

version_lt() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ] && [ "$1" != "$2" ]
}

check_command() {
    command -v "$1" &>/dev/null
}

# 安全提取版本号 (支持 1/2/3/4 段, 失败返回 0.0.0)
extract_version() {
    local input="$1"
    local ver

    # 移除前缀 v 或 V
    input=$(echo "$input" | sed 's/^[vV]//')
    # 移除后缀（-rc, -alpha, -beta 等）
    input=$(echo "$input" | sed 's/-.*$//')

    ver=$(echo "$input" | grep -oP '\d+\.\d+(\.\d+)*' | head -n1 || true)

    if [ -z "$ver" ]; then
        ver=$(echo "$input" | grep -oP '\d+' | head -n1 || true)
        if [ -n "$ver" ]; then
            ver="${ver}.0.0"
        else
            echo "0.0.0"
            return
        fi
    fi

    if [[ "$ver" =~ ^[0-9]+\.[0-9]+$ ]]; then
        ver="${ver}.0"
    fi

    echo "$ver"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ==============================================================================
echo ""
echo "=================================================================="
echo "  ge (Graph Engine) 编译环境依赖检查"
echo "  仓库: https://gitcode.com/cann/ge"
echo "  依据: docs/build.md + requirements.txt"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  系统: $(uname -s) $(uname -m)"
echo "=================================================================="

# ==================== 1. 操作系统 ====================
print_header "1. 操作系统 [可选]"

OS_NAME=$(uname -s)
OS_ARCH=$(uname -m)

if [ "$OS_NAME" = "Linux" ]; then
    log_info "操作系统: Linux"
else
    log_info "其他操作系统: $OS_NAME"
fi

if [ "$OS_ARCH" = "x86_64" ] || [ "$OS_ARCH" = "aarch64" ]; then
    log_info "CPU 架构: $OS_ARCH"
else
    log_info "其他CPU架构: $OS_ARCH (build.md 验证过 x86_64/aarch64)"
fi

if [ -f /etc/os-release ]; then
    DISTRO=$(grep "^ID=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
    DISTRO_VER=$(grep "^VERSION_ID=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
    log_info "发行版: $DISTRO $DISTRO_VER"
fi

# ==================== 2. GCC ====================
# build.md: GCC >= 7.3.x
print_header "2. GCC/G++ [build.md: >= $REQUIRED_GCC_MIN]"

if check_command gcc; then
    GCC_RAW=$(gcc -dumpversion)
    GCC_VER=$(extract_version "$GCC_RAW")
    log_info "gcc -dumpversion: $GCC_RAW → $GCC_VER"

    if version_ge "$GCC_VER" "$REQUIRED_GCC_MIN"; then
        log_pass "GCC $GCC_VER (>= $REQUIRED_GCC_MIN)"
    else
        log_error "GCC 版本过低: $GCC_VER (build.md 要求 >= $REQUIRED_GCC_MIN)"
        log_info "安装 (Ubuntu): sudo apt-get install build-essential"
        log_info "安装 (CentOS): sudo yum install devtoolset-7-gcc devtoolset-7-gcc-c++"
    fi
else
    log_error "未安装 GCC (build.md 要求 >= $REQUIRED_GCC_MIN)"
    log_info "安装 (Ubuntu): sudo apt-get install build-essential"
fi

if check_command g++; then
    GPP_RAW=$(g++ -dumpversion)
    GPP_VER=$(extract_version "$GPP_RAW")
    log_pass "G++ $GPP_VER"
else
    log_error "未安装 G++"
    log_info "安装 (Ubuntu): sudo apt-get install g++"
fi

# ==================== 3. Python3 ====================
# build.md: Python3 >= 3.9.x
# build.md: 还需要额外安装 coverage, 并将 Python3 的 bin 路径添加到 PATH
print_header "3. Python3 [build.md: >= $REQUIRED_PYTHON_MIN]"

PYTHON_CMD=""
if check_command python3; then
    PYTHON_CMD="python3"
    PYTHON_RAW=$($PYTHON_CMD --version 2>&1)
    PYTHON_VER=$(extract_version "$PYTHON_RAW")
elif check_command python; then
    PYTHON_CMD="python"
    PYTHON_RAW=$($PYTHON_CMD --version 2>&1)
    PYTHON_VER=$(extract_version "$PYTHON_RAW")
fi

if [ -n "$PYTHON_CMD" ]; then
    log_info "$PYTHON_CMD --version: $PYTHON_RAW → $PYTHON_VER"

    if version_ge "$PYTHON_VER" "$REQUIRED_PYTHON_MIN"; then
        log_pass "Python $PYTHON_VER (>= $REQUIRED_PYTHON_MIN)"
    else
        log_error "Python 版本过低: $PYTHON_VER (build.md 要求 >= $REQUIRED_PYTHON_MIN)"
        log_info "请安装 Python 3.9 或更高版本"
    fi

    # Python 开发头文件
    PYTHON_INCLUDE=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || true)
    if [ -n "$PYTHON_INCLUDE" ] && [ -f "$PYTHON_INCLUDE/Python.h" ]; then
        log_pass "Python 开发头文件: $PYTHON_INCLUDE/Python.h"
    else
        log_error "缺少 Python.h"
        log_info "安装 (Ubuntu): sudo apt-get install python3-dev"
        log_info "安装 (CentOS): sudo yum install python3-devel"
    fi

    # build.md: 将 Python3 的 bin 路径添加到 PATH
    PYTHON_BIN_DIR=$($PYTHON_CMD -c "import sys; print(sys.prefix + '/bin')" 2>/dev/null || true)
    if [ -n "$PYTHON_BIN_DIR" ] && echo "$PATH" | grep -q "$PYTHON_BIN_DIR" 2>/dev/null; then
        log_pass "Python bin 路径已在 PATH 中"
    else
        log_warn "Python bin 路径可能未在 PATH 中"
        log_info "build.md: export PATH=\$PATH:\$PYTHON3_HOME/bin"
    fi

    # pip
    if $PYTHON_CMD -m pip --version &>/dev/null; then
        PIP_RAW=$($PYTHON_CMD -m pip --version)
        PIP_VER=$(extract_version "$PIP_RAW")
        log_pass "pip $PIP_VER"
    else
        log_error "pip 未安装 (安装 requirements.txt 需要)"
        log_info "安装: $PYTHON_CMD -m ensurepip --upgrade"
    fi
else
    log_error "未安装 Python3 (build.md 要求 >= $REQUIRED_PYTHON_MIN)"
    log_info "安装 (Ubuntu): sudo apt-get install python3 python3-dev python3-pip"
fi

# ==================== 4. CMake ====================
# build.md: CMake >= 3.16.0 (建议使用 3.20.0 版本)
print_header "4. CMake [build.md: >= $REQUIRED_CMAKE_MIN, 建议 $REQUIRED_CMAKE_RECOMMEND]"

if check_command cmake; then
    CMAKE_RAW=$(cmake --version | head -n1)
    CMAKE_VER=$(extract_version "$CMAKE_RAW")
    log_info "cmake --version: $CMAKE_RAW → $CMAKE_VER"

    if version_ge "$CMAKE_VER" "$REQUIRED_CMAKE_RECOMMEND"; then
        log_pass "CMake $CMAKE_VER (>= 建议版本 $REQUIRED_CMAKE_RECOMMEND)"
    elif version_ge "$CMAKE_VER" "$REQUIRED_CMAKE_MIN"; then
        log_pass "CMake $CMAKE_VER (>= $REQUIRED_CMAKE_MIN)"
        log_warn "build.md 建议使用 CMake $REQUIRED_CMAKE_RECOMMEND, 当前 $CMAKE_VER"
    else
        log_error "CMake 版本过低: $CMAKE_VER (build.md 要求 >= $REQUIRED_CMAKE_MIN)"
        log_info "升级 (pip):  pip install cmake --upgrade"
        log_info "升级 (源码): https://cmake.org/download/"
    fi
else
    log_error "未安装 CMake (build.md 要求 >= $REQUIRED_CMAKE_MIN)"
    log_info "安装 (Ubuntu): sudo apt-get install cmake"
    log_info "安装 (pip):    pip install cmake"
fi

# ==================== 5. bash ====================
# build.md: bash >= 5.1.16
print_header "5. bash [build.md: >= $REQUIRED_BASH_MIN]"

if check_command bash; then
    BASH_RAW=$(bash --version | head -n1)
    BASH_VER=$(extract_version "$BASH_RAW")
    log_info "bash --version: $BASH_RAW → $BASH_VER"

    if version_ge "$BASH_VER" "$REQUIRED_BASH_MIN"; then
        log_pass "bash $BASH_VER (>= $REQUIRED_BASH_MIN)"
    else
        log_error "bash 版本过低: $BASH_VER (build.md 要求 >= $REQUIRED_BASH_MIN)"
        log_info "升级 bash 通常需要从源码编译或升级操作系统"
        log_info "源码: https://ftp.gnu.org/gnu/bash/"
    fi
else
    log_error "未安装 bash"
fi

# ==================== 6. 系统工具 (build.md 列表) ====================
# build.md: ccache/asan/autoconf/automake/libtool/gperf/lcov/libasan/patch/graph-easy
print_header "6. 系统工具 [build.md: 必需]"

# --- ccache ---
if check_command ccache; then
    CCACHE_RAW=$(ccache --version | head -n1)
    CCACHE_VER=$(extract_version "$CCACHE_RAW")
    log_pass "ccache $CCACHE_VER"
else
    log_error "未安装 ccache (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install ccache"
fi

# --- autoconf ---
if check_command autoconf; then
    AUTOCONF_RAW=$(autoconf --version | head -n1)
    AUTOCONF_VER=$(extract_version "$AUTOCONF_RAW")
    log_pass "autoconf $AUTOCONF_VER"
else
    log_error "未安装 autoconf (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install autoconf"
fi

# --- automake ---
if check_command automake; then
    AUTOMAKE_RAW=$(automake --version | head -n1)
    AUTOMAKE_VER=$(extract_version "$AUTOMAKE_RAW")
    log_pass "automake $AUTOMAKE_VER"
else
    log_error "未安装 automake (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install automake"
fi

# --- libtool ---
# Ubuntu/Debian 中 libtool 包安装后，可执行文件可能叫 libtoolize 而不是 libtool，或需要额外安装 libtool-bin
if check_command libtool || check_command libtoolize; then
    if command -v libtool &> /dev/null; then
        LIBTOOL_RAW=$(libtool --version | head -n1)
    else
        LIBTOOL_RAW=$(libtoolize --version | head -n1)
    fi
    LIBTOOL_VER=$(extract_version "$LIBTOOL_RAW")
    log_pass "libtool $LIBTOOL_VER"
else
    log_error "未安装 libtool (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install libtool"
fi

# --- gperf ---
if check_command gperf; then
    GPERF_RAW=$(gperf --version | head -n1)
    GPERF_VER=$(extract_version "$GPERF_RAW")
    log_pass "gperf $GPERF_VER"
else
    log_error "未安装 gperf (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install gperf"
fi

# --- lcov ---
if check_command lcov; then
    LCOV_RAW=$(lcov --version 2>&1 | head -n1)
    LCOV_VER=$(extract_version "$LCOV_RAW")
    log_pass "lcov $LCOV_VER"
else
    log_error "未安装 lcov (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install lcov"
fi

# --- patch ---
if check_command patch; then
    log_pass "patch 已安装"
else
    log_error "未安装 patch (build.md 要求)"
    log_info "安装 (Ubuntu): sudo apt-get install patch"
fi

# --- libasan ---
# build.md: "asan以gcc 7.5.0版本为例安装的是libasan4，其他版本请安装对应版本asan"
ASAN_FOUND=false
for path in /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 \
            /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu \
            /usr/lib/gcc/x86_64-linux-gnu/*/  /usr/lib/gcc/aarch64-linux-gnu/*/ \
            /usr/lib/gcc/x86_64-linux-gnu/*/*/  /usr/lib/gcc/aarch64-linux-gnu/*/*/; do
    if ls "$path"/libasan.so* &>/dev/null 2>&1; then
        ASAN_FOUND=true
        ASAN_PATH="$path"
        break
    fi
done

if $ASAN_FOUND; then
    log_pass "libasan 已安装 ($ASAN_PATH)"
else
    log_error "未安装 libasan (build.md 要求)"
    log_info "build.md: 'asan以gcc 7.5.0版本为例安装的是libasan4，其他版本请安装对应版本asan'"

    if check_command gcc; then
        GCC_MAJOR=$(gcc -dumpversion | cut -d'.' -f1)
        case "$GCC_MAJOR" in
            7)  log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan4" ;;
            8)  log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan5" ;;
            9)  log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan5" ;;
            10) log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan6" ;;
            11) log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan6" ;;
            12) log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan8" ;;
            13) log_info ">>> 当前 GCC $GCC_MAJOR → sudo apt-get install libasan8" ;;
            *)  log_info ">>> 当前 GCC $GCC_MAJOR → 请自行确认 libasan 版本" ;;
        esac
    fi
fi

# --- graph-easy (可选) ---
if check_command graph-easy; then
    log_pass "graph-easy 已安装"
else
    log_info "graph-easy 未安装 (build.md 标记为可选, 不影响编译)"
    log_info "安装 (Ubuntu): sudo apt-get install libgraph-easy-perl"
fi

# ==================== 7. CANN Toolkit ====================
# build.md 步骤一: 安装社区版 CANN Toolkit 包
print_header "7. CANN Toolkit [build.md: 步骤一, 必需]"

ASCEND_HOME=""
if [ -n "$ASCEND_HOME_PATH" ]; then
    ASCEND_HOME="$ASCEND_HOME_PATH"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_HOME="/usr/local/Ascend/ascend-toolkit/latest"
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_HOME="$HOME/Ascend/ascend-toolkit/latest"
elif [ -d "/usr/local/Ascend/latest" ]; then
    ASCEND_HOME="/usr/local/Ascend/latest"
fi

if [ -n "$ASCEND_HOME" ] && [ -d "$ASCEND_HOME" ]; then
    log_pass "CANN Toolkit 路径: $ASCEND_HOME"

    if [ -f "$ASCEND_HOME/version.cfg" ]; then
        CANN_VER=$(head -n1 "$ASCEND_HOME/version.cfg")
        log_info "CANN 版本: $CANN_VER"
    fi

    # 环境变量 (build.md 步骤三: source set_env.sh)
    SET_ENV_FOUND=false
    for env_script in \
        "/usr/local/Ascend/cann/set_env.sh" \
        "$HOME/Ascend/cann/set_env.sh" \
        "$ASCEND_HOME/bin/setenv.bash" \
        "$ASCEND_HOME/../set_env.sh"; do
        if [ -f "$env_script" ]; then
            SET_ENV_FOUND=true
            log_pass "set_env.sh: $env_script"
            break
        fi
    done

    if ! $SET_ENV_FOUND; then
        log_warn "未找到 set_env.sh"
        log_info "build.md: source /usr/local/Ascend/cann/set_env.sh"
    fi

    if [ -n "$ASCEND_TOOLKIT_HOME" ]; then
        log_pass "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
    else
        log_warn "ASCEND_TOOLKIT_HOME 未设置 (请先 source set_env.sh)"
    fi

    if echo "$LD_LIBRARY_PATH" | grep -q "Ascend" 2>/dev/null; then
        log_pass "LD_LIBRARY_PATH 包含 Ascend"
    else
        log_warn "LD_LIBRARY_PATH 未包含 Ascend (请先 source set_env.sh)"
    fi
else
    log_error "未检测到 CANN Toolkit"
    log_info "build.md: 安装社区版 CANN Toolkit 包"
    log_info "下载: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    log_info "安装: bash Ascend-cann-toolkit_xxx_linux-xxx.run --full --quiet"
    log_info "环境: source /usr/local/Ascend/cann/set_env.sh"
fi

# ==================== 8. git 和 make ====================
print_header "8. git / make [build.md: 步骤三需要 git clone]"

if check_command git; then
    GIT_RAW=$(git --version)
    GIT_VER=$(extract_version "$GIT_RAW")
    log_pass "git $GIT_VER"
else
    log_error "未安装 git (build.md 步骤三需要 git clone)"
    log_info "安装 (Ubuntu): sudo apt-get install git"
fi

if check_command make; then
    MAKE_RAW=$(make --version | head -n1)
    MAKE_VER=$(extract_version "$MAKE_RAW")
    log_pass "make $MAKE_VER"
else
    log_error "未安装 make"
    log_info "安装 (Ubuntu): sudo apt-get install make"
fi

# ==================== 9. Python 包 (requirements.txt) ====================
# requirements.txt:
#   numpy
#   pybind11>=2.13.6,<3.0.0
#   jinja2
#   setuptools>=59.0.1
#   wheel>=0.37.1
#   coverage          (build.md 也单独提及: pip3 install coverage)
#   cloudpickle
print_header "9. Python 包 [requirements.txt]"

if [ -n "$PYTHON_CMD" ]; then

    # 通用检查函数 (无版本要求)
    check_py_pkg() {
        local pkg_name=$1
        local import_name=$2

        if $PYTHON_CMD -c "import $import_name" &>/dev/null; then
            local ver
            ver=$($PYTHON_CMD -c "import $import_name; print(getattr($import_name, '__version__', 'ok'))" 2>/dev/null || echo "ok")
            log_pass "$pkg_name ($ver) [requirements.txt]"
        else
            log_error "缺少: $pkg_name [requirements.txt]"
            log_info "安装: pip install $pkg_name"
        fi
    }

    # 带版本范围检查的函数
    check_py_pkg_ver() {
        local pkg_name=$1
        local import_name=$2
        local min_ver=$3
        local max_ver=$4  # 可选, 为空则不检查上限

        if $PYTHON_CMD -c "import $import_name" &>/dev/null; then
            local ver
            ver=$($PYTHON_CMD -c "import $import_name; print($import_name.__version__)" 2>/dev/null || echo "0.0.0")
            local parsed_ver
            parsed_ver=$(extract_version "$ver")

            local ok=true

            if [ -n "$min_ver" ] && ! version_ge "$parsed_ver" "$min_ver"; then
                ok=false
            fi

            if [ -n "$max_ver" ] && ! version_lt "$parsed_ver" "$max_ver"; then
                ok=false
            fi

            if $ok; then
                if [ -n "$max_ver" ]; then
                    log_pass "$pkg_name $ver (>= $min_ver, < $max_ver) [requirements.txt]"
                else
                    log_pass "$pkg_name $ver (>= $min_ver) [requirements.txt]"
                fi
            else
                if [ -n "$max_ver" ]; then
                    log_error "$pkg_name $ver 不满足版本要求 (>= $min_ver, < $max_ver)"
                else
                    log_error "$pkg_name $ver 不满足版本要求 (>= $min_ver)"
                fi
                log_info "安装: pip install '$pkg_name>=$min_ver${max_ver:+,<$max_ver}'"
            fi
        else
            log_error "缺少: $pkg_name [requirements.txt]"
            log_info "安装: pip install '$pkg_name>=$min_ver${max_ver:+,<$max_ver}'"
        fi
    }

    # --- 有版本要求的包 ---
    check_py_pkg_ver "pybind11"   "pybind11"   "$REQUIRED_PYBIND11_MIN"   "3.0.0"
    check_py_pkg_ver "setuptools" "setuptools"  "$REQUIRED_SETUPTOOLS_MIN" ""
    check_py_pkg_ver "wheel"      "wheel"       "$REQUIRED_WHEEL_MIN"      ""

    # --- 无版本要求的包 ---
    check_py_pkg "numpy"       "numpy"
    check_py_pkg "jinja2"      "jinja2"
    check_py_pkg "coverage"    "coverage"      # build.md 也单独提及
    check_py_pkg "cloudpickle" "cloudpickle"

    # --- 快捷安装提示 ---
    log_info "一键安装: pip3 install -r requirements.txt"

else
    log_error "Python 未安装, 无法检查 requirements.txt 依赖"
fi

# ==================== 10. 系统资源 ====================
print_header "10. 系统资源 [可选]"

# 磁盘
DISK_AVAIL=$(df -B1 . 2>/dev/null | tail -1 | awk '{printf "%.0f", $4/1024/1024/1024}' || echo "0")
if [ "$DISK_AVAIL" -gt 30 ] 2>/dev/null; then
    log_info "可用磁盘: ${DISK_AVAIL}GB"
elif [ "$DISK_AVAIL" -gt 15 ] 2>/dev/null; then
    log_info "磁盘偏少: ${DISK_AVAIL}GB (ge 编译建议 >30GB)"
else
    log_info "磁盘不足: ${DISK_AVAIL}GB (至少 15GB)"
fi

# 内存
if [ -f /proc/meminfo ]; then
    MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_GB=$((MEM_KB / 1024 / 1024))
    if [ "$MEM_GB" -ge 16 ]; then
        log_info "内存: ${MEM_GB}GB"
    elif [ "$MEM_GB" -ge 8 ]; then
        log_info "内存偏少: ${MEM_GB}GB (ge 编译推荐 >=16GB)"
    else
        log_info "内存不足: ${MEM_GB}GB"
    fi
fi

# CPU
CPU_CORES=$(nproc 2>/dev/null || echo "unknown")
log_info "CPU 核心数: $CPU_CORES"

# ==============================================================================
#                            汇总报告
# ==============================================================================
echo ""
echo "=================================================================="
echo "  ge (Graph Engine) 环境检查完成"
echo "  依据: docs/build.md + requirements.txt"
echo "=================================================================="
echo ""

[ $ERROR_COUNT -gt 0 ]   && echo -e "  ${RED}✗ ERRORS:   $ERROR_COUNT${NC}  (必须修复才能编译)"
[ $WARNING_COUNT -gt 0 ] && echo -e "  ${YELLOW}⚠ WARNINGS: $WARNING_COUNT${NC}  (建议修复)"
echo -e "  ${GREEN}✓ PASSED:   $PASS_COUNT${NC}"
echo ""

if [ $ERROR_COUNT -gt 0 ]; then
    echo -e "  ${RED}结论: 环境不满足 ge 编译要求，请修复上述 ERROR 项。${NC}"
    echo ""
    echo "  build.md 安装命令 (Ubuntu/Debian):"
    echo "    sudo apt-get install cmake ccache bash lcov libasan4 \\"
    echo "      autoconf automake libtool gperf libgraph-easy-perl patch"
    echo ""
    echo "  完整依赖一键安装:"
    echo "    sudo apt-get update && sudo apt-get install -y \\"
    echo "      build-essential cmake ccache lcov patch gperf \\"
    echo "      autoconf automake libtool \\"
    echo "      python3 python3-dev python3-pip \\"
    echo "      git make"
    echo "    pip3 install -r requirements.txt"
    echo ""
    echo "  CANN Toolkit:"
    echo "    下载: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    echo "    安装: bash Ascend-cann-toolkit_xxx_linux-xxx.run --full --quiet"
    echo "    环境: source /usr/local/Ascend/cann/set_env.sh"
    echo ""
    echo "  详细说明: docs/build.md"
    echo ""
    exit 1
elif [ $WARNING_COUNT -gt 0 ]; then
    echo -e "  ${YELLOW}结论: 环境基本可用，建议关注上述 WARNING 项。${NC}"
    exit 0
else
    echo -e "  ${GREEN}结论: 环境检查全部通过！可以编译 ge。${NC}"
    echo -e "  ${GREEN}编译命令: bash build.sh${NC}"
    exit 0
fi
