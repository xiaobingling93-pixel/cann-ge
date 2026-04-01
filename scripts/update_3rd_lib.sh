#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODULE_NAME=""
INSTALL_PATH=""
CLEAN_BUILD=false

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1"
    exit 1
}

usage() {
    echo "Usage: $0 --module=<module_name> [--path=<install_path>] [--clean]"
    echo ""
    echo "Build and install third-party library module."
    echo ""
    echo "Options:"
    echo "  --module=<module_name>   Module name (zlib, etc.)"
    echo "  --path=<install_path>    Installation base path"
    echo "  --clean                  Clean build directory before building"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --module=zlib"
    echo "  $0 --module=zlib --path=/opt"
    echo "  $0 --module=zlib --clean"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --module=*)
                MODULE_NAME="${1#*=}"
                shift
                ;;
            --path=*)
                INSTALL_PATH="${1#*=}"
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                ;;
        esac
    done
}

validate_args() {
    if [ -z "${MODULE_NAME}" ]; then
        print_error "Module name is required. Use --module=<module_name>"
    fi
}

setup_environment() {
    print_info "Setting up environment..."

    BUILD_DIR="${BUILD_DIR:-build}"
    CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

    # Convert BUILD_DIR to absolute path
    if [[ ! "${BUILD_DIR}" = /* ]]; then
        BUILD_DIR="${PROJECT_DIR}/${BUILD_DIR}"
    fi

    if [ -n "${ASCEND_3RD_LIB_PATH}" ]; then
        THIRD_PARTY_BASE="${ASCEND_3RD_LIB_PATH}"
    elif [ -n "${CMAKE_THIRD_PARTY_LIB_DIR}" ]; then
        THIRD_PARTY_BASE="${CMAKE_THIRD_PARTY_LIB_DIR}"
    else
        THIRD_PARTY_BASE="${PROJECT_DIR}/output/third_party"
    fi

    # Set install directory name (can be overridden in build_module)
    INSTALL_DIR="${MODULE_NAME}"

    if [ -z "${INSTALL_PATH}" ]; then
        INSTALL_PATH="${THIRD_PARTY_BASE}/${INSTALL_DIR}"
    else
        INSTALL_PATH="${INSTALL_PATH}/${INSTALL_DIR}"
    fi

    # Convert to absolute path (create directory first if needed)
    mkdir -p "${INSTALL_PATH}"
    INSTALL_PATH="$(cd "${INSTALL_PATH}" 2>/dev/null && pwd)" || print_error "Invalid path: ${INSTALL_PATH}"

    print_info "Module: ${MODULE_NAME}"
    print_info "Installation path: ${INSTALL_PATH}"
    print_info "Build directory: ${BUILD_DIR}"
    print_info "Build type: ${CMAKE_BUILD_TYPE}"
}

clean_build() {
    if [ "${CLEAN_BUILD}" = true ]; then
        print_info "Cleaning build directory..."
        rm -rf "${BUILD_DIR}/${MODULE_NAME}_build"
    fi
}

create_build_dir() {
    print_info "Creating build directory..."
    mkdir -p "${BUILD_DIR}"
}

download_source() {
    local module=$1
    local url=$2
    local download_dir="${THIRD_PARTY_BASE}/${module}"

    print_info "Checking source for ${module}..."

    if [ -f "${download_dir}/$(basename ${url})" ]; then
        print_info "Source already exists: ${download_dir}/$(basename ${url})"
        return 0
    fi

    mkdir -p "${download_dir}"
    print_info "Downloading ${module} from ${url}..."
    if command -v wget &> /dev/null; then
        wget -O "${download_dir}/$(basename ${url})" "${url}"
    elif command -v curl &> /dev/null; then
        curl -L -o "${download_dir}/$(basename ${url})" "${url}"
    else
        print_error "Neither wget nor curl is available"
    fi
}

build_module() {
    print_info "Building ${MODULE_NAME}..."

    # Define module configurations
    case "${MODULE_NAME}" in
        zlib)
            SOURCE_URL="https://gitcode.com/cann-src-third-party/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.gz"
            PATCH_FILE="${PROJECT_DIR}/cmake/third_party/build/modules/patch/zlib_add_minizip_static_lib.patch"
            # Override INSTALL_PATH if needed (uncomment to change subdirectory name)
            CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                       "-DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}"
                       "-DCMAKE_C_FLAGS=-fPIC -fexceptions -O2"
                       "-DUNZ_MAXFILENAMEINZIP=4096")
            ;;
        *)
            print_error "Unsupported module: ${MODULE_NAME}"
            ;;
    esac

    # Download source if not exists
    download_source "${MODULE_NAME}" "${SOURCE_URL}"
    SOURCE_FILE="${THIRD_PARTY_BASE}/${MODULE_NAME}/$(basename ${SOURCE_URL})"

    # Extract source
    EXTRACT_DIR="${BUILD_DIR}/${MODULE_NAME}_src"
    print_info "Extracting ${MODULE_NAME} source..."
    # Clean extract directory to avoid old cmake cache
    rm -rf "${EXTRACT_DIR}"
    mkdir -p "${EXTRACT_DIR}"
    tar -xf "${SOURCE_FILE}" -C "${EXTRACT_DIR}" --strip-components=1

    # Apply patch if exists
    if [ -n "${PATCH_FILE}" ] && [ -f "${PATCH_FILE}" ]; then
        print_info "Applying patch for ${MODULE_NAME}..."
        cd "${EXTRACT_DIR}"
        patch -p1 < "${PATCH_FILE}"
    fi

    # Configure
    print_info "Configuring ${MODULE_NAME}..."
    cd "${EXTRACT_DIR}"
    cmake "${CMAKE_ARGS[@]}"

    # Build
    print_info "Building ${MODULE_NAME}..."
    cmake --build . --parallel $(nproc)

    # Install
    print_info "Installing ${MODULE_NAME}..."
    cmake --install .

    cd "${SCRIPT_DIR}"
}

verify_installation() {
    print_info "Verifying installation..."

    if [ -d "${INSTALL_PATH}" ]; then
        print_info "SUCCESS: ${MODULE_NAME} built successfully"
        print_info "Installation path: ${INSTALL_PATH}"
        ls "${INSTALL_PATH}" | head -10
    else
        print_error "FAILED: Installation path not found: ${INSTALL_PATH}"
    fi
}

main() {
    parse_args "$@"
    validate_args
    setup_environment
    clean_build
    create_build_dir
    build_module
    verify_installation

    print_info "=== ${MODULE_NAME} build completed successfully ==="
}

main "$@"
