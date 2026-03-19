# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(TE_FUSION_SRC_CPP
        src/te_fusion_api.cc
        src/tbe_op_func_manager.cc)

# compile for fe_st
add_library(te_fusion_stub SHARED
        ${TE_FUSION_SRC_CPP})

macro(get_python_path)
    if (DEFINED CUSTOM_PYTHON)
        set(HI_PYTHON ${CUSTOM_PYTHON})
    else()
        set(HI_PYTHON python3)
    endif()

    #检查python版本是否存在
    execute_process(COMMAND ${HI_PYTHON} -c "import sys; print(sys.executable)"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE _TEMP_PYTHON_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (result)
        set(Python3_FIND_STRATEGY VERSION)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(HI_PYTHON ${Python3_EXECUTABLE})
    endif()

    if (DEFINED CUSTOM_PYTHON_PATH)
        set(HI_PYTHON_PATH ${CUSTOM_PYTHON_PATH})
    else()
        execute_process(COMMAND ${HI_PYTHON} -c "import sys; print(sys.executable)"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE _TEMP_PYTHON_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (result)
            message(FATAL_ERROR "Please install python3 software first.")
        endif()
    endif()

    if (DEFINED CUSTOM_PYTHON_INC)
        set(HI_PYTHON_INC ${CUSTOM_PYTHON_INC} CACHE PATH "python Compiler inc")
    else()
        execute_process(COMMAND ${HI_PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE HI_PYTHON_INC
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (result)
            set(HI_PYTHON_INC unknown)
        endif()
    endif()
endmacro(get_python_path)
get_python_path()
message(STATUS "HI_PYTHON = ${HI_PYTHON}")
message(STATUS "HI_PYTHON_INC = ${HI_PYTHON_INC}")

target_include_directories(te_fusion_stub PRIVATE
        ${HI_PYTHON_INC}
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends
        ${AIR_CODE_DIR}/compiler/opcompiler/inc
        )

# add it for slog compile at 12.16
target_compile_definitions(te_fusion_stub PRIVATE
        LOG_CPP
        )

target_compile_options(te_fusion_stub PRIVATE
        -Dgoogle=ascend_private)

target_link_libraries(te_fusion_stub
    PRIVATE
        intf_llt_pub
        msprof_headers
        runtime_headers
        metadef_headers
        c_sec
        json
        graph)

install(TARGETS te_fusion_stub OPTIONAL
        EXPORT te_fusion_stub-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})
