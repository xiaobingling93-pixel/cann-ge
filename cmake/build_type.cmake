# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (CMAKE_BUILD_TYPE MATCHES GCOV)
    message("GCOV test mode")
    set(AIR_COMMON_COMPILE_OPTION
            -O0
            -g
            --coverage -fprofile-arcs -ftest-coverage
            -fsanitize=address -fsanitize=leak -fsanitize-recover=address
            )
    set(AIR_COV_COMPILE_OPTION
            --coverage -fprofile-arcs -ftest-coverage
            )
    set(AIR_COMMON_DYNAMIC_COMPILE_OPTION ${AIR_COMMON_COMPILE_OPTION})
    if (TARGET_SYSTEM_NAME STREQUAL "Android")
        set(AIR_COMMON_LINK_OPTION
                -fsanitize=address -fsanitize=leak -fsanitize-recover=address
                -ldl -lgcov
                )
    else ()
        set(AIR_COMMON_LINK_OPTION
                -fsanitize=address -fsanitize=leak -fsanitize-recover=address
                -lrt -ldl -lgcov
                )
    endif ()
elseif(CMAKE_BUILD_TYPE MATCHES DT)
    message("Dump graph test mode")
    set(AIR_COMMON_COMPILE_OPTION -O0 -g)
    set(AIR_COV_COMPILE_OPTION ${AIR_COMMON_COMPILE_OPTION})
else ()
    if (TARGET_SYSTEM_NAME STREQUAL "Windows")
        if (CMAKE_CONFIGURATION_TYPES STREQUAL "Debug")
            set(AIR_COMMON_COMPILE_OPTION /MTd)
        else ()
            set(AIR_COMMON_COMPILE_OPTION /MT)
        endif ()

    else ()
        set(AIR_COMMON_COMPILE_OPTION -fvisibility=hidden -O2 -Werror -fno-common -Wextra -Wfloat-equal)
        set(AIR_COMMON_DYNAMIC_COMPILE_OPTION -fvisibility=default -O2 -Werror -fno-common -Wextra -Wfloat-equal)
    endif ()

endif (CMAKE_BUILD_TYPE MATCHES GCOV)
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("common compile options ${AIR_COMMON_COMPILE_OPTION}")
message("common link options ${AIR_COMMON_LINK_OPTION}")
