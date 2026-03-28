# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(../cmake/intf_llt_pub.cmake)
if (ENABLE_AICPU_LLT)
    add_subdirectory(cpueng)
endif()
if (ENABLE_FE_LLT OR ENABLE_FE_BENCHMARK)
    add_subdirectory(nn_engine)
endif()
if (ENABLE_FFTS_LLT)
    add_subdirectory(ffts_engine)
endif()
if (ENABLE_DVPP_LLT)
    add_subdirectory(dvppeng)
endif()
if (ENABLE_HCCE_LLT)
    add_subdirectory(hccl_engine)
endif()
if (ENABLE_RTS_LLT)
    add_subdirectory(rts_engine)
endif()