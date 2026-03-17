cmake_minimum_required(VERSION 3.5.1)
project(fuse_matmul_add_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/cann) # 可根据实际修改
endif()

set(INCLUDE_DIR ${ASCEND_PATH}/include)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
    -std=c++17
    -g
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

############ libfuse_matmul_add_pass.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES output_name ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${INCLUDE_DIR}/graph
    ${INCLUDE_DIR}/ge
    ${INCLUDE_DIR}/register
    ${INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/lib64/stub
    ${ASCEND_PATH}/lib64/
)

target_link_options(${PROJECT_NAME} PRIVATE
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    ge_compiler
    -Wl,--as-needed
)
set(FUSION_PASS_DIR fusion_passes) # 可根据实际修改

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${ASCEND_PATH}/opp/vendors/${FUSION_PASS_DIR}/custom_fusion_passes
)
install(CODE "message(STATUS \"so file of ${PROJECT_NAME} is ready！\")")