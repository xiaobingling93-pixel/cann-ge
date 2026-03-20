cmake_minimum_required(VERSION 3.5.1)
project(move_relu_before_concat_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/cann) # 可根据实际修改
endif ()

set(PASS_SO_DIR pass_so_dir) # 可根据实际修改
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
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

add_subdirectory(gen_es_api)

############ move_relu_before_concat_pass.so ############
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
        ${INCLUDE_DIR}/external
        ${CMAKE_CURRENT_BINARY_DIR}/es_output/include/es_all
)

target_link_directories(${PROJECT_NAME} PRIVATE
        ${ASCEND_PATH}/lib64/stub
        ${ASCEND_PATH}/lib64/ #todo delete  eager_style_graph_builder_base change to stub
)
target_link_options(${PROJECT_NAME} PRIVATE
)
target_link_libraries(${PROJECT_NAME} PRIVATE
        -Wl,--no-as-needed
        graph
        register
        ge_compiler
        es_all
        -Wl,--as-needed
)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR}/es_output/lib64")

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${ASCEND_PATH}/opp/vendors/${PASS_SO_DIR}/custom_fusion_passes
)
install(CODE "message(STATUS \"so file of ${PROJECT_NAME} is ready！\")")