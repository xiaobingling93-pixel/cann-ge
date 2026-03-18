add_library(compress_headers INTERFACE)
target_include_directories(compress_headers INTERFACE
    $<BUILD_INTERFACE:${TOP_DIR}/metadef/inc/common/util/compress>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/compress>
    $<INSTALL_INTERFACE:include/compress/common>
    $<INSTALL_INTERFACE:include/compress/common/util>
    $<INSTALL_INTERFACE:include/compress/common/util/compress>
)

file(GLOB FE_SRC_CPP_1 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "compress.cpp"
        "data_compressor.cpp"
        "log.cpp"
        "mode_a_index_generator.cpp"
        "mode_b_index_generator.cpp")

# compile for libcompress.so
add_library(compress SHARED
        ${FE_SRC_CPP_1})

target_include_directories(compress PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/inc
        ${ASCEND_INSTALL_PATH}/include)

target_compile_options(compress PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
        $<$<STREQUAL:${PRODUCT_SIDE},host>:-fexceptions>
        -fno-common
        -fno-strict-aliasing
        )

target_link_libraries(compress
        PRIVATE
                intf_pub
                -Wl,--no-as-needed
                c_sec
                -Wl,--as-needed
        PUBLIC
                compress_headers
        )

# compile for libcompress.a
file(GLOB FE_SRC_CPP_3 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "compress.cpp"
        "data_compressor.cpp"
        "log.cpp"
        "mode_a_index_generator.cpp"
        "mode_b_index_generator.cpp"
        "compress_weight.cpp")

add_library(compress_static STATIC
        ${FE_SRC_CPP_3})

set_target_properties(compress_static
        PROPERTIES
        OUTPUT_NAME compress)

target_include_directories(compress_static PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/inc
        ${ASCEND_INSTALL_PATH}/include)

target_compile_options(compress_static PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
        $<$<STREQUAL:${PRODUCT_SIDE},host>:-fexceptions>
        -fno-common
        -fno-strict-aliasing
        )

target_link_libraries(compress_static PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--no-as-needed
        c_sec
        -Wl,--as-needed
        )

file(GLOB FE_SRC_CPP_2 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "compress_weight.cpp"
        "log.cpp")

# compile for libcompressweight.so
add_library(compressweight SHARED
        ${FE_SRC_CPP_2})

target_include_directories(compressweight PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/inc
        ${ASCEND_INSTALL_PATH}/include)

target_compile_options(compressweight PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
        $<$<STREQUAL:${PRODUCT_SIDE},host>:-fexceptions>
        -fno-common
        -fno-strict-aliasing
        )

target_link_libraries(compressweight PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--no-as-needed
        c_sec
        compress
        -Wl,--as-needed
        )

# compile for libcompressweight.a
add_library(compressweight_static STATIC
        ${FE_SRC_CPP_2})

set_target_properties(compressweight_static
        PROPERTIES
        OUTPUT_NAME compressweight)

target_include_directories(compressweight_static PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/inc
        ${ASCEND_INSTALL_PATH}/include)

target_compile_options(compressweight_static PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
        $<$<STREQUAL:${PRODUCT_SIDE},host>:-fexceptions>
        -fno-common
        -fno-strict-aliasing
        )

target_link_libraries(compressweight_static PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--no-as-needed
        c_sec
        compress
        -Wl,--as-needed
        )

install(TARGETS compress compress_static compressweight compressweight_static
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)
