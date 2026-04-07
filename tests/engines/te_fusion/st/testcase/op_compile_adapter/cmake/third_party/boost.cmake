set(BOOST_NAME "boost")
set(BOOST_PATH ${CMAKE_INSTALL_PREFIX}/third_party/boost_src)

# 默认配置的boost不存在则下载
if (NOT EXISTS "${ASCEND_3RD_LIB_PATH}/boost/include/boost/config.hpp")
    set(BOOST_URL "https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz")
    message(STATUS "Downloading ${BOOST_NAME} from ${BOOST_URL}")

    include(FetchContent)
    FetchContent_Declare(
            ${BOOST_NAME}
            URL ${BOOST_URL}
            URL_HASH SHA256=f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d
            DOWNLOAD_DIR ${BOOST_PATH}
            SOURCE_DIR "${BOOST_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${BOOST_NAME})
endif ()