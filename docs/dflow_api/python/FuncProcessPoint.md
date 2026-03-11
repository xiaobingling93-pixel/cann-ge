# FuncProcessPoint

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

FuncProcessPoint的构造函数，返回一个FuncProcessPoint对象。

## 函数原型

```
class FuncProcessPoint(compile_config_path: Optional[str] = None, name: Optional[str] = None,
py_func: Optional = None, workspace_dir: Optional = None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| compile_config_path | str | UDF的编译配置文件，与py_func互斥。传入py_func参数后compile_config_path不生效。 |
| name | str | 处理点名称，框架会自动保证名称唯一，不设置时会自动生成FuncProcessPoint, FuncProcessPoint_1, FuncProcessPoint_2,...的名称。 |
| py_func | class | 用户开发的自定义Python UDF类（需要使用proc_wrapper(func_list="ix:ox")注册输入输出），与compile_config_path互斥。传入py_func参数后compile_config_path不生效。 |
| workspace_dir | str | 自动生成的Python UDF临时工作空间目录，和py_func配合使用，<br>该字段为空时使用py_func的名字拼接“_ws”使用。 |

UDF的编译配置文件示例如下：

    ```
    {"func_list":[{"func_name":"Add", "inputs_index":[1,0], "outputs_index":[0]}],"input_num":2,"output_num":1,"target_bin":"libadd.so","workspace":"./","cmakelist_path":"CMakeLists.txt","compiler": "./cpu_compile.json","running_resources_info":[{"type":"cpu","num":2},{"type":"memory","num":100}],"heavy_load":false}
    ```

**表 1**  FuncProcessPoint的json配置文件

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| workspace | 必选 | 值为字符串，UDF的工作空间路径。 |
| target_bin | 必选 | 值为字符串，UDF工程编译出来的so名字，为防止被非法篡改，该字符串需要以lib***.so来命名，合法的字符包含大小写字母、数字、下划线和中划线。 |
| input_num | 必选 | 值为数字，表示UDF的输入个数，即FuncProcessPoint的输入个数。 |
| output_num | 必选 | 值为数字，表示UDF的输出个数。即FuncProcessPoint的输出个数。 |
| func_list | 必选 | 值为list，list的元素为单个function的描述，当前只支持一个function。 |
| func_list.func_name | 必选 | 值为字符串，函数名称，要和UDF里定义的function名称一致。多function场景下，func_name不允许重复。 |
| func_list.inputs_index | 可选 | 值为list，list元素为数字，表示该function取FuncProcessPoint的哪些输入，单function情况下当前无效;多function情况下该字段必选。且多个处理函数input index不共享，不能重复。 |
| func_list.outputs_index | 可选 | 值为list，list元素为数字，表示该function对应FuncProcessPoint的哪些输出，单function情况下当前无效。多function情况下output index可共享。 |
| cmakelist_path | 可选 | 值为字符串，源码编译的CMakeLists文件相对于workspace的路径，如果未指定，则取workspace下面的默认CMakeLists文件。 |
| compiler | 可选 | 值为字符串，异构环境下编译源码的交叉编译工具路径配置文件，如果未指定，则取资源类型默认的编译工具。|
| running_resources_info | 可选 | 值为list，运行当前so需要的资源信息，list的元素为单个资源信息的描述。 |
| running_resources_info.type | 可选 | 当配置了running_resources_info时，该字段必选。<br>值为字符串，运行当前so需要的资源信息的类型，可选类型是"cpu"和"memory"。当资源类型是"memory"时，单位是M。 |
| running_resources_info.num | 可选 | 当配置了running_resources_info时，该字段必选。<br>值为数字，运行当前so需要的资源信息的数量。 |
| heavy_load | 可选 | 表示节点对算力的诉求。<br><br>  - true：重载，表示对算力的诉求大。<br>  - false：轻载，表示对算力的诉求小。<br><br>默认值为false。<br>当该参数取值为"true"时会影响UDF的部署位置。 |
| buf_cfg | 可选 | 用户可以自定义配置内存池档位，通过自定义档位可以提升内存申请效率及减少内存碎片，如未设置该参数，将使用默认的档位配置初始化内存模块，该配置最多支持64个档位，超过64编译报错。 |

compiler的json配置内容示例和各字段解释如下。

    ```
    {"compiler":[{"resource_type":"X86","toolchain":"/usr/bin/g++"},{"resource_type":"Aarch","toolchain":"/usr/bin/g++"},{"resource_type":"Ascend","toolchain":"/usr/local/Ascend/hcc"}]}
    ```

**表 2**  compiler的json配置文件

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| compiler | 必选 | 值为list，list的元素为单个资源类型的编译工具的描述。 |
| compiler.resource_type | 必选 | 值为字符串，设备支持的资源类型。 |
| compiler.toolchain | 必选 | 值为字符串，该资源类型对应的编译工具路径。 |

buf\_cfg的json配置内容示例和各字段解释如下。

    ```
    "buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192,"page_type":"normal"},        // 1.total:2M  max:8K
               {"total_size":10485760,"blk_size":4096,"max_buf_size":8388608,"page_type":"normal"},   // 2.total:10M  max:8M
               {"total_size":2097152,"blk_size":256,"max_buf_size":8192,"page_type":"huge"},          // 3.total:2M  max:8K     
               {"total_size":10485760,"blk_size":8192,"max_buf_size":8388608,"page_type":"huge"},     // 4.total:10M  max:8M
               {"total_size":69206016,"blk_size":8192,"max_buf_size":67108864,"page_type":"huge"}]    // 5.total:66M  max:64M
      ```             
  
说明：
<br>- 如上样例共配置了5个内存档位，前两条针对普通内存，后三条针对大页内存。
<br>- 使用该配置初始化内存管理模块后，如果进程申请8M大页内存，驱动会根据第4条配置项，生成并管理一个10M内存池，从其中申请8M内存。
<br>- 如本进程需要再次申请1M大页内存，由于第三条配置项中一次最大只能申请8K，因此仍然会落到第4条配置项对应的内存池中，此时上一次申请10M只使用了8M，剩余的内存仍大于1M，因此会在上一次生成的10M内存池中申请1M内存供本次使用。

<br>默认档位如下：

| ID | total_size | blk_size | max_buf_size | page_type |
| --- | --- | --- | --- | --- |
| 0 | 2M | 256B | 8K | normal |
| 1 | 32M | 8K | 8M | normal |
| 3 | 2M | 256B | 8K | huge |
| 4 | 66M | 8K | 64M | huge |

**表 3**  buf\_cfg的json配置说明

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| total_size | 必选 | 当前档位内存池的大小，单位Byte<br>约束：<br>普通内存total_size是4K的倍数，大页内存total_size是2M的倍数，且total_size是blk_size的倍数 |
| blk_size | 必选 | 当前档位一次可以申请的最小内存值，单位Byte<br>约束：<br>要求满足2^n，且在(0,2M]之间，小于max_buf_size |
| max_buf_size | 必选 | 当前档位一次可以申请的最大内存值，单位Byte<br>约束：小于total_size |
| page_type | 必选 | 当前档位对应的内存类型，<br>约束：<br>有效值huge或normal，分别表示大页内存和普通内存 |

CMakeLists文件相关内容如下：
<br>DataFlow UDF编译模块解析异构环境的resource.json资源配置和cpu\_compiler配置，根据resource.json资源配置类型匹配选择cpu\_compiler中指定的交叉编译工具，如果用户未指定cpu\_compiler.json配置文件或者cpu\_compiler.json未配置该类型的编译工具，则取环境上默认的编译工具进行编译。不同资源类型的编译工具名称和路径如下。$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/cann。

  - X86和Aarch场景下：g++
  - Ascend场景下：$\{INSTALL\_DIR\}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++

    用户的源代码工程要遵从如下规则：

  - 用户提供的源码工程目录下要包括所有执行代码和依赖库源码。
  - 用户要配置好FuncProcessPoint执行代码和依赖库的编译脚本。
  - 编译脚本要使用RELEASE\_DIR变量作为最终输出目录，如果有依赖的so文件，用户需要把依赖的so文件拷贝到该路径下。
  - 编译脚本要使用RESOURCE\_TYPE变量判断资源类型，如果当前UDF不支持某一个资源类型，需要将对应的注释放开。
  - CMakeLists sample如下：

        ```
        cmake_minimum_required(VERSION 3.5)
        PROJECT(UDF)
        if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
          message(STATUS "ascend compiler enter")
          # if unsupport current resource type, please uncomment the next line.
          # message(FATAL_ERROR "Unsupport compile Ascend target!")
        elseif("x${RESOURCE_TYPE}" STREQUAL "xAarch")
          message(STATUS "aarch64 compiler enter")
          # if unsupport current resource type, please uncomment the next line.
          # message(FATAL_ERROR "Unsupport compile Aarch64 target!")
        else()
          message(STATUS "x86 compiler enter")
          # if unsupport current resource type, please uncomment the next line.
          # message(FATAL_ERROR "Unsupport compile X86 target!")
        endif()
        
        if(DEFINED ENV{ASCEND_HOME_PATH})
          set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
          message(STATUS "Read ASCEND_HOME_PATH from ENV: ${ASCEND_HOME_PATH}")
        else()
          
          message(FATAL_ERROR "ASCEND_HOME_PATH is not set, please export ASCEND_HOME_PATH based on actual installation path.")
        endif()
        
        # set dynamic library output path
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
        # set static library output path
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
        
        message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY= ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        
        set(INC_DIR "${ASCEND_HOME_PATH}/include")flow_func")
        file(GLOB SRC_LIST "*.cpp")
        
        # Specify cross compiler
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        
        # set c++ compiler
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN})
        
        # =========================UDF so compile============================
        # check if SRC_LIST is exist
        if("x${SRC_LIST}" STREQUAL "x")
            message(UDF "=========no source file=============")
            add_custom_target(${UDF_TARGET_LIB}
                COMMAND echo "no source to make lib${UDF_TARGET_LIB}.so")
            return(0)
        endif()
        
        add_library(${UDF_TARGET_LIB} SHARED
          ${SRC_LIST}
        )
        
        target_include_directories(${UDF_TARGET_LIB} PRIVATE
          ${INC_DIR}
        )
        
        target_compile_options(${UDF_TARGET_LIB} PRIVATE
          -O2
          -std=c++11
          -ftrapv  
          -fstack-protector-all
          -fPIC
        )
        
        if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
          target_link_libraries(${UDF_TARGET_LIB} PRIVATE 
            -Wl,--whole-archive
            ${ASCEND_HOME_PATH}/devlib/device/libflow_func.so
            -Wl,--no-whole-archive
          )
          # If there have any dependent so, please release the following comments and copy dependent so to ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          # [[execute_process(
            COMMAND cp libdepend_xxx.so ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          )]]
        elseif("x${RESOURCE_TYPE}" STREQUAL "xAarch")
          target_link_libraries(${UDF_TARGET_LIB} PRIVATE 
            -Wl,--whole-archive
           ${ASCEND_HOME_PATH}/devlib/linux/aarch64/libflow_func.so
            -Wl,--no-whole-archive
          )
          # If there have any dependent so, please release the following comments and copy dependent so to ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          # [[execute_process(
            COMMAND cp libdepend_xxx.so ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          )]]
        else()
          target_link_libraries(${UDF_TARGET_LIB} PRIVATE 
            -Wl,--whole-archive
            ${ASCEND_HOME_PATH}/devlib/linux/x86_64/libflow_func.so
            -Wl,--no-whole-archive
          )
          # If there have any dependent so, please release the following comments and copy dependent so to ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          # [[execute_process(
             COMMAND cp libdepend_xxx.so ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          )]]
        endif()
        
        ```

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
```

## 约束说明

无
