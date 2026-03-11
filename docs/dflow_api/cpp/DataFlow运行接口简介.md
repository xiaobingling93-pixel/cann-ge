# DataFlow运行接口简介

本文档主要描述模型执行接口，您可以在“$\{INSTALL\_DIR\}/include/ge”路径下查看对应接口的头文件。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/cann。

| 接口分类 | 头文件路径 | 用途 | 对应的库文件 |
| --- | --- | --- | --- |
| Graph运行接口 | ge_api.h | 用于将数据输入到DataFlow图和获取DataFlow模型执行结果。 | libge_runner.so  libdavinci_executor.so  libgraph_base.so |
| 数据类型 | ge_data_flow_api.h | 支持用户设置和获取DataFlowInfo中的成员变量。<br> 说明： 如果单点编译DataFlowInfo数据类型，建议编译选项增加-Wl,--no-as-needed，确保依赖的so符号在编译时被完整加载。 | libdavinci_executor.so libge_runner.so |
