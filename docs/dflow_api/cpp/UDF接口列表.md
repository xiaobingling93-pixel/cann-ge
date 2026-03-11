# UDF接口列表

本文档主要描述UDF（User Defined Function）模块对外提供的接口，用户可以调用这些接口进行自定义处理函数的开发，然后通过DataFlow构图在CPU上执行该处理函数。

您可以在“$\{INSTALL\_DIR\}/include/flow\_func”查看对应接口的头文件。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/cann。

## 接口分类及对应头文件

**表 1**  接口分类及对应头文件

| 接口分类 | 头文件路径 | 简介 |
| --- | --- | --- |
| AttrValue类 | attr_value.h | 用于获取用户设置的属性值。 |
| AscendString类 | ascend_string.h | 对String类型的封装。 |
| MetaContext类 | meta_context.h | 用于UDF上下文信息相关处理，如申请tensor和获取设置的属性等操作。 |
| FlowMsg类 | flow_msg.h | 用于处理flow func输入输出的相关操作。 |
| Tensor类 | flow_msg.h | 用于执行Tensor的相关操作。 |
| MetaFlowFunc类 | meta_flow_func.h | 该类在meta_flow_func.h中定义，用户继承该类进行自定义的单func处理函数的编写。 |
| MetaMultiFunc类 | meta_multi_func.h | 该类在meta_multi_func.h中定义，用户继承该类进行自定义的多func处理函数的编写。 |
| FlowFuncRegistrar类 | meta_multi_func.h | 该类在meta_multi_func.h中定义，是注册MetaMultiFunc的辅助模板类。 |
| MetaParams类 | meta_params.h | 该类在meta_params.h中定义，在FlowFunc的多func处理函数中使用该类获取共享的变量信息。 |
| MetaRunContext类 | meta_run_context.h | 用于执行FlowFunc的多func处理函数的上下文信息相关处理，如申请Tensor、设置输出、运行FlowModel等操作。 |
| OutOptions类 | out_options.h | 业务发布数据时，为了携带相关的option，提供了输出options的类。 |
| BalanceConfig类 | balance_config.h | 当需要均衡分发时，需要设置输出数据标识和权重矩阵相关配置信息，根据配置调度模块可以完成多实例之间的均衡分发。 |
| FlowBufferFactory类 | flow_msg.h | - |
| FlowMsgQueue类 | flow_msg_queue.h | 流式输入场景下（即flow func函数入参为队列时），用于flow func的输入队列，队列中的数据对象为[FlowMsg类](FlowMsg类.md)。 |
| 注册宏 | meta_flow_func.h<br>meta_multi_func.h | - |
| UDF日志接口 | flow_func_log.h | flow_func_log.h提供了日志接口，方便flowfunc开发中进行日志记录。 |
| 错误码 | flow_func_defines.h | - |

## AttrValue类

**表 2**  AttrValue类接口

| 接口名称 | 简介 |
| --- | --- |
| [AttrValue构造函数和析构函数](AttrValue构造函数和析构函数.md) | AttrValue构造函数和析构函数。 |
| [GetVal(AscendString &value)](GetVal(AscendString-value).md) | 获取string类型的属性值。 |
| [GetVal(std::vector<AscendString> &value)](GetVal(std-vector-AscendString-value).md) | 获取list string类型的属性值。 |
| [GetVal(int64_t &value)](GetVal(int64_t-value).md) | 获取int类型的属性值。 |
| [GetVal(std::vector<int64_t> &value)](GetVal(std-vector-int64_t-value).md) | 获取list int类型的属性值。 |
| [GetVal(std::vector<std::vector<int64_t >> &value)](GetVal(std-vector-std-vector-int64_t-value).md) | 获取list list int类型的属性值。 |
| [GetVal(float &value)](GetVal(float-value).md) | 获取float类型的属性值。 |
| [GetVal(std::vector<float> &value)](GetVal(std-vector-float-value).md) | 获取list float类型的属性值。 |
| [GetVal(bool &value)](GetVal(bool-value).md) | 获取bool类型的属性值。 |
| [GetVal(std::vector<bool> &value)](GetVal(std-vector-bool-value).md) | 获取list bool类型的属性值。 |
| [GetVal(TensorDataType &value)](GetVal(TensorDataType-value).md) | 获取TensorDataType类型的属性值。 |
| [GetVal(std::vector<TensorDataType> &value)](GetVal(std-vector-TensorDataType-value).md) | 获取list TensorDataType类型的属性值。 |

## AscendString类

**表 3**  AscendString类接口

| 接口名称 | 简介 |
| --- | --- |
| [AscendString构造函数和析构函数](AscendString构造函数和析构函数.md) | AscendString构造函数和析构函数。 |
| [GetString](GetString.md) | 获取字符串地址。 |
| [关系符重载](关系符重载.md) | 对于AscendString对象大小比较的使用场景（例如map数据结构的key进行排序），通过重载以下关系符实现。 |
| [GetLength](GetLength.md) | 获取字符串的长度。 |

## MetaContext类

**表 4**  MetaContext类接口

| 接口名称 | 简介 |
| --- | --- |
| [MetaContext构造函数和析构函数](MetaContext构造函数和析构函数.md) | MetaContext构造函数和析构函数。 |
| [AllocTensorMsg（MetaContext类）](AllocTensorMsg（MetaContext类）.md) | 根据shape和data type申请Tensor类型的msg。该函数供[Proc](Proc.md)调用。 |
| [AllocEmptyDataMsg（MetaContext类）](AllocEmptyDataMsg（MetaContext类）.md) | 申请空数据的MsgType类型的message。该函数供[Proc](Proc.md)调用。 |
| [SetOutput（MetaContext类,tensor）](SetOutput（MetaContext类-tensor）.md) | 设置指定index的output的tensor。该函数供[Proc](Proc.md)调用。 |
| [GetAttr（MetaContext类，获取指针）](GetAttr（MetaContext类-获取指针）.md) | 根据属性名获取AttrValue类型的指针。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetAttr（MetaContext类，获取属性值）](GetAttr（MetaContext类-获取属性值）.md) | 根据属性名获取对应的属性值。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [RunFlowModel（MetaContext类）](RunFlowModel（MetaContext类）.md) | 同步执行指定的模型。该函数供[Proc](Proc.md)调用。 |
| [GetInputNum（MetaContext类）](GetInputNum（MetaContext类）.md) | 获取Flowfunc的输入个数。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetOutputNum（MetaContext类）](GetOutputNum（MetaContext类）.md) | 获取Flowfunc的输出个数。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetWorkPath（MetaContext类）](GetWorkPath（MetaContext类）.md) | 获取Flowfunc的工作路径。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetRunningDeviceId（MetaContext类）](GetRunningDeviceId（MetaContext类）.md) | 获取正在运行的设备ID。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetUserData（MetaContext类）](GetUserData（MetaContext类）.md) | 获取用户数据。该函数供[Proc](Proc.md)调用。 |
| [AllocTensorMsgWithAlign（MetaContext类）](AllocTensorMsgWithAlign（MetaContext类）.md) | 根据shape、data type和对齐大小申请Tensor类型的FlowMsg，与[AllocTensorMsg](AllocTensorMsg（MetaContext类）.md)函数区别是AllocTensorMsg默认申请以64字节对齐，此函数可以指定对齐大小，方便性能调优。 |
| [RaiseException（MetaContext类）](RaiseException（MetaContext类）.md) | UDF主动上报异常，该异常可以被同作用域内的其他UDF捕获。 |
| [GetException（MetaContext类）](GetException（MetaContext类）.md) | UDF获取异常，如果开启了异常捕获功能，需要在UDF中Proc函数开始位置尝试捕获异常。 |

## FlowMsg类

**表 5**  FlowMsg类接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowMsg构造函数和析构函数](FlowMsg构造函数和析构函数.md) | FlowMsg构造函数和析构函数。 |
| [GetMsgType（FlowMsg类）](GetMsgType（FlowMsg类）.md) | 获取FlowMsg的消息类型。 |
| [GetTensor（FlowMsg类）](GetTensor（FlowMsg类）.md) | 获取FlowMsg中的Tensor指针。 |
| [SetRetCode（FlowMsg类）](SetRetCode（FlowMsg类）.md) | 设置FlowMsg消息中的错误码。 |
| [GetRetCode（FlowMsg类）](GetRetCode（FlowMsg类）.md) | 获取输入FlowMsg消息中的错误码。 |
| [SetStartTime（FlowMsg类）](SetStartTime（FlowMsg类）.md) | 设置FlowMsg消息头中的开始时间戳。 |
| [GetStartTime（FlowMsg类）](GetStartTime（FlowMsg类）.md) | 获取FlowMsg消息中的开始时间戳。 |
| [SetEndTime（FlowMsg类）](SetEndTime（FlowMsg类）.md) | 设置FlowMsg消息头中的结束时间戳。 |
| [GetEndTime（FlowMsg类）](GetEndTime（FlowMsg类）.md) | 获取FlowMsg消息中的结束时间戳。 |
| [SetFlowFlags（FlowMsg类）](SetFlowFlags（FlowMsg类）.md) | 设置FlowMsg消息头中的flags。 |
| [GetFlowFlags（FlowMsg类）](GetFlowFlags（FlowMsg类）.md) | 获取FlowMsg消息头中的flags。 |
| [SetRouteLabel](SetRouteLabel.md) | 设置路由的标签。 |
| [GetTransactionId（FlowMsg类）](GetTransactionId（FlowMsg类）.md) | 获取FlowMsg消息中的事务ID，事务ID从1开始计数，每feed一批数据，事务ID会加一，可用于识别哪一批数据。 |
| [GetTensorList](GetTensorList.md) | 返回FlowMsg中所有的Tensor指针列表。 |
| [GetRawData（FlowMsg类）](GetRawData（FlowMsg类）.md) | 获取RawData类型的数据对应的数据指针和数据大小。 |
| [SetMsgType（FlowMsg类）](SetMsgType（FlowMsg类）.md) | 设置FlowMsg的消息类型。 |
| [SetTransactionId（FlowMsg类）](SetTransactionId（FlowMsg类）.md) | 设置FlowMsg消息中的事务ID。 |

## Tensor类

**表 6**  Tensor类接口

| 接口名称 | 简介 |
| --- | --- |
| [Tensor构造函数和析构函数](Tensor构造函数和析构函数.md) | Tensor构造函数和析构函数。 |
| [GetShape](GetShape.md) | 获取Tensor的Shape。 |
| [GetDataType](GetDataType.md) | 获取Tensor中的数据类型。 |
| [GetData](GetData.md) | 获取Tensor中的数据。 |
| [GetDataSize](GetDataSize.md) | 获取Tensor中的数据大小。 |
| [GetElementCnt](GetElementCnt.md) | 获取Tensor中的元素的个数。 |
| [GetDataBufferSize](GetDataBufferSize.md) | 获取Tensor中的对齐后的数据大小。 |
| [Reshape](Reshape.md) | 对Tensor进行Reshape操作，不改变Tensor的内容。 |

## MetaFlowFunc类

**表 7**  MetaFlowFunc类接口

| 接口名称 | 简介 |
| --- | --- |
| [MetaFlowFunc构造函数和析构函数](MetaFlowFunc构造函数和析构函数.md) | 用户继承该类进行自定义的单func处理函数的编写。在析构函数中，执行释放相关资源操作。 |
| [SetContext](SetContext.md) | 设置flow func的上下文信息。 |
| [Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md) | 用户自定义flow func的初始化函数。 |
| [Proc](Proc.md) | 用户自定义flow func的处理函数。 |
| [RegisterFlowFunc](RegisterFlowFunc.md) | 注册flow func。<br>不建议直接使用该函数，建议使用[MetaFlowFunc注册函数宏](MetaFlowFunc注册函数宏.md)来注册flow func。 |
| [ResetFlowFuncState（MetaFlowFunc类）](ResetFlowFuncState（MetaFlowFunc类）.md) | 在故障恢复场景下，快速重置FlowFunc为初始化状态。 |
| [其他](其他.md) | REGISTER_FLOW_FUNC_INNER(name, ctr, clazz)和REGISTER_FLOW_FUNC_IMPL(name, ctr, clazz)是[MetaFlowFunc注册函数宏](MetaFlowFunc注册函数宏.md)的实现，不建议用户直接调用。 |

## MetaMultiFunc类

**表 8**  MetaMultiFunc类接口

| 接口名称 | 简介 |
| --- | --- |
| [MetaMultiFunc构造函数和析构函数](MetaMultiFunc构造函数和析构函数.md) | 用户继承该类进行自定义的多func处理函数的编写。在析构函数中，执行释放相关资源操作。 |
| [Init（MetaMultiFunc类）](Init（MetaMultiFunc类）.md) | 用户自定义flow func的初始化函数。 |
| [多func处理函数](多func处理函数.md) | 用户自定义多flow func的处理函数。 |
| [RegisterMultiFunc](RegisterMultiFunc.md) | 注册多flow func。<br>不建议直接使用该函数，建议使用[MetaMultiFunc注册函数宏](MetaMultiFunc注册函数宏.md)来注册flow func。 |
| [ResetFlowFuncState（MetaMultiFunc类）](ResetFlowFuncState（MetaMultiFunc类）.md) | 在故障恢复场景下，快速重置FlowFunc为初始化状态。 |

## FlowFuncRegistrar类

**表 9**  FlowFuncRegistrar类接口

| 接口名称 | 简介 |
| --- | --- |
| [RegProcFunc](RegProcFunc.md) | 注册多flow func处理函数，结合[MetaMultiFunc注册函数宏](MetaMultiFunc注册函数宏.md)来注册flow func。 |
| [CreateMultiFunc](CreateMultiFunc.md) | 创建多func处理对象和处理函数，框架内部使用，用户不直接使用。 |

## MetaParams类

**表 10**  MetaParams类接口

| 接口名称 | 简介 |
| --- | --- |
| [MetaParams构造函数和析构函数](MetaParams构造函数和析构函数.md) | MetaParams构造函数和析构函数。 |
| [GetName](GetName.md) | 获取Flowfunc的实例名。 |
| [GetAttr（MetaParams类，获取指针）](GetAttr（MetaParams类-获取指针）.md) | 根据属性名获取AttrValue类型的指针。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetAttr（MetaParams类，获取属性值）](GetAttr（MetaParams类-获取属性值）.md) | 根据属性名获取对应的属性值。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetInputNum（MetaParams类）](GetInputNum（MetaParams类）.md) | 获取Flowfunc的输入个数。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetOutputNum（MetaParams类）](GetOutputNum（MetaParams类）.md) | 获取Flowfunc的输出个数。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetWorkPath（MetaParams类）](GetWorkPath（MetaParams类）.md) | 获取Flowfunc的工作路径。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。 |
| [GetRunningDeviceId（MetaParams类）](GetRunningDeviceId（MetaParams类）.md) | 获取正在运行的设备ID。 |

## MetaRunContext类

**表 11**  MetaRunContext类接口

| 接口名称 | 简介 |
| --- | --- |
| [MetaRunContext构造函数和析构函数](MetaRunContext构造函数和析构函数.md) | MetaRunContext构造函数和析构函数。 |
| [AllocTensorMsg（MetaRunContext类）](AllocTensorMsg（MetaRunContext类）.md) | 根据shape和data type申请Tensor类型的msg。该函数供[Proc](Proc.md)调用。 |
| [SetOutput（MetaRunContext类,tensor）](SetOutput（MetaRunContext类-tensor）.md) | 设置指定index的output的tensor。该函数供[Proc](Proc.md)调用。 |
| [RunFlowModel（MetaRunContext类）](RunFlowModel（MetaRunContext类）.md) | 同步执行指定的模型。该函数供[Proc](Proc.md)调用。 |
| [AllocEmptyDataMsg（MetaRunContext类）](AllocEmptyDataMsg（MetaRunContext类）.md) | 申请空数据的MsgType类型的message。 |
| [GetUserData（MetaRunContext类）](GetUserData（MetaRunContext类）.md) | 获取用户数据。该函数供[Proc](Proc.md)调用。 |
| [SetOutput（MetaRunContext类,输出）](SetOutput（MetaRunContext类-输出）.md) | 设置指定index和options的输出，该函数供func函数调用。 |
| [SetMultiOutputs](SetMultiOutputs.md) | 批量设置指定index和options的输出，该函数供func函数调用。 |
| [AllocTensorMsgWithAlign（MetaRunContext类）](AllocTensorMsgWithAlign（MetaRunContext类）.md) | 根据shape、data type和对齐大小申请Tensor类型的FlowMsg，与[AllocTensorMsg](AllocTensorMsg（MetaContext类）.md)函数区别是AllocTensorMsg默认申请以64字节对齐，此函数可以指定对齐大小，方便性能调优。 |
| [AllocTensorListMsg](AllocTensorListMsg.md) | 根据输入的dtype shapes数组分配一块连续内存，用于承载Tensor数组。 |
| [RaiseException（MetaRunContext类）](RaiseException（MetaRunContext类）.md) | UDF主动上报异常，该异常可以被同作用域内的其他UDF捕获。 |
| [GetException（MetaRunContext类）](GetException（MetaRunContext类）.md) | UDF获取异常，如果开启了异常捕获功能，需要在UDF中Proc函数开始位置尝试捕获异常。 |
| [AllocRawDataMsg（MetaRunContext类）](AllocRawDataMsg（MetaRunContext类）.md) | 根据输入的size申请一块连续内存，用于承载RawData类型的数据。 |
| [ToFlowMsg](ToFlowMsg.md) | 根据输入的Tensor转换成用于承载Tensor的FlowMsg。 |

## OutOptions类

**表 12**  OutOptions类接口

| 接口名称 | 简介 |
| --- | --- |
| [OutOptions构造函数和析构函数](OutOptions构造函数和析构函数.md) | OutOptions的构造和析构函数。 |
| [MutableBalanceConfig](MutableBalanceConfig.md) | 获取或创建BalanceConfig。 |
| [GetBalanceConfig](GetBalanceConfig.md) | 获取BalanceConfig。 |

## BalanceConfig类

**表 13**  BalanceConfig类接口

| 接口名称 | 简介 |
| --- | --- |
| [BalanceConfig构造函数和析构函数](BalanceConfig构造函数和析构函数.md) | BalanceConfig的构造和析构函数。 |
| [SetAffinityPolicy](SetAffinityPolicy.md) | 设置均衡分发亲和性。 |
| [GetAffinityPolicy](GetAffinityPolicy.md) | 获取亲和性。 |
| [SetBalanceWeight](SetBalanceWeight.md) | 设置均衡分发权重信息。 |
| [GetBalanceWeight](GetBalanceWeight.md) | 获取均衡分发权重信息。 |
| [SetDataPos](SetDataPos.md) | 设置输出数据对应权重矩阵中的位置。 |
| [GetDataPos](GetDataPos.md) | 获取输出数据对应权重矩阵中的位置。 |

## FlowBufferFactory类

**表 14**  FlowBufferFactory类接口

| 接口名称 | 简介 |
| --- | --- |
| [AllocTensor（FlowBufferFactory类）](AllocTensor（FlowBufferFactory类）.md) | 根据shape、data type和对齐大小申请Tensor，默认申请以64字节对齐，可以指定对齐大小，方便性能调优。 |

## FlowMsgQueue类

**表 15**  FlowMsgQueue类接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowMsgQueue构造函数和析构函数](FlowMsgQueue构造函数和析构函数.md) | FlowMsgQueue的构造和析构函数。 |
| [Dequeue](Dequeue.md) | 设置均衡分发亲和性。 |
| [Depth](Depth.md) | 获取队列的深度，即获取队列可容纳元素的最大个数。 |
| [Size](Size.md) | 获取队列中当前元素的个数。 |

## 注册宏

**表 16**  注册宏

| 接口名称 | 简介 |
| --- | --- |
| [MetaFlowFunc注册函数宏](MetaFlowFunc注册函数宏.md) | 注册MetaFlowFunc的实现类。 |
| [MetaMultiFunc注册函数宏](MetaMultiFunc注册函数宏.md) | 注册MetaMultiFunc的实现类。 |

## UDF日志接口

**表 17**  UDF日志接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowFuncLogger构造函数和析构函数](FlowFuncLogger构造函数和析构函数.md) | FlowFuncLogger构造函数和析构函数。 |
| [GetLogger](GetLogger.md) | 获取日志实现类。 |
| [GetLogExtHeader](GetLogExtHeader.md) | 获取日志扩展头信息。 |
| [IsLogEnable](IsLogEnable.md) | 查询对应级别和类型的日志是否开启。 |
| [Error](Error.md) | 记录ERROR级别日志。 |
| [Warn](Warn.md) | 记录Warn级别日志。 |
| [Info](Info.md) | 记录Info级别日志。 |
| [Debug](Debug.md) | 记录Debug级别日志。 |
| [运行日志Error级别日志宏](运行日志Error级别日志宏.md) | 运行日志Error级别日志宏。 |
| [运行日志Info级别日志宏](运行日志Info级别日志宏.md) | 运行日志Info级别日志宏。 |
| [调试日志Error级别日志宏](调试日志Error级别日志宏.md) | 调试日志Error级别日志宏。 |
| [调试日志Warn级别日志宏](调试日志Warn级别日志宏.md) | 调试日志Warn级别日志宏。 |
| [调试日志Info级别日志宏](调试日志Info级别日志宏.md) | 调试日志Info级别日志宏。 |
| [调试日志Debug级别日志宏](调试日志Debug级别日志宏.md) | 调试日志Debug级别日志宏。 |

## 错误码

**表 18**  错误码

| 错误码模块 | 简介 |
| --- | --- |
| [flowfunc](UDF错误码.md#section1390959132616) | 提供了flowfunc的错误码供用户使用，主要用于对异常逻辑的判断处理。 |
| [AICPU](UDF错误码.md#section119131377263) | AICPU在执行模型的过程中，有可能向用户上报的错误码。 |
