# UDF接口列表

本文档主要描述UDF（User Defined Function）模块对外提供的接口，用户可以调用这些接口进行自定义处理函数的开发，然后通过DataFlow构图在CPU上执行该处理函数。

您可以在CANN软件安装后文件存储路径下的“python/site-packages/dataflow/flow\_func/flow\_func.py”查看对应接口的实现。接口列表如下。

## FlowMsg类

用于处理FlowFunc输入输出的相关操作。

**表 1**  FlowMsg类接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowMsg构造函数](FlowMsg构造函数.md) | FlowMsg的构造函数。 |
| [get_msg_type（UDF）](get_msg_type（UDF）.md) | 获取FlowMsg的消息类型。 |
| [get_tensor](get_tensor.md) | 获取FlowMsg中的tensor对象。 |
| [set_ret_code](set_ret_code.md) | 设置FlowMsg消息中的错误码。 |
| [get_ret_code](get_ret_code.md) | 获取输入FlowMsg消息中的错误码。 |
| [set_start_time](set_start_time.md) | 设置FlowMsg消息头中的开始时间戳。 |
| [get_start_time](get_start_time.md) | 获取FlowMsg消息中的开始时间戳。 |
| [set_end_time](set_end_time.md) | 设置FlowMsg消息头中的结束时间戳。 |
| [get_end_time](get_end_time.md) | 获取FlowMsg消息中的结束时间戳。 |
| [set_flow_flags](set_flow_flags.md) | 设置FlowMsg消息头中的flags。 |
| [get_flow_flags](get_flow_flags.md) | 获取FlowMsg消息头中的flags。 |
| [set_route_label](set_route_label.md) | 设置路由的标签。 |
| [get_transaction_id](get_transaction_id.md) | 获取FlowMsg消息中的事务ID，事务Id从1开始计数，每feed一批数据，事务Id会加一，可用于识别哪一批数据。 |
| [set_msg_type](set_msg_type.md) | 设置FlowMsg的消息类型。 |
| [get_raw_data](get_raw_data.md) | 获取rawdata类型的数据。 |
| [set_transaction_id](set_transaction_id.md) | 设置DataFlow数据传输使用的事务ID。 |

## Tensor类

用于执行Tensor的相关操作。这里获取的Tensor是dataflow.Tensor。

**表 2**  Tensor类接口

| 接口名称 | 简介 |
| --- | --- |
| [Tensor构造函数](Tensor构造函数.md) | Tensor构造函数。 |
| [get_shape](get_shape.md) | 获取Tensor的Shape。 |
| [get_data_type](get_data_type.md) | 获取Tensor中的数据类型。 |
| [get_data_size](get_data_size.md) | 获取Tensor中的数据大小。 |
| [get_element_cnt](get_element_cnt.md) | 获取Tensor中的元素的个数。 |
| [reshape](reshape.md) | 对tensor进行Reshape操作，不改变tensor的内容。 |

## MetaParams类

使用该类获取共享的变量信息。

**表 3**  MetaParams类接口

| 接口名称 | 简介 |
| --- | --- |
| [PyMetaParams构造函数](PyMetaParams构造函数.md) | PyMetaParams构造函数。 |
| [get_name](get_name.md) | 获取Flowfunc的实例名。 |
| [get_attr_int](get_attr_int.md) | 获取指定名称的int类型属性值。 |
| [get_attr_bool_list](get_attr_bool_list.md) | 获取指定名称的bool数组类型属性值。 |
| [get_attr_int_list](get_attr_int_list.md) | 获取指定名称的int数组类型属性值。 |
| [get_attr_int_list_list](get_attr_int_list_list.md) | 获取指定名称的int二维数组类型属性值。 |
| [get_attr_bool](get_attr_bool.md) | 获取指定名称的bool类型属性值。 |
| [get_attr_float_list](get_attr_float_list.md) | 获取指定名称的float数组类型属性值。 |
| [get_attr_tensor_dtype](get_attr_tensor_dtype.md) | 获取指定名称的numpy dtype类型的属性值。 |
| [get_attr_tensor_dtype_list](get_attr_tensor_dtype_list.md) | 获取指定名称的numpy dtype数组类型的属性值。 |
| [get_attr_str](get_attr_str.md) | 获取指定名称的string类型的属性值。 |
| [get_attr_str_list](get_attr_str_list.md) | 获取指定名称的string数组类型的属性值。 |
| [get_attr_float](get_attr_float.md) | 获取指定名称的float类型属性值。 |
| [get_input_num](get_input_num.md) | 获取Flowfunc的输入个数。 |
| [get_output_num](get_output_num.md) | 获取Flowfunc的输出个数。 |
| [get_work_path](get_work_path.md) | 获取Flowfunc的工作路径。 |
| [get_running_device_id](get_running_device_id.md) | 获取正在运行的设备ID。 |

## MetaRunContext类

用于FlowFunc处理函数的上下文信息相关处理，如申请Tensor、设置输出、运行FlowModel等操作。

**表 4**  MetaRunContext类接口

| 接口名称 | 简介 |
| --- | --- |
| [MetaRunContext构造函数](MetaRunContext构造函数.md) | MetaRunContext构造函数。 |
| [alloc_tensor_msg](alloc_tensor_msg.md) | 根据shape、data type以及对齐大小申请tensor类型的FlowMsg。 |
| [set_output](set_output.md) | 设置指定index的output的tensor。 |
| [set_multi_outputs](set_multi_outputs.md) | 批量设置指定index的output的tensor。 |
| [run_flow_model](run_flow_model.md) | 同步执行指定的模型。 |
| [alloc_empty_data_msg](alloc_empty_data_msg.md) | 申请空数据的MsgType类型的message。 |
| [get_user_data（UDF）](get_user_data（UDF）.md) | 获取用户定义数据。 |
| [raise_exception](raise_exception.md) | UDF主动上报异常。 |
| [get_exception](get_exception.md) | UDF获取其他UDF节点上报的异常。 |
| [alloc_raw_data_msg](alloc_raw_data_msg.md) | 根据输入的size申请一块连续内存，用于承载raw data类型的FlowMsg。 |
| [to_flow_msg](to_flow_msg.md) | 将dataflow Tensor转换成FlowMsg。 |

## AffinityPolicy类

**表 5**  AffinityPolicy类接口

| 接口名称 | 简介 |
| --- | --- |
| [AffinityPolicy类](AffinityPolicy类.md) | 亲和策略枚举定义。 |

## BalanceConfig类

当需要均衡分发时，需要设置输出数据标识和权重矩阵相关配置信息，根据配置调度模块可以完成多实例之间的均衡分发。

**表 6**  BalanceConfig类接口

| 接口名称 | 简介 |
| --- | --- |
| [BalanceConfig构造函数](BalanceConfig构造函数.md) | BalanceConfig构造函数。 |
| [set_data_pos](set_data_pos.md) | 设置输出数据对应权重矩阵中的位置。 |
| [get_inner_config](get_inner_config.md) | 获取内部配置对象，被[set_output](set_output.md)或者[set_multi_outputs](set_multi_outputs.md)调用。 |

## FlowMsgQueue类

流式输入场景下（即flow func函数入参为队列时），用于flow func的输入队列，队列中的FlowMsg出队后会根据MsgType转换为对应的数据类型返回给用户。

**表 7**  FlowMsgQueue类接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowMsgQueue构造函数](FlowMsgQueue构造函数.md) | FlowMsgQueue构造函数和析构函数。 |
| [get](get.md) | 获取队列中的元素。 |
| [get_nowait](get_nowait.md) | 无等待地获取队列中的元素，功能等同于get(block=False)。 |
| [full](full.md) | 判断队列是否满。 |
| [full](full.md) | 判断队列是否为空。 |
| [qsize](qsize.md) | 获取队列中当前元素的个数。 |

## UDF日志接口

UDF Python开放了日志记录接口，使用时导入flow\_func模块。使用其中定义的logger对象，调用logger对象封装的不同级别的日志接口。

**表 8**  UDF日志接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowFuncLogger构造函数](FlowFuncLogger构造函数.md) | FlowFuncLogger构造函数。 |
| [get_log_header](get_log_header.md) | 获取日志扩展头信息。 |
| [is_log_enable](is_log_enable.md) | 查询对应级别和类型的日志是否开启。 |
| [运行日志Error级别日志宏](运行日志Error级别日志宏.md) | 运行日志Error级别日志宏。 |
| [运行日志Info级别日志宏](运行日志Info级别日志宏.md) | 运行日志Info级别日志宏。 |
| [调试日志Error级别日志宏](调试日志Error级别日志宏.md) | 调试日志Error级别日志宏。 |
| [调试日志Warn级别日志宏](调试日志Warn级别日志宏.md) | 调试日志Warn级别日志宏。 |
| [调试日志Info级别日志宏](调试日志Info级别日志宏.md) | 调试日志Info级别日志宏。 |
| [调试日志Debug级别日志宏](调试日志Debug级别日志宏.md) | 调试日志Debug级别日志宏。 |
