# DataFlow接口列表

使用DataFlow Python接口构造DataFlow图进行推理。支持定义图处理点，UDF处理点，描述处理点之间的数据流关系；支持导入TensorFlow, ONNX, MindSpore的IR文件作为图处理点计算逻辑定义。

## DataFlow构图接口

**表 1**  DataFlow构图接口

| 接口名称 | 简介 |
| --- | --- |
| [dataflow.FlowData](dataflow-FlowData.md) | DataFlow Graph中的数据节点，每个FlowData对应一个输入。 |
| [FlowNode](FlowNode.md) | DataFlow Graph中的计算节点。 |
| [add_process_point](add_process_point.md) | 给FlowNode添加映射的pp，当前一个FlowNode仅能添加一个pp，添加后会默认将FlowNode的输入输出和pp的输入输出按顺序进行映射。 |
| [map_input](map_input.md) | 给FlowNode映射输入，表示将FlowNode的第node_input_index个输入给到ProcessPoint的第pp_input_index个输入，并且给ProcessPoint的该输入设置上attr里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序去映射FlowNode和ProcessPoint的输入。 |
| [map_output](map_output.md) | 给FlowNode映射输出，表示将pp的第pp_output_index个输出给到FlowNode的第node_output_index个输出，返回映射好的FlowNode节点。 |
| [set_attr](set_attr.md) | 设置FlowNode的属性。 |
| [**call**](__call__.md) | 调用FlowNode进行计算。 |
| [set_balance_scatter](set_balance_scatter.md) | 设置节点balance scatter属性，具有balance scatter属性的UDF可以使用balance options设置负载均衡输出。 |
| [set_balance_gather](set_balance_gather.md) | 设置节点balance gather属性，具有balance gather属性的UDF可以使用balance options设置负载均衡亲和输出。 |
| [set_alias](set_alias.md) | 设置节点别名，使用option:ge.experiment.data_flow_deploy_info_path指定节点部署位置时，flow_node_list字段可使用别名进行指定。 |
| [dataflow.FlowFlag](dataflow-FlowFlag.md) | 设置FlowMsg消息头中的flags。 |
| [FlowGraph](FlowGraph.md) | DataFlow的graph，由输入节点FlowData和计算节点FlowNode构成。 |
| [set_contains_n_mapping_node](set_contains_n_mapping_node.md) | 设置FlowGraph是否包含n_mapping节点。 |
| [set_inputs_align_attrs](set_inputs_align_attrs.md) | 设置FlowGraph中的输入对齐属性。 |
| [set_exception_catch](set_exception_catch.md) | 设置用户异常捕获功能是否开启。 |
| [dataflow.FlowOutput](dataflow-FlowOutput.md) | 描述FlowNode的输出。 |
| [dataflow.Framework](dataflow-Framework.md) | 设置原始网络模型的框架类型。 |
| [FuncProcessPoint](FuncProcessPoint.md) | FuncProcessPoint的构造函数，返回一个FuncProcessPoint对象。 |
| [set_init_param](set_init_param.md) | 设置FuncProcessPoint的初始化参数。 |
| [add_invoked_closure](add_invoked_closure.md) | 添加FuncProcessPoint调用的GraphProcessPoint或者FlowGraphProcessPoint，返回添加好的FuncProcessPoint。 |
| [GraphProcessPoint](GraphProcessPoint.md) | GraphProcessPoint构造函数，返回一个GraphProcessPoint对象。 |
| [fnode](fnode.md) | 根据当前的GraphProcessPoint生成一个FlowNode，返回一个FlowNode对象。 |
| [dataflow.FlowGraphProcessPoint](dataflow-FlowGraphProcessPoint.md) | GraphProcessPoint构造函数，返回一个GraphProcessPoint对象。 |
| [Tensor](Tensor.md) | Tensor的构造函数。 |
| [numpy](numpy.md) | 将Tensor转换到numpy的ndarray。 |
| [dataflow.TensorDesc](dataflow-TensorDesc.md) | Tensor的描述函数。 |
| [dataflow.alloc_tensor](dataflow-alloc_tensor.md) | 根据shape、data type以及对齐大小申请dataflow tensor。 |
| [dataflow.utils.generate_deploy_template](dataflow-utils-generate_deploy_template.md) | 根据FlowGraph生成指定部署位置的option:"ge.experiment.data_flow_deploy_info_path"所需要的文件的模板。 |
| [register](register.md) | 注册自定义类型对应的序列化、反序列化、计算size的函数，可结合feed，fetch接口使用，用于feed/fetch任意Python类型。 |
| [registered](registered.md) | 判断消息类型ID是否被注册过。 |
| [get_msg_type（dataflow）](get_msg_type（dataflow）.md) | 根据类型定义获取注册的消息类型ID。 |
| [get_serialize_func](get_serialize_func.md) | 根据消息类型ID获取注册的序列化函数。 |
| [get_deserialize_func](get_deserialize_func.md) | 根据消息类型ID获取注册的反序列化函数。 |
| [get_size_func](get_size_func.md) | 根据消息类型ID获取注册的计算序列化内存大小的函数。 |
| [deserialize_from_file](deserialize_from_file.md) | 从序列化的pickle文件进行反序列化恢复Python对象。 |
| [pyflow](pyflow.md) | 支持将函数作为pipeline任务在本地或者远端运行。 |
| [method](method.md) | 对于复杂场景支持将类作为pipeline任务在本地或者远端运行。 |
| [npu_model](npu_model.md) | 如果UDF部署在host侧，执行时数据需要从device拷贝到本地进行运算。对于PyTorch场景，如果计算全在device侧，输入输出也是在device侧，执行时数据需要从device拷贝到host，执行后PyTorch再将数据搬到device侧，影响执行性能，使用npu_model可以优化为不搬移数据（即直接下沉到device执行）的方式触发执行。 |
| [dataflow.CountBatch](dataflow-CountBatch.md) | CountBatch功能是指基于UDF为计算处理点将多个数据按batchSize组成batch。该功能应用于dataflow异步场景。 |
| [dataflow.TimeBatch](dataflow-TimeBatch.md) | TimeBatch功能是基于UDF为前提的。<br>正常模型每次处理一个数据，当需要一次处理一批数据时，就需要将这批数据组成一个Batch。最基本的Batch方式是将这批N个数据直接拼接，然后shape前加N，而某些场景需要将某段或者某几段时间数据组成一个batch，并且按特定的维度拼接，则可以通过使用TimeBatch功能来组Batch。 |

## DataFlow运行接口

**表 2**  DataFlow运行接口

| 接口名称 | 简介 |
| --- | --- |
| [dataflow.init](dataflow-init.md) | 初始化dataflow时的options。 |
| [FlowInfo](FlowInfo.md) | DataFlow的flow信息。 |
| [set_user_data](set_user_data.md) | 设置用户信息。 |
| [get_user_data（dataflow）](get_user_data（dataflow）.md) | 获取用户信息。 |
| [user_data](user_data.md) | 获取用户信息。 |
| [data_size](data_size.md) | 获取user_data的长度。 |
| [start_time](start_time.md) | 以属性方式读取和设置FlowInfo的开始时间。 |
| [end_time](end_time.md) | 以属性方法读取和设置FlowInfo的结束时间。 |
| [flow_flags](flow_flags.md) | 以属性方法读取和设置FlowInfo的flow_flags。 |
| [transaction_id](transaction_id.md) | 以属性方式读写事务ID。 |
| [feed_data](feed_data.md) | 将数据输入到Graph。 |
| [feed](feed.md) | 将数据输入到Graph，支持可序列化的任意的输入。 |
| [fetch_data](fetch_data.md) | 获取Graph输出数据。 |
| [fetch](fetch.md) | 获取Graph输出数据。支持可序列化的任意的输出。 |
| [dataflow.finalize](dataflow-finalize.md) | 释放dataflow初始化的资源。 |
| [dataflow.get_running_device_id](dataflow-get_running_device_id.md) | UDF执行时获取当前UDF的运行device_id, 信息来源和UDF部署位置的配置。 |
| [dataflow.get_running_instance_id](dataflow-get_running_instance_id.md) | UDF执行时获取当前UDF的运行实例ID，该信息来源于data_flow_deploy_info.json中的logic_device_list配置。 |
| [dataflow.get_running_instance_num](dataflow-get_running_instance_num.md) | UDF执行时获取当前UDF的运行实例个数，该信息来源于data_flow_deploy_info.json中的logic_device_list配置。 |

## 模块

dataflow module：公共接口的命名空间

## 类

- class CountBatch：CountBatch属性的类
- class FlowData：输入节点类
- class FlowFlag：数据标记类
- class FlowGraph：dataflow的图类
- class FlowInfo：指定输入输出数据携带的信息类
- class FlowNode：计算节点类
- class FlowOutput：计算节点的输出类
- class Framework：IR文件的框架类型的枚举类
- class FuncProcessPoint：UDF处理点类
- class GraphProcessPoint：图处理点类
- class Tensor：张量数据类
- class TensorDesc：张量的描述类
- class TimeBatch：TimeBatch属性的类

## 函数

- init\(...\)：dataflow的资源初始化方法
- finalize\(...\)：dataflow的资源释放方法

## 其他成员

**表 3**  其他成员

| 名称 | 简介 |
| --- | --- |
| DT_FLOAT | df.data_type.DType的对象<br>32位单精度浮点数 |
| DT_FLOAT16 | df.data_type.DType的对象<br>16位半精度浮点数 |
| DT_INT8 | df.data_type.DType的对象<br>有符号8位整数 |
| DT_INT16 | df.data_type.DType的对象<br>有符号16位整数 |
| DT_UINT16 | df.data_type.DType的对象<br>无符号16位整数 |
| DT_UINT8 | df.data_type.DType的对象<br>无符号8位整数 |
| DT_INT32 | df.data_type.DType的对象<br>有符号32位整数 |
| DT_INT64 | df.data_type.DType的对象<br>有符号64位整数 |
| DT_UINT32 | df.data_type.DType的对象<br>无符号32位整数 |
| DT_UINT64 | df.data_type.DType的对象<br>无符号64位整数 |
| DT_BOOL | df.data_type.DType的对象<br>布尔类型 |
| DT_DOUBLE | df.data_type.DType的对象<br>64位双精度浮点数 |
| DT_STRING | df.data_type.DType的对象<br>字符串类型 |
