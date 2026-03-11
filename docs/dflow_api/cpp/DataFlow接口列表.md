# DataFlow接口列表

您可以在“$\{INSTALL\_DIR\}/include/flow\_graph”路径下查看对应接口的头文件。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/cann。

| 接口分类 | 头文件路径 | 简介 | 对应的库文件 |
| --- | --- | --- | --- |
| FlowOperator类 | flow_graph.h | FlowOperator类是DataFlow Graph的节点基类，继承于GE的Operator。不支持在外部单独构造使用。 | NA |
| FlowData类 | flow_graph.h | 继承于FlowOperator类，为DataFlow Graph中的数据节点，每个FlowData对应一个输入。 | libflow_graph.so |
| FlowNode类 | flow_graph.h | 继承于FlowOperator类，DataFlow Graph中的计算节点。 | libflow_graph.so |
| FlowGraph类 | flow_graph.h | DataFlow的graph，由输入节点FlowData和计算节点FlowNode构成。 | libflow_graph.so |
| ProcessPoint类 | process_point.h | ProcessPoint是一个虚基类，无法实例化对象。 | NA |
| FunctionPp类 | process_point.h | 继承于[ProcessPoint类](ProcessPoint类.md)，用来表示Function的计算处理点。 | libflow_graph.so |
| GraphPp类 | process_point.h | 继承自[ProcessPoint类](ProcessPoint类.md)，用来表示Graph的计算处理点。 | libflow_graph.so |
| DataFlowInputAttr结构体 | flow_attr.h | 定义timeBatch和countBatch两种功能实现UDF组batch能力。 | libflow_graph.so |

>![](public_sys-resources/icon-note.gif) **说明：**
>头文件中出现的char\_t类型是char类型的别名。

## DataFlow构图接口

**表 1**  DataFlow构图接口

| 接口名称 | 简介 |
| --- | --- |
| [FlowOperator类](FlowOperator类.md) | FlowOperator类是DataFlow Graph的节点基类，继承于GE的Operator。 |
| [FlowData的构造函数和析构函数](FlowData的构造函数和析构函数.md) | FlowData构造函数和析构函数，构造函数会返回一个FlowData节点。 |
| [FlowNode构造函数和析构函数](FlowNode构造函数和析构函数.md) | FlowNode构造函数和析构函数，构造函数返回一个FlowNode节点。 |
| [SetInput](SetInput.md) | 给FlowNode设置输入，表示将src_op的第src_index个输出作为FlowNode的第dst_index个输入，返回设置好输入的FlowNode节点。 |
| [AddPp](AddPp.md) | 给FlowNode添加映射的ProcessPoint，当前一个FlowNode仅能添加一个ProcessPoint，添加后会默认将FlowNode的输入输出和ProcessPoint的输入输出按顺序进行映射。 |
| [MapInput](MapInput.md) | 给FlowNode映射输入，表示将FlowNode的第node_input_index个输入给到ProcessPoint的第pp_input_index个输入，并且给ProcessPoint的该输入设置上attrs里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序映射FlowNode和ProcessPoint的输入。 |
| [MapOutput](MapOutput.md) | 给FlowNode映射输出，表示将ProcessPoint的第pp_output_index个输出给到FlowNode的第node_output_index个输出，返回映射好的FlowNode节点。 |
| [SetBalanceScatter](SetBalanceScatter.md) | 设置节点balance scatter属性，具有balance scatter属性的UDF可以使用balance options设置负载均衡输出。 |
| [SetBalanceGather](SetBalanceGather.md) | 设置节点balance gather属性，具有balance gather属性的UDF可以使用balance options设置负载均衡亲和输出。 |
| [FlowGraph构造函数和析构函数](FlowGraph构造函数和析构函数.md) | FlowGraph构造函数和析构函数，构造函数会返回一张空的FlowGraph图。 |
| [SetInputs](SetInputs.md) | 设置FlowGraph的输入节点，会自动根据节点的输出连接关系构建出一张FlowGraph图，并返回该图。 |
| [SetOutputs](SetOutputs.md) | 设置FlowGraph的输出节点，并返回该图。 |
| [SetOutputs（index）](SetOutputs（index）.md) | 设置FlowGraph中的FlowNode和FlowNode输出index的关联关系，并返回该图。常用于设置FlowNode部分输出场景，比如FlowNode1有2个输出，但是作为FlowNode2输入的时候只需要FlowNode1的一个输出，这种情况下可以设置FlowNode1的一个输出index。 |
| [SetContainsNMappingNode](SetContainsNMappingNode.md) | 设置FlowGraph是否包含n_mapping节点。 |
| [SetInputsAlignAttrs](SetInputsAlignAttrs.md) | 设置FlowGraph中的输入对齐属性。 |
| [const ge::Graph &ToGeGraph() const](const-ge-Graph-ToGeGraph()-const.md) | 将FlowGraph转换到GE的Graph。 |
| [SetGraphPpBuilderAsync](SetGraphPpBuilderAsync.md) | 设置FlowGraph中的GraphPp的Builder是否异步执行。 |
| [SetExceptionCatch](SetExceptionCatch.md) | 设置用户异常捕获功能是否开启。 |
| [ProcessPoint析构函数](ProcessPoint析构函数.md) | ProcessPoint析构函数。 |
| [GetProcessPointType](GetProcessPointType.md) | 获取ProcessPoint的类型。 |
| [GetProcessPointName](GetProcessPointName.md) | 获取ProcessPoint的名称。 |
| [GetCompileConfig](GetCompileConfig.md) | 获取ProcessPoint编译配置的文件。 |
| [Serialize（ProcessPoint类）](Serialize（ProcessPoint类）.md) | ProcessPoint的序列化方法。由ProcessPoint的子类去实现该方法的功能。 |
| [FunctionPp构造函数和析构函数](FunctionPp构造函数和析构函数.md) | FunctionPp的构造函数和析构函数，构造函数会返回一个FunctionPp对象。 |
| [SetCompileConfig（FunctionPp类）](SetCompileConfig（FunctionPp类）.md) | 设置FunctionPp的json配置文件名字和路径，该配置文件用于将FunctionPp和UDF进行映射。 |
| [AddInvokedClosure (添加调用的GraphPp)](AddInvokedClosure-(添加调用的GraphPp).md) | 添加FunctionPp调用的GraphPp，返回添加好的FunctionPp。 |
| [AddInvokedClosure (添加调用的ProcessPoint子类)](AddInvokedClosure-(添加调用的ProcessPoint子类).md) | 添加FunctionPp调用的GraphPp，返回添加好的FunctionPp。 |
| [AddInvokedClosure (添加调用的FlowGraphPp)](AddInvokedClosure-(添加调用的FlowGraphPp).md) | 添加FunctionPp调用的FlowGraphPp，返回添加好的FunctionPp。 |
| [SetInitParam](SetInitParam.md) | 设置FunctionPp的初始化参数，返回设置好的FunctionPp。 |
| [Serialize（FunctionPp类）](Serialize（FunctionPp类）.md) | FunctionPp的序列化方法。 |
| [GetInvokedClosures](GetInvokedClosures.md) | 获取FunctionPp调用的GraphPp。 |
| [GraphPp构造函数和析构函数](GraphPp构造函数和析构函数.md) | GraphPp构造函数和析构函数，构造函数会返回一个GraphPp对象。 |
| [SetCompileConfig（GraphPp类）](SetCompileConfig（GraphPp类）.md) | 设置GraphPp的json配置文件路径和文件名。配置文件用于AscendGraph的描述和编译。 |
| [Serialize（GraphPp类）](Serialize（GraphPp类）.md) | GraphPp的序列化方法。 |
| [GetGraphBuilder（GraphPp类）](GetGraphBuilder（GraphPp类）.md) | 获取GraphPp中Graph的创建函数。 |
| [FlowGraphPp构造函数和析构函数](FlowGraphPp构造函数和析构函数.md) | FlowGraphPp构造函数和析构函数，构造函数会返回一个FlowGraphPp对象。 |
| [Serialize（FlowGraphPp类）](Serialize（FlowGraphPp类）.md) | FlowGraphPp的序列化方法。 |
| [GetGraphBuilder（FlowGraphPp类）](GetGraphBuilder（FlowGraphPp类）.md) | 获取FlowGraphPp中Graph的创建函数。 |
| [TimeBatch](TimeBatch.md) | TimeBatch功能是基于UDF为前提的。<br>正常模型每次处理一个数据，当需要一次处理一批数据时，就需要将这批数据组成一个Batch，最基本的Batch方式是将这批N个数据直接拼接，然后shape前加N，而某些场景需要将某段或者某几段时间数据组成一个batch，并且按特定的维度拼接，则可以通过使用TimeBatch功能来组Batch。 |
| [CountBatch](CountBatch.md) | CountBatch功能是指基于UDF为计算处理点将多个数据按batchSize组成batch。 |

## DataFlow运行接口

**表 2**  DataFlow运行接口

| 接口名称 | 简介 |
| --- | --- |
| [FeedDataFlowGraph（feed所有输入）](FeedDataFlowGraph（feed所有输入）.md) | 将所有数据输入到Graph图。 |
| [FeedDataFlowGraph（按索引feed输入）](FeedDataFlowGraph（按索引feed输入）.md) | 将数据按索引输入到Graph图。 |
| [FeedDataFlowGraph（feed所有FlowMsg）](FeedDataFlowGraph（feed所有FlowMsg）.md) | 将数据输入到Graph图。 |
| [FeedDataFlowGraph（按索引feed FlowMsg）](FeedDataFlowGraph（按索引feed-FlowMsg）.md) | 将数据按索引输入到Graph图。 |
| [FeedRawData](FeedRawData.md) | 将原始数据输入到Graph图。 |
| [FetchDataFlowGraph（获取所有输出数据）](FetchDataFlowGraph（获取所有输出数据）.md) | 获取图输出数据。 |
| [FetchDataFlowGraph（按索引获取输出数据）](FetchDataFlowGraph（按索引获取输出数据）.md) | 按索引获取图输出数据。 |
| [FetchDataFlowGraph（获取所有输出FlowMsg）](FetchDataFlowGraph（获取所有输出FlowMsg）.md) | 获取图输出数据。 |
| [FetchDataFlowGraph（按索引获取输出FlowMsg）](FetchDataFlowGraph（按索引获取输出FlowMsg）.md) | 按索引获取图输出数据。 |
| [DataFlowInfo数据类型构造函数和析构函数](DataFlowInfo数据类型构造函数和析构函数.md) | DataFlowInfo构造函数和析构函数。 |
| [SetUserData（DataFlowInfo数据类型）](SetUserData（DataFlowInfo数据类型）.md) | 设置用户信息。 |
| [GetUserData（DataFlowInfo数据类型）](GetUserData（DataFlowInfo数据类型）.md) | 获取用户信息。 |
| [SetStartTime（DataFlowInfo数据类型）](SetStartTime（DataFlowInfo数据类型）.md) | 设置数据的开始时间戳。 |
| [GetStartTime（DataFlowInfo数据类型）](GetStartTime（DataFlowInfo数据类型）.md) | 获取数据的开始时间戳。 |
| [SetEndTime（DataFlowInfo数据类型）](SetEndTime（DataFlowInfo数据类型）.md) | 设置数据的结束时间戳。 |
| [GetEndTime（DataFlowInfo数据类型）](GetEndTime（DataFlowInfo数据类型）.md) | 获取数据的结束时间戳。 |
| [SetFlowFlags（DataFlowInfo数据类型）](SetFlowFlags（DataFlowInfo数据类型）.md) | 设置数据中的flags。 |
| [GetFlowFlags（DataFlowInfo数据类型）](GetFlowFlags（DataFlowInfo数据类型）.md) | 获取数据中的flags。 |
| [SetTransactionId（DataFlowInfo数据类型）](SetTransactionId（DataFlowInfo数据类型）.md) | 设置DataFlow数据传输使用的事务ID。 |
| [GetTransactionId（DataFlowInfo数据类型）](GetTransactionId（DataFlowInfo数据类型）.md) | 获取DataFlow数据传输使用的事务ID。 |
| [FlowMsg数据类型构造函数和析构函数](FlowMsg数据类型构造函数和析构函数.md) | FlowMsg构造函数和析构函数。 |
| [GetMsgType（FlowMsg数据类型）](GetMsgType（FlowMsg数据类型）.md) | 获取FlowMsg的消息类型。 |
| [SetMsgType（FlowMsg数据类型）](SetMsgType（FlowMsg数据类型）.md) | 设置FlowMsg的消息类型。 |
| [GetTensor（FlowMsg数据类型）](GetTensor（FlowMsg数据类型）.md) | 获取FlowMsg中的Tensor指针。 |
| [GetRetCode（FlowMsg数据类型）](GetRetCode（FlowMsg数据类型）.md) | 获取输入FlowMsg中的错误码。 |
| [SetRetCode（FlowMsg数据类型）](SetRetCode（FlowMsg数据类型）.md) | 设置FlowMsg中的错误码。 |
| [SetStartTime（FlowMsg数据类型）](SetStartTime（FlowMsg数据类型）.md) | 设置FlowMsg消息头中的开始时间戳。 |
| [GetStartTime（FlowMsg数据类型）](GetStartTime（FlowMsg数据类型）.md) | 获取FlowMsg消息中的开始时间戳。 |
| [SetEndTime（FlowMsg数据类型）](SetEndTime（FlowMsg数据类型）.md) | 设置FlowMsg消息头中的结束时间戳。 |
| [GetEndTime（FlowMsg数据类型）](GetEndTime（FlowMsg数据类型）.md) | 获取FlowMsg消息中的结束时间戳。 |
| [SetFlowFlags（FlowMsg数据类型）](SetFlowFlags（FlowMsg数据类型）.md) | 设置FlowMsg消息头中的flags。 |
| [GetFlowFlags（FlowMsg数据类型）](GetFlowFlags（FlowMsg数据类型）.md) | 获取FlowMsg消息头中的flags。 |
| [GetTransactionId（FlowMsg数据类型）](GetTransactionId（FlowMsg数据类型）.md) | 获取FlowMsg消息中的事务ID。 |
| [SetTransactionId（FlowMsg数据类型）](SetTransactionId（FlowMsg数据类型）.md) | 设置FlowMsg消息中的事务ID。 |
| [SetUserData（FlowMsg数据类型）](SetUserData（FlowMsg数据类型）.md) | 设置用户信息。 |
| [GetUserData（FlowMsg数据类型）](GetUserData（FlowMsg数据类型）.md) | 获取用户信息。 |
| [GetRawData（FlowMsg数据类型）](GetRawData（FlowMsg数据类型）.md) | 获取RawData类型的数据对应的数据指针和数据大小。 |
| [AllocTensor（FlowBufferFactory数据类型）](AllocTensor（FlowBufferFactory数据类型）.md) | 根据shape、data type和对齐大小申请Tensor。 |
| [AllocTensorMsg（FlowBufferFactory数据类型）](AllocTensorMsg（FlowBufferFactory数据类型）.md) | 根据shape、data type和对齐大小申请FlowMsg。 |
| [AllocRawDataMsg（FlowBufferFactory数据类型）](AllocRawDataMsg（FlowBufferFactory数据类型）.md) | 根据输入的size申请一块连续内存，用于承载raw data类型的数据。 |
| [AllocEmptyDataMsg（FlowBufferFactory数据类型）](AllocEmptyDataMsg（FlowBufferFactory数据类型）.md) | 申请空数据的MsgType类型的message。 |
| [ToFlowMsg（tensor）](ToFlowMsg（tensor）.md) | 根据输入的Tensor转换成用于承载Tensor的FlowMsg。 |
| [ToFlowMsg（raw data）](ToFlowMsg（raw-data）.md) | 根据输入的raw data转换成用于承载raw data的FlowMsg。 |
