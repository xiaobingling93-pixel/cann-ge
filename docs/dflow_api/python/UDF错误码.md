# UDF错误码

## flowfunc

flow\_func\_defines.h提供了flowfunc的错误码供用户使用，主要用于对异常逻辑的判断处理。每个错误码含义如下。

| 返回码 | 含义 | 解决方法 |
| --- | --- | --- |
| FLOW_FUNC_SUCCESS = 0 | 执行成功 | 不涉及 |
| FLOW_FUNC_ERR_PARAM_INVALID = 164000 | 参数校验无效 | 参数校验失败返回该错误码，包括但不限于输入参数超出系统支持范围，过程中某些参数不匹配。返回该错误码时日志会打印异常的参数及异常原因，请结合具体日志定位原因。 |
| FLOW_FUNC_ERR_ATTR_NOT_EXITS = 164001 | 获取属性时属性不存在 | 请检查获取属性的名称，确认是否在获取前对该属性进行了设置。 |
| FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH = 164002 | 获取属性时属性类型不匹配 | 请检查调用GetAttr接口时入参属性名称所对应的属性值类型与出参变量的数据类型是否一致。该错误码对应错误日志打印属性名对应的实际属性的数据类型。 |
| FLOW_FUNC_FAILED = 564000 | UDF内部错误码 | 请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| FLOW_FUNC_ERR_DATA_ALIGN_FAILED = 364000 | 数据对齐失败 | 可能的原因如下：<br><br>  - Flow func实现不正确，比如给定的输入不匹配。<br>  - 某个节点执行时间超时，导致数据对齐等待超时。 |
| FLOW_FUNC_ERR_TIME_OUT_ERROR = 564001 | 执行NN超时 | 请检查日志中是否存在其他报错导致模型执行失败，若存在其他报错针对实际报错定位。若无报错日志，显示模型正常执行，请调整fetch data接口传递的timeout入参，可增加其值或直接设置为-1。 |
| FLOW_FUNC_ERR_NOT_SUPPORT = 564002 | 功能不支持 | 可能的原因如下：<br><br>  - 单func接口未开放该能力，替换成多func接口可以规避该报错。<br>  - 用户未实现对应的接口，如故障恢复场景ResetFlowFuncState未实现默认会返回不支持。 |
| FLOW_FUNC_STATUS_REDEPLOYING = 564003 | 降级部署中 | 可恢复错误触发降级部署导致当前获取不到数据，等待降级部署结束后会返回其他返回码。若降级部署成功，正常返回数据；若降级部署失败，返回其他不可恢复错误码。 |
| FLOW_FUNC_STATUS_EXIT = 564004 | UDF进程退出中 | Flow func在等待输入数据的过程中，如果进程收到退出信号，会返回该错误码，表示进程准备退出，停止输入数据准备。需要根据日志排查UDF进程收到退出信号的原因。 |
| FLOW_FUNC_ERR_DRV_ERROR = 564100 | driver通用错误 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| FLOW_FUNC_ERR_MEM_BUF_ERROR = 564101 | 驱动内存buffer接口错误 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志报错排查问题，或联系工程师（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| FLOW_FUNC_ERR_QUEUE_ERROR = 564102 | 驱动队列接口错误 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志报错排查问题，或联系工程师（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| FLOW_FUNC_ERR_EVENT_ERROR = 564103 | 驱动事件接口错误 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志报错排查问题，或联系工程师（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| FLOW_FUNC_ERR_USER_DEFINE_START  = 9900000 | 用户自定义错误码，从当前错误码开始定义 | - |
| FLOW_FUNC_ERR_USER_DEFINE_END = 9999999 | 用户自定义错误码，以当前错误码结束 | - |

## AICPU

AICPU在执行模型的过程中，有可能向用户上报以下错误码，每个错误码含义如下。

| 返回码 | 含义 | 解决方法 |
| --- | --- | --- |
| int32_t AICPU_SCHEDULE_ERROR_PARAMETER_NOT_VALID = 521001 | 参数校验无效 | 参数校验失败返回该错误码，包括但不限于输入参数超出系统支持范围，过程中某些参数不匹配。返回该错误码时日志会打印异常的参数及异常原因，请结合具体日志定位原因。 |
| int32_t AICPU_SCHEDULE_ERROR_FROM_DRV = 521003 | Driver接口返回错误 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_NOT_FOUND_LOGICAL_TASK = 521005 | 未找到需要执行的AICPU任务 | 请检查环境驱动包与CANN包版本是否兼容。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_INNER_ERROR = 521008 | AICPU内部错误 | 请检查环境驱动包与CANN包版本是否兼容。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_OVERFLOW = 521011 | 发生溢出 | 乘法或加法运算发生溢出，请结合具体日志定位原因。 |
| int32_t AICPU_SCHEDULE_ERROR_MODEL_EXIT_ERR = 521104 | 模型触发执行失败 | 模型执行过程中，返回值被置为异常标记位，因此模型无法继续执行。请查看日志中是否有其他报错，结合具体日志定位原因。 |
| int32_t AICPU_SCHEDULE_ERROR_MODEL_EXECUTE_FAILED = 521106 | 模型执行过程中TSCH上报的模型执行失败 | 模型执行过程中，收到异常终止消息，需要终止模型（终止原因为模型流执行失败）。请查看日志中是否有其他报错，结合具体日志定位原因。 |
| int32_t AICPU_SCHEDULE_ERROR_TSCH_OTHER_ERROR = 521107 | 模型执行过程中TSCH上报的其他错误 | 模型执行过程中，收到异常终止消息，需要终止模型。请查看日志中是否有其他报错，结合具体日志定位原因。 |
| int32_t AICPU_SCHEDULE_ERROR_DISCARD_DATA = 521108 | 模型执行过程中丢弃Mbuf数据 | 模型执行过程中，缓存的Mbuf数据超过阈值，需要丢弃Mbuf数据。解决方法为调整缓存Mbuf的数量或者时间阈值。 |
| int32_t AICPU_SCHEDULE_ERROR_DRV_ERR = 521206 | driver接口返回错误 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_MALLOC_MEM_FAIL_THROUGH_DRV = 521207 | 通过driver接口申请内存失败 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请检查设备内存使用情况，是否达到设备内存上限。请根据日志报错排查问题，或联系工程师（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_SAFE_FUNCTION_ERR = 521208 | memcpy_s等安全函数执行失败 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_INVAILD_EVENT_SUBMIT = 521209 | AICPU提交事件失败 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| int32_t AICPU_SCHEDULE_ERROR_CALL_HCCL = 521500 | AICPU调用HCCL接口失败 | 请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
