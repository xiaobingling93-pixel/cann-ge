# 样例使用指导

## 功能描述

该样例用于指导如何根据ONNX格式的Qwen离线网络，实现LLM模型的加载、执行和获取执行结果等。

在该样例中：
1.  使用atc工具将ONNX格式的Qwen模型转换为om格式的离线模型。
2.  加载离线模型om文件，给定指定的输入，执行模型并获取结果。
3.  注意本样例不涉及如何获得ONNX格式的离线网络，如何导出离线网络可以参见：[Qwen离线模型导出示例](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/nlp/Qwen2_for_Pytorch/readme.md)。

## 目录结构

```
├── model
│   ├── qwen.om                         // ONNX Qwen网络的模型文件, 需要按指导获取atc转换后的om文件, 放到model目录下

├── scripts
│   ├── build.sh                        // sample编译脚本
│   ├── run.sh                          // sample运行脚本

├── src
│   ├── acl.json                        // 系统初始化的配置文件
│   ├── CMakeLists.txt                  // 编译配置脚本
│   ├── sample_qwen_llm.cpp             // 主函数，LLM推理的实现文件

├── CMakeLists.txt                      // 编译脚本，调用src目录下的CMakeLists文件
```

## 环境要求

- 已完成[昇腾AI软件栈在开发环境上的部署](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)

## 实现步骤

1.  以运行用户登录开发环境。

2.  下载代码并上传至环境后，请先进入根目录下"examples/acl/2_sample_qwen_llm"样例目录。

    请注意，下文中的样例目录均指"examples/acl/2_sample_qwen_llm"目录。

3.  准备Qwen模型。
    1.  获取Qwen AIR格式模型。

        您可以从以下链接中获取Qwen网络的模型文件，并以运行用户将获取的文件上传至开发环境的"样例目录/model"目录下。如果目录不存在，需要自行创建。

        -   Qwen网络的模型文件（qwen.onnx）：单击[Link](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann_test/qwen.onnx)下载该文件。

    2.  将Qwen原始模型转换为适配昇腾AI处理器的离线模型（\*.om文件）。注意如果生成的om文件带有架构后缀（如qwen_linux_aarch64.om），请将文件重命名为qwen.om。

        切换到"样例目录/model", 执行模型转换脚本：

        ```
        cd "样例目录/model"
        atc \
        --model=./qwen.onnx \
        --output=./qwen \
        --soc_version=Ascend910B4-1 \
        --framework=5 \
        --precision_mode=must_keep_origin_dtype \
        --op_select_implmode=high_precision \
        --external_weight=0 \
        --output_type=FP32 \
        --input_shape 'input_ids:1,512;past_key_0.key:1,2,512,64;past_key_0.value:1,2,512,64;past_key_1.key:1,2,512,64;past_key_1.value:1,2,512,64;past_key_2.key:1,2,512,64;past_key_2.value:1,2,512,64;past_key_3.key:1,2,512,64;past_key_3.value:1,2,512,64;past_key_4.key:1,2,512,64;past_key_4.value:1,2,512,64;past_key_5.key:1,2,512,64;past_key_5.value:1,2,512,64;past_key_6.key:1,2,512,64;past_key_6.value:1,2,512,64;past_key_7.key:1,2,512,64;past_key_7.value:1,2,512,64;past_key_8.key:1,2,512,64;past_key_8.value:1,2,512,64;past_key_9.key:1,2,512,64;past_key_9.value:1,2,512,64;past_key_10.key:1,2,512,64;past_key_10.value:1,2,512,64;past_key_11.key:1,2,512,64;past_key_11.value:1,2,512,64;past_key_12.key:1,2,512,64;past_key_12.value:1,2,512,64;past_key_13.key:1,2,512,64;past_key_13.value:1,2,512,64;past_key_14.key:1,2,512,64;past_key_14.value:1,2,512,64;past_key_15.key:1,2,512,64;past_key_15.value:1,2,512,64;past_key_16.key:1,2,512,64;past_key_16.value:1,2,512,64;past_key_17.key:1,2,512,64;past_key_17.value:1,2,512,64;past_key_18.key:1,2,512,64;past_key_18.value:1,2,512,64;past_key_19.key:1,2,512,64;past_key_19.value:1,2,512,64;past_key_20.key:1,2,512,64;past_key_20.value:1,2,512,64;past_key_21.key:1,2,512,64;past_key_21.value:1,2,512,64;past_key_22.key:1,2,512,64;past_key_22.value:1,2,512,64;past_key_23.key:1,2,512,64;past_key_23.value:1,2,512,64' \
        --log=error
        ```

        -   --model：原始模型文件路径。
        -   --input_shape：指定模型输入的shape。
        -   --output：生成的qwen_*.om文件重命名为qwen.om并存放在“样例目录/model“目录下。建议使用命令中的默认设置，否则在编译代码前，您还需要修改sample\_qwen\_llm.cpp 中的omModelPath参数值。

            ```
            ret = sampleQwen.PrepareModel("../model/qwen.om");
            ```
        -   --soc\_version：昇腾AI处理器的版本。版本获取可参考[Link](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion)。
        -   --framework：原始框架类型。0：表示Caffe；1：表示air离线模型；3：表示TensorFlow；5：表示ONNX。
        -   --precision_mode：设置模型精度转换策略，must_keep_origin_dtype表示强制保持原始数据类型。
        -   --op_select_implmode：指定算子实现模式，high_precision表示优先保证计算精度。
        -   --external_weight：是否将权重与模型分离存储，0表示权重内嵌在 .om 文件中
        -   --output_type：指定输出tensor的数据类型。

## 构建验证

1.  以运行用户登录开发环境。

2.  请先进入根目录下"examples/acl/2_sample_qwen_llm"样例目录。

    请注意，下文中的样例目录均指"examples/acl/2_sample_qwen_llm"目录。

3. 设置环境变量，配置程序编译依赖的头文件与库文件路径。

    设置以下环境变量后，编译脚本会根据"{DDK_PATH}环境变量值/include/"目录查找编译依赖的头文件，根据{NPU_HOST_LIB}环境变量指向的目录查找编译依赖的库文件。

    **注意**，在配置{NPU_HOST_LIB}环境变量时，需使用的"devlib"目录下*.so库，确保在编译基于AscendCL接口的应用程序时，不依赖其它组件（例如Driver）的*.so库，编译成功后，运行应用程序时，系统会根据LD_LIBRARY_PATH环境变量查找“Ascend-cann-toolkit安装目录/lib64”目录下的*.so库，同时会自动链接到所依赖的其它组件的*.so库。

    -   配置示例如下所示：

        ```
        # toolkit默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
        export DDK_PATH=/usr/local/Ascend/cann
        export NPU_HOST_LIB=$DDK_PATH/devlib
        # toolkit指定路径安装，${install_path}为toolkit的安装目录。
        export DDK_PATH=${install_path}/cann
        export NPU_HOST_LIB=$DDK_PATH/devlib
        ```

4.  切换到"样例目录/scripts", 编译程序。

    ```
    bash build.sh
    ```

5.  运行程序
    ```
    bash run.sh
    ```

6.  执行结果

执行成功后，在屏幕上的关键提示信息示例如下：
```
[INFO] The sample starts to run
[INFO]  SAMPLE start to execute.
[INFO]  acl init success
[INFO]  set device success
[INFO]  create context success
[INFO]  create stream success
[INFO]  load model ../model/qwen.om success.
[INFO]  Start to Process.
[INFO]  The first five inputs information:
[INFO]    Input[0], tensorName=input_ids, size=4096 bytes, dtype=9, format=0, dims=1 512
[INFO]    Input[1], tensorName=past_key_0.key, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]    Input[2], tensorName=past_key_0.value, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]    Input[3], tensorName=past_key_1.key, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]    Input[4], tensorName=past_key_1.value, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]  Start to execute model.
[INFO]  The first five outputs information:
[INFO]    Output[0], tensorName=/lm_head/MatMul:0:logits, size=311164928 bytes, dtype=0, format=0, dims=1 512 151936
[INFO]    Output[1], tensorName=/model/self_attn/Concat_5:0:present_0.key, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]    Output[2], tensorName=/model/self_attn/Concat_6:0:present_0.value, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]    Output[3], tensorName=/model/self_attn_1/Concat_5:0:present_1.key, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]    Output[4], tensorName=/model/self_attn_1/Concat_6:0:present_1.value, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]  predicted_token_id: 33975
[INFO]  SAMPLE PASSED.
```