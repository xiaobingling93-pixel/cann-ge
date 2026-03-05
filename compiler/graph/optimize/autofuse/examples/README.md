## 上板验证
假设用户的昇腾设备上已经正确的搭建了环境，安装了配套版本的CANN软件包、驱动固件包，搭建了TensorFlow环境。此处基于TensorFlow1.15网络，演示如何创建对应的python脚本，设置环境变量，运行脚本用例，以及性能&精度分析。

### 脚本创建
当前自动融合支持element+element，element+broadcast，element+reduce和element+concat类型算子的融合。

部分复杂计算型算子，例如zerosLike，biasAdd，Squeeze等等当前也已支持融合。更多vectore类型算子的融合能力逐渐开放中。

#### 类型1：elementwise + elementwise
假设想验证几个elementWise类型算子的自动融合场景，可以参考eleandele目录下的各py文件，构造对应的模型并编译执行。

#### 类型2：elementwise + broadcast
假设想验证几个elementWise类型和broadcast类型的算子的自动融合场景，可以参考eleandbroadcast目录下的各py文件，构造对应的模型并编译执行。


#### 类型3：elementwise + reduce
假设想验证几个elementWise类型和reduce类型的算子的自动融合场景，可以参考eleandreduce目录下的各py文件，构造对应的模型并编译执行。

### 其他类型（包含concat，split，slice，gather，transpose等算子的结构）
待构造


### 执行用例

1. 设置环境变量

   执行用例前，需要设置如下环境变量，设置运行NPU设备，并打开自动融合功能。
   ```
    #用户自己的driver包安装路径
 	 source /usr/local/Ascend/driver/bin/setenv.sh
 	 #用户自己的CANN包安装路径，参考(docs/build.md)安装社区toolkit包
 	 source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    export PYTHONPATH=/usr_path/:$PYTHONPATH
    #假设跑在device0
 	 export ASCEND_DEVICE_ID=0
 	 #开启自动融合。--autofuse_enable_pass 表示希望开启的一些额外的融合能力，目前支持reduce,concat,slice,split,gather,transpose
 	 export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat,slice,split,gather,transpose;"
    一般建议第一次测试时，只开启reduce,concat。其他几类搬运类算子，取决于芯片形态以及网络结构来决定是否要开启。

   ```
2. 执行用例
   python3 test.py

### 结果分析
用户开启自动融合功能后，往往想直观的看到融合后的图结构上，图上包含有哪些算子，以及每个融合算子内包含了哪些原始算子节点。开发者可以使用CANN的dump图能力，通过下面的环境变量配置，在设置的dump图路径下生成图编译时期的各种dump图。

    export DUMP_GE_GRAPH=2
    export DUMP_GRAPH_LEVEL=1
    export DUMP_GRAPH_PATH="/xxx/xxx/"

用户可以通过打开dump的相关配置，分析精度问题。可以通过profiling的相关配置，观察打开自动融合后，性能收益情况。详细的精度调试工具以及Profling性能分析工具的使用方法可参见[精度调试工具指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)与[Profiling性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)。
