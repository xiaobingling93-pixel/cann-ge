# ES (Eager Style) Generator
## 前置要求
1. 通过[安装指导](../../build.md#2-安装软件包)正确安装`toolkit`包，并按照指导**正确配置环境变量**
2. 通过[安装指导](../../build.md#2-安装软件包)正确安装算子`ops`包（ES 依赖算子原型进行 API 生成），并按照指导**正确配置环境变量**
## 环境变量要求
gen_esb 所需环境变量列表：
- ASCEND_OPP_PATH: 指向安装目录下的opp路径
- LD_LIBRARY_PATH: 指定动态链接库搜索路径的环境变量

注：上述环境变量无需也不推荐单独配置，默认[前置要求](#前置要求)中已经配置过的环境变量即满足要求

## 功能说明
### 本程序支持两种生成模式
1. 代码生成模式
  生成 ES 图构建器的 C、C++、Python 代码，包括：
  - 所有支持的算子(ops)的C接口
  - 所有支持的算子的C++接口
  - 所有支持的算子的Python接口
  - 聚合头文件，方便用户一次性包含所有算子
  - 聚合Python文件，方便用户一次性导入所有算子
2. 历史原型库生成模式
  生成历史原型结构化数据，包括：
  - 版本索引
  - 版本元信息
  - 该版本的算子原型数据

## 使用方法
### 代码生成模式
```bash
gen_esb [--output_dir=DIR] [--module_name=NAME] [--h_guard_prefix=PREFIX] [--exclude_ops=OP_TYPE1,OP_TYPE2] [--history_registry=PKG_DIR] [--release_version=VER]
```
### 历史原型库生成模式
```bash
gen_esb --mode=extract_history --release_version=VER [--output_dir=DIR] [--release_date=YYYY-MM-DD] [--branch_name=BRANCH]
```

注：因为[前置要求](#前置要求)中已经配置过环境变量，此时`gen_esb`已经被添加到了`PATH`环境变量中，因此可直接执行
### 参数说明
- --mode：可选参数，指定生成模式，支持 `codegen` 和 `extract_history`
  如果不指定，默认codegen
- --output_dir：可选参数，指定生成的目标目录
  如果不指定，默认输出到当前目录
- --module_name：可选参数，控制聚合头文件的命名
  - "math" -> es_math_ops_c.h, es_math_ops.h, es_math_ops.py
  - "all" -> es_all_ops_c.h, es_all_ops.h, es_all_ops.py
  - 不传递 -> 默认为"all"
- --h_guard_prefix：可选参数，控制生成的头文件保护宏前缀，用于可能的内外部算子同名情况的区分
  - 如果不指定，使用默认前缀
  - 指定时，拼接默认前缀
  - python文件不感知此参数，同名场景通过不同的路径避免冲突
- --exclude_ops：可选参数，控制排除代码生成的算子
  - 根据 `,` 分隔算子名
- --history_registry：可选参数，指定代码生产的历史原型库目录
  - 如果不指定，默认不启用历史原型库
  - 指定时，生成的C++接口会包含历史原型库中兼容的版本信息
- --release_version：
  - 代码生成模式：可选参数，与 `--history_registry` 配合使用，指定当前版本号，生成的C++接口包含该版本的兼容版本信息；如果不指定，生成当前日期为基准兼容的历史版本
  - 历史原型库生成模式：必填参数，指定当前历史原型数据对应的版本号
- --release_date：可选参数，控制历史原型结构化数据的发布日期，格式 `YYYY-MM-DD`
  - 如果不指定，使用当前日期
- --branch_name：可选参数，控制历史原型结构化数据的发布分支名

### 输出文件说明
#### 代码生成模式输出
- es_<module>_ops_c.h：C接口聚合头文件
- es_<module>_ops.h：C++接口聚合头文件
- es_<module>_ops.py：Python接口聚合文件
- es_<op_type>_c.h：单个算子的C接口头文件
- es_<op_type>.cpp：单个算子的C接口实现文件
- es_<op_type>.h：单个算子的C++接口头文件
- es_<op_type>.py：单个算子的Python接口文件

#### 历史原型库生成模式输出
- index.json：版本索引
- registry/<ver>/metadata.json：版本元信息
- registry/<ver>/operators.json：该版本算子原型数据

## 使用示例
### 生成代码到当前目录，使用默认模块名"all"，默认保护宏前缀
`gen_esb`
 
### 生成代码到指定目录，使用默认模块名"all"，默认保护宏前缀
`gen_esb --output_dir=./output`
 
### 生成代码到指定目录，使用"math"模块名，默认保护宏前缀
`gen_esb --output_dir=./output --module_name=math`
 
### 生成代码到指定目录，使用"all"模块名，默认保护宏前缀
`gen_esb --output_dir=./output --module_name=all`
 
### 生成代码到指定目录，使用"math"模块名，自定义保护宏前缀"MY_CUSTOM"
`gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM`

### 生成代码到指定目录，使用"math"模块名，自定义保护宏前缀"MY_CUSTOM"，并排除 Add 算子生成
`gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM --exclude_ops=Add`

### 生成代码到指定目录，使用"math"模块名，默认保护宏前缀，生成的C++接口会包含math历史原型目录中以当前日期为基准筛选的兼容版本信息
`./gen_esb --output_dir=./output --module_name=math --history_registry=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math`

### 生成代码到指定目录，使用"math"模块名，默认保护宏前缀，生成的C++接口会包含math历史原型目录中"8.0.RC2"版本兼容的历史版本信息
`./gen_esb --output_dir=./output --module_name=math --history_registry=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math --release_version=8.0.RC2`

### 生成历史原型结构化数据到当前目录，发布版本为"8.0.RC1"，默认发布日期为当前日期
`./gen_esb --mode=extract_history --release_version=8.0.RC1`
 
### 生成历史原型结构化数据到指定目录，发布版本为"8.0.RC1"，默认发布日期为当前日期
`./gen_esb --mode=extract_history --release_version=8.0.RC1 --output_dir=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math`
 
### 生成历史原型结构化数据到指定目录，发布版本为"8.0.RC1"，自定义发布日期"2024-09-30"，分支名为"master"
`./gen_esb --mode=extract_history --release_version=8.0.RC1 --output_dir=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math --release_date=2024-09-30 --branch_name=master`

## 注意事项
1. 确保[环境变量](#环境变量要求)已正确设置
2. 确保有足够的磁盘空间存储生成的代码文件
3. 生成的代码文件数量取决于系统中注册的算子数量
4. 保护宏前缀应该以大写字母和下划线组成，避免与C++关键字冲突
 
## 错误处理
- 如果环境变量未设置，程序会提示错误并退出
- 如果输出目录创建失败，会回退到当前目录
- 不支持的算子会被记录在生成的代码注释中
