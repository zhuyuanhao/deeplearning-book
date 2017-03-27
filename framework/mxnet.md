## MXNet
main site & doc: http://mxnet.io/
中文文档：https://github.com/dmlc/mxnet/tree/master/docs/zh
github: https://github.com/dmlc/mxnet
dmlc site: http://dmlc.ml/ github: https://github.com/dmlc
Model Zoo: https://github.com/dmlc/mxnet-model-gallery
examples: https://github.com/dmlc/mxnet/blob/master/example

### 概述
MXNet(mix-net) 使用C++编写，支持命令式和符号式设计，支持多种语言（C++, Python, R, Scala, Julia），支持多种平台（云，集群，物理机，手持设备）。MXNet 支持多卡和多机的并行。总体设计如图：
![MXNet 总体结构](/assets/mxnet.png)
* Engine: 自动检测命令式和符号式代码中的数据依赖，并有效调度。命令式设计更灵活，易于debug，符号式设计易于全局优化。
* Symbol: 符号式设计接口。包含基本操作和卷积等复杂操作，支持符号自动优化和前向、后向计算。
* NDArray: 命令式的张量计算接口。能和Symbol及宿主语言无缝衔接。
* KVStore: parameter server模式的分布式数据操作接口。使用两级数据同步，机器内同步和机器间同步。支持多种同步模式。
* 内存：使用inplace和co-share的方式减少内存占用。inplace方式统计数据还需要被多少其他单元使用，当不被使用时释放。co-share将不需要并行的数据单元共享同一块内存。
* 其他：提供工具将数据压缩成一个特定格式的包，便于数据预取和多线程操作。

