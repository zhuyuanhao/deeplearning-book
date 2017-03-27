## Google Net (2014)
| | | |
|:-- |:-- |:-- |
|![GoogleNet-Top](/assets/googlenet_top.png) | ![GoogleNet-Middle](/assets/googlenet_mid.png) |![GoogleNet-Bottom](/assets/googlenet_bottom.png) |

主要的创新在于他的Inception，这是一种网中网（Network In Network）的结构，即原来的结点也是一个网络。Inception一直在不断发展，目前已经V2、V3、V4。Inception的结构如下图所示，其中1*1卷积主要用来降维，用了Inception之后整个网络结构的宽度和深度都可扩大，能够带来2-3倍的性能提升。
![GoogleNet-Part](/assets/googlenet_part.png)

- [v1 2014] Going Deeper with Convolutions, 6.67% test error
Inception v1的网络，将1x1，3x3，5x5的conv和3x3的pooling，stack在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性；
![GoogleNet](/assets/googlenet.png)

- [v2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 4.8% test error
v2的网络在v1的基础上，进行了改进，一方面了加入了BN层，减少了Internal Covariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯，另外一方面学习VGG用2个3x3的conv替代inception模块中的5x5，既降低了参数数量，也加速计算；

- [v3] Rethinking the Inception Architecture for Computer Vision, 3.5% test error
v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块；

- [v4] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 3.08% test error
v4研究了Inception模块结合Residual Connection能不能有改进？发现ResNet的结构可以极大地加速训练，同时性能也有提升，得到一个Inception-ResNet v2网络，同时还设计了一个更深更优化的Inception v4模型，能达到与Inception-ResNet v2相媲美的性能。