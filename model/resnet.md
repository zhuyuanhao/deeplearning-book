## Residual Net (2015)
| | |
|:-- |:-- |
|![ResNet](/assets/resnet_top.png)|![ResNet](/assets/resnet_bottom.png)|

Paper:
Deep Residual Learning for Image Recognition(Kaiming He-CVPR2015)

![ResNet](/assets/resnet.png)
主要的创新在残差网络，如下图所示，其实这个网络的提出本质上还是要解决层次比较深的时候无法训练的问题。这种借鉴了Highway Network思想的网络相当于旁边专门开个通道使得输入可以直达输出，而优化的目标由原来的拟合输出H(x)变成输出和输入的差H(x)-x，其中H(X)是某一层原始的的期望映射输出，x是输入。
![ResNet-Part](/assets/resnet-part.png)

注意虚线部分均处于维度增加部分，亦即卷积核数目倍增的过程，这时进行`F(x) + x` 就会出现二者维度不匹配，这里论文中采用两种方法解决这一问题(其实是三种，但通过实验发现第三种方法会使performance急剧下降，故不采用):
* zero_padding: 对恒等层进行0填充的方式将维度补充完整。这种方法不会增加额外的参数
* projection: 在恒等层采用1x1的卷积核来增加维度。这种方法会增加额外的参数