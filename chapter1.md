# 经典模型
## 卷积神经网络 (Convolutional Neural Network, CNN)
在ILSVRC的ImageNet数据集合上，人眼的辨识错误率大概在5.1%。主要的几种CNN模型在ILSVRC上的结果
![compare](img/cnn-compare.png)

# 框架







# 数据集
## MNIST
数据集大小：~12MB
下载地址：http://yann.lecun.com/exdb/mnist/index.html

MNIST是一个手写数字数据库，它有60000个训练样本集和10000个测试样本集，每个样本图像的宽高为28*28，图像只有一个通道（0（白）-255（黑））。此数据集是以二进制存储的，不能直接以图像格式查看，不过很容易找到将其转换成图像格式的工具。

## CIFAR
数据集大小：~170MB
下载地址：http://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10包含10个类别，50,000个训练图像，彩色图像大小：32×32，10,000个测试图像。CIFAR-100与CIFAR-10类似，包含100个类，每类有600张图片，其中500张用于训练，100张用于测试；这100个类分组成20个超类。图像类别均有明确标注。CIFAR对于图像分类算法测试来说是一个非常不错的中小规模数据集。

## PASCAL VOC
数据集大小：~2GB
下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

PASCAL VOC挑战赛是视觉对象的分类识别和检测的一个基准测试，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。PASCAL VOC图片集包括20个目录：人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）。PASCAL VOC挑战赛在2012年后便不再举办，但其数据集图像质量好，标注完备，非常适合用来测试算法性能。

## Imagenet
数据集大小：~1TB（ILSVRC2016比赛全部数据）
下载地址：http://www.image-net.org/about-stats

Imagenet数据集有1400多万幅图片，涵盖2万多个类别；其中有超过百万的图片有明确的类别标注和图像中物体位置的标注，具体信息如下：
1. Total number of non-empty synsets: 21841
2. Total number of images: 14,197,122
3. Number of images with bounding box annotations: 1,034,908
4. Number of synsets with SIFT features: 1000
5. Number of images with SIFT features: 1.2 million

Imagenet数据集是目前深度学习图像领域应用得非常多的一个领域，关于图像分类、定位、检测等研究工作大多基于此数据集展开。Imagenet数据集文档详细，有专门的团队维护，使用非常方便，在计算机视觉领域研究论文中应用非常广，几乎成为了目前深度学习图像领域算法性能检验的“标准”数据集。
与Imagenet数据集对应的有一个享誉全球的“ImageNet国际计算机视觉挑战赛(ILSVRC)”，以往一般是google、MSRA等大公司夺得冠军，今年（2016）ILSVRC2016中国团队包揽全部项目的冠军。
Imagenet数据集是一个非常优秀的数据集，但是标注难免会有错误，几乎每年都会对错误的数据进行修正或是删除，建议下载最新数据集并关注数据集更新。

## COCO
数据集大小：~40GB
下载地址：http://mscoco.org/

COCO(Common Objects in Context)是一个新的图像识别、分割和图像语义数据集，它有如下特点：
1. Object segmentation
2. Recognition in Context
3. Multiple objects per image
4. More than 300,000 images
5. More than 2 Million instances
6. 80 object categories
7. 5 captions per image
8. Keypoints on 100,000 people

COCO数据集由微软赞助，其对于图像的标注信息不仅有类别、位置信息，还有对图像的语义文本描述，COCO数据集的开源使得近两三年来图像分割语义理解取得了巨大的进展，也几乎成为了图像语义理解算法性能评价的“标准”数据集。
Google开源的开源了图说生成模型show and tell就是在此数据集上测试的。

## Open Image
数据集大小：~1.5GB（不包括图片）
下载地址：https://github.com/openimages/dataset

Open Image是Google推出的一个包含~900万张图像URL的数据集（不包含图片），里面的图片通过标签注释被分为6000多类。该数据集中的标签要比ImageNet（1000类）包含更真实生活的实体存在，它足够让我们从头开始训练深度神经网络。

## Youtube-8M
数据集大小：~1.5TB
下载地址：https://research.google.com/youtube8m/

Youtube-8M为谷歌开源的视频数据集，视频来自youtube，共计8百万个视频，总时长50万小时，4800类。为了保证标签视频数据库的稳定性和质量，谷歌只采用浏览量超过1000的公共视频资源。为了让受计算机资源所限的研究者和学生也可以用上这一数据库，谷歌对视频进行了预处理，并提取了帧级别的特征，提取的特征被压缩到可以放到一个硬盘中（小于1.5T）。

## 数据集资源
1. 深度学习数据集收集网站
http://deeplearning.net/datasets/**
收集大量的各深度学习相关的数据集，但并不是所有开源的数据集都能在上面找到相关信息。
2. Tiny Images Dataset
http://horatio.cs.nyu.edu/mit/tiny/data/index.html
包含8000万的32×32图像，CIFAR-10和CIFAR-100便是从中挑选的。
3. CoPhIR
http://cophir.isti.cnr.it/whatis.html
雅虎发布的超大Flickr数据集，包含1亿多张图片。
4. MirFlickr1M
http://press.liacs.nl/mirflickr/
Flickr数据集中挑选出的100万图像集。
5. SBU captioned photo dataset
http://dsl1.cewit.stonybrook.edu/~vicente/sbucaptions/
Flickr的一个子集，包含100万的图像集。
6. NUS-WIDE
http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm
Flickr中的27万的图像集。
7. Large-Scale Image Annotation using Visual Synset(ICCV 2011)
http://cpl.cc.gatech.edu/projects/VisualSynset/
机器标注的一个超大规模数据集，包含2亿图像。
8. SUN dataset
http://people.csail.mit.edu/jxiao/SUN/
包含13万的图像的数据集。
9. MSRA-MM
http://research.microsoft.com/en-us/projects/msrammdata/
包含100万的图像，23000视频；微软亚洲研究院出品，质量应该有保障。

# 名人
## Alex Krizhevsky
主要论文：Imagenet classification with deep convolutional neural networks(2014)
主页：http://www.cs.toronto.edu/~kriz/ (较旧)
Google Scholar：https://scholar.google.com/citations?user=xegzhJcAAAAJ&hl=en


