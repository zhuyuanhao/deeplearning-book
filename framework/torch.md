## Torch
main site：http://www.torch.ch
github：https://github.com/torch/torch7
torch7文档：http://torch7.readthedocs.io/en/latest/index.html
Cheat sheet：https://github.com/torch/torch7/wiki/Cheatsheet
Model Zoo：https://github.com/torch/torch7/wiki/ModelZoo

### luaJIT
使用即时编译（Just-in Time）技术，把 Lua 代码编译成本地机器码后交由 CPU 直接执行，比Lua更快
FFI：Lua代码中申明，连接时直接调用C函数
luarocks：包管理（$ luarocks install image $ luarocks list）
Lua: http://www.lua.org
Lua users: http://lua-users.org/wiki/
LuaJIT: http://luajit.org/luajit.html

TREPL: A REPL for Torch
Torch的一个交互式环境包（Read–eval–print_loop）
https://github.com/torch/trepl/blob/master/README.md
```
$ th -- 打开交互式环境
$ Ctrl+L -- 清屏
$ os.exit() -- 退出，也可用两次 Ctrl+C
```
也可以在luajit解释器中动态加载
```
$ luajit
> repl = require 'trepl'
> repl()
```

功能：
- Tab补全，对命令、函数、变量、文件均有效
- 查看历史，查看历史输出（所有：_RESULTS，上一次：_LAST）
- 自动输出（不需要'='），格式化输出，并且给出执行时间
- 查看帮助：? funcname，环境使用帮助：?
- 执行Shell命令：`$ cmd (example: $ ls)`
- Auto-print after eval (can be stopped with ;)

### itorch
https://github.com/facebook/iTorch
问题：
1.itorch notebook启动时报：socket.error: [Errno 99] Cannot assign requested address
实际上是ipython启动时的问题（默认监听localhost），修改启动参数
$ itorch notebook --ip=127.0.0.1
或者
修改/etc/hosts，确保127.0.0.1和localhost唯一对应。

### Torchnet
抽象化、模块化的Torch，更简单的异步加载数据、多GPU计算
main page：https://github.com/torchnet/torchnet
模块：Datasets, DatasetIterators, Engines, Meters, Logs

### 主要模块
torch：N维数组（Tensor）及线性运算（openmp+sse）
image, gnuplot,ffmpeg,audio：图片处理，绘图，视频，音频
nn,rnn,word2vec,dpnn(reinforcement)：神经网络层（Layers as DAG），Loss Fucntion，最优化算法
optim：最优化算法
cutorch, cunn：GPU端的torch和nn实现
cudnn: NVIDIA CUDNN的包装模块

### 添加层
官方参考：http://torch.ch/docs/developer-docs.html
-- forward过程：
[output] updateOutput(input)
-- backward过程：
[gradInput] updateGradInput(input, gradOutput)
accGradParameters(input, gradOutput, scale) -- 没有参数的层不需要实现该函数

### 多卡和多机
官方没有多机支持，支持多卡。
- 切换GPU：cutorch.setDevice(devID)
- 所有的cuda调用都是异步的，可以使用cutorch.synchronize()同步

基本步骤为
1. load data
2. loop over GPUs (the loop below will be completely anynchronous, so will run parallely)
 2.1. model[gpuX]:forward
 2.2. criterion[gpuX]:forward
 2.3. criterion[gpuX]:backward
 2.4. model[gpuX]:backward
3. cutorch.synchronize()
4. accumulate GPUx'sgradParameters to GPU1's gradParameters
5. do SGD on GPU1
6. copy back GPU1's parameters to GPUx
7. cutorch.synchronize() and print accuracy etc.

### Multi-GPU：
https://github.com/torch/cutorch/issues/42
https://github.com/soumith/imagenet-multiGPU.torch
Multi-Host（parameter server）：https://github.com/sixin-zh/mpiT
lua-mpi(FFI)：https://colberg.org/lua-mpi/README.html
lua-mpi(jzrake)：https://github.com/jzrake/lua-mpi
mpi with docker：https://github.com/ambu50/docker-ib-mpi

### 使用GPU
```
require 'cunn';
require 'cutorch';
net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer:train(trainset)
```
GPU copy
```
cutorch.setDevice(1)
t1 = torch.randn(100):cuda()
cutorch.setDevice(2)
t2 = torch.randn(100):cuda()
-- NvidiaUVA copy
t2:copy(t1)
```

### 模型示例
```
-- load data
trainset= torch.load('cifar10-train.t7')
testset= torch.load('cifar10-test.t7')

-- init dataset
setmetatable(trainset,
  {__index = function(t, i)
  return {t.data[i], t.label[i]}
  end}
);
function trainset:size()
  return self.data:size(1)
end

-- define model
require 'nn';
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.ReLU())
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())

-- define loss function
criterion = nn.ClassNLLCriterion()

-- define trainer
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5

-- training
trainer:train(trainset)

-- predict
predicted = net:forward(testset.data[100])
```

### 项目
动漫图片无损放大：https://github.com/nagadomi/waifu2x
照片油画风格：https://github.com/jcjohnson/neural-style
图片内容解读：https://github.com/karpathy/neuraltalk2
自动生成文本风格：https://github.com/karpathy/char-rnn
torch-rnn：高效的RNN和LSTM库，用它实现的char-rnn模型比原作快1.9x，内存节省7x：https://github.com/jcjohnson/torch-rnn
用Anaconda安装Torch：https://github.com/alexbw/conda-lua-recipes