# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
from mxnet.gluon import loss as gloss
from mxnet import nd
import mxnet as mx
import d2lzh as d2l
import os
import time
try:
    os.chdir(os.path.join(os.getcwd(), 'chapter_deep-learning-basics'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# # 多层感知机的从零开始实现
#
# 我们已经从上一节里了解了多层感知机的原理。下面，我们一起来动手实现一个多层感知机。首先导入实现所需的包或模块。

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## 获取和读取数据
#
# 这里继续使用Fashion-MNIST数据集。我们将使用多层感知机对图像进行分类。

# %%
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %% [markdown]
# ## 定义模型参数
#
# 我们在[“softmax回归的从零开始实现”](softmax-regression-scratch.ipynb)一节里已经介绍了，Fashion-MNIST数据集中图像形状为$28 \times 28$，类别数为10。本节中我们依然使用长度为$28 \times 28 = 784$的向量表示每一张图像。因此，输入个数为784，输出个数为10。实验中，我们设超参数隐藏单元个数为256。

# %%
num_inputs, num_outputs, num_hiddens = 784, 10, 784

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

# %% [markdown]
# ## 定义激活函数
#
# 这里我们使用基础的`maximum`函数来实现ReLU，而非直接调用`relu`函数。

# %%


def relu(X):
    return nd.maximum(X, 0)

# %% [markdown]
# ## 定义模型
#
# 同softmax回归一样，我们通过`reshape`函数将每张原始图像改成长度为`num_inputs`的向量。然后我们实现上一节中多层感知机的计算表达式。

# %%


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

# %% [markdown]
# ## 定义损失函数
#
# 为了得到更好的数值稳定性，我们直接使用Gluon提供的包括softmax运算和交叉熵损失计算的函数。


# %%
loss = gloss.SoftmaxCrossEntropyLoss()

# %% [markdown]
# ## 训练模型
#
# 训练多层感知机的步骤和[“softmax回归的从零开始实现”](softmax-regression-scratch.ipynb)一节中训练softmax回归的步骤没什么区别。我们直接调用`d2lzh`包中的`train_ch3`函数，它的实现已经在[“softmax回归的从零开始实现”](softmax-regression-scratch.ipynb)一节里介绍过。我们在这里设超参数迭代周期数为5，学习率为0.5。

# %%
start = time.clock()

num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
print("Time consume : " + str(time.clock() - start))

# %% Show results
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[10:19], titles[10:19])

# %% [markdown]
# ## 小结
#
# * 可以通过手动定义模型及其参数来实现简单的多层感知机。
# * 当多层感知机的层数较多时，本节的实现方法会显得较烦琐，例如在定义模型参数的时候。
#
# ## 练习
#
# * 改变超参数`num_hiddens`的值，看看对实验结果有什么影响。
# * 试着加入一个新的隐藏层，看看对实验结果有什么影响。
#
#
#
# ## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/739)
#
# ![](../img/qr_mlp-scratch.svg)
