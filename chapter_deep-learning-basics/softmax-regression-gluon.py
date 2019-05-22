# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
from mxnet.gluon import loss as gloss, nn
from mxnet import gluon, init
import d2lzh as d2l
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'chapter_deep-learning-basics'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# # softmax回归的简洁实现
#
# 我们在[“线性回归的简洁实现”](linear-regression-gluon.ipynb)一节中已经了解了使用Gluon实现模型的便利。下面，让我们再次使用Gluon来实现一个softmax回归模型。首先导入所需的包或模块。

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## 获取和读取数据
#
# 我们仍然使用Fashion-MNIST数据集和上一节中设置的批量大小。

# %%
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %% [markdown]
# ## 定义和初始化模型
#
# 在[“softmax回归”](softmax-regression.ipynb)一节中提到，softmax回归的输出层是一个全连接层。因此，我们添加一个输出个数为10的全连接层。我们使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。

# %%
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

# %% [markdown]
# ## softmax和交叉熵损失函数
#
# 如果做了上一节的练习，那么你可能意识到了分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。因此，Gluon提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳定性更好。

# %%
loss = gloss.SoftmaxCrossEntropyLoss()

# %% [markdown]
# ## 定义优化算法
#
# 我们使用学习率为0.1的小批量随机梯度下降作为优化算法。

# %%
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# %% [markdown]
# ## 训练模型
#
# 接下来，我们使用上一节中定义的训练函数来训练模型。

# %%
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)

# %% [markdown]
# ## 小结
#
# * Gluon提供的函数往往具有更好的数值稳定性。
# * 可以使用Gluon更简洁地实现softmax回归。
#
# ## 练习
#
# * 尝试调一调超参数，如批量大小、迭代周期和学习率，看看结果会怎样。
#
#
#
# ## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/740)
#
# ![](../img/qr_softmax-regression-gluon.svg)
