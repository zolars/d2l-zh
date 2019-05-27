# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'chapter_convolutional-neural-networks'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# # 池化层
#
# 回忆一下，在[“二维卷积层”](conv-layer.ipynb)一节里介绍的图像物体边缘检测应用中，我们构造卷积核从而精确地找到了像素变化的位置。设任意二维数组`X`的`i`行`j`列的元素为`X[i, j]`。如果我们构造的卷积核输出`Y[i, j]=1`，那么说明输入中`X[i, j]`和`X[i, j+1]`数值不一样。这可能意味着物体边缘通过这两个元素之间。但实际图像里，我们感兴趣的物体不会总出现在固定位置：即使我们连续拍摄同一个物体也极有可能出现像素位置上的偏移。这会导致同一个边缘对应的输出可能出现在卷积输出`Y`中的不同位置，进而对后面的模式识别造成不便。
#
# 在本节中我们介绍池化（pooling）层，它的提出是为了缓解卷积层对位置的过度敏感性。
#
# ## 二维最大池化层和平均池化层
#
# 同卷积层一样，池化层每次对输入数据的一个固定形状窗口（又称池化窗口）中的元素计算输出。不同于卷积层里计算输入和核的互相关性，池化层直接计算池化窗口内元素的最大值或者平均值。该运算也分别叫做最大池化或平均池化。在二维最大池化中，池化窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动。当池化窗口滑动到某一位置时，窗口中的输入子数组的最大值即输出数组中相应位置的元素。
#
# ![池化窗口形状为$2\times 2$的最大池化](../img/pooling.svg)
#
# 图5.6展示了池化窗口形状为$2\times 2$的最大池化，阴影部分为第一个输出元素及其计算所使用的输入元素。输出数组的高和宽分别为2，其中的4个元素由取最大值运算$\text{max}$得出：
#
# $$
# \max(0,1,3,4)=4,\\
# \max(1,2,4,5)=5,\\
# \max(3,4,6,7)=7,\\
# \max(4,5,7,8)=8.\\
# $$
#
#
# 二维平均池化的工作原理与二维最大池化类似，但将最大运算符替换成平均运算符。池化窗口形状为$p \times q$的池化层称为$p \times q$池化层，其中的池化运算叫作$p \times q$池化。
#
# 让我们再次回到本节开始提到的物体边缘检测的例子。现在我们将卷积层的输出作为$2\times 2$最大池化的输入。设该卷积层输入是`X`、池化层输出为`Y`。无论是`X[i, j]`和`X[i, j+1]`值不同，还是`X[i, j+1]`和`X[i, j+2]`不同，池化层输出均有`Y[i, j]=1`。也就是说，使用$2\times 2$最大池化层时，只要卷积层识别的模式在高和宽上移动不超过一个元素，我们依然可以将它检测出来。
#
# 下面把池化层的前向计算实现在`pool2d`函数里。它跟[“二维卷积层”](conv-layer.ipynb)一节里`corr2d`函数非常类似，唯一的区别在计算输出`Y`上。

# %%
from mxnet import nd
from mxnet.gluon import nn


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

# %% [markdown]
# 我们可以构造图5.6中的输入数组`X`来验证二维最大池化层的输出。


# %%
X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))

# %% [markdown]
# 同时我们实验一下平均池化层。

# %%
pool2d(X, (2, 2), 'avg')

# %% [markdown]
# ## 填充和步幅
#
# 同卷积层一样，池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状。池化层填充和步幅与卷积层填充和步幅的工作机制一样。我们将通过`nn`模块里的二维最大池化层MaxPool2D来演示池化层填充和步幅的工作机制。我们先构造一个形状为(1, 1, 4, 4)的输入数据，前两个维度分别是批量和通道。

# %%
X = nd.arange(16).reshape((1, 1, 4, 4))
X

# %% [markdown]
# 默认情况下，`MaxPool2D`实例里步幅和池化窗口形状相同。下面使用形状为(3, 3)的池化窗口，默认获得形状为(3, 3)的步幅。

# %%
pool2d = nn.MaxPool2D(3)
pool2d(X)  # 因为池化层没有模型参数，所以不需要调用参数初始化函数

# %% [markdown]
# 我们可以手动指定步幅和填充。

# %%
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)

# %% [markdown]
# 当然，我们也可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅。

# %%
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)

# %% [markdown]
# ## 多通道
#
# 在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。下面将数组`X`和`X+1`在通道维上连结来构造通道数为2的输入。

# %%
X = nd.concat(X, X + 1, dim=1)
X

# %% [markdown]
# 池化后，我们发现输出通道数仍然是2。

# %%
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)

# %% [markdown]
# ## 小结
#
# * 最大池化和平均池化分别取池化窗口中输入元素的最大值和平均值作为输出。
# * 池化层的一个主要作用是缓解卷积层对位置的过度敏感性。
# * 可以指定池化层的填充和步幅。
# * 池化层的输出通道数跟输入通道数相同。
#
#
# ## 练习
#
# * 分析池化层的计算复杂度。假设输入形状为$c\times h\times w$，我们使用形状为$p_h\times p_w$的池化窗口，而且使用$(p_h, p_w)$填充和$(s_h, s_w)$步幅。这个池化层的前向计算复杂度有多大？
# * 想一想，最大池化层和平均池化层在作用上可能有哪些区别？
# * 你觉得最小池化层这个想法有没有意义？
#
#
#
# ## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6406)
#
# ![](../img/qr_pooling.svg)
