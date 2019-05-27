# %%
import numpy as np
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd, lr_scheduler
from mxnet.gluon import loss as gloss, nn, data as gdata

import time
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt


# GPU
ctx = d2l.try_gpu()

# Load net
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
        # (批量大小, 通道 * 高 * 宽)形状的输入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
net.load_parameters('../model/mlp.params')

# Load pic
img = Image.open('./1.png')
print(type(img), np.min(img), np.max(img))
img = nd.NDArray(np.array(img))
print(img.shape)
fig = plt.figure()
fig.add_subplot(2, 1, 1).imshow(img[0].asnumpy())
img = img.as_in_context(ctx)
print(net(img).argmax(axis=1))

print("Time consume : %.2fs" % (time.process_time() / 1000))
