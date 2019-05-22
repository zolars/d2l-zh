# %%
import mxnet as mx
import matplotlib.pyplot as plt
import d2lzh as d2l
import numpy as np
from mxnet.gluon import loss as gloss, nn
from mxnet import gluon, init, autograd, nd


# def comp_conv2d(conv2d, X):
#     conv2d.initialize()

#     X = X.reshape((1, 1) + X.shape)
#     Y = conv2d(X)
#     return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道


# batch_size = 2
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# K1 = nd.array([[1, -1]])
# K2 = nd.array([[1], [-1]])


# for X, y in train_iter:

#     fig = plt.figure()

#     X = X[0][0]

#     fig.add_subplot(411).imshow(X.asnumpy())

#     conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
#     Y = comp_conv2d(conv2d, X)

#     fig.add_subplot(414).imshow(Y.asnumpy())
#     print(Y.size)
#     break

# %%

net = nn.Sequential()
net.add(
    nn.Dense(256, activation='relu'),
    nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
print(net)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

for X, y in train_iter:
    print(X.shape)
    break

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
#               None, trainer)
