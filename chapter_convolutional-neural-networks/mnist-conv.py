# %%
import mxnet as mx
import matplotlib.pyplot as plt
import d2lzh as d2l
import numpy as np
from mxnet.gluon import loss as gloss, nn
from mxnet import gluon, init, autograd, nd

# %%


def comp_conv2d(conv2d, X):
    conv2d.initialize()

    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道


batch_size = 48
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

K1 = nd.array([[1, -1]])
K2 = nd.array([[1], [-1]])


for X, y in train_iter:
    fig = plt.figure()
    fig.add_subplot(211).imshow(X[0][0].asnumpy())

    print(X.size)

    conv2d = nn.Conv2D(1, kernel_size=3, padding=1)

    for x in X:
        for pic in x:
            x[0] = comp_conv2d(conv2d, pic)

    fig.add_subplot(212).imshow(X[0][0].asnumpy())

    print(X.size)
    break

# %%
a = nd.array([1, 2, 3], ctx=mx.gpu())
a


# %%
net = nn.Sequential()
net.add(
    nn.Dense(256, activation='relu'),
    nn.Dense(10))
# net.initialize(init.Normal(sigma=0.01), ctx=mx.gpu())
net.initialize(init.Normal(sigma=0.01))
print(net)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5


# %%
def comp_conv2d(conv2d, X):
    conv2d.initialize()

    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道


conv2d = nn.Conv2D(1, kernel_size=3, padding=1)

for X, y in train_iter:
    for x in X:
        for pic in x:
            x[0] = comp_conv2d(conv2d, pic)

# %%

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y).sum()
        l.backward()
        trainer.step(batch_size)
        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size

    fig = plt.figure()
    fig.add_subplot(211).imshow(X[0][0].asnumpy())

    test_acc = d2l.evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
