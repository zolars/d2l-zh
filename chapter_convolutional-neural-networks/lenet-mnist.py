# %%
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd, lr_scheduler
from mxnet.gluon import loss as gloss, nn, data as gdata
import time
import sys
import os


class SimpleLRScheduler(lr_scheduler.LRScheduler):
    def __init__(self, learning_rate=0.5):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        return self.learning_rate


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter


def load_data_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.MNIST(root=root, train=True)
    mnist_test = gdata.vision.MNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter


# net Initialize
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

# Load data
batch_size = 256
train_iter, test_iter = load_data_mnist(batch_size=batch_size)


def train(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            net.save_parameters('../model/MNIST/mlp.params')
        else:
            lr_scheduler.learning_rate *= 0.9

        print('epoch %d, lr %.2f, loss %.4f, train acc %.3f, test acc %.3f, best test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, lr_scheduler.learning_rate, train_l_sum / n, train_acc_sum / n, test_acc, best_test_acc,
                 time.time() - start))


start_time = time.clock()
lr, num_epochs = 1, 50
ctx = d2l.try_gpu()
# Initialize learning rate to lr
lr_scheduler = SimpleLRScheduler(learning_rate=lr)
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        optimizer_params={'lr_scheduler': lr_scheduler})
train(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
print("Time consume : ", time.clock() - start_time)
