import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import mxnet as mx


def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params=None, lr=None, trainer=None):
    """Train and evaluate a model"""
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X) * net(X)
                l = loss(y_hat, y).sum()
            l.backward()

            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def main():

    net = nn.Sequential()

    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(256, activation='relu'), nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
                            'learning_rate': 0.5})

    num_epochs = 5
    train(net, train_iter, test_iter, loss,
          num_epochs, batch_size, None,  None, trainer)


if __name__ == "__main__":
    main()
