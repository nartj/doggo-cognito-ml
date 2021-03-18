import os
from glob import glob

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn
from mxnet.ndarray import softmax_cross_entropy
from tqdm import tqdm


def build_model(ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.BatchNorm())
        net.add(nn.Dense(1024))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        #         net.add(nn.Dropout(0.5))
        net.add(nn.Dense(512))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        #         net.add(nn.Dropout(0.5))
        net.add(nn.Dense(120))
    net.initialize(ctx=ctx)
    return net


def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()


def evaluate(net, data_iter, ctx):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss / steps, acc / steps


def create_model(gpu):
    if gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()

    data_dir = 'data'
    batch_size = 128
    learning_rate = 1e-3
    epochs = int(input("Epoch? "))
    lr_decay = 0.95
    lr_decay2 = 0.8
    lr_period = 100
    models_dir = "models"

    synset = np.unique(pd.read_csv(os.path.join(data_dir, 'labels.csv')).breed).tolist()
    n = len(glob(os.path.join('.', data_dir, 'Images', '*', '*.jpg')))

    y = nd.zeros((n,))
    print('Aggregating labels')
    for i, file_name in tqdm(enumerate(glob(os.path.join('.', data_dir, 'Images', '*', '*.jpg'))), total=n):
        y[i] = synset.index(file_name.split('/')[3][10:].lower())
        nd.waitall()

    features = [nd.load(os.path.join(models_dir, 'features_incep.nd'))[0], \
                nd.load(os.path.join(models_dir, 'features_res.nd'))[0]]
    features = nd.concat(*features, dim=1)

    data_iter_train = gluon.data.DataLoader(gluon.data.ArrayDataset(features, y), batch_size, shuffle=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    net = build_model(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

    print('Starting training')
    for epoch in range(epochs):
        if epoch <= lr_period:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        else:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay2)
        train_loss = 0.
        train_acc = 0.
        steps = len(data_iter_train)
        for data, label in data_iter_train:
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        val_loss, val_acc = evaluate(net, data_iter_train, ctx)

        print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
            epoch + 1, train_loss / steps, train_acc / steps * 100, val_loss, val_acc * 100))

    print('Model saved under ' + models_dir + '/model' + str(epochs) + '.params')
    net.save_parameters(os.path.join(models_dir, 'model' + str(epochs) + '.params'))
    input("Press Enter to continue...")
