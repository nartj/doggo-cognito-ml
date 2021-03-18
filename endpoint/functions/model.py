import json
import os

import mxnet as mx
import numpy as np
from mxnet import gluon, nd, image
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

ctx = mx.cpu()

res151 = models.resnet152_v1(pretrained=True, ctx=ctx).features
with res151.name_scope():
    res151.add(gluon.nn.GlobalAvgPool2D())
res151.collect_params().reset_ctx(ctx)
res151.hybridize()

inception = models.inception_v3(pretrained=True, ctx=ctx)
inception_net = inception.features
inception_net.collect_params().reset_ctx(ctx)
inception_net.hybridize()

with open(os.path.join('functions', 'labels.json'), 'r') as fp:
    dic = json.load(fp)


def build_model_mxnet(ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.BatchNorm())
        net.add(nn.Dense(1024))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Dense(512))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Dense(120))
    net.initialize(ctx=ctx)
    return net


def transform(data):
    im1 = image.imresize(data.astype('float32') / 255, 288, 288)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224),
                                     resize=0,
                                     mean=np.array([0.485, 0.456, 0.406]),
                                     std=np.array([0.229, 0.224, 0.225]))

    im2 = image.imresize(data.astype('float32') / 255, 363, 363)
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299),
                                     resize=0,
                                     mean=np.array([0.485, 0.456, 0.406]),
                                     std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist1:
        im1 = aug(im1)
    im1 = nd.transpose(im1, (2, 0, 1))

    for aug in auglist2:
        im2 = aug(im2)
    im2 = nd.transpose(im2, (2, 0, 1))

    return im1.expand_dims(axis=0), im2.expand_dims(axis=0)


def get_features(net1, net2, data):
    res_features = []
    inception_features = []

    feature1 = net1(data[0].as_in_context(ctx))
    res_features.append(feature1.asnumpy())

    feature2 = net2(data[1].as_in_context(ctx))
    inception_features.append(feature2.asnumpy())

    res_features = np.concatenate(res_features, axis=0)
    inception_features = np.concatenate(inception_features, axis=0)

    return res_features, inception_features


net = build_model_mxnet(ctx)
net.load_parameters(os.path.join('functions', 'model.params'))


def process(img):
    img = transform(mx.nd.array(img))

    img_res151, img_inception = get_features(res151, inception_net, img)
    img_res151 = img_res151.reshape(img_res151.shape[:2])
    img_inception = img_inception.reshape(img_inception.shape[:2])

    img = nd.concat(mx.nd.array(img_inception), mx.nd.array(img_res151))

    predict_mx = nd.softmax(net(nd.array(img).as_in_context(ctx)))

    result = []

    for i in predict_mx.topk(k=3).asnumpy().flatten():
        result.append({
            'label': list(dic.keys())[list(dic.values()).index(i)],
            'probability': "{:.2f}".format(predict_mx[0][int(i)].asscalar() * 100)
        })

    return json.dumps(result)
