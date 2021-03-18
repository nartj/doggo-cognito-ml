import os

import cv2
import mxnet as mx
import numpy as np
import pandas as pd
from keras.preprocessing import image as tf_image
from keras_preprocessing.image import img_to_array
from mxnet import gluon, nd, image
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from tabulate import tabulate
from tensorflow_core.python.keras.models import load_model
from tqdm import tqdm


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


def transform_test(data):
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


def get_features_test(net1, net2, data, ctx):
    res_features = []
    inception_features = []

    feature1 = net1(data[0].as_in_context(ctx))
    res_features.append(feature1.asnumpy())

    feature2 = net2(data[1].as_in_context(ctx))
    inception_features.append(feature2.asnumpy())

    res_features = np.concatenate(res_features, axis=0)
    inception_features = np.concatenate(inception_features, axis=0)

    return res_features, inception_features


def test_models():
    ctx = mx.cpu()

    data_dir = 'data'
    models_dir = 'models'
    target_size = (128, 128, 1)

    dataset = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    dic = dict(zip(np.unique(dataset.breed), range(0, np.unique(dataset.breed).__len__() + 1)))

    net = build_model_mxnet(ctx)
    net.load_parameters(os.path.join(models_dir, 'model.params'))

    model = load_model(os.path.join(models_dir, 'dog-recognition.h5'))

    test_set = dataset.sample(20).reset_index()

    result = []

    res151 = models.resnet152_v1(pretrained=True, ctx=ctx).features
    with res151.name_scope():
        res151.add(gluon.nn.GlobalAvgPool2D())
    res151.collect_params().reset_ctx(ctx)
    res151.hybridize()

    inception = models.inception_v3(pretrained=True, ctx=ctx)
    inception_net = inception.features
    inception_net.collect_params().reset_ctx(ctx)
    inception_net.hybridize()

    for i in tqdm(range(20)):
        # -- Tensorflow
        img = tf_image.load_img(os.path.join(data_dir, 'train', test_set['id'][i]) + '.jpg', target_size=target_size,
                                grayscale=False)

        img = img_to_array(img)
        img = img / 255

        predict_tensorflow = model.predict_classes(np.array([img]))

        # -- MXNet

        img = mx.nd.array(cv2.imread(os.path.join(data_dir, 'train', test_set['id'][i]) + '.jpg'))
        img = transform_test(img)

        img_res151, img_inception = get_features_test(res151, inception_net, img, ctx)
        img_res151 = img_res151.reshape(img_res151.shape[:2])
        img_inception = img_inception.reshape(img_inception.shape[:2])

        img = nd.concat(mx.nd.array(img_inception), mx.nd.array(img_res151))

        predict_mx = nd.softmax(net(nd.array(img).as_in_context(ctx)))

        result.append({
            'id': test_set['id'][i],
            'expected': test_set['breed'][i],
            'tensor': list(dic.keys())[list(dic.values()).index(predict_tensorflow)],
            'mx': list(dic.keys())[list(dic.values()).index(predict_mx.topk(k=1).asnumpy()[0][0])],
            'mx_percentage': predict_mx[0, 0].asscalar()
        })
    print(tabulate(result))
    input("Press Enter to continue...")
