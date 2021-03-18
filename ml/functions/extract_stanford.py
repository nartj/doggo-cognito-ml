import os
from glob import glob

import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision as models
from tqdm import tqdm


def extract_stanford(gpu):
    if gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()

    data_dir = "data"
    models_dir = "models"
    imageSize_resnet = 288
    imageSize_inception = 363
    n = len(glob(os.path.join('.', data_dir, "Images", "*", "*.jpg")))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("Using ResNet")
    net = models.get_model('resnet152_v1', pretrained=True, ctx=ctx)
    features = []

    for j in tqdm(range(0, 161)):
        i = 0
        temp = nd.zeros((128, 3, imageSize_resnet, imageSize_resnet))
        for file_name in glob(os.path.join(data_dir, "Images", "*", "*.jpg"))[128 * j:128 * (j + 1)]:
            img = cv2.imread(file_name)
            img_224 = ((cv2.resize(img, (imageSize_resnet, imageSize_resnet))[:, :, ::-1] \
                        / 255.0 - mean) / std).transpose((2, 0, 1))
            temp[i] = nd.array(img_224)
            nd.waitall()
            i += 1
        if j == 160:
            temp = temp[0:100]
        data_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(temp), batch_size=128)
        for data in data_iter_224:
            feature = net.features(data.as_in_context(ctx))
            feature = gluon.nn.Flatten()(feature)
            features.append(feature.as_in_context(mx.cpu()))
        nd.waitall()
    features = nd.concat(*features, dim=0)
    print(features.shape)
    nd.save(os.path.join(models_dir, 'features_res.nd'), features)

    input("Press Enter to continue...")


    print('Using Inception')
    net = models.get_model('inceptionv3', pretrained=True, ctx=ctx)
    features = []
    for j in tqdm(range(0, 161)):
        i = 0
        temp = nd.zeros((128, 3, imageSize_inception, imageSize_inception))
        for file_name in glob(os.path.join(data_dir, "Images", "*", "*.jpg"))[128 * j:128 * (j + 1)]:
            img = cv2.imread(file_name)
            img_299 = ((cv2.resize(img, (imageSize_inception, imageSize_inception))[:, :, ::-1] \
                        / 255.0 - mean) / std).transpose((2, 0, 1))
            temp[i] = nd.array(img_299)
            nd.waitall()
            i += 1
        if j == 160:
            temp = temp[0:100]
        data_iter_299 = gluon.data.DataLoader(gluon.data.ArrayDataset(temp), batch_size=128)
        for data in data_iter_299:
            feature = net.features(data.as_in_context(ctx))
            feature = gluon.nn.Flatten()(feature)
            features.append(feature.as_in_context(mx.cpu()))
        nd.waitall()
    features = nd.concat(*features, dim=0)
    print(features.shape)
    nd.save(os.path.join(models_dir, 'features_incep.nd'), features)

    input("Press Enter to continue...")
