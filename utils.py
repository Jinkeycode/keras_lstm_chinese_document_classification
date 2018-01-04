# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras as krs

BINARY_FLAG = "binary"
MULTI_FLAG = "multi"

def load_data(catalogue=BINARY_FLAG):
    titles = []
    print("正在加载健康类别的数据...")
    with open("data/health.txt", "r") as f:
        for line in f.readlines():
            titles.append(line.strip())

    print("正在加载科技类别的数据...")
    with open("data/tech.txt", "r") as f:
        for line in f.readlines():
            titles.append(line.strip())

    if catalogue == MULTI_FLAG:
        print("正在加载设计类别的数据...")
        with open("data/design.txt", "r") as f:
            for line in f.readlines():
                titles.append(line.strip())

    print("一共加载了 %s 个标题" % len(titles))

    return titles


def load_label(catalogue=BINARY_FLAG):
    if catalogue == BINARY_FLAG:
        arr0 = np.zeros(shape=[12000, ])
        arr1 = np.ones(shape=[12000, ])
        target = np.hstack([arr0, arr1])
        print("一共加载了 %s 个标签" % target.shape)
        return target
    elif catalogue == MULTI_FLAG:
        arr0 = np.zeros(shape=[12000, ])
        arr1 = np.ones(shape=[12000, ])
        arr2 = np.array([2]).repeat(7318)
        target = np.hstack([arr0, arr1, arr2])
        print("一共加载了 %s 个标签" % target.shape)

        encoder = LabelEncoder()
        encoder.fit(target)
        encoded_target = encoder.transform(target)
        dummy_target = krs.utils.np_utils.to_categorical(encoded_target)

        return dummy_target


def build_netword(dict, catalogue=BINARY_FLAG, embedding_size=50, max_sequence_length=30):
    if catalogue == BINARY_FLAG:
        # 配置网络结构
        model = krs.Sequential()
        model.add(krs.layers.Embedding(len(dict.items()), embedding_size, input_length=max_sequence_length))
        model.add(krs.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        model.add(krs.layers.Dense(1))
        model.add(krs.layers.Activation("sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    elif catalogue == MULTI_FLAG:
        # 配置网络结构
        model = krs.Sequential()
        model.add(krs.layers.Embedding(len(dict.items()), embedding_size, input_length=max_sequence_length))
        model.add(krs.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        model.add(krs.layers.Dense(3))
        model.add(krs.layers.Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model