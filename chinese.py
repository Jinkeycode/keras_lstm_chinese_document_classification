# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

import tensorflow as tf
import jieba as jb
import numpy as np
import keras as krs


titles = []
print("正在加载健康类别的数据...")
with open("data/health.txt", "r") as f:
    for line in f.readlines():
        titles.append(line.strip())

print("正在加载科技类别的数据...")
with open("data/tech.txt", "r") as f:
    for line in f.readlines():
        titles.append(line.strip())

print("一共加载了 %s 个标题" %len(titles))

arr0 = np.zeros(shape=[12000, ])
arr1 = np.ones(shape=[12000, ])
target = np.hstack([arr0, arr1])
print("一共加载了 %s 个标签" %target.shape)

max_sequence_length = 30
embedding_size = 50
rnn_size = 10

# 标题分词
titles = [".".join(jb.cut(t, cut_all=True)) for t in titles]

# word2vec 词袋化
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=1)
text_processed = np.array(list(vocab_processor.fit_transform(titles)))

# 读取标签
dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(dict.items(), key = lambda x : x[1])

# 配置网络结构
model = krs.Sequential()
model.add(krs.layers.Embedding(len(dict.items()), embedding_size,input_length=max_sequence_length))
model.add(krs.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(krs.layers.Dense(1))
model.add(krs.layers.Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# 训练模型
# model.fit(text_processed, target, batch_size=512, epochs=10, )
# 保存模型
# model.save("health_and_tech.h5")

# 加载预训练的模型
model.load_weights("health_and_tech.h5")

# 预测样本
sen = "中消协约谈摩拜、ofo等7家共享单车企业 建议尽可能免收押金租车"
sen_prosessed = " ".join(jb.cut(sen, cut_all=True))
sen_prosessed = vocab_processor.transform([sen_prosessed])
sen_prosessed = np.array(list(sen_prosessed))
result = model.predict(sen_prosessed)

print(result)
