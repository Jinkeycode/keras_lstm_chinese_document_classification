# -*- coding: utf-8 -*-
# Created by Jinkey on 2018/1/4.
__author__ = 'Jinkey'

import tensorflow as tf
import jieba as jb
import numpy as np

import utils

titles = utils.load_data(catalogue=utils.MULTI_FLAG)
target = utils.load_label(catalogue=utils.MULTI_FLAG)

max_sequence_length = 30
embedding_size = 50

# 标题分词
titles = [".".join(jb.cut(t, cut_all=True)) for t in titles]

# word2vec 词袋化
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=1)
text_processed = np.array(list(vocab_processor.fit_transform(titles)))

# 读取标签
dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(dict.items(), key = lambda x : x[1])

# 配置网络结构
model = utils.build_netword(catalogue=utils.MULTI_FLAG, dict=dict, embedding_size=embedding_size, max_sequence_length=max_sequence_length)


# 训练模型
# model.fit(text_processed, target, batch_size=512, epochs=10, )
# 保存模型
# model.save("health_and_tech_design.h5")

# 加载预训练的模型
model.load_weights("health_and_tech_design.h5")

# 预测样本
sen = "做好商业设计需要学习的小技巧"
sen_prosessed = " ".join(jb.cut(sen, cut_all=True))
sen_prosessed = vocab_processor.transform([sen_prosessed])
sen_prosessed = np.array(list(sen_prosessed))
result = model.predict(sen_prosessed)

catalogue = list(result[0]).index(max(result[0]))
if max(result[0]) > 0.8:
    if catalogue == 0:
        print("这是一篇关于健康的文章")
    elif catalogue == 1:
        print("这是一篇关于科技的文章")
    elif catalogue == 2:
        print("这是一篇关于设计的文章")
    else:
        print("这篇文章没有可信分类")