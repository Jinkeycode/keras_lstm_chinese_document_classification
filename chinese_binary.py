# -*- coding: utf-8 -*-
# Created by Jinkey on 2017/12/05.
__author__ = 'Jinkey'

import tensorflow as tf
import jieba as jb
import numpy as np
import utils

titles = utils.load_data(catalogue=utils.BINARY_FLAG)
target = utils.load_label(catalogue=utils.BINARY_FLAG)

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
model = utils.build_netword(catalogue=utils.BINARY_FLAG, dict=dict, embedding_size=embedding_size, max_sequence_length=max_sequence_length)

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

if result == 0:
    print("这是一篇关于健康的文章")
else:
    print("这是一篇关于科技的文章")