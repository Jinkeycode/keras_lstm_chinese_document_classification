# 下载代码
git clone https://github.com/Jinkeycode/keras_lstm_chinese_document_classification.git

# 安装依赖
pip install -r requirements.txt

# 二分类
标签形式为[0, 0, 1, 1] 代表分类的四个样本分别为第 0 类和第 1 类

具体代码在 `chinese_binary.py`

# 多分类
标签形式为[[1, 0, 0], [0, 1, 0], [0, 0, 1]] 代表分类的四个样本分别为第 0 类、第 1 类、第 2 类

具体代码在 `chinese_multi.py`

# 训练
取消掉这几行的注释即可
```
# 训练模型
# model.fit(text_processed, target, batch_size=512, epochs=10, )
# 保存模型
# model.save("health_and_tech.h5")
```

# 预测
sen = "*" 改为 自己的句子
