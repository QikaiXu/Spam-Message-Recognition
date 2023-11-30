# 垃圾短信识别

> 浙江大学《机器学习及其应用》作业，垃圾短信识别。
>
> 项目来源于：<https://mo.zju.edu.cn/workspace/5fc0eadb7ceb533cc49bce17?type=app&tab=2>（只有我自己的号能打开）

具体处理信息及说明查看 `main.ipynb`。

**注意先下载数据集 [sms_pub.csv.zip](https://wwtk.lanzoub.com/i3eFAk6r4cj) 并解压到 `Spam-Message-Recognition/datasets/5f9ae242cae5285cd734b91e-momodel/` 目录下**。

由于训练数据中正负样本不均衡（正负样本⽐例约为 1:10），将会导致拟合效果较差，因此读⼊后在负样本中随机取出⼀定数量作为实验⽤样本，使正负样本数量相同。

借助 sklearn 中的 `TfidfVectorizer` 来实现⽂本的向量化，之后用朴素贝叶斯进行分类。

最后在测试集上评估模型

- f1-score：0.9705720403793209 

- 准确率：0.969614655716993

