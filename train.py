# 导入相关的包
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import MaxAbsScaler


# 数据集的路径
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
# 读取数据
sms = pd.read_csv(data_path, encoding='utf-8')
sms_pos = sms[(sms['label'] == 1)]
sms_neg = sms[(sms['label'] == 0)].sample(frac=1.0)[: len(sms_pos)]
sms = pd.concat([sms_pos, sms_neg], axis=0).sample(frac=1.0)


def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表
    """
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    return stopwords


# 停用词库路径
stopwords_path = 'scu_stopwords.txt'
# 读取停用词
stopwords = read_stopwords(stopwords_path)

# 构建训练集和测试集
X = np.array(sms.msg_new)
y = np.array(sms.label)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    # ('MaxAbsScaler', MaxAbsScaler()),
    ('clf', MultinomialNB())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'results/pipeline.model')

y_pred = pipeline.predict(X_test)

print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))
print('在测试集上的准确率：')
print(metrics.accuracy_score(y_test, y_pred))
