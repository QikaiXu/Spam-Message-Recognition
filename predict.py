import warnings
warnings.filterwarnings('ignore')
from sklearn.externals import joblib

pipeline_path = 'results/pipeline.model'
pipeline = joblib.load(pipeline_path)


def predict(message):
    """
    预测短信短信的类别和每个类别的概率
    param: message: 经过jieba分词的短信，如"医生 拿 着 我 的 报告单 说 ： 幸亏 你 来 的 早 啊"
    return: label: 整数类型，短信的类别，0 代表正常，1 代表恶意
            proba: 列表类型，短信属于每个类别的概率，如[0.3, 0.7]，认为短信属于 0 的概率为 0.3，属于 1 的概率为 0.7
    """
    label = pipeline.predict([message])[0]
    proba = list(pipeline.predict_proba([message])[0])

    return label, proba


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    data_eval = pd.read_csv('sms_eval.csv', encoding='utf8')
    y_eval = np.array(data_eval['label'])
    X_eval = np.array(data_eval['msg_new'])
    total = y_eval.shape[0]
    count = 0
    for x, y in zip(X_eval, y_eval):
        y_pred, _ = predict(x)
        if y_pred == y:
            count += 1
    print('{} / {}'.format(count, total))
