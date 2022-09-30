from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def model_evaluate(X_test, y_test, model):
    actual = y_test.argmax(axis=1)  # 真实的类别标签（将one-hot)标签逆向
    predict_x = model.predict(X_test)  # 预测标签
    predicted = np.argmax(np.array(predict_x), axis=1)  # one-hot编码逆向

    # 计算总的精度
    acc = accuracy_score(actual, predicted)
    print("准确率： ", acc)
    # 计算混淆矩阵
    print("混淆矩阵：")
    print(confusion_matrix(actual, predicted))
    # 计算 precision_score
    print("precision_score:", end="  ")
    print(precision_score(actual, predicted, average="binary", pos_label=1))  # pos_label设置为1，代表标签为1的样本是正例，标签为2的样本是负例。
    #	recall
    print("recall: ", end=" ")
    print(recall_score(actual, predicted, average="binary", pos_label=1))
    #	F1
    print("f1 score: ", end=" ")
    print(f1_score(actual, predicted, average="binary", pos_label=1))
