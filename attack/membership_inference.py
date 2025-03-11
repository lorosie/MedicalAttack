import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 数据加载和预处理
data_path = r"E:\PycharmProjects\PythonProject1\data\diabetes_processed2.csv"
data1 = pd.read_csv(data_path)  # 使用正确的分隔符

# 检查数据
print("数据形状:", data1.shape)
print("前5行数据:\n", data1.head())

# 分割特征和标签
X = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values

# 检查特征和标签的形状
print("X 的形状:", X.shape)
print("y 的形状:", y.shape)

# 数据分割
test_size = 0.2
random_state = 42
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# 加载模型
model_path = r"E:\网安\Attack(2)\Attack\models\随机森林模型\糖尿病\model_randomforest_10.pkl"
model = joblib.load(model_path)

# 加载 ONNX 模型
# model_path = r"\Download\牛头马面\Attack\models\随机森林模型\心脑血管\model2_randomforest_10.onnx"
# model = ort.InferenceSession(model_path)

# 封装为 ART 支持的分类器
art_classifier = ScikitlearnRandomForestClassifier(model)

# 计算模型准确率
train_accuracy = accuracy_score(y_train, model.predict(x_train))
test_accuracy = accuracy_score(y_test, model.predict(x_test))

# 创建基于规则的黑盒成员资格推断攻击对象
attack = MembershipInferenceBlackBoxRuleBased(art_classifier)

# 推断攻击特征
inferred_train = attack.infer(x_train, y_train)
inferred_test = attack.infer(x_test, y_test)

# 检查准确度
train_member_acc = np.mean(inferred_train)
test_nonmember_acc = 1 - np.mean(inferred_test)
overall_acc = (train_member_acc * len(inferred_train) + test_nonmember_acc * len(inferred_test)) / (
                len(inferred_train) + len(inferred_test))

# 计算ROC曲线下的AUC
y_true = np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))
y_pred = np.concatenate((inferred_train, inferred_test))
auc_score = roc_auc_score(y_true, y_pred)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_pred)

# 找到0.1%FPR下的TPR值
fpr_at_0_1 = 0.001
tpr_at_0_1 = np.interp(fpr_at_0_1, fpr, tpr)  # 线性插值找到0.1%FPR下的TPR

# 定义精确率/召回率计算函数
def calc_precision_recall(predicted, actual, positive_value=1):
    """改进版精确率/召回率计算（推荐直接使用sklearn内置函数）"""
    # 更稳健的实现方式
    tp = np.sum((predicted == positive_value) & (actual == positive_value))
    fp = np.sum((predicted == positive_value) & (actual != positive_value))
    fn = np.sum((predicted != positive_value) & (actual == positive_value))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

# 使用sklearn计算精确率和召回率
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)


# 输出结果
print(f"Model Training Accuracy: {train_accuracy:.4f}")
print(f"Model Testing Accuracy: {test_accuracy:.4f}")
print(f"Member Accuracy (TPR): {train_member_acc:.4f}")
print(f"Non-Member Accuracy (TNR): {test_nonmember_acc:.4f}")
print(f"Overall Attack Accuracy: {overall_acc:.4f}")
print(f"Precision (sklearn): {precision:.4f}")
print(f"Recall (sklearn): {recall:.4f}")
print(f"AUC: {auc_score:.4f}")
print(f"TPR@0.1%FPR: {tpr_at_0_1:.4f}")
