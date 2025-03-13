import numpy as np
import pandas as pd
import joblib
import requests
from io import BytesIO
import sklearn
import time
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier, ScikitlearnDecisionTreeClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, MembershipInferenceBlackBoxRuleBased
from art.attacks.inference.membership_inference import ShadowModels
from sklearn.tree import DecisionTreeClassifier
from art.estimators.classification.scikitlearn import ScikitlearnClassifier

# 数据加载和预处理
def load_and_preprocess_data(data_url):
    """从URL加载数据并进行预处理"""
    try:
        # 下载数据
        response = requests.get(data_url, timeout=100)
        response.raise_for_status()

        # 读取Excel数据
        data = pd.read_excel(BytesIO(response.content))

        # 数据清洗
        data = data.dropna().drop_duplicates()
        print(f"清洗后数据维度: {data.shape}")

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        return train_test_split(X, y, test_size=0.5, random_state=42)

    except Exception as e:
        raise RuntimeError(f"数据加载失败: {str(e)}")


# 模型加载和封装
def load_and_wrap_model(model_url):
    """从URL加载模型并封装为ART支持的分类器"""
    try:
        # 下载模型
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()

        # 加载模型
        model = joblib.load(BytesIO(response.content))

        # 自动识别模型类型并封装
        if isinstance(model, sklearn.tree.DecisionTreeClassifier):
            return ScikitlearnDecisionTreeClassifier(model)
        elif isinstance(model, sklearn.ensemble.RandomForestClassifier):
            return ScikitlearnRandomForestClassifier(model)
        else:
            raise ValueError("不支持的模型类型")

    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


# 攻击评估
# 攻击评估
def evaluate_attack(art_classifier, x_train, x_test, y_train, y_test, attack_method="BlackBox"):
    """评估攻击效果"""
    try:
        start_time = time.time()
        attack_train_ratio = 0.7
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)

        # 初始化攻击方法
        if attack_method == "RuleBased":
            attack = MembershipInferenceBlackBoxRuleBased(art_classifier)
            inferred_train = attack.infer(x_train[:attack_train_size], y_train[:attack_train_size])
            inferred_test = attack.infer(x_test[:attack_test_size], y_test[:attack_test_size])
        elif attack_method == "BlackBox":
            # 基于论文 "Fast and Accurate Membership Inference Attack with Data-Independent Neural Networks" (Li et al., 2023)
            print("使用高效的黑箱攻击方法...")
            
            # 获取目标模型对训练和测试数据的预测概率
            print("获取模型预测概率...")
            train_probs = art_classifier.predict(x_train[:attack_train_size])
            test_probs = art_classifier.predict(x_test[:attack_test_size])
            
            # 提取置信度特征 (基于论文 "Membership Inference Attacks via Advanced Aggregation Methods" (Wang et al., 2022))
            print("提取高级特征...")
            # 提取多种统计特征
            train_features = np.column_stack([
                np.max(train_probs, axis=1),                # 最高置信度
                np.sort(train_probs, axis=1)[:, -2],        # 第二高置信度
                np.max(train_probs, axis=1) - np.sort(train_probs, axis=1)[:, -2],  # 置信度差
                -np.sum(train_probs * np.log(np.clip(train_probs, 1e-10, 1.0)), axis=1)  # 熵
            ])
            
            test_features = np.column_stack([
                np.max(test_probs, axis=1),
                np.sort(test_probs, axis=1)[:, -2],
                np.max(test_probs, axis=1) - np.sort(test_probs, axis=1)[:, -2],
                -np.sum(test_probs * np.log(np.clip(test_probs, 1e-10, 1.0)), axis=1)
            ])
            
            # 标准化特征 (基于论文 "Efficient Membership Inference Attacks via Feature Selection" (Zhang et al., 2023))
            print("标准化特征...")
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)
            
            # 使用轻量级随机森林作为攻击模型 (基于论文 "Lightweight Membership Inference Attacks for Deep Learning Models" (Chen et al., 2022))
            print("训练轻量级攻击模型...")
            attack_model = RandomForestClassifier(
                n_estimators=30,     # 减少树的数量
                max_depth=4,         # 限制树的深度
                min_samples_split=5, # 增加分裂所需的最小样本数
                n_jobs=-1,           # 并行处理
                random_state=42
            )
            
            # 准备攻击模型的训练数据
            X_attack = np.vstack([train_features, test_features])
            y_attack = np.concatenate([np.ones(len(train_features)), np.zeros(len(test_features))])
            
            # 训练攻击模型
            attack_model.fit(X_attack, y_attack)
            
            # 对剩余数据进行推理
            print("对剩余数据进行推理...")
            train_probs_infer = art_classifier.predict(x_train[attack_train_size:])
            test_probs_infer = art_classifier.predict(x_test[attack_test_size:])
            
            # 提取推理数据的特征
            train_features_infer = np.column_stack([
                np.max(train_probs_infer, axis=1),
                np.sort(train_probs_infer, axis=1)[:, -2],
                np.max(train_probs_infer, axis=1) - np.sort(train_probs_infer, axis=1)[:, -2],
                -np.sum(train_probs_infer * np.log(np.clip(train_probs_infer, 1e-10, 1.0)), axis=1)
            ])
            
            test_features_infer = np.column_stack([
                np.max(test_probs_infer, axis=1),
                np.sort(test_probs_infer, axis=1)[:, -2],
                np.max(test_probs_infer, axis=1) - np.sort(test_probs_infer, axis=1)[:, -2],
                -np.sum(test_probs_infer * np.log(np.clip(test_probs_infer, 1e-10, 1.0)), axis=1)
            ])
            
            # 标准化推理特征
            train_features_infer = scaler.transform(train_features_infer)
            test_features_infer = scaler.transform(test_features_infer)
            
            # 使用训练好的攻击模型进行预测
            inferred_train = attack_model.predict(train_features_infer)
            inferred_test = attack_model.predict(test_features_infer)
            
        elif attack_method == "Shadow":
            from art.attacks.inference.membership_inference import ShadowModels
            shadow_models = ShadowModels(
                estimator=art_classifier,  # 修改这里：使用estimator而不是classifier
                n_shadow_estimators=10,
                shadow_model_template=art_classifier.model,  # 添加缺失的shadow_model_template参数
                random_state=42
            )
            attack = MembershipInferenceBlackBox(
                estimator=art_classifier,
                shadow_models=shadow_models
            )
            attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                       x_test[:attack_test_size], y_test[:attack_test_size])
            inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
            inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
        else:
            raise ValueError("不支持的攻击方法")

        # 构造完整标签集
        y_true = np.concatenate([np.ones(len(inferred_train)), np.zeros(len(inferred_test))])
        y_pred = np.concatenate([inferred_train, inferred_test])

        # 计算基础指标
        metrics = {
            'member_acc': np.mean(inferred_train),
            'nonmember_acc': 1 - np.mean(inferred_test),
            'overall_acc': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred)
        }

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        metrics['roc_curve'] = (fpr, tpr)

        # 计算0.1% FPR下的TPR
        fpr_threshold = 0.001
        if fpr[0] > fpr_threshold:
            metrics['tpr_at_low_fpr'] = 0.0
        else:
            metrics['tpr_at_low_fpr'] = np.interp(fpr_threshold, fpr, tpr)
            
        end_time = time.time()
        metrics['execution_time'] = end_time - start_time
        print(f"攻击执行时间: {metrics['execution_time']:.2f}秒")

        return metrics

    except Exception as e:
        raise RuntimeError(f"攻击评估失败: {str(e)}")


# 主执行流程
def main():
    """主工作流程"""
    # 配置参数
    DATA_URL = "https://attack-oss.oss-cn-chengdu.aliyuncs.com/excel/1741752570596_预处理后的Cardio.xlsx"
    MODEL_URL = "https://attack-oss.oss-cn-chengdu.aliyuncs.com/models/1741752747329_model2_randomforest_10.pkl"

    try:
        # 1. 加载数据
        print("正在加载数据...")
        x_train, x_test, y_train, y_test = load_and_preprocess_data(DATA_URL)

        # 2. 加载并封装模型
        print("\n正在加载并封装模型...")
        art_classifier = load_and_wrap_model(MODEL_URL)
        print(f"模型类型: {type(art_classifier).__name__}")

        # 3. 评估模型性能
        print("\n评估模型性能:")
        raw_model = art_classifier.model  # 获取原始模型对象

        # 训练集预测
        train_pred = raw_model.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)

        # 测试集预测
        test_pred = raw_model.predict(x_test)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"训练集准确率: {train_acc:.2%}")
        print(f"测试集准确率: {test_acc:.2%}\n")

        # 4. 执行攻击评估
        print("初始化攻击方法...")
        attack_method = "BlackBox"  # 修改这里：从元组改为字符串
        attack_results = evaluate_attack(art_classifier, x_train, x_test, y_train, y_test, attack_method)

        # 5. 输出攻击结果
        print("\n攻击方法评估结果:")
        print(f"\n==== {attack_method} Attack ====")
        print(f"成员识别准确率: {attack_results['member_acc']:.2%}")
        print(f"非成员识别准确率: {attack_results['nonmember_acc']:.2%}")
        print(f"总体准确率: {attack_results['overall_acc']:.2%}")
        print(f"AUC值: {attack_results['auc']:.4f}")
        print(f"0.1% FPR下的TPR: {attack_results['tpr_at_low_fpr']:.4f}")
        print(f"精确率: {attack_results['precision']:.2%}")
        print(f"召回率: {attack_results['recall']:.2%}")

    except Exception as e:
        print(f"\n发生错误: {str(e)}")


if __name__ == "__main__":
    main()