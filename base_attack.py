import numpy as np
import pandas as pd
import joblib
import requests
from io import BytesIO
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
def evaluate_attack(art_classifier, x_train, x_test, y_train, y_test, attack_method="Shadow"):
    """评估攻击效果"""
    try:
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)

        # 初始化攻击方法
        if attack_method == "RuleBased":
            attack = MembershipInferenceBlackBoxRuleBased(art_classifier)
            inferred_train = attack.infer(x_train[:attack_train_size], y_train[:attack_train_size])
            inferred_test = attack.infer(x_test[:attack_test_size], y_test[:attack_test_size])
        elif attack_method == "BlackBox":
            attack = MembershipInferenceBlackBox(art_classifier)
            # 黑箱攻击需要训练攻击模型
            attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                       x_test[:attack_test_size], y_test[:attack_test_size])
            inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
            inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
        elif attack_method == "Shadow":
            # 创建一个与原始模型相同类型的模型作为模板
            if isinstance(art_classifier, ScikitlearnRandomForestClassifier):
                shadow_model_template = sklearn.ensemble.RandomForestClassifier(
                    n_estimators=5, random_state=42
                )
            elif isinstance(art_classifier, ScikitlearnDecisionTreeClassifier):
                shadow_model_template = sklearn.tree.DecisionTreeClassifier(random_state=42)
            else:
                shadow_model_template = None

            # 尝试不同的参数组合
            try:
                # 尝试方法1：使用最新版本的参数
                shadow_models = ShadowModels(
                    classifier=art_classifier,
                    num_shadow_models=5,  # 减少影子模型数量以加快训练
                    random_state=42
                )
            except TypeError:
                try:
                    # 尝试方法2：使用旧版本的参数
                    shadow_models = ShadowModels(
                        classifier=art_classifier,
                        n_shadow_estimators=5,
                        random_state=42
                    )
                except TypeError:
                    # 尝试方法3：添加shadow_model_template参数
                    shadow_models = ShadowModels(
                        classifier=art_classifier,
                        n_shadow_estimators=5,
                        shadow_model_template=shadow_model_template,
                        random_state=42
                    )

            # 创建攻击实例
            attack = MembershipInferenceBlackBox(
                classifier=art_classifier,
                shadow_models=shadow_models
            )

            print("正在训练影子模型，这可能需要一些时间...")
            attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                       x_test[:attack_test_size], y_test[:attack_test_size])

            print("影子模型训练完成，开始推理...")
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
        attack_method = "Shadow"  # 修改这里：从元组改为字符串
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