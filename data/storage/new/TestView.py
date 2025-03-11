
from io import StringIO
import requests
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path


# ================== 数据预处理 ==================
class DataProcessor:
    def __init__(self):
        self.preprocessors = {
            'cardio': self._preprocess_cardio,
            'diabetes': self._preprocess_diabetes
        }
        self.transformer = QuantileTransformer(output_distribution='normal', random_state=42)

    def _preprocess_cardio(self, df):
        """心血管数据集处理"""
        # 年龄转换（天→年）
        df['age'] = df['age'] // 365
        # 数值特征标准化
        num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        df[num_cols] = self.transformer.fit_transform(df[num_cols])
        # 分类特征编码
        cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        df.rename(columns={'cardio': 'target'}, inplace=True)
        return df

    def _preprocess_diabetes(self, df):
        """糖尿病数据集处理"""
        num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        df[num_cols] = self.transformer.fit_transform(df[num_cols])
        df.rename(columns={'Outcome': 'target'}, inplace=True)
        return df


# ================== 可视化验证 ==================
def plot_distributions(raw_df, processed_df, dataset_type):
    """分布对比可视化"""
    plt.figure(figsize=(15, 10))

    # 选择典型特征
    features = {
        'cardio': ['age', 'ap_hi', 'weight'],
        'diabetes': ['Glucose', 'BMI', 'Age']
    }[dataset_type]

    for i, col in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(raw_df[col], label='原始数据', fill=True, alpha=0.5)
        sns.kdeplot(processed_df[col], label='处理后数据', fill=True, alpha=0.5)
        plt.title(f"{col}分布对比")
        plt.legend()

    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{dataset_type}_distribution_comparison.png')
    plt.show()


def evaluate_model_performance(raw_df, processed_df, dataset_type):
    """模型性能对比"""
    # 数据准备
    target_col = 'cardio' if dataset_type == 'cardio' else 'Outcome'
    X_raw = raw_df.drop(target_col, axis=1)
    y_raw = raw_df[target_col]

    X_proc = processed_df.drop('target', axis=1)
    y_proc = processed_df['target']

    # 模型训练对比
    model = LogisticRegression(max_iter=1000, random_state=42)

    # 原始数据性能
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2)
    model.fit(X_train, y_train)
    raw_acc = accuracy_score(y_test, model.predict(X_test))
    raw_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # 处理后数据性能
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y_proc, test_size=0.2)
    model.fit(X_train, y_train)
    proc_acc = accuracy_score(y_test, model.predict(X_test))
    proc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # 结果展示
    result_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC'],
        '原始数据': [raw_acc, raw_auc],
        '处理后数据': [proc_acc, proc_auc]
    }).set_index('Metric')

    print(f"\n=== {dataset_type.upper()}模型性能对比 ===")
    print(result_df.round(3))
    return result_df


def plot_pca(processed_df):
    """PCA降维可视化"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(processed_df.drop('target', axis=1))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1],
                    hue=processed_df['target'], palette='coolwarm', alpha=0.6)
    plt.title("PCA降维可视化（处理后数据）")
    plt.savefig('pca_visualization.png')
    plt.show()


# ================== 主程序 ==================


# 修改 load_data_from_url 函数以支持 Excel 文件
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    if url.endswith('.xlsx'):
        return pd.read_excel(BytesIO(response.content))  # 使用 read_excel 处理 Excel 文件
    elif url.endswith('.csv'):
        return pd.read_csv(StringIO(response.text))  # 使用 read_csv 处理 CSV 文件
    else:
        raise ValueError("Unsupported file format. Only .csv and .xlsx are supported.")

if __name__ == "__main__":
    cardio_url = "https://attack-oss.oss-cn-chengdu.aliyuncs.com/excel/1741430335128_新CardioTrain.xlsx"
    diabetes_url = "https://attack-oss.oss-cn-chengdu.aliyuncs.com/excel/1739779683828_Diabetes.xlsx"

    # 初始化处理器
    processor = DataProcessor()

    # 处理心血管数据
    raw_cardio = load_data_from_url(cardio_url)
    processed_cardio = processor._preprocess_cardio(raw_cardio.copy())
    plot_distributions(raw_cardio, processed_cardio, 'cardio')
    evaluate_model_performance(raw_cardio, processed_cardio, 'cardio')

    # 处理糖尿病数据
    raw_diabetes = load_data_from_url(diabetes_url)
    processed_diabetes = processor._preprocess_diabetes(raw_diabetes.copy())
    plot_distributions(raw_diabetes, processed_diabetes, 'diabetes')
    evaluate_model_performance(raw_diabetes, processed_diabetes, 'diabetes')

    # 保存处理结果
    processed_cardio.to_csv('processed_cardio.csv', index=False)
    processed_diabetes.to_csv('processed_diabetes.csv', index=False)
    joblib.dump(processor.transformer, 'quantile_transformer.pkl')  # 保存预处理器

    print("\n 处理完成！查看生成图片：")
    print(list(Path().absolute().glob('*.png')))

