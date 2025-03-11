from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from load_selector import DataLoader


def preprocess_cardio(df):
    """心血管数据集预处理"""
    # 1. 年龄转换（天转年）
    df['age'] = df['age'] // 365

    # 2. 数值特征标准化
    num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    qt = QuantileTransformer(output_distribution='normal', random_state=0)  # [正态分布转换]
    df[num_cols] = qt.fit_transform(df[num_cols])

    # 3. 分类特征编码
    cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)  # [独热编码]

    # 4. 目标变量处理
    df.rename(columns={'cardio': 'target'}, inplace=True)
    return df


def preprocess_diabetes(df):
    """糖尿病数据集预处理"""
    # 数值特征标准化
    num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    qt = QuantileTransformer(output_distribution='normal', random_state=0)  #
    df[num_cols] = qt.fit_transform(df[num_cols])

    # 目标变量处理
    df.rename(columns={'Outcome': 'target'}, inplace=True)
    return df


class TabDDPMProcessor(DataLoader):
    def __init__(self):
        super().__init__()
        self.preprocessors = {
            'cardio': preprocess_cardio,
            'diabetes': preprocess_diabetes
        }

    def process_dataset(self, dataset_type):
        """
        完整处理流程
        :param dataset_type: 数据集类型 (cardio/diabetes)
        """
        # 加载原始数据
        raw_data = self.interactive_upload()

        # 执行预处理
        processor = self.preprocessors[dataset_type]
        processed_data = processor(raw_data)

        # 数据分割 (训练集80%, 验证集10%, 测试集10%)
        train, test = train_test_split(processed_data, test_size=0.2, random_state=42)
        val, test = train_test_split(test, test_size=0.5, random_state=42)

        # 保存处理结果
        self._save_dataset(train, f"{dataset_type}_train.csv")
        self._save_dataset(val, f"{dataset_type}_val.csv")
        self._save_dataset(test, f"{dataset_type}_test.csv")

        print(f"{dataset_type}数据集处理完成！")

    def _save_dataset(self, data, filename):
        """保存处理后的数据集"""
        save_path = self.storage / "processed" / filename
        save_path.parent.mkdir(exist_ok=True)
        data.to_csv(save_path, index=False)


# 使用示例
if __name__ == "__main__":
    processor = TabDDPMProcessor()

    # 处理心血管数据集
    print("=== 开始处理心血管数据集 ===")
    processor.process_dataset('cardio')

    # 处理糖尿病数据集
    print("\n=== 开始处理糖尿病数据集 ===")
    processor.process_dataset('diabetes')
