from datetime import time
import json
import pandas as pd
import joblib
from pathlib import Path
import requests
from io import BytesIO

class DataLoader:
    def __init__(self):
        """
        初始化数据加载器
        """
        self.storage = Path("storage/old")

    def _load_uploaded(self, identifier: str):
        """
        加载上传的数据文件或远程数据
        Args:
            identifier: 本地文件路径或URL地址（支持http/https）
        """
        # 判断是否为URL
        if identifier.startswith(('http://', 'https://')):
            try:
                # 支持CSV格式的URL数据加载
                if identifier.endswith('.csv'):
                    return pd.read_csv(identifier)
                # 支持PKL格式的URL数据加载
                elif identifier.endswith('.pkl'):
                    response = requests.get(identifier)
                    return joblib.load(BytesIO(response.content))
                # 新增支持Excel格式的URL数据加载
                elif identifier.endswith(('.xlsx', '.xls')):
                    response = requests.get(identifier)
                    return pd.read_excel(BytesIO(response.content))
                else:
                    raise ValueError("远程URL仅支持.csv、.pkl和.xlsx/.xls格式")
            except Exception as e:
                raise ConnectionError(f"无法加载远程数据: {str(e)}") from e
        else:
            # 保持原有本地文件加载逻辑
            filepath = Path(identifier)
            if not filepath.exists():
                raise FileNotFoundError(f"本地文件 {identifier} 不存在")

            # 支持多种格式
            if filepath.suffix == ".csv":
                return pd.read_csv(filepath)
            elif filepath.suffix == ".pkl":
                return joblib.load(filepath)
            elif filepath.suffix in [".xlsx", ".xls"]:
                return pd.read_excel(filepath)  # 支持本地Excel文件
            else:
                raise ValueError("不支持的文件格式")

    def interactive_upload(self):
        """
        交互式上传数据集
        """
        print("请输入数据源（本地文件路径或远程URL地址）：")
        identifier = input("请输入路径或URL：")

        try:
            data = self._load_uploaded(identifier)
            print("数据加载成功！")
            return data
        except Exception as e:
            print(f"加载失败：{str(e)}")
            return None

    def archive_old_data(self, filename: str):
        """
        将处理前的数据移动到old目录
        Args:
            filename: 需要归档的文件名
        """
        old_path = Path("old") / f"{filename.split('.')[0]}_{int(time.time())}.bak"
        filepath = Path(filename)
        filepath.rename(old_path)

    def save_processed(self, data, filename: str, versioning=True):
        """
        保存处理后的数据
        Args:
            data: 需要保存的数据
            filename: 数据文件名
            versioning: 是否自动创建版本备份
        """
        save_path = Path("new") / filename

        if versioning and save_path.exists():
            self.archive_old_data(filename)

        if isinstance(data, pd.DataFrame):
            data.to_csv(save_path, index=False)
        else:
            joblib.dump(data, save_path)

        print(f"数据已保存至 {save_path}")
        return save_path

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.interactive_upload()
    if data is not None:
        print("加载的数据预览：")
        print(data.head())
