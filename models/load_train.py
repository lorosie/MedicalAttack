import io  # 添加导入
import joblib
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from art.estimators.classification import SklearnClassifier, PyTorchClassifier


class ModelManager:
    def __init__(self, dataset_name: str):
        """
        模型生命周期管理系统
        :param dataset_name: 数据集标识 (cardio/diabetes)
        """
        self.dataset = dataset_name
        self.supported_models = ['tree', 'forest', 'nn']
        self.model_dir = Path("model_store") / dataset_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self, data_source: str, target_column='target'):
        """
        根据数据源加载数据集
        :param data_source: 数据集来源 (本地文件路径或URL)
        :param target_column: 目标列名称，默认为 'target'
        :return: 特征和标签
        """
        if urlparse(data_source).scheme in ('http', 'https'):
            import requests
            response = requests.get(data_source, timeout=30)
            response.raise_for_status()
            # 检查文件扩展名以确定使用哪种方法读取数据
            if data_source.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(io.BytesIO(response.content))
            else:
                data = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
        else:
            # 检查文件扩展名以确定使用哪种方法读取数据
            if data_source.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(data_source)
            else:
                data = pd.read_csv(data_source, on_bad_lines='skip')

        print("数据集列名:", data.columns)  # 打印列名以确认目标列

        if target_column not in data.columns:
            raise KeyError(f"数据集中没有名为 '{target_column}' 的列。请检查数据集并确认目标列名称。")

        X, y = data.drop(target_column, axis=1), data[target_column]
        return X, y





    def load_model(self, model_path: str):
        """
        通用模型加载方法
        :param model_path: 本地路径或URL
        :return: ART兼容的模型包装器
        """
        print(f"加载模型111 {model_path}")
        # 处理远程模型
        if urlparse(model_path).scheme in ('http', 'https'):
            print(f"加载模型2 {model_path}")
            return model_path



    def _get_input_size(self):
        """根据数据集确定输入大小"""
        return 12 if self.dataset == 'cardio' else 8

    @staticmethod
    def _get_format(model_obj) -> str:
        """自动判断保存格式"""
        if isinstance(model_obj, (DecisionTreeClassifier, RandomForestClassifier)):
            return 'pkl'
        elif isinstance(model_obj, dict):
            return 'pt'
        else:
            raise TypeError("未知的模型对象类型")


# 使用示例
if __name__ == "__main__":
    # 初始化管理器
    mgr = ModelManager('cardio')





    # 加载远程模型
    remote_model = mgr.load_model("https://attack-oss.oss-cn-chengdu.aliyuncs.com/models/1741431442047_decision_tree_model.onnx")
