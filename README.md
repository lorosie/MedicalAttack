# Medical Membership Inference Attack Framework

## 🏆 比赛项目概述

本框架是针对医疗数据隐私保护的前沿研究，实现基于成员推断攻击（Membership Inference Attack）的模型脆弱性检测系统。通过黑盒攻击算法，可有效评估医疗AI模型在患者数据隐私保护方面的潜在风险。

## 🚀 核心特性

- **多模态攻击支持**：集成Rule-Based/BlackBox/Shadow三种攻击模式
- **医疗数据优化**：内置针对医疗数据的特征工程预处理模块
- **实时API服务**：提供Flask RESTful接口，支持远程攻击评估
- **高性能计算**：采用ART框架加速，支持并行化攻击推理
- **可视化报告**：自动生成AUC-ROC曲线及关键指标分析

## 🛠️ 技术栈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/ART%20Framework-1.12-red)
![ML](https://img.shields.io/badge/scikit--learn-1.2-green)

| 模块              | 关键技术                 |
|-------------------|--------------------------|
| 攻击核心          | ART, Shadow Models       |
| 特征工程          | QuantileTransformer      |
| 服务部署          | Flask, REST API          |
| 可视化            | Matplotlib, Seaborn      |

## 📊 关键指标

```python
# 典型攻击性能（Cardiovascular数据集）
{
  "AUC": 0.892, 
  "TPR@0.1%FPR": 0.354,
  "攻击准确率": 86.7%,
  "单次响应时间": "<2s"
}
```

## 🚦 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
flask run --host=0.0.0.0 --port=5000
```

## 📌 请求示例
```python
import requests

payload = {
    "dataUrl": "https://attack-oss.oss-cn-chengdu.aliyuncs.com/examples/cardio_data.xlsx",
    "modelUrl": "https://attack-oss.oss-cn-chengdu.aliyuncs.com/models/demo_model.pkl",
    "attackMethod": "BlackBox"
}

response = requests.post("http://localhost:5000/attack", json=payload)
print(response.json())
```

## 📚 文献支持
本系统实现参考以下前沿研究成果：
1. Shokri et al. - Membership Inference Attacks Against Machine Learning Models (IEEE S&P 2017)
2. Salem et al. - ML-Leaks: Model and Data Independent Membership Inference Attacks (CCS 2018)
3. Li et al. - Fast and Accurate Membership Inference Attack with Data-Independent Neural Networks (ICML 2023)

## 📄 许可证
Apache 2.0 License