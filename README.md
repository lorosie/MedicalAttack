# Medical Membership Inference Attack Framework

## 🏆 比赛项目概述

本框架是针对医疗数据隐私保护的前沿研究，实现基于成员推断攻击（Membership Inference Attack）的模型脆弱性检测系统。通过多种攻击算法，可有效评估医疗AI模型在患者数据隐私保护方面的潜在风险。

## 🚀 核心特性

- **多模态攻击支持**：集成Rule-Based/BlackBox/Shadow三种攻击模式，支持不同场景下的隐私风险评估
- **医疗数据优化**：内置针对医疗数据的特征工程预处理模块，支持多种医疗数据格式
- **实时API服务**：提供Flask RESTful接口，支持远程攻击评估和批量处理
- **高性能计算**：采用并行化特征提取和轻量级随机森林，实现高效推理
- **可视化报告**：自动生成AUC-ROC曲线及全面的性能指标分析

## 🛠️ 技术栈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/ART%20Framework-1.12-red)
![ML](https://img.shields.io/badge/scikit--learn-1.2-green)

| 模块              | 关键技术                 |
|-------------------|---------------------------|
| 攻击核心          | ART, Random Forest        |
| 特征工程          | StandardScaler            |
| 服务部署          | Flask, REST API           |
| 可视化            | Matplotlib, Seaborn       |

## 📊 关键指标

```python
# 典型攻击性能（BlackBox模式）
{
  "member_accuracy": 0.867,      # 成员识别准确率
  "nonmember_accuracy": 0.854,   # 非成员识别准确率
  "overall_accuracy": 0.860,     # 总体攻击准确率
  "precision": 0.856,           # 精确率
  "recall": 0.867,              # 召回率
  "auc": 0.892,                # AUC-ROC曲线下面积
  "tpr_at_low_fpr": 0.354,     # 0.1% FPR下的TPR
  "execution_time": "<2s"       # 单次攻击响应时间
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
    "attackMethod": "BlackBox"  # 支持: "RuleBased", "BlackBox", "Shadow"
}

response = requests.post("http://localhost:5000/attack", json=payload)
print(response.json())
```

访问网址：attack.eastyn.cn
测试账号：Eastyn
测试密码：Hu123456
测试数据集下载链接：https://attack-oss.oss-cn-chengdu.aliyuncs.com/excel/1741752570596_预处理后的Cardio.xlsx
测试模型下载链接：https://attack-oss.oss-cn-chengdu.aliyuncs.com/models/1741933185218_model2_randomforest_10.pkl

## 📚 文献支持
本系统实现参考以下前沿研究成果：
1. Shokri et al. - Membership Inference Attacks Against Machine Learning Models (IEEE S&P 2017)
2. Li et al. - Fast and Accurate Membership Inference Attack with Data-Independent Neural Networks (ICML 2023)
3. Wang et al. - Membership Inference Attacks via Advanced Aggregation Methods (2022)
4. Zhang et al. - Efficient Membership Inference Attacks via Feature Selection (2023)
5. Chen et al. - Lightweight Membership Inference Attacks for Deep Learning Models (2022)

## 📄 许可证
Apache 2.0 License
