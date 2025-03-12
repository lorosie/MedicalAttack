from flask import Flask, request, jsonify
import requests
from io import BytesIO
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import os
from base_attack import load_and_preprocess_data, load_and_wrap_model, evaluate_attack

app = Flask(__name__)

# API接口
@app.route('/attack', methods=['POST'])
def attack():
    try:
        # 获取参数
        data = request.json
        data_url = data.get('dataUrl')
        model_url = data.get('modelUrl')
        attack_method = data.get('attackMethod')

        if not data_url or not model_url or not attack_method:
            return jsonify({'error': '缺少必要的参数'}), 400

        # 1. 加载数据
        print("正在加载数据...")
        x_train, x_test, y_train, y_test = load_and_preprocess_data(data_url)

        # 2. 加载并封装模型
        print("\n正在加载并封装模型...")
        art_classifier = load_and_wrap_model(model_url)
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
        print(f"初始化攻击方法: {attack_method}...")
        attack_results = evaluate_attack(art_classifier, x_train, x_test, y_train, y_test, attack_method)

        # 5. 返回攻击结果
        return jsonify({
            'model_performance': {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc)
            },
            'attack_results': attack_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)