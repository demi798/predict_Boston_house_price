# TensorFlow 数据并行 - Boston房价预测

## 项目概述
本项目使用 **TensorFlow Keras Sequential API** 和 **tf.distribute.MirroredStrategy()** 实现Boston房价预测任务的数据并行训练。

## 核心要求实现

✅ **TensorFlow框架**：使用Keras Sequential API构建神经网络  
✅ **网络架构**：输入层(13) → 隐藏层(10, ReLU) → 输出层(1)  
✅ **激活函数**：隐藏层使用ReLU激活函数  
✅ **优化器**：使用Adam优化器（学习率=0.01）  
✅ **损失函数**：使用MSE（均方误差）  
✅ **数据并行**：使用 `tf.distribute.MirroredStrategy()` 自动检测并利用所有可用GPU设备  

## 数据处理流程

```
housing.csv (506 samples, 13 features)
    ↓
[1] 加载数据: 特征(506, 13) + 目标(506,)
    ↓
[2] 数据规范化: MinMaxScaler 缩放到[0, 1]
    ↓
[3] 分割数据集: 训练集(379) + 测试集(127) [比例 75:25]
    ↓
[4] 初始化分布式策略
    ↓
[5] 在分布式策略内构建模型
```

## 网络架构详解

```
输入层 (13个特征)
    ↓
隐藏层 (10个神经元, ReLU激活)
    参数数: 13×10 + 10(偏置) = 140
    ↓
输出层 (1个神经元, 线性激活)
    参数数: 10×1 + 1(偏置) = 11
    ↓
总参数数: 151
```

### 数学表达

**隐藏层输出**:
$$h = \text{ReLU}(W_1 x + b_1)$$

**网络输出**:
$$\hat{y} = W_2 h + b_2$$

**损失函数** (MSE):
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

## 数据并行策略

### MirroredStrategy 工作原理

```python
strategy = tf.distribute.MirroredStrategy()
# 自动检测可用设备数量
print('设备数量:', strategy.num_replicas_in_sync)

with strategy.scope():
    # 在分布式策略内构建模型
    model = tf.keras.Sequential([...])
    model.compile(...)

# 使用相同的训练代码
model.fit(train_x, train_y, epochs=300)
```

**关键特性**：
- 自动检测所有物理GPU设备
- 在每个GPU上复制模型
- 数据自动分片到各GPU
- 梯度自动同步和平均
- 无需修改训练代码

## 训练配置

| 参数 | 值 |
|------|-----|
| 总Epoch数 | 300 |
| 批大小 | 32 |
| 优化器 | Adam(lr=0.01) |
| 损失函数 | MSE |
| 验证集比例 | 20% (从训练集中分出) |
| 评估指标 | MAE |

## 运行结果

### 性能指标

```
============================================================
训练完成！性能统计
============================================================
测试集 MSE: 17.5008
测试集 MAE: 2.7030
平均绝对误差(MAE): $2.7030K
均方根误差(RMSE): $4.1834K
最小预测误差: $0.0176K
最大预测误差: $22.7296K
============================================================
```

### 预测示例

前10个样本的预测结果 vs 真实值：
```
预测值: [27.51, 36.20, 15.89, 24.94, 17.13, 22.01, 17.25, 15.57, 22.51, 19.23]
真实值: [23.6,  32.4,  13.6,  22.8,  16.1,  20.0,  17.8,  14.0,  19.6,  16.8]
```

## 生成的可视化图表

程序自动生成4个关键图表：

1. **损失函数变化曲线** - 显示训练集和验证集MSE损失的变化趋势
2. **MAE变化曲线** - 平均绝对误差在训练过程中的变化
3. **预测值vs真实值散点图** - 测试集上的预测精度可视化
4. **残差分布直方图** - 预测残差的分布特征

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `tensorflow_boston_dataparallel.py` | 完整的数据并行实现代码 |
| `housing.csv` | Boston房价数据集 |
| `README_dataparallel.md` | 本文档 |

## 运行方法

```bash
# 创建Python 3.11虚拟环境（TensorFlow兼容性）
python3.11 -m venv .venv_tf

# 激活虚拟环境
source .venv_tf/bin/activate

# 安装依赖
pip install numpy pandas tensorflow scikit-learn matplotlib

# 运行程序
python tensorflow_boston_dataparallel.py
```

## 关键代码片段

### 1. 初始化分布式策略
```python
strategy = tf.distribute.MirroredStrategy()
print(f'可用设备/副本数量: {strategy.num_replicas_in_sync}')
```

### 2. 在分布式策略内构建模型
```python
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(13,), 
                            name='hidden_layer'),
        tf.keras.layers.Dense(1, name='output_layer')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )
```

### 3. 模型训练
```python
history = model.fit(
    train_x, train_y,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

## 性能分析

### 收敛特性
- **快速收敛**：在约100个Epoch内，模型损失从600+快速降至20左右
- **稳定性**：100个Epoch后，损失保持在19-20范围，说明模型趋于稳定
- **验证效果**：验证集损失与训练集损失相近，无明显过拟合

### 预测精度
- 平均绝对误差(MAE)为2.70K$，意味着平均预测误差约为房价的±$2,700
- 对于房价范围(~$5K-$50K)，这个精度是可接受的
- 残差分布相对均衡，说明模型对不同价位的房屋都有良好的预测能力

## 扩展方向

1. **模型优化**
   - 增加隐藏层数量和神经元数
   - 添加正则化(L1/L2)减少过拟合
   - 使用Dropout层提高泛化能力

2. **数据优化**
   - 特征工程：特征选择、特征组合
   - 异常值处理
   - 不同的归一化方式

3. **分布式训练优化**
   - 多机多GPU训练(tf.distribute.MultiWorkerMirroredStrategy)
   - 参数服务器策略(tf.distribute.ParameterServerStrategy)
   - 混合精度训练提高速度

4. **模型评估**
   - 交叉验证
   - 学习曲线分析
   - 特征重要性分析

## 技术要点总结

✅ **MirroredStrategy自动化**：无需手动分片数据，框架自动处理  
✅ **模型定义灵活**：支持Sequential和Functional API  
✅ **即插即用**：现有训练代码无需修改，仅需在模型构建处加入strategy.scope()  
✅ **性能可扩展**：可轻松扩展到多GPU/多机训练  
✅ **完整的数据处理流程**：包含加载、规范化、分割、训练、评估、可视化  

---

**相关文件**：
- [tensorflow_boston.py](tensorflow_boston.py) - 基础版本（单GPU）
- [tensorflow_boston_custom.py](tensorflow_boston_custom.py) - 自定义实现版本
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - 项目指导文档
