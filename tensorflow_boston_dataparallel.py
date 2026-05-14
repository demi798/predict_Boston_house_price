import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ==================== 数据加载和预处理 ====================
print("=" * 60)
print("TensorFlow数据并行神经网络 - Boston房价预测")
print("=" * 60)

# 数据加载（从housing.csv读取，无表头，空格分隔）
print("\n[1] 加载数据...")
data = pd.read_csv('housing.csv', header=None, sep=r'\s+')
x = data.iloc[:, :-1].values  # 前13列为特征 (506, 13)
y = data.iloc[:, -1].values   # 最后一列为目标 (506,)
print(f"特征数据形状: {x.shape}")
print(f"目标数据形状: {y.shape}")

# 将y转换形状
y = y.reshape(-1, 1)  # (506, 1)

# 数据规范化（MinMaxScaler将数据缩放到[0,1]范围）
print("\n[2] 数据规范化...")
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)
print(f"特征数据已规范化到[0,1]范围")

# 划分训练集和测试集（75% 训练，25% 测试）
print("\n[3] 划分训练集和测试集...")
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)
print(f"训练集大小: {train_x.shape[0]}, 测试集大小: {test_x.shape[0]}")

# ==================== 数据并行配置 ====================
print("\n[4] 配置数据并行策略 (MirroredStrategy)...")
strategy = tf.distribute.MirroredStrategy()
print(f"可用设备/副本数量: {strategy.num_replicas_in_sync}")
print(f"设备列表: {tf.config.list_physical_devices()}")

# ==================== 神经网络构建 ====================
print("\n[5] 在分布式策略内构建模型...")
with strategy.scope():
    # 构建神经网络模型
    # 输入层: 13个特征
    # 隐藏层: 10个神经元，ReLU激活函数
    # 输出层: 1个神经元，线性激活（回归问题）
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(13,), 
                            name='hidden_layer'),
        tf.keras.layers.Dense(1, name='output_layer')
    ])
    
    # 编译模型
    # 优化器: Adam (自适应学习率优化器，推荐用于神经网络)
    # 损失函数: MSE (均方误差，用于回归问题)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']  # 添加MAE作为评估指标
    )

print("模型架构:")
model.summary()

# ==================== 模型训练 ====================
print("\n[6] 开始训练模型...")
max_epoch = 300
history = model.fit(
    train_x, train_y,
    epochs=max_epoch,
    batch_size=32,
    validation_split=0.2,  # 从训练集中分出20%作为验证集
    verbose=1
)

# ==================== 结果评估 ====================
print("\n[7] 模型评估...")
# 在测试集上评估
test_loss = model.evaluate(test_x, test_y, verbose=0)
print(f"测试集 MSE: {test_loss[0]:.4f}")
print(f"测试集 MAE: {test_loss[1]:.4f}")

# ==================== 预测结果 ====================
print("\n[8] 进行预测...")
predict_list = model.predict(test_x)
print(f"前10个预测值: {predict_list[:10].flatten()}")
print(f"前10个真实值: {test_y[:10].flatten()}")

# ==================== 可视化结果 ====================
print("\n[9] 生成可视化图表...")

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TensorFlow数据并行 - Boston房价预测结果分析', fontsize=16, fontweight='bold')

# 1. 训练损失曲线
axes[0, 0].plot(np.arange(max_epoch), history.history['loss'], label='训练集MSE', linewidth=2)
axes[0, 0].plot(np.arange(max_epoch), history.history['val_loss'], label='验证集MSE', linewidth=2)
axes[0, 0].set_title('损失函数变化曲线', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE损失')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. MAE曲线
axes[0, 1].plot(np.arange(max_epoch), history.history['mae'], label='训练集MAE', linewidth=2)
axes[0, 1].plot(np.arange(max_epoch), history.history['val_mae'], label='验证集MAE', linewidth=2)
axes[0, 1].set_title('平均绝对误差(MAE)变化曲线', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 预测值vs真实值散点图
x_idx = np.arange(test_x.shape[0])
axes[1, 0].scatter(x_idx, predict_list.flatten(), c='red', alpha=0.6, s=30, label='预测值')
axes[1, 0].scatter(x_idx, test_y.flatten(), c='blue', alpha=0.6, s=30, label='真实值')
axes[1, 0].set_title('测试集预测值vs真实值', fontweight='bold')
axes[1, 0].set_xlabel('样本索引')
axes[1, 0].set_ylabel('房价(单位: $1000)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 残差分析
residuals = predict_list.flatten() - test_y.flatten()
axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='green')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
axes[1, 1].set_title('预测残差分布', fontweight='bold')
axes[1, 1].set_xlabel('残差')
axes[1, 1].set_ylabel('频数')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 性能统计 ====================
print("\n" + "=" * 60)
print("训练完成！性能统计")
print("=" * 60)
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals ** 2))
print(f"MAE (平均绝对误差): ${mae:.4f}K")
print(f"RMSE (均方根误差): ${rmse:.4f}K")
print(f"最小预测误差: ${np.min(np.abs(residuals)):.4f}K")
print(f"最大预测误差: ${np.max(np.abs(residuals)):.4f}K")
print("=" * 60) 