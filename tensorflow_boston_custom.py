import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 如果上面的字体不可用，尝试使用系统字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
except:
    pass

# 数据加载（从housing.csv读取，无表头，空格分隔）
data = pd.read_csv('housing.csv', header=None, sep=r'\s+')
x = data.iloc[:, :-1].values  # 前13列为特征
y = data.iloc[:, -1].values  # 最后一列为目标
print("数据形状:", x.shape, y.shape)

# 将y转换形状
y = y.reshape(-1, 1)

# 数据规范化（MinMaxScaler归一化到[0,1]区间）
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)

# 构建神经网络（TensorFlow Sequential API）
# 输入层(13) -> 隐藏层(16, ReLU) -> 输出层(1, 线性)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(1)
])

# 定义损失函数和优化器
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae'])

# 显示模型结构
model.summary()

# 训练模型
max_epochs = 300
history = model.fit(train_x, train_y,
                   epochs=max_epochs,
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)

# 绘制训练损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(np.arange(max_epochs), history.history['loss'], label='训练损失')
plt.plot(np.arange(max_epochs), history.history['val_loss'], label='验证损失')
plt.title('训练和验证损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# 在测试集上进行预测
predictions = model.predict(test_x)

plt.subplot(1, 2, 2)
plt.scatter(test_y, predictions, alpha=0.6)
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', linewidth=2)
plt.title('预测值 vs 真实值')
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.grid(True)

plt.tight_layout()
plt.show()

# 计算测试集上的性能指标
test_loss, test_mae = model.evaluate(test_x, test_y, verbose=0)
print(".2f")
print(".2f")

# 显示前10个测试样本的预测结果
print("\n前10个测试样本预测结果:")
print("真实值\t预测值\t误差")
print("-" * 30)
for i in range(10):
    real = test_y[i][0]
    pred = predictions[i][0]
    error = abs(real - pred)
    print(".2f")