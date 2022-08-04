import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import pandas as pd
import numpy as np
import model

data = pd.read_csv('mnist-dataset\\sign_mnist_train.csv')
data = np.array(data)
pixel = []
label = []
for item in data:
    pixel.append(item[1:].reshape(28,28,1))
    label.append(item[0])
train_x = np.array(pixel)
train_y = np.array(label)

data = pd.read_csv('mnist-dataset\\sign_mnist_test.csv')
data = np.array(data)
pixel = []
label = []
for item in data:
    pixel.append(item[1:].reshape(28,28,1))
    label.append(item[0])
test_x = np.array(pixel)
test_y = np.array(label)


# 图像归一化
train_x1 = train_x / 255
test_x1 = test_x / 255

# 标签独热编码
train_y1 = tf.keras.utils.to_categorical(train_y)
test_y1 = tf.keras.utils.to_categorical(test_y)

print("训练集样本数：", train_x1.shape[0])
print("训练集图像：", train_x1.shape)
print("训练集标签：", train_y1.shape)
print("测试集样本数：", train_x1.shape[0])
print("测试集图像：", train_x1.shape)
print("测试集标签", train_y1.shape)

parameters = model.model(train_x1, train_y1, test_x1, test_y1, lr=0.01, num_epoch=1000, train = True)