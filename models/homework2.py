# %%
import pandas as pd # 导入pandas
facescore = pd.read_csv("../data/FaceScore.csv") # 读取颜值评分的数据
print(facescore.shape) # 输出数据的行列数，判断数据是否完整
facescore[0:10] # 观察数据格式

# %%
facescore.hist(); # 通过绘制直方图观察颜值评分的分布，输入分号只显示图

# %%
import numpy as np # 导入numpy
# 分离X和Y，将Y转换成数组
picture_name = facescore["Filename"] # 提取Filename列
N = len(picture_name) # 获得样本量
y = np.array(facescore["Rating"]).reshape(N, 1) # 提取Y，转换成二维数组

y # 查看Y

# %%
# 准备图片数据
from PIL import Image # 导入Image
size = 128 # 图片尺寸

x = np.zeros([N, size, size, 3]) # 初始化X为零，可以存储N个大小size*size、通道是3的图片矩阵

# 读取图片数据
for i in range(N):
    name = picture_name[i] # 第i张图片的名称
    image = Image.open("../data/images/" + name) # 按照文件名读取图片
    image = image.resize([size, size]) # 更改图片大小
    image = np.array(image)/255 # 控制数据范围，不让图片显示为白色（imshow默认数据范围为0-1），并将图片数据转换成二维数组
    x[i, ] = image # 存入x中

# %%
# 绘制前15张图片
import matplotlib.pyplot as plt # 导入绘图模块

plt.figure() # 绘制画布
fig, ax = plt.subplot(3, 5) # 排列为3×5
fig.set_figheight(9.5) # 设置高度
fig.set_figwidth(15) # 设置宽度

ax = ax.flatten() # 降维函数，使得能使用ax[i]，不必使用ax[i, j]

for i in range(15):
    ax[i].imshow(x[i, :, :, :]) # 绘制第i张图片
    ax[i].set_title(np.round(y[i], 2)) # 设置标题为得分，并保留两位小数

# %%
# 切分数据集
from sklearn.model_selection import train_test_split # 导入划分数据集模块
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 123) # 切分训练集和测试集，70%训练集

# %%
from keras.layers import Input, Flatten, Dense
# 导入输入层和全连接层模块，通过Flatten在全连接层之前将多维数组压缩成一维数组
from keras import Model # 导入Model模块
input_layers = Input([size, size, 3]) # 设置张量维度，输出时为[None, size, size, 3]，None是batch size
a = input_layers # 通过a得到输出层
a = Flatten()(a) # 拉直a，从而传入全连接层
a = Dense(256)(a)
a = Dense(256)(a)
a = Dense(1)(a) # 输出维度1
output_layers = a # 输出层

# 构建模型
model = Model(input_layers, output_layers)
model.summary() # 输出模型描述

# %%
# 与已知结果进行对比
test01 = Image.open("../data/images/" + picture_name[76])
test01

# %%
model.compile(loss='mse', metrics=['mse', 'mae'])

# %%
model.fit(x=x_train, y=y_train, batch_size=32, epochs=10)

# %%
# 处理图片
image01 = test01
image01 = image01.resize([size, size]) # 重新设置图片尺寸
image01 = np.array(image01)/255 # 改变数据范围
image01 = image01.reshape((1, size, size, 3)) # 改变数组形状，使得能够输入模型
print("原始评分：", y[77], "预测得分：", model.predict(image01))

# %%
# 为输入的图片打分
test02 = Image.open("颜值模型待打分图片.jpg")
test02

# %%
# 处理图片
image02 = test02
image02 = image02.resize([size, size]) # 重新设置图片尺寸
image02 = np.array(image02)/255 # 改变数据范围
image02 = image02.reshape((1, size, size, 3)) # 改变数组形状，使得能够输入模型
print( "颜值评分结果：", model.predict(image02))


