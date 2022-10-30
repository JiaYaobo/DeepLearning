# %% [markdown]
# ### 几中不同优化算法的比较

# %%
# 载入必要的函数库
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Model
from keras.layers import Dense,Flatten,Input
from keras.optimizers import SGD,RMSprop,Adam

from keras.utils import to_categorical 

# %%
## 载入mnist数据集
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

## 转换为one - hot型向量
Y_train=to_categorical(y_train)
Y_test=to_categorical(y_test)

print(Y_train.shape)
print(Y_train[0])

# %%
'''
 实验：构建 Multi-layer Nueral Network 模型 
'''

##  第一步  创建模型结构 ##

IMSIZE = 28                                               
input_layer = Input([IMSIZE,IMSIZE])       # MNIST图像为28*28的单层图片
x = input_layer                              
x = Flatten()(input_layer)                   # 将28*28*1的Tensor拉直为784维向量
x = Dense(1000,activation = 'relu')(x)       # 全连接到1000个节点，并采用relu激活函数
x = Dense(10,activation = 'softmax')(x)      # 全连接到10个节点，并采用softmax激活函数转化为(0,1)取值
output_layer=x
model=Model(input_layer,output_layer)    # Model函数将input_layer 和 output_layer中间的部分连接起来
model.summary()

##  第二步  模型编译 ##

model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.001),metrics=['accuracy'])

##  第三步  模型拟合 ##

history1 = model.fit(X_train,Y_train, validation_data=(X_test,Y_test), batch_size=1000, epochs=50)

# 第四部  提取loss指标
# model.fit会返回一个history对象，里面记录了训练集和测试集的loss以及acc
# 我们将这些指标取出，绘制折线图

train_loss1 = history1.history["loss"]

# %% [markdown]
# ### 请编写出Momentum、RMSprop和Adam优化算法下的train loss情况，并将四种优化算法绘制在一张折线图下，如下

# %%
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy'])
history2 = model.fit(X_train,Y_train, validation_data=(X_test,Y_test), batch_size=1000, epochs=50)

# %%
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
history3 = model.fit(X_train,Y_train, validation_data=(X_test,Y_test), batch_size=1000, epochs=50)

# %%
model.compile(loss='categorical_crossentropy',optimizer=SGD(momentum=0.9),metrics=['accuracy'])
history4 = model.fit(X_train,Y_train, validation_data=(X_test,Y_test), batch_size=100, epochs=50)


