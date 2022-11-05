# %%
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Conv2D, Dropout, MaxPooling2D, Flatten, Activation
from keras import Model

# %%
df = pd.read_csv("../data/FaceScore.csv")
df["female"], df["male"] = np.zeros(len(df)), np.ones(len(df))
# 产生性别分类数据
for i in range(len(df)):
    if df["Filename"][i][0] == "f":
        # 如果图片名以“f”开头，则为女性，female和male分别赋值为1和0
        df["female"][i] = 1
        df["male"][i] = 0
df.head()

# %%
images = []
for i in tqdm(range(df.shape[0])):
    img = image.image_utils.load_img('../data/images/'+df['Filename'][i])
    img = image.image_utils.img_to_array(img)
    img = image.image_utils.smart_resize(img, (227, 227))
    img = img/255
    images.append(img)

# %%
images = np.array(images, dtype=np.float32)
genders = np.array(df[["female", "male"]], dtype=np.float32)

# %%
X_train, X_test, y_train, y_test = train_test_split(images, genders, train_size = 0.7, random_state = 1)

# %%
from sklearn.decomposition import PCA

# %%
x_train = X_train.reshape((len(X_train), 227*227*3))
x_test = X_test.reshape((len(X_test), 227*227*3))
# 降维
pca = PCA(n_components = 1000)
# 训练
pca.fit(x_train)
pca_train = pca.transform(x_train)
pca_test = pca.transform(x_test)

# %%
layer_input = keras.layers.Input((1000, ))
# 中间变量
x = layer_input
# 输出个数为2，因为二分类问题
x = Dense(2)(x)
# 使用softmax回归将输出结果转换到0-1
x = Activation("softmax")(x)
# 输出层
layer_output = x
# 建立模型
model1 = keras.Model(layer_input, layer_output)
# 模型编译
# 损失函数为对数似然函数，监控目标为精度
model1.compile(optimizer = tf.optimizers.Adam(0.0005), loss = "categorical_crossentropy", metrics = ["accuracy"])
# 模型拟合
# 小批量300个，迭代50次
model1.fit(pca_train, y_train, validation_data = (pca_test, y_test), batch_size = 32, epochs = 50)

# %% [markdown]
# # LeNet

# %% [markdown]
# * Conv2d(6, [5, 5]):  6 * (5 * 5 * 3 + 1) = 456
# * Conv2d(16, [5, 5]): 16 * (5 * 5* 6 + 1) = 2,416
# * Dense(120):         (46656 + 1) * 120 = 5,598,840
# * Dense(84):          (120 + 1) * 84 = 10,164
# * Dense(2):           (84 + 1) * 2

# %%
input_layer = Input([227,227,3])
x = input_layer
x = Conv2D(6,[5,5],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)    
x = Conv2D(16,[5,5],padding = "valid", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)
x = Flatten()(x)   
x = Dense(120,activation = 'relu')(x)
x = Dense(84,activation = 'relu')(x)
x = Dense(2,activation = 'softmax')(x)
output_layer=x
model2=Model(input_layer,output_layer)
model2.summary()

# %%
model2.compile(optimizer = tf.optimizers.Adam(0.0005), loss = "categorical_crossentropy", metrics = ["accuracy"])
history2 = model2.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 32, epochs = 50)

# %% [markdown]
# # AlexNet

# %% [markdown]
# * Conv2d(96, [11, 11]): 96 * (11 * 11 * 3 + 1) = 34,944
# * Conv2d(96, [11, 11]): 256 * (5 * 5 * 96 + 1) = 614,656
# * Conv2d(96, [11, 11]): 384 * (3 * 3 * 256 + 1) = 885,120
# * Conv2d(96, [11, 11]): 384 * (3 * 3 * 384 + 1) = 1,327,488
# * Conv2d(96, [11, 11]): 256 * (3 * 3 * 384 + 1) = 884,992
# * Dense(4096)          (6 * 6 * 256 + 1) * 4096 = 37,752,832
# * Dense(4096)          (4096 + 1) * 4096 = 16,781,312
# * Dense(2)             (4096 + 1) * 2

# %%
IMSIZE = 227
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer
x = Conv2D(96,[11,11],strides = [4,4], activation = 'relu')(x) 
x = MaxPooling2D([3,3], strides = [2,2])(x)    
x = Conv2D(256,[5,5],padding = "same", activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x) 
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x) 
x = Conv2D(256,[3,3],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Flatten()(x)   
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(2,activation = 'softmax')(x) 
output_layer=x
model3=keras.Model(input_layer,output_layer)
model3.summary()


# %%
model3.compile(optimizer = tf.optimizers.Adam(0.0005), loss = "categorical_crossentropy", metrics = ["accuracy"])
history3 = model3.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 32, epochs = 50)

# %%
img = image.image_utils.load_img('../data/jyb.jpg')
img = image.image_utils.img_to_array(img)
img = image.image_utils.smart_resize(img, (227, 227))
img = img/255

# %%
img = img.reshape((1, 227, 227, 3))

# %%
model3.predict(img)


