# %%
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
import matplotlib.pyplot as plt

# %%
data_dir = "../data/"

# %%
df = pd.read_csv("../data/FaceScore.csv")

# %%
df.head()

# %%
rating = df.pop('Rating')
# rating = (rating - np.mean(rating))/np.std(rating)

# %%
images = []
for i in tqdm(range(df.shape[0])):
    img = image.image_utils.load_img('../data/images/'+df['Filename'][i])
    img = image.image_utils.img_to_array(img)
    img = image.image_utils.smart_resize(img, (227, 227))
    img = img/255
    images.append(img)

# %%
images = tf.convert_to_tensor(images, dtype=tf.float32)

# %%
dataset = tf.data.Dataset.from_tensor_slices((images, rating.values))

# %%
train_size = int(5500 * 0.7)
test_size = int(5500 * 0.15)
val_size = int(5500 * 0.15)

# %%
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)
val_ds = test_ds.skip(test_size)
test_ds = test_ds.take(test_size)

# %%
train_ds = train_ds.shuffle(buffer_size=train_size).batch(32, drop_remainder=True)
test_ds = test_ds.shuffle(buffer_size=test_size).batch(32, drop_remainder=True)
val_ds = val_ds.shuffle(buffer_size=test_size).batch(32, drop_remainder=True)

# %%
plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(3)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image[0])
    plt.title(str(label[0].numpy()))
    plt.axis('off')

# %%
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

# %%
model.compile(loss='mse', optimizer=tf.optimizers.SGD(learning_rate=0.01), metrics=['mae', 'mse'])
model.summary()

# %%
history = model.fit(train_ds, epochs=20, validation_data=val_ds,
          validation_freq=1)

# %%
fig, axs = plt.subplots(2, 1, figsize=(15,15))  
axs[0].plot(history.history['loss']) 
axs[0].plot(history.history['val_loss']) 
axs[0].title.set_text('Training Loss vs Validation Loss') 
axs[0].legend(['Train', 'Val'])  
axs[1].plot(history.history['mse']) 
axs[1].plot(history.history['val_mse']) 
axs[1].title.set_text('Training MSE vs Validation MSE') 
axs[1].legend(['Train', 'Val'])

# %%
model.evaluate(test_ds)

# %%



