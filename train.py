import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout, Flatten,Dense
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_save_path = './checkpoint/dazhuan.tf'

'''cifar10=tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()'''
#制作数据集
train_dir='x_train'
test_dir='x_test'
classname=['贝','之','其','出','在','大','小','我','是','月','石','雨','河','华','将','文']
x_train=[];y_train=[];x_test=[];y_test=[]
for dir in os.listdir(train_dir):
    dirname=os.path.join(train_dir,dir)
    for file in os.listdir(dirname):
        impath=os.path.join(dirname,file)
        img=cv2.imread(impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_train.append(img)
        y_train.append(int(dir))

for dir in os.listdir(test_dir):
    dirname=os.path.join(test_dir,dir)
    for file in os.listdir(dirname):
        impath=os.path.join(dirname,file)
        img=cv2.imread(impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_test.append(img)
        y_test.append(int(dir))

#对训练集进行shuffle
random.seed(116)
random.shuffle(x_train)
random.seed(116)
random.shuffle(y_train)

x_train=np.array(x_train);y_train=np.array(y_train)
x_test=np.array(x_test);y_test=np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

#数据增强
'''image_gen_train = ImageDataGenerator(
                                     rescale=1./255,#归至0～1
                                     rotation_range=0,#随机45度旋转
                                     width_shift_range=.15,#宽度偏移
                                     height_shift_range=.15,#高度偏移
                                     horizontal_flip=False,#水平翻转
                                     zoom_range=0.5#将图像随机缩放到50％
                                     )
image_gen_train.fit(x_train)'''

'''x_train_subset1 = x_train[:10]
x_train_subset2 = x_train[:10]  # 一次显示12张图片

fig = plt.figure(figsize=(20,2))

# 显示原始图片,这里显示的图像的RGB通道是反的
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i+1)
    x_train_subset1[i].reshape((32,32))
    ax.imshow(x_train_subset1[i], cmap='gray')
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

# 显示增强后的图片
fig = plt.figure(figsize=(20,2))
for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12,shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(np.squeeze(x_batch[i]), cmap='gray')
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break'''



x_train, x_test = x_train / 255.0, x_test / 255.0
x_train=tf.convert_to_tensor(x_train)
x_test=tf.convert_to_tensor(x_test)
#y_train=tf.squeeze(y_train, axis=1)
#y_test=tf.squeeze(y_test, axis=1)


model = tf.keras.models.Sequential([
    Conv2D(filters=32,kernel_size=(5,5),padding='same',input_shape=(32,32,1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    Dropout(0.2),

    Conv2D(64, kernel_size=(5,5), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if os.path.exists(model_save_path+'.index'):
    print('-------------load the model-----------------')
    model.load_weights(model_save_path)
for i in range(5):
    history=model.fit(x_train, y_train, epochs=1,batch_size=20, validation_data=(x_test, y_test), validation_freq=2)
    model.save_weights(model_save_path, save_format='tf')
'''for i in range(5):
    history=model.fit(image_gen_train.flow(x_train, y_train,batch_size=20), epochs=5, validation_data=(x_test, y_test), validation_freq=1)
    model.save_weights(model_save_path, save_format='tf')'''

model.summary()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
#val_acc = history.history['val_sparse_categorical_accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()