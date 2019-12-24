import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout, Flatten,Dense
import cv2
import numpy as np
from PIL import Image,ImageDraw, ImageFont
classname=['贝','之','其','出','在','大','小','我','是','月','石','雨','河','华','将','文']
model_save_path = './checkpoint/dazhuan.tf'
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
model.load_weights(model_save_path)
testPath='train'
def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

'''def showReInCv2():
    for dir in os.listdir(testPath):
        count = 1
        picNum = len(files)
        colNum = 5
        # rowNum = (picNum - 1) // colNum + 1
        rowNum = 1
        imgs = []
        dirname=os.path.join(testPath, dir)
        for filename in os.listdir(dirname):
            picPath = os.path.join(dirname, filename)
            # image_path = input("the path of test picture:")

            img = cv2.imread(picPath)
            img_arr= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_arr=cv2.resize(img_arr,(32,32))


            for i in range(28):
                for j in range(28):
                    if img_arr[i][j] < 200:
                        img_arr[i][j] = 0
                    else:
                        img_arr[i][j] = 255

            img_arr = img_arr / 255.0

            x_predict = img_arr.reshape(32, 32, 1)
            x_predict = x_predict[tf.newaxis, ...]
            #

            result = model.predict(x_predict)#ndarray
            pred = tf.argmax(result, axis=1)[0]#pred为tensor
            #title="pre:" + classname[pred] + " ground true:" + classname[int(dir.split('_')[0])]
            #img = cv2.imread(picPath)
            img = cv2ImgAddText(img, "pre:" + classname[pred], 0, 30)
            img = cv2ImgAddText(img, " true:" , 0, 60)
            img = cv2ImgAddText(img, classname[int(dir.split('_')[0])], 0, 100)
            cv2.imshow("predicted image",img)
            cv2.waitKey(0)'''

def infer(imgs,minThred):
    img_arr = imgs / 255.0

    x_predict = img_arr.reshape(np.shape(img_arr)[0],32, 32, 1)
    #x_predict = x_predict[tf.newaxis, ...]

    result = model.predict(x_predict)
    preds = np.argmax(result, axis=1)
    re=''
    for i,pre in enumerate(preds):
        if result[i][pre]<minThred:#如果预测值小于阈值，视为无效
            re+="?"
        else:
            re+=classname[pre]
    return re


#showReInCv2()