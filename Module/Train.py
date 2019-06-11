# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,Dropout,BatchNormalization,UpSampling2D
from tensorflow.keras.layers import PReLU,Add,LeakyReLU
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import sqlite3
import json
import cv2
import os
############################可调整参数##########################
batchSize = 40#批处理量
epochs = 500#迭代次数
genFilters = 64#生成核数
disFilters = 64#判别核数
genLearnRate = 5e-5#生成学习率
disLearnRate = 1e-4#判别学习率
##############################################################

#GPU加速
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
set_session(tf.Session(config=config)) 

#路径目录
curDir = os.path.dirname(os.path.abspath(__file__))#当前目录
baseDir = os.path.dirname(curDir)#根目录
staticDir = os.path.join(baseDir,'Static')#静态文件目录
dbPath = os.path.join(staticDir,'db.sqlite3')#数据库路径

#数据库
conn = sqlite3.connect(dbPath)
cursor = conn.cursor()

def createGenerator():
    '''创建生成器'''
    def resBlock(xIn):
        '''残差块'''
        x = Conv2D(filters=genFilters,kernel_size=3,padding='same')(xIn)
        x = BatchNormalization(momentum=0.8)(x)
        x = PReLU()(x)
        x = Conv2D(filters=genFilters,kernel_size=3,padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([xIn, x])
        return x

    #输入层
    inputLayer = Input(shape=(32,32,3))
    
    #第一层
    firstLayer = Conv2D(filters=genFilters,kernel_size=3,padding='same')(inputLayer)
    firstLayer = PReLU()(firstLayer)
    
    #中间层
    middle = firstLayer
    for num in range(5):
        middle = resBlock(middle)   
    middle = Conv2D(filters=genFilters,kernel_size=3,padding='same')(middle)
    middle = BatchNormalization()(middle)
    middle = Add()([firstLayer,middle])
    middle = UpSampling2D(size=2)(middle)
    
    #输出层
    outputLayer = Conv2D(filters=3,kernel_size=9,padding="same",activation='tanh')(middle)
    
    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    model.summary()
    return model

def createDiscriminator():
    '''创建判别器'''
    def block(xIn,strides):
        '''判别层'''
        x = Conv2D(disFilters,3,strides=strides,padding='same')(xIn)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(0.2)(x)
        return x
    
    #输入层
    inputLayer = Input(shape=(64,64,3))
    
    #中间层
    middle = inputLayer
    middle = block(middle,2)
    middle = block(middle,2)
    middle = block(middle,1)
    middle = block(middle,2)
    middle = block(middle,1)
    middle = block(middle,2)
    middle = block(middle,1)
    middle = block(middle,2)
    middle = Flatten()(middle)
    middle = Dense(1024)(middle)
    middle = LeakyReLU(0.2)(middle)
    
    #输出层
    outputLayer = Dense(1, activation='sigmoid')(middle)
    
    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    model.summary()
    return model

def createGAN(generator,discriminator):
    '''对抗网'''
    discriminator.trainable = False
    #生成器输入
    lowImg = generator.input
    #生成器输出
    fakeHighImg = generator(lowImg)
    #生成器判断
    judge = discriminator(fakeHighImg)
    model = Model(inputs=lowImg,outputs=[judge,fakeHighImg])
    model.summary()
    return model

#特征提取器
vgg19 = VGG19(include_top=False, weights='imagenet')
vgg19 = Model(vgg19.input, vgg19.output)

def restoreImg(img):
    '''还原图片'''
    img = (img/2 + 0.5)*255
    return img

def contentLoss(y_true, y_pred):
    '''内容损失'''
    y_true = restoreImg(y_true)
    y_pred = restoreImg(y_pred)
    y_true = preprocess_input(y_true)
    y_pred = preprocess_input(y_pred)
    sr = vgg19(y_pred)
    hr = vgg19(y_true)
    return mse(y_true, y_pred)

#生成器   
generator = createGenerator()
#判别器
discriminator = createDiscriminator()
if os.path.exists(staticDir+'/generator.h5'):
    generator = load_model(staticDir+'/generator.h5')
    discriminator = load_model(staticDir+'/discriminator.h5')
    print('load model success~')
optimizer = Adam(lr=disLearnRate)
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')
#对抗网
GAN = createGAN(generator,discriminator)
optimizer = Adam(lr=genLearnRate)
GAN.compile(optimizer=optimizer, loss=['binary_crossentropy', contentLoss],loss_weights=[0.01, 0.001])



def getImgs(startId,endId):
    '''根据id获得图片信息'''
    sql = 'select desc from img where id>=%s and id<=%s'%(startId,endId)
    cursor.execute(sql)
    imgs = {'img32One':[],'img64One':[]}
    for desc in cursor.fetchall():
        desc = json.loads(desc[0])
        img32One = np.array(desc['img32One'])
        img64One = np.array(desc['img64One'])
        if img32One.shape[-1]!=3:
            continue
        imgs['img32One'].append(img32One)
        imgs['img64One'].append(img64One)
    imgs['img32One'] = np.array(imgs['img32One'])
    imgs['img64One'] = np.array(imgs['img64One'])
    return imgs

for epoch in range(1,epochs+1):
    dLossTotal = list()
    gLossTotal = list()
    cLossTotal = list()
    for index in range(int(5000/batchSize)):
        start = time.time()#开始时间
        startId = index * batchSize + 1
        endId = startId + batchSize - 1
        imgs = getImgs(startId,endId)
        #低像素图片
        low = imgs['img32One']
        #生成高像素图片
        fakeHigh = generator.predict(low)
        #原高像素图片
        realHigh = imgs['img64One']
        #真伪标签
        realBool = np.random.uniform(0.7,1,size=(low.shape[0],))
        fakeBool = np.random.uniform(0,0.3,size=(low.shape[0],))

        #鉴别器训练
        discriminator.trainable = True
        dRealLoss = discriminator.train_on_batch(x=realHigh, y=realBool)
        dFakeLoss = discriminator.train_on_batch(x=fakeHigh, y=fakeBool)
        dLoss = 0.5 * (dRealLoss + dFakeLoss)

        #对抗网络训练
        discriminator.trainable = False
        ganLoss = GAN.train_on_batch(x=low, y=[realBool,realHigh])
        gLoss = ganLoss[0]
        cLoss = ganLoss[2]
        past = time.time()-start

        dLossTotal.append(dLoss)
        gLossTotal.append(gLoss)
        cLossTotal.append(cLoss)

    dLoss = sum(dLossTotal)/len(dLossTotal)
    gLoss = sum(gLossTotal)/len(gLossTotal)
    cLoss = sum(cLossTotal)/len(cLossTotal)
    print('epoch:%d/%d dLoss:%.4f gLoss:%.4f cLoss:%.4f pst:%.2fs'%(epoch,epochs,dLoss,gLoss,cLoss,past))
    #保存模型
    generator.save(staticDir+'/generator.h5')
    discriminator.save(staticDir+'/discriminator.h5')
    try:
        #低像素传统放大
        testIndex = (epoch%low.shape[0]) - 1
        tranLow = restoreImg(low[testIndex:testIndex+1]).reshape(32,32,3)
        tranLow = Image.fromarray(tranLow.astype(np.uint8))
        tranLow = tranLow.resize((64,64),resample=Image.ANTIALIAS)
        tranLow = np.array(tranLow).astype(np.int32)    
        #生成图像
        fakeHigh = generator.predict(low[testIndex:testIndex+1]).reshape((64,64,3))
        fakeHigh = restoreImg(fakeHigh).astype(np.int32)
        #原图像
        oriHigh = realHigh[testIndex:testIndex+1].reshape(64,64,3)
        oriHigh = restoreImg(oriHigh).astype(np.int32)
        #组合图片
        combine = np.concatenate((tranLow,fakeHigh,oriHigh), axis=1)
        img = Image.fromarray(combine.astype(np.uint8))
        img.save(os.path.join(staticDir+'/测试','%s.jpg'%(epoch)))
    except:
        continue
