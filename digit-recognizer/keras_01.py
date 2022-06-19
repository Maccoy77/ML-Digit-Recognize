# 本模型为深度学习入门demo
# Keras 手写数据集MNIST
# 每张图片都是28px*28px 采集的不同从0-9的手写数字
# train data has 694 images data
# test data has 697 images data
# Auther: Yang Zhenhua  Date:2022
import csv
import numpy as np
import tensorflow as tf
import keras
#from keras.utils import to_categorical
#from keras import models, layers, regularizers
#from keras.optimizers import  RMSprop

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'





##Input Data##

with open('./train.csv','r') as csvfile: # r 表示以只读的方式读取csv文件
    reader = csv.reader(csvfile)         # 此时reader返回的值是csv文件中的每行的列表
    header = next(reader)                # next只调用一次得到的文件第一行 返回值存储在header pixel1。。。
    traindata = []                            # 保存所有数据
    for line in reader:                  # 统计共有几行数据
        traindata.append(line)

#print(type(traindata))           # data数据类型
#print(len(traindata))            # 数据数量
#print(len(traindata[0]))         # 28*28+1=785 1列标签 784数据


with open('./test.csv','r') as testfile:
    reader = csv.reader(testfile)
    header = next(reader)
    testdata = []
    for line in reader:
        testdata.append(line)

#print(type(testdata))
#print(len(testdata))
#print(len(testdata[0]))

#42000条数据训练 其中40000条训练 2000条测试
#28000条数据检验测试

#添加训练集
train_img = [] # train data
train_tar = [] # train target label
for i in range(40000):
    train_tar.append(traindata[i][0])
    train_img.append(traindata[i][1:])

#添加测试集
test_img = []
test_tar = []

for i in range(2000):
    test_tar.append((traindata[i+40000][0]))
    test_img.append((traindata[i+40000][1:]))


#切分数据并检验
#print('train_img\'s len:',len(train_img),'       pixel of each line:', len(train_img[0]))
#print('train_tar\'s len:',len(train_tar))

#print('test_img\'s len:',len(test_img),'       pixel of each line:', len(test_img[0]))
#print('test_tar\'s len:',len(test_tar))

##Change Data Type##

#转换数据类型
#之前的数组都是字符串，需要把数字转换为整型和浮点数
#数据转换为浮点数，标签转化为整型

for index in range(40000):
    for i,v in enumerate(train_img[index]):
        train_img[index][i] = float(v)

for i,v in enumerate(train_tar):
    train_tar[i] = int(v)

for index in range(2000):
    for i,v in enumerate(test_img[index]):
        train_img[index][i] = float(v)

for i,v in enumerate(test_tar):
    test_tar[i] = int(v)

#前面的数据都是list，把数据专程数组
#print(type(train_img))
#把训练数据集列表转为数组
#40000行784列 训练数据
train_data = np.zeros((40000,784))
#40000行784列 训练结果标签
train_res = np.zeros(40000)

for i in range(40000):
    for j in range(784):
        train_data[i][j] = train_img[i][j]

for i in range(40000):
    train_res[i] = train_tar[i]
#print(type(train_data))

#同理，将测试集也转化
#2000行784列 测试数据
test_data = np.zeros((2000,784))
#2000行 测试结果标签
test_res = np.zeros(2000)

for i in range(2000):
    for j in range(784):
        test_data[i][j] = test_img[i][j]

for i in range(2000):
    test_res[i] = int(test_tar[i])
print(type(test_data))



##Build Model##


















