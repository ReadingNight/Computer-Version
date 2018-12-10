# 利用python实现多种方法来实现图像识别

import cv2
import os
import glob as gb
import numpy as np
from matplotlib import pyplot as plt


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def classify_gray_hist(image1, image2, size=(256, 256)):
    # 先计算直方图
    # 几个参数必须用方括号括起来
    # 这里直接用灰度图计算直方图，所以是使用第一个通道，
    # 也可以进行通道分离后，得到多个通道的直方图
    # bins 取为16
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 可以比较下直方图
    plt.plot(range(256), hist1, 'r')  # 红色折线
    plt.plot(range(256), hist2, 'b')  # 蓝色折线
    plt.show()
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree  # 返回两张图片的重合程度


# 平均哈希算法计算
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 颜色空间转换函数(B,G,R->gray) 变成灰度图
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)  # 返回汉明距离

#感知哈希
def classify_pHash(image1,image2):
    image1 = cv2.resize(image1,(32,32))
    image2 = cv2.resize(image2,(32,32))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8,0:8]
    dct2_roi = dct2[0:8,0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1,hash2)


# 输入灰度图，返回hash
def getHash(image):
    # 计算进行灰度处理后图片的所有像素点的平均值。
    # 遍历灰度图片每一个像素，如果大于平均值记录为1，否则为0.
    # 得到并返回哈希序列
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


#找到最小汉明距离
def find_min(array):
    min = array[0]
    index = []
    for i in range(0,len(array)):
        if array[i] <min:
            min = array[i]
    for i in range(0,len(array)):
        if array[i] == min:
            index.append(i)
    return index

#找到最大重合度
def find_max(array):
    max = array[0]
    index = 0
    for i in range(0, len(array)):
        if array[i] > max:
            max = array[i]
            index = i
    return index

if __name__ == '__main__':
    img1 = cv_imread('F:\\Python\\Hash\\1.jpg')
    cv2.imshow('img1', img1)


#用感知哈希的方法遍历图像文件夹中每一个图片，与img1比较
    imgs = os.listdir("F:\\Python\\Hash\\test")
    imgNum = len(imgs)
    Dict = []
    for i in range(imgNum):
        img = cv_imread("F:\\Python\\Hash\\test" + "\\" + imgs[i])
        #degree = classify_gray_hist(img1, img)
        degree = classify_pHash(img1,img)
        print(degree)
        Dict.append(degree)

    index = find_min(Dict)
    print(index)
    img = []
    for i in range(len(index)):
        img.append(cv_imread("F:\\Python\\Hash\\test" + "\\" + imgs[index[i]]))
        cv2.imshow('img' + str(i + 2), img[i])


    #用直方图方法进行第二次遍历:
    Dict2 = []
    for i in range(len(img)):
        degree2 = classify_gray_hist(img1, img[i])
        print(degree2)
        Dict2.append(degree2)

    index2 = find_max(Dict2)#找到最大相似度的索引
    print(index2)
    cv2.imshow('final_img',img[index2])

    cv2.waitKey(0)
