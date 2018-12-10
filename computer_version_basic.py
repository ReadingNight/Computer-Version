# coding: utf-8
import cv2
import numpy as np

def show_gray_img_hist(hist, window_title):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    hist_img = np.zeros([256, 256], np.uint8)
    for h in range(256):
        intensity = int(256 * hist[h] / max_val)
        cv2.line(hist_img, (h, 256), (h, 256 - intensity), [255, 0, 0])

    cv2.imshow(window_title, hist_img)


def get_acc_prob_hist(hist):
    acc_hist = np.zeros([256, 1], np.float32)
    pre_val = 0.
    for i in range(256):
        acc_hist[i, 0] = pre_val + hist[i, 0]
        pre_val = acc_hist[i, 0]

    acc_hist /= pre_val
    return acc_hist


def hist(src_img, dst_img):
    # 均衡化
    result_equl = cv2.equalizeHist(dst_img)
    # cv2.imshow('dst_img', dst_img)
    cv2.imwrite('dst_img_equl.png', result_equl)

    # 计算源图像和规定化之后图像的累计直方图
    src_hist = cv2.calcHist([src_img], [0], None, [256], [0.0, 255.])
    dst_hist = cv2.calcHist([dst_img], [0], None, [256], [0.0, 255.])
    src_acc_prob_hist = get_acc_prob_hist(src_hist)
    dst_acc_prob_hist = get_acc_prob_hist(dst_hist)

    # 计算源图像的各阶灰度到规定化之后图像各阶灰度的差值的绝对值，得到一个256*256的矩阵，第i行表示源图像的第i阶累计直方图到规定化后图像各
    # 阶灰度累计直方图的差值的绝对值，
    diff_acc_prob = abs(np.tile(src_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_acc_prob_hist.reshape(1, 256))

    # 求出各阶灰度对应的差值的最小值，该最小值对应的灰度阶即为映射之后的灰度阶
    table = np.argmin(diff_acc_prob, axis=0)
    table = table.astype(np.uint8)  # @注意 对于灰度图像cv2.LUT的table必须是uint8类型

    # 将源图像按照求出的映射关系做映射
    result = cv2.LUT(dst_img, table)

    # 显示各种图像
    show_gray_img_hist(src_hist, 'src_hist')
    show_gray_img_hist(dst_hist, 'dst_hist')
    # cv2.imshow('src_img', src_img)
    # cv2.imshow('dst_img', dst_img)
    # cv2.imshow('result', result)
    cv2.imwrite('dst_img_hist.png', result)

    result_hist = cv2.calcHist([result], [0], None, [256], [0.0, 255.])
    show_gray_img_hist(result_hist, 'result_hist')

    return result_equl


def salt(dst_img, n):
    for k in range(n):
        i = int(np.random.random() * dst_img.shape[1]);
        j = int(np.random.random() * dst_img.shape[0]);
        if dst_img.ndim == 2:
            dst_img[j, i] = 255
        elif dst_img.ndim == 3:
            dst_img[j, i, 0] = 255
            dst_img[j, i, 1] = 255
            dst_img[j, i, 2] = 255
    cv2.imwrite('salt_img.png', dst_img)
    return dst_img


def median(dst_img, n):
    result_salt = salt(dst_img, 1000)
    result = cv2.medianBlur(result_salt, n)
    cv2.imwrite('dst_img_salt.png', result_salt)
    cv2.imwrite('dst_img_median'+str(n)+'.png', result)

    return result


def sobel(dst_img):
    # 使用16位有符号的数据类型
    x = cv2.Sobel(dst_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(dst_img, cv2.CV_16S, 0, 1)

    # 转回uint8
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)

    result = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    cv2.imwrite("dst_img_sobelX.png", abs_x)
    cv2.imwrite("dst_img_sobelY.png", abs_y)
    cv2.imwrite("dst_img_sobel.png", result)


def prewitt(dst_img):
    grad_x = np.matrix('-1,-1,-1;'
                       '0,0,0;'
                       '1,1,1')

    grad_y = np.matrix('-1,0,1;'
                       '-1,0,1;'
                       '-1,0,1')

    abs_x = cv2.filter2D(dst_img, -1, grad_x)
    abs_y = cv2.filter2D(dst_img, -1, grad_y)

    img_prewitt = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    cv2.imwrite('dst_img_prewitt.png', img_prewitt)


def RobertsOperator(img):
    operator_first = np.array([[-1, 0],[0, 1]])
    operator_second = np.array([[0, -1],[1, 0]])
    return np.abs(np.sum(img[1:, 1:]*operator_first))+np.abs(np.sum(img[1:, 1:]*operator_second))


def Roberts(dst_img):
    dst_img = cv2.copyMakeBorder(dst_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

    for i in range(1, dst_img.shape[0]):
        for j in range(1, dst_img.shape[1]):
            dst_img[i, j] = RobertsOperator(dst_img[i-1:i+2, j-1:j+2])

    cv2.imwrite('dst_img_roberts.png', dst_img[1:dst_img.shape[0], 1:dst_img.shape[1]])


if __name__ == '__main__':
    src_img = cv2.imread('pan.jpg', 0)
    dst_img = cv2.imread('pan.png', 0)

    dst_img_equl = hist(src_img, dst_img)

    sobel_img = median(dst_img_equl, 3)
    sobel(sobel_img)

    roberts_img = median(dst_img_equl, 5)
    Roberts(roberts_img)

    prewitt_img = median(dst_img_equl, 7)
    prewitt(prewitt_img)

    cv2.waitKey()
