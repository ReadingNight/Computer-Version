#coding=utf-8
import cv2
import numpy as np
from skimage import morphology, measure
import matplotlib.pyplot as plt

img_path = "data/morphology/test.bmp"
erosion_path = "data/morphology/erosion.bmp"
dilation_path = "data/morphology/dilation.bmp"
close_path = "data/morphology/close.bmp"
open_path = "data/morphology/open.bmp"

img = cv2.imread(img_path, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#腐蚀
erosion = cv2.erode(img, kernel)
cv2.imwrite(erosion_path, erosion);

#膨胀
dilation = cv2.dilate(img, kernel)
cv2.imwrite(dilation_path, dilation);

#闭运算
close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(close_path, close);

#开运算
open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite(open_path, open);

# 显示
pic = [img_path, erosion_path, dilation_path, close_path, open_path]
plt.figure(figsize=(8, 6))
for i in range(5):
    img = cv2.imread(pic[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(pic[i])
plt.tight_layout()
plt.show();

# 方法一：去除较小的连通分量
def remove_objects(img):
    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域
    # is_del = False
    if len(jj) == 1:
        out = img
        # is_del = False
    else:
        # 通过与质心之间的距离进行判断
        num = labels.max()  #连通域的个数
        del_array = np.array([0] * (num + 1))# 生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):# 这里如果遇到全黑的图像的话会报错
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
        # is_del = True
    return out

# 方法二：保留最大的连通区域
def largestConnectComponent(bw_img):
    labeled_img, num = measure.label(bw_img, neighbors=8, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc

# dst = largestConnectComponent(img)
# dst = remove_objects(img)

# 方法三：调参
dst = morphology.remove_small_objects(img, min_size=300, connectivity=2)
dst = morphology.remove_small_holes(img, min_size=6500, connectivity=2)

# 显示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))
ax1.imshow(img, plt.cm.gray)
ax1.axis("off")
ax2.imshow(dst, plt.cm.gray)
ax2.axis("off")
fig.tight_layout()
plt.show()


