from PIL import Image
import os
import torch
import numpy as np
import cv2
import numpy as np
normDir = r'E:\zcq\ThyroidUS\data1012\norm\1\1'
saveDir = r'E:\zcq\ThyroidUS\data1012\mask\1\1'
imageDir = r'E:\zcq\ThyroidUS\data1012\1\1'
for img_name in os.listdir(imageDir):
    # 读取图像
    color_image = cv2.imread(os.path.join(imageDir, img_name))
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    #resized_image = cv2.resize(gray_image, (224, 224))
    #normalized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

    # 获取图像的高度和宽度
    height, width = normalized_image.shape

    # 创建一个与图像大小相同的全零矩阵
    zero_matrix = np.zeros((height, width), dtype=np.uint8)

    # 计算中心区域的位置
    center_x = width // 2
    center_y = height // 2

    # 定义中心区域的尺寸（64x64）
    center_size = 64

    # 将中心区域设置为全一
    zero_matrix[center_y - center_size // 2:center_y + center_size // 2, center_x - center_size // 2:center_x + center_size // 2] = 255

    # 保存结果
    save_nor_pth = os.path.join(normDir, img_name)
    cv2.imwrite(save_nor_pth, normalized_image)

    save_pth = os.path.join(saveDir, img_name)
    cv2.imwrite(save_pth, zero_matrix)

print('OK')


