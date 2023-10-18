#coding=utf-8
import numpy as np
import cv2

import scipy.io as sio

out = sio.loadmat('./m.mat')
output = out['num']
x_visualize = output
print(x_visualize.shape)
# for i in range(x_visualize.shape[0]):
# print(x_visualize.shape)

    # print(i)
x_visualize=x_visualize[:1,:,:]
# print(x_visualize.shape)
x_visualize = np.mean(x_visualize,axis=1).reshape(8,8) #shape为[h,w]，二维
x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
savedir =  './'
x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
# cv2.imshow('imshow',x_visualize)
cv2.imwrite(savedir+'GCN_result1.jpg',x_visualize) #保存可视化图像