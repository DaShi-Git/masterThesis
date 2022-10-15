#use conda environment: cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
# import cv2
# from skimage import measure

# lossestxt = np.loadtxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_losses.txt', dtype=float)
# countertxt = np.loadtxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_counter.txt', dtype=int)

# # print(lossestxt, countertxt)
# losses =lossestxt.tolist()
# counter= countertxt.tolist()

counter = ['16\n40', '16\n80', '32\n40', '32\n80', '64\n40', '64\n80', '128\n40', '128\n80',]
kernel2 = [0.007604, 0.007951, 0.0088188, 0.0089249, 0.01357, 0.01528, 0.03159, 0.03385]
kernel1 = [0.006625, 0.0072864, 0.0070796, 0.007601, 0.010164, 0.010753, 0.02052116, 0.021313]
py2 = [5.0, 10.375, 6.6687, 13.9, 11.028, 23.09, 22.2, 46.75]
py1 = [2.48, 5.31, 3.36, 7.12, 5.54, 11.6, 11.058, 23.1]

kernel2 = np.log10(kernel2)
kernel1 = np.log10(kernel1)
py2 = np.log10(py2)
py1 = np.log10(py1)

plt.plot(counter, py2, marker='o', markersize=3)
plt.plot(counter, py1, marker='o', markersize=3)
plt.plot(counter, kernel2, marker='o', markersize=3)
plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Log(Time in Seconds)',fontsize=12,color='k')
#plt.xlabel('Iterations',fontsize=12,color='k', loc='left')
#plt.title('Training Process', fontsize=20)

plt.grid(linestyle=':')
plt.axvline(1.5, linewidth=0.5, color="k")
plt.axvline(3.5, linewidth=0.5, color="k")
plt.axvline(5.5, linewidth=0.5, color="k")
plt.text(-1.375, -2.775, 'Channels:\n    Layers:')

plt.subplots_adjust(left=0.15, )
plt.legend(['PyTorch FP16, batch size 2 millions', 'PyTorch FP16, batch size 1 million', 'Fully-fused MLP, batch size 2 millions', 'Fully-fused MLP, batch size 1 million'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plot1.jpg')

