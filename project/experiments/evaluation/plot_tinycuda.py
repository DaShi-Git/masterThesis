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

counter = ['16\n20', '32\n20', '64\n20', '128\n20']
# kernel2 = [1.08, 1.83, 1.86, 1.88]
# kernel1 = [0.96, 1.78, 1.84, 1.86]
kernel2 = [1.85, 1.88, 1.96, 1.97]
kernel1 = [1.78, 1.86, 1.90, 1.93]
py2 = [2.65, 6.45, 13.97, 28.95]
py1 = [2.34, 4.73, 9.32, 19.12]

kernel2 = np.log10(kernel2)
kernel1 = np.log10(kernel1)
py2 = np.log10(py2)
py1 = np.log10(py1)

plt.plot(counter, py2, marker='o', markersize=3)
plt.plot(counter, py1, marker='o', markersize=3)
plt.plot(counter, kernel2, marker='o', markersize=3)
plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Log(Time in Milliseconds)',fontsize=12,color='k')
#plt.xlabel('Iterations',fontsize=12,color='k', loc='left')
#plt.title('Training Process', fontsize=20)

plt.grid(linestyle=':')
# plt.axvline(1.5, linewidth=0.5, color="k")
# plt.axvline(3.5, linewidth=0.5, color="k")
# plt.axvline(5.5, linewidth=0.5, color="k")
plt.text(-0.6, 0.06, 'Channels:\n    Layers:')

plt.subplots_adjust(left=0.15, )
plt.legend(['Tiny-cuda-nn, 1.5 millions samples', 'Tiny-cuda-nn, 1 million samples', 'Fully-fused MLP, 1.5 millions samples', 'Fully-fused MLP, 1 million samples'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plot2.jpg')

