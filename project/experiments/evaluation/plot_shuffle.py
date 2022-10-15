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

counter = ['32\n5', '32\n10', '64\n5', '64\n10', '96\n5', '96\n10', '128\n5','128\n10']

py2 = [7.58, 14.36, 20.9, 40, 46, 95, 75.19, 140]
#py2 = [11.47, 20.65, 25.55, 49.97, 50.36, 99.47, 87.69, 173.4]
kernel2 = [10.55, 19.86, 23.95, 45.24, 45.07, 87.09, 73.38, 138.18]

# kernel2 = np.log10(kernel2)
# # # kernel1 = np.log10(kernel1)
# py2 = np.log10(py2)
# # py1 = np.log10(py1)
plt.plot(counter, kernel2, marker='o', markersize=3)
plt.plot(counter, py2, marker='o', markersize=3)
#plt.plot(counter, py1, marker='o', markersize=3)

#plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Kernel time in ms',fontsize=12,color='k')
#plt.xlabel('Iterations',fontsize=12,color='k', loc='left')
plt.text(-1.5, -15, 'Channels:\n    Layers:')

plt.grid(linestyle=':')
plt.axvline(1.5, linewidth=0.5, color="k")
plt.axvline(3.5, linewidth=0.5, color="k")
plt.axvline(5.5, linewidth=0.5, color="k")
#plt.axvline(7.5, linewidth=0.5, color="k")

plt.subplots_adjust(left=0.15, )
plt.legend(['WMMA loading and storing', 'Threads shuffle'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plotshuffle.jpg')

