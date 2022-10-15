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

counter = ['32\n2', '32\n22', '64\n2', '64\n6', '96\n2', '96\n3', '128\n2',]


py2 = [25, 162, 71.82, 169, 199, 147, 248]
kernel2 = [25, 162, 71.67, 168, 173, 132, 212]

# kernel2 = np.log10(kernel2)
# # kernel1 = np.log10(kernel1)
# py2 = np.log10(py2)
# # py1 = np.log10(py1)
plt.plot(counter, kernel2, marker='o', markersize=3)
plt.plot(counter, py2, marker='o', markersize=3)
#plt.plot(counter, py1, marker='o', markersize=3)

#plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Kernel Time in Milliseconds',fontsize=12,color='k')
#plt.xlabel('Iterations',fontsize=12,color='k', loc='left')
plt.text(-1.2, -9, 'Channels:\n    Layers:')

plt.grid(linestyle=':')
plt.axvline(1.5, linewidth=0.5, color="k")
plt.axvline(3.5, linewidth=0.5, color="k")
plt.axvline(5.5, linewidth=0.5, color="k")
#plt.axvline(7.5, linewidth=0.5, color="k")

plt.subplots_adjust(left=0.15, )
plt.legend(['Fully-fused MLP', 'FV-SRN hidden layers'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plot4.jpg')

