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

counter = ['64\n2', '64\n4', '64\n6', '64\n10', '64\n16', '96\n2', '96\n4', '96\n6', '96\n10', '96\n16', '128\n2','128\n4','128\n6','128\n10','128\n16']


py2 = [71.4, 124, 169, 240, 330, 147, 242, 314, 428, 665, 250, 364, 458, 762, 1120]
kernel2 = [64.2, 115, 166, 239, 339, 126, 199, 297, 419, 645, 219, 342, 455, 764, 1127]


kernel2 = np.log10(kernel2)
# # # kernel1 = np.log10(kernel1)
py2 = np.log10(py2)
# # py1 = np.log10(py1)
plt.plot(counter, kernel2, marker='o', markersize=3)
plt.plot(counter, py2, marker='o', markersize=3)

#plt.plot(counter, py1, marker='o', markersize=3)

#plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Log(Kernel time in ms)',fontsize=12,color='k')
#plt.xlabel('Iterations',fontsize=12,color='k', loc='left')
plt.text(-3, 1.61, 'Channels:\n    Layers:')

plt.grid(linestyle=':')
plt.axvline(4.5, linewidth=0.5, color="k")
#plt.axvline(3.5, linewidth=0.5, color="k")
plt.axvline(9.5, linewidth=0.5, color="k")
#plt.axvline(7.5, linewidth=0.5, color="k")

plt.subplots_adjust(left=0.15, )
plt.legend(['Warp batch size = 16', 'Warp batch size = 32'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plotbs.jpg')

