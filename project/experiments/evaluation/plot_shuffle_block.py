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

counter = [128, 192, 256, 384, 512, 640, 768, 896]

c128 = [156, 137, 146, 160, 166, 169, 172.8, 173.7]
c96 = [103, 95, 95, 97, 102, 103, 104.6, 106]
c64 = [44.6, 46, 42, 40, 42, 45.4, 49.9, 51.9]
c32 =[18.8, 14.4, 11.95, 10.8, 12.1, 13.1, 7.45, 15]

best = [137, 95, 40, 7.45]
best_x = [192, 256, 384, 768]

# kernel2 = np.log10(kernel2)
# # # kernel1 = np.log10(kernel1)
# py2 = np.log10(py2)
# c128 = np.log10(c128)
# c96 = np.log10(c96)
# c64 = np.log10(c64)
# c32 = np.log10(c32)
# best = np.log10(best)

plt.plot(best_x, best, marker='o', markersize=3, linestyle=':')
plt.plot(counter, c128, marker='o', markersize=3)
plt.plot(counter, c96, marker='o', markersize=3)
plt.plot(counter, c64, marker='o', markersize=3)
plt.plot(counter, c32, marker='o', markersize=3)
#plt.plot(counter, py1, marker='o', markersize=3)

#plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Kernel time in ms',fontsize=12,color='k')
plt.xlabel('Block size',fontsize=12,color='k', loc='left')
#plt.text(-1.5, -15, 'Channels:\n    Layers:')

plt.grid(linestyle=':')
# plt.axvline(1.5, linewidth=0.5, color="k")
# plt.axvline(3.5, linewidth=0.5, color="k")
# plt.axvline(5.5, linewidth=0.5, color="k")
#plt.axvline(7.5, linewidth=0.5, color="k")

plt.subplots_adjust(left=0.15, )
plt.legend(['Best block size','128 hidden channels', '96 hidden channels', '64 hidden channels', '32 hidden channels'], framealpha=1, loc='upper left', bbox_to_anchor=(0.52, 0.92))
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plotshuffle_block.jpg')

