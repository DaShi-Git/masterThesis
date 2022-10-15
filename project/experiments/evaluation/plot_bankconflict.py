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
#256 threads

counter = [10000000, 100000000, 1000000000, 10000000000]
base = [5.1, 56, 467.1, 4202]
int1 = [2.32, 21.75, 220.62, 1770]
int2 = [1.60, 14.2, 134, 1005]
int4 = [0.82, 6.94, 66, 559]

counter = np.log10(counter)
int1 = np.log10(int1)
int2 = np.log10(int2)
int4 = np.log10(int4)
base = np.log10(base)

plt.plot(counter, int4, marker='o', markersize=3)
plt.plot(counter, int2, marker='o', markersize=3)
plt.plot(counter, int1, marker='o', markersize=3)
plt.plot(counter, base, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Log(Time in Milliseconds)',fontsize=12,color='k')
plt.xlabel('Log(Number of elements)',fontsize=12,color='k')
#plt.title('Training Process', fontsize=20)

plt.grid(linestyle=':')
# plt.axvline(1.5, linewidth=0.5, color="k")
# plt.axvline(3.5, linewidth=0.5, color="k")
# plt.axvline(5.5, linewidth=0.5, color="k")
#plt.text(-0.6, -0.25, 'Channels:\n    Layers:')

plt.subplots_adjust(left=0.15, )
plt.legend(['loading 16 byte at once', 'loading 8 byte at once', 'loading 4 byte at once', 'loading 2 byte at once'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plot5.jpg')

