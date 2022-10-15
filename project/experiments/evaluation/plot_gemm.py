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

counter = ['32\n20', '32\n100', '32\n200', '48\n20', '48\n100', '48\n200', '64\n20','64\n100', '64\n200']


py2 = [296306383, 25320727, 6833366, 681182608, 53109152, 15130082, 864993788, 82404733, 25459597]
kernel2 = [1121589262, 726594782, 616667158, 1025256442, 546133333, 326400000, 908243478, 418839097, 257102769]

kernel2 = np.log10(kernel2)
# # kernel1 = np.log10(kernel1)
py2 = np.log10(py2)
# # py1 = np.log10(py1)
plt.plot(counter, kernel2, marker='o', markersize=3)
plt.plot(counter, py2, marker='o', markersize=3)
#plt.plot(counter, py1, marker='o', markersize=3)

#plt.plot(counter, kernel1, marker='o', markersize=3) # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('Throughput: log(Elements)',fontsize=12,color='k')
#plt.xlabel('Iterations',fontsize=12,color='k', loc='left')
plt.text(-1.5, 6.49, 'Channels:\n    Layers:')

plt.grid(linestyle=':')
plt.axvline(2.5, linewidth=0.5, color="k")
#plt.axvline(3.5, linewidth=0.5, color="k")
plt.axvline(5.5, linewidth=0.5, color="k")
#plt.axvline(7.5, linewidth=0.5, color="k")

plt.subplots_adjust(left=0.15, )
plt.legend(['Warp-level GEMM', 'Block-level GEMM'], framealpha=1)
plt.savefig('/home/dashi/projects/tmp6/masterThesis/project/experiments/evaluation/plotgemm.jpg')

