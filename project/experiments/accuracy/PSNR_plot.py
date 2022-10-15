#use conda environment: cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure

lossestxt = np.loadtxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_losses.txt', dtype=float)
countertxt = np.loadtxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_counter.txt', dtype=int)

# print(lossestxt, countertxt)
losses =lossestxt.tolist()
counter= countertxt.tolist()

plt.scatter(counter, losses, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.ylabel('MSE Loss',fontsize=12,color='k')
plt.xlabel('Iterations',fontsize=12,color='k')
plt.title('Training Process', fontsize=20)
plt.grid()
plt.subplots_adjust(left=0.15, )
plt.savefig('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/plot1.jpg')


# img1 = cv2.imread(r'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/albert.jpg')
# img2 = cv2.imread(r'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result1_FP32.jpg')
# img3 = cv2.imread(r'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result1_FP16.jpg')
# img4 = cv2.imread(r'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result1_FP16_kernel.jpg')
# psnr = measure.compare_psnr(img1, img2)
# ssim = measure.compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
# mse = measure.compare_mse(img1, img2)

# print('1-2 PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))

# psnr = measure.compare_psnr(img1, img3)
# ssim = measure.compare_ssim(img1, img3, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
# mse = measure.compare_mse(img1, img3)

# print('1-3 PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))

# psnr = measure.compare_psnr(img1, img4)
# ssim = measure.compare_ssim(img1, img4, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
# mse = measure.compare_mse(img1, img4)

# print('1-4 PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))

# psnr = measure.compare_psnr(img2, img3)
# ssim = measure.compare_ssim(img2, img3, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
# mse = measure.compare_mse(img2, img3)

# print('2-3 PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))

# psnr = measure.compare_psnr(img2, img4)
# ssim = measure.compare_ssim(img2, img4, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
# mse = measure.compare_mse(img2, img4)

# print('2-4 PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))

# psnr = measure.compare_psnr(img3, img4)
# ssim = measure.compare_ssim(img3, img4, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
# mse = measure.compare_mse(img3, img4)

# print('3-4 PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))