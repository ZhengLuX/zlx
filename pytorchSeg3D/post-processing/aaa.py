import torch
import SimpleITK as sitk
import numpy as np
image_path = 'E:/results/predicted_result0.mhd'
image = sitk.ReadImage(image_path)
image_array = sitk.GetArrayFromImage(image)

channel1 = image_array[0, 0, :, :, :]

threshold = 0.2

binary_image = (channel1 < threshold).astype(np.float32)

final_image = sitk.GetImageFromArray(binary_image)

sitk.WriteImage(final_image,'E:/results/predicted_result1.mhd')