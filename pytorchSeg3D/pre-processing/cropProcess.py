import os
import numpy as np
import SimpleITK as sitk
from utils.patchCrop import centerCrop

# 设置根目录、输出目录和裁剪尺寸
root_dir = 'D:/Coronary Artery Project/data/Aorta/256z/512xy'
output_dir = 'D:/Coronary Artery Project/data/Aorta/256z/512xy/Crop/256xyz'
patch_size = (256, 256, 256)  # 设置裁剪尺寸

output_image_dir = os.path.join(output_dir, 'image')
output_mask_dir = os.path.join(output_dir, 'mask')
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 定义输入目录
image_dir = os.path.join(root_dir, 'image')
mask_dir = os.path.join(root_dir, 'mask')

for image_file in os.listdir(image_dir):
    if image_file.endswith('.mhd'):
        image_path = os.path.join(image_dir, image_file)
        mask_file = image_file  # 假设图像文件和掩码文件有相同的文件名
        mask_path = os.path.join(mask_dir, mask_file)

        # 读取图像和掩码
        itk_image = sitk.ReadImage(image_path)
        itk_mask = sitk.ReadImage(mask_path)

        image_array = sitk.GetArrayFromImage(itk_image)
        mask_array = sitk.GetArrayFromImage(itk_mask)

        image = centerCrop(image_array, patch_size)
        mask = centerCrop(mask_array, patch_size)

        crop_image = sitk.GetImageFromArray(image)
        crop_mask = sitk.GetImageFromArray(mask)

        # 保存处理后的图像和掩码
        output_image_path = os.path.join(output_image_dir, image_file)
        output_mask_path = os.path.join(output_mask_dir, mask_file)

        sitk.WriteImage(crop_image, output_image_path)
        sitk.WriteImage(crop_mask, output_mask_path)