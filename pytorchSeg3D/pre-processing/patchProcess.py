import random
import numpy as np
import os
import SimpleITK as sitk
from utils.patchCrop import get_patch, get_patch_valid
from utils.common import Z_Score_normalization, Min_Max_normalization

def main(root_dir_train, patch_size, neg_ratio):

    # 构建图像和掩码文件夹的路径
    image_dir = os.path.join(root_dir_train, 'image')
    mask_dir = os.path.join(root_dir_train, 'mask')

    # 创建图像和掩码patch的保存路径
    image_patch_dir = os.path.join(root_dir_train, 'image_patch')
    mask_patch_dir = os.path.join(root_dir_train, 'mask_patch')

    # 如果文件夹不存在，则创建
    os.makedirs(image_patch_dir, exist_ok=True)
    os.makedirs(mask_patch_dir, exist_ok=True)

    # 获取图像文件夹下所有文件的完整路径，筛选出具有指定扩展名的文件
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.mhd', '.nii.gz'))]

    # 获取掩码文件夹下所有文件的完整路径，筛选出具有指定扩展名的文件
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(('.mhd', '.nii.gz'))]

    for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):

        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkInt16))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkUInt8))

        # 归一化
        # image = Z_Score_normalization(image)
        image = Min_Max_normalization(image)
        image_patches, mask_patches, _ = get_patch(image, mask, patch_size)

        # 保存patch
        for patch_idx, (image_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):

            # 将patch转换为SimpleITK图像
            image_itk_patch = sitk.GetImageFromArray(image_patch)
            mask_itk_patch = sitk.GetImageFromArray(mask_patch)

            # 设置保存路径
            image_patch_path = os.path.join(image_patch_dir, f'image_patch_{idx}_{patch_idx}.nii.gz')
            mask_patch_path = os.path.join(mask_patch_dir, f'mask_patch_{idx}_{patch_idx}.nii.gz')

            # 保存patch
            sitk.WriteImage(image_itk_patch, image_patch_path)
            sitk.WriteImage(mask_itk_patch, mask_patch_path)

if __name__ == "__main__":
    root_dir_train = 'D:/ImageCAS/1-89'  # 替换为您的训练数据路径
    patch_size = 64
    neg_ratio = 0
    main(root_dir_train, patch_size, neg_ratio)


