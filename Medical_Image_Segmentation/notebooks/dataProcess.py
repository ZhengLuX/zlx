#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from typing import List


# In[2]:


# 获取子文件夹中的所有文件
def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

# 获取 NIfTI 文件列表
def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')

# 创建目录（如果不存在）
def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


# # 处理图像灰度

# In[ ]:


IMAGE_PATH = "D:/zlx/Medical_Image_Segmentation/data/raw/"
LABEL_PATH = "D:/zlx/Medical_Image_Segmentation/data/annotations/"
PROCESSED_PATH = "D:/zlx/Medical_Image_Segmentation/data/processed/"

# 创建 processed 文件夹（如果不存在）
os.makedirs(PROCESSED_PATH, exist_ok=True)

# 列出数据集中的文件
image_files = os.listdir(IMAGE_PATH)
label_files = os.listdir(LABEL_PATH)

image_files = [f for f in image_files if f.endswith(".nii.gz")]  # 只选择 NIfTI 文件
label_files = [f for f in label_files if f.endswith(".nii.gz")]  # 只选择 NIfTI 文件

if len(image_files) == 0 or len(label_files) == 0:
    raise ValueError("数据集中没有找到 NIfTI 文件。")

# 初始化用于统计的变量
gray_values = []

# 遍历所有病人文件，计算标签区域的灰度统计信息并进行预处理
for image_file, label_file in zip(image_files, label_files):
    # 加载数据文件
    image_path = os.path.join(IMAGE_PATH, image_file)
    label_path = os.path.join(LABEL_PATH, label_file)

    image_data = nib.load(image_path)
    label_data = nib.load(label_path)

    image_array = image_data.get_fdata()
    label_array = label_data.get_fdata()
    
    # 仅选择标签区域的灰度值
    label_mask = label_array > 0  # 假设标签区域大于0
    gray_values.extend(image_array[label_mask])

    # 对图像进行预处理（例如裁剪、标准化等）
    image_array = np.clip(image_array, 0, 800)  # 裁剪极端灰度值

    # 保存处理后的图像到 processed 文件夹
    processed_image = nib.Nifti1Image(image_array, affine=image_data.affine)
    processed_image_path = os.path.join(PROCESSED_PATH, image_file)
    nib.save(processed_image, processed_image_path)


# # 调整Spacing

# In[ ]:


def resample_image(itk_image, new_spacing=[0.5, 0.5, 0.5]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    # 计算新的尺寸
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    # 设置重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputPixelType(itk_image.GetPixelID())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # 重采样图像
    resampled_image = resampler.Execute(itk_image)
    return resampled_image

def resample_label(itk_label, new_spacing=[0.5, 0.5, 0.5]):
    original_spacing = itk_label.GetSpacing()
    original_size = itk_label.GetSize()
    
    # 计算新的尺寸
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    # 设置重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_label.GetDirection())
    resampler.SetOutputOrigin(itk_label.GetOrigin())
    resampler.SetOutputPixelType(itk_label.GetPixelID())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 对于标签使用最近邻插值
    
    # 重采样标签
    resampled_label = resampler.Execute(itk_label)
    return resampled_label


IMAGE_PATH = "D:/zlx/Medical_Image_Segmentation/data/processed/image/"
LABEL_PATH = "D:/zlx/Medical_Image_Segmentation/data/processed/label/"

# 列出数据集中的文件
image_files = os.listdir(IMAGE_PATH)
label_files = os.listdir(LABEL_PATH)

image_files = [f for f in image_files if f.endswith(".nii.gz")]  # 只选择 NIfTI 文件
label_files = [f for f in label_files if f.endswith(".nii.gz")]  # 只选择 NIfTI 文件

if len(image_files) == 0 or len(label_files) == 0:
    raise ValueError("数据集中没有找到 NIfTI 文件。")

# 初始化用于统计的变量
gray_values = []

# 遍历所有病人文件，计算标签区域的灰度统计信息
for image_file, label_file in zip(image_files, label_files):

    # 加载数据文件
    image_path = os.path.join(IMAGE_PATH, image_file)
    label_path = os.path.join(LABEL_PATH, label_file)

    # 读取图像和标签
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # 重采样
    resampled_image = resample_image(image)
    resampled_label = resample_label(label)

    #保存
    sitk.WriteImage(resampled_image, image_path)
    sitk.WriteImage(resampled_label, label_path)


# 主处理函数
def process_all_images_and_labels(base_dir):
    # 定义图像和标签子目录
    image_dir = os.path.join(base_dir, "image")
    label_dir = os.path.join(base_dir, "label")

    # 获取所有图像和标签文件
    image_files = nifti_files(image_dir)
    label_files = nifti_files(label_dir)

    if len(image_files) != len(label_files):
        raise ValueError("图像和标签文件数量不匹配，请检查数据。")

    # 遍历所有图像和标签文件，进行批量处理
    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # 读取并预处理图像
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        # 重采样图像和标签
        resampled_image = resample_image(image)
        resampled_label = resample_label(label)

        # 保存重采样后的图像和标签
        sitk.WriteImage(resampled_image, image_path)
        sitk.WriteImage(resampled_label, label_path)

# 定义基础路径
BASE_DIR = "D:/zlx/Medical_Image_Segmentation/data/processed"

# 运行主处理函数
process_all_images_and_labels(BASE_DIR)


# # getPatch

# In[ ]:


def Min_Max_normalization(image_array):
    return (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

def extract_patches(image_array, patch_size=(64, 64, 32), step=(32, 32, 16)):

    patches = []
    coords = []

    for x in range(0, image_array.shape[2] - patch_size[0] + 1, step[0]):
        for y in range(0, image_array.shape[1] - patch_size[1] + 1, step[1]):
            for z in range(0, image_array.shape[0] - patch_size[2] + 1, step[2]):
                patch = image_array[z:z + patch_size[2],
                                    y:y + patch_size[1],
                                    x:x + patch_size[0]]
                patches.append(patch)
                coords.append((x, y, z))

    return np.array(patches), np.array(coords)


# 主处理函数
def process_all_images_and_labels(base_dir, patches_base_dir, patch_size, step):
    # 定义图像和标签子目录
    image_dir = os.path.join(base_dir, "image")
    label_dir = os.path.join(base_dir, "label")
    patches_image_dir = os.path.join(patches_base_dir, "image")
    patches_label_dir = os.path.join(patches_base_dir, "label")

    # 创建目录（如果不存在）
    maybe_mkdir_p(patches_image_dir)
    maybe_mkdir_p(patches_label_dir)

    # 获取所有图像和标签文件
    image_files = nifti_files(image_dir)
    label_files = nifti_files(label_dir)

    if len(image_files) != len(label_files):
        raise ValueError("图像和标签文件数量不匹配，请检查数据。")

    # 遍历所有图像和标签文件，进行批量处理
    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # 读取并预处理图像
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = Min_Max_normalization(image_array)

        # 读取标签
        label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label)

        # 提取图像和标签的 patches
        image_patches, coords = extract_patches(image_array, patch_size, step)
        label_patches, _ = extract_patches(label_array, patch_size, step)

        # 保存提取的 patches
        for i, (image_patch, label_patch) in enumerate(zip(image_patches, label_patches)):
            patches_image_path = f'{patches_image_dir}/{os.path.basename(image_file)[:-7]}_patch_{coords[i][0]}_{coords[i][1]}_{coords[i][2]}.nii.gz'
            patches_label_path = f'{patches_label_dir}/{os.path.basename(label_file)[:-7]}_patch_{coords[i][0]}_{coords[i][1]}_{coords[i][2]}.nii.gz'

            sitk.WriteImage(sitk.GetImageFromArray(image_patch), patches_image_path)
            sitk.WriteImage(sitk.GetImageFromArray(label_patch), patches_label_path)

# 定义基础路径
BASE_DIR = "D:/zlx/Medical_Image_Segmentation/data/processed"
PATCHES_BASE_DIR = "D:/zlx/Medical_Image_Segmentation/data/patches"

# 定义块的大小和步长
patch_size = (128, 128, 64)
step = (64, 64, 32)

# 运行主处理函数
process_all_images_and_labels(BASE_DIR, PATCHES_BASE_DIR, patch_size, step)


# # 拼接图像

# In[ ]:


def reconstruct_image(patches, coords, image_shape, patch_size=(128, 128, 64)):
    reconstructed = np.zeros(image_shape)
    count = np.zeros(image_shape)

    for patch, (x, y, z) in zip(patches, coords):
        reconstructed[z:z + patch_size[2],
                      y:y + patch_size[1],
                      x:x + patch_size[0]] += patch
        count[z:z + patch_size[2],
              y:y + patch_size[1],
              x:x + patch_size[0]] += 1

    # 防止除以0
    count[count == 0] = 1
    reconstructed /= count
    
    return reconstructed
# 还原图像
image_shape = sitk.GetArrayFromImage(image).shape
reconstructed_array = reconstruct_image(image_patches, coords, image_shape, patch_size)

# 转换回SimpleITK图像
reconstructed_image = sitk.GetImageFromArray(reconstructed_array)
reconstructed_image.CopyInformation(image)

# 保存还原图像
sitk.WriteImage(reconstructed_image, 'D:/zlx/Medical_Image_Segmentation/data/patches/reconstructed_image.nii')

